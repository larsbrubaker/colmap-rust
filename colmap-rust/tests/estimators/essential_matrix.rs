// Port of COLMAP's src/colmap/estimators/solvers/essential_matrix_test.cc (1:1, same values
// and tolerances), testing colmap_rust::estimators::solvers::essential_matrix. The TEST_P
// suites EssentialMatrixFivePointEstimatorTests.Nominal and
// EssentialMatrixEightPointEstimatorTests.Nominal become one test per instantiation, named
// after the suite, the case and its ray count. Tier B for the solvers, Tier C for the refiner
// and LO-RANSAC.
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does; random_seed stays -1,
// so RANSAC draws from that thread PRNG. Arguments COLMAP draws inside one call are drawn
// here in clang's left-to-right order.

use colmap_rust::estimators::solvers::{
    EssentialMatrixEightPointEstimator, EssentialMatrixFivePointEstimator,
    EssentialMatrixTangentSampsonEstimator,
};
use colmap_rust::geometry::essential_matrix::{
    compute_squared_sampson_errors_homogeneous, essential_matrix_from_pose,
};
use colmap_rust::geometry::pose::CamRayWithJac;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Quaterniond, Vector3d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::optim::{LoRansac, RansacOptions};
use colmap_rust::scene::camera::Camera;
use colmap_rust::sensor::models::CameraModelId;

// Rejection thresholds used by random_epipolar_correspondences to condition a minimal sample:
// minimum depth in front of either camera, and minimum parallax between the two rays as
// sin^2 of their angle (1 - a^2 in the cheirality form).
const K_MIN_DEPTH: f64 = 0.2;
const K_MIN_PARALLAX: f64 = 1e-2; // ~5.7 degrees.

/// Eigen's `normalized()` of a matrix: divided by its Frobenius norm.
fn normalized(m: Matrix3d) -> Matrix3d {
    m / m.norm()
}

/// min(|E - E*|, |E + E*|): distance up to the sign of a null vector.
fn signless_distance(e: Matrix3d, expected: Matrix3d) -> f64 {
    (e - expected).norm().min((e + expected).norm())
}

// Generates a random relative pose with a unit-norm baseline. Bounding the baseline away from
// zero avoids the (near) pure-rotation degeneracy, where the essential matrix vanishes and the
// cheirality of every correspondence becomes ill-defined.
fn test_cam2_from_cam1() -> Rigid3d {
    let rotation = random_eigen_quaterniond();
    Rigid3d::new(rotation, random_eigen_vector3d().normalized())
}

// When reject_degenerate is set, resamples correspondences that make a minimal 5-point solve
// ill-conditioned (near-zero depth or near-parallel rays). Only the minimal case needs this;
// larger samples are robust to such points.
fn random_epipolar_correspondences(
    cam2_from_cam1: &Rigid3d,
    num_rays: usize,
    reject_degenerate: bool,
    rays1: &mut Vec<Vector3d>,
    rays2: &mut Vec<Vector3d>,
) {
    for _ in 0..num_rays {
        let (mut ray1, mut point_in_cam2);
        loop {
            ray1 = random_eigen_vector3d().normalized();
            let random_depth = random_uniform_real(K_MIN_DEPTH, 2.0);
            point_in_cam2 = *cam2_from_cam1 * (random_depth * ray1);
            let ray1_in_cam2 = cam2_from_cam1.rotation * ray1;
            let cos_parallax = ray1_in_cam2.dot(point_in_cam2.normalized());
            let degenerate = point_in_cam2.norm() < K_MIN_DEPTH
                || 1.0 - cos_parallax * cos_parallax < K_MIN_PARALLAX;
            if !(reject_degenerate && degenerate) {
                break;
            }
        }
        rays1.push(ray1);
        rays2.push(point_in_cam2.normalized());
    }
}

fn expect_at_least_one_valid_model(
    rays1: &[Vector3d],
    rays2: &[Vector3d],
    expected_e: Matrix3d,
    models: &[Matrix3d],
    e_eps: f64,
    r_eps: f64,
) {
    let expected_e = normalized(expected_e);
    for model in models {
        let e = normalized(*model);
        if signless_distance(e, expected_e) > e_eps {
            continue;
        }

        // The five/eight-point solvers no longer expose Residuals (bearing Sampson was
        // retired). Verify the recovered model directly with the plain Sampson error, which is
        // ~0 for these noiseless, in-front correspondences.
        let residuals = compute_squared_sampson_errors_homogeneous(rays1, rays2, &e).unwrap();
        for residual in residuals {
            assert!(residual < r_eps, "{residual} >= {r_eps}");
        }
        return;
    }
    panic!("No essential matrix is equal up to scale.");
}

fn five_point_nominal(num_rays: usize) {
    set_prng_seed(0);
    // The minimal case has no redundancy, so it conditions its sample to stay well-posed and
    // accepts the solver's numerical accuracy with a looser tolerance.
    let is_minimal = num_rays == EssentialMatrixFivePointEstimator::MIN_NUM_SAMPLES;
    for _ in 0..100 {
        let cam2_from_cam1 = test_cam2_from_cam1();
        let expected_e = essential_matrix_from_pose(&cam2_from_cam1);
        let mut rays1 = Vec::new();
        let mut rays2 = Vec::new();
        random_epipolar_correspondences(
            &cam2_from_cam1,
            num_rays,
            is_minimal,
            &mut rays1,
            &mut rays2,
        );

        let mut models = Vec::new();
        EssentialMatrixFivePointEstimator::estimate(&rays1, &rays2, &mut models).unwrap();

        let e_eps = if is_minimal { 5e-3 } else { 1e-4 };
        expect_at_least_one_valid_model(&rays1, &rays2, expected_e, &models, e_eps, 1e-5);
    }
}

#[test]
fn essential_matrix_five_point_estimator_essential_matrix_five_point_estimator_tests_nominal_5() {
    five_point_nominal(5);
}

#[test]
fn essential_matrix_five_point_estimator_essential_matrix_five_point_estimator_tests_nominal_20() {
    five_point_nominal(20);
}

#[test]
fn essential_matrix_five_point_estimator_essential_matrix_five_point_estimator_tests_nominal_1000()
{
    five_point_nominal(1000);
}

fn eight_point_nominal(num_rays: usize) {
    set_prng_seed(0);
    for _ in 0..1 {
        let cam2_from_cam1 = test_cam2_from_cam1();
        let expected_e = essential_matrix_from_pose(&cam2_from_cam1);
        let mut rays1 = Vec::new();
        let mut rays2 = Vec::new();
        random_epipolar_correspondences(&cam2_from_cam1, num_rays, false, &mut rays1, &mut rays2);

        let mut models = Vec::new();
        EssentialMatrixEightPointEstimator::estimate(&rays1, &rays2, &mut models).unwrap();

        expect_at_least_one_valid_model(&rays1, &rays2, expected_e, &models, 1e-4, 1e-5);
    }
}

#[test]
fn essential_matrix_eight_point_estimator_essential_matrix_eight_point_estimator_tests_nominal_8() {
    eight_point_nominal(8);
}

#[test]
fn essential_matrix_eight_point_estimator_essential_matrix_eight_point_estimator_tests_nominal_64()
{
    eight_point_nominal(64);
}

#[test]
fn essential_matrix_eight_point_estimator_essential_matrix_eight_point_estimator_tests_nominal_1024(
) {
    eight_point_nominal(1024);
}

fn equirectangular_camera() -> Camera {
    Camera::create_from_model_id(1, CameraModelId::Equirectangular, 0.0, 1000, 500).unwrap()
}

// Attaches an unprojection Jacobian to each bearing via a spherical camera, which maps every
// direction to a valid pixel.
fn with_jacobians(camera: &Camera, rays: &[Vector3d]) -> Vec<CamRayWithJac> {
    rays.iter()
        .map(|ray| {
            let img = camera.img_from_cam(*ray, true).unwrap();
            camera.cam_ray_from_img_with_jac(img).unwrap()
        })
        .collect()
}

// Refine recovers the true pose from a perturbed initial E on exact rays.
#[test]
fn essential_matrix_tangent_sampson_estimator_refine_recovers_pose() {
    set_prng_seed(0);
    let camera = equirectangular_camera();
    for _ in 0..30 {
        let cam2_from_cam1 = test_cam2_from_cam1();
        let expected = normalized(essential_matrix_from_pose(&cam2_from_cam1));
        let mut rays1 = Vec::new();
        let mut rays2 = Vec::new();
        random_epipolar_correspondences(&cam2_from_cam1, 50, false, &mut rays1, &mut rays2);
        let crj1 = with_jacobians(&camera, &rays1);
        let crj2 = with_jacobians(&camera, &rays2);

        // Seed the refinement from a slightly perturbed pose.
        let axis = random_eigen_vector3d().normalized();
        let rotation =
            cam2_from_cam1.rotation * Quaterniond::from_angle_axis(AngleAxisd::new(0.01, axis));
        let translation =
            (cam2_from_cam1.translation + 0.01 * random_eigen_vector3d()).normalized();
        let init = Rigid3d::new(rotation, translation);
        let mut e = essential_matrix_from_pose(&init);

        let dist = |m: Matrix3d| signless_distance(normalized(m), expected);
        let init_dist = dist(e);
        assert!(EssentialMatrixTangentSampsonEstimator::refine(&crj1, &crj2, &mut e).unwrap());
        assert!(dist(e) < 1e-4, "{}", dist(e));
        assert!(dist(e) < init_dist, "{} >= {init_dist}", dist(e));
    }
}

// The LO-RANSAC estimator recovers the pose despite 30% gross outliers.
#[test]
fn essential_matrix_tangent_sampson_estimator_loransac_with_outliers() {
    set_prng_seed(0);
    let camera = equirectangular_camera();
    for _ in 0..10 {
        let cam2_from_cam1 = test_cam2_from_cam1();
        let expected = normalized(essential_matrix_from_pose(&cam2_from_cam1));
        let mut rays1 = Vec::new();
        let mut rays2 = Vec::new();
        random_epipolar_correspondences(&cam2_from_cam1, 200, false, &mut rays1, &mut rays2);
        for ray in rays2.iter_mut().take(60) {
            // 30% gross outliers.
            *ray = random_eigen_vector3d().normalized();
        }
        let crj1 = with_jacobians(&camera, &rays1);
        let crj2 = with_jacobians(&camera, &rays2);

        let options = RansacOptions {
            max_error: 2.0, // pixels
            ..RansacOptions::default()
        };
        let mut ransac: LoRansac<
            EssentialMatrixTangentSampsonEstimator,
            EssentialMatrixTangentSampsonEstimator,
        > = LoRansac::new(
            &options,
            EssentialMatrixTangentSampsonEstimator,
            EssentialMatrixTangentSampsonEstimator,
        )
        .unwrap();
        let report = ransac.estimate(&crj1, &crj2).unwrap();

        assert!(report.success);
        let e = normalized(report.model);
        let distance = signless_distance(e, expected);
        assert!(distance < 1e-2, "{distance}");
    }
}
