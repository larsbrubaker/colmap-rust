// Port of COLMAP's src/colmap/estimators/solvers/relpose_one_sided_focal_test.cc (1:1, same
// values and tolerances), testing colmap_rust::estimators::solvers::relpose_one_sided_focal.
// Tier B for the minimal solver and residuals, Tier C for the refiner.
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. Arguments COLMAP
// draws inside one call are drawn here in clang's left-to-right order.

use colmap_rust::estimators::solvers::{
    RelativePoseOneSidedFocalEstimator, RelativePoseOneSidedFocalModel,
};
use colmap_rust::geometry::essential_matrix::essential_matrix_from_pose;
use colmap_rust::geometry::pose::CamRayWithJac;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Quaterniond, Vector2d, Vector3d};
use colmap_rust::math::deg_to_rad;
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::random_eigen_vector3d;
use colmap_rust::scene::camera::Camera;
use colmap_rust::sensor::models::CameraModelId;

// The unknown focal of the first (uncalibrated) view. The second view is calibrated and
// enters through a real camera, whose unprojection Jacobian lets its share of the residual be
// measured in its own pixels.
const K_FOCAL1: f64 = 1000.0;
const K_FOCAL2: f64 = 800.0;
const K_WIDTH2: usize = 1600;
const K_HEIGHT2: usize = 1200;

// The calibrated second camera. A distortion-free model keeps the projection round trip used
// to build exact correspondences exact.
fn test_camera2(model_id: CameraModelId) -> Camera {
    Camera::create_from_model_id(2, model_id, K_FOCAL2, K_WIDTH2, K_HEIGHT2).unwrap()
}

// Rejection thresholds used to condition a minimal sample: minimum depth in front of the
// second camera and minimum parallax (sin^2 of the ray angle).
const K_MIN_DEPTH: f64 = 0.5;
const K_MIN_PARALLAX: f64 = 1e-2; // ~5.7 degrees.

/// Eigen's `normalized()` of a matrix: divided by its Frobenius norm.
fn normalized(m: Matrix3d) -> Matrix3d {
    m / m.norm()
}

// Random relative pose with a unit-norm baseline (away from the pure-rotation degeneracy) and
// a rotation bounded so the two view frustums overlap, which keeps the sampled points in
// front of both cameras.
fn test_cam2_from_cam1(max_angle_deg: f64) -> Rigid3d {
    let axis = random_eigen_vector3d().normalized();
    let angle = deg_to_rad(random_uniform_real(0.0, max_angle_deg));
    let rotation = Quaterniond::from_angle_axis(AngleAxisd::new(angle, axis));
    Rigid3d::new(rotation, random_eigen_vector3d().normalized())
}

// Generates correspondences from random 3D points: centered image points scaled by `focal1`
// for the uncalibrated view, and for the calibrated one the bearing and unprojection Jacobian
// read back from `camera2`. Clearing require_front_of_cam2 admits rays at any angle, as a
// fisheye or spherical second camera would observe.
//
// The second view is round-tripped through the camera rather than synthesized as a bare
// bearing, because the residual now needs d(ray2)/d(pixel2), which only the camera model can
// supply.
#[allow(clippy::too_many_arguments)]
fn random_one_sided_focal_correspondences(
    cam2_from_cam1: &Rigid3d,
    camera2: &Camera,
    focal1: f64,
    num_points: usize,
    reject_degenerate: bool,
    points1: &mut Vec<Vector2d>,
    cam_rays2_with_jac: &mut Vec<CamRayWithJac>,
    require_front_of_cam2: bool,
) {
    for _ in 0..num_points {
        let mut point_in_cam1;
        let mut cam_ray2_with_jac: Option<CamRayWithJac>;
        loop {
            // Moderate field of view (|x/z|, |y/z| <= ~0.5): wide-angle points span a
            // magnitude range that degrades the conditioning of the minimal solve.
            let mut ray1 = random_eigen_vector3d();
            ray1.z = ray1.z.abs() + 2.0;
            let depth = random_uniform_real(1.0, 3.0);
            point_in_cam1 = depth * ray1.normalized();
            let point_in_cam2 = *cam2_from_cam1 * point_in_cam1;
            let ray1_in_cam2 = cam2_from_cam1.rotation * point_in_cam1.normalized();
            let cos_parallax = ray1_in_cam2.dot(point_in_cam2.normalized());
            // Cheirality in cam1 holds by construction; requiring it in cam2 is only
            // meaningful for a pinhole second view.
            let mut degenerate = 1.0 - cos_parallax * cos_parallax < K_MIN_PARALLAX
                || (require_front_of_cam2 && point_in_cam2.z < K_MIN_DEPTH);
            cam_ray2_with_jac = None;
            if let Some(xy) = camera2.img_from_cam(point_in_cam2, true) {
                cam_ray2_with_jac = camera2.cam_ray_from_img_with_jac(xy);
            }
            // An unprojectable or rank-deficient pixel carries no usable Jacobian, so it is
            // always resampled, whatever the caller asked for.
            degenerate = degenerate || cam_ray2_with_jac.is_none();
            if !(degenerate && (reject_degenerate || cam_ray2_with_jac.is_none())) {
                break;
            }
        }
        points1.push(Vector2d::new(
            focal1 * point_in_cam1.x / point_in_cam1.z,
            focal1 * point_in_cam1.y / point_in_cam1.z,
        ));
        cam_rays2_with_jac.push(cam_ray2_with_jac.expect("resampled until unprojectable"));
    }
}

// Whether at least one model recovers the essential matrix (up to scale/sign) and the unknown
// focal length, with small residuals on the exact correspondences. Returns a bool rather than
// asserting, so that FullSphereCalibratedRays can tolerate the solver's intrinsic failure
// rate over many draws.
#[allow(clippy::too_many_arguments)]
fn has_valid_model(
    points1: &[Vector2d],
    cam_rays2_with_jac: &[CamRayWithJac],
    expected_e: &Matrix3d,
    expected_focal: f64,
    models: &[RelativePoseOneSidedFocalModel],
    e_eps: f64,
    focal_rel_eps: f64,
    r_eps: f64,
) -> bool {
    let expected_e_n = normalized(*expected_e);
    for model in models {
        let e = normalized(model.e);
        if (e - expected_e_n).norm().min((e + expected_e_n).norm()) > e_eps {
            continue;
        }
        if (model.focal - expected_focal).abs() / expected_focal > focal_rel_eps {
            continue;
        }
        let mut residuals = Vec::new();
        RelativePoseOneSidedFocalEstimator::residuals(
            points1,
            cam_rays2_with_jac,
            model,
            &mut residuals,
        )
        .unwrap();
        if residuals.iter().any(|&r| r >= r_eps) {
            continue;
        }
        return true;
    }
    false
}

/// [`has_valid_model`] with C++'s default tolerances.
fn has_valid_model_default(
    points1: &[Vector2d],
    cam_rays2_with_jac: &[CamRayWithJac],
    expected_e: &Matrix3d,
    expected_focal: f64,
    models: &[RelativePoseOneSidedFocalModel],
) -> bool {
    has_valid_model(
        points1,
        cam_rays2_with_jac,
        expected_e,
        expected_focal,
        models,
        5e-3,
        1e-2,
        1e-2,
    )
}

// The minimal 6-point solver recovers the pose and the unknown focal on clean samples.
#[test]
fn relative_pose_one_sided_focal_estimator_nominal() {
    set_prng_seed(0);
    let camera2 = test_camera2(CameraModelId::Pinhole);
    for _ in 0..100 {
        let cam2_from_cam1 = test_cam2_from_cam1(60.0);
        let expected_e = essential_matrix_from_pose(&cam2_from_cam1);
        let mut points1 = Vec::new();
        let mut cam_rays2_with_jac = Vec::new();
        random_one_sided_focal_correspondences(
            &cam2_from_cam1,
            &camera2,
            K_FOCAL1,
            RelativePoseOneSidedFocalEstimator::MIN_NUM_SAMPLES,
            true,
            &mut points1,
            &mut cam_rays2_with_jac,
            true,
        );

        let mut models = Vec::new();
        RelativePoseOneSidedFocalEstimator::estimate(&points1, &cam_rays2_with_jac, &mut models)
            .unwrap();

        assert!(has_valid_model_default(
            &points1,
            &cam_rays2_with_jac,
            &expected_e,
            K_FOCAL1,
            &models
        ));
    }
}

// Nominal residual behavior: zero on exact correspondences, growing with the error, in
// first-view pixels, and max on degenerate inputs (non-positive focal or a zero ray). The
// exact formula is pinned by the cost-function tests.
#[test]
fn relative_pose_one_sided_focal_estimator_residuals() {
    set_prng_seed(0);
    let camera2 = test_camera2(CameraModelId::Pinhole);
    let cam2_from_cam1 = test_cam2_from_cam1(60.0);
    let mut points1 = Vec::new();
    let mut cam_rays2_with_jac = Vec::new();
    random_one_sided_focal_correspondences(
        &cam2_from_cam1,
        &camera2,
        K_FOCAL1,
        30,
        false,
        &mut points1,
        &mut cam_rays2_with_jac,
        true,
    );

    let model = RelativePoseOneSidedFocalModel {
        e: essential_matrix_from_pose(&cam2_from_cam1),
        focal: K_FOCAL1,
    };

    // Exact correspondences lie on the epipolar variety.
    let mut residuals = Vec::new();
    RelativePoseOneSidedFocalEstimator::residuals(
        &points1,
        &cam_rays2_with_jac,
        &model,
        &mut residuals,
    )
    .unwrap();
    for &residual in &residuals {
        assert!(residual.sqrt() < 1e-6, "{residual}");
    }

    // Shift the view-1 points off their epipolar lines along the line normal. The constraint
    // is linear in the first view's pixel, so the residual is at most the offset, and its
    // mean is of pixel order rather than the ~offset/focal of ray units. The mean stays
    // robust where a single point's sensitivity vanishes near an epipole.
    let m = model.e * Matrix3d::from_diagonal(Vector3d::new(1.0 / K_FOCAL1, 1.0 / K_FOCAL1, 1.0));
    let mut prev_mean = 0.0;
    for offset_pixels in [1.0, 4.0, 16.0] {
        let mut shifted_points1 = points1.clone();
        for i in 0..points1.len() {
            let mt_ray2 = m.transpose() * cam_rays2_with_jac[i].ray;
            let line_normal = Vector2d::new(mt_ray2.x, mt_ray2.y).normalized();
            shifted_points1[i] += offset_pixels * line_normal;
        }
        RelativePoseOneSidedFocalEstimator::residuals(
            &shifted_points1,
            &cam_rays2_with_jac,
            &model,
            &mut residuals,
        )
        .unwrap();
        let mut mean = 0.0;
        for &residual in &residuals {
            assert!(residual.sqrt() < offset_pixels + 1e-6);
            mean += residual.sqrt();
        }
        mean /= residuals.len() as f64;
        assert!(mean > 0.1 * offset_pixels);
        assert!(mean > prev_mean);
        prev_mean = mean;
    }

    let mut invalid_model = model;
    invalid_model.focal = 0.0;
    RelativePoseOneSidedFocalEstimator::residuals(
        &points1,
        &cam_rays2_with_jac,
        &invalid_model,
        &mut residuals,
    )
    .unwrap();
    for &residual in &residuals {
        assert_eq!(residual, f64::MAX);
    }

    let mut degenerate_rays2 = cam_rays2_with_jac.clone();
    degenerate_rays2[0] = CamRayWithJac::zero();
    RelativePoseOneSidedFocalEstimator::residuals(
        &points1,
        &degenerate_rays2,
        &model,
        &mut residuals,
    )
    .unwrap();
    assert_eq!(residuals[0], f64::MAX);
    assert!(residuals[1].sqrt() < 1e-6);
}

// Refinement pulls a perturbed pose + focal back to the ground truth.
#[test]
fn relative_pose_one_sided_focal_estimator_refine_from_initial_model() {
    set_prng_seed(0);
    let camera2 = test_camera2(CameraModelId::Pinhole);
    for _ in 0..50 {
        let cam2_from_cam1 = test_cam2_from_cam1(60.0);
        let expected_e = essential_matrix_from_pose(&cam2_from_cam1);
        let mut points1 = Vec::new();
        let mut cam_rays2_with_jac = Vec::new();
        random_one_sided_focal_correspondences(
            &cam2_from_cam1,
            &camera2,
            K_FOCAL1,
            50,
            false,
            &mut points1,
            &mut cam_rays2_with_jac,
            true,
        );

        let seed_rotation = cam2_from_cam1.rotation
            * Quaterniond::from_angle_axis(AngleAxisd::new(
                0.02,
                random_eigen_vector3d().normalized(),
            ));
        let seed_translation =
            (cam2_from_cam1.translation + 0.02 * random_eigen_vector3d()).normalized();
        let mut model = RelativePoseOneSidedFocalModel {
            e: essential_matrix_from_pose(&Rigid3d::new(seed_rotation, seed_translation)),
            focal: 1.1 * K_FOCAL1,
        };

        assert!(RelativePoseOneSidedFocalEstimator::refine(
            &points1,
            &cam_rays2_with_jac,
            &mut model
        )
        .unwrap());

        let models = [model];
        assert!(has_valid_model(
            &points1,
            &cam_rays2_with_jac,
            &expected_e,
            K_FOCAL1,
            &models,
            1e-3,
            1e-2,
            1e-2
        ));
    }
}

// The calibrated view enters as bearing rays, which span the full sphere rather than a
// pinhole image plane: rays with z <= 0, which only a camera seeing beyond 180 degrees
// observes and which no pinhole image plane can represent, must be usable correspondences. An
// equirectangular second camera supplies both those rays and the Jacobians that put its share
// of the residual in its own pixels.
#[test]
fn relative_pose_one_sided_focal_estimator_full_sphere_calibrated_rays() {
    set_prng_seed(0);
    let camera2 = test_camera2(CameraModelId::Equirectangular);
    // Without the cheirality filter the configurations are harsher than any pinhole pair, and
    // the minimal solve occasionally loses the true root. Never exceeded 2% over 199 measured
    // runs; a real regression exceeds it at once.
    const K_NUM_TRIALS: usize = 100;
    const K_MAX_FAILURE_RATE: f64 = 0.03;
    let mut num_failures = 0;
    for _ in 0..K_NUM_TRIALS {
        let mut cam2_from_cam1;
        let mut points1 = Vec::new();
        let mut cam_rays2_with_jac = Vec::new();
        // Resample until the sample really contains such a ray, so that every trial exercises
        // the property rather than a random subset of them. A large rotation puts a good
        // share of the points outside any pinhole frustum.
        loop {
            points1.clear();
            cam_rays2_with_jac.clear();
            cam2_from_cam1 = test_cam2_from_cam1(140.0);
            random_one_sided_focal_correspondences(
                &cam2_from_cam1,
                &camera2,
                K_FOCAL1,
                RelativePoseOneSidedFocalEstimator::MIN_NUM_SAMPLES,
                true,
                &mut points1,
                &mut cam_rays2_with_jac,
                false,
            );
            if cam_rays2_with_jac.iter().any(|r| r.ray.z <= 0.0) {
                break;
            }
        }

        let mut models = Vec::new();
        RelativePoseOneSidedFocalEstimator::estimate(&points1, &cam_rays2_with_jac, &mut models)
            .unwrap();

        if !has_valid_model_default(
            &points1,
            &cam_rays2_with_jac,
            &essential_matrix_from_pose(&cam2_from_cam1),
            K_FOCAL1,
            &models,
        ) {
            num_failures += 1;
        }
    }
    assert!(num_failures as f64 / K_NUM_TRIALS as f64 <= K_MAX_FAILURE_RATE);
}
