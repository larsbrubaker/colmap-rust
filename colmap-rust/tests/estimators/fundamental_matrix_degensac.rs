// Port of COLMAP's src/colmap/estimators/fundamental_matrix_degensac_test.cc (same values
// and tolerances), testing colmap_rust::estimators::fundamental_matrix_degensac and
// fundamental_matrix_degensac_estimator, plus a rust_only_* test of a minimal
// (7-correspondence) estimate. Tier B for the epipole and the
// compatible homography; Tier C for the degeneracy test and the robust estimates (RANSAC).
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. Arguments COLMAP
// draws inside one expression (the comma initializer of RandomCalibrationMatrix,
// `Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>())`) are drawn here in clang's
// left-to-right order.

use crate::expect_near;
use colmap_rust::estimators::fundamental_matrix_degensac::{
    epipole_from_fundamental_matrix, homography_from_fundamental_and_points, is_sample_h_degenerate,
};
use colmap_rust::estimators::fundamental_matrix_degensac_estimator::{
    estimate_fundamental_matrix_degensac, FundamentalMatrixDegensacOptions,
};
use colmap_rust::estimators::solvers::{
    FundamentalMatrixEightPointEstimator, FundamentalMatrixSevenPointEstimator,
};
use colmap_rust::geometry::essential_matrix::{
    compute_squared_sampson_errors, essential_matrix_from_pose, fundamental_from_essential_matrix,
};
use colmap_rust::geometry::homography_matrix::compute_squared_homography_error;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Matrix3d, Vector2d, Vector3d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::{
    random_eigen_quaterniond, random_eigen_vector2d, random_eigen_vector3d,
};
use colmap_rust::optim::{EstimateAsLocal, LoRansac, RansacOptions};

fn random_calibration_matrix() -> Matrix3d {
    let fx = random_uniform_real(800.0, 1200.0);
    let cx = random_uniform_real(400.0, 600.0);
    let fy = random_uniform_real(800.0, 1200.0);
    let cy = random_uniform_real(400.0, 600.0);
    Matrix3d::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
}

fn random_rigid3d(normalize_translation: bool) -> Rigid3d {
    let rotation = random_eigen_quaterniond();
    let translation = random_eigen_vector3d();
    let translation = if normalize_translation {
        translation.normalized()
    } else {
        translation
    };
    Rigid3d::new(rotation, translation)
}

fn true_fundamental(k: &Matrix3d, cam2_from_cam1: &Rigid3d) -> Matrix3d {
    fundamental_from_essential_matrix(k, &essential_matrix_from_pose(cam2_from_cam1), k)
}

fn plane_normal() -> Vector3d {
    Vector3d::new(0.2, -0.1, 1.0).normalized()
}

struct Scene {
    points1: Vec<Vector2d>,
    points2: Vec<Vector2d>,
    on_plane_mask: Vec<bool>,
}

// Generates a two-view scene with a dominant plane. The first `num_on_plane` correspondences
// lie on the plane `normal . X = distance` (in the first camera frame); the remaining ones
// are at random depths off the plane. All correspondences are consistent with the true
// epipolar geometry; only the on-plane ones are additionally consistent with the plane
// homography.
fn generate_dominant_plane_scene(
    cam2_from_cam1: &Rigid3d,
    k: &Matrix3d,
    normal: Vector3d,
    distance: f64,
    num_points: usize,
    num_on_plane: usize,
    noise: f64,
) -> Scene {
    let k_inv = k.inverse();
    let mut scene = Scene {
        points1: Vec::new(),
        points2: Vec::new(),
        on_plane_mask: Vec::new(),
    };
    for i in 0..num_points {
        let v = random_eigen_vector2d().homogeneous();
        // K.topRows<2>() * v.
        let point1 = Vector2d::new(k.row(0).dot(v), k.row(1).dot(v));
        let ray = k_inv * point1.homogeneous();
        let on_plane = i < num_on_plane;
        let depth = if on_plane {
            distance / normal.dot(ray)
        } else {
            random_uniform_real(0.5, 3.0)
        };
        let point3d_in_cam1 = depth * ray;
        let point2 = (*k * (*cam2_from_cam1 * point3d_in_cam1)).hnormalized();
        scene.points1.push(point1 + noise * random_eigen_vector2d());
        scene.points2.push(point2 + noise * random_eigen_vector2d());
        scene.on_plane_mask.push(on_plane);
    }
    scene
}

fn mean_sampson_error_on_subset(
    points1: &[Vector2d],
    points2: &[Vector2d],
    f: &Matrix3d,
    subset_mask: &[bool],
) -> f64 {
    let residuals = compute_squared_sampson_errors(points1, points2, f).unwrap();
    let mut sum = 0.0;
    let mut count = 0usize;
    for (residual, &in_subset) in residuals.iter().zip(subset_mask) {
        if in_subset {
            sum += residual;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

fn test_ransac_options() -> RansacOptions {
    RansacOptions {
        max_error: 1.0,
        confidence: 0.9999,
        min_inlier_ratio: 0.1,
        max_num_trials: 10000,
        random_seed: 0,
        ..RansacOptions::default()
    }
}

fn is_approx_up_to_sign(f: &Matrix3d, expected_f: &Matrix3d, precision: f64) -> bool {
    f.is_approx_with(*expected_f, precision) || (-*f).is_approx_with(*expected_f, precision)
}

#[test]
fn epipole_from_fundamental_matrix_nominal() {
    set_prng_seed(0);
    for _ in 0..20 {
        let k = random_calibration_matrix();
        let cam2_from_cam1 = random_rigid3d(false);
        let f = true_fundamental(&k, &cam2_from_cam1);
        let epipole2 = epipole_from_fundamental_matrix(&f);
        expect_near(epipole2.norm(), 1.0, 1e-9);
        assert!((f.transpose() * epipole2).norm() < 1e-6);
    }
}

#[test]
fn homography_from_fundamental_and_points_nominal() {
    set_prng_seed(0);
    let k = random_calibration_matrix();
    let cam2_from_cam1 = random_rigid3d(false);
    let f = true_fundamental(&k, &cam2_from_cam1);
    let epipole2 = epipole_from_fundamental_matrix(&f);

    const DISTANCE: f64 = 2.0;
    let scene =
        generate_dominant_plane_scene(&cam2_from_cam1, &k, plane_normal(), DISTANCE, 6, 6, 0.0);

    let tri_points1 = [scene.points1[0], scene.points1[1], scene.points1[2]];
    let tri_points2 = [scene.points2[0], scene.points2[1], scene.points2[2]];
    let h = homography_from_fundamental_and_points(&f, epipole2, &tri_points1, &tri_points2)
        .expect("H");

    // All on-plane correspondences must be explained by the recovered homography.
    for (&p1, &p2) in scene.points1.iter().zip(&scene.points2) {
        assert!(compute_squared_homography_error(p1, p2, &h) < 1e-6);
    }
}

#[test]
fn homography_from_fundamental_and_points_collinear_returns_nullopt() {
    set_prng_seed(0);
    let k = random_calibration_matrix();
    let cam2_from_cam1 = random_rigid3d(false);
    let f = true_fundamental(&k, &cam2_from_cam1);
    let epipole2 = epipole_from_fundamental_matrix(&f);

    // Three collinear first-image points make M rank-deficient.
    let tri_points1 = [
        Vector2d::new(100.0, 201.0),
        Vector2d::new(200.0, 401.0),
        Vector2d::new(300.0, 601.0),
    ];
    let tri_points2 = [
        Vector2d::new(110.0, 210.0),
        Vector2d::new(220.0, 430.0),
        Vector2d::new(330.0, 650.0),
    ];
    assert!(
        homography_from_fundamental_and_points(&f, epipole2, &tri_points1, &tri_points2).is_none()
    );
}

#[test]
fn is_sample_h_degenerate_detects_and_rejects() {
    set_prng_seed(0);
    let k = random_calibration_matrix();
    let cam2_from_cam1 = random_rigid3d(false);
    let f = true_fundamental(&k, &cam2_from_cam1);
    let normal = plane_normal();

    // A sample with 5 of 7 correspondences on the plane is H-degenerate.
    {
        let scene = generate_dominant_plane_scene(&cam2_from_cam1, &k, normal, 2.0, 7, 5, 0.0);
        assert!(
            is_sample_h_degenerate(&f, &scene.points1, &scene.points2, 4.0, 5.0 / 7.0).unwrap()
        );
    }

    // A general (non-planar) sample is not H-degenerate.
    {
        let scene = generate_dominant_plane_scene(&cam2_from_cam1, &k, normal, 2.0, 7, 0, 0.0);
        assert!(
            !is_sample_h_degenerate(&f, &scene.points1, &scene.points2, 4.0, 5.0 / 7.0).unwrap()
        );
    }
}

// On a general non-planar scene, DEGENSAC recovers the fundamental matrix as well as the
// plain LO-RANSAC estimator.
#[test]
fn fundamental_matrix_degensac_non_planar_parity() {
    set_prng_seed(0);
    let k = random_calibration_matrix();
    let cam2_from_cam1 = random_rigid3d(false);
    let mut expected_f = true_fundamental(&k, &cam2_from_cam1);
    expected_f = expected_f / expected_f[(2, 2)];

    let scene =
        generate_dominant_plane_scene(&cam2_from_cam1, &k, plane_normal(), 2.0, 200, 0, 0.0);

    let options = FundamentalMatrixDegensacOptions {
        ransac: test_ransac_options(),
        ..FundamentalMatrixDegensacOptions::default()
    };
    let report =
        estimate_fundamental_matrix_degensac(&scene.points1, &scene.points2, &options).unwrap();
    assert!(report.success);

    let f = report.model / report.model[(2, 2)];
    assert!(is_approx_up_to_sign(&f, &expected_f, 1e-3));
    assert!(report.support.num_inliers >= scene.points1.len() - 2);
}

// On a dominant-plane scene with real off-plane parallax, DEGENSAC recovers the correct
// fundamental matrix via plane-and-parallax completion, explaining the off-plane points
// well.
#[test]
fn fundamental_matrix_degensac_recovers_f_on_dominant_plane() {
    set_prng_seed(0);
    let k = random_calibration_matrix();
    let cam2_from_cam1 = random_rigid3d(true);
    let mut expected_f = true_fundamental(&k, &cam2_from_cam1);
    expected_f = expected_f / expected_f[(2, 2)];

    const NUM_POINTS: usize = 200;
    const NUM_ON_PLANE: usize = 190; // 95% dominant plane.
    let scene = generate_dominant_plane_scene(
        &cam2_from_cam1,
        &k,
        plane_normal(),
        2.0,
        NUM_POINTS,
        NUM_ON_PLANE,
        0.1,
    );
    let off_plane_mask: Vec<bool> = scene.on_plane_mask.iter().map(|&on| !on).collect();

    let options = FundamentalMatrixDegensacOptions {
        ransac: test_ransac_options(),
        ..FundamentalMatrixDegensacOptions::default()
    };
    let report =
        estimate_fundamental_matrix_degensac(&scene.points1, &scene.points2, &options).unwrap();
    assert!(report.success);

    let f = report.model / report.model[(2, 2)];
    assert!(is_approx_up_to_sign(&f, &expected_f, 1e-2));

    // DEGENSAC must explain the off-plane parallax points well, which a plane-corrupted
    // fundamental matrix cannot.
    assert!(
        mean_sampson_error_on_subset(
            &scene.points1,
            &scene.points2,
            &report.model,
            &off_plane_mask
        ) < 1.0
    );
}

// Aggregated over many dominant-plane scenes, DEGENSAC recovers the correct epipolar
// geometry at least as often as plain LO-RANSAC, which is prone to terminating on a
// plane-corrupted model.
#[test]
fn fundamental_matrix_degensac_outperforms_lo_ransac_on_dominant_plane() {
    const NUM_SCENES: u32 = 40;
    const NUM_POINTS: usize = 200;
    const NUM_ON_PLANE: usize = 190; // 95% dominant plane.
    const OFF_PLANE_ERROR_THRESHOLD: f64 = 1.0;

    let mut degensac_successes = 0;
    let mut loransac_successes = 0;
    for s in 0..NUM_SCENES {
        set_prng_seed(s);
        let k = random_calibration_matrix();
        let cam2_from_cam1 = random_rigid3d(true);
        let scene = generate_dominant_plane_scene(
            &cam2_from_cam1,
            &k,
            plane_normal(),
            2.0,
            NUM_POINTS,
            NUM_ON_PLANE,
            0.1,
        );
        let off_plane_mask: Vec<bool> = scene.on_plane_mask.iter().map(|&on| !on).collect();

        let ransac_options = test_ransac_options();

        let degensac_options = FundamentalMatrixDegensacOptions {
            ransac: ransac_options,
            ..FundamentalMatrixDegensacOptions::default()
        };
        let degensac_report =
            estimate_fundamental_matrix_degensac(&scene.points1, &scene.points2, &degensac_options)
                .unwrap();
        if degensac_report.success
            && mean_sampson_error_on_subset(
                &scene.points1,
                &scene.points2,
                &degensac_report.model,
                &off_plane_mask,
            ) < OFF_PLANE_ERROR_THRESHOLD
        {
            degensac_successes += 1;
        }

        let mut loransac: LoRansac<
            FundamentalMatrixSevenPointEstimator,
            EstimateAsLocal<FundamentalMatrixEightPointEstimator>,
        > = LoRansac::new(
            &ransac_options,
            FundamentalMatrixSevenPointEstimator,
            EstimateAsLocal(FundamentalMatrixEightPointEstimator),
        )
        .unwrap();
        let loransac_report = loransac.estimate(&scene.points1, &scene.points2).unwrap();
        if loransac_report.success
            && mean_sampson_error_on_subset(
                &scene.points1,
                &scene.points2,
                &loransac_report.model,
                &off_plane_mask,
            ) < OFF_PLANE_ERROR_THRESHOLD
        {
            loransac_successes += 1;
        }
    }

    assert!(degensac_successes > 0);
    assert!(degensac_successes >= loransac_successes);
}

// Seven general correspondences: every 7-point model fits all of them. LO-RANSAC runs its
// local optimization (Refine, whose Sampson polish needs eight points) only on more than
// kMinNumSamples = 7 inliers, so the minimal estimate succeeds without it.
#[test]
fn rust_only_fundamental_matrix_degensac_seven_correspondences() {
    set_prng_seed(0);
    let k = random_calibration_matrix();
    let cam2_from_cam1 = random_rigid3d(false);
    let scene = generate_dominant_plane_scene(&cam2_from_cam1, &k, plane_normal(), 2.0, 7, 0, 0.0);

    let options = FundamentalMatrixDegensacOptions {
        ransac: test_ransac_options(),
        ..FundamentalMatrixDegensacOptions::default()
    };
    let report =
        estimate_fundamental_matrix_degensac(&scene.points1, &scene.points2, &options).unwrap();
    assert!(report.success);
    assert_eq!(report.support.num_inliers, 7);
}
