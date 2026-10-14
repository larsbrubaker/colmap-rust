// Rust-only tests (PoseLib ships no unit tests COLMAP runs; COLMAP tests these solvers only
// through RelativePoseSharedFocalEstimator and RelativePoseOneSidedFocalEstimator, whose
// refinement waits on the tiny cost functions) of
// colmap_rust::estimators::solvers::poselib::relpose_6pt_shared_focal and
// relpose_6pt_onesided_focal. Tier B: on random minimal problems with a known pose and focal,
// one returned image pair has the true focal and pose; every pair carries SIMPLE_PINHOLE
// cameras and a pose under which every correspondence is cheiral.

use colmap_rust::estimators::solvers::poselib::essential::check_cheirality;
use colmap_rust::estimators::solvers::poselib::relpose_6pt_onesided_focal::relpose_6pt_onesided_focal;
use colmap_rust::estimators::solvers::poselib::relpose_6pt_shared_focal::relpose_6pt_shared_focal;
use colmap_rust::estimators::solvers::poselib::{ImagePair, PoseLibCamera};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Vector3d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::random_eigen_vector3d;

/// A rotation of 0.1 to 0.5 rad about a random axis and a unit baseline: enough rotation that
/// the optical axes are generically skew (a pure translation is a focal-length singularity).
fn random_cam2_from_cam1() -> Rigid3d {
    let axis = random_eigen_vector3d().normalized();
    let angle = random_uniform_real(0.1, 0.5);
    let rotation = AngleAxisd::new(angle, axis).to_quaternion();
    Rigid3d::new(rotation, random_eigen_vector3d().normalized())
}

/// Six 3D points (camera-1 frame) in front of both cameras.
fn random_points(cam2_from_cam1: &Rigid3d) -> Vec<(Vector3d, Vector3d)> {
    let mut points = Vec::new();
    while points.len() < 6 {
        let depth = random_uniform_real(2.0, 6.0);
        let point1 = Vector3d::new(
            random_uniform_real(-1.0, 1.0),
            random_uniform_real(-1.0, 1.0),
            1.0,
        ) * depth;
        let point2 = *cam2_from_cam1 * point1;
        if point2.z < 0.5 {
            continue;
        }
        points.push((point1, point2));
    }
    points
}

/// Homogeneous principal-point-centered image point `(f X / Z, f Y / Z, 1)`.
fn image_point(point: Vector3d, focal: f64) -> Vector3d {
    Vector3d::new(focal * point.x / point.z, focal * point.y / point.z, 1.0)
}

/// Relative focal and pose tolerance. Well-conditioned problems agree to 1e-12; the
/// degree-15 polynomial of the shared-focal template leaves an occasional problem
/// ill-conditioned (observed 5e-6 off). COLMAP's own estimator tests allow 1e-2.
const TOL: f64 = 1e-4;

/// Whether `pair` is the true pose, with a relative focal error below `focal_tol`.
fn matches(pair: &ImagePair, truth: &Rigid3d, focal: f64, focal_tol: f64, pose_tol: f64) -> bool {
    let r_error = (pair.pose.r() - truth.rotation.to_rotation_matrix()).norm();
    let t_error = (pair.pose.t.normalized() - truth.translation.normalized()).norm();
    let f_error = (pair.camera1.focal() - focal).abs() / focal;
    r_error < pose_tol && t_error < pose_tol && f_error < focal_tol
}

/// Every correspondence is cheiral under the pair's pose, on calibrated unit bearings.
fn assert_cheiral(pair: &ImagePair, x1: &[Vector3d], x2: &[Vector3d]) {
    let (f1, f2) = (pair.camera1.focal(), pair.camera2.focal());
    for (a, b) in x1.iter().zip(x2) {
        let b1 = Vector3d::new(a.x / f1, a.y / f1, a.z).normalized();
        let b2 = Vector3d::new(b.x / f2, b.y / f2, b.z).normalized();
        assert!(check_cheirality(&pair.pose, b1, b2, 0.0));
    }
}

#[test]
fn rust_only_shared_focal_recovers_true_pose_and_focal() {
    set_prng_seed(4);
    let mut found_count = 0;
    let trials = 50;
    for _ in 0..trials {
        let cam2_from_cam1 = random_cam2_from_cam1();
        let focal = random_uniform_real(0.5, 2.0);
        let points = random_points(&cam2_from_cam1);
        let x1: Vec<Vector3d> = points.iter().map(|p| image_point(p.0, focal)).collect();
        let x2: Vec<Vector3d> = points.iter().map(|p| image_point(p.1, focal)).collect();

        let unit = PoseLibCamera::new(0, &[1.0, 0.0, 0.0], -1, -1);
        let mut pairs = vec![ImagePair::new(Default::default(), unit.clone(), unit)];
        let n = relpose_6pt_shared_focal(&x1, &x2, &mut pairs);
        assert_eq!(n, pairs.len(), "the output is cleared first");

        let mut found = false;
        for pair in &pairs {
            assert_eq!(pair.camera1.model_name(), "SIMPLE_PINHOLE");
            assert_eq!(pair.camera1, pair.camera2, "the focal is shared");
            assert!(pair.camera1.focal() > 0.0);
            assert_cheiral(pair, &x1, &x2);
            found |= matches(pair, &cam2_from_cam1, focal, TOL, TOL);
        }
        if found {
            found_count += 1;
        }
    }
    // Every random problem here is well away from the focal singularities.
    assert_eq!(found_count, trials);
}

#[test]
fn rust_only_onesided_focal_recovers_true_pose_and_focal() {
    set_prng_seed(5);
    let mut found_count = 0;
    let trials = 50;
    for _ in 0..trials {
        let cam2_from_cam1 = random_cam2_from_cam1();
        // Pixel-scale focal: the solver rescales the uncalibrated view itself.
        let focal = random_uniform_real(300.0, 3000.0);
        let points = random_points(&cam2_from_cam1);
        let x1: Vec<Vector3d> = points.iter().map(|p| image_point(p.0, focal)).collect();
        let x2: Vec<Vector3d> = points.iter().map(|p| p.1.normalized()).collect();

        let mut pairs = Vec::new();
        let n = relpose_6pt_onesided_focal(&x1, &x2, &mut pairs);
        assert_eq!(n, pairs.len());

        let mut found = false;
        for pair in &pairs {
            assert_eq!(pair.camera1.model_name(), "SIMPLE_PINHOLE");
            assert_eq!(pair.camera2.params, vec![1.0, 0.0, 0.0]);
            assert!(pair.camera1.focal() > 0.0);
            assert_cheiral(pair, &x1, &x2);
            found |= matches(pair, &cam2_from_cam1, focal, TOL, TOL);
        }
        if found {
            found_count += 1;
        }
    }
    assert_eq!(found_count, trials);
}
