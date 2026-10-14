// Rust-only tests (PoseLib's p4pf has no unit test in COLMAP or colmap-sharp; COLMAP tests it
// through P4PFEstimator in absolute_pose_test.cc) of
// colmap_rust::estimators::solvers::poselib::p4pf, and PoseLibCamera's parameter
// initialization. Tier B: the ground-truth pose and focal length are among the solutions.

use colmap_rust::estimators::solvers::poselib::p4pf::{p4pf, p4pf_fx_fy};
use colmap_rust::estimators::solvers::poselib::poselib_camera::{
    id_from_string, name_from_id, PoseLibCamera,
};
use colmap_rust::estimators::solvers::poselib::CameraPose;
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Vector2d, Vector3d};

fn scene(fx: f64, fy: f64) -> (Matrix3d, Vector3d, Vec<Vector2d>, Vec<Vector3d>) {
    let r = AngleAxisd::new(0.4, Vector3d::new(0.3, -0.8, 0.2).normalized()).to_rotation_matrix();
    let t = Vector3d::new(0.3, -0.2, 2.0);
    let big_x = vec![
        Vector3d::new(1.0, 0.5, 4.0),
        Vector3d::new(-1.2, 0.3, 5.0),
        Vector3d::new(0.4, -0.9, 3.0),
        Vector3d::new(-0.6, -0.7, 6.0),
    ];
    let x = big_x
        .iter()
        .map(|&p| {
            let c = r * p + t;
            Vector2d::new(fx * c.x / c.z, fy * c.y / c.z)
        })
        .collect();
    (r, t, x, big_x)
}

fn matches(pose: &CameraPose, r: Matrix3d, t: Vector3d) -> bool {
    (pose.r() - r).norm() < 1e-6 && (pose.t - t).norm() < 1e-6
}

#[test]
fn rust_only_p4pf_shared_focal_recovers_ground_truth() {
    let focal = 1200.0;
    let (r, t, x, big_x) = scene(focal, focal);

    let mut poses = Vec::new();
    let mut focals = Vec::new();
    let n = p4pf(&x, &big_x, &mut poses, &mut focals, false);
    assert_eq!(n, poses.len());
    assert_eq!(n, focals.len());
    let found = poses
        .iter()
        .zip(&focals)
        .any(|(pose, &f)| matches(pose, r, t) && (f - focal).abs() < 1e-6 * focal);
    assert!(found, "{poses:?} {focals:?}");

    // Filtering keeps exactly the solution with aspect ratio closest to 1: the true one.
    let mut poses = Vec::new();
    let mut focals = Vec::new();
    assert_eq!(p4pf(&x, &big_x, &mut poses, &mut focals, true), 1);
    assert!(matches(&poses[0], r, t));
    assert!((focals[0] - focal).abs() < 1e-6 * focal);
}

#[test]
fn rust_only_p4pf_separate_focals_recover_ground_truth() {
    let (fx, fy) = (1100.0, 1300.0);
    let (r, t, x, big_x) = scene(fx, fy);

    let mut poses = Vec::new();
    let mut out_fx = Vec::new();
    let mut out_fy = Vec::new();
    let n = p4pf_fx_fy(&x, &big_x, &mut poses, &mut out_fx, &mut out_fy, true);
    assert!(n >= 1);
    let found = (0..n).any(|i| {
        matches(&poses[i], r, t)
            && (out_fx[i] - fx).abs() < 1e-6 * fx
            && (out_fy[i] - fy).abs() < 1e-6 * fy
    });
    assert!(found, "{poses:?} {out_fx:?} {out_fy:?}");

    // Cheirality filtering: every kept pose sees all four points in front of it.
    for pose in &poses {
        for p in &big_x {
            assert!((pose.r() * *p + pose.t).z >= 0.0);
        }
    }
}

#[test]
fn rust_only_poselib_camera_init_params() {
    // No parameters given: focal 1.2 * max(w, h), principal point at the center.
    let camera = PoseLibCamera::from_name("PINHOLE", &[], 640, 480);
    assert_eq!(camera.model_id, 1);
    assert_eq!(camera.params, vec![768.0, 768.0, 320.0, 240.0]);
    assert_eq!(camera.focal(), 768.0);
    assert_eq!(camera.model_name(), "PINHOLE");

    // Given parameters are resized to the model's count, not re-initialized.
    let camera = PoseLibCamera::from_name("SIMPLE_RADIAL", &[1000.0, 1.0], 0, 0);
    assert_eq!(camera.params, vec![1000.0, 1.0, 0.0, 0.0]);
    let camera = PoseLibCamera::from_name("SIMPLE_PINHOLE", &[5.0, 1.0, 2.0, 9.0], 0, 0);
    assert_eq!(camera.params, vec![5.0, 1.0, 2.0]);

    // Without a size: focal 1, principal point 0.
    let camera = PoseLibCamera::from_name("SIMPLE_PINHOLE", &[], 0, 0);
    assert_eq!(camera.params, vec![1.0, 0.0, 0.0]);

    // An empty camera has focal 1; so do models without focal parameters and unknown ids.
    // An unknown name maps to INVALID (-1), which is also the NULL model's id, so its
    // parameters are resized to NULL's zero.
    assert_eq!(PoseLibCamera::from_name("NULL", &[], 0, 0).focal(), 1.0);
    assert_eq!(
        PoseLibCamera::from_name("SPHERICAL", &[3.0, 4.0], 0, 0).focal(),
        1.0
    );
    let unknown = PoseLibCamera::from_name("NOT_A_MODEL", &[3.0], 0, 0);
    assert_eq!(unknown.model_id, -1);
    assert_eq!(unknown.model_name(), "NULL");
    assert!(unknown.params.is_empty());
    assert_eq!(unknown.focal(), 1.0);
    let unknown_id = PoseLibCamera::new(42, &[3.0], 0, 0);
    assert_eq!(unknown_id.model_name(), "INVALID_MODEL");
    assert_eq!(unknown_id.params, vec![3.0]);
    assert_eq!(unknown_id.focal(), 1.0);

    assert_eq!(id_from_string("1D_RADIAL"), 99);
    assert_eq!(name_from_id(102), "SIMPLE_DIVISION");
}
