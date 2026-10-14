// Rust-only tests (PoseLib's misc/essential.cc has no unit test COLMAP ships) of
// colmap_rust::estimators::solvers::poselib::essential. Tier B: motion_from_essential on the
// essential matrix of a known pose returns exactly that pose (translation at unit scale)
// when every point is in front of both cameras; check_cheirality rejects a point behind a
// camera; quat_rotate agrees with the rotation matrix.

use colmap_rust::estimators::solvers::poselib::camera_pose::quat_to_rotmat;
use colmap_rust::estimators::solvers::poselib::essential::{
    check_cheirality, motion_from_essential, quat_rotate,
};
use colmap_rust::estimators::solvers::poselib::CameraPose;
use colmap_rust::geometry::essential_matrix::essential_matrix_from_pose;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Vector3d, Vector4d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::random_eigen_vector3d;

fn random_cam2_from_cam1() -> Rigid3d {
    let axis = random_eigen_vector3d().normalized();
    let rotation = AngleAxisd::new(random_uniform_real(-0.5, 0.5), axis).to_quaternion();
    Rigid3d::new(rotation, random_eigen_vector3d().normalized())
}

/// `n` correspondences of points in front of both cameras, as unit bearings.
fn random_bearings(cam2_from_cam1: &Rigid3d, n: usize) -> (Vec<Vector3d>, Vec<Vector3d>) {
    let (mut x1, mut x2) = (Vec::new(), Vec::new());
    while x1.len() < n {
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
        x1.push(point1.normalized());
        x2.push(point2.normalized());
    }
    (x1, x2)
}

#[test]
fn rust_only_motion_from_essential_recovers_true_pose() {
    set_prng_seed(1);
    for _ in 0..100 {
        let cam2_from_cam1 = random_cam2_from_cam1();
        let (x1, x2) = random_bearings(&cam2_from_cam1, 8);
        let e = essential_matrix_from_pose(&cam2_from_cam1);

        let mut poses = Vec::new();
        motion_from_essential(&e, &x1, &x2, &mut poses);

        assert_eq!(poses.len(), 1, "exactly one of the four poses is cheiral");
        let r_true = cam2_from_cam1.rotation.to_rotation_matrix();
        assert!((poses[0].r() - r_true).norm() < 1e-10);
        let t_true = cam2_from_cam1.translation.normalized();
        assert!((poses[0].t - t_true).norm() < 1e-10);
        assert!((poses[0].q.norm() - 1.0).abs() < 1e-12);
    }
}

#[test]
fn rust_only_motion_from_essential_appends() {
    set_prng_seed(2);
    let cam2_from_cam1 = random_cam2_from_cam1();
    let (x1, x2) = random_bearings(&cam2_from_cam1, 6);
    let e = essential_matrix_from_pose(&cam2_from_cam1);

    // Existing entries are kept, as in PoseLib (the solvers clear the vector themselves).
    let mut poses = vec![CameraPose::identity()];
    motion_from_essential(&e, &x1, &x2, &mut poses);
    assert_eq!(poses.len(), 2);
    assert_eq!(poses[0], CameraPose::identity());
}

#[test]
fn rust_only_check_cheirality() {
    // Camera 2 is camera 1 moved one unit along -x: x_cam2 = x_cam1 + (1, 0, 0).
    let pose = CameraPose::new(
        Vector4d::new(1.0, 0.0, 0.0, 0.0),
        Vector3d::new(1.0, 0.0, 0.0),
    );
    let front = Vector3d::new(0.0, 0.0, 4.0);
    let x1 = front.normalized();
    let x2 = (front + pose.t).normalized();
    assert!(check_cheirality(&pose, x1, x2, 0.0));
    // The point is 4 units deep in camera 1: a minimum depth above that rejects it.
    assert!(check_cheirality(&pose, x1, x2, 3.9));
    assert!(!check_cheirality(&pose, x1, x2, 4.1));

    // The same rays pointing backwards triangulate behind both cameras.
    assert!(!check_cheirality(&pose, -x1, -x2, 0.0));
}

#[test]
fn rust_only_quat_rotate_matches_rotation_matrix() {
    set_prng_seed(3);
    for _ in 0..20 {
        let q = Vector4d::new(
            random_uniform_real(-1.0, 1.0),
            random_uniform_real(-1.0, 1.0),
            random_uniform_real(-1.0, 1.0),
            random_uniform_real(-1.0, 1.0),
        )
        .normalized();
        let p = random_eigen_vector3d();
        let expected = quat_to_rotmat(q) * p;
        assert!((quat_rotate(q, p) - expected).norm() < 1e-14);
    }
}
