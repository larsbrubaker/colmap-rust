// Rust-only tests (PoseLib's camera_pose.h has no test in COLMAP or colmap-sharp) of
// colmap_rust::estimators::solvers::poselib::camera_pose. Tier B: rotation round trips
// within 1e-12, quaternions in PoseLib's (w, x, y, z) order.

use colmap_rust::estimators::solvers::poselib::camera_pose::{
    quat_multiply, quat_to_rotmat, rotmat_to_quat,
};
use colmap_rust::estimators::solvers::poselib::CameraPose;
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Matrix3x4d, Vector3d, Vector4d};

#[test]
fn rust_only_identity_and_round_trip() {
    let identity = CameraPose::default();
    assert_eq!(identity.q, Vector4d::new(1.0, 0.0, 0.0, 0.0));
    assert_eq!(identity.r(), Matrix3d::identity());
    assert_eq!(identity.rt(), Matrix3x4d::identity());

    let r = AngleAxisd::new(0.9, Vector3d::new(0.2, -0.6, 0.3).normalized()).to_rotation_matrix();
    let t = Vector3d::new(1.0, -2.0, 0.5);
    let pose = CameraPose::from_rt(r, t);
    assert!((pose.q.norm() - 1.0).abs() < 1e-15);
    assert!((pose.r() - r).norm() < 1e-12);
    assert_eq!(pose.rt(), Matrix3x4d::from_blocks(pose.r(), t));
}

#[test]
fn rust_only_quaternion_order_is_w_first() {
    // A quarter turn about z: (w, x, y, z) = (cos 45, 0, 0, sin 45).
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let q = Vector4d::new(h, 0.0, 0.0, h);
    let r = quat_to_rotmat(q);
    let expected = Matrix3d::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    assert!((r - expected).norm() < 1e-15);
    assert!((rotmat_to_quat(r) - q).norm() < 1e-15);

    // Composing two quarter turns gives a half turn about z.
    let q2 = quat_multiply(q, q);
    assert!((q2 - Vector4d::new(0.0, 0.0, 0.0, 1.0)).norm() < 1e-15);
    let r2 = quat_to_rotmat(q2);
    assert!((r2 - r * r).norm() < 1e-15);
}
