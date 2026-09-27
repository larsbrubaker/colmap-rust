// Port of COLMAP's src/colmap/geometry/pose_test.cc, the cases whose functions need no
// matrix decomposition, plus Rust-only tests of `transform_camera_world` and
// `CamRayWithJac::zero` (pose_test.cc has none).
//
// Deferred until the dynamic-size decompositions land (they test SVD/QR-based functions):
// ComputeClosestRotationMatrix.Nominal, DecomposeProjectionMatrix.Nominal,
// AverageQuaternions.Nominal, AverageUnitVectors.Nominal, AverageDirections.Nominal,
// GravityAlignedRotation.Nominal.

use colmap_rust::geometry::pose::{
    angle_axis_to_rotation_matrix, check_cheirality, euler_angles_to_rotation_matrix,
    interpolate_camera_poses, left_jacobian_from_angle_axis, quaternion_from_angle_axis,
    right_jacobian_from_angle_axis, rotation_from_y_axis_angle, rotation_matrix_to_angle_axis,
    rotation_matrix_to_euler_angles, transform_camera_world, y_axis_angle_from_rotation,
};
use colmap_rust::geometry::{cross_product_matrix, CamRayWithJac, Rigid3d, Sim3d};
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Matrix3x2d, Quaterniond, Vector3d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};

use super::{eigen_matrix_near, eigen_matrix_near_default, expect_near};

fn random_omega() -> Vector3d {
    let aa = AngleAxisd::from_quaternion(random_eigen_quaterniond());
    aa.angle * aa.axis
}

#[test]
fn rotation_matrix_to_angle_axis_roundtrip() {
    set_prng_seed(0);
    let r = random_eigen_quaterniond().to_rotation_matrix();
    assert!(eigen_matrix_near(
        &angle_axis_to_rotation_matrix(rotation_matrix_to_angle_axis(&r)),
        &r,
        1e-6
    ));
}

#[test]
fn angle_axis_to_rotation_matrix_roundtrip() {
    set_prng_seed(0);
    let w = random_omega();
    assert!(eigen_matrix_near(
        &rotation_matrix_to_angle_axis(&angle_axis_to_rotation_matrix(w)),
        &w,
        1e-6
    ));
}

fn check_euler_roundtrip(rx: f64, ry: f64, rz: f64) {
    let (rxx, ryy, rzz) =
        rotation_matrix_to_euler_angles(&euler_angles_to_rotation_matrix(rx, ry, rz));
    expect_near(rx, rxx, 1e-6);
    expect_near(ry, ryy, 1e-6);
    expect_near(rz, rzz, 1e-6);
}

#[test]
fn euler_angles_x() {
    check_euler_roundtrip(0.3, 0.0, 0.0);
}

#[test]
fn euler_angles_y() {
    check_euler_roundtrip(0.0, 0.3, 0.0);
}

#[test]
fn euler_angles_z() {
    check_euler_roundtrip(0.0, 0.0, 0.3);
}

#[test]
fn euler_angles_xyz() {
    check_euler_roundtrip(0.1, 0.2, 0.3);
}

#[test]
fn interpolate_camera_poses_nominal() {
    set_prng_seed(0);
    let cam_from_world1 = Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d());
    let cam_from_world2 = Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d());

    let interp1 = interpolate_camera_poses(&cam_from_world1, &cam_from_world2, 0.0);
    assert!(eigen_matrix_near_default(
        &interp1.translation,
        &cam_from_world1.translation
    ));

    let interp2 = interpolate_camera_poses(&cam_from_world1, &cam_from_world2, 1.0);
    assert!(eigen_matrix_near_default(
        &interp2.translation,
        &cam_from_world2.translation
    ));

    let interp3 = interpolate_camera_poses(&cam_from_world1, &cam_from_world2, 0.5);
    assert!(eigen_matrix_near_default(
        &interp3.translation,
        &((cam_from_world1.translation + cam_from_world2.translation) / 2.0)
    ));
}

#[test]
fn check_cheirality_nominal() {
    let cam2_from_cam1 = Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0));
    let mut rays1 = vec![Vector3d::new(0.0, 0.0, 1.0).normalized()];
    let mut rays2 = vec![Vector3d::new(0.1, 0.0, 1.0).normalized()];
    let valid = check_cheirality(&cam2_from_cam1, &rays1, &rays2).unwrap();
    assert!(!valid.is_empty());
    assert_eq!(valid.len(), 1);

    rays1.push(Vector3d::new(0.0, 0.0, 1.0).normalized());
    rays2.push(Vector3d::new(-0.1, 0.0, 1.0).normalized());
    let valid = check_cheirality(&cam2_from_cam1, &rays1, &rays2).unwrap();
    assert!(!valid.is_empty());
    assert_eq!(valid.len(), 1);

    rays2[1].x = 0.2;
    let valid = check_cheirality(&cam2_from_cam1, &rays1, &rays2).unwrap();
    assert!(!valid.is_empty());
    assert_eq!(valid.len(), 2);

    rays2[0].x = -0.2;
    rays2[1].x = -0.2;
    let valid = check_cheirality(&cam2_from_cam1, &rays1, &rays2).unwrap();
    assert!(valid.is_empty());
    assert_eq!(valid.len(), 0);
}

#[test]
fn y_axis_angle_from_rotation_roundtrip() {
    set_prng_seed(0);
    for _ in 0..100 {
        let angle = random_uniform_real::<f64>(-std::f64::consts::PI, std::f64::consts::PI);
        let r = rotation_from_y_axis_angle(angle);
        let recovered_angle = y_axis_angle_from_rotation(&r);
        assert!(eigen_matrix_near(
            &rotation_from_y_axis_angle(recovered_angle),
            &r,
            1e-6
        ));
    }
}

#[test]
fn rotation_from_y_axis_angle_nominal() {
    let r = rotation_from_y_axis_angle(0.0);
    assert!(eigen_matrix_near(&r, &Matrix3d::identity(), 1e-6));
    expect_near(y_axis_angle_from_rotation(&r), 0.0, 1e-6);

    let r = rotation_from_y_axis_angle(std::f64::consts::PI / 2.0);
    assert!(eigen_matrix_near(
        &(r * Vector3d::unit_x()),
        &(-Vector3d::unit_z()),
        1e-6
    ));
}

#[test]
fn quaternion_from_angle_axis_zero() {
    let q = quaternion_from_angle_axis(Vector3d::zeros());
    expect_near(q.w, 1.0, 1e-12);
    expect_near(q.vec().norm(), 0.0, 1e-12);
}

#[test]
fn quaternion_from_angle_axis_small_angle() {
    // Just above and below the threshold — results should be continuous.
    let axis = Vector3d::new(1.0, 2.0, 3.0).normalized();
    let q_small = quaternion_from_angle_axis(axis * 1e-11);
    let q_medium = quaternion_from_angle_axis(axis * 1e-9);
    // Both should be near identity but preserve direction.
    expect_near(q_small.angular_distance(q_medium), 0.0, 1e-8);
    // Small angle should NOT snap to identity — it should preserve direction.
    let recovered = AngleAxisd::from_quaternion(q_small).axis.normalized();
    expect_near(recovered.dot(axis).abs(), 1.0, 1e-6);
}

#[test]
fn quaternion_from_angle_axis_roundtrip() {
    set_prng_seed(0);
    for _ in 0..100 {
        let aa = AngleAxisd::from_quaternion(random_eigen_quaterniond());
        let omega = aa.angle * aa.axis;
        let q = quaternion_from_angle_axis(omega);
        assert!(eigen_matrix_near(
            &q.to_rotation_matrix(),
            &aa.to_rotation_matrix(),
            1e-10
        ));
    }
}

#[test]
fn left_jacobian_from_angle_axis_identity_at_zero() {
    let jl = left_jacobian_from_angle_axis(Vector3d::zeros());
    assert!(eigen_matrix_near(&jl, &Matrix3d::identity(), 1e-10));
}

#[test]
fn left_jacobian_from_angle_axis_relation_to_right() {
    // Jr(w) = Jl(-w) for all w.
    set_prng_seed(0);
    for _ in 0..100 {
        let omega = random_omega();
        let jr = right_jacobian_from_angle_axis(omega);
        let jl_neg = left_jacobian_from_angle_axis(-omega);
        assert!(eigen_matrix_near(&jr, &jl_neg, 1e-10));
    }
}

#[test]
fn right_jacobian_from_angle_axis_small_angle() {
    // Near zero, Jr ≈ I - 0.5 * [w]_x.
    let omega = Vector3d::new(1e-12, 2e-12, 3e-12);
    let jr = right_jacobian_from_angle_axis(omega);
    let mut expected = Matrix3d::identity();
    expected -= 0.5 * cross_product_matrix(omega);
    assert!(eigen_matrix_near(&jr, &expected, 1e-10));
}

// Numeric Jacobian of Exp: column k is Log(combine(R, R_perturbed)) / eps.
fn numeric_jacobian(omega: Vector3d, combine: impl Fn(Matrix3d, Matrix3d) -> Matrix3d) -> Matrix3d {
    let eps = 1e-7;
    let r = angle_axis_to_rotation_matrix(omega);
    let mut columns = [Vector3d::zeros(); 3];
    for (k, column) in columns.iter_mut().enumerate() {
        let mut dw = Vector3d::zeros();
        dw[k] = eps;
        let r_perturbed = angle_axis_to_rotation_matrix(omega + dw);
        let log_dr = rotation_matrix_to_angle_axis(&combine(r, r_perturbed));
        *column = log_dr / eps;
    }
    Matrix3d::from_columns(columns[0], columns[1], columns[2])
}

#[test]
fn right_jacobian_from_angle_axis_numeric_derivative() {
    // Verify Jr by numeric differentiation of Exp(w + dw) ≈ Exp(w) * Exp(Jr*dw).
    set_prng_seed(0);
    for _ in 0..50 {
        let omega = random_omega();
        let jr = right_jacobian_from_angle_axis(omega);
        // R_perturbed ≈ R * Exp(Jr * dw), so Exp(Jr*dw) ≈ R^T * R_perturbed
        let jr_numeric = numeric_jacobian(omega, |r, rp| r.transpose() * rp);
        assert!(eigen_matrix_near(&jr, &jr_numeric, 1e-5));
    }
}

#[test]
fn left_jacobian_from_angle_axis_numeric_derivative() {
    // Verify Jl by numeric differentiation of Exp(w + dw) ≈ Exp(Jl*dw) * Exp(w).
    set_prng_seed(0);
    for _ in 0..50 {
        let omega = random_omega();
        let jl = left_jacobian_from_angle_axis(omega);
        // R_perturbed ≈ Exp(Jl * dw) * R, so Exp(Jl*dw) ≈ R_perturbed * R^T.
        let jl_numeric = numeric_jacobian(omega, |r, rp| rp * r.transpose());
        assert!(eigen_matrix_near(&jl, &jl_numeric, 1e-5));
    }
}

#[test]
fn rust_only_transform_camera_world() {
    // A camera center c in the old world is new_from_old * c in the new world, and the
    // transformed pose must project the new-world point to the same camera-frame point.
    set_prng_seed(0);
    let new_from_old = Sim3d::new(
        random_uniform_real::<f64>(0.5, 2.0),
        random_eigen_quaterniond(),
        random_eigen_vector3d(),
    );
    let cam_from_world = Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d());
    let cam_from_new_world = transform_camera_world(&new_from_old, &cam_from_world);
    for _ in 0..10 {
        let x_old = random_eigen_vector3d();
        let x_cam = cam_from_world * x_old;
        let x_cam_new = cam_from_new_world * (new_from_old * x_old);
        assert!(eigen_matrix_near(
            &(x_cam_new / new_from_old.scale),
            &x_cam,
            1e-9
        ));
    }
}

#[test]
fn rust_only_cam_ray_with_jac_zero() {
    let zero = CamRayWithJac::zero();
    assert_eq!(zero.ray, Vector3d::zeros());
    assert_eq!(zero.jacobian, Matrix3x2d::zeros());
}
