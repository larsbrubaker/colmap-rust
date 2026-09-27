//! Port of the decomposition-free part of COLMAP's `colmap/geometry/pose.h` and `pose.cc`:
//! [`CamRayWithJac`], angle-axis and Euler-angle conversions, the so(3) exponential and its
//! left/right Jacobians, pose interpolation, the cheirality test, re-expressing a camera pose
//! in a similarity-transformed world, and the yaw-about-Y helpers. Port of colmap-sharp's
//! `Geometry/Pose.cs` and `CamRayWithJac.cs`. Neighbors: [`super::rigid3`], [`super::sim3`].
//! Tests: `tests/geometry/pose.rs` (pose_test.cc).
//!
//! Not here yet, because they need an SVD or a QR (the dynamic-size decompositions):
//! `AverageUnitVectors`, `AverageDirections`, `AverageQuaternions`,
//! `ComputeClosestRotationMatrix`, `DecomposeProjectionMatrix` and `GravityAlignedRotation`.
//!
//! Tier A (scalar arithmetic in COLMAP's order; transcendentals through [`crate::math::fns`],
//! docs/CPP_DIVERGENCES.md entry 1).

use super::rigid3::{cross_product_matrix, Rigid3d};
use super::sim3::Sim3d;
use crate::linalg::{AngleAxisd, Matrix3d, Matrix3x2d, Quaterniond, Vector3d};
use crate::math::fns;
use crate::{check_eq, Result};

/// A unit bearing and the Jacobian d(ray)/d(pixel) of its unprojection (see
/// `Camera::CamRayFromImgWithJac`), bundled so RANSAC subsampling keeps them index-aligned.
/// Port of `colmap::CamRayWithJac`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamRayWithJac {
    /// The unit bearing.
    pub ray: Vector3d,
    /// d(ray)/d(pixel).
    pub jacobian: Matrix3x2d,
}

impl CamRayWithJac {
    /// `CamRayWithJac::Zero()`: the fallback when unprojection fails. The estimators take a
    /// dense list, not options, so a failed ray is kept as zero, which the tangent Sampson
    /// residual scores as infinite (rejected).
    pub const fn zero() -> Self {
        Self {
            ray: Vector3d::new(0.0, 0.0, 0.0),
            jacobian: Matrix3x2d::zeros(),
        }
    }
}

/// Port of `colmap::RotationMatrixToAngleAxis`: `angle * axis` of `Eigen::AngleAxisd(R)`.
pub fn rotation_matrix_to_angle_axis(r: &Matrix3d) -> Vector3d {
    let aa = AngleAxisd::from_rotation_matrix(*r);
    aa.angle * aa.axis
}

/// Port of `colmap::AngleAxisToRotationMatrix`. Below an angle of 1e-12 it uses the small
/// angle approximation `I + [w]_x`.
pub fn angle_axis_to_rotation_matrix(w: Vector3d) -> Matrix3d {
    let angle = w.norm();
    if angle > 1e-12 {
        AngleAxisd::new(angle, w / angle).to_rotation_matrix()
    } else {
        // Small angle approximation: I + [w]_x.
        Matrix3d::new(
            1.0, -w.z, w.y, //
            w.z, 1.0, -w.x, //
            -w.y, w.x, 1.0,
        )
    }
}

/// Port of `colmap::RotationMatrixToEulerAngles`: `(rx, ry, rz)` in radians for the
/// convention `R = Rz * Ry * Rx` (right-handed). A NaN angle becomes 0.
pub fn rotation_matrix_to_euler_angles(r: &Matrix3d) -> (f64, f64, f64) {
    let rx = fns::atan2(r[(2, 1)], r[(2, 2)]);
    let ry = fns::asin(-r[(2, 0)]);
    let rz = fns::atan2(r[(1, 0)], r[(0, 0)]);
    let zero_nan = |v: f64| if v.is_nan() { 0.0 } else { v };
    (zero_nan(rx), zero_nan(ry), zero_nan(rz))
}

/// Port of `colmap::EulerAnglesToRotationMatrix`: `Rz * Ry * Rx` from angles in radians.
pub fn euler_angles_to_rotation_matrix(rx: f64, ry: f64, rz: f64) -> Matrix3d {
    let r_x = AngleAxisd::new(rx, Vector3d::unit_x()).to_rotation_matrix();
    let r_y = AngleAxisd::new(ry, Vector3d::unit_y()).to_rotation_matrix();
    let r_z = AngleAxisd::new(rz, Vector3d::unit_z()).to_rotation_matrix();
    r_z * r_y * r_x
}

/// Below this rotation angle the so(3) helpers use their first-order expansions.
const SMALL_ANGLE_THRESHOLD: f64 = 1e-10;

/// Port of `colmap::QuaternionFromAngleAxis`: the quaternion of `omega` in so(3). Below
/// 1e-10 rad it uses the first-order expansion `(1, omega / 2)`, normalized, which keeps the
/// rotation direction instead of snapping to the identity.
pub fn quaternion_from_angle_axis(omega: Vector3d) -> Quaterniond {
    let theta = omega.norm();
    if theta < SMALL_ANGLE_THRESHOLD {
        // First-order Taylor expansion preserving rotation direction.
        return Quaterniond::new(1.0, 0.5 * omega.x, 0.5 * omega.y, 0.5 * omega.z).normalized();
    }
    Quaterniond::from_angle_axis(AngleAxisd::new(theta, omega / theta))
}

/// Port of `colmap::LeftJacobianFromAngleAxis`: the left Jacobian `J_l(omega)` of so(3),
/// used for bias Jacobian propagation in IMU preintegration.
pub fn left_jacobian_from_angle_axis(omega: Vector3d) -> Matrix3d {
    let theta = omega.norm();
    if theta < SMALL_ANGLE_THRESHOLD {
        return Matrix3d::identity() + 0.5 * cross_product_matrix(omega);
    }
    let a = omega / theta;
    let a_x = cross_product_matrix(a);
    let sin_theta = fns::sin(theta);
    let sinc_theta = sin_theta / theta;
    // Eigen evaluates `(1 - sinc) * a * a.transpose()` as the outer product of the scaled
    // vector `(1 - sinc) * a` with `a`.
    let scaled_a = (1.0 - sinc_theta) * a;
    let outer = Matrix3d::from_columns(scaled_a * a.x, scaled_a * a.y, scaled_a * a.z);
    sinc_theta * Matrix3d::identity() + outer + ((1.0 - fns::cos(theta)) / theta) * a_x
}

/// Port of `colmap::RightJacobianFromAngleAxis`: `J_r(omega) = J_l(-omega)`.
pub fn right_jacobian_from_angle_axis(omega: Vector3d) -> Matrix3d {
    left_jacobian_from_angle_axis(-omega)
}

/// Port of `colmap::InterpolateCameraPoses`: slerp of the rotations and linear
/// interpolation of the translations, `t` in [0, 1].
pub fn interpolate_camera_poses(
    cam1_from_world: &Rigid3d,
    cam2_from_world: &Rigid3d,
    t: f64,
) -> Rigid3d {
    let translation12 = cam2_from_world.translation - cam1_from_world.translation;
    Rigid3d::new(
        cam1_from_world.rotation.slerp(t, cam2_from_world.rotation),
        cam1_from_world.translation + translation12 * t,
    )
}

/// Port of `colmap::CheckCheirality`: the indices of the correspondences whose rays
/// triangulate to a point in front of both cameras. COLMAP's `bool` result is
/// `!valid_indices.empty()`.
///
/// The closed-form depth test assumes both rays are unit-normalized; rays of arbitrary scale
/// give incorrect results. The two lists must have equal length.
pub fn check_cheirality(
    cam2_from_cam1: &Rigid3d,
    cam_rays1: &[Vector3d],
    cam_rays2: &[Vector3d],
) -> Result<Vec<usize>> {
    check_eq!(cam_rays1.len(), cam_rays2.len());
    let cam2_from_cam1_rot = cam2_from_cam1.rotation.to_rotation_matrix();
    let t = cam2_from_cam1.translation;
    let mut valid_indices = Vec::new();
    for (i, (&ray1, &ray2)) in cam_rays1.iter().zip(cam_rays2).enumerate() {
        // Solve the 2x2 system for the depths of the point along both rays; both must be
        // positive for the point to lie in front of both cameras. This assumes unit-norm
        // rays: the common positive factor 1 / (1 - a^2) is dropped since it does not affect
        // the sign (a = cos angle between the rays, so |a| <= 1).
        let ray1_in_cam2 = cam2_from_cam1_rot * ray1;
        let a = -ray1_in_cam2.dot(ray2);
        let b1 = -ray1_in_cam2.dot(t);
        let b2 = ray2.dot(t);
        if b1 - a * b2 > 0.0 && b2 - a * b1 > 0.0 {
            valid_indices.push(i);
        }
    }
    Ok(valid_indices)
}

/// Port of `colmap::TransformCameraWorld`: `cam_from_world` re-expressed as
/// `cam_from_new_world`, given `new_from_old_world`.
pub fn transform_camera_world(new_from_old_world: &Sim3d, cam_from_world: &Rigid3d) -> Rigid3d {
    let cam_from_new_world = Sim3d::new(1.0, cam_from_world.rotation, cam_from_world.translation)
        * new_from_old_world.inverse();
    Rigid3d::new(
        cam_from_new_world.rotation,
        cam_from_new_world.translation * new_from_old_world.scale,
    )
}

/// Port of `colmap::YAxisAngleFromRotation`: the yaw (rotation about Y) of a gravity-aligned
/// rotation, the y component of its angle-axis vector.
pub fn y_axis_angle_from_rotation(rotation: &Matrix3d) -> f64 {
    rotation_matrix_to_angle_axis(rotation).y
}

/// Port of `colmap::RotationFromYAxisAngle`: the rotation by `angle` about the Y axis.
pub fn rotation_from_y_axis_angle(angle: f64) -> Matrix3d {
    angle_axis_to_rotation_matrix(Vector3d::new(0.0, angle, 0.0))
}
