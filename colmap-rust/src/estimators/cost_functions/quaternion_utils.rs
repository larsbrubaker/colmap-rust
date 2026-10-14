//! Port of COLMAP's `colmap/estimators/cost_functions/quaternion_utils.h` (double forms), via
//! colmap-sharp's `Estimators/CostFunctions/QuaternionUtils.cs`, with the two Ceres
//! `rotation.h` conversions it wraps (`ceres::AngleAxisToQuaternion`,
//! `ceres::QuaternionToAngleAxis`; Ceres Solver, BSD-3-Clause, see THIRD_PARTY_NOTICES.md,
//! via colmap-sharp's `Solver/Rotation.cs`). Tests: `tests/estimators/quaternion_utils.rs`
//! (`quaternion_utils_test.cc`).
//!
//! Convention: Eigen quaternion storage order (x, y, z, w), Hamilton product. A quaternion
//! q = (x, y, z, w) represents the rotation matrix R(q) = (w^2 - ||v||^2) I + 2 v v^T +
//! 2 w [v]_x, where v = (x, y, z) is the vector part. The 4-vector representation used in
//! the matrices is [x, y, z, w] (Eigen's `coeffs()`). The Ceres conversions use Ceres'
//! order [w, x, y, z]; the Eigen-order wrappers swap.
//!
//! [`super::tiny_manifold`] retracts with [`eigen_quaternion_from_angle_axis`] and takes its
//! Plus Jacobian from [`quaternion_left_mult_matrix`]. The generic (Jet) forms of the
//! angle-axis wrappers arrive with the solver phase, which owns the general Jet.

use crate::linalg::{Matrix4d, Quaterniond, Vector3d};
use crate::math::fns;

/// Port of `ceres::AngleAxisToQuaternion` on doubles: the unit quaternion [w, x, y, z] of the
/// rotation by |angle_axis| radians about angle_axis. At the origin it uses the first-order
/// Taylor expansion (w = 1, xyz = angle_axis / 2), as Ceres does so Jets get correct
/// derivatives there.
pub fn ceres_angle_axis_to_quaternion(angle_axis: &[f64], quaternion: &mut [f64]) {
    let a0 = angle_axis[0];
    let a1 = angle_axis[1];
    let a2 = angle_axis[2];
    let theta = fns::hypot3(a0, a1, a2);

    // For points not at the origin, the full conversion is numerically stable.
    if theta != 0.0 {
        let half_theta = theta * 0.5;
        let k = fns::sin(half_theta) / theta;
        quaternion[0] = fns::cos(half_theta);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    } else {
        // At the origin, sqrt() will produce NaN in the derivative since the argument is
        // zero. By approximating with a Taylor series, and truncating at one term, the value
        // and first derivatives will be computed correctly when Jets are used.
        let k = 0.5;
        quaternion[0] = 1.0;
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
}

/// Port of `ceres::QuaternionToAngleAxis` on doubles: the angle-axis vector of the rotation
/// by a unit quaternion [w, x, y, z], normalized to an angle of at most pi. For zero rotation
/// it uses the first-order Taylor expansion (angle_axis = 2 xyz).
pub fn ceres_quaternion_to_angle_axis(quaternion: &[f64], angle_axis: &mut [f64]) {
    let q1 = quaternion[1];
    let q2 = quaternion[2];
    let q3 = quaternion[3];
    let sin_theta = fns::hypot3(q1, q2, q3);

    // For quaternions representing non-zero rotation, the conversion is numerically stable.
    if sin_theta != 0.0 {
        let cos_theta = quaternion[0];

        // If cos_theta is negative, theta is greater than pi/2, which means that angle for
        // the angle_axis vector which is 2 * theta would be greater than pi. While this
        // would still be the correct rotation, it would not be a normalized angle-axis
        // vector. In that case 2 * theta ~ 2 * theta - 2 * pi, which is to say
        // theta - pi = atan(sin(theta - pi), cos(theta - pi)) = atan(-sin(theta), -cos(theta)).
        let two_theta = if cos_theta < 0.0 {
            2.0 * fns::atan2(-sin_theta, -cos_theta)
        } else {
            2.0 * fns::atan2(sin_theta, cos_theta)
        };
        let k = two_theta / sin_theta;
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    } else {
        // For zero rotation, sqrt() will produce NaN in the derivative since the argument is
        // zero. By approximating with a Taylor series, and truncating at one term, the value
        // and first derivatives will be computed correctly when Jets are used.
        let k = 2.0;
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }
}

/// Port of `colmap::AngleAxisFromEigenQuaternion` (double form): the angle-axis vector of
/// an (x, y, z, w) quaternion, through `ceres::QuaternionToAngleAxis`.
pub fn angle_axis_from_eigen_quaternion(eigen_quaternion: &[f64], angle_axis: &mut [f64]) {
    let quaternion = [
        eigen_quaternion[3],
        eigen_quaternion[0],
        eigen_quaternion[1],
        eigen_quaternion[2],
    ];
    ceres_quaternion_to_angle_axis(&quaternion, angle_axis);
}

/// Port of `colmap::EigenQuaternionFromAngleAxis` (double form): the (x, y, z, w) unit
/// quaternion of an angle-axis vector, through `ceres::AngleAxisToQuaternion`.
pub fn eigen_quaternion_from_angle_axis(angle_axis: &[f64], eigen_quaternion: &mut [f64]) {
    let mut quaternion = [0.0; 4];
    ceres_angle_axis_to_quaternion(angle_axis, &mut quaternion);
    eigen_quaternion[0] = quaternion[1];
    eigen_quaternion[1] = quaternion[2];
    eigen_quaternion[2] = quaternion[3];
    eigen_quaternion[3] = quaternion[0];
}

/// Port of `colmap::QuaternionLeftMultMatrix`: the Hamilton left-multiplication matrix in
/// xyzw storage, `QuaternionLeftMultMatrix(q) * p = q * p` (as 4-vectors).
pub fn quaternion_left_mult_matrix(q: Quaterniond) -> Matrix4d {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    Matrix4d::new(
        w, -z, y, x, //
        z, w, -x, y, //
        -y, x, w, z, //
        -x, -y, -z, w,
    )
}

/// Port of `colmap::QuaternionRightMultMatrix`: the Hamilton right-multiplication matrix in
/// xyzw storage, `QuaternionRightMultMatrix(p) * q = q * p` (as 4-vectors).
pub fn quaternion_right_mult_matrix(q: Quaterniond) -> Matrix4d {
    let (x, y, z, w) = (q.x, q.y, q.z, q.w);
    Matrix4d::new(
        w, z, -y, x, //
        -z, w, x, y, //
        y, -x, w, z, //
        -x, -y, -z, w,
    )
}

/// Port of `colmap::QuaternionRotatePointWithJac`: rotates `pt` by the (x, y, z, w)
/// quaternion `q` and, when `j_out` is given (C++'s non-null `J_out`), writes the row-major
/// 3x4 Jacobian of R(q) * pt with respect to q.
pub fn quaternion_rotate_point_with_jac(
    q: &[f64],
    pt: &[f64],
    j_out: Option<&mut [f64]>,
) -> Vector3d {
    let (qx, qy, qz, qw) = (q[0], q[1], q[2], q[3]);
    let (px, py, pz) = (pt[0], pt[1], pt[2]);

    let qx_py = qx * py;
    let qx_pz = qx * pz;
    let qy_px = qy * px;
    let qy_pz = qy * pz;
    let qz_px = qz * px;
    let qz_py = qz * py;

    // R(q) * p = p + 2*w*(v x p) + 2*(v x (v x p))
    let v_x_p0 = qy_pz - qz_py;
    let v_x_p1 = qz_px - qx_pz;
    let v_x_p2 = qx_py - qy_px;

    let v_x_v_x_p0 = qy * v_x_p2 - qz * v_x_p1;
    let v_x_v_x_p1 = qz * v_x_p0 - qx * v_x_p2;
    let v_x_v_x_p2 = qx * v_x_p1 - qy * v_x_p0;

    let pt_out = Vector3d::new(
        px + 2.0 * (qw * v_x_p0 + v_x_v_x_p0),
        py + 2.0 * (qw * v_x_p1 + v_x_v_x_p1),
        pz + 2.0 * (qw * v_x_p2 + v_x_v_x_p2),
    );

    if let Some(j) = j_out {
        let qx_px = qx * px;
        let qy_py = qy * py;
        let qz_pz = qz * pz;
        let qw_px = qw * px;
        let qw_py = qw * py;
        let qw_pz = qw * pz;

        j[0] = 2.0 * (qy_py + qz_pz);
        j[1] = 2.0 * (-2.0 * qy_px + qx_py + qw_pz);
        j[2] = 2.0 * (-2.0 * qz_px - qw_py + qx_pz);
        j[3] = 2.0 * (-qz_py + qy_pz);

        j[4] = 2.0 * (qy_px - 2.0 * qx_py - qw_pz);
        j[5] = 2.0 * (qx_px + qz_pz);
        j[6] = 2.0 * (qw_px - 2.0 * qz_py + qy_pz);
        j[7] = 2.0 * (qz_px - qx_pz);

        j[8] = 2.0 * (qz_px + qw_py - 2.0 * qx_pz);
        j[9] = 2.0 * (-qw_px + qz_py - 2.0 * qy_pz);
        j[10] = 2.0 * (qx_px + qy_py);
        j[11] = 2.0 * (-qy_px + qx_py);
    }

    pt_out
}
