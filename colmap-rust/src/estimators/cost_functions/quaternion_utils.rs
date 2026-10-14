//! Port of COLMAP's `colmap/estimators/cost_functions/quaternion_utils.h`, via colmap-sharp's
//! `Estimators/CostFunctions/QuaternionUtils.cs`. The two Ceres `rotation.h` conversions it
//! wraps (`ceres::AngleAxisToQuaternion`, `ceres::QuaternionToAngleAxis`) live in
//! [`crate::solver::rotation`] and are re-exported here under their `ceres_` names. Tests:
//! `tests/estimators/quaternion_utils.rs` (`quaternion_utils_test.cc`).
//!
//! Convention: Eigen quaternion storage order (x, y, z, w), Hamilton product. A quaternion
//! q = (x, y, z, w) represents the rotation matrix R(q) = (w^2 - ||v||^2) I + 2 v v^T +
//! 2 w `[v]_x`, where v = (x, y, z) is the vector part. The 4-vector representation used in
//! the matrices is [x, y, z, w] (Eigen's `coeffs()`). The Ceres conversions use Ceres'
//! order [w, x, y, z]; the Eigen-order wrappers swap.
//!
//! [`super::tiny_manifold`] retracts with [`eigen_quaternion_from_angle_axis`] and takes its
//! Plus Jacobian from [`quaternion_left_mult_matrix`]. The angle-axis wrappers are generic
//! over [`Scalar`], so autodiff cost functors call them on Jets.

use crate::linalg::{Matrix4d, Quaterniond, Vector3d};
use crate::solver::Scalar;

/// Port of `ceres::AngleAxisToQuaternion`: [`crate::solver::rotation::angle_axis_to_quaternion`]
/// under the name this module has always exported.
pub use crate::solver::rotation::angle_axis_to_quaternion as ceres_angle_axis_to_quaternion;

/// Port of `ceres::QuaternionToAngleAxis`: [`crate::solver::rotation::quaternion_to_angle_axis`]
/// under the name this module has always exported.
pub use crate::solver::rotation::quaternion_to_angle_axis as ceres_quaternion_to_angle_axis;

/// Port of `colmap::AngleAxisFromEigenQuaternion`: the angle-axis vector of an
/// (x, y, z, w) quaternion, through `ceres::QuaternionToAngleAxis`. Generic over [`Scalar`]
/// (doubles or Jets), as the C++ template is.
pub fn angle_axis_from_eigen_quaternion<T: Scalar>(eigen_quaternion: &[T], angle_axis: &mut [T]) {
    let quaternion = [
        eigen_quaternion[3],
        eigen_quaternion[0],
        eigen_quaternion[1],
        eigen_quaternion[2],
    ];
    ceres_quaternion_to_angle_axis(&quaternion, angle_axis);
}

/// Port of `colmap::EigenQuaternionFromAngleAxis`: the (x, y, z, w) unit quaternion of an
/// angle-axis vector, through `ceres::AngleAxisToQuaternion`. Generic over [`Scalar`]
/// (doubles or Jets), as the C++ template is.
pub fn eigen_quaternion_from_angle_axis<T: Scalar>(angle_axis: &[T], eigen_quaternion: &mut [T]) {
    let mut quaternion = [T::from_f64(0.0); 4];
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
