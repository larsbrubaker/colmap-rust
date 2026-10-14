//! Port of the parts of PoseLib's `PoseLib/misc/essential.cc` that the six-point relative
//! pose solvers use (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
//! THIRD_PARTY_NOTICES.md; commit fa7280fee27f97aff31ae7f98bab7f583fac7d08), through
//! colmap-sharp's `Estimators/Solvers/PoseLib/Essential.cs`:
//!
//! - [`motion_from_essential`]: the SVD-free decomposition of an essential matrix into the
//!   four candidate poses, each kept only if every correspondence passes cheirality;
//! - [`check_cheirality`] and [`check_cheirality_generalized`]: the central and generalized
//!   (ray origin plus direction) cheirality tests (the generalized one from colmap-sharp's
//!   `GenRelpose6pt.cs`, which keeps it private);
//! - [`quat_rotate`] from `misc/quaternion.h`, which the cheirality test rotates with.
//!
//! Callers: [`super::relpose_6pt_shared_focal`], [`super::relpose_6pt_onesided_focal`] and
//! [`super::gen_relpose_6pt`].
//! COLMAP itself decomposes its essential matrices with `geometry/essential_matrix`.
//!
//! Tier B: every step is scalar and follows PoseLib's operation order; the rotation goes
//! through [`super::camera_pose::rotmat_to_quat`] (Eigen's matrix-to-quaternion). Tests:
//! `tests/estimators/rust_only_poselib_essential.rs`.

use super::camera_pose::{rotmat_to_quat, CameraPose};
use crate::linalg::{Matrix3d, Vector3d, Vector4d};
use crate::math::fns;

/// Decomposes `e` into the (up to four) relative poses `cam2_from_cam1` under which every
/// correspondence (unit bearings `x1`, `x2`) lies in front of both cameras, and appends them
/// to `relative_poses`. Port of `poselib::motion_from_essential`.
pub fn motion_from_essential(
    e: &Matrix3d,
    x1: &[Vector3d],
    x2: &[Vector3d],
    relative_poses: &mut Vec<CameraPose>,
) {
    // Compute the necessary cross products
    let (e0, e1, e2) = (e.col(0), e.col(1), e.col(2));
    let u12 = e0.cross(e1);
    let u13 = e0.cross(e2);
    let u23 = e1.cross(e2);
    let n12 = u12.squared_norm();
    let n13 = u13.squared_norm();
    let n23 = u23.squared_norm();

    // Compute the U*W factor
    let (uw1, uw2) = if n12 > n13 {
        if n12 > n23 {
            (e0.normalized(), u12 / fns::sqrt(n12))
        } else {
            (e1.normalized(), u23 / fns::sqrt(n23))
        }
    } else if n13 > n23 {
        (e0.normalized(), u13 / fns::sqrt(n13))
    } else {
        (e1.normalized(), u23 / fns::sqrt(n23))
    };

    let uw0 = -uw2.cross(uw1);

    // Compute the V factor: row0 = UW.col(1)^T E, row1 = -UW.col(0)^T E.
    let et = e.transpose();
    let vt0 = (et * uw1).normalized();
    let mut vt1 = et * -uw0;

    // Here v1 and v2 should be orthogonal. However, if E is not exactly an essential matrix
    // they might not be. To ensure we end up with a rotation matrix we orthogonalize them
    // again here, this should be a nop for good data
    vt1 -= vt0.dot(vt1) * vt0;

    vt1 = vt1.normalized();
    let vt2 = vt0.cross(vt1);
    let vt = Matrix3d::from_rows(vt0, vt1, vt2);

    let mut pose = CameraPose::from_rt(Matrix3d::from_columns(uw0, uw1, uw2) * vt, uw2);
    add_if_cheiral(&pose, x1, x2, relative_poses);
    pose.t = -pose.t;
    add_if_cheiral(&pose, x1, x2, relative_poses);

    // U * W.transpose()
    pose.q = rotmat_to_quat(Matrix3d::from_columns(-uw0, -uw1, uw2) * vt);
    add_if_cheiral(&pose, x1, x2, relative_poses);
    pose.t = -pose.t;
    add_if_cheiral(&pose, x1, x2, relative_poses);
}

/// Whether the triangulated depths of the unit bearings `x1`, `x2` under `pose` are both above
/// `min_depth`. Port of `poselib::check_cheirality` (central form).
pub fn check_cheirality(pose: &CameraPose, x1: Vector3d, x2: Vector3d, min_depth: f64) -> bool {
    // This code assumes that x1 and x2 are unit vectors
    let rx1 = quat_rotate(pose.q, x1);

    // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
    // [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)
    let a = -rx1.dot(x2);
    let b1 = -rx1.dot(pose.t);
    let b2 = x2.dot(pose.t);

    // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
    let lambda1 = b1 - a * b2;
    let lambda2 = -a * b1 + b2;

    let min_depth = min_depth * (1.0 - a * a);
    lambda1 > min_depth && lambda2 > min_depth
}

/// Whether the depths along the rays `p1 + lambda1 x1` (rig 1) and `p2 + lambda2 x2` (rig 2)
/// at their closest points under `pose` are both above `min_depth`. Port of the generalized
/// `poselib::check_cheirality` overload.
pub fn check_cheirality_generalized(
    pose: &CameraPose,
    p1: Vector3d,
    x1: Vector3d,
    p2: Vector3d,
    x2: Vector3d,
    min_depth: f64,
) -> bool {
    // This code assumes that x1 and x2 are unit vectors
    let rx1 = quat_rotate(pose.q, x1);

    // [1 a; a 1] * [lambda1; lambda2] = [b1; b2]
    // [lambda1; lambda2] = [1 -a; -a 1] * [b1; b2] / (1 - a*a)
    let rhs = pose.t + quat_rotate(pose.q, p1) - p2;
    let a = -rx1.dot(x2);
    let b1 = -rx1.dot(rhs);
    let b2 = x2.dot(rhs);

    // Note that we drop the factor 1.0/(1-a*a) since it is always positive.
    let lambda1 = b1 - a * b2;
    let lambda2 = -a * b1 + b2;

    let min_depth = min_depth * (1.0 - a * a);
    lambda1 > min_depth && lambda2 > min_depth
}

/// Port of `poselib::quat_rotate` (`misc/quaternion.h`): rotates `p` by the unit quaternion
/// `q = (w, x, y, z)`.
pub fn quat_rotate(q: Vector4d, p: Vector3d) -> Vector3d {
    let (q1, q2, q3, q4) = (q.x, q.y, q.z, q.w);
    let (p1, p2, p3) = (p.x, p.y, p.z);
    let px1 = -p1 * q2 - p2 * q3 - p3 * q4;
    let px2 = p1 * q1 - p2 * q4 + p3 * q3;
    let px3 = p2 * q1 + p1 * q4 - p3 * q2;
    let px4 = p2 * q2 - p1 * q3 + p3 * q1;
    Vector3d::new(
        px2 * q1 - px1 * q2 - px3 * q4 + px4 * q3,
        px3 * q1 - px1 * q3 + px2 * q4 - px4 * q2,
        px3 * q2 - px2 * q3 - px1 * q4 + px4 * q1,
    )
}

/// The vector wrapper of `check_cheirality` (min depth 0): every correspondence must pass.
fn add_if_cheiral(
    pose: &CameraPose,
    x1: &[Vector3d],
    x2: &[Vector3d],
    relative_poses: &mut Vec<CameraPose>,
) {
    if x1
        .iter()
        .zip(x2)
        .all(|(&a, &b)| check_cheirality(pose, a, b, 0.0))
    {
        relative_poses.push(*pose);
    }
}
