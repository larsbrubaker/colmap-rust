//! Port of PoseLib's `PoseLib/camera_pose.h` (`struct CameraPose`) and the two quaternion
//! helpers of `PoseLib/misc/quaternion.h` it is built on (`rotmat_to_quat`, `quat_to_rotmat`)
//! plus `quat_multiply` (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
//! THIRD_PARTY_NOTICES.md; commit fa7280fee27f97aff31ae7f98bab7f583fac7d08, COLMAP 4.2.0's
//! pin). Port of colmap-sharp's `Estimators/Solvers/PoseLib/CameraPose.cs`.
//!
//! [`CameraPose`] is the pose type every PoseLib solver returns ([`super::p3p`],
//! [`super::gp3p`]). Only the members COLMAP's callers use are ported (construction from R
//! and t, `R()`, `Rt()`); PoseLib's rotate/compose/center helpers are used only inside
//! PoseLib's own refinement code, which COLMAP does not call. Tests:
//! `tests/estimators/rust_only_camera_pose.rs`.

use crate::linalg::{Matrix3d, Matrix3x4d, Quaterniond, Vector3d, Vector4d};

/// A camera pose `x_cam = R x_world + t` with the rotation as a unit quaternion. Port of
/// `poselib::CameraPose`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraPose {
    /// The rotation quaternion in PoseLib's order, real part first: `q.x = qw`, `q.y = qx`,
    /// `q.z = qy`, `q.w = qz`. (A [`Vector4d`] rather than a [`Quaterniond`] because PoseLib
    /// stores and normalizes it as a plain 4-vector in that order.)
    pub q: Vector4d,
    /// The translation.
    pub t: Vector3d,
}

impl Default for CameraPose {
    /// The identity pose, PoseLib's default constructor.
    fn default() -> Self {
        Self::identity()
    }
}

impl CameraPose {
    /// The identity pose.
    pub const fn identity() -> Self {
        Self {
            q: Vector4d::new(1.0, 0.0, 0.0, 0.0),
            t: Vector3d::zeros(),
        }
    }

    /// A pose from a `(qw, qx, qy, qz)` quaternion and a translation.
    pub const fn new(q: Vector4d, t: Vector3d) -> Self {
        Self { q, t }
    }

    /// A pose from a rotation matrix and a translation (`rotmat_to_quat`).
    pub fn from_rt(r: Matrix3d, t: Vector3d) -> Self {
        Self {
            q: rotmat_to_quat(r),
            t,
        }
    }

    /// The rotation matrix, PoseLib's `R()`.
    pub fn r(&self) -> Matrix3d {
        quat_to_rotmat(self.q)
    }

    /// The 3x4 matrix `[R | t]`, PoseLib's `Rt()`.
    pub fn rt(&self) -> Matrix3x4d {
        Matrix3x4d::from_blocks(quat_to_rotmat(self.q), self.t)
    }
}

/// Port of `poselib::quat_to_rotmat` on a `(w, x, y, z)` vector.
pub fn quat_to_rotmat(q: Vector4d) -> Matrix3d {
    Quaterniond::new(q.x, q.y, q.z, q.w).to_rotation_matrix()
}

/// Port of `poselib::rotmat_to_quat`: Eigen's matrix-to-quaternion, reordered to
/// `(w, x, y, z)` and normalized.
pub fn rotmat_to_quat(r: Matrix3d) -> Vector4d {
    let q_flip = Quaterniond::from_rotation_matrix(r);
    Vector4d::new(q_flip.w, q_flip.x, q_flip.y, q_flip.z).normalized()
}

/// Port of `poselib::quat_multiply` on `(w, x, y, z)` vectors.
pub fn quat_multiply(qa: Vector4d, qb: Vector4d) -> Vector4d {
    let (qa1, qa2, qa3, qa4) = (qa.x, qa.y, qa.z, qa.w);
    let (qb1, qb2, qb3, qb4) = (qb.x, qb.y, qb.z, qb.w);

    Vector4d::new(
        qa1 * qb1 - qa2 * qb2 - qa3 * qb3 - qa4 * qb4,
        qa1 * qb2 + qa2 * qb1 + qa3 * qb4 - qa4 * qb3,
        qa1 * qb3 + qa3 * qb1 - qa2 * qb4 + qa4 * qb2,
        qa1 * qb4 + qa2 * qb3 - qa3 * qb2 + qa4 * qb1,
    )
}
