//! Port of the parts of PoseLib (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
//! THIRD_PARTY_NOTICES.md) that COLMAP's `estimators/solvers/` call, at commit
//! fa7280fee27f97aff31ae7f98bab7f583fac7d08 (COLMAP 4.2.0's pin), ported from colmap-sharp's
//! `Estimators/Solvers/PoseLib/`, one file per PoseLib source:
//!
//! - [`sturm`]: `misc/sturm.h`, Sturm-sequence root bracketing and Danilevsky's
//!   characteristic polynomial.
//! - [`camera_pose`]: `camera_pose.h`'s [`CameraPose`] and the quaternion helpers of
//!   `misc/quaternion.h`.
//! - [`re3q3`]: `misc/re3q3.cc`, three quadratics in three unknowns (its generated
//!   elimination step and random change of variables are in private sibling files).
//! - [`p3p`]: `solvers/p3p.cc`, the minimal absolute pose.
//! - [`gp3p`]: `solvers/gp3p.cc`, the minimal generalized absolute pose.
//! - [`p4pf`]: `solvers/p4pf.cc`, the minimal absolute pose with unknown focal length.
//! - [`relpose_5pt`]: `solvers/relpose_5pt.cc`, the minimal five-point essential matrix
//!   (its generated determinant polynomial is in a private sibling file).
//! - [`poselib_camera`]: the camera description of `misc/camera_models.h`.

pub mod camera_pose;
pub mod gp3p;
pub mod p3p;
pub mod p4pf;
pub mod poselib_camera;
pub mod re3q3;
mod re3q3_coeffs;
mod re3q3_var_change;
pub mod relpose_5pt;
mod relpose_5pt_det;
pub mod sturm;

pub use camera_pose::CameraPose;
pub use poselib_camera::PoseLibCamera;
