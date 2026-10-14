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
//! - [`essential`]: `misc/essential.cc`'s `motion_from_essential` and `check_cheirality`.
//! - [`gen_relpose_6pt`]: `solvers/gen_relpose_6pt.cc`, the minimal generalized relative pose
//!   (its generated equations and template are in private sibling files).
//! - [`relpose_6pt_shared_focal`]: `solvers/relpose_6pt_focal.cc`, the minimal relative pose
//!   with a shared unknown focal length (generated coefficients and template in private
//!   sibling files).
//! - [`relpose_6pt_onesided_focal`]: `solvers/relpose_6pt_onesided_focal.cc`, the minimal
//!   relative pose with one unknown focal length (likewise).

pub mod camera_pose;
pub mod essential;
pub mod gen_relpose_6pt;
mod gen_relpose_6pt_coeff_matrix;
mod gen_relpose_6pt_poly;
mod gen_relpose_6pt_template0;
mod gen_relpose_6pt_template1;
pub mod gp3p;
pub mod p3p;
pub mod p4pf;
pub mod poselib_camera;
pub mod re3q3;
mod re3q3_coeffs;
mod re3q3_var_change;
pub mod relpose_5pt;
mod relpose_5pt_det;
pub mod relpose_6pt_onesided_focal;
mod relpose_6pt_onesided_focal_coeffs1;
mod relpose_6pt_onesided_focal_coeffs2;
mod relpose_6pt_onesided_focal_coeffs3;
mod relpose_6pt_onesided_focal_coeffs4;
mod relpose_6pt_onesided_focal_template;
pub mod relpose_6pt_shared_focal;
mod relpose_6pt_shared_focal_coeffs1;
mod relpose_6pt_shared_focal_coeffs2;
mod relpose_6pt_shared_focal_coeffs3;
mod relpose_6pt_shared_focal_coeffs4;
mod relpose_6pt_shared_focal_template;
pub mod sturm;

pub use camera_pose::CameraPose;
pub use camera_pose::ImagePair;
pub use poselib_camera::PoseLibCamera;
