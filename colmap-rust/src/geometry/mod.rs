//! Port of COLMAP's `src/colmap/geometry/` (the parts that need no matrix decomposition):
//!
//! - [`rigid3`]: the rigid transform [`Rigid3d`] and [`cross_product_matrix`].
//! - [`sim3`]: the similarity transform [`Sim3d`].
//! - [`pose`]: angle-axis/Euler conversions, the so(3) exponential and Jacobians, pose
//!   interpolation, the cheirality test and [`CamRayWithJac`].
//! - [`pose_prior`]: [`PosePrior`] and the EXIF gravity helpers.
//! - [`gps`]: [`GpsTransform`], ellipsoid/ECEF/ENU/UTM conversions.
//! - [`normalization`]: percentile bounding boxes and Hartley normalization.
//! - [`bbox`]: splitting a box into equal parts.
//!
//! `essential_matrix`, `homography_matrix`, `triangulation`, the averaging and
//! decomposition functions of `pose` and the 12x12 covariance helpers of `rigid3` need the
//! dynamic-size matrices and decompositions and arrive with them. Tests:
//! `colmap-rust/tests/geometry.rs`.

pub mod bbox;
pub mod gps;
pub mod normalization;
pub mod pose;
pub mod pose_prior;
pub mod rigid3;
pub mod sim3;

pub use gps::{Ellipsoid, GpsTransform};
pub use pose::CamRayWithJac;
pub use pose_prior::{CoordinateSystem, PosePrior};
pub use rigid3::{cross_product_matrix, get_covariance_for_rigid3d_inverse, Rigid3d};
pub use sim3::Sim3d;
