//! Port of COLMAP's `src/colmap/geometry/`:
//!
//! - [`rigid3`]: the rigid transform [`Rigid3d`], [`cross_product_matrix`] and the
//!   covariance helpers.
//! - [`sim3`]: the similarity transform [`Sim3d`].
//! - [`pose`]: angle-axis/Euler conversions, the so(3) exponential and Jacobians, pose
//!   interpolation, the cheirality test, [`CamRayWithJac`], the closest rotation, projection
//!   matrix decomposition, unit-vector and quaternion averaging, the gravity-aligned frame.
//! - [`pose_prior`]: [`PosePrior`] and the EXIF gravity helpers.
//! - [`gps`]: [`GpsTransform`], ellipsoid/ECEF/ENU/UTM conversions.
//! - [`normalization`]: percentile bounding boxes and Hartley normalization.
//! - [`bbox`]: splitting a box into equal parts.
//! - [`essential_matrix`]: essential/fundamental matrices, their decomposition and the
//!   (tangent) Sampson errors.
//! - [`homography_matrix`]: homography decomposition, pose from a homography, the transfer
//!   error.
//! - [`triangulation`]: two-view, mid-point, multi-view and optimal triangulation, and the
//!   triangulation angles.
//!
//! Tests: `colmap-rust/tests/geometry.rs`.

pub mod bbox;
pub mod essential_matrix;
pub mod gps;
pub mod homography_matrix;
pub mod normalization;
pub mod pose;
pub mod pose_prior;
pub mod rigid3;
pub mod sim3;
pub mod triangulation;

pub use gps::{Ellipsoid, GpsTransform};
pub use pose::CamRayWithJac;
pub use pose_prior::{CoordinateSystem, PosePrior};
pub use rigid3::{
    cross_product_matrix, get_covariance_for_composed_rigid3d, get_covariance_for_relative_rigid3d,
    get_covariance_for_rigid3d_inverse, Rigid3d,
};
pub use sim3::Sim3d;
