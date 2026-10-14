//! Port of COLMAP's `src/colmap/estimators/solvers/`:
//!
//! - [`absolute_pose`]: [`P3PEstimator`], [`P4PFEstimator`], [`Point2DWithRay`],
//!   [`ImgFromCamFunc`] and [`compute_squared_reprojection_error`].
//! - [`epnp`]: [`EPNPEstimator`], the rest of `absolute_pose.h`.
//! - [`generalized_absolute_pose`]: [`GP3PEstimator`] for multi-camera rigs.
//! - [`similarity_transform`]: [`SimilarityTransformEstimator`] and
//!   [`EuclideanTransformEstimator`] (Umeyama), [`estimate_rigid3d`], [`estimate_sim3d`] and
//!   their LO-RANSAC variants.
//! - [`poselib`]: the PoseLib solvers and helpers they are built on.
//! - [`poselib_utils`]: conversions between COLMAP's camera and pose types and PoseLib's.

pub mod absolute_pose;
pub mod epnp;
pub mod generalized_absolute_pose;
pub mod poselib;
pub mod poselib_utils;
pub mod similarity_transform;

pub use absolute_pose::{
    compute_squared_reprojection_error, ImgFromCamFunc, P3PEstimator, P4PFEstimator, P4PFModel,
    Point2DWithRay,
};
pub use epnp::EPNPEstimator;
pub use generalized_absolute_pose::{GP3PEstimator, GP3PObservation, GP3PResidualType};
pub use similarity_transform::{
    estimate_rigid3d, estimate_rigid3d_robust, estimate_sim3d, estimate_sim3d_robust,
    EuclideanTransformEstimator, SimilarityTransformEstimator, SimilarityTransformReport,
};
