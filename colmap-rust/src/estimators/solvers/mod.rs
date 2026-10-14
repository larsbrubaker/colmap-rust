//! Port of COLMAP's `src/colmap/estimators/solvers/`:
//!
//! - [`affine_transform`]: [`AffineTransformEstimator`], [`estimate_affine2d`] and
//!   [`estimate_affine2d_robust`].
//! - [`translation_transform`]: [`TranslationTransformEstimator`] (2D).
//! - [`fundamental_matrix`]: [`FundamentalMatrixSevenPointEstimator`] and
//!   [`FundamentalMatrixEightPointEstimator`].
//! - [`generalized_relative_pose`]: [`GR6PEstimator`] and its [`GRNPObservation`].
//! - [`homography_matrix`]: [`HomographyMatrixEstimator`] and
//!   [`HomographyMatrixRayEstimator`] (DLT between points or bearing rays).
//! - [`absolute_pose`]: [`P3PEstimator`], [`P4PFEstimator`], [`Point2DWithRay`],
//!   [`ImgFromCamFunc`] and [`compute_squared_reprojection_error`].
//! - [`epnp`]: [`EPNPEstimator`], the rest of `absolute_pose.h`.
//! - [`essential_matrix`]: the essential matrix estimators
//!   [`EssentialMatrixFivePointEstimator`], [`EssentialMatrixEightPointEstimator`] and
//!   [`EssentialMatrixTangentSampsonEstimator`].
//! - [`generalized_absolute_pose`]: [`GP3PEstimator`] for multi-camera rigs.
//! - [`similarity_transform`]: [`SimilarityTransformEstimator`] and
//!   [`EuclideanTransformEstimator`] (Umeyama), [`estimate_rigid3d`], [`estimate_sim3d`] and
//!   their LO-RANSAC variants.
//! - [`poselib`]: the PoseLib solvers and helpers they are built on.
//! - [`poselib_utils`]: conversions between COLMAP's camera and pose types and PoseLib's.

pub mod absolute_pose;
pub mod affine_transform;
pub mod epnp;
pub mod essential_matrix;
mod essential_matrix_coeffs;
mod essential_matrix_poly1;
mod essential_matrix_poly2;
mod essential_matrix_poly3;
pub mod fundamental_matrix;
pub mod generalized_absolute_pose;
pub mod generalized_relative_pose;
pub mod homography_matrix;
pub mod poselib;
pub mod poselib_utils;
pub mod similarity_transform;
pub mod translation_transform;

pub use absolute_pose::{
    compute_squared_reprojection_error, ImgFromCamFunc, P3PEstimator, P4PFEstimator, P4PFModel,
    Point2DWithRay,
};
pub use affine_transform::{
    estimate_affine2d, estimate_affine2d_robust, AffineTransformEstimator, AffineTransformReport,
};
pub use epnp::EPNPEstimator;
pub use essential_matrix::{
    EssentialMatrixEightPointEstimator, EssentialMatrixFivePointEstimator,
    EssentialMatrixTangentSampsonEstimator,
};
pub use fundamental_matrix::{
    FundamentalMatrixEightPointEstimator, FundamentalMatrixSevenPointEstimator,
};
pub use generalized_absolute_pose::{GP3PEstimator, GP3PObservation, GP3PResidualType};
pub use generalized_relative_pose::{GR6PEstimator, GRNPObservation};
pub use homography_matrix::{
    CamRayWithImgPoint, HomographyMatrixEstimator, HomographyMatrixRayEstimator,
};
pub use similarity_transform::{
    estimate_rigid3d, estimate_rigid3d_robust, estimate_sim3d, estimate_sim3d_robust,
    EuclideanTransformEstimator, SimilarityTransformEstimator, SimilarityTransformReport,
};
pub use translation_transform::TranslationTransformEstimator;
