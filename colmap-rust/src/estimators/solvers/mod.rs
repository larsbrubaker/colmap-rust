//! Port of COLMAP's `src/colmap/estimators/solvers/`:
//!
//! - [`similarity_transform`]: [`SimilarityTransformEstimator`] and
//!   [`EuclideanTransformEstimator`] (Umeyama), [`estimate_rigid3d`], [`estimate_sim3d`] and
//!   their LO-RANSAC variants.

pub mod similarity_transform;

pub use similarity_transform::{
    estimate_rigid3d, estimate_rigid3d_robust, estimate_sim3d, estimate_sim3d_robust,
    EuclideanTransformEstimator, SimilarityTransformEstimator, SimilarityTransformReport,
};
