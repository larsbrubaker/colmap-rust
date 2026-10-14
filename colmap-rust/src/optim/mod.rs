//! Port of COLMAP's `src/colmap/optim/` robust-estimation framework:
//!
//! - [`estimator`]: the [`Estimator`] trait, COLMAP's implicit Estimator concept (`X_t`,
//!   `Y_t`, `M_t`, `kMinNumSamples`, `Estimate`, `Residuals`), and LO-RANSAC's
//!   [`LocalEstimator`] (`Refine` or, via [`EstimateAsLocal`], `Estimate` on the inliers).
//! - [`sampler`] (the [`Sampler`] trait, `sampler.h`), [`random_sampler`],
//!   [`progressive_sampler`] (PROSAC) and [`combination_sampler`].
//! - [`support_measurement`]: inlier, unique-inlier and M-estimator (MSAC) support.
//! - [`ransac`] ([`Ransac`], [`RansacOptions`], [`RansacReport`], [`compute_num_trials`]) and
//!   [`loransac`] ([`LoRansac`]).
//! - [`sprt`]: the Sequential Probability Ratio Test.
//! - [`sparse_cholesky`] ([`SparseCholeskyWithFallbackSolver`]) and
//!   [`least_absolute_deviations`] ([`LeastAbsoluteDeviationSolver`]), over the sparse
//!   Cholesky of [`crate::linalg::SimplicialCholesky`].
//! - [`tiny_solver`]: [`TinySolver`], the small dense Levenberg-Marquardt solver.

pub mod combination_sampler;
pub mod estimator;
pub mod least_absolute_deviations;
pub mod loransac;
pub mod progressive_sampler;
pub mod random_sampler;
pub mod ransac;
pub mod sampler;
pub mod sparse_cholesky;
pub mod sprt;
pub mod support_measurement;
pub mod tiny_solver;

pub use combination_sampler::CombinationSampler;
pub use estimator::{EstimateAsLocal, Estimator, LocalEstimator};
pub use least_absolute_deviations::{
    LeastAbsoluteDeviationSolver, LeastAbsoluteDeviationSolverOptions,
    LeastAbsoluteDeviationSolverType,
};
pub use loransac::LoRansac;
pub use progressive_sampler::ProgressiveSampler;
pub use random_sampler::RandomSampler;
pub use ransac::{compute_num_trials, Ransac, RansacOptions, RansacReport};
pub use sampler::Sampler;
pub use sparse_cholesky::SparseCholeskyWithFallbackSolver;
pub use sprt::{Sprt, SprtEvaluation, SprtOptions};
pub use support_measurement::{
    InlierSupport, InlierSupportMeasurer, MEstimatorSupport, MEstimatorSupportMeasurer,
    MeasuredSupport, SupportMeasurer, UniqueInlierSupport, UniqueInlierSupportMeasurer,
};
pub use tiny_solver::{
    TinyEuclideanManifold, TinyManifold, TinySolver, TinySolverFunction, TinySolverOptions,
    TinySolverStatus, TinySolverSummary,
};
