//! Port of COLMAP's `src/colmap/optim/` robust-estimation framework:
//!
//! - [`estimator`]: the [`Estimator`] trait, COLMAP's implicit Estimator concept (`X_t`,
//!   `Y_t`, `M_t`, `kMinNumSamples`, `Estimate`, `Residuals`, LO-RANSAC's `Refine` hook).
//! - [`sampler`] (the [`Sampler`] trait, `sampler.h`), [`random_sampler`],
//!   [`progressive_sampler`] (PROSAC) and [`combination_sampler`].
//! - [`support_measurement`]: inlier, unique-inlier and M-estimator (MSAC) support.
//! - [`ransac`] ([`Ransac`], [`RansacOptions`], [`RansacReport`], [`compute_num_trials`]) and
//!   [`loransac`] ([`LoRansac`]).
//! - [`sprt`]: the Sequential Probability Ratio Test.
//!
//! Not yet ported (later Phase 5 slices): `least_absolute_deviations` (needs the sparse
//! Cholesky), `sparse_cholesky` and `tiny_solver`.

pub mod combination_sampler;
pub mod estimator;
pub mod loransac;
pub mod progressive_sampler;
pub mod random_sampler;
pub mod ransac;
pub mod sampler;
pub mod sprt;
pub mod support_measurement;

pub use combination_sampler::CombinationSampler;
pub use estimator::Estimator;
pub use loransac::LoRansac;
pub use progressive_sampler::ProgressiveSampler;
pub use random_sampler::RandomSampler;
pub use ransac::{compute_num_trials, Ransac, RansacOptions, RansacReport};
pub use sampler::Sampler;
pub use sprt::{Sprt, SprtEvaluation, SprtOptions};
pub use support_measurement::{
    InlierSupport, InlierSupportMeasurer, MEstimatorSupport, MEstimatorSupportMeasurer,
    MeasuredSupport, SupportMeasurer, UniqueInlierSupport, UniqueInlierSupportMeasurer,
};
