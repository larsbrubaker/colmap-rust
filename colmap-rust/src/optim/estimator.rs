//! The Rust face of COLMAP's implicit "Estimator" template concept that
//! `colmap/optim/ransac.h` and `loransac.h` are written against: `X_t`, `Y_t`, `M_t`,
//! `kMinNumSamples`, `Estimate`, `Residuals` ([`Estimator`]), and `loransac.h`'s local
//! estimator with its optional `Refine` hook ([`LocalEstimator`], [`EstimateAsLocal`]).
//! COLMAP never spells the concept out; every class in `estimators/` and
//! `estimators/solvers/` satisfies it by shape. Consumers: [`super::ransac`] and
//! [`super::loransac`]. Counterpart of colmap-sharp's `Optim/Estimator.cs`.
//!
//! Translation notes (these shape every Phase 6 estimator):
//! - `X_t`, `Y_t`, `M_t` are associated types and `kMinNumSamples` an associated constant,
//!   so generic code reads `E::MIN_NUM_SAMPLES` with no instance, as C++ does.
//! - `Estimate` and `Residuals` take `&mut self`: some COLMAP estimators carry state
//!   (options, a camera, DEGENSAC's inner estimator) and their methods are non-const.
//!   Stateless C++ estimators declare them `static`; here they are methods of a unit struct.
//!   RANSAC clones the estimator for its worker (`Estimator thread_estimator = estimator;`),
//!   hence `Clone`.
//! - Both return [`crate::Result`], so an estimator's own `THROW_CHECK`s propagate.
//! - `Residuals` resizes the caller's vector, as every COLMAP estimator does; RANSAC reuses
//!   the vector across calls, so an estimator that leaves slots untouched sees the previous
//!   values there, as in C++.
//! - `M_t: Default` because RANSAC's `Report` default-constructs its model.
//! - `loransac.h` detects at compile time whether the local estimator has
//!   `Refine(X, Y, M_t*)` and then refines the current best model instead of calling
//!   `Estimate` on the inliers. Rust cannot branch on "has a method", so LO-RANSAC's local
//!   estimator is its own trait, [`LocalEstimator`], with one required
//!   [`LocalEstimator::estimate_local`] that receives the current best model. An estimator
//!   whose C++ class has `Refine` implements it by refining a copy and pushing it on success
//!   (COLMAP's `FundamentalMatrixSampsonEstimator` has only `Refine` and `Residuals`, so it
//!   implements [`LocalEstimator`] alone). An estimator without `Refine` opts in explicitly
//!   through [`EstimateAsLocal`], which calls its `Estimate` on the inliers. With no default
//!   method, forgetting the `Refine` path is a compile error, not a silent divergence. Same
//!   split as colmap-sharp's `IEstimator` / `ILocalEstimator`.

use crate::Result;

/// A model estimator for RANSAC: estimates candidate models from a minimal sample and
/// measures every data pair against a model. Port of COLMAP's Estimator concept.
pub trait Estimator: Clone {
    /// Independent variable (`X_t`).
    type X: Clone;
    /// Dependent variable (`Y_t`).
    type Y: Clone;
    /// Model (`M_t`).
    type M: Clone + Default;

    /// The minimum number of samples needed to estimate a model (`kMinNumSamples`).
    const MIN_NUM_SAMPLES: usize;

    /// Estimate zero or more models from the sampled pairs and push them to `models`, which
    /// the caller has cleared.
    fn estimate(&mut self, x: &[Self::X], y: &[Self::Y], models: &mut Vec<Self::M>) -> Result<()>;

    /// Compute the residual of every pair `(x[i], y[i])` under `model` into `residuals`
    /// (resized to `x.len()`). RANSAC compares the residuals with the squared max error, so
    /// they are squared errors.
    fn residuals(
        &mut self,
        x: &[Self::X],
        y: &[Self::Y],
        model: &Self::M,
        residuals: &mut Vec<f64>,
    ) -> Result<()>;
}

/// The local optimizer of LO-RANSAC (`LORANSAC`'s `LocalEstimator` parameter): re-estimates
/// the model from the current inlier set, either by refining the current best model
/// (COLMAP's `Refine` hook) or from scratch (COLMAP's `Estimate`, via [`EstimateAsLocal`]).
/// `LoRansac` requires its `X`, `Y` and `M` to be those of the minimal estimator, as
/// `LORANSAC` assigns between them.
pub trait LocalEstimator: Clone {
    /// Independent variable (`X_t`).
    type X: Clone;
    /// Dependent variable (`Y_t`).
    type Y: Clone;
    /// Model (`M_t`).
    type M: Clone + Default;

    /// The minimum number of inliers needed for a local estimate (`kMinNumSamples`);
    /// LO-RANSAC skips local optimization below it.
    const MIN_NUM_SAMPLES: usize;

    /// Estimate locally optimized models from the inliers `x`, `y` and push them to `models`
    /// (cleared by the caller). `initial_model` is the current best model: an estimator whose
    /// C++ class has `Refine(X, Y, M_t*)` refines a copy of it and pushes the copy only when
    /// `Refine` succeeds; [`EstimateAsLocal`] ignores it and runs `Estimate`.
    fn estimate_local(
        &mut self,
        x: &[Self::X],
        y: &[Self::Y],
        initial_model: &Self::M,
        models: &mut Vec<Self::M>,
    ) -> Result<()>;

    /// Compute the residual of every pair under `model`, as [`Estimator::residuals`].
    fn residuals(
        &mut self,
        x: &[Self::X],
        y: &[Self::Y],
        model: &Self::M,
        residuals: &mut Vec<f64>,
    ) -> Result<()>;
}

/// Uses an [`Estimator`] whose C++ class has no `Refine` as LO-RANSAC's local estimator:
/// `loransac.h`'s `Estimate(X_inlier, Y_inlier, &local_models)` branch.
#[derive(Clone, Debug, Default)]
pub struct EstimateAsLocal<E>(pub E);

impl<E: Estimator> LocalEstimator for EstimateAsLocal<E> {
    type X = E::X;
    type Y = E::Y;
    type M = E::M;
    const MIN_NUM_SAMPLES: usize = E::MIN_NUM_SAMPLES;

    fn estimate_local(
        &mut self,
        x: &[E::X],
        y: &[E::Y],
        _initial_model: &E::M,
        models: &mut Vec<E::M>,
    ) -> Result<()> {
        self.0.estimate(x, y, models)
    }

    fn residuals(
        &mut self,
        x: &[E::X],
        y: &[E::Y],
        model: &E::M,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        self.0.residuals(x, y, model, residuals)
    }
}
