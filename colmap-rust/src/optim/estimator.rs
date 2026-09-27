//! The Rust face of COLMAP's implicit "Estimator" template concept that
//! `colmap/optim/ransac.h` and `loransac.h` are written against: `X_t`, `Y_t`, `M_t`,
//! `kMinNumSamples`, `Estimate`, `Residuals`, and `loransac.h`'s optional `Refine` hook.
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
//!   `Estimate` on the inliers. Rust cannot branch on "has a method", so the choice is the
//!   provided method [`Estimator::estimate_local`]: it forwards to `estimate` (ignoring the
//!   model), and an estimator whose C++ class has `Refine` overrides it to refine a copy of
//!   the model and push it only on success. The choice stays with the estimator's author,
//!   where COLMAP's overload detection puts it.

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

    /// LO-RANSAC's local optimization step on the inliers `x`, `y`: push the locally
    /// optimized models to `models` (cleared by the caller). `initial_model` is the current
    /// best model. The default calls [`Estimator::estimate`] and ignores it; an estimator
    /// whose C++ class has `Refine(X, Y, M_t*)` overrides this to refine a copy of
    /// `initial_model` and push the copy only when `Refine` succeeds.
    fn estimate_local(
        &mut self,
        x: &[Self::X],
        y: &[Self::Y],
        initial_model: &Self::M,
        models: &mut Vec<Self::M>,
    ) -> Result<()> {
        let _ = initial_model;
        self.estimate(x, y, models)
    }
}
