//! Port of COLMAP's `colmap/optim/sampler.h`: the [`Sampler`] trait every RANSAC sampler
//! implements (`Initialize`, `MaxNumSamples`, `Sample`), COLMAP's `is_randomized_sampler`
//! trait as an associated constant, and the `SampleX`/`SampleXY` helpers that gather the
//! sampled elements. Implementations: [`super::random_sampler`], [`super::progressive_sampler`]
//! and [`super::combination_sampler`]; consumers: [`super::ransac`] and [`super::loransac`].
//! Port of colmap-sharp's `Optim/Sampler.cs`.
//!
//! Translation notes:
//! - COLMAP's abstract base class plus template parameter becomes one trait. What RANSAC
//!   needs from the *type* — `Sampler thread_sampler(kMinNumSamples)` and
//!   `is_randomized_sampler<Sampler>` / `std::is_same_v<Sampler, RandomSampler>` — are
//!   [`Sampler::new`], [`Sampler::IS_RANDOMIZED`] and [`Sampler::IS_RANDOM_SAMPLER`].
//! - `SampleX`/`SampleXY` write `X_rand->size()` elements into pre-sized vectors. Here they
//!   clear the output vectors and push one element per sampled index, so element types need
//!   `Clone` but not `Default`. RANSAC always sizes `X_rand` to `kMinNumSamples`, which is
//!   the `num_samples` of the sampler it builds, so the two agree for every COLMAP caller.
//! - The `thread_local std::vector<size_t>` index scratch stays thread-local, so the RANSAC
//!   loop does not allocate per trial.
//! - Every sampled index is checked against the data length. COLMAP reads past the end of
//!   the data when `ProgressiveSampler` yields index `total_num_samples` (undefined
//!   behavior); here that is a check failure (`docs/CPP_DIVERGENCES.md`, entry 140).

use crate::{check_eq, check_lt, Result};
use std::cell::RefCell;

/// Port of `colmap::Sampler`: a sampling method for RANSAC-based estimators.
pub trait Sampler {
    /// Port of `colmap::is_randomized_sampler<Sampler>`: whether the sampler draws from the
    /// PRNG, in which case RANSAC seeds it when `random_seed != -1`.
    const IS_RANDOMIZED: bool;

    /// Whether this is [`super::RandomSampler`]: RANSAC's
    /// `std::is_same_v<Sampler, RandomSampler>`, the only sampler COLMAP allows with more than
    /// one thread.
    const IS_RANDOM_SAMPLER: bool = false;

    /// Construct a sampler drawing `num_samples` elements per sample (the `explicit
    /// Sampler(size_t num_samples)` constructor every COLMAP sampler has).
    fn new(num_samples: usize) -> Self
    where
        Self: Sized;

    /// Initialize the sampler, before calling the [`Sampler::sample`] method.
    fn initialize(&mut self, total_num_samples: usize) -> Result<()>;

    /// Maximum number of unique samples that can be generated.
    fn max_num_samples(&self) -> usize;

    /// Sample `num_samples` elements from all samples. `sampled_idxs` is overwritten.
    fn sample(&mut self, sampled_idxs: &mut Vec<usize>) -> Result<()>;

    /// Port of `Sampler::SampleX`: sample elements from `x` into `x_rand` (cleared first).
    fn sample_x<X: Clone>(&mut self, x: &[X], x_rand: &mut Vec<X>) -> Result<()> {
        with_scratch(|sampled_idxs| {
            self.sample(sampled_idxs)?;
            x_rand.clear();
            for &idx in sampled_idxs.iter() {
                // Entry 140: COLMAP reads past the end here.
                check_lt!(idx, x.len());
                x_rand.push(x[idx].clone());
            }
            Ok(())
        })
    }

    /// Port of `Sampler::SampleXY`: sample elements from `x` and `y` into `x_rand` and
    /// `y_rand` (both cleared first).
    fn sample_xy<X: Clone, Y: Clone>(
        &mut self,
        x: &[X],
        y: &[Y],
        x_rand: &mut Vec<X>,
        y_rand: &mut Vec<Y>,
    ) -> Result<()> {
        check_eq!(x.len(), y.len());
        with_scratch(|sampled_idxs| {
            self.sample(sampled_idxs)?;
            x_rand.clear();
            y_rand.clear();
            for &idx in sampled_idxs.iter() {
                // Entry 140: COLMAP reads past the end here.
                check_lt!(idx, x.len());
                x_rand.push(x[idx].clone());
                y_rand.push(y[idx].clone());
            }
            Ok(())
        })
    }
}

thread_local! {
    // `thread_local std::vector<size_t> sampled_idxs;` of SampleX/SampleXY.
    static SAMPLED_IDXS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

// Lends the thread's index scratch to `f`. A nested call (a sampler whose `sample` itself
// calls `sample_x`) falls back to a fresh vector instead of panicking on the borrow.
fn with_scratch<R>(f: impl FnOnce(&mut Vec<usize>) -> R) -> R {
    SAMPLED_IDXS.with(|cell| match cell.try_borrow_mut() {
        Ok(mut scratch) => f(&mut scratch),
        Err(_) => f(&mut Vec::new()),
    })
}
