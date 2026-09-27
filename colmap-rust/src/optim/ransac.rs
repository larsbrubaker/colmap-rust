//! Port of COLMAP's `colmap/optim/ransac.h`: [`RansacOptions`], the [`RansacReport`] and
//! [`Ransac`], generic over an [`Estimator`], a [`SupportMeasurer`] and a [`Sampler`]
//! (COLMAP's template parameters, defaulting to [`InlierSupportMeasurer`] and
//! [`RandomSampler`]). [`super::loransac`] builds on the options, the report and
//! [`compute_num_trials`]. Port of colmap-sharp's `Optim/Ransac.cs`. Tests:
//! `tests/optim/ransac.rs` (`ransac_test.cc`) and `tests/optim/rust_only_ransac.rs`.
//!
//! Tier: [`compute_num_trials`] is Tier A (exact trial counts); estimation is Tier C
//! (outcome), though with a seeded [`RandomSampler`] the draw sequence is COLMAP's.
//!
//! Translation notes:
//! - The trial loop always runs serially on the calling thread, which is COLMAP's own
//!   `num_threads == 1` path and its non-OpenMP build for every `num_threads`
//!   (`docs/CPP_DIVERGENCES.md`, entry 141). `num_threads` is still validated exactly as
//!   COLMAP does, and the thread-0 seed (`random_seed + 0`) is used. The atomic trial
//!   counter and abort flag become plain locals with the same update order, so
//!   `report.num_trials` counts the final `fetch_add` that ends the loop, as COLMAP's does.
//! - The `int` options convert to `size_t` where COLMAP converts them (sign-extending, so a
//!   negative `min_num_trials` becomes huge, as in C++), and the constructor's clamped
//!   `max_num_trials` converts back to `int` by truncation, as C++'s assignment does.
//! - `static_cast<size_t>(std::ceil(...))` is `as usize`, which saturates where C++ is
//!   undefined (a trial count beyond `size_t`).

use super::{
    Estimator, InlierSupportMeasurer, MeasuredSupport, RandomSampler, Sampler, SupportMeasurer,
};
use crate::math::fns::ln;
use crate::math::random::set_prng_seed;
use crate::util::threading::get_effective_num_threads;
use crate::{check_eq, check_ge, check_gt, check_le, check_ne, Result};

/// Port of `colmap::RANSACOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RansacOptions {
    /// Maximum error for a sample to be considered as an inlier. Note that the residual of an
    /// estimator corresponds to a squared error.
    pub max_error: f64,
    /// A priori assumed minimum inlier ratio, which determines the maximum number of
    /// iterations. Only applies if smaller than `max_num_trials`.
    pub min_inlier_ratio: f64,
    /// Abort the iteration if minimum probability that one sample is free from outliers is
    /// reached.
    pub confidence: f64,
    /// The num_trials_multiplier to the dynamically computed maximum number of iterations
    /// based on the specified confidence value.
    pub dyn_num_trials_multiplier: f64,
    /// Number of random trials to estimate model from random subset.
    pub min_num_trials: i32,
    /// See `min_num_trials`.
    pub max_num_trials: i32,
    /// PRNG seed for randomized samplers. Set to -1 for nondeterministic behavior, or a fixed
    /// value to make results reproducible.
    pub random_seed: i32,
    /// Number of threads for parallel RANSAC. 1 = serial (default). -1 uses all available
    /// hardware threads. (This port always runs serially; see the file header.)
    pub num_threads: i32,
}

impl Default for RansacOptions {
    fn default() -> Self {
        Self {
            max_error: 0.0,
            min_inlier_ratio: 0.1,
            confidence: 0.99,
            dyn_num_trials_multiplier: 3.0,
            min_num_trials: 0,
            max_num_trials: i32::MAX,
            random_seed: -1,
            num_threads: 1,
        }
    }
}

impl RansacOptions {
    /// Port of `RANSACOptions::Check`.
    pub fn check(&self) -> Result<()> {
        check_gt!(self.max_error, 0.0);
        check_ge!(self.min_inlier_ratio, 0.0);
        check_le!(self.min_inlier_ratio, 1.0);
        check_ge!(self.confidence, 0.0);
        check_le!(self.confidence, 1.0);
        check_le!(self.min_num_trials, self.max_num_trials);
        check_ge!(self.random_seed, -1);
        check_ge!(self.num_threads, -1);
        check_ne!(self.num_threads, 0);
        Ok(())
    }

    // The constructor's adjustment: validate, then cap `max_num_trials` by the number of
    // trials the assumed `min_inlier_ratio` needs.
    pub(super) fn checked_for_min_num_samples(&self, min_num_samples: usize) -> Result<Self> {
        self.check()?;
        let mut options = *self;
        // Determine max_num_trials based on assumed `min_inlier_ratio`.
        const NUM_SAMPLES: usize = 100000;
        let dyn_max_num_trials = compute_num_trials(
            min_num_samples,
            (options.min_inlier_ratio * NUM_SAMPLES as f64) as usize,
            NUM_SAMPLES,
            options.confidence,
            options.dyn_num_trials_multiplier,
        );
        // `std::min<size_t>(int, size_t)` assigned back to `int`: truncating conversion.
        options.max_num_trials =
            int_to_size_t(options.max_num_trials).min(dyn_max_num_trials) as i32;
        Ok(options)
    }
}

/// C++'s implicit `int` -> `size_t` conversion (sign-extending, then modulo `2^N`).
pub(super) fn int_to_size_t(value: i32) -> usize {
    value as isize as usize
}

/// Port of `RANSAC<...>::Report` (shared with `LORANSAC<...>::Report`).
#[derive(Clone, Debug, PartialEq)]
pub struct RansacReport<M, S> {
    /// Whether the estimation was successful.
    pub success: bool,
    /// The number of RANSAC trials / iterations.
    pub num_trials: usize,
    /// The support of the estimated model.
    pub support: S,
    /// Boolean mask which is true if a sample is an inlier.
    pub inlier_mask: Vec<bool>,
    /// The estimated model.
    pub model: M,
}

impl<M: Default, S: Default> Default for RansacReport<M, S> {
    fn default() -> Self {
        Self {
            success: false,
            num_trials: 0,
            support: S::default(),
            inlier_mask: Vec::new(),
            model: M::default(),
        }
    }
}

/// Port of `RANSAC<Estimator, ...>::ComputeNumTrials` with `Estimator::kMinNumSamples` passed
/// as `min_num_samples`: the maximum number of trials required to sample at least one
/// outlier-free random set of samples with the specified confidence, given the inlier
/// ratio. [`Ransac::compute_num_trials`] fills in the estimator's constant.
pub fn compute_num_trials(
    min_num_samples: usize,
    num_inliers: usize,
    num_samples: usize,
    confidence: f64,
    num_trials_multiplier: f64,
) -> usize {
    let prob_failure = 1.0 - confidence;
    if prob_failure <= 0.0 {
        return usize::MAX;
    }

    // Not using pow(inlier_ratio, Estimator::kMinNumSamples).
    // See "Fixing the RANSAC stopping criterion"
    // by Schönberger, Larsson, Pollefeys, 2025.
    // size_t subtraction wraps as in C++ when num_inliers < min_num_samples; an earlier
    // factor is then 0, so the product is 0 either way.
    let mut prob_inlier = 1.0;
    for i in 0..min_num_samples {
        prob_inlier *= num_inliers.wrapping_sub(i) as f64 / num_samples.wrapping_sub(i) as f64;
    }

    let prob_outlier = 1.0 - prob_inlier;
    if prob_outlier <= 0.0 {
        return 1;
    }

    // Prevent division by zero below.
    if prob_outlier == 1.0 {
        return usize::MAX;
    }

    (ln(prob_failure) / ln(prob_outlier) * num_trials_multiplier).ceil() as usize
}

/// Port of `colmap::RANSAC<Estimator, SupportMeasurer, Sampler>`: RANdom SAmple Consensus.
#[derive(Clone, Debug)]
pub struct Ransac<E, S = InlierSupportMeasurer, Sa = RandomSampler> {
    /// The minimal-sample estimator.
    pub estimator: E,
    /// Scores each model.
    pub support_measurer: S,
    /// The sampler. As in COLMAP, `estimate` only initializes it and reads its
    /// `max_num_samples`; the draws come from a fresh sampler built per run.
    pub sampler: Sa,
    options: RansacOptions,
}

impl<E: Estimator, S: SupportMeasurer + Default, Sa: Sampler> Ransac<E, S, Sa> {
    /// Port of `RANSAC(options)`: default support measurer and a sampler for
    /// `E::MIN_NUM_SAMPLES`.
    pub fn new(options: &RansacOptions, estimator: E) -> Result<Self> {
        Self::with_parts(
            options,
            estimator,
            S::default(),
            Sa::new(E::MIN_NUM_SAMPLES),
        )
    }
}

impl<E: Estimator, S: SupportMeasurer, Sa: Sampler> Ransac<E, S, Sa> {
    /// Port of `RANSAC(options, estimator, support_measurer, sampler)`.
    pub fn with_parts(
        options: &RansacOptions,
        estimator: E,
        support_measurer: S,
        sampler: Sa,
    ) -> Result<Self> {
        Ok(Self {
            estimator,
            support_measurer,
            sampler,
            options: options.checked_for_min_num_samples(E::MIN_NUM_SAMPLES)?,
        })
    }

    /// The options after the constructor's adjustment of `max_num_trials`.
    pub fn options(&self) -> &RansacOptions {
        &self.options
    }

    /// Port of `RANSAC::ComputeNumTrials` (see [`compute_num_trials`]).
    pub fn compute_num_trials(
        num_inliers: usize,
        num_samples: usize,
        confidence: f64,
        num_trials_multiplier: f64,
    ) -> usize {
        compute_num_trials(
            E::MIN_NUM_SAMPLES,
            num_inliers,
            num_samples,
            confidence,
            num_trials_multiplier,
        )
    }

    /// Port of `RANSAC::Estimate`: robustly estimate a model from the independent variables
    /// `x` and the dependent variables `y`.
    pub fn estimate(&mut self, x: &[E::X], y: &[E::Y]) -> Result<RansacReport<E::M, S::Support>> {
        check_eq!(x.len(), y.len());

        let num_samples = x.len();

        let mut report = RansacReport::<E::M, S::Support>::default();

        if num_samples < E::MIN_NUM_SAMPLES {
            return Ok(report);
        }

        let mut best_support = S::Support::default();
        let mut best_model: Option<E::M> = None;

        let options = self.options;
        let max_residual = options.max_error * options.max_error;
        let min_num_trials = int_to_size_t(options.min_num_trials);

        self.sampler.initialize(num_samples)?;
        let max_num_trials =
            int_to_size_t(options.max_num_trials).min(self.sampler.max_num_samples());

        let num_threads = get_effective_num_threads(options.num_threads);
        if !Sa::IS_RANDOM_SAMPLER {
            check_eq!(
                num_threads,
                1,
                "Parallel RANSAC only supports RandomSampler"
            );
        }

        let mut trial_counter: usize = 0;
        let mut dyn_max_num_trials = max_num_trials;
        let mut abort_flag = false;

        // COLMAP's parallel region, run once on this thread (entry 141). Per-thread copies
        // of the mutable objects.
        let mut thread_sampler = Sa::new(E::MIN_NUM_SAMPLES);
        thread_sampler.initialize(num_samples)?;
        let mut thread_estimator = self.estimator.clone();
        let mut thread_support_measurer = self.support_measurer.clone();

        // Seed the (thread 0) PRNG.
        if Sa::IS_RANDOMIZED && options.random_seed != -1 {
            set_prng_seed(options.random_seed as u32);
        }

        // Per-thread working buffers.
        let mut residuals = Vec::new();
        let mut x_rand = Vec::with_capacity(E::MIN_NUM_SAMPLES);
        let mut y_rand = Vec::with_capacity(E::MIN_NUM_SAMPLES);
        let mut sample_models = Vec::new();

        loop {
            let curr_thread_trial = trial_counter;
            trial_counter += 1;
            if curr_thread_trial >= max_num_trials || abort_flag {
                break;
            }

            thread_sampler.sample_xy(x, y, &mut x_rand, &mut y_rand)?;

            // Estimate model for current subset.
            sample_models.clear();
            thread_estimator.estimate(&x_rand, &y_rand, &mut sample_models)?;

            // Iterate through all estimated models.
            for sample_model in &sample_models {
                thread_estimator.residuals(x, y, sample_model, &mut residuals)?;
                check_eq!(residuals.len(), num_samples);

                let support = thread_support_measurer.evaluate(&residuals, max_residual)?;

                // Save as best subset if better than all previous subsets.
                if thread_support_measurer.is_left_better(&support, &best_support) {
                    best_support = support;
                    best_model = Some(sample_model.clone());

                    dyn_max_num_trials = compute_num_trials(
                        E::MIN_NUM_SAMPLES,
                        best_support.num_inliers(),
                        num_samples,
                        options.confidence,
                        options.dyn_num_trials_multiplier,
                    );
                }

                if curr_thread_trial >= dyn_max_num_trials && curr_thread_trial >= min_num_trials {
                    abort_flag = true;
                    break;
                }
            }

            if abort_flag {
                break;
            }
        }

        report.num_trials = trial_counter;

        let Some(best_model) = best_model else {
            return Ok(report);
        };

        report.support = best_support;
        report.model = best_model;

        // No valid model was found.
        if report.support.num_inliers() < E::MIN_NUM_SAMPLES {
            return Ok(report);
        }

        report.success = true;

        // Determine inlier mask. Note that this calculates the residuals for the best model
        // twice, but saves to copy and fill the inlier mask for each evaluated model. Some
        // benchmarking revealed that this approach is faster.
        let mut residuals = vec![0.0; num_samples];
        self.estimator
            .residuals(x, y, &report.model, &mut residuals)?;
        check_eq!(residuals.len(), num_samples);

        report.inlier_mask = residuals.iter().map(|&r| r <= max_residual).collect();

        Ok(report)
    }
}
