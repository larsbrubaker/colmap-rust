//! Port of COLMAP's `colmap/optim/loransac.h`: [`LoRansac`], Locally Optimized RANSAC
//! ("Locally Optimized RANSAC", Ondrej Chum, Jiri Matas, Josef Kittler, DAGM 2003). It
//! shares [`RansacOptions`], [`RansacReport`] and [`compute_num_trials`] with
//! [`super::ransac`]; the local estimator is any [`Estimator`] with the same `X`, `Y` and
//! `M` as the minimal one (COLMAP assigns between them), and its
//! [`Estimator::estimate_local`] stands for `loransac.h`'s `Refine` detection. Port of
//! colmap-sharp's `Optim/LoRansac.cs`. Tests: `tests/optim/loransac.rs`
//! (`loransac_test.cc`) and `tests/optim/rust_only_ransac.rs`.
//!
//! Tier C (outcome), as RANSAC. The same serial-loop translation as [`super::ransac`]
//! applies (`docs/CPP_DIVERGENCES.md`, entry 141): COLMAP's speculative "is better" check
//! and the locked commit become two plain checks in the same order.

use super::ransac::int_to_size_t;
use super::{
    compute_num_trials, Estimator, InlierSupportMeasurer, MeasuredSupport, RandomSampler,
    RansacOptions, RansacReport, Sampler, SupportMeasurer,
};
use crate::math::random::set_prng_seed;
use crate::util::threading::get_effective_num_threads;
use crate::{check_eq, Result};

// Recursive local optimization to expand the inlier set: at most this many refits.
const MAX_NUM_LOCAL_TRIALS: usize = 10;

/// Port of `colmap::LORANSAC<Estimator, LocalEstimator, SupportMeasurer, Sampler>`.
#[derive(Clone, Debug)]
pub struct LoRansac<E, L, S = InlierSupportMeasurer, Sa = RandomSampler> {
    /// The minimal-sample estimator.
    pub estimator: E,
    /// The local (non-minimal) estimator run on the inliers of a better model.
    pub local_estimator: L,
    /// Scores each model.
    pub support_measurer: S,
    /// The sampler. As in COLMAP, `estimate` only initializes it and reads its
    /// `max_num_samples`; the draws come from a fresh sampler built per run.
    pub sampler: Sa,
    options: RansacOptions,
}

impl<E, L, S, Sa> LoRansac<E, L, S, Sa>
where
    E: Estimator,
    L: Estimator<X = E::X, Y = E::Y, M = E::M>,
    S: SupportMeasurer + Default,
    Sa: Sampler,
{
    /// Port of `LORANSAC(options)`: default support measurer and a sampler for
    /// `E::MIN_NUM_SAMPLES`.
    pub fn new(options: &RansacOptions, estimator: E, local_estimator: L) -> Result<Self> {
        Self::with_parts(
            options,
            estimator,
            local_estimator,
            S::default(),
            Sa::new(E::MIN_NUM_SAMPLES),
        )
    }
}

impl<E, L, S, Sa> LoRansac<E, L, S, Sa>
where
    E: Estimator,
    L: Estimator<X = E::X, Y = E::Y, M = E::M>,
    S: SupportMeasurer,
    Sa: Sampler,
{
    /// Port of `LORANSAC(options, estimator, local_estimator, support_measurer, sampler)`.
    pub fn with_parts(
        options: &RansacOptions,
        estimator: E,
        local_estimator: L,
        support_measurer: S,
        sampler: Sa,
    ) -> Result<Self> {
        Ok(Self {
            estimator,
            local_estimator,
            support_measurer,
            sampler,
            options: options.checked_for_min_num_samples(E::MIN_NUM_SAMPLES)?,
        })
    }

    /// The options after the constructor's adjustment of `max_num_trials`.
    pub fn options(&self) -> &RansacOptions {
        &self.options
    }

    /// Port of `LORANSAC::Estimate`: robustly estimate a model from the independent
    /// variables `x` and the dependent variables `y`.
    pub fn estimate(&mut self, x: &[E::X], y: &[E::Y]) -> Result<RansacReport<E::M, S::Support>> {
        check_eq!(x.len(), y.len());

        let num_samples = x.len();

        let mut report = RansacReport::<E::M, S::Support>::default();

        if num_samples < E::MIN_NUM_SAMPLES {
            return Ok(report);
        }

        let options = self.options;
        let max_residual = options.max_error * options.max_error;
        let min_num_trials = int_to_size_t(options.min_num_trials);

        self.sampler.initialize(num_samples)?;
        let max_num_trials =
            int_to_size_t(options.max_num_trials).min(self.sampler.max_num_samples());

        let mut best_support = S::Support::default();
        let mut best_model: Option<E::M> = None;
        let mut best_model_is_local = false;

        let num_threads = get_effective_num_threads(options.num_threads);
        if !Sa::IS_RANDOM_SAMPLER {
            check_eq!(
                num_threads,
                1,
                "Parallel LORANSAC only supports RandomSampler"
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
        let mut thread_local_estimator = self.local_estimator.clone();
        let mut thread_support_measurer = self.support_measurer.clone();

        // Seed the (thread 0) PRNG.
        if Sa::IS_RANDOMIZED && options.random_seed != -1 {
            set_prng_seed(options.random_seed as u32);
        }

        // Per-thread working buffers.
        let mut residuals: Vec<f64> = Vec::new();
        let mut best_local_residuals: Vec<f64> = Vec::new();

        let mut x_inlier: Vec<E::X> = Vec::new();
        let mut y_inlier: Vec<E::Y> = Vec::new();

        let mut x_rand = Vec::with_capacity(E::MIN_NUM_SAMPLES);
        let mut y_rand = Vec::with_capacity(E::MIN_NUM_SAMPLES);
        let mut sample_models: Vec<E::M> = Vec::new();
        let mut local_models: Vec<E::M> = Vec::new();

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

                // Check if better than global best.
                if thread_support_measurer.is_left_better(&support, &best_support) {
                    // Do local optimization.
                    let mut local_best_support = support.clone();
                    let mut local_best_model = sample_model.clone();
                    let mut local_best_is_local = false;

                    // Estimate locally optimized model from inliers.
                    if support.num_inliers() > E::MIN_NUM_SAMPLES
                        && support.num_inliers() >= L::MIN_NUM_SAMPLES
                    {
                        // Recursive local optimization to expand inlier set.
                        for _ in 0..MAX_NUM_LOCAL_TRIALS {
                            x_inlier.clear();
                            y_inlier.clear();
                            x_inlier.reserve(num_samples);
                            y_inlier.reserve(num_samples);
                            for (i, &residual) in residuals.iter().enumerate() {
                                if residual <= max_residual {
                                    x_inlier.push(x[i].clone());
                                    y_inlier.push(y[i].clone());
                                }
                            }

                            local_models.clear();
                            // `Refine` from the current best model, or `Estimate` on the
                            // inliers (see `Estimator::estimate_local`).
                            thread_local_estimator.estimate_local(
                                &x_inlier,
                                &y_inlier,
                                &local_best_model,
                                &mut local_models,
                            )?;

                            let mut improved_support = false;
                            for local_model in &local_models {
                                thread_local_estimator.residuals(
                                    x,
                                    y,
                                    local_model,
                                    &mut residuals,
                                )?;
                                check_eq!(residuals.len(), num_samples);

                                let local_support =
                                    thread_support_measurer.evaluate(&residuals, max_residual)?;

                                // Check if locally optimized model is better.
                                if thread_support_measurer
                                    .is_left_better(&local_support, &local_best_support)
                                {
                                    local_best_support = local_support;
                                    local_best_model = local_model.clone();
                                    local_best_is_local = true;
                                    improved_support = true;
                                    std::mem::swap(&mut residuals, &mut best_local_residuals);
                                }
                            }

                            // Keep expanding only while the refit improves the support.
                            if !improved_support {
                                break;
                            }

                            // Swap back the residuals, so we can extract the best inlier set
                            // in the next recursion of local optimization.
                            std::mem::swap(&mut residuals, &mut best_local_residuals);
                        }
                    }

                    // Commit the local optimization result to the global best.
                    if thread_support_measurer.is_left_better(&local_best_support, &best_support) {
                        best_support = local_best_support;
                        best_model = Some(local_best_model);
                        best_model_is_local = local_best_is_local;

                        dyn_max_num_trials = compute_num_trials(
                            E::MIN_NUM_SAMPLES,
                            best_support.num_inliers(),
                            num_samples,
                            options.confidence,
                            options.dyn_num_trials_multiplier,
                        );
                    }
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
        if best_model_is_local {
            self.local_estimator
                .residuals(x, y, &report.model, &mut residuals)?;
        } else {
            self.estimator
                .residuals(x, y, &report.model, &mut residuals)?;
        }

        check_eq!(residuals.len(), num_samples);

        report.inlier_mask = residuals.iter().map(|&r| r <= max_residual).collect();

        Ok(report)
    }
}
