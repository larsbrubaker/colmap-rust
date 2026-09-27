//! Port of COLMAP's `colmap/optim/sprt.h/.cc`: Wald's Sequential Probability Ratio Test for
//! early rejection of bad RANSAC hypotheses (Matas and Chum, "Randomized RANSAC with
//! Sequential Probability Ratio Test", ICCV 2005). Standalone; nothing in COLMAP 4.2.0's
//! RANSAC calls it yet. Port of colmap-sharp's `Optim/Sprt.cs`. Tests: `tests/optim/sprt.rs`
//! (`sprt_test.cc` 1:1).
//!
//! Tier A (exact) except the decision threshold, which goes through `log`
//! ([`crate::math::fns::ln`], `docs/CPP_DIVERGENCES.md` entry 1: 1-2 ulp from Apple libm).
//!
//! Translation note: `Evaluate`'s two out-parameters are returned in [`SprtEvaluation`]
//! with the accept/reject flag; COLMAP writes both on every path, so nothing is lost.

use crate::math::fns::ln;

/// Port of `SPRT::Options`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SprtOptions {
    /// Probability of rejecting a good model.
    pub delta: f64,
    /// A priori assumed minimum inlier ratio.
    pub epsilon: f64,
    /// The ratio of the time it takes to estimate a model from a random sample over the time
    /// it takes to decide whether one data sample is an inlier or not. Matas et al. propose
    /// 200 for the 7-point algorithm.
    pub eval_time_ratio: f64,
    /// Number of models per random sample, that have to be verified. E.g. 1-3 for the
    /// 7-point fundamental matrix algorithm, or 1-10 for the 5-point essential matrix
    /// algorithm.
    pub num_models_per_sample: i32,
}

impl Default for SprtOptions {
    fn default() -> Self {
        Self {
            delta: 0.01,
            epsilon: 0.1,
            eval_time_ratio: 200.0,
            num_models_per_sample: 1,
        }
    }
}

/// The outcome of [`Sprt::evaluate`]: the return value and both out-parameters of
/// `SPRT::Evaluate`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SprtEvaluation {
    /// Whether the model is accepted (the return value).
    pub accepted: bool,
    /// The number of inliers among the evaluated samples.
    pub num_inliers: usize,
    /// The number of samples evaluated before the decision.
    pub num_eval_samples: usize,
}

/// Port of `colmap::SPRT`.
#[derive(Clone, Copy, Debug)]
pub struct Sprt {
    options: SprtOptions,
    delta_epsilon: f64,
    delta_1_epsilon_1: f64,
    decision_threshold: f64,
}

impl Sprt {
    /// Port of `SPRT(const Options&)`.
    pub fn new(options: SprtOptions) -> Self {
        let mut sprt = Self {
            options,
            delta_epsilon: 0.0,
            delta_1_epsilon_1: 0.0,
            decision_threshold: 0.0,
        };
        sprt.update(options);
        sprt
    }

    /// Port of `SPRT::Update`.
    pub fn update(&mut self, options: SprtOptions) {
        self.options = options;
        self.delta_epsilon = options.delta / options.epsilon;
        self.delta_1_epsilon_1 = (1.0 - options.delta) / (1.0 - options.epsilon);
        self.update_decision_threshold();
    }

    /// Port of `SPRT::Evaluate`.
    pub fn evaluate(&self, residuals: &[f64], max_residual: f64) -> SprtEvaluation {
        let mut num_inliers = 0;

        let mut likelihood_ratio = 1.0;

        for (i, residual) in residuals.iter().enumerate() {
            if residual.abs() <= max_residual {
                num_inliers += 1;
                likelihood_ratio *= self.delta_epsilon;
            } else {
                likelihood_ratio *= self.delta_1_epsilon_1;
            }

            if likelihood_ratio > self.decision_threshold {
                return SprtEvaluation {
                    accepted: false,
                    num_inliers,
                    num_eval_samples: i + 1,
                };
            }
        }

        SprtEvaluation {
            accepted: true,
            num_inliers,
            num_eval_samples: residuals.len(),
        }
    }

    fn update_decision_threshold(&mut self) {
        let o = &self.options;
        // Equation 2
        let c = (1.0 - o.delta) * ln((1.0 - o.delta) / (1.0 - o.epsilon))
            + o.delta * ln(o.delta / o.epsilon);

        // Equation 6
        let a0 = o.eval_time_ratio * c / o.num_models_per_sample as f64 + 1.0;

        let mut a = a0;

        const EPS: f64 = 1.5e-8;

        // Compute A using the recursive relation
        //    A* = lim(n->inf) A
        // The series typically converges within 4 iterations
        for _ in 0..100 {
            let a1 = a0 + ln(a);

            if (a1 - a).abs() < EPS {
                break;
            }

            a = a1;
        }

        self.decision_threshold = a;
    }
}
