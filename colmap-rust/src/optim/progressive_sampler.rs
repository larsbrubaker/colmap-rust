//! Port of COLMAP's `colmap/optim/progressive_sampler.h/.cc`: the PROSAC sampler (Chum and
//! Matas, CVPR 2005), which draws from a growing prefix of quality-sorted data. Implements
//! [`super::Sampler`]; siblings are [`super::random_sampler`] and
//! [`super::combination_sampler`]. Port of colmap-sharp's `Optim/ProgressiveSampler.cs`.
//! Tests: `tests/optim/progressive_sampler.rs` (`progressive_sampler_test.cc` 1:1) and
//! `tests/optim/rust_only_sampler_sequence.rs` (the seeded index sequence).
//!
//! Tier A (exact): the growth schedule is plain double arithmetic in COLMAP's evaluation
//! order, and every draw goes through `RandomUniformInteger<uint32_t>`, so a seeded thread
//! yields COLMAP's index sequence.
//!
//! Translation notes:
//! - `t_` (the number of `Sample` calls) is compared with the double `T_n_p_` after
//!   conversion to double, as the C++ `t_ == T_n_p_` does.
//! - Faithful to COLMAP, not to the paper: in progressive mode the random part is drawn from
//!   `[0, n - 2]` and the mandatory element is index `n`, so index `n - 1` is skipped and
//!   index `n` can equal `total_num_samples` (for example on the first call when
//!   `num_samples == total_num_samples`). COLMAP's RANSAC then reads past the end of the data;
//!   [`super::Sampler::sample_xy`] fails a check instead (`docs/CPP_DIVERGENCES.md`, entry 140).

use super::Sampler;
use crate::math::random::random_uniform_integer;
use crate::util::misc::vector_contains_value;
use crate::{check_le, Result};

// Number of iterations before PROSAC behaves like RANSAC. Default value is chosen according
// to the recommended value in the paper.
const NUM_PROGRESSIVE_ITERATIONS: usize = 200000;

/// Port of `colmap::ProgressiveSampler`: random sampler for PROSAC (Progressive Sample
/// Consensus), as described in "Matching with PROSAC - Progressive Sample Consensus",
/// Ondrej Chum and Matas, CVPR 2005.
///
/// A separate sampler should be instantiated per thread, and the data to be sampled from is
/// assumed to be sorted by the quality function in descending order, i.e. higher quality
/// data is closer to the front of the list.
#[derive(Clone, Debug)]
pub struct ProgressiveSampler {
    num_samples: usize,
    total_num_samples: usize,
    // The number of generated samples, i.e. the number of calls to `sample`.
    t: usize,
    n: usize,
    // Variables defined in equation 3.
    t_n: f64,
    t_n_p: f64,
}

impl Sampler for ProgressiveSampler {
    const IS_RANDOMIZED: bool = true;

    fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            total_num_samples: 0,
            t: 0,
            n: 0,
            t_n: 0.0,
            t_n_p: 0.0,
        }
    }

    fn initialize(&mut self, total_num_samples: usize) -> Result<()> {
        check_le!(self.num_samples, total_num_samples);
        self.total_num_samples = total_num_samples;

        self.t = 0;
        self.n = self.num_samples;

        // Compute T_n using recurrent relation in equation 3 (first part).
        self.t_n = NUM_PROGRESSIVE_ITERATIONS as f64;
        self.t_n_p = 1.0;
        for i in 0..self.num_samples {
            self.t_n *= (self.num_samples - i) as f64 / (self.total_num_samples - i) as f64;
        }
        Ok(())
    }

    fn max_num_samples(&self) -> usize {
        usize::MAX
    }

    fn sample(&mut self, sampled_idxs: &mut Vec<usize>) -> Result<()> {
        self.t += 1;

        sampled_idxs.clear();
        sampled_idxs.reserve(self.num_samples);

        // Compute T_n_p using recurrent relation in equation 3 (second part).
        #[allow(clippy::float_cmp)]
        if self.t as f64 == self.t_n_p && self.n < self.total_num_samples {
            let t_n_plus_1 =
                self.t_n * (self.n as f64 + 1.0) / (self.n as f64 + 1.0 - self.num_samples as f64);
            self.t_n_p += (t_n_plus_1 - self.t_n).ceil();
            self.t_n = t_n_plus_1;
            self.n += 1;
        }

        // Decide how many samples to draw from which part of the data as specified in
        // equation 5. size_t arithmetic wraps as in C++ (only reachable with no random draw).
        let mut num_random_samples = self.num_samples;
        let mut max_random_sample_idx = self.n.wrapping_sub(1);
        if self.t_n_p >= self.t as f64 {
            num_random_samples = num_random_samples.wrapping_sub(1);
            max_random_sample_idx = max_random_sample_idx.wrapping_sub(1);
        }

        // Draw semi-random samples as described in algorithm 1.
        for _ in 0..num_random_samples {
            loop {
                // `RandomUniformInteger<uint32_t>(0, max_random_sample_idx)`: the size_t
                // bound converts to uint32_t.
                let random_idx = random_uniform_integer::<u32>(0, max_random_sample_idx as u32);
                let random_idx = random_idx as usize;
                if !vector_contains_value(sampled_idxs, &random_idx) {
                    sampled_idxs.push(random_idx);
                    break;
                }
            }
        }

        // In progressive sampling mode, the last element is mandatory.
        if self.t_n_p >= self.t as f64 {
            sampled_idxs.push(self.n);
        }
        Ok(())
    }
}
