//! Port of COLMAP's `colmap/optim/combination_sampler.h/.cc`: a deterministic sampler that
//! walks all N-choose-K combinations in lexicographic order and wraps around. Implements
//! [`super::Sampler`]; siblings are [`super::random_sampler`] and
//! [`super::progressive_sampler`]. Port of colmap-sharp's `Optim/CombinationSampler.cs`.
//! Tests: `tests/optim/combination_sampler.rs` (`combination_sampler_test.cc` 1:1).
//!
//! Tier A (exact): no PRNG; [`crate::math::next_combination`] is `NextCombination`.

use super::Sampler;
use crate::math::{n_choose_k, next_combination};
use crate::{check_le, Result};

/// Port of `colmap::CombinationSampler`: random sampler for RANSAC-based methods that
/// generates unique samples.
///
/// A separate sampler should be instantiated per thread, and it assumes that the input data
/// is shuffled in advance.
#[derive(Clone, Debug)]
pub struct CombinationSampler {
    num_samples: usize,
    total_sample_idxs: Vec<usize>,
}

impl CombinationSampler {
    // Note that the samples must be in increasing order for `NextCombination`.
    fn reset_idxs(&mut self, total_num_samples: usize) {
        self.total_sample_idxs.clear();
        self.total_sample_idxs.extend(0..total_num_samples);
    }
}

impl Sampler for CombinationSampler {
    const IS_RANDOMIZED: bool = false;

    fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            total_sample_idxs: Vec::new(),
        }
    }

    fn initialize(&mut self, total_num_samples: usize) -> Result<()> {
        check_le!(self.num_samples, total_num_samples);
        self.reset_idxs(total_num_samples);
        Ok(())
    }

    fn max_num_samples(&self) -> usize {
        // NChooseK is uint64_t; size_t on 64-bit targets. On 32-bit targets the conversion
        // truncates, as C++'s does.
        n_choose_k(self.total_sample_idxs.len() as u64, self.num_samples as u64) as usize
    }

    fn sample(&mut self, sampled_idxs: &mut Vec<usize>) -> Result<()> {
        sampled_idxs.clear();
        sampled_idxs.extend_from_slice(&self.total_sample_idxs[..self.num_samples]);

        if !next_combination(&mut self.total_sample_idxs, self.num_samples) {
            // Reached all possible combinations, so reset to original state.
            let total = self.total_sample_idxs.len();
            self.reset_idxs(total);
        }
        Ok(())
    }
}
