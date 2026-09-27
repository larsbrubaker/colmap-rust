//! Port of COLMAP's `colmap/optim/random_sampler.h/.cc`: RANSAC's default sampler, which
//! draws `num_samples` distinct indices by a partial Fisher-Yates shuffle of a running
//! permutation. Implements [`super::Sampler`]; siblings are [`super::progressive_sampler`]
//! and [`super::combination_sampler`]. Port of colmap-sharp's `Optim/RandomSampler.cs`.
//! Tests: `tests/optim/random_sampler.rs` (`random_sampler_test.cc` 1:1) and
//! `tests/optim/rust_only_sampler_sequence.rs` (the seeded index sequence).
//!
//! Tier A (exact): every draw goes through [`crate::math::random::shuffle`], and the
//! permutation is *not* reset between samples (as in COLMAP), so a seeded thread yields
//! COLMAP's index sequence.

use super::Sampler;
use crate::math::random::shuffle;
use crate::{check_le, Result};

/// Port of `colmap::RandomSampler`: random sampler for RANSAC-based methods. A separate
/// sampler is needed per thread.
#[derive(Clone, Debug)]
pub struct RandomSampler {
    num_samples: usize,
    sample_idxs: Vec<usize>,
}

impl Sampler for RandomSampler {
    const IS_RANDOMIZED: bool = true;
    const IS_RANDOM_SAMPLER: bool = true;

    fn new(num_samples: usize) -> Self {
        Self {
            num_samples,
            sample_idxs: Vec::new(),
        }
    }

    fn initialize(&mut self, total_num_samples: usize) -> Result<()> {
        check_le!(self.num_samples, total_num_samples);
        self.sample_idxs.clear();
        self.sample_idxs.extend(0..total_num_samples);
        Ok(())
    }

    fn max_num_samples(&self) -> usize {
        usize::MAX
    }

    fn sample(&mut self, sampled_idxs: &mut Vec<usize>) -> Result<()> {
        // `static_cast<uint32_t>(num_samples_)`.
        shuffle(self.num_samples as u32, &mut self.sample_idxs)?;

        sampled_idxs.clear();
        sampled_idxs.extend_from_slice(&self.sample_idxs[..self.num_samples]);
        Ok(())
    }
}
