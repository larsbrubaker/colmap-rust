// Port of COLMAP's src/colmap/optim/progressive_sampler_test.cc, 1:1 (Suite_Name ->
// suite_name), testing colmap_rust::optim::ProgressiveSampler. The seeded index sequence is
// pinned by rust_only_sampler_sequence.rs. Each test seeds with 0 first, as gtest_main does.

use colmap_rust::math::random::set_prng_seed;
use colmap_rust::optim::{ProgressiveSampler, Sampler};
use std::collections::HashSet;

#[test]
fn progressive_sampler_less_samples() {
    set_prng_seed(0);
    let mut sampler = ProgressiveSampler::new(2);
    sampler.initialize(5).unwrap();
    assert_eq!(sampler.max_num_samples(), usize::MAX);
    for _ in 0..100 {
        let mut samples = Vec::new();
        sampler.sample(&mut samples).unwrap();
        assert_eq!(samples.len(), 2);
        assert_eq!(samples.iter().collect::<HashSet<_>>().len(), 2);
    }
}

#[test]
fn progressive_sampler_equal_samples() {
    set_prng_seed(0);
    let mut sampler = ProgressiveSampler::new(5);
    sampler.initialize(5).unwrap();
    assert_eq!(sampler.max_num_samples(), usize::MAX);
    for _ in 0..100 {
        let mut samples = Vec::new();
        sampler.sample(&mut samples).unwrap();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples.iter().collect::<HashSet<_>>().len(), 5);
    }
}

#[test]
fn progressive_sampler_progressive() {
    set_prng_seed(0);
    const NUM_SAMPLES: usize = 5;
    let mut sampler = ProgressiveSampler::new(NUM_SAMPLES);
    sampler.initialize(50).unwrap();
    let mut prev_last_sample = 5;
    for _ in 0..100 {
        let mut samples = Vec::new();
        sampler.sample(&mut samples).unwrap();
        let last = *samples.last().unwrap();
        for &sample in &samples[..samples.len() - 1] {
            assert!(sample < last);
            assert!(last >= prev_last_sample);
            prev_last_sample = last;
        }
    }
}
