// Port of COLMAP's src/colmap/optim/random_sampler_test.cc, 1:1 (Suite_Name -> suite_name),
// testing colmap_rust::optim::RandomSampler. These check sizes and uniqueness only; the
// seeded index sequence is pinned by rust_only_sampler_sequence.rs. Each test seeds with 0
// first, as gtest_main does (see tests/math/random.rs for why).

use colmap_rust::math::random::set_prng_seed;
use colmap_rust::optim::{RandomSampler, Sampler};
use std::collections::HashSet;

#[test]
fn random_sampler_less_samples() {
    set_prng_seed(0);
    let mut sampler = RandomSampler::new(2);
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
fn random_sampler_equal_samples() {
    set_prng_seed(0);
    let mut sampler = RandomSampler::new(5);
    sampler.initialize(5).unwrap();
    assert_eq!(sampler.max_num_samples(), usize::MAX);
    for _ in 0..100 {
        let mut samples = Vec::new();
        sampler.sample(&mut samples).unwrap();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples.iter().collect::<HashSet<_>>().len(), 5);
    }
}
