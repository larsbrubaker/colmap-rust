// Port of COLMAP's src/colmap/optim/combination_sampler_test.cc, 1:1 (Suite_Name ->
// suite_name), testing colmap_rust::optim::CombinationSampler. Tier A (exact, no PRNG).

use colmap_rust::optim::{CombinationSampler, Sampler};
use std::collections::HashSet;

#[test]
fn combination_sampler_less_samples() {
    let mut sampler = CombinationSampler::new(2);
    sampler.initialize(5).unwrap();
    assert_eq!(sampler.max_num_samples(), 10);
    let mut sample_sets: Vec<HashSet<usize>> = Vec::new();
    for i in 0..10 {
        let mut samples = Vec::new();
        sampler.sample(&mut samples).unwrap();
        assert_eq!(samples.len(), 2);
        sample_sets.push(samples.iter().copied().collect());
        assert_eq!(sample_sets.last().unwrap().len(), 2);
        for set in &sample_sets[..i] {
            assert!(!set.contains(&samples[0]) || !set.contains(&samples[1]));
        }
    }
    let mut samples = Vec::new();
    sampler.sample(&mut samples).unwrap();
    assert!(sample_sets[0].contains(&samples[0]) && sample_sets[0].contains(&samples[1]));
}

#[test]
fn combination_sampler_equal_samples() {
    let mut sampler = CombinationSampler::new(5);
    sampler.initialize(5).unwrap();
    assert_eq!(sampler.max_num_samples(), 1);
    for _ in 0..100 {
        let mut samples = Vec::new();
        sampler.sample(&mut samples).unwrap();
        assert_eq!(samples.len(), 5);
        assert_eq!(samples.iter().collect::<HashSet<_>>().len(), 5);
    }
}
