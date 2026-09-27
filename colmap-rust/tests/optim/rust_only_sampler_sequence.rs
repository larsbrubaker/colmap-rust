// Rust-only tests (no COLMAP counterpart), ported from colmap-sharp's
// ColmapSharp.Tests/Optim/SamplerSequenceTests.cs. They pin the Tier A claim of
// random_sampler.rs and progressive_sampler.rs - same seed, same index sequence as COLMAP -
// which the ported *_test.cc files cannot see, since those only check sizes and uniqueness.
// They also cover Sampler::sample_xy.
//
// Expected sequences come from colmap-sharp's differential harness: COLMAP 4.2.0's
// random.cc, random_sampler.cc and progressive_sampler.cc compiled unmodified with Apple
// clang and libc++ (macOS arm64), calling SetPRNGSeed(0), constructing the sampler,
// Initialize, then Sample repeatedly.

use colmap_rust::math::random::set_prng_seed;
use colmap_rust::optim::{ProgressiveSampler, RandomSampler, Sampler};

fn draw(sampler: &mut impl Sampler, count: usize) -> Vec<Vec<usize>> {
    (0..count)
        .map(|_| {
            let mut samples = Vec::new();
            sampler.sample(&mut samples).unwrap();
            samples
        })
        .collect()
}

#[test]
fn rust_only_random_sampler_seeded_sequence_matches_cpp() {
    set_prng_seed(0);
    let expected: [[usize; 3]; 8] = [
        [5, 1, 0],
        [3, 8, 5],
        [0, 6, 4],
        [5, 1, 6],
        [4, 9, 8],
        [9, 7, 1],
        [4, 8, 3],
        [2, 7, 1],
    ];
    let mut sampler = RandomSampler::new(3);
    sampler.initialize(10).unwrap();
    let actual = draw(&mut sampler, expected.len());
    assert_eq!(actual, expected.map(|s| s.to_vec()).to_vec());
}

#[test]
fn rust_only_progressive_sampler_seeded_sequence_matches_cpp() {
    set_prng_seed(0);
    let expected: [[usize; 3]; 8] = [
        [0, 1, 4],
        [0, 1, 4],
        [1, 2, 4],
        [0, 2, 4],
        [0, 2, 4],
        [1, 2, 4],
        [2, 0, 4],
        [1, 0, 4],
    ];
    let mut sampler = ProgressiveSampler::new(3);
    sampler.initialize(20).unwrap();
    let actual = draw(&mut sampler, expected.len());
    assert_eq!(actual, expected.map(|s| s.to_vec()).to_vec());
}

#[test]
fn rust_only_sample_xy_gathers_sampled_elements() {
    // Same seed and sampler as the random sequence test, so the first sample is {5, 1, 0}.
    set_prng_seed(0);
    let x: Vec<i32> = (0..10).map(|i| i * 10).collect();
    let y: Vec<String> = (0..10).map(|i| format!("y{i}")).collect();
    let mut x_rand = Vec::new();
    let mut y_rand = Vec::new();

    let mut sampler = RandomSampler::new(3);
    sampler.initialize(10).unwrap();
    sampler.sample_xy(&x, &y, &mut x_rand, &mut y_rand).unwrap();

    assert_eq!(x_rand, vec![50, 10, 0]);
    assert_eq!(y_rand, vec!["y5", "y1", "y0"]);
}
