// Port of COLMAP's src/colmap/math/random_test.cc (1:1, same test names in snake_case).
// These check ranges, moments and repeatability; the bit-exact draws are pinned by
// rust_only_random_oracle.rs.
//
// gtest_main reseeds the PRNG with 0 before every test. libtest gives each test a fresh
// thread (whose PRNG lazily seeds with kDefaultPRNGSeed = 0, the same state) natively, but
// wasm32-wasip1 runs every test on one thread, so tests that draw call `seed_like_gtest()`
// first.

use colmap_rust::math::random::{
    prng_is_set, random_gaussian, random_uniform_integer, random_uniform_real, reset_prng,
    set_prng_seed, set_prng_seed_default, shuffle,
};
use colmap_rust::math::{mean, std_dev};

fn seed_like_gtest() {
    set_prng_seed(0);
}

#[test]
fn prng_seed_nominal() {
    reset_prng();
    assert!(!prng_is_set());
    set_prng_seed_default();
    assert!(prng_is_set());
    set_prng_seed(0);
    assert!(prng_is_set());
    // Each thread defines their own PRNG instance. wasm32 has no threads to spawn.
    #[cfg(not(target_family = "wasm"))]
    std::thread::spawn(|| {
        assert!(!prng_is_set());
        set_prng_seed_default();
        assert!(prng_is_set());
        set_prng_seed(0);
        assert!(prng_is_set());
    })
    .join()
    .unwrap();
}

#[test]
fn repeatability_nominal() {
    const K_NUM_POINTS: usize = 100;

    set_prng_seed(0);
    let numbers1: Vec<i32> = (0..K_NUM_POINTS)
        .map(|_| random_uniform_integer(0, 10000))
        .collect();

    set_prng_seed(1);
    let numbers2: Vec<i32> = (0..K_NUM_POINTS)
        .map(|_| random_uniform_integer(0, 10000))
        .collect();

    set_prng_seed(0);
    let numbers3: Vec<i32> = (0..K_NUM_POINTS)
        .map(|_| random_uniform_integer(0, 10000))
        .collect();

    assert_eq!(numbers1, numbers3);
    let all_equal = numbers1.iter().zip(&numbers2).all(|(a, b)| a == b);
    assert!(!all_equal);
}

#[test]
fn random_uniform_integer_nominal() {
    seed_like_gtest();
    for _ in 0..1000 {
        assert!(random_uniform_integer(-100, 100) >= -100);
        assert!(random_uniform_integer(-100, 100) <= 100);
    }
}

#[test]
fn random_uniform_real_nominal() {
    seed_like_gtest();
    for _ in 0..1000 {
        assert!(random_uniform_real(-100.0, 100.0) >= -100.0);
        assert!(random_uniform_real(-100.0, 100.0) <= 100.0);
    }
}

#[test]
fn random_gaussian_nominal() {
    seed_like_gtest();
    const K_MEAN: f64 = 1.0;
    const K_SIGMA: f64 = 1.0;
    const K_NUM_VALUES: usize = 100000;
    let values: Vec<f64> = (0..K_NUM_VALUES)
        .map(|_| random_gaussian(K_MEAN, K_SIGMA))
        .collect();
    assert!((mean(&values).unwrap() - K_MEAN).abs() <= 1e-2);
    assert!((std_dev(&values).unwrap() - K_SIGMA).abs() <= 1e-2);
}

#[test]
fn shuffle_none_nominal() {
    seed_like_gtest();
    let mut numbers: Vec<i32> = Vec::new();
    shuffle(0, &mut numbers).unwrap();
    numbers = vec![1, 2, 3, 4, 5];
    let mut shuffled_numbers = numbers.clone();
    shuffle(0, &mut shuffled_numbers).unwrap();
    assert_eq!(numbers, shuffled_numbers);
}

#[test]
fn shuffle_all_nominal() {
    seed_like_gtest();
    let numbers: Vec<i32> = (0..1000).collect();
    let mut shuffled_numbers = numbers.clone();
    shuffle(1000, &mut shuffled_numbers).unwrap();
    let num_shuffled = numbers
        .iter()
        .zip(&shuffled_numbers)
        .filter(|(a, b)| a != b)
        .count();
    assert!(num_shuffled > 0);
}
