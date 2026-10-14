// Port of COLMAP's src/colmap/optim/ransac_test.cc (Suite_Name -> suite_name), testing
// colmap_rust::optim::ransac with the production SimilarityTransformEstimator<3>.
// ComputeNumTrials is Tier A (exact trial counts); the estimation cases are Tier C (exact
// inlier set, model within 1e-6). Each estimation case starts with set_prng_seed(0), as
// COLMAP's gtest_main does.

use crate::test_estimators::{generate_test_data, validate_report};
use colmap_rust::estimators::solvers::SimilarityTransformEstimator;
use colmap_rust::linalg::Matrix3x4d;
use colmap_rust::math::random::{default_prng_seed, set_prng_seed};
use colmap_rust::optim::{InlierSupport, Ransac, RansacOptions, RansacReport};

type Ransac3 = Ransac<SimilarityTransformEstimator>;

#[test]
fn ransac_options() {
    let options = RansacOptions::default();
    assert_eq!(options.max_error, 0.0);
    assert_eq!(options.min_inlier_ratio, 0.1);
    assert_eq!(options.confidence, 0.99);
    assert_eq!(options.min_num_trials, 0);
    assert_eq!(options.max_num_trials, i32::MAX);
}

#[test]
fn ransac_report() {
    let report = RansacReport::<Matrix3x4d, InlierSupport>::default();
    assert!(!report.success);
    assert_eq!(report.num_trials, 0);
    assert_eq!(report.support.num_inliers, 0);
    assert_eq!(report.support.residual_sum, f64::MAX);
    assert_eq!(report.inlier_mask.len(), 0);
}

#[test]
fn ransac_num_trials() {
    // size_t's maximum on a 64-bit target, as COLMAP's literal.
    assert_eq!(
        Ransac3::compute_num_trials(1, 100, 0.99, 1.0) as u64,
        18446744073709551615u64
    );
    assert_eq!(Ransac3::compute_num_trials(10, 100, 0.99, 1.0), 6204);
    assert_eq!(Ransac3::compute_num_trials(10, 100, 0.999, 1.0), 9305);
    assert_eq!(Ransac3::compute_num_trials(10, 100, 0.999, 2.0), 18610);
    assert_eq!(Ransac3::compute_num_trials(50, 100, 0.99, 1.0), 36);
    assert_eq!(Ransac3::compute_num_trials(50, 100, 0.999, 1.0), 54);
    assert_eq!(Ransac3::compute_num_trials(100, 100, 0.99, 1.0), 1);
    assert_eq!(Ransac3::compute_num_trials(100, 100, 0.999, 1.0), 1);
    assert_eq!(Ransac3::compute_num_trials(100, 100, 0.0, 1.0), 1);
}

#[test]
fn ransac_similarity_transform() {
    set_prng_seed(0);
    let data = generate_test_data(1000, 400);

    let options = RansacOptions {
        max_error: 10.0,
        random_seed: default_prng_seed(),
        ..Default::default()
    };
    let mut ransac = Ransac3::new(&options, SimilarityTransformEstimator).unwrap();
    let report = ransac.estimate(&data.src, &data.tgt).unwrap();

    validate_report(&report, &data);
}

#[test]
fn ransac_parallel_similarity_transform() {
    set_prng_seed(0);
    let data = generate_test_data(1000, 400);

    let options = RansacOptions {
        max_error: 10.0,
        random_seed: default_prng_seed(),
        num_threads: 4,
        ..Default::default()
    };
    let mut ransac = Ransac3::new(&options, SimilarityTransformEstimator).unwrap();
    let report = ransac.estimate(&data.src, &data.tgt).unwrap();

    validate_report(&report, &data);
}

#[test]
fn ransac_reproducibility_with_random_seed() {
    set_prng_seed(0);
    let data = generate_test_data(1000, 400);

    let options1 = RansacOptions {
        max_error: 10.0,
        random_seed: 42,
        ..Default::default()
    };
    let mut ransac1 = Ransac3::new(&options1, SimilarityTransformEstimator).unwrap();
    let report1 = ransac1.estimate(&data.src, &data.tgt).unwrap();

    let mut options2 = options1;
    let mut ransac2 = Ransac3::new(&options2, SimilarityTransformEstimator).unwrap();
    let report2 = ransac2.estimate(&data.src, &data.tgt).unwrap();

    assert!(report1.success);
    assert!(report2.success);

    // Results should be exactly the same.
    assert_eq!(report1.support.num_inliers, report2.support.num_inliers);
    assert_eq!(report1.inlier_mask, report2.inlier_mask);
    assert_eq!(report1.model, report2.model);

    // Now change the seed.
    options2.random_seed = 123;
    let mut ransac3 = Ransac3::new(&options2, SimilarityTransformEstimator).unwrap();
    let report3 = ransac3.estimate(&data.src, &data.tgt).unwrap();

    assert!(report3.success);

    // Results should now differ.
    assert_ne!(report1.model, report3.model);
}
