// Port of COLMAP's src/colmap/optim/loransac_test.cc (Suite_Name -> suite_name), testing
// colmap_rust::optim::loransac with the production SimilarityTransformEstimator<3> as both
// the minimal and the local estimator (it has no Refine, so the local one goes through
// EstimateAsLocal). The estimation cases are Tier C (exact inlier set, model within 1e-6) and
// start with set_prng_seed(0), as COLMAP's gtest_main does.

use crate::test_estimators::{generate_test_data, validate_report};
use colmap_rust::estimators::solvers::SimilarityTransformEstimator;
use colmap_rust::linalg::Matrix3x4d;
use colmap_rust::math::random::{default_prng_seed, set_prng_seed};
use colmap_rust::optim::{EstimateAsLocal, InlierSupport, LoRansac, RansacOptions, RansacReport};

type LoRansac3 =
    LoRansac<SimilarityTransformEstimator, EstimateAsLocal<SimilarityTransformEstimator>>;

fn new_loransac(options: &RansacOptions) -> LoRansac3 {
    LoRansac3::new(
        options,
        SimilarityTransformEstimator,
        EstimateAsLocal(SimilarityTransformEstimator),
    )
    .unwrap()
}

#[test]
fn loransac_report() {
    let report = RansacReport::<Matrix3x4d, InlierSupport>::default();
    assert!(!report.success);
    assert_eq!(report.num_trials, 0);
    assert_eq!(report.support.num_inliers, 0);
    assert_eq!(report.support.residual_sum, f64::MAX);
    assert_eq!(report.inlier_mask.len(), 0);
}

#[test]
fn loransac_similarity_transform() {
    set_prng_seed(0);
    let data = generate_test_data(1000, 400);

    let options = RansacOptions {
        max_error: 10.0,
        random_seed: default_prng_seed(),
        ..Default::default()
    };
    let report = new_loransac(&options)
        .estimate(&data.src, &data.tgt)
        .unwrap();

    validate_report(&report, &data);
}

#[test]
fn loransac_parallel_similarity_transform() {
    set_prng_seed(0);
    let data = generate_test_data(1000, 400);

    let options = RansacOptions {
        max_error: 10.0,
        random_seed: default_prng_seed(),
        num_threads: 4,
        ..Default::default()
    };
    let report = new_loransac(&options)
        .estimate(&data.src, &data.tgt)
        .unwrap();

    validate_report(&report, &data);
}
