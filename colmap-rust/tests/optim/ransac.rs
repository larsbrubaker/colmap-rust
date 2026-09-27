// Port of COLMAP's src/colmap/optim/ransac_test.cc (Suite_Name -> suite_name), testing
// colmap_rust::optim::ransac. ComputeNumTrials is Tier A (exact trial counts).
//
// RANSAC<SimilarityTransformEstimator<3>> is instantiated with MinSamples3Estimator
// (test_estimators.rs), which has the same kMinNumSamples = 3, the only thing Report and
// ComputeNumTrials read from the estimator. The estimation cases SimilarityTransform,
// ParallelSimilarityTransform and ReproducibilityWithRandomSeed need the production
// SimilarityTransformEstimator<3> and land with it in Phase 6; rust_only_ransac.rs runs the
// same checks on a line estimator meanwhile.

use crate::test_estimators::MinSamples3Estimator;
use colmap_rust::optim::{InlierSupport, Ransac, RansacOptions, RansacReport};

type Ransac3 = Ransac<MinSamples3Estimator>;

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
    let report = RansacReport::<f64, InlierSupport>::default();
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
