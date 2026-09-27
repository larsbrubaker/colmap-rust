// Port of COLMAP's src/colmap/optim/loransac_test.cc (Suite_Name -> suite_name), testing
// colmap_rust::optim::loransac.
//
// LORANSAC<...>::Report is RansacReport, instantiated with the model type of
// MinSamples3Estimator (test_estimators.rs), which stands in for
// SimilarityTransformEstimator<3>. The estimation cases SimilarityTransform and
// ParallelSimilarityTransform need the production SimilarityTransformEstimator<3> and land
// with it in Phase 6; rust_only_ransac.rs runs the same checks on line estimators meanwhile.

use crate::test_estimators::MinSamples3Estimator;
use colmap_rust::optim::{Estimator, InlierSupport, RansacReport};

#[test]
fn loransac_report() {
    let report = RansacReport::<<MinSamples3Estimator as Estimator>::M, InlierSupport>::default();
    assert!(!report.success);
    assert_eq!(report.num_trials, 0);
    assert_eq!(report.support.num_inliers, 0);
    assert_eq!(report.support.residual_sum, f64::MAX);
    assert_eq!(report.inlier_mask.len(), 0);
}
