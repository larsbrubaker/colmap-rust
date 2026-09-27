// Port of COLMAP's src/colmap/optim/sprt_test.cc, 1:1 (Suite_Name -> suite_name), testing
// colmap_rust::optim::Sprt. `Evaluate`'s out-parameters are the fields of the returned
// SprtEvaluation.

use colmap_rust::optim::{Sprt, SprtOptions};

#[test]
fn sprt_evaluate_all_inliers() {
    let options = SprtOptions {
        delta: 0.05,
        epsilon: 0.5,
        ..Default::default()
    };
    let sprt = Sprt::new(options);

    // All residuals are small (inliers)
    let residuals = vec![0.1; 100];
    let result = sprt.evaluate(&residuals, 1.0);

    assert!(result.accepted);
    assert_eq!(result.num_inliers, 100);
    assert_eq!(result.num_eval_samples, 100);
}

#[test]
fn sprt_evaluate_all_outliers() {
    let options = SprtOptions {
        delta: 0.05,
        epsilon: 0.5,
        ..Default::default()
    };
    let sprt = Sprt::new(options);

    // All residuals are large (outliers) - should trigger early rejection
    let residuals = vec![10.0; 100];
    let result = sprt.evaluate(&residuals, 1.0);

    assert!(!result.accepted);
    assert_eq!(result.num_inliers, 0);
    assert!(result.num_eval_samples < 100);
}

#[test]
fn sprt_evaluate_mixed_early_reject() {
    let options = SprtOptions {
        delta: 0.05,
        epsilon: 0.9,
        ..Default::default()
    };
    let sprt = Sprt::new(options);

    // Mostly outliers - should reject early
    let mut residuals = vec![10.0; 1000];
    // Sprinkle a few inliers
    residuals[0] = 0.1;
    residuals[10] = 0.1;

    let result = sprt.evaluate(&residuals, 1.0);

    assert!(!result.accepted);
    // With epsilon=0.9 and delta=0.05, the likelihood ratio exceeds the decision threshold
    // after processing the inlier at index 0 and 4 subsequent outliers.
    assert_eq!(result.num_inliers, 1);
    assert_eq!(result.num_eval_samples, 5);
}

#[test]
fn sprt_evaluate_empty() {
    let options = SprtOptions::default();
    let sprt = Sprt::new(options);

    let residuals: Vec<f64> = Vec::new();
    let result = sprt.evaluate(&residuals, 1.0);

    assert!(result.accepted);
    assert_eq!(result.num_inliers, 0);
    assert_eq!(result.num_eval_samples, 0);
}
