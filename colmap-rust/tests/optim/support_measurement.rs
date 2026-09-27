// Port of COLMAP's src/colmap/optim/support_measurement_test.cc, 1:1 (Suite_Name ->
// suite_name), testing colmap_rust::optim::support_measurement. Tier A (exact).

use colmap_rust::optim::{
    InlierSupport, InlierSupportMeasurer, MEstimatorSupport, MEstimatorSupportMeasurer,
    SupportMeasurer, UniqueInlierSupport, UniqueInlierSupportMeasurer,
};

#[test]
fn inlier_support_measurer_nominal() {
    let mut support1 = InlierSupport::default();
    assert_eq!(support1.num_inliers, 0);
    assert_eq!(support1.residual_sum, f64::MAX);
    let mut measurer = InlierSupportMeasurer;
    let residuals = [-1.0, 0.0, 1.0, 2.0];
    support1 = measurer.evaluate(&residuals, 1.0).unwrap();
    assert_eq!(support1.num_inliers, 3);
    assert_eq!(support1.residual_sum, 0.0);
    let mut support2 = InlierSupport {
        num_inliers: 2,
        ..Default::default()
    };
    assert!(measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.residual_sum = support1.residual_sum;
    assert!(measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.num_inliers = support1.num_inliers;
    support2.residual_sum += 0.01;
    assert!(measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.residual_sum -= 0.01;
    assert!(!measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.residual_sum -= 0.01;
    assert!(!measurer.is_left_better(&support1, &support2));
    assert!(measurer.is_left_better(&support2, &support1));
}

#[test]
fn unique_inlier_support_measurer_nominal() {
    let mut support1 = UniqueInlierSupport::default();
    assert_eq!(support1.num_inliers, 0);
    assert_eq!(support1.num_unique_inliers, 0);
    assert_eq!(support1.residual_sum, f64::MAX);

    let mut measurer = UniqueInlierSupportMeasurer::new(vec![1, 2, 2, 3]);
    let residuals = [-1.0, 0.0, 1.0, 2.0];
    support1 = measurer.evaluate(&residuals, 1.0).unwrap();
    assert_eq!(support1.num_inliers, 3);
    assert_eq!(support1.num_unique_inliers, 2);
    assert_eq!(support1.residual_sum, 0.0);

    let mut support2 = UniqueInlierSupport {
        num_unique_inliers: support1.num_unique_inliers - 1,
        ..Default::default()
    };
    assert!(measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.num_inliers = support1.num_inliers + 1;
    assert!(measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.num_inliers = support1.num_inliers;
    assert!(measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.residual_sum = support1.residual_sum - 0.01;
    assert!(measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.residual_sum = support1.residual_sum;
    assert!(measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.num_unique_inliers = support1.num_unique_inliers;
    assert!(!measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.residual_sum = support1.residual_sum - 0.01;
    assert!(!measurer.is_left_better(&support1, &support2));
    assert!(measurer.is_left_better(&support2, &support1));
    support2.num_inliers = support1.num_inliers + 1;
    support2.residual_sum = support1.residual_sum + 0.01;
    assert!(!measurer.is_left_better(&support1, &support2));
    assert!(measurer.is_left_better(&support2, &support1));
    support2.num_unique_inliers = support1.num_unique_inliers + 1;
    support2.num_inliers = support1.num_inliers - 1;
    support2.residual_sum = support1.residual_sum + 0.01;
    assert!(!measurer.is_left_better(&support1, &support2));
    assert!(measurer.is_left_better(&support2, &support1));
}

#[test]
fn m_estimator_support_measurer_nominal() {
    let mut support1 = MEstimatorSupport::default();
    assert_eq!(support1.num_inliers, 0);
    assert_eq!(support1.score, f64::MAX);
    let mut measurer = MEstimatorSupportMeasurer;
    let residuals = [-1.0, 0.0, 1.0, 2.0];
    support1 = measurer.evaluate(&residuals, 1.0).unwrap();
    assert_eq!(support1.num_inliers, 3);
    assert_eq!(support1.score, 1.0);
    let mut support2 = support1;
    assert!(!measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.num_inliers -= 1;
    support2.score += 0.01;
    assert!(measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.score -= 0.01;
    assert!(!measurer.is_left_better(&support1, &support2));
    assert!(!measurer.is_left_better(&support2, &support1));
    support2.score -= 0.01;
    assert!(!measurer.is_left_better(&support1, &support2));
    assert!(measurer.is_left_better(&support2, &support1));
}
