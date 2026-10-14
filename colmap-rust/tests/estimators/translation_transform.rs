// Port of COLMAP's src/colmap/estimators/solvers/translation_transform_test.cc
// (Suite_Name -> suite_name), testing colmap_rust::estimators::solvers::translation_transform.
// Tier A code (means and a difference); COLMAP's 1e-6 tolerances are kept. The test starts
// with set_prng_seed(0), as COLMAP's gtest_main does; the two coordinates of each
// `Eigen::Vector2d(RandomUniformReal(...), RandomUniformReal(...))` are drawn in clang's
// left-to-right order.

use crate::expect_near;
use colmap_rust::estimators::solvers::TranslationTransformEstimator;
use colmap_rust::linalg::Vector2d;
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::optim::Estimator;

fn random_point() -> Vector2d {
    let x = random_uniform_real(-1000.0, 1000.0);
    let y = random_uniform_real(-1000.0, 1000.0);
    Vector2d::new(x, y)
}

#[test]
fn translation_transform_estimate() {
    set_prng_seed(0);
    const K_NUM_POINTS: usize = 100;

    let src: Vec<Vector2d> = (0..K_NUM_POINTS).map(|_| random_point()).collect();

    let translation = random_point();

    let dst: Vec<Vector2d> = src.iter().map(|&p| p + translation).collect();

    let mut models = Vec::new();
    TranslationTransformEstimator
        .estimate(&src, &dst, &mut models)
        .unwrap();

    assert_eq!(models.len(), 1);
    let estimated_translation = models[0];

    expect_near(translation.x, estimated_translation.x, 1e-6);
    expect_near(translation.y, estimated_translation.y, 1e-6);

    let mut residuals = Vec::new();
    TranslationTransformEstimator
        .residuals(&src, &dst, &estimated_translation, &mut residuals)
        .unwrap();

    for residual in residuals {
        assert!(residual < 1e-6);
    }
}
