// Port of COLMAP's src/colmap/estimators/solvers/affine_transform_test.cc
// (Suite_Name -> suite_name), testing colmap_rust::estimators::solvers::affine_transform.
// The estimates are Tier B (an LU or SVD solve) and the robust one Tier C (LO-RANSAC);
// COLMAP's tolerances are the bars.
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. The robust test leaves
// random_seed at -1, so RANSAC keeps drawing from that same thread PRNG, as in COLMAP.

use crate::eigen_matchers::eigen_matrix_near;
use colmap_rust::estimators::solvers::{estimate_affine2d, estimate_affine2d_robust};
use colmap_rust::linalg::{Matrix2x3d, Vector2d};
use colmap_rust::math::fns;
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::random_eigen_matrixd;
use colmap_rust::optim::RansacOptions;

fn generate_data(
    num_inliers: usize,
    num_outliers: usize,
    tgt_from_src: &Matrix2x3d,
) -> (Vec<Vector2d>, Vec<Vector2d>) {
    let mut src = Vec::new();
    let mut tgt = Vec::new();

    let point = |i: usize| Vector2d::new(i as f64, fns::sqrt(i as f64) + 2.0);

    // Generate inlier data.
    for i in 0..num_inliers {
        src.push(point(i));
        tgt.push(*tgt_from_src * point(i).homogeneous());
    }

    // Add some faulty data.
    for i in 0..num_outliers {
        src.push(point(i));
        let x = random_uniform_real(-3000.0, -2000.0);
        let y = random_uniform_real(-4000.0, -3000.0);
        tgt.push(Vector2d::new(x, y));
    }

    (src, tgt)
}

fn test_estimate_affine2d_with_num_coords(num_coords: usize) {
    set_prng_seed(0);
    let gt_tgt_from_src: Matrix2x3d = random_eigen_matrixd();
    let (src, tgt) = generate_data(num_coords, 0, &gt_tgt_from_src);

    let mut tgt_from_src = Matrix2x3d::zeros();
    assert!(estimate_affine2d(&src, &tgt, &mut tgt_from_src).unwrap());
    assert!(eigen_matrix_near(
        tgt_from_src.as_slice(),
        gt_tgt_from_src.as_slice(),
        1e-6
    ));
}

#[test]
fn affine2d_estimate_minimal() {
    test_estimate_affine2d_with_num_coords(3);
}

#[test]
fn affine2d_estimate_over_determined() {
    test_estimate_affine2d_with_num_coords(100);
}

#[test]
fn affine2d_estimate_minimal_degenerate() {
    set_prng_seed(0);
    let degenerate_src_tgt = vec![Vector2d::zeros(); 3];
    let mut tgt_from_src = Matrix2x3d::zeros();
    assert!(
        !estimate_affine2d(&degenerate_src_tgt, &degenerate_src_tgt, &mut tgt_from_src).unwrap()
    );
}

#[test]
fn affine2d_estimate_non_minimal_degenerate() {
    set_prng_seed(0);
    let degenerate_src_tgt = vec![Vector2d::zeros(); 5];
    let mut tgt_from_src = Matrix2x3d::zeros();
    assert!(
        !estimate_affine2d(&degenerate_src_tgt, &degenerate_src_tgt, &mut tgt_from_src).unwrap()
    );
}

#[test]
fn affine2d_estimate_robust() {
    set_prng_seed(0);
    let num_inliers = 1000;
    let num_outliers = 400;

    let gt_tgt_from_src: Matrix2x3d = random_eigen_matrixd();
    let (src, tgt) = generate_data(num_inliers, num_outliers, &gt_tgt_from_src);

    // Robustly estimate transformation using RANSAC.
    let options = RansacOptions {
        max_error: 10.0,
        ..RansacOptions::default()
    };
    let mut tgt_from_src = Matrix2x3d::zeros();
    let report = estimate_affine2d_robust(&src, &tgt, &options, &mut tgt_from_src).unwrap();

    assert!(report.success);
    assert!(report.num_trials > 0);

    // Make sure outliers were detected correctly.
    assert_eq!(report.support.num_inliers, num_inliers);
    for i in 0..num_inliers + num_outliers {
        if i >= num_inliers {
            assert!(!report.inlier_mask[i]);
        } else {
            assert!(report.inlier_mask[i]);
        }
    }

    assert!(eigen_matrix_near(
        tgt_from_src.as_slice(),
        gt_tgt_from_src.as_slice(),
        1e-6
    ));
}
