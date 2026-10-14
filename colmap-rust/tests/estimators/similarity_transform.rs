// Port of COLMAP's src/colmap/estimators/solvers/similarity_transform_test.cc
// (Suite_Name -> suite_name), testing colmap_rust::estimators::solvers::similarity_transform.
// The estimates are Tier B (a 3x3 SVD) and the robust ones Tier C (LO-RANSAC); COLMAP's
// tolerances are the bars.
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. The robust tests leave
// random_seed at -1, so RANSAC keeps drawing from that same thread PRNG, as in COLMAP.
// Arguments COLMAP draws inside one call (`Rigid3d(RandomEigenQuaterniond(),
// RandomEigenVectord<3>())`) are drawn here in clang's left-to-right order.

use crate::expect_near;
use crate::transform_matchers::{rigid3d_near, sim3d_near};
use colmap_rust::estimators::solvers::{
    estimate_rigid3d, estimate_rigid3d_robust, estimate_sim3d, estimate_sim3d_robust,
};
use colmap_rust::geometry::{Rigid3d, Sim3d};
use colmap_rust::linalg::{Matrix3x4d, Vector3d};
use colmap_rust::math::fns;
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::optim::RansacOptions;

fn generate_data(
    num_inliers: usize,
    num_outliers: usize,
    tgt_from_src: &Matrix3x4d,
) -> (Vec<Vector3d>, Vec<Vector3d>) {
    let mut src = Vec::new();
    let mut tgt = Vec::new();

    let point = |i: usize| {
        let i = i as f64;
        Vector3d::new(i, fns::sqrt(i) + 2.0, fns::sqrt(2.0 * i + 2.0))
    };

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
        let z = random_uniform_real(-5000.0, -4000.0);
        tgt.push(Vector3d::new(x, y, z));
    }

    (src, tgt)
}

fn test_estimate_rigid3d_with_num_coords(num_coords: usize) {
    set_prng_seed(0);
    let rotation = random_eigen_quaterniond();
    let translation = random_eigen_vector3d();
    let gt_tgt_from_src = Rigid3d::new(rotation, translation);
    let (src, tgt) = generate_data(num_coords, 0, &gt_tgt_from_src.to_matrix());

    let mut tgt_from_src = Rigid3d::default();
    assert!(estimate_rigid3d(&src, &tgt, &mut tgt_from_src).unwrap());
    assert!(
        gt_tgt_from_src
            .rotation
            .angular_distance(tgt_from_src.rotation)
            < 1e-6
    );
    assert!((gt_tgt_from_src.translation - tgt_from_src.translation).norm() < 1e-6);
}

#[test]
fn rigid3d_estimate_minimal() {
    test_estimate_rigid3d_with_num_coords(3);
}

#[test]
fn rigid3d_estimate_over_determined() {
    test_estimate_rigid3d_with_num_coords(100);
}

#[test]
fn rigid3d_estimate_minimal_degenerate() {
    let degenerate_src_tgt = vec![Vector3d::zeros(); 3];
    let mut tgt_from_src = Rigid3d::default();
    assert!(
        !estimate_rigid3d(&degenerate_src_tgt, &degenerate_src_tgt, &mut tgt_from_src).unwrap()
    );
}

#[test]
fn rigid3d_estimate_non_minimal_degenerate() {
    let degenerate_src_tgt = vec![Vector3d::zeros(); 5];
    let mut tgt_from_src = Rigid3d::default();
    assert!(
        !estimate_rigid3d(&degenerate_src_tgt, &degenerate_src_tgt, &mut tgt_from_src).unwrap()
    );
}

fn validate_robust_report(
    report: &colmap_rust::estimators::solvers::SimilarityTransformReport,
    num_inliers: usize,
    num_outliers: usize,
) {
    assert!(report.success);
    assert!(report.num_trials > 0);

    // Make sure outliers were detected correctly.
    assert_eq!(report.support.num_inliers, num_inliers);
    for i in 0..num_inliers + num_outliers {
        assert_eq!(report.inlier_mask[i], i < num_inliers, "sample {i}");
    }
}

#[test]
fn rigid3d_estimate_robust() {
    set_prng_seed(0);
    let num_inliers = 1000;
    let num_outliers = 400;

    let gt_tgt_from_src =
        Rigid3d::new(random_eigen_quaterniond(), Vector3d::new(100.0, 10.0, 10.0));
    let (src, tgt) = generate_data(num_inliers, num_outliers, &gt_tgt_from_src.to_matrix());

    // Robustly estimate transformation using RANSAC.
    let options = RansacOptions {
        max_error: 10.0,
        ..Default::default()
    };
    let mut tgt_from_src = Rigid3d::default();
    let report = estimate_rigid3d_robust(&src, &tgt, &options, &mut tgt_from_src).unwrap();

    validate_robust_report(&report, num_inliers, num_outliers);
    assert!(rigid3d_near(&tgt_from_src, &gt_tgt_from_src, 1e-6, 1e-6));
}

fn test_estimate_sim3d_with_num_coords(num_coords: usize) {
    set_prng_seed(0);
    let scale = random_uniform_real(0.1, 10.0);
    let rotation = random_eigen_quaterniond();
    let translation = random_eigen_vector3d();
    let gt_tgt_from_src = Sim3d::new(scale, rotation, translation);
    let (src, tgt) = generate_data(num_coords, 0, &gt_tgt_from_src.to_matrix());

    let mut tgt_from_src = Sim3d::default();
    assert!(estimate_sim3d(&src, &tgt, &mut tgt_from_src).unwrap());
    expect_near(gt_tgt_from_src.scale, tgt_from_src.scale, 1e-6);
    assert!(
        gt_tgt_from_src
            .rotation
            .angular_distance(tgt_from_src.rotation)
            < 1e-6
    );
    assert!((gt_tgt_from_src.translation - tgt_from_src.translation).norm() < 1e-6);
}

#[test]
fn sim3d_estimate_minimal() {
    test_estimate_sim3d_with_num_coords(3);
}

#[test]
fn sim3d_estimate_over_determined() {
    test_estimate_sim3d_with_num_coords(100);
}

#[test]
fn sim3d_estimate_minimal_degenerate() {
    let degenerate_src_tgt = vec![Vector3d::zeros(); 3];
    let mut tgt_from_src = Sim3d::default();
    assert!(!estimate_sim3d(&degenerate_src_tgt, &degenerate_src_tgt, &mut tgt_from_src).unwrap());
}

#[test]
fn sim3d_estimate_non_minimal_degenerate() {
    let degenerate_src_tgt = vec![Vector3d::zeros(); 5];
    let mut tgt_from_src = Sim3d::default();
    assert!(!estimate_sim3d(&degenerate_src_tgt, &degenerate_src_tgt, &mut tgt_from_src).unwrap());
}

#[test]
fn sim3d_estimate_robust() {
    set_prng_seed(0);
    let num_inliers = 1000;
    let num_outliers = 400;

    let gt_tgt_from_src = Sim3d::new(
        2.0,
        random_eigen_quaterniond(),
        Vector3d::new(100.0, 10.0, 10.0),
    );
    let (src, tgt) = generate_data(num_inliers, num_outliers, &gt_tgt_from_src.to_matrix());

    // Robustly estimate transformation using RANSAC.
    let options = RansacOptions {
        max_error: 10.0,
        ..Default::default()
    };
    let mut tgt_from_src = Sim3d::default();
    let report = estimate_sim3d_robust(&src, &tgt, &options, &mut tgt_from_src).unwrap();

    validate_robust_report(&report, num_inliers, num_outliers);
    assert!(sim3d_near(
        &tgt_from_src,
        &gt_tgt_from_src,
        1e-8,
        1e-8,
        1e-8
    ));
}
