// Rust-only tests (PoseLib's p3p and univariate helpers have no unit test in COLMAP or
// colmap-sharp; COLMAP tests p3p through P3PEstimator in absolute_pose_test.cc) of
// colmap_rust::estimators::solvers::poselib::p3p. Tier B: the ground-truth pose is among the
// solutions, every solution reprojects its three points onto their bearings, and the cubic
// and quadratic helpers return roots of their polynomials.

use colmap_rust::estimators::solvers::poselib::p3p::{p3p, root2real, solve_cubic_single_real};
use colmap_rust::estimators::solvers::poselib::CameraPose;
use colmap_rust::linalg::{AngleAxisd, Vector3d};

fn max_abs3(v: Vector3d) -> f64 {
    let a = v.cwise_abs();
    a.x.max(a.y).max(a.z)
}

/// Bearings of three points seen from several poses, including point orders that exercise
/// each branch of p3p's largest-distance switch.
#[test]
fn rust_only_recovers_ground_truth_pose() {
    let points = [
        [
            Vector3d::new(1.0, 0.5, 4.0),
            Vector3d::new(-1.2, 0.3, 5.0),
            Vector3d::new(0.4, -0.9, 3.0),
        ],
        [
            Vector3d::new(-2.0, 0.1, 6.0),
            Vector3d::new(0.3, 0.2, 5.0),
            Vector3d::new(2.5, -0.4, 4.0),
        ],
        [
            Vector3d::new(0.2, 1.5, 3.0),
            Vector3d::new(1.0, -1.8, 4.5),
            Vector3d::new(-0.5, 0.0, 5.5),
        ],
    ];
    let poses = [
        (
            0.3,
            Vector3d::new(0.1, 1.0, -0.2),
            Vector3d::new(0.2, -0.4, 1.5),
        ),
        (
            1.1,
            Vector3d::new(-0.7, 0.2, 0.5),
            Vector3d::new(-0.3, 0.1, 0.8),
        ),
        (
            0.05,
            Vector3d::new(0.0, 0.0, 1.0),
            Vector3d::new(0.0, 0.0, 0.5),
        ),
    ];

    for big_x in &points {
        for &(angle, axis, t) in &poses {
            let r = AngleAxisd::new(angle, axis.normalized()).to_rotation_matrix();
            let x: Vec<Vector3d> = big_x.iter().map(|&p| (r * p + t).normalized()).collect();

            let mut output = vec![CameraPose::identity()];
            let n = p3p(&x, big_x, &mut output);
            assert_eq!(n, output.len());
            assert!((1..=4).contains(&n), "{n} solutions");

            let found = output
                .iter()
                .any(|pose| (pose.r() - r).norm() < 1e-6 && max_abs3(pose.t - t) < 1e-6);
            assert!(found, "ground truth missing from {output:?}");

            for pose in &output {
                for (xi, &pi) in x.iter().zip(big_x) {
                    let proj = (pose.r() * pi + pose.t).normalized();
                    assert!(max_abs3(proj - *xi) < 1e-6, "{proj:?} vs {xi:?}");
                }
            }
        }
    }
}

#[test]
fn rust_only_solve_cubic_single_real() {
    // (x - 2)(x^2 + 1) = x^3 - 2x^2 + x - 2: one real root.
    let (single, root) = solve_cubic_single_real(-2.0, 1.0, -2.0);
    assert!(single);
    assert!((root - 2.0).abs() < 1e-12, "{root}");

    // (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6: three real roots, one is returned.
    let (single, root) = solve_cubic_single_real(-6.0, 11.0, -6.0);
    assert!(!single);
    let residual = ((root - 6.0) * root + 11.0) * root - 6.0;
    assert!(residual.abs() < 1e-12, "{root}");
}

#[test]
fn rust_only_root2real() {
    // x^2 - 3x + 2 = (x - 1)(x - 2).
    let (r1, r2) = root2real(-3.0, 2.0).expect("real roots");
    assert!((r1 - 2.0).abs() < 1e-15 && (r2 - 1.0).abs() < 1e-15);
    // x^2 + 3x + 2 = (x + 1)(x + 2); for b >= 0 the roots come from 2c / (-b -+ y), so the
    // larger-magnitude root is first.
    let (r1, r2) = root2real(3.0, 2.0).expect("real roots");
    assert!((r1 + 2.0).abs() < 1e-15 && (r2 + 1.0).abs() < 1e-15);
    // No real roots.
    assert_eq!(root2real(0.0, 1.0), None);
    // A discriminant just below zero counts as a double root, with PoseLib's -2 second root.
    assert_eq!(root2real(2.0, 1.0 + 1e-14), Some((-1.0, -2.0)));
}
