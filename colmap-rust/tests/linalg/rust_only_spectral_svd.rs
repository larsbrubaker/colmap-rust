// Rust-only (COLMAP has no test for Eigen itself): hand-checked conventions of JacobiSvd and
// Svd3d/Svd4d that the numpy comparison in rust_only_spectral_oracle.rs does not pin: sorting
// and sign of singular values, thin/full factor shapes (the wide-input allocation bound is in
// the separate `tests/linalg_alloc.rs` binary), non-finite input, tiny entries, and
// convergence of exactly rank-deficient inputs in a few sweeps. Port of the SVD half of
// colmap-sharp's `ColmapSharp.Tests/LinearAlgebra/SpectralTests.cs` (the eigen-solver and
// FullPivLU half is rust_only_spectral_eigen.rs). Tier B where a tolerance appears; exact
// where the value is exactly representable.

use colmap_rust::linalg::{
    ComputationInfo, JacobiSvd, Matrix3d, Matrix4d, MatrixXd, Quaterniond, Svd3d, Svd4d,
    SvdOptions, Vector3d, VectorXd,
};
use colmap_rust::math::fns;

fn max_abs(m: &MatrixXd) -> f64 {
    m.as_slice().iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

#[test]
fn rust_only_jacobi_svd_diagonal_is_sorted_and_non_negative() {
    // diag(1, -3, 2): singular values (3, 2, 1); the negative entry flips its U column.
    let svd = JacobiSvd::new(
        &MatrixXd::from_diagonal(&VectorXd::from_slice(&[1.0, -3.0, 2.0])),
        SvdOptions::FULL_UV,
    );
    let s = svd.singular_values();
    assert_eq!(s[0], 3.0);
    assert_eq!(s[1], 2.0);
    assert_eq!(s[2], 1.0);
    let u = svd.matrix_u();
    let v = svd.matrix_v();
    assert_eq!(u[(1, 0)] * v[(1, 0)], -1.0);
    assert_eq!(u[(2, 1)] * v[(2, 1)], 1.0);
    assert_eq!(u[(0, 2)] * v[(0, 2)], 1.0);
}

#[test]
fn rust_only_jacobi_svd_thin_and_full_shapes() {
    let tall = MatrixXd::from_row_major(5, 2, &[1., 2., 3., 4., 5., 6., 7., 8., 9., 11.]);
    let thin = JacobiSvd::new(&tall, SvdOptions::THIN_UV);
    assert_eq!(thin.matrix_u().cols(), 2);
    assert_eq!(thin.matrix_v().cols(), 2);
    let wide = JacobiSvd::new(&tall.transpose(), SvdOptions::FULL_UV);
    assert_eq!(wide.matrix_u().rows(), 2);
    assert_eq!(wide.matrix_v().rows(), 5);
    assert_eq!(wide.matrix_v().cols(), 5);
    assert_eq!(wide.rank(), 2);
    let unrequested =
        std::panic::catch_unwind(|| JacobiSvd::new(&tall, SvdOptions::NONE).matrix_u());
    assert!(unrequested.is_err());
}

#[test]
fn rust_only_jacobi_svd_non_finite_input_is_invalid() {
    let mut a = MatrixXd::identity(3);
    a[(1, 2)] = f64::NAN;
    assert_eq!(
        JacobiSvd::new(&a, SvdOptions::NONE).info(),
        ComputationInfo::InvalidInput
    );
    assert_eq!(
        Svd3d::compute(&a.to_matrix3d()).info,
        ComputationInfo::InvalidInput
    );
}

// Regression: the 2x2 step's hypot underflowed to 0 and produced NaN with Info Success.
#[test]
fn rust_only_jacobi_svd_tiny_entries_do_not_underflow() {
    let a = MatrixXd::from_row_major(3, 3, &[0.0, 1e-170, 1.0, 2e-170, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let svd = JacobiSvd::new(&a, SvdOptions::FULL_UV);
    let s = svd.singular_values();
    assert_eq!(svd.info(), ComputationInfo::Success);
    assert!((s[0] - 1.0).abs() <= 1e-15);
    assert!(s[1] <= 1e-15);
    let u = svd.matrix_u();
    let v = svd.matrix_v();
    assert!(u.as_slice().iter().all(|x| x.is_finite()));
    assert!(v.as_slice().iter().all(|x| x.is_finite()));
    let error = max_abs(&(&(&(&u * &MatrixXd::from_diagonal(&s)) * &v.transpose()) - &a));
    assert!(error <= 1e-15, "error {error}");
}

// Exactly rank-deficient inputs (COLMAP's everyday case) must converge in a few sweeps: the
// absolute floor eps * ||A||_F stops rotations on roundoff next to a zero diagonal.
const FEW_SWEEPS: usize = 8;

fn assert_converges_with_null_space(label: &str, a: &MatrixXd, expected_rank: usize) {
    let svd = JacobiSvd::new(a, SvdOptions::FULL_UV);
    assert_eq!(svd.info(), ComputationInfo::Success, "{label}");
    assert!(
        svd.sweeps() <= FEW_SWEEPS,
        "{label}: {} sweeps",
        svd.sweeps()
    );
    assert_eq!(svd.rank(), expected_rank, "{label}");
    let v = svd.matrix_v();
    let scale = svd.singular_values()[0];
    for i in expected_rank..a.cols() {
        let residual = (a * &v.col(i)).norm();
        assert!(
            residual <= 1e-13 * scale,
            "{label} null column {i}: {residual}"
        );
    }
}

fn test_rotation() -> Matrix3d {
    Quaterniond::new(0.9, 0.2, -0.3, 0.25)
        .normalized()
        .to_rotation_matrix()
}

const TEST_TRANSLATION: Vector3d = Vector3d::new(0.4, -0.7, 0.3);

/// `[t]_x`, colmap-sharp's `Rigid3d.CrossProductMatrix` (COLMAP's CrossProductMatrix).
fn cross_product_matrix(t: Vector3d) -> Matrix3d {
    Matrix3d::new(0.0, -t.z, t.y, t.z, 0.0, -t.x, -t.y, t.x, 0.0)
}

#[test]
fn rust_only_jacobi_svd_exact_essential_matrix_converges() {
    let essential = cross_product_matrix(TEST_TRANSLATION) * test_rotation();
    assert_converges_with_null_space("essential", &MatrixXd::from(essential), 2);
    let fixed3 = Svd3d::compute(&essential);
    assert_eq!(fixed3.info, ComputationInfo::Success);
    assert_eq!(fixed3.rank(), 2);
}

fn noise_free_eight_point_system_converges(n: usize) {
    // Rows kron(x2, x1) of the epipolar constraint x2^T E x1 = 0 for exact correspondences.
    let rotation = test_rotation();
    let mut a = MatrixXd::zeros(n, 9);
    for k in 0..n {
        let kf = k as f64;
        let point = Vector3d::new(
            fns::sin(1.3 * kf),
            fns::cos(0.7 * kf + 0.2),
            4.0 + fns::sin(0.37 * kf),
        );
        let x1 = point / point.z;
        let moved = rotation * point + TEST_TRANSLATION;
        let x2 = moved / moved.z;
        for r in 0..3 {
            for c in 0..3 {
                a[(k, 3 * r + c)] = x2[r] * x1[c];
            }
        }
    }

    assert_converges_with_null_space(&format!("8-point n={n}"), &a, 8);
}

#[test]
fn rust_only_jacobi_svd_noise_free_eight_point_system_converges_8() {
    noise_free_eight_point_system_converges(8);
}

#[test]
fn rust_only_jacobi_svd_noise_free_eight_point_system_converges_50() {
    noise_free_eight_point_system_converges(50);
}

#[test]
fn rust_only_jacobi_svd_zero_column_converges() {
    let a = MatrixXd::from_row_major(
        4,
        4,
        &[
            1.0, 0.0, 2.0, -1.0, //
            0.5, 0.0, -1.0, 3.0, //
            2.0, 0.0, 0.3, 0.7, //
            -1.0, 0.0, 1.5, 2.0,
        ],
    );
    assert_converges_with_null_space("zero column", &a, 3);
}

// A zero column of A is an exact null vector, and it must come out exact: COLMAP's
// TriangulatePoint rejects parallel rays by testing that vector's last coordinate for == 0
// (triangulation_test.cc TriangulatePoint.ParallelRays uses this very matrix). The general
// two-angle rotation left cos(pi/2) = 6.1e-17 there.
#[test]
fn rust_only_svd4d_zero_column_null_vector_is_exact() {
    let a = Matrix4d::new(
        -1.0, 0.0, 0.0, 0.0, //
        0.0, -1.0, 0.0, 0.0, //
        -1.0, 0.0, 0.0, -1.0, //
        0.0, -1.0, 0.0, 0.0,
    );
    let svd = Svd4d::compute(&a);
    let v = svd.matrix_v;
    assert_eq!(svd.singular_values.w, 0.0);
    assert_eq!(v[(0, 3)], 0.0);
    assert_eq!(v[(1, 3)], 0.0);
    assert_eq!(v[(2, 3)].abs(), 1.0);
    assert_eq!(v[(3, 3)], 0.0);
}

#[test]
fn rust_only_jacobi_svd_rank_one_outer_product_converges() {
    let mut a = MatrixXd::zeros(9, 9);
    for r in 0..9 {
        for c in 0..9 {
            a[(r, c)] = fns::sin(r as f64 + 1.0) * fns::cos(0.5 * c as f64 + 0.3);
        }
    }
    assert_converges_with_null_space("rank-1 9x9", &a, 1);
}
