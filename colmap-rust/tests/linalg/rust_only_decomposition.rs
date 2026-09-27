// Rust-only (COLMAP has no test for Eigen itself): PartialPivLu, Llt, Ldlt, HouseholderQr and
// ColPivHouseholderQr in colmap_rust::linalg. Port of colmap-sharp's
// `ColmapSharp.Tests/LinearAlgebra/DecompositionTests.cs`. Each decomposition must
// reconstruct its input from its factors and its solve must leave a small residual; small
// hand-checked cases pin the conventions (pivot choice, reflector signs, rank threshold,
// non-positive-definite detection). Tier B: tolerances are 1e-12 relative (is_approx's
// default) for reconstructions of well-conditioned O(1) inputs. numpy agreement is in
// `rust_only_decomposition_oracle.rs`.

use colmap_rust::linalg::{
    ColPivHouseholderQr, ComputationInfo, HouseholderQr, Ldlt, Llt, Matrix3d, MatrixXd,
    PartialPivLu, VectorXd, DUMMY_PRECISION,
};

fn general() -> MatrixXd {
    MatrixXd::from_row_major(
        4,
        4,
        &[
            0.5, -1.2, 2.0, 0.3, //
            1.7, 0.4, -0.6, 1.1, //
            -0.9, 2.2, 0.8, -1.4, //
            0.2, -0.3, 1.5, 0.9,
        ],
    )
}

fn tall() -> MatrixXd {
    MatrixXd::from_row_major(
        6,
        3,
        &[
            1.0, 0.2, -0.5, //
            0.3, -1.1, 0.7, //
            -0.8, 0.4, 1.3, //
            0.6, 0.9, -0.2, //
            -0.1, -0.7, 0.4, //
            1.2, 0.5, 0.8,
        ],
    )
}

fn spd() -> MatrixXd {
    let gram = tall().transpose_times_self();
    &gram + &(&MatrixXd::identity(3) * 0.1)
}

fn residual(a: &MatrixXd, x: &VectorXd, b: &VectorXd) -> f64 {
    (&(a * x) - b).norm()
}

#[test]
fn rust_only_partial_piv_lu_reconstructs_solves_and_inverts() {
    let general = general();
    let lu = PartialPivLu::new(&general);
    let b = VectorXd::from_slice(&[1.0, -2.0, 0.5, 3.0]);
    let x = lu.solve(&b);
    assert!((&lu.permutation_p() * &general).is_approx(&(&lu.matrix_l() * &lu.matrix_u())));
    assert!(residual(&general, &x, &b) < 1e-13);
    assert!((&general * &lu.inverse()).is_approx(&MatrixXd::identity(4)));
    // The first row's pivot is the largest |entry| of column 0 (1.7, row 1).
    assert_eq!(lu.permutation_indices()[0], 1);
}

#[test]
fn rust_only_partial_piv_lu_determinant_matches_closed_form() {
    let m3 = Matrix3d::new(2.0, -1.0, 0.5, 3.0, 4.0, -2.0, 1.0, 0.25, 5.0);
    assert!((MatrixXd::from(m3).determinant() - m3.determinant()).abs() <= 1e-12);
    let m4 = general().to_matrix4d();
    assert!((MatrixXd::from(m4).determinant() - m4.determinant()).abs() <= 1e-12);
    assert_eq!(
        MatrixXd::from_row_major(2, 2, &[0.0, 1.0, 1.0, 0.0]).determinant(),
        -1.0
    );
}

#[test]
fn rust_only_llt_reconstructs_and_solves() {
    let a = spd();
    let llt = Llt::new(&a);
    let b = VectorXd::from_slice(&[0.3, -1.0, 2.0]);
    let l = llt.matrix_l();
    assert_eq!(llt.info(), ComputationInfo::Success);
    assert!((&l * &llt.matrix_u()).is_approx(&a));
    assert_eq!(l[(0, 1)], 0.0);
    assert!(residual(&a, &llt.solve(&b), &b) < 1e-13);
}

#[test]
fn rust_only_llt_reports_not_positive_definite() {
    let indefinite = MatrixXd::from_row_major(2, 2, &[1.0, 2.0, 2.0, 1.0]);
    assert_eq!(
        Llt::new(&indefinite).info(),
        ComputationInfo::NumericalIssue
    );
}

#[test]
fn rust_only_ldlt_reconstructs_and_solves() {
    // Symmetric indefinite but with a nonsingular diagonally pivoted LDL^T, plus the SPD
    // case; A = P^T L D L^T P.
    let indefinite =
        MatrixXd::from_row_major(3, 3, &[1.0, 0.5, 0.2, 0.5, -3.0, 0.4, 0.2, 0.4, 2.0]);
    for a in [spd(), indefinite.clone()] {
        let ldlt = Ldlt::new(&a);
        let p = ldlt.permutation_p();
        let reconstructed = &(&(&(&p.transpose() * &ldlt.matrix_l())
            * &MatrixXd::from_diagonal(&ldlt.vector_d()))
            * &ldlt.matrix_l().transpose())
            * &p;
        let b = VectorXd::from_slice(&[1.0, 2.0, -0.5]);
        assert_eq!(ldlt.info(), ComputationInfo::Success);
        assert!(reconstructed.is_approx(&a));
        assert!(residual(&a, &ldlt.solve(&b), &b) < 1e-13);
    }

    let indefinite_ldlt = Ldlt::new(&indefinite);
    // |-3| is the largest diagonal entry, so it is pivoted first.
    assert_eq!(indefinite_ldlt.transpositions()[0], 1);
    assert!(!indefinite_ldlt.is_positive());
    assert!(!indefinite_ldlt.is_negative());
    assert!(Ldlt::new(&spd()).is_positive());
}

#[test]
fn rust_only_ldlt_semidefinite_solve_is_consistent() {
    // Rank-1 PSD matrix v v^T: D has one nonzero entry, and a b in the range is solved.
    let v = MatrixXd::from_column_major(3, 1, &[1.0, 2.0, -1.0]);
    let a = &v * &v.transpose();
    let ldlt = Ldlt::new(&a);
    let b = &v.col(0) * 3.0;
    assert_eq!(ldlt.vector_d()[1], 0.0);
    assert_eq!(ldlt.vector_d()[2], 0.0);
    assert!(residual(&a, &ldlt.solve(&b), &b) < 1e-13);
}

#[test]
fn rust_only_ldlt_zero_pivot_with_nonzero_column_is_numerical_issue() {
    // [[0, 1], [1, 0]] has no diagonally pivoted LDL^T: both diagonal entries are 0 but the
    // off-diagonal is not. tiny_solver rejects a step when info() != Success.
    let a = MatrixXd::from_row_major(2, 2, &[0.0, 1.0, 1.0, 0.0]);
    assert_eq!(Ldlt::new(&a).info(), ComputationInfo::NumericalIssue);
}

#[test]
fn rust_only_householder_qr_reconstructs_and_is_orthogonal() {
    for a in [general(), tall(), tall().transpose()] {
        let qr = HouseholderQr::new(&a);
        let q = qr.householder_q();
        assert!((&q * &qr.matrix_r()).is_approx(&a));
        assert!(q.is_unitary(DUMMY_PRECISION));
        assert!(qr.matrix_r().is_upper_triangular(DUMMY_PRECISION));
    }
}

#[test]
fn rust_only_householder_qr_sign_convention_and_least_squares() {
    // LAPACK/Eigen convention: R(0,0) = -sign(a00) * ||a0||.
    let column = MatrixXd::from_column_major(3, 1, &[3.0, 0.0, 4.0]);
    let qr = HouseholderQr::new(&column);
    let b = VectorXd::from_slice(&[1.0, 0.0, -1.0, 2.0, 0.5, 1.0]);
    let tall = tall();
    let x = HouseholderQr::new(&tall).solve(&b);

    // Normal equations residual A^T (A x - b) vanishes at the least-squares solution.
    let normal_residual = tall.transpose_times_vector(&(&(&tall * &x) - &b));
    assert!((qr.matrix_r()[(0, 0)] - -5.0).abs() <= 1e-15);
    assert!((qr.householder_q()[(0, 0)] - -0.6).abs() <= 1e-15);
    assert!(normal_residual.norm() < 1e-13);
    let zero_column = MatrixXd::from_column_major(2, 1, &[0.0, 0.0]);
    assert_eq!(HouseholderQr::new(&zero_column).h_coeffs()[0], 0.0);
}

#[test]
fn rust_only_col_piv_householder_qr_reconstructs_rank_and_solve() {
    let tall = tall();
    let qr = ColPivHouseholderQr::new(&tall);
    let reconstructed = &qr.householder_q() * &qr.matrix_r();
    let b = VectorXd::from_slice(&[1.0, 0.0, -1.0, 2.0, 0.5, 1.0]);
    let x = qr.solve(&b);
    let expected = HouseholderQr::new(&tall).solve(&b);

    // Rank-deficient: the third column is the sum of the first two.
    let mut deficient = tall.clone();
    deficient.set_col(2, &(&tall.col(0) + &tall.col(1)));
    assert!(reconstructed.is_approx(&(&tall * &qr.cols_permutation())));
    assert_eq!(qr.rank(), 3);
    assert!((&x - &expected).norm() < 1e-13);
    assert_eq!(ColPivHouseholderQr::new(&deficient).rank(), 2);
    assert_eq!(ColPivHouseholderQr::new(&MatrixXd::zeros(3, 3)).rank(), 0);
    assert_eq!(
        ColPivHouseholderQr::new(&MatrixXd::identity_rect(3, 5)).rank(),
        3
    );

    // The largest-norm column is pivoted first.
    let scaled = MatrixXd::from_row_major(2, 3, &[1.0, 0.0, 5.0, 0.0, 1.0, 0.0]);
    assert_eq!(
        ColPivHouseholderQr::new(&scaled).cols_permutation_indices()[0],
        2
    );
}
