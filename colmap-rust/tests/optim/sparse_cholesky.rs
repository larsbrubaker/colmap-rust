// Port of COLMAP's `src/colmap/optim/sparse_cholesky_test.cc` (1:1, same names, values and
// tolerances), plus colmap-sharp's C#-only ComputeNewPatternAfterFallback as a `rust_only_*`
// test. Tests `colmap_rust::optim::SparseCholeskyWithFallbackSolver` (Tier B).
//
// IllConditionedChain draws its right-hand side from `random_eigen_vector_xd`, COLMAP's
// RandomEigenVectorXd; the test only checks the residual, so the values do not matter.

use colmap_rust::linalg::{SparseMatrixCsc, SparseTriplet, Vector3d, VectorXd};
use colmap_rust::math::random_eigen::random_eigen_vector_xd;
use colmap_rust::optim::SparseCholeskyWithFallbackSolver;

use crate::eigen_vector_xd_near;

/// 1D Laplacian of a chain of n nodes with the first diagonal entry increased by 1 to fix
/// the gauge.
fn chain_laplacian_gauge_fixed(n: usize) -> SparseMatrixCsc {
    let mut triplets = Vec::new();
    for i in 0..n {
        let mut diag = 0.0;
        if i > 0 {
            triplets.push(SparseTriplet::new(i, i - 1, -1.0));
            diag += 1.0;
        }
        if i < n - 1 {
            triplets.push(SparseTriplet::new(i, i + 1, -1.0));
            diag += 1.0;
        }
        triplets.push(SparseTriplet::new(i, i, diag));
    }
    // Gauge fix: pin node 0.
    triplets.push(SparseTriplet::new(0, 0, 1.0));
    SparseMatrixCsc::from_triplets(n, n, &triplets).unwrap()
}

/// Eigen's `VectorXd::LinSpaced(size, low, high)`.
fn lin_spaced(size: usize, low: f64, high: f64) -> VectorXd {
    VectorXd::from_vec(
        (0..size)
            .map(|i| {
                if size == 1 {
                    high
                } else {
                    low + (high - low) * i as f64 / (size - 1) as f64
                }
            })
            .collect(),
    )
}

fn any_nan(v: &VectorXd) -> bool {
    v.as_slice().iter().any(|x| x.is_nan())
}

fn diagonal(values: &[f64]) -> SparseMatrixCsc {
    let triplets: Vec<_> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| SparseTriplet::new(i, i, v))
        .collect();
    SparseMatrixCsc::from_triplets(values.len(), values.len(), &triplets).unwrap()
}

#[test]
fn sparse_cholesky_with_fallback_solver_compute_and_solve_diagonal() {
    let a = diagonal(&[2.0, 3.0, 4.0]);
    let mut solver = SparseCholeskyWithFallbackSolver::new();
    assert!(solver.compute(&a).unwrap());

    let b = VectorXd::from_slice(&[2.0, 6.0, 12.0]);
    let x = solver.solve(&b).expect("solve succeeds");
    assert!(eigen_vector_xd_near(
        &x,
        &Vector3d::new(1.0, 2.0, 3.0).into(),
        1e-12
    ));
}

#[test]
fn sparse_cholesky_with_fallback_solver_compute_and_solve_chain() {
    let a = chain_laplacian_gauge_fixed(10);
    let b = lin_spaced(a.rows(), 1.0, 10.0);
    let mut solver = SparseCholeskyWithFallbackSolver::new();
    assert!(solver.compute(&a).unwrap());

    let x = solver.solve(&b).expect("solve succeeds");
    assert!(eigen_vector_xd_near(&(&a * &x), &b, 1e-9));
}

#[test]
fn sparse_cholesky_with_fallback_solver_analyze_and_factorize_reused_across_matrices() {
    // Same sparsity, different numeric values — mirrors IRLS reuse pattern.
    let a1 = chain_laplacian_gauge_fixed(8);
    let mut a2 = a1.clone();
    for i in 0..a2.cols() {
        *a2.coeff_ref(i, i).unwrap() += 0.5;
    }

    let mut solver = SparseCholeskyWithFallbackSolver::new();
    solver.analyze_pattern(&a1).unwrap();
    let b = lin_spaced(a1.rows(), 1.0, 8.0);

    assert!(solver.factorize(&a1).unwrap());
    let x = solver.solve(&b).expect("solve succeeds");
    assert!(eigen_vector_xd_near(&(&a1 * &x), &b, 1e-9));

    assert!(solver.factorize(&a2).unwrap());
    let x = solver.solve(&b).expect("solve succeeds");
    assert!(eigen_vector_xd_near(&(&a2 * &x), &b, 1e-9));
}

#[test]
fn sparse_cholesky_with_fallback_solver_compute_returns_false_on_singular_matrix() {
    // 2x2 rank-1 matrix: [[1,1],[1,1]]. Singular, so both supernodal and LDLT must detect
    // the failure and Compute must return false rather than NaN.
    let a = SparseMatrixCsc::from_triplets(
        2,
        2,
        &[
            SparseTriplet::new(0, 0, 1.0),
            SparseTriplet::new(0, 1, 1.0),
            SparseTriplet::new(1, 0, 1.0),
            SparseTriplet::new(1, 1, 1.0),
        ],
    )
    .unwrap();
    let mut solver = SparseCholeskyWithFallbackSolver::new();
    assert!(!solver.compute(&a).unwrap());
}

#[test]
fn sparse_cholesky_with_fallback_solver_ridge_makes_singular_matrix_solvable() {
    // Same singular matrix as above with a small ridge added to the diagonal is PD and must
    // factorize successfully.
    let a = SparseMatrixCsc::from_triplets(
        2,
        2,
        &[
            SparseTriplet::new(0, 0, 1.0 + 1e-6),
            SparseTriplet::new(0, 1, 1.0),
            SparseTriplet::new(1, 0, 1.0),
            SparseTriplet::new(1, 1, 1.0 + 1e-6),
        ],
    )
    .unwrap();
    let mut solver = SparseCholeskyWithFallbackSolver::new();
    assert!(solver.compute(&a).unwrap());

    let b = VectorXd::from_slice(&[2.0, 2.0]);
    let x = solver.solve(&b).expect("solve succeeds");
    assert!(!any_nan(&x));
}

#[test]
fn sparse_cholesky_with_fallback_solver_falls_back_to_ldlt_on_indefinite_matrix() {
    // diag(1, 1, -1e-20) is mathematically indefinite. Supernodal LLT fundamentally requires
    // strictly positive pivots (it takes square roots), so CHOLMOD reliably rejects this
    // across versions. SimplicialLDLT accepts indefinite matrices by allowing negative entries
    // in D. The wrapper must fall back transparently and Solve must return the correct (exact
    // for a diagonal system) solution rather than NaN.
    let a = diagonal(&[1.0, 1.0, -1e-20]);
    let mut solver = SparseCholeskyWithFallbackSolver::new();
    assert!(solver.compute(&a).unwrap());

    let b = VectorXd::from_slice(&[2.0, 3.0, -5e-20]);
    let x = solver.solve(&b).expect("solve succeeds");
    assert!(eigen_vector_xd_near(
        &x,
        &Vector3d::new(2.0, 3.0, 5.0).into(),
        1e-10
    ));
}

#[test]
fn sparse_cholesky_with_fallback_solver_ill_conditioned_chain() {
    // Long chain Laplacian + gauge fix. Mathematically PD but condition number grows as
    // O(n^2). Exercises the regime that motivated the fallback.
    let a = chain_laplacian_gauge_fixed(500);
    let b = random_eigen_vector_xd(a.rows());
    let mut solver = SparseCholeskyWithFallbackSolver::new();
    assert!(solver.compute(&a).unwrap());

    let x = solver.solve(&b).expect("solve succeeds");
    assert!(!any_nan(&x));
    assert!(eigen_vector_xd_near(&(&a * &x), &b, 1e-6));
}

// colmap-sharp's C#-only test (no counterpart in sparse_cholesky_test.cc): once the solver
// has fallen back to LDLT, a compute on a matrix with a different pattern must re-analyze the
// LDLT stage too, not factorize the new values against the old pattern
// (see the translation notes in `optim/sparse_cholesky.rs`).
#[test]
fn rust_only_compute_new_pattern_after_fallback() {
    let indefinite = diagonal(&[1.0, 1.0, -1e-20]);
    let chain = chain_laplacian_gauge_fixed(6);
    let b = lin_spaced(chain.rows(), 1.0, 6.0);

    let mut solver = SparseCholeskyWithFallbackSolver::new();
    assert!(solver.compute(&indefinite).unwrap());
    assert!(solver.uses_ldlt());

    assert!(solver.compute(&chain).unwrap());
    let x = solver.solve(&b).expect("solve succeeds");
    assert!(eigen_vector_xd_near(&(&chain * &x), &b, 1e-9));
}
