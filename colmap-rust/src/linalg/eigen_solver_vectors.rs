//! EigenSolver eigenvectors from the real Schur form `T = Z^T A Z` (`eigen_solver.rs`). Port of
//! colmap-sharp's `ColmapSharp/LinearAlgebra/EigenSolver.Vectors.cs`; Eigen (MPL-2.0) is not
//! ported.
//!
//! For each eigenvalue lambda at diagonal block k, the eigenvector y of T is 1 (real) or the
//! 2x2 block's own eigenvector (complex pair) at the block, zero below, and above it the
//! back-substitution of `(T - lambda I) y = 0` block by block (1x1 blocks divide, 2x2 blocks
//! solve a 2x2 complex system by Cramer's rule): Golub & Van Loan, "Matrix Computations",
//! 4th ed., §7.6.4 (eigenvectors of a quasi-triangular matrix). The eigenvector of A is
//! `Z y`, normalized to unit norm. A zero pivot (a repeated eigenvalue) is replaced by
//! `epsilon * ||T||`, the perturbation of EISPACK's hqr2 (Smith et al., "Matrix Eigensystem
//! Routines - EISPACK Guide", Lecture Notes in Computer Science 6, Springer, 1976; only this
//! published idea is used, no code).
//!
//! Complex-eigenvector phase: ours is fixed by the back-substitution (the block's eigenvector
//! starts as `(b, lambda - a)`, then the vector is scaled to unit norm without rotating its
//! phase), which need not be Eigen's (docs/CPP_DIVERGENCES.md). Complex arithmetic follows
//! .NET's `System.Numerics.Complex` operators (`complex.rs`), so the bits match colmap-sharp.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_spectral_eigen.rs` and
//! `rust_only_spectral_oracle.rs`.

use super::complex::{Complex, ComplexMatrixXd};
use super::{MatrixXd, MACHINE_EPSILON};
use crate::math::fns;

/// The unit-norm eigenvectors of `A = Z T Z^T` for the Schur-order `eigenvalues` of `t`
/// (scaled `t`, unscaled eigenvalues of it).
pub(super) fn compute_eigenvectors(
    t: &MatrixXd,
    z: &MatrixXd,
    eigenvalues: &[Complex],
) -> ComplexMatrixXd {
    let n = t.rows();
    let mut vectors = ComplexMatrixXd::zeros(n, n);
    // t is scaled to max |entry| = 1, so its norm is 0 only for the zero matrix.
    let t_norm = t.norm();
    let small_pivot = MACHINE_EPSILON * (if t_norm > 0.0 { t_norm } else { 1.0 });
    let mut y = vec![Complex::ZERO; n];
    let mut column = vec![Complex::ZERO; n];
    let mut k = 0;
    while k < n {
        let lambda = eigenvalues[k];
        let complex_block = k < n - 1 && t[(k + 1, k)] != 0.0;
        y.fill(Complex::ZERO);
        if complex_block {
            // (B - lambda I) u = 0 for the block B = [a b; c d]: u = (b, lambda - a).
            y[k] = Complex::from_real(t[(k, k + 1)]);
            y[k + 1] = lambda - t[(k, k)];
        } else {
            y[k] = Complex::from_real(1.0);
        }
        let top = k as isize - 1;

        let end = if complex_block { k + 1 } else { k };
        back_substitute(t, lambda, &mut y, top, end, small_pivot);
        let mut squared_norm = 0.0;
        for (r, entry) in column.iter_mut().enumerate() {
            let mut sum = Complex::ZERO;
            for (c, &yc) in y.iter().enumerate().take(end + 1) {
                sum += z[(r, c)] * yc;
            }
            *entry = sum;
            squared_norm += sum.re * sum.re + sum.im * sum.im;
        }

        let norm = fns::sqrt(squared_norm);
        for (r, &entry) in column.iter().enumerate() {
            vectors[(r, k)] = if norm > 0.0 { entry / norm } else { entry };
        }

        if complex_block {
            for r in 0..n {
                vectors[(r, k + 1)] = vectors[(r, k)].conj();
            }
            k += 1;
        }
        k += 1;
    }
    vectors
}

/// Solves rows `top..=0` of `(T - lambda I) y = 0` given `y[top+1..=end]`, block by block.
fn back_substitute(
    t: &MatrixXd,
    lambda: Complex,
    y: &mut [Complex],
    top: isize,
    end: usize,
    small_pivot: f64,
) {
    let mut i = top;
    while i >= 0 {
        let iu = i as usize;
        let block = iu > 0 && t[(iu, iu - 1)] != 0.0;
        if !block {
            let rhs = -row_sum(t, y, iu, iu + 1, end);
            let mut pivot = t[(iu, iu)] - lambda;
            if pivot == Complex::ZERO {
                pivot = Complex::from_real(small_pivot);
            }
            y[iu] = rhs / pivot;
            i -= 1;
        } else {
            let r0 = iu - 1;
            let rhs0 = -row_sum(t, y, r0, iu + 1, end);
            let rhs1 = -row_sum(t, y, iu, iu + 1, end);
            let a = t[(r0, r0)] - lambda;
            let b = Complex::from_real(t[(r0, iu)]);
            let c = Complex::from_real(t[(iu, r0)]);
            let d = t[(iu, iu)] - lambda;
            let mut det = a * d - b * c;
            if det == Complex::ZERO {
                det = Complex::from_real(small_pivot);
            }
            y[r0] = (rhs0 * d - b * rhs1) / det;
            y[iu] = (a * rhs1 - c * rhs0) / det;
            i -= 2;
        }
    }
}

/// `sum_{j = from..=to} t(row, j) * y[j]`, left to right from zero.
fn row_sum(t: &MatrixXd, y: &[Complex], row: usize, from: usize, to: usize) -> Complex {
    let mut sum = Complex::ZERO;
    for (j, &yj) in y.iter().enumerate().take(to + 1).skip(from) {
        sum += t[(row, j)] * yj;
    }
    sum
}
