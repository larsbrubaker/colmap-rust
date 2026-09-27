//! Llt: Cholesky factorization `A = L L^T` of a symmetric positive-definite matrix, the
//! replacement for `Eigen::LLT` / `matrix.llt()` (COLMAP: cost_functions/utils.h takes
//! `cov.inverse().llt().matrixL()` as a square-root information matrix). Sibling of
//! [`Ldlt`](super::Ldlt) (the pivoted, semidefinite-tolerant variant). Port of colmap-sharp's
//! `ColmapSharp/LinearAlgebra/LLT.cs` (whose `ComputationInfo` lives in `linalg/mod.rs`
//! here), written from Golub & Van Loan / colmap-sharp; Eigen (MPL-2.0) is not ported.
//!
//! Algorithm: column-oriented ("gaxpy") Cholesky, Golub & Van Loan, "Matrix Computations",
//! 4th ed., Algorithm 4.2.2. Like Eigen's default (Lower) LLT, only the lower triangle of
//! the input is read. A non-positive pivot stops the factorization and reports
//! `NumericalIssue`, Eigen's documented `info()` contract. Tier B.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_decomposition.rs` and
//! `rust_only_decomposition_oracle.rs`.

use super::{ComputationInfo, MatrixXd, VectorXd};
use crate::math::fns;

/// Cholesky factorization `A = L L^T`. Replacement for `Eigen::LLT<MatrixXd>`.
#[derive(Clone, Debug)]
pub struct Llt {
    l: MatrixXd,
    info: ComputationInfo,
}

impl Llt {
    /// Factorizes a symmetric positive-definite matrix, reading its lower triangle.
    ///
    /// # Panics
    /// When `a` is not square.
    pub fn new(a: &MatrixXd) -> Self {
        assert!(
            a.rows() == a.cols(),
            "Llt needs a square matrix, got {}x{}.",
            a.rows(),
            a.cols()
        );

        let n = a.rows();
        let mut l_matrix = MatrixXd::zeros(n, n);
        let l = l_matrix.as_mut_slice();
        let src = a.as_slice();
        let mut info = ComputationInfo::Success;
        for j in 0..n {
            // v = A(j:n, j) - L(j:n, 0:j) * L(j, 0:j)^T
            for i in j..n {
                let mut v = src[j * n + i];
                for k in 0..j {
                    v -= l[k * n + i] * l[k * n + j];
                }

                l[j * n + i] = v;
            }

            let pivot = l[j * n + j];
            // colmap-sharp's `!(pivot > 0)`: a NaN pivot also stops.
            if pivot.is_nan() || pivot <= 0.0 {
                info = ComputationInfo::NumericalIssue;
                break;
            }

            let root = fns::sqrt(pivot);
            l[j * n + j] = root;
            for value in &mut l[j * n + j + 1..(j + 1) * n] {
                *value /= root;
            }
        }

        Self { l: l_matrix, info }
    }

    /// `Success`, or `NumericalIssue` when the matrix is not positive definite.
    pub fn info(&self) -> ComputationInfo {
        self.info
    }

    /// The lower-triangular factor L.
    pub fn matrix_l(&self) -> MatrixXd {
        self.l.clone()
    }

    /// The upper-triangular factor `U = L^T`.
    pub fn matrix_u(&self) -> MatrixXd {
        self.l.transpose()
    }

    /// Solves `A x = b` by forward then backward substitution.
    ///
    /// # Panics
    /// When `b` has the wrong length.
    pub fn solve(&self, b: &VectorXd) -> VectorXd {
        let n = self.l.rows();
        assert!(
            b.len() == n,
            "Right-hand side has {} rows, expected {n}.",
            b.len()
        );

        let mut x = b.clone();
        solve_in_place(self.l.as_slice(), n, x.as_mut_slice());
        x
    }

    /// Solves `A X = B` column by column.
    ///
    /// # Panics
    /// When `b` has the wrong number of rows.
    pub fn solve_matrix(&self, b: &MatrixXd) -> MatrixXd {
        let n = self.l.rows();
        assert!(
            b.rows() == n,
            "Right-hand side has {} rows, expected {n}.",
            b.rows()
        );

        let mut x = b.clone();
        for j in 0..x.cols() {
            solve_in_place(self.l.as_slice(), n, x.column_mut(j));
        }

        x
    }
}

fn solve_in_place(l: &[f64], n: usize, x: &mut [f64]) {
    for i in 0..n {
        let mut sum = x[i];
        for k in 0..i {
            sum -= l[k * n + i] * x[k];
        }

        x[i] = sum / l[i * n + i];
    }

    for i in (0..n).rev() {
        let mut sum = x[i];
        for k in i + 1..n {
            sum -= l[i * n + k] * x[k];
        }

        x[i] = sum / l[i * n + i];
    }
}
