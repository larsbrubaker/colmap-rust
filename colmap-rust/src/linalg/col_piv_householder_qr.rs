//! ColPivHouseholderQr: rank-revealing QR with column pivoting, `A P = Q R`, the replacement
//! for `Eigen::ColPivHouseholderQR` / `matrix.colPivHouseholderQr()`. COLMAP uses `rank()`
//! (the P3P/EPnP degeneracy check in estimators/solvers/absolute_pose.cc, the fixed-point
//! gauge check in bundle_adjustment_ceres.cc) and `solve()` (absolute_pose.cc's 4-vector
//! least-squares step). Port of colmap-sharp's
//! `ColmapSharp/LinearAlgebra/ColPivHouseholderQR.cs`, written from Golub & Van Loan /
//! colmap-sharp; Eigen (MPL-2.0) is not ported.
//!
//! Algorithm: Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm 5.4.1 (at step j
//! the remaining column of largest norm is swapped in), with the reflector convention in
//! [`super::householder`]. The partial column norms are downdated after each step and
//! recomputed when cancellation makes the downdate unreliable, following Drmač and
//! Bujanović, LAPACK Working Note 176 (2008), the scheme LAPACK's `dlaqp2` documents. Ties
//! pick the first column of largest norm.
//!
//! Rank: Eigen's documented default. A pivot `|R(i,i)|` counts as nonzero when it is strictly
//! greater than `threshold * max_pivot`, with `threshold = min(rows, cols) * epsilon` and
//! `max_pivot` the largest `|R(i,i)|`. `solve` returns the basic solution: the components
//! past the rank are zero, then undone through the permutation. Tier B.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_decomposition.rs` and
//! `rust_only_decomposition_oracle.rs`.

use super::householder;
use super::{dot, MatrixXd, VectorXd, MACHINE_EPSILON};
use crate::math::fns;

/// Householder QR with column pivoting, `A P = Q R`. Replacement for
/// `Eigen::ColPivHouseholderQR<MatrixXd>`.
#[derive(Clone, Debug)]
pub struct ColPivHouseholderQr {
    qr: MatrixXd,
    h_coeffs: Vec<f64>,
    /// Column j of `A P` is column `permutation[j]` of A.
    permutation: Vec<usize>,
    max_pivot: f64,
}

/// C#'s `Math.Max(double, double)`: NaN when either argument is NaN.
fn math_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a < b {
        b
    } else {
        a
    }
}

fn column_norm(column: &[f64]) -> f64 {
    fns::sqrt(dot(column, column))
}

impl ColPivHouseholderQr {
    /// Factorizes a matrix of any shape. The input is not modified.
    pub fn new(a: &MatrixXd) -> Self {
        let mut qr = a.clone();
        let m = a.rows();
        let n = a.cols();
        let k = m.min(n);
        let mut h_coeffs = vec![0.0; k];
        let mut permutation: Vec<usize> = (0..n).collect();
        let mut max_pivot = 0.0;

        // partial_norms: current norm of column c's rows j..m-1; reference_norms: the norm at
        // the last full recomputation (LAWN 176's vn1 / vn2).
        let mut partial_norms: Vec<f64> = (0..n).map(|c| column_norm(qr.column(c))).collect();
        let mut reference_norms = partial_norms.clone();

        let tolerance = fns::sqrt(MACHINE_EPSILON);
        let data = qr.as_mut_slice();
        for j in 0..k {
            let mut pivot = j;
            for c in j + 1..n {
                if partial_norms[c] > partial_norms[pivot] {
                    pivot = c;
                }
            }

            if pivot != j {
                for r in 0..m {
                    data.swap(j * m + r, pivot * m + r);
                }

                permutation.swap(j, pivot);
                partial_norms.swap(j, pivot);
                reference_norms.swap(j, pivot);
            }

            let (head, rest) = data.split_at_mut((j + 1) * m);
            let column = &mut head[j * m + j..];
            let tau = householder::make_in_place(column);
            h_coeffs[j] = tau;
            max_pivot = math_max(max_pivot, column[0].abs());
            let essential = &column[1..];
            for c in j + 1..n {
                let target = &mut rest[(c - j - 1) * m..(c - j) * m];
                householder::apply_left(essential, tau, &mut target[j..]);

                // Downdate the norm of rows j+1.. by removing R(j, c).
                if partial_norms[c] != 0.0 {
                    let ratio = target[j].abs() / partial_norms[c];
                    let temp = math_max(0.0, 1.0 - ratio * ratio);
                    let norm_ratio = partial_norms[c] / reference_norms[c];
                    if temp * norm_ratio * norm_ratio <= tolerance {
                        partial_norms[c] = column_norm(&target[j + 1..]);
                        reference_norms[c] = partial_norms[c];
                    } else {
                        partial_norms[c] *= fns::sqrt(temp);
                    }
                }
            }
        }

        Self {
            qr,
            h_coeffs,
            permutation,
            max_pivot,
        }
    }

    /// The packed factor (Eigen's `matrixQR()`).
    pub fn matrix_qr(&self) -> &MatrixXd {
        &self.qr
    }

    /// The reflector coefficients tau (Eigen's `hCoeffs()`).
    pub fn h_coeffs(&self) -> VectorXd {
        VectorXd::from_slice(&self.h_coeffs)
    }

    /// The upper-triangular (trapezoidal) factor R, m x n.
    pub fn matrix_r(&self) -> MatrixXd {
        householder::upper_part(&self.qr)
    }

    /// The full orthogonal factor Q (m x m).
    pub fn householder_q(&self) -> MatrixXd {
        householder::accumulate_q(&self.qr, &self.h_coeffs)
    }

    /// Column j of `A P` is column `cols_permutation_indices()[j]` of A.
    pub fn cols_permutation_indices(&self) -> &[usize] {
        &self.permutation
    }

    /// The permutation matrix P with `A P = Q R` (Eigen's `colsPermutation()`).
    pub fn cols_permutation(&self) -> MatrixXd {
        let n = self.permutation.len();
        let mut p = MatrixXd::zeros(n, n);
        for (j, &row) in self.permutation.iter().enumerate() {
            p[(row, j)] = 1.0;
        }

        p
    }

    /// Numerical rank: the number of `|R(i,i)|` strictly greater than
    /// `min(rows, cols) * epsilon * max |R(i,i)|`.
    pub fn rank(&self) -> usize {
        let threshold = self.h_coeffs.len() as f64 * MACHINE_EPSILON * self.max_pivot;
        (0..self.h_coeffs.len())
            .filter(|&i| self.qr[(i, i)].abs() > threshold)
            .count()
    }

    /// Basic least-squares solution of `A x = b`: solves the leading rank x rank triangle of R
    /// and sets the remaining components to zero.
    ///
    /// # Panics
    /// When `b` has the wrong length.
    pub fn solve(&self, b: &VectorXd) -> VectorXd {
        let m = self.qr.rows();
        assert!(
            b.len() == m,
            "Right-hand side has {} rows, expected {m}.",
            b.len()
        );

        let rank = self.rank();
        let mut y = b.clone();
        householder::apply_q_transpose(self.qr.as_slice(), m, &self.h_coeffs, y.as_mut_slice());
        householder::back_substitute(self.qr.as_slice(), m, rank, y.as_mut_slice());
        let mut x = VectorXd::zeros(self.qr.cols());
        for i in 0..rank {
            x[self.permutation[i]] = y[i];
        }

        x
    }
}
