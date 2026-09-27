//! FullPivLu: LU with complete pivoting, `P A Q = L U`, the replacement for
//! `Eigen::FullPivLU`. COLMAP calls `rank()` (estimators/solvers/similarity_transform.h
//! rejects 3 x N point sets of rank below the sample count); Ceres' line-search polynomial
//! fit calls `setThreshold(0).solve(b)`, which is [`FullPivLu::solve`] here. Port of
//! colmap-sharp's `ColmapSharp/LinearAlgebra/FullPivLU.cs`, written from Golub & Van Loan /
//! colmap-sharp; Eigen (MPL-2.0) is not ported.
//!
//! Algorithm: Gaussian elimination with complete pivoting, Golub & Van Loan, "Matrix
//! Computations", 4th ed., Algorithm 3.4.3: at step k the entry of largest magnitude in the
//! remaining submatrix is swapped to (k, k) (ties: the first in column-major order).
//! Elimination stops once the remaining submatrix is exactly zero.
//!
//! Solve (G&VL §3.4.4): `c = P b`, forward substitution with the unit lower factor, back
//! substitution with the leading r x r block of U over the r pivots above the threshold, and
//! `x = Q [y; 0]`, so the coordinates past the rank are zero (Eigen's documented behavior).
//!
//! Rank: Eigen's documented default threshold. A pivot counts as nonzero when its magnitude
//! is strictly greater than `min(rows, cols) * epsilon * |largest pivot|` (the first one).
//! Tier B.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_decomposition.rs`.

use super::{MatrixXd, VectorXd, MACHINE_EPSILON};

/// LU decomposition with complete pivoting; exposes the numerical rank. Replacement for
/// `Eigen::FullPivLU`.
#[derive(Clone, Debug)]
pub struct FullPivLu {
    pivots: Vec<f64>,
    lu: MatrixXd,
    row_transpositions: Vec<usize>,
    col_transpositions: Vec<usize>,
    diagonal_size: usize,
}

fn swap_rows(m: &mut MatrixXd, a: usize, b: usize) {
    if a == b {
        return;
    }

    for c in 0..m.cols() {
        let t = m[(a, c)];
        m[(a, c)] = m[(b, c)];
        m[(b, c)] = t;
    }
}

fn swap_cols(m: &mut MatrixXd, a: usize, b: usize) {
    if a == b {
        return;
    }

    for r in 0..m.rows() {
        let t = m[(r, a)];
        m[(r, a)] = m[(r, b)];
        m[(r, b)] = t;
    }
}

impl FullPivLu {
    /// Factorizes a matrix of any shape. The input is not modified.
    pub fn new(a: &MatrixXd) -> Self {
        let mut lu = a.clone();
        let m = lu.rows();
        let n = lu.cols();
        let size = m.min(n);
        let mut pivots = Vec::with_capacity(size);
        let mut row_transpositions = vec![0; size];
        let mut col_transpositions = vec![0; size];
        for k in 0..size {
            let mut pivot_row = k;
            let mut pivot_col = k;
            let mut largest = 0.0;
            for c in k..n {
                for r in k..m {
                    let magnitude = lu[(r, c)].abs();
                    if magnitude > largest {
                        largest = magnitude;
                        pivot_row = r;
                        pivot_col = c;
                    }
                }
            }

            if largest == 0.0 {
                break;
            }

            row_transpositions[k] = pivot_row;
            col_transpositions[k] = pivot_col;
            swap_rows(&mut lu, k, pivot_row);
            swap_cols(&mut lu, k, pivot_col);
            let pivot = lu[(k, k)];
            pivots.push(pivot);
            for r in k + 1..m {
                let factor = lu[(r, k)] / pivot;
                lu[(r, k)] = factor;
                for c in k + 1..n {
                    let update = factor * lu[(k, c)];
                    lu[(r, c)] -= update;
                }
            }
        }

        for k in pivots.len()..size {
            row_transpositions[k] = k;
            col_transpositions[k] = k;
        }

        Self {
            pivots,
            lu,
            row_transpositions,
            col_transpositions,
            diagonal_size: size,
        }
    }

    /// `min(rows, cols)`.
    pub fn diagonal_size(&self) -> usize {
        self.diagonal_size
    }

    /// Numerical rank: the number of pivots with magnitude strictly greater than
    /// `min(rows, cols) * epsilon * |largest pivot|`.
    pub fn rank(&self) -> usize {
        if self.pivots.is_empty() {
            return 0;
        }

        let threshold = self.diagonal_size as f64 * MACHINE_EPSILON * self.pivots[0].abs();
        self.pivots.iter().filter(|p| p.abs() > threshold).count()
    }

    /// Solves `A x = b` (least squares is not attempted: rows past the rank are ignored),
    /// counting a pivot as nonzero when its magnitude is strictly greater than `threshold`
    /// times the largest pivot's; the unknowns beyond that rank are zero. Eigen's
    /// `setThreshold(threshold).solve(b)`.
    ///
    /// # Panics
    /// When `b` has the wrong length.
    pub fn solve(&self, b: &VectorXd, threshold: f64) -> VectorXd {
        let m = self.lu.rows();
        let n = self.lu.cols();
        assert_eq!(b.len(), m, "Right-hand side has the wrong length.");
        let limit = if self.pivots.is_empty() {
            0.0
        } else {
            threshold * self.pivots[0].abs()
        };
        let rank = self.pivots.iter().filter(|p| p.abs() > limit).count();

        let mut x = VectorXd::zeros(n);
        if rank == 0 {
            return x;
        }

        // c = P b.
        let mut c = b.as_slice().to_vec();
        for k in 0..self.diagonal_size {
            c.swap(k, self.row_transpositions[k]);
        }

        // L y = c over the leading diagonal_size rows (unit diagonal).
        for i in 0..self.diagonal_size {
            let mut sum = c[i];
            for (j, &cj) in c[..i].iter().enumerate() {
                sum -= self.lu[(i, j)] * cj;
            }

            c[i] = sum;
        }

        // U z = y over the leading rank x rank block.
        for i in (0..rank).rev() {
            let mut sum = c[i];
            for (j, &cj) in c.iter().enumerate().take(rank).skip(i + 1) {
                sum -= self.lu[(i, j)] * cj;
            }

            c[i] = sum / self.lu[(i, i)];
        }

        // x = Q [z; 0]: undo the column transpositions in reverse.
        let y = x.as_mut_slice();
        y[..rank].copy_from_slice(&c[..rank]);
        for k in (0..self.diagonal_size).rev() {
            y.swap(k, self.col_transpositions[k]);
        }

        x
    }
}
