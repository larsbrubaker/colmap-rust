//! Ldlt: pivoted "robust Cholesky" `A = P^T L D L^T P` of a symmetric positive or negative
//! semidefinite matrix, the replacement for `Eigen::LDLT` (COLMAP: optim/tiny_solver.h solves
//! its Levenberg-Marquardt normal equations with it and checks `info()`). Sibling of
//! [`Llt`](super::Llt). Port of colmap-sharp's `ColmapSharp/LinearAlgebra/LDLT.cs`, written
//! from Golub & Van Loan / colmap-sharp; Eigen (MPL-2.0) is not ported.
//!
//! Algorithm: outer-product LDL^T with symmetric (diagonal) pivoting: at step k the
//! remaining diagonal entry of largest magnitude is swapped to position k (rows and columns
//! together), then the trailing block is updated with the rank-1 term. Golub & Van Loan,
//! "Matrix Computations", 4th ed., §4.2.9 applied to the LDL^T form of §4.1.2; Higham,
//! "Accuracy and Stability of Numerical Algorithms", 2nd ed., §10.3. Eigen documents the same
//! factorization shape (`P^T L D L^* P` with a unit lower L), so the factors correspond;
//! Tier B.
//!
//! Only the lower triangle of the input is read, like Eigen's default (Lower). Choices of
//! colmap-sharp's, not taken from Eigen's code: the first diagonal entry of largest magnitude
//! wins a tie; a zero pivot whose column below is also zero (the semidefinite case) leaves
//! its L column zero and `solve` maps that component to 0; `info` is `NumericalIssue` when a
//! pivot is NaN or infinite, or when a zero pivot has a nonzero entry below it (no diagonally
//! pivoted LDL^T exists, e.g. `[[0, 1], [1, 0]]`; the factors are then not usable).
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_decomposition.rs` and
//! `rust_only_decomposition_oracle.rs`.

use super::{ComputationInfo, MatrixXd, VectorXd};

/// Pivoted LDL^T factorization `A = P^T L D L^T P`. Replacement for `Eigen::LDLT<MatrixXd>`.
#[derive(Clone, Debug)]
pub struct Ldlt {
    /// Packed: D on the diagonal, the strictly lower part of the unit lower L below it.
    ldl: MatrixXd,
    /// `transpositions[k]` is the index swapped with k at step k (Eigen's `transpositionsP()`).
    transpositions: Vec<usize>,
    info: ComputationInfo,
}

fn swap_symmetric(m: &mut [f64], n: usize, a: usize, b: usize) {
    for c in 0..n {
        m.swap(c * n + a, c * n + b);
    }

    for r in 0..n {
        m.swap(a * n + r, b * n + r);
    }
}

impl Ldlt {
    /// Factorizes a symmetric matrix, reading its lower triangle.
    ///
    /// # Panics
    /// When `a` is not square.
    pub fn new(a: &MatrixXd) -> Self {
        assert!(
            a.rows() == a.cols(),
            "Ldlt needs a square matrix, got {}x{}.",
            a.rows(),
            a.cols()
        );

        let n = a.rows();

        // Work on a full symmetric copy built from the lower triangle; the trailing block is
        // kept symmetric so the row-and-column swaps stay consistent.
        let mut ldl = MatrixXd::zeros(n, n);
        for c in 0..n {
            for r in c..n {
                ldl[(r, c)] = a[(r, c)];
                ldl[(c, r)] = a[(r, c)];
            }
        }

        let mut transpositions = vec![0; n];
        let mut info = ComputationInfo::Success;
        let m = ldl.as_mut_slice();
        for k in 0..n {
            let mut pivot = k;
            let mut best = m[k * n + k].abs();
            for i in k + 1..n {
                let v = m[i * n + i].abs();
                if v > best {
                    best = v;
                    pivot = i;
                }
            }

            transpositions[k] = pivot;
            if pivot != k {
                swap_symmetric(m, n, k, pivot);
            }

            let d = m[k * n + k];
            if !d.is_finite() {
                info = ComputationInfo::NumericalIssue;
            }

            if d == 0.0 {
                // A zero pivot is only consistent when its whole column below is zero too (a
                // semidefinite matrix); then that column of L is zero. Otherwise, e.g.
                // [[0, 1], [1, 0]], no diagonally pivoted LDL^T exists: report it so callers
                // like tiny_solver reject the step instead of using a wrong solve.
                for value in &mut m[k * n + k + 1..(k + 1) * n] {
                    if *value != 0.0 {
                        info = ComputationInfo::NumericalIssue;
                    }

                    *value = 0.0;
                }

                continue;
            }

            // l(i, k) = a(i, k) / d, then A(k+1:, k+1:) -= l * d * l^T (lower part).
            for value in &mut m[k * n + k + 1..(k + 1) * n] {
                *value /= d;
            }

            for c in k + 1..n {
                let dlc = m[k * n + c] * d;
                for r in c..n {
                    // Mirror into the upper part so later symmetric swaps see a symmetric
                    // trailing block.
                    let updated = m[c * n + r] - m[k * n + r] * dlc;
                    m[c * n + r] = updated;
                    m[r * n + c] = updated;
                }
            }
        }

        // Clear the strictly upper part so the packed factor is unambiguous.
        for c in 1..n {
            for r in 0..c {
                m[c * n + r] = 0.0;
            }
        }

        Self {
            ldl,
            transpositions,
            info,
        }
    }

    /// `Success`, or `NumericalIssue` when a pivot was NaN or infinite, or zero with a nonzero
    /// entry below it.
    pub fn info(&self) -> ComputationInfo {
        self.info
    }

    /// The diagonal of D, Eigen's `vectorD()`.
    pub fn vector_d(&self) -> VectorXd {
        let n = self.ldl.rows();
        VectorXd::from_vec((0..n).map(|i| self.ldl[(i, i)]).collect())
    }

    /// The unit lower-triangular factor L.
    pub fn matrix_l(&self) -> MatrixXd {
        let mut l = self.ldl.clone();
        for i in 0..l.rows() {
            l[(i, i)] = 1.0;
        }

        l
    }

    /// The transpositions, Eigen's `transpositionsP()`: step k swapped k with
    /// `transpositions()[k]`.
    pub fn transpositions(&self) -> &[usize] {
        &self.transpositions
    }

    /// The permutation matrix P with `A = P^T L D L^T P`.
    pub fn permutation_p(&self) -> MatrixXd {
        let mut p = MatrixXd::identity(self.ldl.rows());
        for (k, &t) in self.transpositions.iter().enumerate() {
            for c in 0..p.cols() {
                let swap = p[(k, c)];
                p[(k, c)] = p[(t, c)];
                p[(t, c)] = swap;
            }
        }

        p
    }

    /// True when every entry of D is `>= 0` (Eigen's `isPositive()`).
    pub fn is_positive(&self) -> bool {
        !(0..self.ldl.rows()).any(|i| self.ldl[(i, i)] < 0.0)
    }

    /// True when every entry of D is `<= 0` (Eigen's `isNegative()`).
    pub fn is_negative(&self) -> bool {
        !(0..self.ldl.rows()).any(|i| self.ldl[(i, i)] > 0.0)
    }

    /// Solves `A x = b`.
    ///
    /// # Panics
    /// When `b` has the wrong length.
    pub fn solve(&self, b: &VectorXd) -> VectorXd {
        let n = self.ldl.rows();
        assert!(
            b.len() == n,
            "Right-hand side has {} rows, expected {n}.",
            b.len()
        );

        let mut x = b.clone();
        self.solve_in_place(x.as_mut_slice());
        x
    }

    /// Solves `A X = B` column by column.
    ///
    /// # Panics
    /// When `b` has the wrong number of rows.
    pub fn solve_matrix(&self, b: &MatrixXd) -> MatrixXd {
        let n = self.ldl.rows();
        assert!(
            b.rows() == n,
            "Right-hand side has {} rows, expected {n}.",
            b.rows()
        );

        let mut x = b.clone();
        for j in 0..x.cols() {
            self.solve_in_place(x.column_mut(j));
        }

        x
    }

    fn solve_in_place(&self, x: &mut [f64]) {
        let n = self.ldl.rows();
        let m = self.ldl.as_slice();
        for k in 0..n {
            x.swap(k, self.transpositions[k]);
        }

        for i in 1..n {
            let mut sum = x[i];
            for k in 0..i {
                sum -= m[k * n + i] * x[k];
            }

            x[i] = sum;
        }

        for i in 0..n {
            let d = m[i * n + i];
            x[i] = if d == 0.0 { 0.0 } else { x[i] / d };
        }

        for i in (0..n).rev() {
            let mut sum = x[i];
            for k in i + 1..n {
                sum -= m[i * n + k] * x[k];
            }

            x[i] = sum;
        }

        for k in (0..n).rev() {
            x.swap(k, self.transpositions[k]);
        }
    }
}
