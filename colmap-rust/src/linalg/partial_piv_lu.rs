//! PartialPivLu: LU factorization with partial (row) pivoting of a square matrix,
//! `P A = L U`, the replacement for `Eigen::PartialPivLU` / `matrix.partialPivLu()`. COLMAP
//! uses it to solve small square systems (affine and homography estimators, the 5-point
//! essential solver's 10x10 block, the camera-model Newton step in sensor/models.h) and,
//! through [`MatrixXd::determinant`] / [`MatrixXd::inverse`], for dynamic determinants and
//! inverses. Port of colmap-sharp's `ColmapSharp/LinearAlgebra/PartialPivLU.cs` (and
//! `MatrixXd.Determinant` / `Inverse`), written from Golub & Van Loan / colmap-sharp; Eigen
//! (MPL-2.0) is not ported.
//!
//! Algorithm: right-looking Gaussian elimination with partial pivoting, Golub & Van Loan,
//! "Matrix Computations", 4th ed., Algorithm 3.4.1 (the scheme of LAPACK's `dgetf2`). The
//! pivot is the first entry of largest magnitude in the column, the L multipliers are
//! computed by division, and a zero pivot is left in place (U is singular; like Eigen,
//! solving then yields inf/NaN rather than an error). Tier B: Eigen blocks the factorization
//! for large matrices, so results agree within rounding, not bit for bit.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_decomposition.rs` and
//! `rust_only_decomposition_oracle.rs`.

use super::{MatrixXd, VectorXd};

/// LU with partial pivoting, `P A = L U`. Replacement for `Eigen::PartialPivLU`.
#[derive(Clone, Debug)]
pub struct PartialPivLu {
    lu: MatrixXd,
    /// `permutation[i]` is the row of A that ended up as row i of `P A`.
    permutation: Vec<usize>,
    determinant_sign: i32,
}

impl PartialPivLu {
    /// Factorizes a square matrix. The input is not modified.
    ///
    /// # Panics
    /// When `a` is not square.
    pub fn new(a: &MatrixXd) -> Self {
        assert!(
            a.rows() == a.cols(),
            "PartialPivLu needs a square matrix, got {}x{}.",
            a.rows(),
            a.cols()
        );

        let n = a.rows();
        let mut lu = a.clone();
        let mut permutation = vec![0; n];
        let determinant_sign = factor_in_place(lu.as_mut_slice(), n, &mut permutation);
        Self {
            lu,
            permutation,
            determinant_sign,
        }
    }

    /// The packed factors (Eigen's `matrixLU()`): U on and above the diagonal, the strictly
    /// lower part of the unit lower-triangular L below it.
    pub fn matrix_lu(&self) -> &MatrixXd {
        &self.lu
    }

    /// The row permutation as indices: row i of `P A` is row `permutation_indices()[i]` of A.
    pub fn permutation_indices(&self) -> &[usize] {
        &self.permutation
    }

    /// The permutation matrix P with `P A = L U`.
    pub fn permutation_p(&self) -> MatrixXd {
        let n = self.permutation.len();
        let mut p = MatrixXd::zeros(n, n);
        for (i, &col) in self.permutation.iter().enumerate() {
            p[(i, col)] = 1.0;
        }

        p
    }

    /// The unit lower-triangular factor L.
    pub fn matrix_l(&self) -> MatrixXd {
        let n = self.lu.rows();
        let mut l = MatrixXd::identity(n);
        for c in 0..n {
            for r in c + 1..n {
                l[(r, c)] = self.lu[(r, c)];
            }
        }

        l
    }

    /// The upper-triangular factor U.
    pub fn matrix_u(&self) -> MatrixXd {
        let n = self.lu.rows();
        let mut u = MatrixXd::zeros(n, n);
        for c in 0..n {
            for r in 0..=c {
                u[(r, c)] = self.lu[(r, c)];
            }
        }

        u
    }

    /// `det(A)`: the permutation sign times the product of U's diagonal, left to right.
    pub fn determinant(&self) -> f64 {
        let mut det = f64::from(self.determinant_sign);
        for i in 0..self.lu.rows() {
            det *= self.lu[(i, i)];
        }

        det
    }

    /// Solves `A x = b`.
    ///
    /// # Panics
    /// When `b` has the wrong length.
    pub fn solve(&self, b: &VectorXd) -> VectorXd {
        let x = MatrixXd::from_column_major(b.len(), 1, b.as_slice());
        self.solve_matrix(&x).col(0)
    }

    /// Solves `A X = B` column by column.
    ///
    /// # Panics
    /// When `b` has the wrong number of rows.
    pub fn solve_matrix(&self, b: &MatrixXd) -> MatrixXd {
        let n = self.lu.rows();
        assert!(
            b.rows() == n,
            "Right-hand side has {} rows, expected {n}.",
            b.rows()
        );

        let mut x = MatrixXd::zeros(n, b.cols());
        for j in 0..b.cols() {
            solve_in_place(
                self.lu.as_slice(),
                n,
                &self.permutation,
                b.column(j),
                x.column_mut(j),
            );
        }

        x
    }

    /// `A^-1`, by solving against the identity.
    pub fn inverse(&self) -> MatrixXd {
        self.solve_matrix(&MatrixXd::identity(self.lu.rows()))
    }
}

impl MatrixXd {
    /// Determinant through [`PartialPivLu`], like Eigen does for a dynamic square matrix.
    ///
    /// # Panics
    /// When the matrix is not square.
    pub fn determinant(&self) -> f64 {
        PartialPivLu::new(self).determinant()
    }

    /// Inverse through [`PartialPivLu`] (no singularity check, like Eigen).
    ///
    /// # Panics
    /// When the matrix is not square.
    pub fn inverse(&self) -> MatrixXd {
        PartialPivLu::new(self).inverse()
    }
}

/// The factorization kernel over a column-major n x n slice, factorized in place into the
/// packed L\U form; `permutation` (length n) receives the row order. Returns the
/// permutation's sign. Allocation-free, for the minimal solvers.
pub(crate) fn factor_in_place(lu: &mut [f64], n: usize, permutation: &mut [usize]) -> i32 {
    factor_augmented_in_place(lu, n, 0, permutation)
}

/// [`factor_in_place`] on the n x (n + extra_cols) column-major slice `[A | B]`: A becomes
/// its packed L\U factors and B is carried through the same row swaps and elimination,
/// ending as `L^-1 P B`, the forward-substituted right-hand sides, ready for
/// [`back_substitute_trailing_rows`]. Each entry of B receives `b - l(i,0) y(0) -
/// l(i,1) y(1) - ...` in the same order as [`solve_in_place`]'s forward substitution, so the
/// results are bit-identical to factoring and then solving each column. Returns the
/// permutation's sign. Allocation-free.
pub(crate) fn factor_augmented_in_place(
    lu: &mut [f64],
    n: usize,
    extra_cols: usize,
    permutation: &mut [usize],
) -> i32 {
    for (i, p) in permutation.iter_mut().enumerate().take(n) {
        *p = i;
    }

    let total_cols = n + extra_cols;
    let mut sign = 1;
    for k in 0..n {
        let mut pivot = k;
        let mut best = lu[k * n + k].abs();
        for i in k + 1..n {
            let v = lu[k * n + i].abs();
            if v > best {
                best = v;
                pivot = i;
            }
        }

        if pivot != k {
            for c in 0..total_cols {
                lu.swap(c * n + k, c * n + pivot);
            }

            permutation.swap(k, pivot);
            sign = -sign;
        }

        let below = n - k - 1;
        let diagonal = lu[k * n + k];
        let first_updated = if diagonal == 0.0 {
            // U is singular; the column is left as it is, but the right-hand sides still get
            // the forward-substitution step (a solve would subtract l * y too).
            n
        } else {
            for value in &mut lu[k * n + k + 1..k * n + k + 1 + below] {
                *value /= diagonal;
            }

            k + 1
        };

        // The rank-1 update, column by column: each entry is a - l * u (no FMA).
        let (head, rest) = lu.split_at_mut((k + 1) * n);
        let multipliers = &head[k * n + k + 1..k * n + k + 1 + below];
        for c in first_updated..total_cols {
            let column = &mut rest[(c - k - 1) * n..(c - k) * n];
            let ukc = column[k];
            for (y, &l) in column[k + 1..].iter_mut().zip(multipliers) {
                *y -= l * ukc;
            }
        }
    }

    sign
}

/// The back-substitution half of [`solve_trailing_rows_in_place`]: with `y = L^-1 P b` in
/// `x`, finishes `x[first_row..n]` of `A x = b` (rows above `first_row` keep y). Back
/// substitution of row i reads only the rows below it.
pub(crate) fn back_substitute_trailing_rows(lu: &[f64], n: usize, x: &mut [f64], first_row: usize) {
    for i in (first_row..n).rev() {
        let mut sum = x[i];
        for k in i + 1..n {
            sum -= lu[k * n + i] * x[k];
        }

        x[i] = sum / lu[i * n + i];
    }
}

/// Solves `A x = b` for one right-hand side from the packed factors of [`factor_in_place`].
/// Allocation-free.
pub(crate) fn solve_in_place(
    lu: &[f64],
    n: usize,
    permutation: &[usize],
    b: &[f64],
    x: &mut [f64],
) {
    solve_trailing_rows_in_place(lu, n, permutation, b, x, 0);
}

/// [`solve_in_place`] when only `x[first_row..n]` is wanted: the forward substitution runs
/// in full, the back substitution stops at `first_row` (x above it is left holding
/// `L^-1 P b`). The wanted entries are bit-identical to a full solve. Allocation-free.
pub(crate) fn solve_trailing_rows_in_place(
    lu: &[f64],
    n: usize,
    permutation: &[usize],
    b: &[f64],
    x: &mut [f64],
    first_row: usize,
) {
    for i in 0..n {
        x[i] = b[permutation[i]];
    }

    // Forward substitution with the unit lower-triangular L, column-oriented: x[i] still
    // receives x[i] - l(i,0) x[0] - l(i,1) x[1] - ... in the same order as the row-oriented
    // sum, but walking L down its contiguous columns.
    for k in 0..n.saturating_sub(1) {
        let xk = x[k];
        let column = &lu[k * n..(k + 1) * n];
        for i in k + 1..n {
            x[i] -= column[i] * xk;
        }
    }

    // Back substitution with U.
    back_substitute_trailing_rows(lu, n, x, first_row);
}
