// Small dense row-major kernels and PSD block inversion for the Schur solvers. Port of
// colmap-sharp's `ColmapSharp/Solver/SmallBlas.cs` (our MIT code), which follows Ceres Solver
// 2.2.0 `internal/ceres/small_blas.h` (the "Naive" loop kernels) and
// `internal/ceres/invert_psd_matrix.h` (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// The products dispatch the bundle adjustment shapes (2 or 3 inner rows, inner dimension 3) to
// `small_blas_fixed.rs`, whose const-generic kernels unroll the k loop but keep its order, so
// every shape rounds exactly as the loops here do.
//
// Every matrix is row-major. Each output element is a sum over k in increasing order into a
// temporary that starts at 0.0, then stored with the [`Operation`] (=, +=, -=), which is how
// Ceres' naive kernels round. Ceres switches to Eigen's kernels when all sizes are
// compile-time constants (its template specializations such as <2, 3, 6>, which the Schur
// eliminator picks from the Jacobian's static block sizes), whose summation order is Eigen's.
//
// That is a deliberate divergence: every shape here rounds as the naive loops do, while
// Eigen's fixed-size products may sum in another order (vectorized pairs), so results agree
// with Ceres' to rounding, not bit for bit. Eigen is replaced rather than ported (CLAUDE.md
// contract 2), and one summation order for every shape makes the Schur solvers' results
// independent of which kernel a shape dispatches to, which the later determinism tests rely on.
// Bundle adjustment is Tier C. Evidence: `tests/solver/rust_only_small_blas.rs` (Ceres'
// `small_blas_test.cc` via colmap-sharp: every product, operation, stride and start offset
// against MatrixXd within 5 * epsilon) and `tests/solver/rust_only_small_blas_fixed.rs` (the
// fixed kernels give the naive loops' bits for every operation, with negative zeros and
// strided, offset output blocks; reversing the fixed kernels' k order makes it fail). Same as
// colmap-sharp entry 35 (item 2, the kernel part).
//
// `invert_upper_psd` is Ceres' InvertPSDMatrix with assume_full_rank = true for dynamic sizes:
// selfadjointView<Upper>().llt().solve(Identity), a Cholesky factorization of the upper
// triangle (Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm 4.2.2, as in
// `linalg/llt.rs`; Eigen is not ported) and two triangular solves per column of the identity.
// A matrix that is not positive definite yields NaN, a deliberate divergence: Eigen's LLT
// stops there and the solve returns unspecified values, which cannot be reproduced without
// porting Eigen and are garbage either way. `invert_upper_psd` (and `_in_place`, which
// `BlockRandomAccessDiagonalMatrix::invert` uses for the SCHUR_JACOBI preconditioner, and which
// the Schur eliminator will use for E'E) fills the whole inverse with NaN as soon as a Cholesky
// pivot is not positive or is NaN. NaN makes the linear solve, and so the trust-region step,
// invalid; the minimizer then rejects the step and shrinks the radius, which is what Ceres'
// minimizer does when the garbage happens to be non-finite. Evidence:
// `tests/solver/rust_only_sparse_matrix.rs`
// `rust_only_invert_upper_psd_matches_llt_and_rejects_indefinite` (an indefinite 2x2 block
// gives all NaN; an SPD block with garbage below the diagonal matches `linalg::Llt` within
// 1e-12, in place bit-identical to out of place); the ported
// `BlockRandomAccessDiagonalMatrixTest.Invert` checks the positive definite case at 1e-14.
// Same as colmap-sharp entry 35 (item 5).
//
// Users: `block_sparse_matrix.rs` and the block random access matrices; later the Schur
// eliminator and the implicit Schur complement. No kernel allocates.

use crate::math::fns;

use super::small_blas_fixed;

/// How a kernel stores each computed element into C: Ceres' `kOperation` template argument
/// (`1` is +=, `-1` is -=, `0` is =).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    /// `C = product` (Ceres' 0).
    Assign,
    /// `C += product` (Ceres' 1).
    Add,
    /// `C -= product` (Ceres' -1).
    Subtract,
}

impl Operation {
    /// Stores `tmp` into `c` with this operation.
    #[inline(always)]
    pub fn store(self, c: &mut f64, tmp: f64) {
        match self {
            Operation::Add => *c += tmp,
            Operation::Subtract => *c -= tmp,
            Operation::Assign => *c = tmp,
        }
    }
}

/// `C(start_row_c.., start_col_c..) op= A' B` for A (`rows_a x cols_a`) and B
/// (`rows_a x cols_b`); C has row stride `col_stride_c`. Port of Ceres'
/// `MatrixTransposeMatrixMultiply` with dynamic sizes.
#[allow(clippy::too_many_arguments)]
pub fn matrix_transpose_matrix_multiply(
    a: &[f64],
    rows_a: usize,
    cols_a: usize,
    b: &[f64],
    cols_b: usize,
    c: &mut [f64],
    start_row_c: usize,
    start_col_c: usize,
    col_stride_c: usize,
    operation: Operation,
) {
    let index0 = start_row_c * col_stride_c + start_col_c;
    // The bundle adjustment shapes: residual blocks of 2 rows (E'E, E'F, F'F) and the 3-row
    // E'F blocks times (E'E)^-1.
    match rows_a {
        2 => small_blas_fixed::matrix_transpose_matrix_multiply_rows::<2>(
            a,
            cols_a,
            b,
            cols_b,
            c,
            index0,
            col_stride_c,
            operation,
        ),
        3 => small_blas_fixed::matrix_transpose_matrix_multiply_rows::<3>(
            a,
            cols_a,
            b,
            cols_b,
            c,
            index0,
            col_stride_c,
            operation,
        ),
        _ => matrix_transpose_matrix_multiply_naive(
            a,
            rows_a,
            cols_a,
            b,
            cols_b,
            c,
            start_row_c,
            start_col_c,
            col_stride_c,
            operation,
        ),
    }
}

/// The loop kernel behind [`matrix_transpose_matrix_multiply`] for every shape; the
/// fixed-size kernels reproduce it bit for bit. Port of Ceres'
/// `MatrixTransposeMatrixMultiplyNaive`.
#[allow(clippy::too_many_arguments)]
pub fn matrix_transpose_matrix_multiply_naive(
    a: &[f64],
    rows_a: usize,
    cols_a: usize,
    b: &[f64],
    cols_b: usize,
    c: &mut [f64],
    start_row_c: usize,
    start_col_c: usize,
    col_stride_c: usize,
    operation: Operation,
) {
    for row in 0..cols_a {
        let index = (row + start_row_c) * col_stride_c + start_col_c;
        for col in 0..cols_b {
            let mut tmp = 0.0;
            for k in 0..rows_a {
                tmp += a[k * cols_a + row] * b[k * cols_b + col];
            }

            operation.store(&mut c[index + col], tmp);
        }
    }
}

/// `C(start_row_c.., start_col_c..) op= A B` for A (`rows_a x cols_a`) and B
/// (`cols_a x cols_b`). Port of Ceres' `MatrixMatrixMultiply` with dynamic sizes.
#[allow(clippy::too_many_arguments)]
pub fn matrix_matrix_multiply(
    a: &[f64],
    rows_a: usize,
    cols_a: usize,
    b: &[f64],
    cols_b: usize,
    c: &mut [f64],
    start_row_c: usize,
    start_col_c: usize,
    col_stride_c: usize,
    operation: Operation,
) {
    // b_i' (E'E)^-1 times an E'F block with a 3-dimensional E block (points).
    if cols_a == 3 {
        small_blas_fixed::matrix_matrix_multiply_inner::<3>(
            a,
            rows_a,
            b,
            cols_b,
            c,
            start_row_c * col_stride_c + start_col_c,
            col_stride_c,
            operation,
        );
        return;
    }

    matrix_matrix_multiply_naive(
        a,
        rows_a,
        cols_a,
        b,
        cols_b,
        c,
        start_row_c,
        start_col_c,
        col_stride_c,
        operation,
    );
}

/// The loop kernel behind [`matrix_matrix_multiply`] for every shape. Port of Ceres'
/// `MatrixMatrixMultiplyNaive`.
#[allow(clippy::too_many_arguments)]
pub fn matrix_matrix_multiply_naive(
    a: &[f64],
    rows_a: usize,
    cols_a: usize,
    b: &[f64],
    cols_b: usize,
    c: &mut [f64],
    start_row_c: usize,
    start_col_c: usize,
    col_stride_c: usize,
    operation: Operation,
) {
    for row in 0..rows_a {
        let index = (row + start_row_c) * col_stride_c + start_col_c;
        for col in 0..cols_b {
            let mut tmp = 0.0;
            for k in 0..cols_a {
                tmp += a[row * cols_a + k] * b[k * cols_b + col];
            }

            operation.store(&mut c[index + col], tmp);
        }
    }
}

/// `c op= A b` for A (`rows_a x cols_a`). Port of Ceres' `MatrixVectorMultiply`.
pub fn matrix_vector_multiply(
    a: &[f64],
    rows_a: usize,
    cols_a: usize,
    b: &[f64],
    c: &mut [f64],
    operation: Operation,
) {
    for row in 0..rows_a {
        let mut tmp = 0.0;
        for col in 0..cols_a {
            tmp += a[row * cols_a + col] * b[col];
        }

        operation.store(&mut c[row], tmp);
    }
}

/// `c op= A' b` for A (`rows_a x cols_a`). Port of Ceres' `MatrixTransposeVectorMultiply`.
pub fn matrix_transpose_vector_multiply(
    a: &[f64],
    rows_a: usize,
    cols_a: usize,
    b: &[f64],
    c: &mut [f64],
    operation: Operation,
) {
    for col in 0..cols_a {
        let mut tmp = 0.0;
        for row in 0..rows_a {
            tmp += a[row * cols_a + col] * b[row];
        }

        operation.store(&mut c[col], tmp);
    }
}

/// `inverse = M^-1` for the symmetric positive definite `n x n` row-major M, reading only its
/// upper triangle. `scratch` needs `n * n` values. A matrix that is not positive definite
/// yields NaN in every entry (Eigen's LLT leaves unspecified values there; NaN makes the step
/// invalid, which is what the minimizer does with such garbage). Port of Ceres'
/// `InvertPSDMatrix<Dynamic>` with `assume_full_rank = true`.
pub fn invert_upper_psd(m: &[f64], n: usize, inverse: &mut [f64], scratch: &mut [f64]) {
    let l = &mut scratch[..n * n];
    if cholesky_of_upper(m, n, l) {
        solve_identity(l, n, inverse);
    } else {
        inverse[..n * n].fill(f64::NAN);
    }
}

/// [`invert_upper_psd`] in place: `m` is replaced by its inverse. (colmap-sharp passes the
/// same span as both matrices; the factorization reads all of M before the solves write.)
pub fn invert_upper_psd_in_place(m: &mut [f64], n: usize, scratch: &mut [f64]) {
    let l = &mut scratch[..n * n];
    if cholesky_of_upper(m, n, l) {
        solve_identity(l, n, m);
    } else {
        m[..n * n].fill(f64::NAN);
    }
}

// L L' = A, L stored row-major in `l`: L(i, j) = l[i * n + j], i >= j, with A(i, j) = M(j, i)
// for i >= j (the lower triangle of the symmetric matrix the upper triangle describes).
// False when a pivot is not positive (or NaN).
fn cholesky_of_upper(m: &[f64], n: usize, l: &mut [f64]) -> bool {
    for j in 0..n {
        for i in j..n {
            let mut v = m[j * n + i];
            for k in 0..j {
                v -= l[i * n + k] * l[j * n + k];
            }

            l[i * n + j] = v;
        }

        let pivot = l[j * n + j];
        // colmap-sharp's `!(pivot > 0)`: a NaN pivot also fails.
        if pivot.is_nan() || pivot <= 0.0 {
            return false;
        }

        let root = fns::sqrt(pivot);
        l[j * n + j] = root;
        for i in j + 1..n {
            l[i * n + j] /= root;
        }
    }

    true
}

// Column c of the inverse solves L L' x = e_c: forward substitution with L, then back
// substitution with L', both in place in column c of `inverse`.
fn solve_identity(l: &[f64], n: usize, inverse: &mut [f64]) {
    for c in 0..n {
        for i in 0..n {
            let mut v = if i == c { 1.0 } else { 0.0 };
            for k in 0..i {
                v -= l[i * n + k] * inverse[k * n + c];
            }

            inverse[i * n + c] = v / l[i * n + i];
        }

        for i in (0..n).rev() {
            let mut v = inverse[i * n + c];
            for k in i + 1..n {
                v -= l[k * n + i] * inverse[k * n + c];
            }

            inverse[i * n + c] = v / l[i * n + i];
        }
    }
}
