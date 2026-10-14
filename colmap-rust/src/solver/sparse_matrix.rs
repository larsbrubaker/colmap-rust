// The Jacobian as the trust-region minimizer sees it: a linear operator it can scale by
// columns, multiply with and take column norms of. Port of colmap-sharp's
// `ColmapSharp/Solver/SparseMatrix.cs` (our MIT code) and of its `ILinearOperator`
// (`IterativeSchurSolver.cs`), which follow Ceres Solver 2.2.0
// `internal/ceres/linear_operator.h`, `sparse_matrix.h` and `dense_sparse_matrix.cc`
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//
// [`DenseSparseMatrix`] (a dense column-major matrix behind the [`SparseMatrix`] interface, for
// DENSE_QR and DENSE_NORMAL_CHOLESKY) lives here; `block_sparse_matrix.rs` is the
// block-structured one the sparse and Schur solvers use. Every product is a left-to-right sum
// in a fixed order, so results do not depend on the thread count.

use crate::linalg::MatrixXd;

/// `ceres::internal::LinearOperator` (the part the conjugate gradients solver needs): an
/// operator it can apply without forming it.
pub trait LinearOperator {
    /// `y += A x`.
    fn right_multiply_and_accumulate(&self, x: &[f64], y: &mut [f64]);
}

/// `ceres::internal::SparseMatrix`: the minimizer's view of a Jacobian.
pub trait SparseMatrix: LinearOperator {
    /// Number of rows (residuals).
    fn num_rows(&self) -> usize;

    /// Number of columns (tangent parameters).
    fn num_cols(&self) -> usize;

    /// Sets every stored value to zero, keeping the structure.
    fn set_zero(&mut self);

    /// `y += A' x`.
    fn left_multiply_and_accumulate(&self, x: &[f64], y: &mut [f64]);

    /// `x[j]` = squared norm of column j.
    fn squared_column_norm(&self, x: &mut [f64]);

    /// `A = A * diag(scale)`.
    fn scale_columns(&mut self, scale: &[f64]);
}

/// `ceres::internal::DenseSparseMatrix`: a dense Jacobian, stored column-major.
#[derive(Clone, Debug)]
pub struct DenseSparseMatrix {
    matrix: MatrixXd,
}

impl DenseSparseMatrix {
    /// A zero `rows x cols` matrix.
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            matrix: MatrixXd::zeros(rows, cols),
        }
    }

    /// The column-major storage.
    pub fn matrix(&self) -> &MatrixXd {
        &self.matrix
    }

    /// The column-major storage, writable (the evaluator writes the Jacobian here).
    pub fn matrix_mut(&mut self) -> &mut MatrixXd {
        &mut self.matrix
    }
}

impl LinearOperator for DenseSparseMatrix {
    fn right_multiply_and_accumulate(&self, x: &[f64], y: &mut [f64]) {
        let rows = self.matrix.rows();
        let y = &mut y[..rows];
        for (c, &xc) in x[..self.matrix.cols()].iter().enumerate() {
            for (yr, &v) in y.iter_mut().zip(self.matrix.column(c)) {
                *yr += v * xc;
            }
        }
    }
}

impl SparseMatrix for DenseSparseMatrix {
    fn num_rows(&self) -> usize {
        self.matrix.rows()
    }

    fn num_cols(&self) -> usize {
        self.matrix.cols()
    }

    fn set_zero(&mut self) {
        self.matrix.as_mut_slice().fill(0.0);
    }

    fn left_multiply_and_accumulate(&self, x: &[f64], y: &mut [f64]) {
        let x = &x[..self.matrix.rows()];
        for (c, yc) in y[..self.matrix.cols()].iter_mut().enumerate() {
            let mut sum = 0.0;
            for (&v, &xr) in self.matrix.column(c).iter().zip(x) {
                sum += v * xr;
            }

            *yc += sum;
        }
    }

    fn squared_column_norm(&self, x: &mut [f64]) {
        for (c, xc) in x[..self.matrix.cols()].iter_mut().enumerate() {
            let mut sum = 0.0;
            for &v in self.matrix.column(c) {
                sum += v * v;
            }

            *xc = sum;
        }
    }

    fn scale_columns(&mut self, scale: &[f64]) {
        for (c, &s) in scale[..self.matrix.cols()].iter().enumerate() {
            for v in self.matrix.column_mut(c) {
                *v *= s;
            }
        }
    }
}
