//! `MatrixXd`: a heap-backed, dynamically sized matrix of doubles, the replacement for
//! `Eigen::MatrixXd` and the partly dynamic shapes COLMAP's estimators build
//! (`Eigen::Matrix<double, Dynamic, 9>` for the 8-point and homography DLT systems,
//! `Dynamic x 6` for affine, `12 x 12` for DLT pose).
//!
//! Port of colmap-sharp's `ColmapSharp/LinearAlgebra/MatrixXd.cs` (MIT), written there to
//! Eigen's documented semantics; Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md).
//! Neighbors: [`super::VectorXd`] (`vector_x.rs`), the arithmetic and fixed-size conversions
//! (`matrix_x_ops.rs`), and the decompositions (`partial_piv_lu.rs` adds `determinant` and
//! `inverse`, like Eigen evaluating them through `PartialPivLU` for a dynamic matrix).
//!
//! Storage: one column-major `Vec<f64>`, Eigen's default memory order, so a COLMAP walk over
//! `matrix.data()` maps to the same flat index here. `column(j)` / `column_mut(j)` are
//! contiguous slice views for hot loops; `block`/`row`/`col` return copies and
//! `set_block`/`set_row`/`set_col` write back. Shape mismatches are programming errors and
//! panic, like Eigen's asserts. Unlike Eigen, a new matrix is zero-filled.
//!
//! Arithmetic order: the module contract in `linalg/mod.rs`. Eigen evaluates dynamic products
//! with a blocked, vectorized kernel, so they are Tier B against COLMAP
//! (docs/CPP_DIVERGENCES.md, entry 20).
//! Tests: `colmap-rust/tests/linalg/rust_only_dynamic_matrix.rs`.

use super::{dot, is_approx_slices, maxi, VectorXd, DUMMY_PRECISION};
use crate::math::fns;

/// Dynamically sized, column-major matrix of doubles, Eigen's `MatrixXd`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MatrixXd {
    rows: usize,
    cols: usize,
    // Column-major: element (row, col) is at col * rows + row.
    data: Vec<f64>,
}

impl MatrixXd {
    /// A zero matrix, Eigen's `MatrixXd::Zero(rows, cols)`.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    /// Eigen's `MatrixXd::Identity(n, n)`.
    pub fn identity(n: usize) -> Self {
        Self::identity_rect(n, n)
    }

    /// Eigen's `MatrixXd::Identity(rows, cols)`: ones on the main diagonal.
    pub fn identity_rect(rows: usize, cols: usize) -> Self {
        let mut m = Self::zeros(rows, cols);
        for i in 0..rows.min(cols) {
            m.data[i * rows + i] = 1.0;
        }
        m
    }

    /// Eigen's `MatrixXd::Constant(rows, cols, value)`.
    pub fn constant(rows: usize, cols: usize, value: f64) -> Self {
        Self {
            rows,
            cols,
            data: vec![value; rows * cols],
        }
    }

    /// A matrix from values in column-major order (Eigen's memory layout).
    pub fn from_column_major(rows: usize, cols: usize, values: &[f64]) -> Self {
        assert_eq!(
            values.len(),
            rows * cols,
            "Expected {} values.",
            rows * cols
        );
        Self {
            rows,
            cols,
            data: values.to_vec(),
        }
    }

    /// A matrix that takes ownership of column-major `values`.
    pub fn from_column_major_vec(rows: usize, cols: usize, values: Vec<f64>) -> Self {
        assert_eq!(
            values.len(),
            rows * cols,
            "Expected {} values.",
            rows * cols
        );
        Self {
            rows,
            cols,
            data: values,
        }
    }

    /// A matrix from values in row-major reading order, like Eigen's comma initializer.
    pub fn from_row_major(rows: usize, cols: usize, values: &[f64]) -> Self {
        assert_eq!(
            values.len(),
            rows * cols,
            "Expected {} values.",
            rows * cols
        );
        let mut m = Self::zeros(rows, cols);
        for r in 0..rows {
            for c in 0..cols {
                m.data[c * rows + r] = values[r * cols + c];
            }
        }
        m
    }

    /// A diagonal matrix, Eigen's `v.asDiagonal()`.
    pub fn from_diagonal(diagonal: &VectorXd) -> Self {
        let n = diagonal.len();
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = diagonal[i];
        }
        m
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of coefficients, `rows * cols` (Eigen's `size()`).
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the matrix has no coefficients.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The column-major coefficients, Eigen's `data()`.
    pub fn as_slice(&self) -> &[f64] {
        &self.data
    }

    /// The column-major coefficients, writable.
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Column `j` as a contiguous view.
    pub fn column(&self, j: usize) -> &[f64] {
        assert!(j < self.cols, "column {j} out of range");
        &self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// Column `j` as a writable contiguous view.
    pub fn column_mut(&mut self, j: usize) -> &mut [f64] {
        assert!(j < self.cols, "column {j} out of range");
        &mut self.data[j * self.rows..(j + 1) * self.rows]
    }

    /// A copy of column `j`.
    pub fn col(&self, j: usize) -> VectorXd {
        VectorXd::from_slice(self.column(j))
    }

    /// A copy of row `i`.
    pub fn row(&self, i: usize) -> VectorXd {
        assert!(i < self.rows, "row {i} out of range");
        VectorXd::from_vec(
            (0..self.cols)
                .map(|c| self.data[c * self.rows + i])
                .collect(),
        )
    }

    /// Overwrites column `j`.
    pub fn set_col(&mut self, j: usize, values: &VectorXd) {
        require_same_len(values.len(), self.rows);
        self.column_mut(j).copy_from_slice(values.as_slice());
    }

    /// Overwrites row `i`.
    pub fn set_row(&mut self, i: usize, values: &VectorXd) {
        assert!(i < self.rows, "row {i} out of range");
        require_same_len(values.len(), self.cols);
        for c in 0..self.cols {
            self.data[c * self.rows + i] = values[c];
        }
    }

    /// A copy of the `rows x cols` block starting at `(row, col)`, Eigen's `block()`.
    pub fn block(&self, row: usize, col: usize, rows: usize, cols: usize) -> Self {
        self.check_block(row, col, rows, cols);
        let mut b = Self::zeros(rows, cols);
        for c in 0..cols {
            let src = (col + c) * self.rows + row;
            b.data[c * rows..(c + 1) * rows].copy_from_slice(&self.data[src..src + rows]);
        }
        b
    }

    /// Writes `block` into this matrix with its top-left corner at `(row, col)`.
    pub fn set_block(&mut self, row: usize, col: usize, block: &Self) {
        self.check_block(row, col, block.rows, block.cols);
        for c in 0..block.cols {
            let dst = (col + c) * self.rows + row;
            self.data[dst..dst + block.rows].copy_from_slice(block.column(c));
        }
    }

    /// A copy of the first `n` rows, Eigen's `topRows(n)`.
    pub fn top_rows(&self, n: usize) -> Self {
        self.block(0, 0, n, self.cols)
    }

    /// A copy of the last `n` rows, Eigen's `bottomRows(n)`.
    pub fn bottom_rows(&self, n: usize) -> Self {
        self.block(self.rows - n, 0, n, self.cols)
    }

    /// A copy of the first `n` columns, Eigen's `leftCols(n)`.
    pub fn left_cols(&self, n: usize) -> Self {
        self.block(0, 0, self.rows, n)
    }

    /// A copy of the last `n` columns, Eigen's `rightCols(n)`.
    pub fn right_cols(&self, n: usize) -> Self {
        self.block(0, self.cols - n, self.rows, n)
    }

    /// The transpose, as a new matrix.
    pub fn transpose(&self) -> Self {
        let mut t = Self::zeros(self.cols, self.rows);
        for c in 0..self.cols {
            for r in 0..self.rows {
                t.data[r * self.cols + c] = self.data[c * self.rows + r];
            }
        }
        t
    }

    /// Rows in reverse order, Eigen's `colwise().reverse()`.
    pub fn reverse_rows(&self) -> Self {
        let mut m = Self::zeros(self.rows, self.cols);
        for c in 0..self.cols {
            for r in 0..self.rows {
                m.data[c * self.rows + r] = self.data[c * self.rows + (self.rows - 1 - r)];
            }
        }
        m
    }

    /// Columns in reverse order, Eigen's `rowwise().reverse()`.
    pub fn reverse_cols(&self) -> Self {
        let mut m = Self::zeros(self.rows, self.cols);
        for c in 0..self.cols {
            m.column_mut(c)
                .copy_from_slice(self.column(self.cols - 1 - c));
        }
        m
    }

    /// Sum of the diagonal, left to right (0 when empty).
    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        if n == 0 {
            return 0.0;
        }
        let mut sum = self.data[0];
        for i in 1..n {
            sum += self.data[i * self.rows + i];
        }
        sum
    }

    /// Squared Frobenius norm, a left-to-right sum over the column-major buffer.
    pub fn squared_norm(&self) -> f64 {
        dot(&self.data, &self.data)
    }

    /// Frobenius norm, Eigen's `norm()` on a matrix.
    pub fn norm(&self) -> f64 {
        fns::sqrt(self.squared_norm())
    }

    /// Eigen's `isUpperTriangular`: every coefficient strictly below the diagonal is within
    /// `precision` of 0 relative to the largest coefficient on or above it.
    pub fn is_upper_triangular(&self, precision: f64) -> bool {
        let mut max_abs_on_upper_part: f64 = 0.0;
        for c in 0..self.cols {
            if self.rows == 0 {
                break;
            }
            for r in 0..=c.min(self.rows - 1) {
                max_abs_on_upper_part =
                    maxi(max_abs_on_upper_part, self.data[c * self.rows + r].abs());
            }
        }
        let threshold = max_abs_on_upper_part * precision;
        for c in 0..self.cols {
            for r in c + 1..self.rows {
                if self.data[c * self.rows + r].abs() > threshold {
                    return false;
                }
            }
        }
        true
    }

    /// Eigen's `isUnitary`: the columns are orthonormal within `precision` (every column dot
    /// product is approximately 0 or 1).
    pub fn is_unitary(&self, precision: f64) -> bool {
        for i in 0..self.cols {
            let ci = self.column(i);
            if (dot(ci, ci) - 1.0).abs() > precision {
                return false;
            }
            for j in 0..i {
                if dot(ci, self.column(j)).abs() > precision {
                    return false;
                }
            }
        }
        true
    }

    /// Eigen's `isApprox` at [`DUMMY_PRECISION`].
    pub fn is_approx(&self, other: &Self) -> bool {
        self.is_approx_with(other, DUMMY_PRECISION)
    }

    /// Eigen's `isApprox`, `||a - b|| <= precision * min(||a||, ||b||)` with Frobenius norms;
    /// false when the shapes differ.
    pub fn is_approx_with(&self, other: &Self, precision: f64) -> bool {
        self.rows == other.rows
            && self.cols == other.cols
            && is_approx_slices(&self.data, &other.data, precision)
    }

    fn check_block(&self, row: usize, col: usize, rows: usize, cols: usize) {
        assert!(
            row + rows <= self.rows && col + cols <= self.cols,
            "Block ({row}, {col}, {rows}x{cols}) is outside a {}x{} matrix.",
            self.rows,
            self.cols
        );
    }

    fn check_index(&self, row: usize, col: usize) {
        assert!(
            row < self.rows && col < self.cols,
            "MatrixXd index ({row}, {col}) outside {}x{}",
            self.rows,
            self.cols
        );
    }
}

/// Panics with colmap-sharp's message when two dimensions that must agree differ.
pub(crate) fn require_same_len(a: usize, b: usize) {
    assert_eq!(a, b, "Dimension mismatch: {a} vs {b}.");
}

/// Coefficient at `(row, col)`; panics when either is out of range.
impl std::ops::Index<(usize, usize)> for MatrixXd {
    type Output = f64;
    fn index(&self, (row, col): (usize, usize)) -> &f64 {
        self.check_index(row, col);
        &self.data[col * self.rows + row]
    }
}

impl std::ops::IndexMut<(usize, usize)> for MatrixXd {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut f64 {
        self.check_index(row, col);
        &mut self.data[col * self.rows + row]
    }
}

impl std::fmt::Display for MatrixXd {
    /// `[[a, b], [c, d]]` in row order, each value with Rust's round-trip formatting.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        for r in 0..self.rows {
            write!(f, "{}", if r == 0 { "[" } else { ", [" })?;
            for c in 0..self.cols {
                if c > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:?}", self.data[c * self.rows + r])?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")
    }
}
