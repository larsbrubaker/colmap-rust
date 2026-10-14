//! SparseMatrixCsc: a compressed sparse column matrix, the replacement for
//! `Eigen::SparseMatrix<double>` (column-major, the Eigen default COLMAP uses everywhere:
//! optim/sparse_cholesky, optim/least_absolute_deviations, rotation averaging, and later the
//! bundle-adjustment normal equations). The sparse Cholesky
//! ([`SimplicialCholesky`](super::SimplicialCholesky), with
//! [`amd_ordering`](super::amd_ordering)) factors these. Port of colmap-sharp's
//! `ColmapSharp/LinearAlgebra/SparseMatrixCsc.cs` (MIT, our own code).
//!
//! Storage is the textbook CSC layout (Davis, "Direct Methods for Sparse Linear Systems",
//! SIAM 2006, ch. 2, used as a description only; no CSparse code is ported, it is LGPL):
//! `col_ptr` has `cols + 1` offsets, and column j's entries are `row_indices/values[col_ptr[j]
//! .. col_ptr[j+1]]`. Every matrix built here keeps row indices strictly increasing within a
//! column, with no duplicates, which `coeff`'s binary search and the Cholesky's pattern check
//! rely on. Explicitly stored zeros stay stored, as they do in Eigen when a caller inserts a 0
//! (least_absolute_deviations_test.cc does), so they are part of the pattern.
//!
//! Unlike Eigen there is no random-access insert: [`SparseMatrixCsc::coeff_ref`] only reaches
//! stored entries, because every COLMAP call site updates an existing pattern (the diagonal of
//! A^T A). [`SparseMatrixCsc::add_to_diagonal`] covers the one case that may need new entries.
//!
//! Errors: building from caller data (`from_triplets`, `from_csc`, `coeff_ref`, `block`)
//! returns `Err`; shape mismatches in the arithmetic operators panic, as the dense
//! [`MatrixXd`] operators do. Tests: `colmap-rust/tests/linalg/rust_only_sparse_matrix.rs`.

use super::{MatrixXd, VectorXd};
use crate::util::check::{ColmapError, ErrorKind};
use crate::Result;

/// One (row, col, value) entry used to build a sparse matrix, `Eigen::Triplet<double>`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SparseTriplet {
    /// Row index.
    pub row: usize,
    /// Column index.
    pub col: usize,
    /// Value; duplicates of one (row, col) are summed.
    pub value: f64,
}

impl SparseTriplet {
    /// A triplet `(row, col, value)`.
    pub const fn new(row: usize, col: usize, value: f64) -> Self {
        Self { row, col, value }
    }
}

/// Which triangle of a symmetric matrix is stored or read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymmetricPart {
    /// Entries with row >= col (`Eigen::Lower`, the default of Eigen's sparse Cholesky).
    Lower,
    /// Entries with row <= col (`Eigen::Upper`).
    Upper,
}

impl SymmetricPart {
    /// Whether entry (i, j) lies in this triangle (diagonal included).
    #[inline]
    pub fn contains(self, i: usize, j: usize) -> bool {
        match self {
            SymmetricPart::Lower => i >= j,
            SymmetricPart::Upper => i <= j,
        }
    }
}

/// Compressed sparse column matrix of doubles. Replacement for `Eigen::SparseMatrix<double>`.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseMatrixCsc {
    rows: usize,
    cols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
}

fn invalid(message: String) -> ColmapError {
    ColmapError::new(ErrorKind::InvalidArgument, message)
}

impl SparseMatrixCsc {
    /// An all-zero `rows x cols` matrix with no stored entries.
    pub fn zero(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            col_ptr: vec![0; cols + 1],
            row_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// The `n x n` identity.
    pub fn identity(n: usize) -> Self {
        Self {
            rows: n,
            cols: n,
            col_ptr: (0..=n).collect(),
            row_idx: (0..n).collect(),
            values: vec![1.0; n],
        }
    }

    /// Builds a matrix from triplets, summing duplicates (Eigen's `setFromTriplets`).
    /// Duplicates are summed in the order they appear in `triplets`. A triplet outside the
    /// matrix is an error.
    pub fn from_triplets(rows: usize, cols: usize, triplets: &[SparseTriplet]) -> Result<Self> {
        // Counting sort by column, then by row within each column. Both passes are stable,
        // so duplicates keep their input order and are summed in it.
        let mut count = vec![0usize; cols + 1];
        for t in triplets {
            if t.row >= rows || t.col >= cols {
                return Err(invalid(format!(
                    "Triplet ({}, {}) is outside a {rows}x{cols} matrix.",
                    t.row, t.col
                )));
            }
            count[t.col + 1] += 1;
        }
        for j in 0..cols {
            count[j + 1] += count[j];
        }

        let mut by_col = vec![SparseTriplet::new(0, 0, 0.0); triplets.len()];
        let mut next = count.clone();
        for t in triplets {
            by_col[next[t.col]] = *t;
            next[t.col] += 1;
        }

        let mut col_ptr = vec![0usize; cols + 1];
        let mut row_idx = Vec::with_capacity(triplets.len());
        let mut values = Vec::with_capacity(triplets.len());
        // usize::MAX marks "no slot yet"; a slot below `start` belongs to an earlier column.
        let mut last_slot = vec![usize::MAX; rows];
        for j in 0..cols {
            let start = row_idx.len();
            for t in &by_col[count[j]..count[j + 1]] {
                let slot = last_slot[t.row];
                if slot != usize::MAX && slot >= start {
                    values[slot] += t.value;
                } else {
                    last_slot[t.row] = row_idx.len();
                    row_idx.push(t.row);
                    values.push(t.value);
                }
            }
            sort_column(&mut row_idx[start..], &mut values[start..]);
            col_ptr[j + 1] = row_idx.len();
        }

        Ok(Self {
            rows,
            cols,
            col_ptr,
            row_idx,
            values,
        })
    }

    /// Builds a matrix from raw CSC arrays (copied). Row indices must be in range and
    /// strictly increasing within each column.
    pub fn from_csc(
        rows: usize,
        cols: usize,
        col_ptr: &[usize],
        row_indices: &[usize],
        values: &[f64],
    ) -> Result<Self> {
        if col_ptr.len() != cols + 1
            || col_ptr[0] != 0
            || row_indices.len() != col_ptr[cols]
            || values.len() != col_ptr[cols]
        {
            return Err(invalid("CSC arrays have inconsistent lengths.".into()));
        }
        for j in 0..cols {
            if col_ptr[j + 1] < col_ptr[j] {
                return Err(invalid("CSC column offsets must be non-decreasing.".into()));
            }
            for p in col_ptr[j]..col_ptr[j + 1] {
                let i = row_indices[p];
                if i >= rows || (p > col_ptr[j] && row_indices[p - 1] >= i) {
                    return Err(invalid(
                        "CSC row indices must be in range and strictly increasing per column."
                            .into(),
                    ));
                }
            }
        }
        Ok(Self {
            rows,
            cols,
            col_ptr: col_ptr.to_vec(),
            row_idx: row_indices.to_vec(),
            values: values.to_vec(),
        })
    }

    /// Converts a dense matrix, storing only its nonzero entries.
    pub fn from_dense(dense: &MatrixXd) -> Self {
        let mut col_ptr = vec![0usize; dense.cols() + 1];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for j in 0..dense.cols() {
            for (i, &v) in dense.column(j).iter().enumerate() {
                if v != 0.0 {
                    row_idx.push(i);
                    values.push(v);
                }
            }
            col_ptr[j + 1] = row_idx.len();
        }
        Self {
            rows: dense.rows(),
            cols: dense.cols(),
            col_ptr,
            row_idx,
            values,
        }
    }

    /// Converts to a dense matrix.
    pub fn to_dense(&self) -> MatrixXd {
        let mut dense = MatrixXd::zeros(self.rows, self.cols);
        self.for_each(|i, j, v| dense[(i, j)] = v);
        dense
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of stored entries (Eigen's `nonZeros()`, explicit zeros included).
    pub fn non_zeros(&self) -> usize {
        self.col_ptr[self.cols]
    }

    /// Column offsets, length `cols + 1`.
    pub fn col_ptr(&self) -> &[usize] {
        &self.col_ptr
    }

    /// Row index of each stored entry, increasing within each column.
    pub fn row_indices(&self) -> &[usize] {
        &self.row_idx
    }

    /// Value of each stored entry.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// Mutable values: changing them keeps the pattern.
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }

    /// Entry (i, j), zero if it is not stored (Eigen's `coeff()`). Panics outside the matrix.
    pub fn coeff(&self, i: usize, j: usize) -> f64 {
        self.find(i, j).map_or(0.0, |p| self.values[p])
    }

    /// Reference to the stored entry (i, j). Eigen's `coeffRef` inserts a missing entry; this
    /// type has a fixed pattern, so a structural zero is an error instead.
    pub fn coeff_ref(&mut self, i: usize, j: usize) -> Result<&mut f64> {
        match self.find(i, j) {
            Some(p) => Ok(&mut self.values[p]),
            None => Err(invalid(format!(
                "Entry ({i}, {j}) is not stored in the sparse pattern."
            ))),
        }
    }

    /// This matrix plus `value` times the identity, adding diagonal entries to the pattern
    /// where they are missing.
    pub fn add_to_diagonal(&self, value: f64) -> Self {
        let n = self.rows.min(self.cols);
        let mut triplets = Vec::with_capacity(self.non_zeros() + n);
        self.for_each(|i, j, v| triplets.push(SparseTriplet::new(i, j, v)));
        triplets.extend((0..n).map(|i| SparseTriplet::new(i, i, value)));
        Self::from_triplets(self.rows, self.cols, &triplets)
            .expect("triplets of a valid matrix are in range")
    }

    /// Calls `action(row, col, value)` for every stored entry, column by column.
    pub fn for_each(&self, mut action: impl FnMut(usize, usize, f64)) {
        for j in 0..self.cols {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                action(self.row_idx[p], j, self.values[p]);
            }
        }
    }

    /// The transpose, as a new CSC matrix.
    pub fn transpose(&self) -> Self {
        let nnz = self.non_zeros();
        let mut col_ptr = vec![0usize; self.rows + 1];
        for &r in &self.row_idx[..nnz] {
            col_ptr[r + 1] += 1;
        }
        for i in 0..self.rows {
            col_ptr[i + 1] += col_ptr[i];
        }
        let mut next = col_ptr.clone();
        let mut row_idx = vec![0usize; nnz];
        let mut values = vec![0.0; nnz];
        // Visiting columns in increasing order writes each transposed column's rows in
        // increasing order, so no sort is needed.
        for j in 0..self.cols {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                let q = next[self.row_idx[p]];
                next[self.row_idx[p]] += 1;
                row_idx[q] = j;
                values[q] = self.values[p];
            }
        }
        Self {
            rows: self.cols,
            cols: self.rows,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// The stored triangle of this square matrix that `part` names (Eigen's
    /// `triangularView<Lower/Upper>()` materialized), diagonal included.
    pub fn triangular_part(&self, part: SymmetricPart) -> Self {
        let mut col_ptr = vec![0usize; self.cols + 1];
        let mut row_idx = Vec::with_capacity(self.non_zeros());
        let mut values = Vec::with_capacity(self.non_zeros());
        for j in 0..self.cols {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                if part.contains(self.row_idx[p], j) {
                    row_idx.push(self.row_idx[p]);
                    values.push(self.values[p]);
                }
            }
            col_ptr[j + 1] = row_idx.len();
        }
        Self {
            rows: self.rows,
            cols: self.cols,
            col_ptr,
            row_idx,
            values,
        }
    }

    /// y = A x. Each y(i) accumulates its terms column by column from 0.
    pub fn multiply(&self, x: &VectorXd) -> VectorXd {
        assert_eq!(
            x.len(),
            self.cols,
            "Cannot multiply a {}x{} matrix by a vector of length {}.",
            self.rows,
            self.cols,
            x.len()
        );
        let mut y = vec![0.0; self.rows];
        let xs = x.as_slice();
        for (j, &xj) in xs.iter().enumerate() {
            for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                y[self.row_idx[p]] += self.values[p] * xj;
            }
        }
        VectorXd::from_vec(y)
    }

    /// y = A^T x, without forming the transpose.
    pub fn transpose_multiply(&self, x: &VectorXd) -> VectorXd {
        assert_eq!(
            x.len(),
            self.rows,
            "Cannot multiply the transpose of a {}x{} matrix by a vector of length {}.",
            self.rows,
            self.cols,
            x.len()
        );
        let xs = x.as_slice();
        let y = (0..self.cols)
            .map(|j| {
                let mut sum = 0.0;
                for p in self.col_ptr[j]..self.col_ptr[j + 1] {
                    sum += self.values[p] * xs[self.row_idx[p]];
                }
                sum
            })
            .collect();
        VectorXd::from_vec(y)
    }

    /// A^T A with both triangles stored (the normal-equations matrix).
    pub fn transpose_times_self(&self) -> Self {
        &self.transpose() * self
    }

    /// A^T A with only the `part` triangle stored.
    pub fn transpose_times_self_part(&self, part: SymmetricPart) -> Self {
        self.transpose_times_self().triangular_part(part)
    }

    /// The stored entries inside the `rows x cols` block at (`row`, `col`), as a new matrix
    /// (Eigen's `SparseMatrix::block()` assigned to a `SparseMatrix`).
    pub fn block(&self, row: usize, col: usize, rows: usize, cols: usize) -> Result<Self> {
        if row + rows > self.rows || col + cols > self.cols {
            return Err(invalid(format!(
                "Block ({row}, {col}, {rows}, {cols}) is outside a {}x{} matrix.",
                self.rows, self.cols
            )));
        }
        let mut col_ptr = vec![0usize; cols + 1];
        let mut row_idx = Vec::new();
        let mut values = Vec::new();
        for j in 0..cols {
            for p in self.col_ptr[col + j]..self.col_ptr[col + j + 1] {
                let i = self.row_idx[p];
                if i >= row && i < row + rows {
                    row_idx.push(i - row);
                    values.push(self.values[p]);
                }
            }
            col_ptr[j + 1] = row_idx.len();
        }
        Ok(Self {
            rows,
            cols,
            col_ptr,
            row_idx,
            values,
        })
    }

    /// Position of entry (i, j) in the value array, or `None` when it is not stored. Panics
    /// outside the matrix.
    fn find(&self, i: usize, j: usize) -> Option<usize> {
        assert!(
            i < self.rows && j < self.cols,
            "Entry ({i}, {j}) is outside a {}x{} matrix.",
            self.rows,
            self.cols
        );
        let start = self.col_ptr[j];
        self.row_idx[start..self.col_ptr[j + 1]]
            .binary_search(&i)
            .ok()
            .map(|k| start + k)
    }
}

/// Sorts one column's (row, value) pairs by row. Rows are unique, so the order is total and
/// an unstable sort is deterministic.
fn sort_column(rows: &mut [usize], values: &mut [f64]) {
    if rows.len() < 2 || rows.windows(2).all(|w| w[0] < w[1]) {
        return;
    }
    let mut pairs: Vec<(usize, f64)> = rows.iter().copied().zip(values.iter().copied()).collect();
    pairs.sort_unstable_by_key(|&(r, _)| r);
    for (k, (r, v)) in pairs.into_iter().enumerate() {
        rows[k] = r;
        values[k] = v;
    }
}

impl std::ops::Mul<&VectorXd> for &SparseMatrixCsc {
    type Output = VectorXd;
    /// A x, [`SparseMatrixCsc::multiply`].
    fn mul(self, x: &VectorXd) -> VectorXd {
        self.multiply(x)
    }
}

impl std::ops::Mul for &SparseMatrixCsc {
    type Output = SparseMatrixCsc;
    /// Sparse product A B (Gustavson's column-by-column algorithm: column j of the product
    /// is the combination of A's columns weighted by column j of B). Every product term
    /// stays in the pattern, even one that cancels to zero.
    fn mul(self, b: &SparseMatrixCsc) -> SparseMatrixCsc {
        let a = self;
        assert_eq!(
            a.cols, b.rows,
            "Cannot multiply a {}x{} by a {}x{} sparse matrix.",
            a.rows, a.cols, b.rows, b.cols
        );
        let mut col_ptr = vec![0usize; b.cols + 1];
        let mut row_idx = Vec::with_capacity(a.non_zeros() + b.non_zeros());
        let mut values: Vec<f64> = Vec::with_capacity(a.non_zeros() + b.non_zeros());
        let mut slot_of = vec![usize::MAX; a.rows];
        for j in 0..b.cols {
            let start = row_idx.len();
            for pb in b.col_ptr[j]..b.col_ptr[j + 1] {
                let k = b.row_idx[pb];
                let bkj = b.values[pb];
                for pa in a.col_ptr[k]..a.col_ptr[k + 1] {
                    let i = a.row_idx[pa];
                    let slot = slot_of[i];
                    if slot != usize::MAX && slot >= start {
                        values[slot] += a.values[pa] * bkj;
                    } else {
                        slot_of[i] = row_idx.len();
                        row_idx.push(i);
                        values.push(a.values[pa] * bkj);
                    }
                }
            }
            sort_column(&mut row_idx[start..], &mut values[start..]);
            col_ptr[j + 1] = row_idx.len();
        }
        SparseMatrixCsc {
            rows: a.rows,
            cols: b.cols,
            col_ptr,
            row_idx,
            values,
        }
    }
}

impl std::ops::Sub for &SparseMatrixCsc {
    type Output = SparseMatrixCsc;
    /// A - B over the union of both patterns (an entry only in B stores -b).
    fn sub(self, b: &SparseMatrixCsc) -> SparseMatrixCsc {
        let a = self;
        assert!(
            a.rows == b.rows && a.cols == b.cols,
            "Cannot subtract a {}x{} from a {}x{} sparse matrix.",
            b.rows,
            b.cols,
            a.rows,
            a.cols
        );
        let mut col_ptr = vec![0usize; a.cols + 1];
        let mut row_idx = Vec::with_capacity(a.non_zeros() + b.non_zeros());
        let mut values = Vec::with_capacity(a.non_zeros() + b.non_zeros());
        for j in 0..a.cols {
            // Merge the two sorted row lists of column j.
            let (mut pa, ea) = (a.col_ptr[j], a.col_ptr[j + 1]);
            let (mut pb, eb) = (b.col_ptr[j], b.col_ptr[j + 1]);
            while pa < ea || pb < eb {
                let ia = if pa < ea { a.row_idx[pa] } else { usize::MAX };
                let ib = if pb < eb { b.row_idx[pb] } else { usize::MAX };
                if ia == ib {
                    row_idx.push(ia);
                    values.push(a.values[pa] - b.values[pb]);
                    pa += 1;
                    pb += 1;
                } else if ia < ib {
                    row_idx.push(ia);
                    values.push(a.values[pa]);
                    pa += 1;
                } else {
                    row_idx.push(ib);
                    values.push(-b.values[pb]);
                    pb += 1;
                }
            }
            col_ptr[j + 1] = row_idx.len();
        }
        SparseMatrixCsc {
            rows: a.rows,
            cols: a.cols,
            col_ptr,
            row_idx,
            values,
        }
    }
}
