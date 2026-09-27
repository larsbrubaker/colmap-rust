//! `RowMajorMatrix<T>`: a dynamic-size, row-major matrix of plain values, the counterpart of
//! COLMAP's `Eigen::Matrix<T, Dynamic, Dynamic, RowMajor>` typedefs that carry feature data
//! rather than linear algebra: `FeatureDescriptorsData` (uint8), `FeatureDescriptorsFloatData`
//! and `FeatureKeypointsBlob` (float), `FeatureMatchesBlob` (uint32).
//!
//! Port of colmap-sharp's `ColmapSharp/LinearAlgebra/RowMajorMatrix.cs` (MIT, written there,
//! not ported from Eigen). The only Eigen behavior it mirrors is the storage order: element
//! `(r, c)` lives at `data[r * cols + c]`, so `data.data()[i]` in COLMAP is `as_slice()[i]`
//! here. Numeric linear algebra stays in [`super::MatrixXd`]. Unlike Eigen, a new matrix is
//! zero-filled (`T::default()`) rather than uninitialized; COLMAP never reads an uninitialized
//! one. Equality is same shape and same elements.
//!
//! colmap-sharp's `FixedBuffers.cs` (.NET inline arrays behind the fixed-size matrices) has no
//! counterpart: Rust's `[f64; N]` fields already give allocation-free value storage.
//! Tests: `colmap-rust/tests/linalg/rust_only_dynamic_matrix.rs`.

/// A dense row-major matrix of `T` with value equality (same shape and elements).
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct RowMajorMatrix<T> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Copy + Default> RowMajorMatrix<T> {
    /// A `rows x cols` matrix filled with `T::default()` (zero for numbers).
    pub fn new(rows: usize, cols: usize) -> Self {
        let len = rows
            .checked_mul(cols)
            .expect("RowMajorMatrix size overflow");
        Self {
            rows,
            cols,
            data: vec![T::default(); len],
        }
    }
}

impl<T> RowMajorMatrix<T> {
    /// A `rows x cols` matrix over `data` (row-major); panics unless
    /// `data.len() == rows * cols`.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<T>) -> Self {
        let len = rows
            .checked_mul(cols)
            .expect("RowMajorMatrix size overflow");
        assert_eq!(
            data.len(),
            len,
            "RowMajorMatrix data length {} != {rows}x{cols}",
            data.len()
        );
        Self { rows, cols, data }
    }

    /// Number of rows.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Number of columns.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Number of elements, Eigen's `size()`.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True when the matrix has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// The row-major element storage, Eigen's `data()`.
    pub fn as_slice(&self) -> &[T] {
        &self.data
    }

    /// The row-major element storage, writable.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }

    /// Consumes the matrix and returns its row-major storage.
    pub fn into_vec(self) -> Vec<T> {
        self.data
    }

    /// The elements of row `row`.
    pub fn row(&self, row: usize) -> &[T] {
        assert!(row < self.rows, "row {row} outside {} rows", self.rows);
        &self.data[row * self.cols..(row + 1) * self.cols]
    }

    /// The elements of row `row`, writable.
    pub fn row_mut(&mut self, row: usize) -> &mut [T] {
        assert!(row < self.rows, "row {row} outside {} rows", self.rows);
        &mut self.data[row * self.cols..(row + 1) * self.cols]
    }

    fn check_index(&self, row: usize, col: usize) {
        assert!(
            row < self.rows && col < self.cols,
            "({row}, {col}) outside {}x{}",
            self.rows,
            self.cols
        );
    }
}

/// Element `(row, col)`; panics when either is out of range.
impl<T> std::ops::Index<(usize, usize)> for RowMajorMatrix<T> {
    type Output = T;
    fn index(&self, (row, col): (usize, usize)) -> &T {
        self.check_index(row, col);
        &self.data[row * self.cols + col]
    }
}

impl<T> std::ops::IndexMut<(usize, usize)> for RowMajorMatrix<T> {
    fn index_mut(&mut self, (row, col): (usize, usize)) -> &mut T {
        self.check_index(row, col);
        &mut self.data[row * self.cols + col]
    }
}
