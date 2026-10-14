// The block-sparse Jacobian: a `CompressedRowBlockStructure` (`block_structure.rs`) and one
// value array holding every cell, row-major per cell. Port of `BlockSparseMatrix` in
// colmap-sharp's `ColmapSharp/Solver/BlockSparseMatrix.cs` (our MIT code), which follows Ceres
// Solver 2.2.0 `internal/ceres/block_sparse_matrix.cc` (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// The evaluator writes each cell in place (no copy), at the offsets the block Jacobian layout
// assigns: Ceres puts the cells of the first num_eliminate_blocks column blocks (the Schur "E"
// blocks) before all the others in the value array, and the Schur solvers rely on that; with
// no eliminated blocks the layout is simply residual-block order. The products run each cell
// through `small_blas.rs`' vector kernels, as Ceres' do, so every row (column) of a cell is a
// left-to-right sum added to y once. The sparse normal Cholesky solver forms J'J from this
// structure.

use super::block_structure::CompressedRowBlockStructure;
use super::small_blas::{self, Operation};
use super::sparse_matrix::{LinearOperator, SparseMatrix};

/// `ceres::internal::BlockSparseMatrix`.
#[derive(Clone, Debug)]
pub struct BlockSparseMatrix {
    structure: CompressedRowBlockStructure,
    values: Vec<f64>,
    num_rows: usize,
    num_cols: usize,
}

impl BlockSparseMatrix {
    /// A zero matrix with the given structure; the value array is sized from the cells (the
    /// cells' positions must lie within it).
    pub fn new(structure: CompressedRowBlockStructure) -> Self {
        let mut num_rows = 0;
        let mut num_nonzeros = 0;
        for row in &structure.rows {
            num_rows += row.block.size;
            for cell in &row.cells {
                num_nonzeros += row.block.size * structure.cols[cell.block_id].size;
            }
        }

        let num_cols = structure.cols.iter().map(|col| col.size).sum();
        Self {
            structure,
            values: vec![0.0; num_nonzeros],
            num_rows,
            num_cols,
        }
    }

    /// The block structure.
    pub fn structure(&self) -> &CompressedRowBlockStructure {
        &self.structure
    }

    /// The cell values; each cell is row-major, row block size x column block size.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The cell values, writable (the evaluator writes the Jacobian cells here).
    pub fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
}

impl LinearOperator for BlockSparseMatrix {
    fn right_multiply_and_accumulate(&self, x: &[f64], y: &mut [f64]) {
        let cols = &self.structure.cols;
        for row in &self.structure.rows {
            let row_block = row.block;
            for cell in &row.cells {
                let col = cols[cell.block_id];
                small_blas::matrix_vector_multiply(
                    &self.values[cell.position..cell.position + row_block.size * col.size],
                    row_block.size,
                    col.size,
                    &x[col.position..col.position + col.size],
                    &mut y[row_block.position..row_block.position + row_block.size],
                    Operation::Add,
                );
            }
        }
    }
}

impl SparseMatrix for BlockSparseMatrix {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn num_cols(&self) -> usize {
        self.num_cols
    }

    fn set_zero(&mut self) {
        self.values.fill(0.0);
    }

    fn left_multiply_and_accumulate(&self, x: &[f64], y: &mut [f64]) {
        let cols = &self.structure.cols;
        for row in &self.structure.rows {
            let row_block = row.block;
            for cell in &row.cells {
                let col = cols[cell.block_id];
                small_blas::matrix_transpose_vector_multiply(
                    &self.values[cell.position..cell.position + row_block.size * col.size],
                    row_block.size,
                    col.size,
                    &x[row_block.position..row_block.position + row_block.size],
                    &mut y[col.position..col.position + col.size],
                    Operation::Add,
                );
            }
        }
    }

    fn squared_column_norm(&self, x: &mut [f64]) {
        x[..self.num_cols].fill(0.0);
        let cols = &self.structure.cols;
        for row in &self.structure.rows {
            let row_size = row.block.size;
            for cell in &row.cells {
                let col = cols[cell.block_id];
                for c in 0..col.size {
                    let mut sum = 0.0;
                    for r in 0..row_size {
                        let v = self.values[cell.position + r * col.size + c];
                        sum += v * v;
                    }

                    x[col.position + c] += sum;
                }
            }
        }
    }

    fn scale_columns(&mut self, scale: &[f64]) {
        let cols = &self.structure.cols;
        for row in &self.structure.rows {
            let row_size = row.block.size;
            for cell in &row.cells {
                let col = cols[cell.block_id];
                if col.size == 0 {
                    continue;
                }

                let scale = &scale[col.position..col.position + col.size];
                let cell_values =
                    &mut self.values[cell.position..cell.position + row_size * col.size];
                for cell_row in cell_values.chunks_exact_mut(col.size) {
                    for (v, &s) in cell_row.iter_mut().zip(scale) {
                        *v *= s;
                    }
                }
            }
        }
    }
}
