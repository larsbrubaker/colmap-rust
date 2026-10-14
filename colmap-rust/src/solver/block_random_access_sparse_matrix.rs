// The SPARSE_SCHUR reduced camera matrix: one row-major cell per block pair (i, j), i <= j,
// that two F blocks share through an E block or a row. Port of `BlockRandomAccessSparseMatrix`
// in colmap-sharp's `ColmapSharp/Solver/BlockRandomAccessMatrix.cs` (our MIT code), which
// follows Ceres Solver 2.2.0 `internal/ceres/block_random_access_sparse_matrix.cc`
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md). The trait and the dense and diagonal variants
// are in `block_random_access_matrix.rs`.
//
// The cells are laid out in (row block, column block) order, the row-major layout of Ceres'
// underlying BlockSparseMatrix. Ceres finds a cell through an unordered_map keyed by its
// bit-packed (row, col); here the cells are sorted, so `cell_location` binary-searches the
// row's cells (no hashing, no iteration order to keep deterministic). `to_lower_csc` hands the
// lower triangle to the sparse Cholesky (`linalg::SimplicialCholesky`), with the value index
// each CSC entry is copied from.

use std::collections::BTreeSet;

use crate::linalg::SparseMatrixCsc;
use crate::Result;

use super::block_random_access_matrix::{BlockRandomAccessMatrix, CellLocation};
use super::block_structure::Block;
use super::small_blas::{self, Operation};

#[derive(Clone, Copy, Debug)]
struct StoredCell {
    row: usize,
    col: usize,
    offset: usize,
}

/// `ceres::internal::BlockRandomAccessSparseMatrix`.
#[derive(Clone, Debug)]
pub struct BlockRandomAccessSparseMatrix {
    blocks: Vec<Block>,
    // Cells by (row block, column block).
    cells: Vec<StoredCell>,
    // The cells of block row i are cells[row_starts[i]..row_starts[i + 1]].
    row_starts: Vec<usize>,
    num_rows: usize,
    values: Vec<f64>,
}

impl BlockRandomAccessSparseMatrix {
    /// A zero matrix over `blocks` storing the cells `block_pairs` (each `(i, j)` with
    /// `i <= j`). The blocks' positions must be the running sum of their sizes.
    pub fn new(blocks: &[Block], block_pairs: &BTreeSet<(usize, usize)>) -> Self {
        let num_rows = blocks.iter().map(|block| block.size).sum();
        let mut cells = Vec::with_capacity(block_pairs.len());
        let mut row_starts = vec![0; blocks.len() + 1];
        let mut offset = 0;
        for &(row, col) in block_pairs {
            cells.push(StoredCell { row, col, offset });
            row_starts[row + 1] += 1;
            offset += blocks[row].size * blocks[col].size;
        }

        for i in 0..blocks.len() {
            row_starts[i + 1] += row_starts[i];
        }

        Self {
            blocks: blocks.to_vec(),
            cells,
            row_starts,
            num_rows,
            values: vec![0.0; offset],
        }
    }

    /// `y += S x` for the symmetric S whose upper block triangle is stored: each off-diagonal
    /// cell also does the corresponding lower-triangle multiply.
    pub fn symmetric_right_multiply_and_accumulate(&self, x: &[f64], y: &mut [f64]) {
        for cell in &self.cells {
            let row_block = self.blocks[cell.row];
            let col_block = self.blocks[cell.col];
            let values = &self.values[cell.offset..cell.offset + row_block.size * col_block.size];
            small_blas::matrix_vector_multiply(
                values,
                row_block.size,
                col_block.size,
                &x[col_block.position..col_block.position + col_block.size],
                &mut y[row_block.position..row_block.position + row_block.size],
                Operation::Add,
            );
            if cell.row != cell.col {
                small_blas::matrix_transpose_vector_multiply(
                    values,
                    row_block.size,
                    col_block.size,
                    &x[row_block.position..row_block.position + row_block.size],
                    &mut y[col_block.position..col_block.position + col_block.size],
                    Operation::Add,
                );
            }
        }
    }

    /// The lower triangle of the symmetric matrix as CSC (zero values), with, for every stored
    /// entry, the index in [`BlockRandomAccessMatrix::values`] it is copied from (column c of
    /// the lower triangle is row c of the stored upper triangle). Fails only if the blocks'
    /// positions are not increasing.
    pub fn to_lower_csc(&self) -> Result<(SparseMatrixCsc, Vec<usize>)> {
        let mut col_ptr = vec![0; self.num_rows + 1];
        let mut row_indices = Vec::new();
        let mut sources = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            // The cells of block row i, by increasing column block.
            let row_cells = &self.cells[self.row_starts[i]..self.row_starts[i + 1]];
            for r in 0..block.size {
                for cell in row_cells {
                    let col_block = self.blocks[cell.col];
                    let width = col_block.size;
                    let first = if cell.col == i { r } else { 0 };
                    for c in first..width {
                        row_indices.push(col_block.position + c);
                        sources.push(cell.offset + r * width + c);
                    }
                }

                col_ptr[block.position + r + 1] = row_indices.len();
            }
        }

        let values = vec![0.0; row_indices.len()];
        let csc = SparseMatrixCsc::from_csc(
            self.num_rows,
            self.num_rows,
            &col_ptr,
            &row_indices,
            &values,
        )?;
        Ok((csc, sources))
    }
}

impl BlockRandomAccessMatrix for BlockRandomAccessSparseMatrix {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn cell_location(&self, row_block_id: usize, col_block_id: usize) -> Option<CellLocation> {
        let row_cells =
            &self.cells[self.row_starts[row_block_id]..self.row_starts[row_block_id + 1]];
        let k = row_cells
            .binary_search_by(|cell| cell.col.cmp(&col_block_id))
            .ok()?;
        let stride = self.blocks[col_block_id].size;
        Some(CellLocation::new(
            row_cells[k].offset,
            stride,
            self.blocks[row_block_id].size,
            stride,
        ))
    }

    fn values(&self) -> &[f64] {
        &self.values
    }

    fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
}
