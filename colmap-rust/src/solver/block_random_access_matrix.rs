// The reduced camera matrix S the Schur eliminator accumulates into, addressed by (row block,
// column block). Port of colmap-sharp's `ColmapSharp/Solver/BlockRandomAccessMatrix.cs` (our
// MIT code), which follows Ceres Solver 2.2.0 `internal/ceres/block_random_access_matrix.h`,
// `block_random_access_dense_matrix.cc` and `block_random_access_diagonal_matrix.cc`
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md). The sparse variant is in
// `block_random_access_sparse_matrix.rs`.
//
// Only the upper block triangle is written.
// - Dense (DENSE_SCHUR, [`BlockRandomAccessDenseMatrix`]): one row-major n x n array. Read as
//   column-major, its upper triangle is the lower triangle `linalg::Llt` factors, so the
//   matrix goes to the dense Cholesky with a plain copy.
// - Sparse (SPARSE_SCHUR): one row-major cell per stored block pair (i, j), i <= j.
// - Diagonal (the SCHUR_JACOBI preconditioner, [`BlockRandomAccessDiagonalMatrix`]): only the
//   diagonal blocks, inverted in place.
//
// Rust shape: Ceres' `GetCell` hands out a pointer and colmap-sharp's `TryGetCell` a span into
// the shared value array. Here the primitive is [`BlockRandomAccessMatrix::cell_location`],
// the cell's offset, row stride and extent in [`BlockRandomAccessMatrix::values`]: the Schur
// eliminator resolves every cell it touches once per structure (colmap-sharp's `CellOffset`)
// and then writes through `values_mut()` with the small BLAS kernels, so no borrow of the
// matrix outlives a single kernel call. `cell` / `cell_mut` are the one-off conveniences.
// Ceres guards each cell with a mutex for its multithreaded elimination; these matrices have no
// locks, because `values_mut()` needs exclusive access: a writer serializes its own writes, in
// an order it fixes.

use super::block_structure::Block;
use super::small_blas::{self, Operation};

/// Where a stored cell lives in a [`BlockRandomAccessMatrix`]'s values: entry (r, c) of the
/// cell is at `offset + r * stride + c`, and the cell spans `len` values from `offset` (up to
/// and including its last entry).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CellLocation {
    /// Index of the cell's (0, 0) entry in the values.
    pub offset: usize,
    /// Distance between the starts of consecutive cell rows.
    pub stride: usize,
    /// Number of values from `offset` through the cell's last entry.
    pub len: usize,
}

impl CellLocation {
    /// The location of a `rows x cols` cell at `offset` with row stride `stride`.
    pub fn new(offset: usize, stride: usize, rows: usize, cols: usize) -> Self {
        let len = if rows == 0 || cols == 0 {
            0
        } else {
            (rows - 1) * stride + cols
        };
        Self {
            offset,
            stride,
            len,
        }
    }
}

/// `ceres::internal::BlockRandomAccessMatrix`: a square matrix over blocks whose cells can be
/// addressed by (row block, column block).
pub trait BlockRandomAccessMatrix {
    /// Scalar rows (= columns).
    fn num_rows(&self) -> usize;

    /// The cell (`row_block_id`, `col_block_id`) if it is stored, `None` if it is not.
    fn cell_location(&self, row_block_id: usize, col_block_id: usize) -> Option<CellLocation>;

    /// Every stored value; the cells are views into it.
    fn values(&self) -> &[f64];

    /// Every stored value, writable.
    fn values_mut(&mut self) -> &mut [f64];

    /// Zeroes every stored value.
    fn set_zero(&mut self) {
        self.values_mut().fill(0.0);
    }

    /// The stored cell starting at its (0, 0) entry, with its row stride.
    fn cell(&self, row_block_id: usize, col_block_id: usize) -> Option<(&[f64], usize)> {
        let location = self.cell_location(row_block_id, col_block_id)?;
        Some((
            &self.values()[location.offset..location.offset + location.len],
            location.stride,
        ))
    }

    /// The stored cell starting at its (0, 0) entry, writable, with its row stride.
    fn cell_mut(
        &mut self,
        row_block_id: usize,
        col_block_id: usize,
    ) -> Option<(&mut [f64], usize)> {
        let location = self.cell_location(row_block_id, col_block_id)?;
        Some((
            &mut self.values_mut()[location.offset..location.offset + location.len],
            location.stride,
        ))
    }
}

/// `ceres::internal::BlockRandomAccessDenseMatrix`: every cell, in one row-major n x n array.
#[derive(Clone, Debug)]
pub struct BlockRandomAccessDenseMatrix {
    blocks: Vec<Block>,
    // Scalar position of each block (the running sum of the sizes, as Ceres computes it).
    block_positions: Vec<usize>,
    num_rows: usize,
    values: Vec<f64>,
}

impl BlockRandomAccessDenseMatrix {
    /// A zero matrix over `blocks` (only their sizes are used).
    pub fn new(blocks: &[Block]) -> Self {
        let mut block_positions = Vec::with_capacity(blocks.len());
        let mut n = 0;
        for block in blocks {
            block_positions.push(n);
            n += block.size;
        }

        Self {
            blocks: blocks.to_vec(),
            block_positions,
            num_rows: n,
            values: vec![0.0; n * n],
        }
    }
}

impl BlockRandomAccessMatrix for BlockRandomAccessDenseMatrix {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn cell_location(&self, row_block_id: usize, col_block_id: usize) -> Option<CellLocation> {
        Some(CellLocation::new(
            self.block_positions[row_block_id] * self.num_rows + self.block_positions[col_block_id],
            self.num_rows,
            self.blocks[row_block_id].size,
            self.blocks[col_block_id].size,
        ))
    }

    fn values(&self) -> &[f64] {
        &self.values
    }

    fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
}

/// `ceres::internal::BlockRandomAccessDiagonalMatrix`: only the diagonal blocks, each
/// row-major.
#[derive(Clone, Debug)]
pub struct BlockRandomAccessDiagonalMatrix {
    blocks: Vec<Block>,
    offsets: Vec<usize>,
    num_rows: usize,
    values: Vec<f64>,
    // The Cholesky factor of one block while `invert` runs, sized for the largest block.
    scratch: Vec<f64>,
}

impl BlockRandomAccessDiagonalMatrix {
    /// A zero block-diagonal matrix over `blocks`; their positions place the blocks in the
    /// vectors of [`Self::right_multiply_and_accumulate`].
    pub fn new(blocks: &[Block]) -> Self {
        let mut offsets = Vec::with_capacity(blocks.len());
        let mut offset = 0;
        let mut n = 0;
        let mut max_size = 0;
        for block in blocks {
            offsets.push(offset);
            offset += block.size * block.size;
            n += block.size;
            max_size = max_size.max(block.size);
        }

        Self {
            blocks: blocks.to_vec(),
            offsets,
            num_rows: n,
            values: vec![0.0; offset],
            scratch: vec![0.0; max_size * max_size],
        }
    }

    /// Replaces every block by its inverse, computed from its upper triangle
    /// (`small_blas::invert_upper_psd_in_place`; a block that is not positive definite
    /// becomes NaN).
    pub fn invert(&mut self) {
        for (block, &offset) in self.blocks.iter().zip(&self.offsets) {
            let size = block.size;
            small_blas::invert_upper_psd_in_place(
                &mut self.values[offset..offset + size * size],
                size,
                &mut self.scratch,
            );
        }
    }

    /// `y += M x`.
    pub fn right_multiply_and_accumulate(&self, x: &[f64], y: &mut [f64]) {
        for (block, &offset) in self.blocks.iter().zip(&self.offsets) {
            let (size, position) = (block.size, block.position);
            small_blas::matrix_vector_multiply(
                &self.values[offset..offset + size * size],
                size,
                size,
                &x[position..position + size],
                &mut y[position..position + size],
                Operation::Add,
            );
        }
    }
}

impl BlockRandomAccessMatrix for BlockRandomAccessDiagonalMatrix {
    fn num_rows(&self) -> usize {
        self.num_rows
    }

    fn cell_location(&self, row_block_id: usize, col_block_id: usize) -> Option<CellLocation> {
        if row_block_id != col_block_id {
            return None;
        }

        let size = self.blocks[row_block_id].size;
        Some(CellLocation::new(
            self.offsets[row_block_id],
            size,
            size,
            size,
        ))
    }

    fn values(&self) -> &[f64] {
        &self.values
    }

    fn values_mut(&mut self) -> &mut [f64] {
        &mut self.values
    }
}
