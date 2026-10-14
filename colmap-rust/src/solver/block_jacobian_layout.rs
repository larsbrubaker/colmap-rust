//! Where each residual block's Jacobian cells live in a [`BlockSparseMatrix`]'s value array,
//! and that matrix's block structure. Port of Ceres Solver 2.2.0
//! `internal/ceres/block_jacobian_writer.cc` (`BuildJacobianLayout`, `CreateJacobian`;
//! BSD-3-Clause, see THIRD_PARTY_NOTICES.md); colmap-sharp's `BlockJacobianLayout` in
//! `ColmapSharp/Solver/BlockSparseMatrix.cs` is the C# form.
//!
//! The cells of the first `num_eliminate_blocks` column blocks (the Schur "E" blocks) go before
//! all the others in the value array; the Schur solvers rely on that. Within a residual block,
//! cells are laid out in column-block order. Constant parameter blocks get no cell. Ceres'
//! `CreateJacobian` CHECKs that the program holds no constant block; `Problem::evaluate` keeps
//! constant blocks listed in its options as (empty) columns, as colmap-sharp's port does, so
//! here they get a column and no cells instead. The layout is stored flat (one offset list
//! for all residual blocks), so building it allocates twice whatever the problem size.

use super::block_sparse_matrix::BlockSparseMatrix;
use super::block_structure::{Block, Cell, CompressedRow, CompressedRowBlockStructure};
use super::problem::ProblemBlocks;
use super::program::Program;

/// Port of `ceres::internal::BlockJacobianWriter`'s layout.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BlockJacobianLayout {
    // positions[starts[i] + k]: value offset of the k-th variable parameter block (in residual
    // block order) of residual block i.
    positions: Vec<usize>,
    starts: Vec<usize>,
}

impl BlockJacobianLayout {
    /// Computes the layout of `program` (whose parameter blocks carry their program indices,
    /// `Program::set_parameter_offsets_and_index`).
    pub fn new(program: &Program, blocks: &ProblemBlocks, num_eliminate_blocks: usize) -> Self {
        let is_e_block = |index: isize| index >= 0 && (index as usize) < num_eliminate_blocks;
        let mut f_block_pos = 0;
        let mut num_jacobian_blocks = 0;
        for &id in &program.residual_blocks {
            let residual_block = blocks.residual_block(id);
            for &pid in residual_block.parameter_blocks() {
                let parameter_block = blocks.parameter_block(pid);
                if !parameter_block.is_constant() {
                    num_jacobian_blocks += 1;
                    if is_e_block(parameter_block.index) {
                        f_block_pos +=
                            residual_block.num_residuals() * parameter_block.tangent_size();
                    }
                }
            }
        }

        let mut positions = vec![0; num_jacobian_blocks];
        let mut starts = Vec::with_capacity(program.residual_blocks.len() + 1);
        let mut e_block_pos = 0;
        let mut cursor = 0;
        // (k, j): the k-th variable block of the residual block is its j-th parameter block.
        let mut active: Vec<(usize, usize)> = Vec::new();
        for &id in &program.residual_blocks {
            let residual_block = blocks.residual_block(id);
            let parameter_blocks = residual_block.parameter_blocks();
            starts.push(cursor);
            active.clear();
            for (j, &pid) in parameter_blocks.iter().enumerate() {
                if !blocks.parameter_block(pid).is_constant() {
                    active.push((active.len(), j));
                }
            }

            // Cells go in column-block order. Parameter block indices within one residual
            // block are distinct, so this order has no ties.
            active.sort_by_key(|&(_, j)| blocks.parameter_block(parameter_blocks[j]).index);
            for &(k, j) in &active {
                let parameter_block = blocks.parameter_block(parameter_blocks[j]);
                let size = residual_block.num_residuals() * parameter_block.tangent_size();
                if is_e_block(parameter_block.index) {
                    positions[cursor + k] = e_block_pos;
                    e_block_pos += size;
                } else {
                    positions[cursor + k] = f_block_pos;
                    f_block_pos += size;
                }
            }
            cursor += active.len();
        }
        starts.push(cursor);
        Self { positions, starts }
    }

    /// Value offsets of the `i`-th residual block's variable parameter blocks, in its order.
    pub fn positions(&self, i: usize) -> &[usize] {
        &self.positions[self.starts[i]..self.starts[i + 1]]
    }

    /// The (zero) block-sparse Jacobian of `program`: one column block per parameter block
    /// (its tangent size), one row block per residual block, cells sorted by column block.
    pub fn create_jacobian(&self, program: &Program, blocks: &ProblemBlocks) -> BlockSparseMatrix {
        let mut cols = Vec::with_capacity(program.parameter_blocks.len());
        let mut cursor = 0;
        for &pid in &program.parameter_blocks {
            let size = blocks.parameter_block(pid).tangent_size();
            cols.push(Block::new(size, cursor));
            cursor += size;
        }

        let mut rows = Vec::with_capacity(program.residual_blocks.len());
        let mut row_block_position = 0;
        for (i, &id) in program.residual_blocks.iter().enumerate() {
            let residual_block = blocks.residual_block(id);
            let positions = self.positions(i);
            let mut cells = Vec::with_capacity(positions.len());
            let mut k = 0;
            for &pid in residual_block.parameter_blocks() {
                let parameter_block = blocks.parameter_block(pid);
                if !parameter_block.is_constant() {
                    cells.push(Cell::new(parameter_block.index as usize, positions[k]));
                    k += 1;
                }
            }
            cells.sort_by_key(|cell| cell.block_id);
            let num_residuals = residual_block.num_residuals();
            rows.push(CompressedRow::new(
                Block::new(num_residuals, row_block_position),
                cells,
            ));
            row_block_position += num_residuals;
        }
        BlockSparseMatrix::new(CompressedRowBlockStructure::new(cols, rows))
    }
}
