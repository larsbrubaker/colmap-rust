//! Evaluates a [`Program`] at a state vector: total cost, residual vector, gradient and
//! Jacobian (dense or block-sparse). Port of Ceres Solver 2.2.0
//! `internal/ceres/program_evaluator.h`, `evaluator.cc`, `dense_jacobian_writer.h` and
//! `block_evaluate_preparer.cc` (BSD-3-Clause, see THIRD_PARTY_NOTICES.md); colmap-sharp's
//! `ColmapSharp/Solver/ProgramEvaluator.cs` is the C# form.
//!
//! The core is single-threaded (CLAUDE.md), so the residual blocks run in order: the cost is
//! summed over residual blocks in order and the gradient is J'r, accumulated row block by row
//! block, which is what Ceres computes on one thread (colmap-sharp keeps that order for any
//! thread count, its divergence 18). A block-sparse Jacobian's cells are written in place at
//! the offsets of a [`BlockJacobianLayout`]; a dense Jacobian's through a per-residual-block
//! scratch, copied into its rows. A gradient asked for without a Jacobian or residuals goes
//! through a Jacobian and residual vector the evaluator owns, allocated on first use and
//! reused: after construction an evaluation allocates nothing.

use super::block_jacobian_layout::BlockJacobianLayout;
use super::block_sparse_matrix::BlockSparseMatrix;
use super::problem::ProblemBlocks;
use super::program::Program;
use super::residual_block::{ParameterValues, MAX_STACK_BLOCKS};
use super::sparse_matrix::{DenseSparseMatrix, SparseMatrix};
use crate::Result;

/// A Jacobian of the kind a [`ProgramEvaluator`] writes.
#[derive(Clone, Debug)]
pub enum EvaluatorJacobian {
    /// For DENSE_QR and DENSE_NORMAL_CHOLESKY.
    Dense(DenseSparseMatrix),
    /// For the sparse and Schur solvers (and `Problem::evaluate`).
    BlockSparse(BlockSparseMatrix),
}

impl EvaluatorJacobian {
    /// The Jacobian as the minimizer sees it.
    pub fn as_sparse_matrix(&self) -> &dyn SparseMatrix {
        match self {
            Self::Dense(matrix) => matrix,
            Self::BlockSparse(matrix) => matrix,
        }
    }

    /// The Jacobian as the minimizer sees it, writable.
    pub fn as_sparse_matrix_mut(&mut self) -> &mut dyn SparseMatrix {
        match self {
            Self::Dense(matrix) => matrix,
            Self::BlockSparse(matrix) => matrix,
        }
    }
}

/// Port of `ceres::internal::ProgramEvaluator` for the dense and block-sparse Jacobians.
#[derive(Clone, Debug)]
pub struct ProgramEvaluator {
    program: Program,
    dense_jacobian: bool,
    block_layout: Option<BlockJacobianLayout>,
    residual_layout: Vec<usize>,
    evaluate_scratch: Vec<f64>,
    residual_block_residuals: Vec<f64>,
    jacobian_scratch: Vec<f64>,
    gradient_jacobian: Option<EvaluatorJacobian>,
    gradient_residuals: Option<Vec<f64>>,
    num_parameters: usize,
    num_effective_parameters: usize,
    num_residuals: usize,
    num_residual_evaluations: usize,
    num_jacobian_evaluations: usize,
}

impl ProgramEvaluator {
    /// The evaluator of `program`, whose parameter blocks carry their offsets
    /// (`Program::set_parameter_offsets_and_index`). `dense_jacobian` selects
    /// [`DenseSparseMatrix`] over [`BlockSparseMatrix`]; `num_eliminate_blocks` is the Schur
    /// E-block count (0 otherwise).
    pub fn new(
        program: Program,
        blocks: &ProblemBlocks,
        dense_jacobian: bool,
        num_eliminate_blocks: usize,
    ) -> Self {
        let block_layout = (!dense_jacobian)
            .then(|| BlockJacobianLayout::new(&program, blocks, num_eliminate_blocks));
        let mut residual_layout = Vec::with_capacity(program.residual_blocks.len());
        let mut position = 0;
        for &id in &program.residual_blocks {
            residual_layout.push(position);
            position += blocks.residual_block(id).num_residuals();
        }
        let jacobian_scratch = if dense_jacobian {
            vec![0.0; program.max_derivatives_per_residual_block(blocks)]
        } else {
            Vec::new()
        };
        Self {
            evaluate_scratch: vec![0.0; program.max_scratch_doubles_needed_for_evaluate(blocks)],
            residual_block_residuals: vec![0.0; program.max_residuals_per_residual_block(blocks)],
            jacobian_scratch,
            num_parameters: program.num_parameters(blocks),
            num_effective_parameters: program.num_effective_parameters(blocks),
            num_residuals: program.num_residuals(blocks),
            program,
            dense_jacobian,
            block_layout,
            residual_layout,
            gradient_jacobian: None,
            gradient_residuals: None,
            num_residual_evaluations: 0,
            num_jacobian_evaluations: 0,
        }
    }

    /// The program evaluated.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Ambient size of the state vector.
    pub fn num_parameters(&self) -> usize {
        self.num_parameters
    }

    /// Tangent size (Jacobian columns).
    pub fn num_effective_parameters(&self) -> usize {
        self.num_effective_parameters
    }

    /// Number of residuals (Jacobian rows).
    pub fn num_residuals(&self) -> usize {
        self.num_residuals
    }

    /// Evaluations asking for neither a gradient nor a Jacobian (Ceres times them as
    /// "Evaluator::Residual").
    pub fn num_residual_evaluations(&self) -> usize {
        self.num_residual_evaluations
    }

    /// Evaluations asking for a gradient or a Jacobian ("Evaluator::Jacobian").
    pub fn num_jacobian_evaluations(&self) -> usize {
        self.num_jacobian_evaluations
    }

    /// A zero Jacobian of the evaluator's kind.
    pub fn create_jacobian(&self, blocks: &ProblemBlocks) -> EvaluatorJacobian {
        match &self.block_layout {
            Some(layout) if !self.dense_jacobian => {
                EvaluatorJacobian::BlockSparse(layout.create_jacobian(&self.program, blocks))
            }
            _ => EvaluatorJacobian::Dense(DenseSparseMatrix::new(
                self.num_residuals,
                self.num_effective_parameters,
            )),
        }
    }

    /// Evaluates at `state`. Each of `residuals`, `gradient` and `jacobian` is optional.
    /// Without `apply_loss_function` the cost and residuals are the plain squared ones.
    /// Returns `Ok(None)` if a residual block fails or the cost is not finite.
    pub fn evaluate(
        &mut self,
        blocks: &mut ProblemBlocks,
        state: &[f64],
        apply_loss_function: bool,
        residuals: Option<&mut [f64]>,
        gradient: Option<&mut [f64]>,
        jacobian: Option<&mut EvaluatorJacobian>,
    ) -> Result<Option<f64>> {
        if gradient.is_some() || jacobian.is_some() {
            self.num_jacobian_evaluations += 1;
        } else {
            self.num_residual_evaluations += 1;
        }

        // The gradient is J'r, so it needs both; borrow the evaluator's own when the caller
        // did not ask for them.
        let mut own_jacobian = None;
        let mut own_residuals = None;
        if gradient.is_some() {
            if jacobian.is_none() {
                own_jacobian = Some(
                    self.gradient_jacobian
                        .take()
                        .unwrap_or_else(|| self.create_jacobian(blocks)),
                );
            }
            if residuals.is_none() {
                own_residuals = Some(
                    self.gradient_residuals
                        .take()
                        .unwrap_or_else(|| vec![0.0; self.num_residuals]),
                );
            }
        }
        let jacobian = jacobian.or(own_jacobian.as_mut());
        let residuals = residuals.or(own_residuals.as_deref_mut());
        let result = self.evaluate_into(
            blocks,
            state,
            apply_loss_function,
            residuals,
            gradient,
            jacobian,
        );
        if own_jacobian.is_some() {
            self.gradient_jacobian = own_jacobian;
        }
        if own_residuals.is_some() {
            self.gradient_residuals = own_residuals;
        }
        result
    }

    fn evaluate_into(
        &mut self,
        blocks: &mut ProblemBlocks,
        state: &[f64],
        apply_loss_function: bool,
        mut residuals: Option<&mut [f64]>,
        gradient: Option<&mut [f64]>,
        mut jacobian: Option<&mut EvaluatorJacobian>,
    ) -> Result<Option<f64>> {
        if !self.program.state_vector_to_parameter_blocks(blocks, state) {
            return Ok(None);
        }
        if let Some(residuals) = residuals.as_deref_mut() {
            residuals.fill(0.0);
        }
        if let Some(jacobian) = jacobian.as_deref_mut() {
            jacobian.as_sparse_matrix_mut().set_zero();
        }

        let blocks: &ProblemBlocks = blocks;
        let values = ParameterValues {
            user: &blocks.values,
            state: Some(state),
        };
        let mut cost = 0.0;
        for i in 0..self.program.residual_blocks.len() {
            let residual_block = blocks.residual_block(self.program.residual_blocks[i]);
            let num_residuals = residual_block.num_residuals();
            let n = residual_block.num_parameter_blocks();
            let block_residuals: Option<&mut [f64]> = match residuals.as_deref_mut() {
                Some(residuals) => {
                    let start = self.residual_layout[i];
                    Some(&mut residuals[start..start + num_residuals])
                }
                None if jacobian.is_some() => {
                    Some(&mut self.residual_block_residuals[..num_residuals])
                }
                None => None,
            };

            let mut stack_views: [Option<&mut [f64]>; MAX_STACK_BLOCKS] = Default::default();
            let mut heap_views: Vec<Option<&mut [f64]>> = Vec::new();
            let views: &mut [Option<&mut [f64]>] = if n <= MAX_STACK_BLOCKS {
                &mut stack_views[..n]
            } else {
                heap_views.resize_with(n, || None);
                &mut heap_views
            };

            let block_cost = match jacobian.as_deref_mut() {
                None => residual_block.evaluate(
                    &blocks.parameter_blocks,
                    values,
                    apply_loss_function,
                    block_residuals,
                    None,
                    &mut self.evaluate_scratch,
                )?,
                Some(EvaluatorJacobian::BlockSparse(matrix)) => {
                    // BlockEvaluatePreparer: each variable block's cell, in place.
                    let layout = self
                        .block_layout
                        .as_ref()
                        .map(|l| l.positions(i))
                        .unwrap_or(&[]);
                    prepare_block_sparse(
                        blocks,
                        residual_block.parameter_blocks(),
                        num_residuals,
                        layout,
                        matrix.values_mut(),
                        views,
                    );
                    residual_block.evaluate(
                        &blocks.parameter_blocks,
                        values,
                        apply_loss_function,
                        block_residuals,
                        Some(views),
                        &mut self.evaluate_scratch,
                    )?
                }
                Some(EvaluatorJacobian::Dense(matrix)) => {
                    // ScratchEvaluatePreparer, then DenseJacobianWriter::Write.
                    let mut rest: &mut [f64] = &mut self.jacobian_scratch;
                    for (slot, &pid) in views.iter_mut().zip(residual_block.parameter_blocks()) {
                        let block = blocks.parameter_block(pid);
                        *slot = if block.is_constant() {
                            None
                        } else {
                            let (head, tail) = std::mem::take(&mut rest)
                                .split_at_mut(num_residuals * block.tangent_size());
                            rest = tail;
                            Some(head)
                        };
                    }
                    let block_cost = residual_block.evaluate(
                        &blocks.parameter_blocks,
                        values,
                        apply_loss_function,
                        block_residuals,
                        Some(&mut *views),
                        &mut self.evaluate_scratch,
                    )?;
                    if block_cost.is_some() {
                        let row_offset = self.residual_layout[i];
                        for (view, &pid) in views.iter().zip(residual_block.parameter_blocks()) {
                            let Some(cell) = view.as_deref() else {
                                continue;
                            };
                            let block = blocks.parameter_block(pid);
                            let tangent_size = block.tangent_size();
                            for c in 0..tangent_size {
                                let column = matrix.matrix_mut().column_mut(block.delta_offset + c);
                                for r in 0..num_residuals {
                                    column[row_offset + r] = cell[r * tangent_size + c];
                                }
                            }
                        }
                    }
                    block_cost
                }
            };
            match block_cost {
                Some(block_cost) => cost += block_cost,
                None => return Ok(None),
            }
        }

        if let Some(gradient) = gradient {
            gradient.fill(0.0);
            if let (Some(jacobian), Some(residuals)) = (jacobian.as_deref(), residuals.as_deref()) {
                jacobian
                    .as_sparse_matrix()
                    .left_multiply_and_accumulate(residuals, gradient);
            }
        }

        // Ceres logs "Accumulated cost = ... is not a finite number. Evaluation failed."
        Ok(cost.is_finite().then_some(cost))
    }
}

// Points each variable parameter block's Jacobian view at its cell of `values` (cells of one
// residual block are disjoint, so they are split off in increasing position order).
fn prepare_block_sparse<'a>(
    blocks: &ProblemBlocks,
    parameter_blocks: &[super::problem::ParameterBlockId],
    num_residuals: usize,
    positions: &[usize],
    values: &'a mut [f64],
    views: &mut [Option<&'a mut [f64]>],
) {
    // (position, size, slot) of each variable block, by position.
    let mut stack_cells = [(0usize, 0usize, 0usize); MAX_STACK_BLOCKS];
    let mut heap_cells = Vec::new();
    let cells: &mut [(usize, usize, usize)] = if parameter_blocks.len() <= MAX_STACK_BLOCKS {
        &mut stack_cells[..positions.len()]
    } else {
        heap_cells.resize(positions.len(), (0, 0, 0));
        &mut heap_cells
    };
    let mut k = 0;
    for (j, (&pid, view)) in parameter_blocks.iter().zip(views.iter_mut()).enumerate() {
        *view = None;
        let block = blocks.parameter_block(pid);
        if !block.is_constant() {
            cells[k] = (positions[k], num_residuals * block.tangent_size(), j);
            k += 1;
        }
    }
    cells.sort_unstable_by_key(|&(position, _, _)| position);

    let mut rest = values;
    let mut consumed = 0;
    for &(position, size, j) in cells.iter() {
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(position - consumed);
        let (cell, tail) = tail.split_at_mut(size);
        views[j] = Some(cell);
        rest = tail;
        consumed = position + size;
    }
}
