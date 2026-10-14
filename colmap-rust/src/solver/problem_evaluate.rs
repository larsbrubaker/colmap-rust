//! `Problem::Evaluate`: cost, residuals, gradient and Jacobian of a [`Problem`] at its current
//! values, without solving. Port of Ceres Solver 2.2.0 `include/ceres/problem.h`
//! (`Problem::EvaluateOptions`), `include/ceres/crs_matrix.h` and
//! `internal/ceres/problem_impl.cc` (`ProblemImpl::Evaluate`) (BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md); colmap-sharp's `ColmapSharp/Solver/Problem.Evaluate.cs` is the C#
//! form.
//!
//! COLMAP calls it twice: covariance estimation takes the Jacobian over a chosen list of
//! parameter blocks, and view-graph calibration takes the residuals without the loss. The
//! evaluation runs on a temporary [`Program`] over the chosen residual and parameter blocks
//! (all by default, in problem order); variable blocks left out of an explicit list are held
//! constant for the call, so they get no Jacobian columns, and constant blocks in the list keep
//! their (zero) columns. Ceres writes the Jacobian with its compressed-row writer; this port
//! evaluates into the block-sparse Jacobian (`program_evaluator.rs`) and converts it row by
//! row with each row's columns ascending, which is the same matrix in the same layout.

use super::problem::{ParameterBlockId, Problem, ResidualBlockId};
use super::program::Program;
use super::program_evaluator::{EvaluatorJacobian, ProgramEvaluator};
use crate::{ColmapError, ErrorKind, Result};

/// Port of `ceres::Problem::EvaluateOptions` (without `num_threads`: the core is
/// single-threaded, and Ceres' result does not depend on it).
#[derive(Clone, Debug)]
pub struct EvaluateOptions {
    /// The parameter blocks, in Jacobian column order. Empty means every block of the problem
    /// in insertion order; otherwise the variable blocks not listed are held constant.
    pub parameter_blocks: Vec<ParameterBlockId>,
    /// The residual blocks, in residual order. Empty means every one, in insertion order.
    pub residual_blocks: Vec<ResidualBlockId>,
    /// Apply each residual block's loss function (default true).
    pub apply_loss_function: bool,
}

impl Default for EvaluateOptions {
    fn default() -> Self {
        Self {
            parameter_blocks: Vec::new(),
            residual_blocks: Vec::new(),
            apply_loss_function: true,
        }
    }
}

/// Port of `ceres::CRSMatrix`: a compressed row sparse matrix.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CrsMatrix {
    /// Number of rows.
    pub num_rows: usize,
    /// Number of columns.
    pub num_cols: usize,
    /// Row i's entries are `cols`/`values[rows[i]..rows[i + 1]]`; `num_rows + 1` long.
    pub rows: Vec<usize>,
    /// Column index of each entry.
    pub cols: Vec<usize>,
    /// Value of each entry.
    pub values: Vec<f64>,
}

impl Problem {
    /// `Problem::Evaluate` at the current values of the parameter blocks. Each of `residuals`,
    /// `gradient` and `jacobian` is optional and is resized and filled; the gradient and the
    /// Jacobian's columns follow the tangent sizes of the chosen parameter blocks. Returns
    /// `Ok(Some(cost))`, or `Ok(None)` if a residual block fails to evaluate or the cost is not
    /// finite (the Jacobian is then left untouched). An unknown block in `options` is an
    /// error, where Ceres `LOG(FATAL)`s.
    pub fn evaluate(
        &mut self,
        options: &EvaluateOptions,
        residuals: Option<&mut Vec<f64>>,
        gradient: Option<&mut Vec<f64>>,
        jacobian: Option<&mut CrsMatrix>,
    ) -> Result<Option<f64>> {
        for (i, &id) in options.parameter_blocks.iter().enumerate() {
            if !self.has_parameter_block(id) {
                return Err(ColmapError::new(
                    ErrorKind::InvalidArgument,
                    format!(
                        "No known parameter block for Problem::Evaluate::Options.parameter_blocks[{i}] = {}",
                        id.0
                    ),
                ));
            }
        }
        for &id in &options.residual_blocks {
            self.get_cost_function_for_residual_block(id)?;
        }

        let (problem_program, blocks) = self.program_and_blocks_mut();

        // If the user supplied residual blocks, then use them, otherwise take the residual
        // blocks from the underlying program.
        let mut program = Program::new();
        program.residual_blocks = if options.residual_blocks.is_empty() {
            problem_program.residual_blocks.clone()
        } else {
            options.residual_blocks.clone()
        };

        let mut variable_parameter_blocks = Vec::new();
        if options.parameter_blocks.is_empty() {
            // The user did not provide any parameter blocks, so default to using all the
            // parameter blocks in the order that they are in the underlying program object.
            program.parameter_blocks = problem_program.parameter_blocks.clone();
        } else {
            program.parameter_blocks = options.parameter_blocks.clone();

            // The user may have only supplied a subset of parameter blocks, so the variable
            // ones not supplied are made constant during the evaluation (so that they get no
            // Jacobian columns) and variable again afterwards.
            let mut included = options.parameter_blocks.clone();
            included.sort_unstable();
            for &id in &problem_program.parameter_blocks {
                let block = &mut blocks.parameter_blocks[id.0];
                if included.binary_search(&id).is_err() && !block.is_constant() {
                    variable_parameter_blocks.push(id);
                    block.set_constant(true);
                }
            }
        }

        // Setup the parameter indices and offsets before an evaluator can be constructed and
        // used.
        program.set_parameter_offsets_and_index(blocks);
        let mut evaluator = ProgramEvaluator::new(program, blocks, false, 0);
        let (mut residuals, mut gradient) = (residuals, gradient);
        if let Some(residuals) = residuals.as_deref_mut() {
            residuals.clear();
            residuals.resize(evaluator.num_residuals(), 0.0);
        }
        if let Some(gradient) = gradient.as_deref_mut() {
            gradient.clear();
            gradient.resize(evaluator.num_effective_parameters(), 0.0);
        }
        let mut block_jacobian = jacobian
            .is_some()
            .then(|| evaluator.create_jacobian(blocks));

        // The values live in the problem's arena; copy them into a state vector, since the
        // evaluator needs its input as such.
        let program = evaluator.program().clone();
        program.set_parameter_block_state_ptrs_to_user_state_ptrs(blocks);
        let mut parameters = vec![0.0; program.num_parameters(blocks)];
        program.parameter_blocks_to_state_vector(blocks, &mut parameters);

        let status = evaluator.evaluate(
            blocks,
            &parameters,
            options.apply_loss_function,
            residuals.map(|r| r.as_mut_slice()),
            gradient.map(|g| g.as_mut_slice()),
            block_jacobian.as_mut(),
        );

        // Make the parameter blocks that were temporarily marked constant, variable again.
        for id in variable_parameter_blocks {
            blocks.parameter_blocks[id.0].set_constant(false);
        }
        problem_program.set_parameter_block_state_ptrs_to_user_state_ptrs(blocks);
        problem_program.set_parameter_offsets_and_index(blocks);

        let cost = status?;
        if cost.is_some() {
            if let (Some(jacobian), Some(EvaluatorJacobian::BlockSparse(matrix))) =
                (jacobian, block_jacobian.as_ref())
            {
                to_crs_matrix(matrix, jacobian);
            }
        }
        Ok(cost)
    }
}

// BlockSparseMatrix -> CRSMatrix: rows in order, each row's cells by increasing column block
// (the structure keeps them sorted) and each cell's columns in order.
fn to_crs_matrix(matrix: &super::BlockSparseMatrix, crs: &mut CrsMatrix) {
    use super::SparseMatrix;
    crs.num_rows = matrix.num_rows();
    crs.num_cols = matrix.num_cols();
    crs.rows.clear();
    crs.cols.clear();
    crs.values.clear();
    crs.rows.push(0);
    let structure = matrix.structure();
    let values = matrix.values();
    for row in &structure.rows {
        for r in 0..row.block.size {
            for cell in &row.cells {
                let col = structure.cols[cell.block_id];
                let start = cell.position + r * col.size;
                for c in 0..col.size {
                    crs.cols.push(col.position + c);
                    crs.values.push(values[start + c]);
                }
            }
            crs.rows.push(crs.cols.len());
        }
    }
}
