//! The nonlinear least-squares problem: parameter blocks and the residual blocks over them.
//! Port of Ceres Solver 2.2.0 `include/ceres/problem.h` and `internal/ceres/problem_impl.cc`
//! (BSD-3-Clause, see THIRD_PARTY_NOTICES.md); colmap-sharp's `ColmapSharp/Solver/Problem.cs` is
//! the C# form.
//!
//! Design (an API bundle adjustment and every refinement build on; divergence 306):
//! - Ceres keys a parameter block by the user's `double*` and reads and writes the user's memory.
//!   Here the problem owns the values: [`Problem::add_parameter_block`] copies them into one
//!   arena (`ProblemBlocks::values`, one allocation for every block) and returns a
//!   [`ParameterBlockId`]; the caller keeps the ids it needs (bundle adjustment maps image, point
//!   and camera ids to them, where COLMAP asks `HasParameterBlock(ptr)`), reads the solution
//!   back with [`Problem::parameter_block`] and writes it into its own objects. Blocks cannot
//!   overlap, so Ceres' aliasing and same-pointer-different-size checks have nothing to check.
//!   A `Problem` borrowing the caller's arrays instead would need every block borrowed mutably
//!   at once for the problem's lifetime (impossible for COLMAP's shared camera blocks without
//!   ids anyway) and would make every solver type generic over that lifetime.
//! - Cost functions, losses and manifolds are `Arc`s: a problem may share one among many
//!   blocks, and between problems, as Ceres' `DO_NOT_TAKE_OWNERSHIP` allows (the only problem
//!   option COLMAP sets). Ceres' other `Problem::Options` (fast removal, safety checks, the
//!   context, the evaluation callback) have no use in COLMAP and are not ported, nor are
//!   `RemoveResidualBlock` / `RemoveParameterBlock` / `EvaluateResidualBlock`, which COLMAP
//!   never calls.
//! - Where Ceres `LOG(FATAL)`s or CHECK-fails (an unknown block, a size mismatch, duplicate
//!   blocks), the port returns the error with Ceres' message.
//!
//! The problem's [`Program`] (`program.rs`) lists every block in insertion order;
//! `problem_evaluate.rs` holds [`Problem::evaluate`].

use std::sync::Arc;

use super::autodiff_cost_function::CostFunction;
use super::loss_function::LossFunction;
use super::manifold::Manifold;
use super::parameter_block::ParameterBlock;
use super::program::Program;
use super::residual_block::ResidualBlock;
use crate::{ColmapError, ErrorKind, Result};

/// A parameter block of a [`Problem`]: its position in the problem's block list (blocks are
/// numbered in the order they are added). Ceres names a block by its `double*`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParameterBlockId(pub usize);

/// A residual block of a [`Problem`] (`ceres::ResidualBlockId`): its position in the
/// problem's residual block list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResidualBlockId(pub usize);

/// What a [`Problem`] owns: the parameter values, the parameter blocks and the residual
/// blocks, indexed by their ids. [`Program`]s and the evaluator work on it.
#[derive(Clone, Debug, Default)]
pub struct ProblemBlocks {
    /// Every parameter block's values, back to back; a block's live at its
    /// [`ParameterBlock::user_offset`] (Ceres' user state).
    pub values: Vec<f64>,
    /// The parameter blocks, by [`ParameterBlockId`].
    pub parameter_blocks: Vec<ParameterBlock>,
    /// The residual blocks, by [`ResidualBlockId`].
    pub residual_blocks: Vec<ResidualBlock>,
}

impl ProblemBlocks {
    /// The parameter block `id`.
    pub fn parameter_block(&self, id: ParameterBlockId) -> &ParameterBlock {
        &self.parameter_blocks[id.0]
    }

    /// The residual block `id`.
    pub fn residual_block(&self, id: ResidualBlockId) -> &ResidualBlock {
        &self.residual_blocks[id.0]
    }

    /// The values of the parameter block `id` in the arena.
    pub fn user_values(&self, id: ParameterBlockId) -> &[f64] {
        let block = &self.parameter_blocks[id.0];
        &self.values[block.user_offset()..block.user_offset() + block.size()]
    }
}

/// Port of `ceres::Problem`.
#[derive(Clone, Debug, Default)]
pub struct Problem {
    blocks: ProblemBlocks,
    program: Program,
}

impl Problem {
    /// An empty problem.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a parameter block holding a copy of `values` (Ceres' `AddParameterBlock(values,
    /// size)`).
    pub fn add_parameter_block(&mut self, values: &[f64]) -> ParameterBlockId {
        let id = ParameterBlockId(self.blocks.parameter_blocks.len());
        let offset = self.blocks.values.len();
        self.blocks.values.extend_from_slice(values);
        // The index keeps the block in sync with its position in the program's list.
        self.blocks
            .parameter_blocks
            .push(ParameterBlock::new(offset, values.len(), id.0 as isize));
        self.program.parameter_blocks.push(id);
        id
    }

    /// Adds a parameter block holding a copy of `values` on `manifold` (Ceres'
    /// `AddParameterBlock(values, size, manifold)`).
    pub fn add_parameter_block_with_manifold(
        &mut self,
        values: &[f64],
        manifold: Arc<dyn Manifold>,
    ) -> Result<ParameterBlockId> {
        let id = self.add_parameter_block(values);
        self.set_manifold(id, Some(manifold))?;
        Ok(id)
    }

    /// Adds a residual block over `parameter_blocks`, in the cost function's order; `None`
    /// for `loss_function` means plain least squares.
    pub fn add_residual_block(
        &mut self,
        cost_function: Arc<dyn CostFunction>,
        loss_function: Option<Arc<dyn LossFunction>>,
        parameter_blocks: &[ParameterBlockId],
    ) -> Result<ResidualBlockId> {
        let sizes = cost_function.parameter_block_sizes();
        if parameter_blocks.len() != sizes.len() {
            return Err(invalid(format!(
                "Check failed: num_parameter_blocks == cost_function->parameter_block_sizes().size() ({} vs. {}) Number of blocks input is different than the number of blocks that the cost function expects.",
                parameter_blocks.len(),
                sizes.len()
            )));
        }

        // Check for duplicate parameter blocks.
        let mut sorted = parameter_blocks.to_vec();
        sorted.sort_unstable();
        if sorted.windows(2).any(|w| w[0] == w[1]) {
            let blocks: String = parameter_blocks
                .iter()
                .map(|b| format!(" {} ", b.0))
                .collect();
            return Err(invalid(format!(
                "Duplicate parameter blocks in a residual parameter are not allowed. Parameter block pointers: [{blocks}]"
            )));
        }

        for &id in parameter_blocks {
            self.find(id, "be used in a residual block")?;
        }

        // Check that the block sizes match the block sizes expected by the cost_function.
        for (i, (&id, &size)) in parameter_blocks.iter().zip(sizes).enumerate() {
            let block_size = self.blocks.parameter_blocks[id.0].size();
            if size != block_size {
                return Err(invalid(format!(
                    "Check failed: cost_function->parameter_block_sizes()[i] == parameter_block_ptrs[i]->Size() ({size} vs. {block_size}) The cost function expects parameter block {i} of size {size} but was given a block of size {block_size}"
                )));
            }
        }

        let id = ResidualBlockId(self.blocks.residual_blocks.len());
        self.blocks.residual_blocks.push(ResidualBlock::new(
            cost_function,
            loss_function,
            parameter_blocks.to_vec(),
        ));
        self.program.residual_blocks.push(id);
        Ok(id)
    }

    /// The current values of a parameter block (after a solve, the solution).
    pub fn parameter_block(&self, id: ParameterBlockId) -> Result<&[f64]> {
        self.find(id, "be read")?;
        Ok(self.blocks.user_values(id))
    }

    /// The values of a parameter block, writable.
    pub fn parameter_block_mut(&mut self, id: ParameterBlockId) -> Result<&mut [f64]> {
        let block = self.find(id, "be written")?;
        let (offset, size) = (block.user_offset(), block.size());
        Ok(&mut self.blocks.values[offset..offset + size])
    }

    /// Holds the block constant during optimization.
    pub fn set_parameter_block_constant(&mut self, id: ParameterBlockId) -> Result<()> {
        self.find_mut(id, "it can be set constant")?
            .set_constant(true);
        Ok(())
    }

    /// Lets a constant block vary again.
    pub fn set_parameter_block_variable(&mut self, id: ParameterBlockId) -> Result<()> {
        self.find_mut(id, "it can be set varying")?
            .set_constant(false);
        Ok(())
    }

    /// True if the block is held constant (set constant, or a zero-dimensional tangent space).
    pub fn is_parameter_block_constant(&self, id: ParameterBlockId) -> Result<bool> {
        Ok(self.find(id, "it can be queried")?.is_constant())
    }

    /// Sets (or with `None`, clears) the block's manifold. Ceres CHECK-fails when the manifold's
    /// ambient size differs from the block's or its Plus Jacobian cannot be computed at the
    /// block's values; the port returns the error.
    pub fn set_manifold(
        &mut self,
        id: ParameterBlockId,
        manifold: Option<Arc<dyn Manifold>>,
    ) -> Result<()> {
        self.find(id, "you can set its manifold")?;
        let ProblemBlocks {
            values,
            parameter_blocks,
            ..
        } = &mut self.blocks;
        let block = &mut parameter_blocks[id.0];
        let x = &values[block.user_offset()..block.user_offset() + block.size()];
        block.set_manifold(manifold, x)
    }

    /// The block's manifold, or `None`.
    pub fn get_manifold(&self, id: ParameterBlockId) -> Result<Option<&Arc<dyn Manifold>>> {
        Ok(self.find(id, "you can get its manifold")?.manifold())
    }

    /// True if the block has a manifold.
    pub fn has_manifold(&self, id: ParameterBlockId) -> Result<bool> {
        Ok(self.get_manifold(id)?.is_some())
    }

    /// Sets the lower bound of coordinate `index` (`-f64::MAX` means none).
    pub fn set_parameter_lower_bound(
        &mut self,
        id: ParameterBlockId,
        index: usize,
        lower_bound: f64,
    ) -> Result<()> {
        self.find_mut(id, "you can set a lower bound on one of its components")?
            .set_lower_bound(index, lower_bound)
    }

    /// Sets the upper bound of coordinate `index` (`f64::MAX` means none).
    pub fn set_parameter_upper_bound(
        &mut self,
        id: ParameterBlockId,
        index: usize,
        upper_bound: f64,
    ) -> Result<()> {
        self.find_mut(id, "you can set an upper bound on one of its components")?
            .set_upper_bound(index, upper_bound)
    }

    /// The lower bound of coordinate `index` (`-f64::MAX` if none).
    pub fn get_parameter_lower_bound(&self, id: ParameterBlockId, index: usize) -> Result<f64> {
        let block = self.find(id, "you can get the lower bound on one of its components")?;
        crate::check_lt!(index, block.size());
        Ok(block.lower_bound_for_parameter(index))
    }

    /// The upper bound of coordinate `index` (`f64::MAX` if none).
    pub fn get_parameter_upper_bound(&self, id: ParameterBlockId, index: usize) -> Result<f64> {
        let block = self.find(id, "you can get the upper bound on one of its components")?;
        crate::check_lt!(index, block.size());
        Ok(block.upper_bound_for_parameter(index))
    }

    /// Ambient size of the block.
    pub fn parameter_block_size(&self, id: ParameterBlockId) -> Result<usize> {
        Ok(self.find(id, "you can get its size")?.size())
    }

    /// Tangent size of the block.
    pub fn parameter_block_tangent_size(&self, id: ParameterBlockId) -> Result<usize> {
        Ok(self
            .find(id, "you can get its tangent size")?
            .tangent_size())
    }

    /// True if `id` names a block of this problem.
    pub fn has_parameter_block(&self, id: ParameterBlockId) -> bool {
        id.0 < self.blocks.parameter_blocks.len()
    }

    /// Every parameter block, in the order they were added (Ceres lists them in address
    /// order; divergence 306).
    pub fn get_parameter_blocks(&self) -> Vec<ParameterBlockId> {
        self.program.parameter_blocks.clone()
    }

    /// Every residual block, in the order they were added.
    pub fn get_residual_blocks(&self) -> Vec<ResidualBlockId> {
        self.program.residual_blocks.clone()
    }

    /// The parameter blocks of a residual block, in its cost function's order.
    pub fn get_parameter_blocks_for_residual_block(
        &self,
        id: ResidualBlockId,
    ) -> Result<&[ParameterBlockId]> {
        Ok(self.find_residual(id)?.parameter_blocks())
    }

    /// The cost function of a residual block.
    pub fn get_cost_function_for_residual_block(
        &self,
        id: ResidualBlockId,
    ) -> Result<&Arc<dyn CostFunction>> {
        Ok(self.find_residual(id)?.cost_function())
    }

    /// The loss function of a residual block, or `None`.
    pub fn get_loss_function_for_residual_block(
        &self,
        id: ResidualBlockId,
    ) -> Result<Option<&Arc<dyn LossFunction>>> {
        Ok(self.find_residual(id)?.loss_function())
    }

    /// Number of parameter blocks.
    pub fn num_parameter_blocks(&self) -> usize {
        self.program.parameter_blocks.len()
    }

    /// Total number of scalar parameters (ambient sizes).
    pub fn num_parameters(&self) -> usize {
        self.program.num_parameters(&self.blocks)
    }

    /// Number of residual blocks.
    pub fn num_residual_blocks(&self) -> usize {
        self.program.residual_blocks.len()
    }

    /// Total number of residuals.
    pub fn num_residuals(&self) -> usize {
        self.program.num_residuals(&self.blocks)
    }

    /// The problem's program: every block, in insertion order unless reordered.
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// The blocks and values the problem owns.
    pub fn blocks(&self) -> &ProblemBlocks {
        &self.blocks
    }

    /// The program and the blocks together, writable (for the solver's preprocessing).
    pub fn program_and_blocks_mut(&mut self) -> (&mut Program, &mut ProblemBlocks) {
        (&mut self.program, &mut self.blocks)
    }

    fn find(&self, id: ParameterBlockId, action: &str) -> Result<&ParameterBlock> {
        self.blocks
            .parameter_blocks
            .get(id.0)
            .ok_or_else(|| not_found(id, action))
    }

    fn find_mut(&mut self, id: ParameterBlockId, action: &str) -> Result<&mut ParameterBlock> {
        self.blocks
            .parameter_blocks
            .get_mut(id.0)
            .ok_or_else(|| not_found(id, action))
    }

    fn find_residual(&self, id: ResidualBlockId) -> Result<&ResidualBlock> {
        self.blocks.residual_blocks.get(id.0).ok_or_else(|| {
            invalid(format!(
                "Residual block not found: {}. You must add the residual block to the problem first.",
                id.0
            ))
        })
    }
}

fn invalid(message: String) -> ColmapError {
    ColmapError::new(ErrorKind::InvalidArgument, message)
}

// Ceres' LOG(FATAL) text, with the block id where Ceres prints the pointer.
fn not_found(id: ParameterBlockId, action: &str) -> ColmapError {
    invalid(format!(
        "Parameter block not found: {}. You must add the parameter block to the problem before {action}.",
        id.0
    ))
}
