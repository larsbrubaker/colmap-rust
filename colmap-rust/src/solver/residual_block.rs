//! A residual block: a [`CostFunction`] and an optional [`LossFunction`] over some parameter
//! blocks, evaluated the way the solver needs it. Port of colmap-sharp's
//! `ColmapSharp/Solver/ResidualBlock.cs` (our MIT code), from Ceres Solver 2.2.0
//! `internal/ceres/residual_block.cc` (BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//!
//! [`ResidualBlock::evaluate`] gives the residuals, the cost 1/2 rho(|f|^2), and Jacobians in
//! the tangent space (the cost function's ambient Jacobian times the manifold's Plus
//! Jacobian), all corrected for the robust loss by [`Corrector`] (`corrector.rs`). The
//! parameter blocks are ids into the problem's block list (`problem.rs`); their values come
//! from [`ParameterValues`]. Evaluation allocates nothing for up to [`MAX_STACK_BLOCKS`]
//! parameter blocks (every COLMAP cost function has fewer); the caller supplies the scratch.

use std::sync::Arc;

use super::autodiff_cost_function::CostFunction;
use super::corrector::Corrector;
use super::loss_function::LossFunction;
use super::parameter_block::{invalidate_array, is_array_valid, ParameterBlock};
use super::problem::ParameterBlockId;
use crate::Result;

/// Residual blocks with at most this many parameter blocks evaluate without allocating.
pub const MAX_STACK_BLOCKS: usize = 16;

/// Where an evaluation reads parameter values from: a variable block reads `state` at its
/// [`ParameterBlock::state_offset`] when there is a state vector, every other block reads the
/// problem's arena at its [`ParameterBlock::user_offset`] (Ceres' `state()` pointer).
#[derive(Clone, Copy, Debug)]
pub struct ParameterValues<'a> {
    /// The problem's value arena.
    pub user: &'a [f64],
    /// The program's state vector, if the variable blocks have been moved there.
    pub state: Option<&'a [f64]>,
}

impl<'a> ParameterValues<'a> {
    /// The current values of `block`.
    pub fn block(&self, block: &ParameterBlock) -> &'a [f64] {
        match self.state {
            Some(state) if !block.is_constant() => {
                &state[block.state_offset..block.state_offset + block.size()]
            }
            _ => &self.user[block.user_offset()..block.user_offset() + block.size()],
        }
    }
}

/// Port of `ceres::internal::ResidualBlock`.
#[derive(Clone)]
pub struct ResidualBlock {
    cost_function: Arc<dyn CostFunction>,
    loss_function: Option<Arc<dyn LossFunction>>,
    parameter_blocks: Vec<ParameterBlockId>,
}

impl std::fmt::Debug for ResidualBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidualBlock")
            .field("num_residuals", &self.num_residuals())
            .field("has_loss", &self.loss_function.is_some())
            .field("parameter_blocks", &self.parameter_blocks)
            .finish()
    }
}

impl ResidualBlock {
    /// A residual block over the parameter blocks with the given ids, in the cost function's
    /// order.
    pub fn new(
        cost_function: Arc<dyn CostFunction>,
        loss_function: Option<Arc<dyn LossFunction>>,
        parameter_blocks: Vec<ParameterBlockId>,
    ) -> Self {
        Self {
            cost_function,
            loss_function,
            parameter_blocks,
        }
    }

    /// The cost function.
    pub fn cost_function(&self) -> &Arc<dyn CostFunction> {
        &self.cost_function
    }

    /// The robust loss, or `None` for plain least squares.
    pub fn loss_function(&self) -> Option<&Arc<dyn LossFunction>> {
        self.loss_function.as_ref()
    }

    /// The parameter block ids, in the cost function's order.
    pub fn parameter_blocks(&self) -> &[ParameterBlockId] {
        &self.parameter_blocks
    }

    /// Number of residuals.
    pub fn num_residuals(&self) -> usize {
        self.cost_function.num_residuals()
    }

    /// Number of parameter blocks.
    pub fn num_parameter_blocks(&self) -> usize {
        self.parameter_blocks.len()
    }

    /// Scratch [`Self::evaluate`] needs: room for the residuals when the caller does not want
    /// them, plus an ambient Jacobian for every block with a manifold.
    pub fn num_scratch_doubles_for_evaluate(&self, blocks: &[ParameterBlock]) -> usize {
        let mut scratch_doubles = 1;
        for &id in &self.parameter_blocks {
            if blocks[id.0].plus_jacobian().is_some() {
                scratch_doubles += blocks[id.0].size();
            }
        }
        scratch_doubles * self.num_residuals()
    }

    /// Evaluates the block. `residuals` is `None` when the caller only wants the cost;
    /// `jacobians` is `None` for no Jacobians, otherwise one entry per parameter block: `None`
    /// to skip it, else room for a row-major `num_residuals x tangent_size` matrix. `scratch`
    /// holds at least [`Self::num_scratch_doubles_for_evaluate`] values. Returns `Ok(None)` if
    /// the cost function fails or leaves a non-finite or unwritten value, `Ok(Some(cost))`
    /// otherwise; `Err` is the loss correction's check (Ceres CHECK-fails there).
    pub fn evaluate(
        &self,
        blocks: &[ParameterBlock],
        values: ParameterValues<'_>,
        apply_loss_function: bool,
        residuals: Option<&mut [f64]>,
        mut jacobians: Option<&mut [Option<&mut [f64]>]>,
        scratch: &mut [f64],
    ) -> Result<Option<f64>> {
        let n = self.parameter_blocks.len();
        let num_residuals = self.num_residuals();

        let mut stack_views: [&[f64]; MAX_STACK_BLOCKS] = [&[]; MAX_STACK_BLOCKS];
        let mut heap_views: Vec<&[f64]> = Vec::new();
        let views: &mut [&[f64]] = if n <= MAX_STACK_BLOCKS {
            &mut stack_views[..n]
        } else {
            heap_views.resize(n, &[]);
            &mut heap_views
        };
        for (view, &id) in views.iter_mut().zip(&self.parameter_blocks) {
            *view = values.block(&blocks[id.0]);
        }

        // Blocks with a manifold get their ambient Jacobian in scratch; the product with the
        // Plus Jacobian then lands in the caller's tangent-space view. The residuals go after
        // those Jacobians when the caller does not want them.
        let mut ambient_total = 0;
        if let Some(jacobians) = jacobians.as_deref() {
            for (jacobian, &id) in jacobians.iter().zip(&self.parameter_blocks) {
                if jacobian.is_some() && blocks[id.0].plus_jacobian().is_some() {
                    ambient_total += num_residuals * blocks[id.0].size();
                }
            }
        }
        let (ambient_scratch, residual_scratch) = scratch.split_at_mut(ambient_total);
        let outputting_residuals = residuals.is_some();
        let residuals: &mut [f64] = match residuals {
            Some(residuals) => &mut residuals[..num_residuals],
            None => &mut residual_scratch[..num_residuals],
        };

        // InvalidateEvaluation: a value the cost function leaves unwritten is caught below.
        invalidate_array(residuals);
        let evaluated = {
            let mut stack_eval: [Option<&mut [f64]>; MAX_STACK_BLOCKS] = Default::default();
            let mut heap_eval: Vec<Option<&mut [f64]>> = Vec::new();
            let mut eval: Option<&mut [Option<&mut [f64]>]> = None;
            if let Some(jacobians) = jacobians.as_deref_mut() {
                let slots: &mut [Option<&mut [f64]>] = if n <= MAX_STACK_BLOCKS {
                    &mut stack_eval[..n]
                } else {
                    heap_eval.resize_with(n, || None);
                    &mut heap_eval
                };
                let mut rest: &mut [f64] = &mut *ambient_scratch;
                for ((slot, out), &id) in slots
                    .iter_mut()
                    .zip(jacobians.iter_mut())
                    .zip(&self.parameter_blocks)
                {
                    let block = &blocks[id.0];
                    *slot = match out {
                        None => None,
                        Some(_) if block.plus_jacobian().is_some() => {
                            let (head, tail) = std::mem::take(&mut rest)
                                .split_at_mut(num_residuals * block.size());
                            rest = tail;
                            Some(head)
                        }
                        Some(out) => Some(&mut out[..num_residuals * block.tangent_size()]),
                    };
                    if let Some(slot) = slot.as_deref_mut() {
                        invalidate_array(slot);
                    }
                }
                eval = Some(slots);
            }

            let ok = self
                .cost_function
                .evaluate(views, residuals, eval.as_deref_mut());
            ok && is_array_valid(residuals)
                && eval
                    .is_none_or(|eval| eval.iter().all(|j| j.as_deref().is_none_or(is_array_valid)))
        };
        if !evaluated {
            return Ok(None);
        }

        let mut squared_norm = 0.0;
        for &r in residuals.iter() {
            squared_norm += r * r;
        }

        if let Some(jacobians) = jacobians.as_deref_mut() {
            let mut offset = 0;
            for (out, &id) in jacobians.iter_mut().zip(&self.parameter_blocks) {
                let block = &blocks[id.0];
                if let (Some(out), Some(plus)) = (out.as_deref_mut(), block.plus_jacobian()) {
                    let ambient_size = num_residuals * block.size();
                    multiply_by_plus_jacobian(
                        &ambient_scratch[offset..offset + ambient_size],
                        num_residuals,
                        block,
                        plus,
                        out,
                    );
                    offset += ambient_size;
                }
            }
        }

        let loss = match &self.loss_function {
            Some(loss) if apply_loss_function => loss,
            _ => return Ok(Some(0.5 * squared_norm)),
        };

        let mut rho = [0.0; 3];
        loss.evaluate(squared_norm, &mut rho);
        let cost = 0.5 * rho[0];
        if jacobians.is_none() && !outputting_residuals {
            return Ok(Some(cost));
        }

        let corrector = Corrector::new(squared_norm, &rho)?;
        if let Some(jacobians) = jacobians {
            for (out, &id) in jacobians.iter_mut().zip(&self.parameter_blocks) {
                if let Some(out) = out.as_deref_mut() {
                    let tangent_size = blocks[id.0].tangent_size();
                    corrector.correct_jacobian(num_residuals, tangent_size, residuals, out);
                }
            }
        }
        if outputting_residuals {
            corrector.correct_residuals(residuals);
        }
        Ok(Some(cost))
    }
}

// tangent (r x t) = ambient (r x s) * plus_jacobian (s x t), all row-major; each entry a
// left-to-right sum over the ambient index.
fn multiply_by_plus_jacobian(
    ambient: &[f64],
    num_residuals: usize,
    block: &ParameterBlock,
    plus: &[f64],
    tangent: &mut [f64],
) {
    let size = block.size();
    let tangent_size = block.tangent_size();
    for r in 0..num_residuals {
        for c in 0..tangent_size {
            let mut sum = 0.0;
            for k in 0..size {
                sum += ambient[r * size + k] * plus[k * tangent_size + c];
            }
            tangent[r * tangent_size + c] = sum;
        }
    }
}
