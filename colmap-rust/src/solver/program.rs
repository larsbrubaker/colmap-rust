//! An ordered list of parameter blocks and residual blocks: the problem's own (every block),
//! the temporary one `Problem::evaluate` builds, and the reduced one the minimizer works on.
//! Port of Ceres Solver 2.2.0 `internal/ceres/program.cc` (BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md); colmap-sharp's `ColmapSharp/Solver/Program.cs` is the C# form.
//!
//! A program holds block ids; the blocks themselves live in the problem's [`ProblemBlocks`],
//! which every method takes. The state vector holds the program's blocks' values back to back
//! (ambient sizes), the delta vector their steps (tangent sizes);
//! [`Program::set_parameter_offsets_and_index`] lays both out in list order. Ceres moves each
//! variable block's `state_` pointer into the state vector during a solve; here a variable
//! block reads the state vector at its offset whenever an evaluation is given one
//! (`residual_block.rs`' `ParameterValues`), and the problem's arena otherwise, so
//! [`Program::parameter_blocks_to_state_vector`] reads the arena and
//! [`Program::copy_parameter_block_state_to_user_state`] writes a state vector back to it.
//! A program's residual blocks are numbered by their position in its list (Ceres also stores
//! that index in each residual block).

use std::collections::BTreeSet;

use super::parameter_block::IMPOSSIBLE_VALUE;
use super::problem::{ParameterBlockId, ProblemBlocks, ResidualBlockId};
use super::residual_block::ParameterValues;
use crate::util::stream_format::{format_double, format_exponential};
use crate::Result;

/// Port of `ceres::internal::Program`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Program {
    /// The parameter blocks, in state-vector order.
    pub parameter_blocks: Vec<ParameterBlockId>,
    /// The residual blocks, in residual-vector order.
    pub residual_blocks: Vec<ResidualBlockId>,
}

impl Program {
    /// An empty program.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of residual blocks.
    pub fn num_residual_blocks(&self) -> usize {
        self.residual_blocks.len()
    }

    /// Number of parameter blocks.
    pub fn num_parameter_blocks(&self) -> usize {
        self.parameter_blocks.len()
    }

    /// Total number of residuals.
    pub fn num_residuals(&self, blocks: &ProblemBlocks) -> usize {
        self.residual_blocks
            .iter()
            .map(|&id| blocks.residual_block(id).num_residuals())
            .sum()
    }

    /// Total ambient size of the parameter blocks.
    pub fn num_parameters(&self, blocks: &ProblemBlocks) -> usize {
        self.parameter_blocks
            .iter()
            .map(|&id| blocks.parameter_block(id).size())
            .sum()
    }

    /// Total tangent size of the parameter blocks.
    pub fn num_effective_parameters(&self, blocks: &ProblemBlocks) -> usize {
        self.parameter_blocks
            .iter()
            .map(|&id| blocks.parameter_block(id).tangent_size())
            .sum()
    }

    /// Largest scratch any residual block needs for its evaluation.
    pub fn max_scratch_doubles_needed_for_evaluate(&self, blocks: &ProblemBlocks) -> usize {
        self.residual_blocks
            .iter()
            .map(|&id| {
                blocks
                    .residual_block(id)
                    .num_scratch_doubles_for_evaluate(&blocks.parameter_blocks)
            })
            .max()
            .unwrap_or(0)
    }

    /// Most Jacobian entries (over its variable blocks) of one residual block.
    pub fn max_derivatives_per_residual_block(&self, blocks: &ProblemBlocks) -> usize {
        let mut max_derivatives = 0;
        for &id in &self.residual_blocks {
            let residual_block = blocks.residual_block(id);
            let mut derivatives = 0;
            for &pid in residual_block.parameter_blocks() {
                let parameter_block = blocks.parameter_block(pid);
                if !parameter_block.is_constant() {
                    derivatives += residual_block.num_residuals() * parameter_block.tangent_size();
                }
            }
            max_derivatives = max_derivatives.max(derivatives);
        }
        max_derivatives
    }

    /// Most parameter blocks in one residual block.
    pub fn max_parameters_per_residual_block(&self, blocks: &ProblemBlocks) -> usize {
        self.residual_blocks
            .iter()
            .map(|&id| blocks.residual_block(id).num_parameter_blocks())
            .max()
            .unwrap_or(0)
    }

    /// Most residuals in one residual block.
    pub fn max_residuals_per_residual_block(&self, blocks: &ProblemBlocks) -> usize {
        self.residual_blocks
            .iter()
            .map(|&id| blocks.residual_block(id).num_residuals())
            .max()
            .unwrap_or(0)
    }

    /// Ceres' `StateVectorToParameterBlocks`: the variable blocks now read `state`; refreshes
    /// their Plus Jacobians there. False if one cannot be computed.
    pub fn state_vector_to_parameter_blocks(
        &self,
        blocks: &mut ProblemBlocks,
        state: &[f64],
    ) -> bool {
        let mut offset = 0;
        for &id in &self.parameter_blocks {
            let block = &mut blocks.parameter_blocks[id.0];
            let size = block.size();
            if !block.is_constant() && !block.set_state(&state[offset..offset + size]) {
                return false;
            }
            offset += size;
        }
        true
    }

    /// Ceres' `ParameterBlocksToStateVector`: copies every block's values (the problem's
    /// arena, where they live between evaluations) into `state`.
    pub fn parameter_blocks_to_state_vector(&self, blocks: &ProblemBlocks, state: &mut [f64]) {
        let mut offset = 0;
        for &id in &self.parameter_blocks {
            let values = blocks.user_values(id);
            state[offset..offset + values.len()].copy_from_slice(values);
            offset += values.len();
        }
    }

    /// Ceres' `CopyParameterBlockStateToUserState`: writes the variable blocks' values in
    /// `state` back to the problem's arena (constant blocks already read it).
    pub fn copy_parameter_block_state_to_user_state(
        &self,
        blocks: &mut ProblemBlocks,
        state: &[f64],
    ) {
        let mut offset = 0;
        for &id in &self.parameter_blocks {
            let block = &blocks.parameter_blocks[id.0];
            let (user, size) = (block.user_offset(), block.size());
            if !block.is_constant() {
                blocks.values[user..user + size].copy_from_slice(&state[offset..offset + size]);
            }
            offset += size;
        }
    }

    /// Ceres' `SetParameterBlockStatePtrsToUserStatePtrs`: the variable blocks read the
    /// problem's arena again; refreshes their Plus Jacobians there.
    pub fn set_parameter_block_state_ptrs_to_user_state_ptrs(
        &self,
        blocks: &mut ProblemBlocks,
    ) -> bool {
        let ProblemBlocks {
            values,
            parameter_blocks,
            ..
        } = blocks;
        for &id in &self.parameter_blocks {
            let block = &mut parameter_blocks[id.0];
            let x = &values[block.user_offset()..block.user_offset() + block.size()];
            if !block.is_constant() && !block.set_state(x) {
                return false;
            }
        }
        true
    }

    /// `state_plus_delta = Plus(state, delta)`, block by block at their state and delta
    /// offsets. Each block writes only its own slice.
    pub fn plus(
        &self,
        blocks: &ProblemBlocks,
        state: &[f64],
        delta: &[f64],
        state_plus_delta: &mut [f64],
    ) -> bool {
        for &id in &self.parameter_blocks {
            let block = blocks.parameter_block(id);
            let (s, d) = (block.state_offset, block.delta_offset);
            let (size, tangent_size) = (block.size(), block.tangent_size());
            if !block.plus(
                &state[s..s + size],
                &delta[d..d + tangent_size],
                &mut state_plus_delta[s..s + size],
            ) {
                return false;
            }
        }
        true
    }

    /// Assigns each parameter block its index and state/delta offsets, in list order; blocks
    /// that residual blocks use but the program does not list get index -1.
    pub fn set_parameter_offsets_and_index(&self, blocks: &mut ProblemBlocks) {
        for &id in &self.residual_blocks {
            for &pid in blocks.residual_blocks[id.0].parameter_blocks() {
                blocks.parameter_blocks[pid.0].index = -1;
            }
        }
        let mut state_offset = 0;
        let mut delta_offset = 0;
        for (i, &id) in self.parameter_blocks.iter().enumerate() {
            let block = &mut blocks.parameter_blocks[id.0];
            block.index = i as isize;
            block.state_offset = state_offset;
            block.delta_offset = delta_offset;
            state_offset += block.size();
            delta_offset += block.tangent_size();
        }
    }

    /// Ceres' `IsValid`: every parameter block's index and offsets match its position.
    pub fn is_valid(&self, blocks: &ProblemBlocks) -> bool {
        let mut state_offset = 0;
        let mut delta_offset = 0;
        for (i, &id) in self.parameter_blocks.iter().enumerate() {
            let block = blocks.parameter_block(id);
            if block.index != i as isize
                || block.state_offset != state_offset
                || block.delta_offset != delta_offset
            {
                return false;
            }
            state_offset += block.size();
            delta_offset += block.tangent_size();
        }
        true
    }

    /// Ceres' `ParameterBlocksAreFinite`: every value is finite, else the message naming the
    /// first invalid one.
    pub fn parameter_blocks_are_finite(
        &self,
        blocks: &ProblemBlocks,
    ) -> std::result::Result<(), String> {
        for &id in &self.parameter_blocks {
            let values = blocks.user_values(id);
            if let Some(invalid_index) = values.iter().position(|v| !v.is_finite()) {
                return Err(format!(
                    "ParameterBlock: {} with size {} has at least one invalid value.\nFirst invalid value is at index: {invalid_index}.\nParameter block values: {}",
                    id.0,
                    values.len(),
                    array_to_string(values)
                ));
            }
        }
        Ok(())
    }

    /// Ceres' `IsBoundsConstrained`: some variable block has a finite bound.
    pub fn is_bounds_constrained(&self, blocks: &ProblemBlocks) -> bool {
        for &id in &self.parameter_blocks {
            let block = blocks.parameter_block(id);
            if block.is_constant() {
                continue;
            }
            for j in 0..block.size() {
                if block.lower_bound_for_parameter(j) > -f64::MAX
                    || block.upper_bound_for_parameter(j) < f64::MAX
                {
                    return true;
                }
            }
        }
        false
    }

    /// Ceres' `IsFeasible`: constant blocks must start inside their bounds (they cannot move),
    /// and variable blocks must have a non-empty box.
    pub fn is_feasible(&self, blocks: &ProblemBlocks) -> std::result::Result<(), String> {
        for &id in &self.parameter_blocks {
            let block = blocks.parameter_block(id);
            let parameters = blocks.user_values(id);
            let size = block.size();
            for j in 0..size {
                let lower_bound = block.lower_bound_for_parameter(j);
                let upper_bound = block.upper_bound_for_parameter(j);
                if block.is_constant() {
                    // Constant parameter blocks must start in the feasible region to
                    // ultimately produce a feasible solution, since Ceres cannot change them.
                    if parameters[j] < lower_bound || parameters[j] > upper_bound {
                        return Err(format!(
                            "ParameterBlock: {} with size {size} has at least one infeasible value.\nFirst infeasible value is at index: {j}.\nLower bound: {}, value: {}, upper bound: {}\nParameter block values: {}",
                            id.0,
                            format_exponential(lower_bound, 6),
                            format_exponential(parameters[j], 6),
                            format_exponential(upper_bound, 6),
                            array_to_string(parameters)
                        ));
                    }
                } else if lower_bound >= upper_bound {
                    // Variable parameter blocks must have non-empty feasible regions,
                    // otherwise there is no way to produce a feasible solution.
                    return Err(format!(
                        "ParameterBlock: {} with size {size} has at least one infeasible bound.\nFirst infeasible bound is at index: {j}.\nLower bound: {}, upper bound: {}\nParameter block values: {}",
                        id.0,
                        format_exponential(lower_bound, 6),
                        format_exponential(upper_bound, 6),
                        array_to_string(parameters)
                    ));
                }
            }
        }
        Ok(())
    }

    /// Ceres' `CreateReducedProgram`: a copy without constant parameter blocks and without the
    /// residual blocks that depend only on constant ones, whose cost (with the loss) is added
    /// to `fixed_cost`; the removed parameter blocks go to `removed_parameter_blocks`. `Ok(None)`
    /// with `error` set if such a residual block fails to evaluate.
    pub fn create_reduced_program(
        &self,
        blocks: &mut ProblemBlocks,
        removed_parameter_blocks: &mut Vec<ParameterBlockId>,
        fixed_cost: &mut f64,
        error: &mut String,
    ) -> Result<Option<Program>> {
        let mut reduced = self.clone();
        if !reduced.remove_fixed_blocks(blocks, removed_parameter_blocks, fixed_cost, error)? {
            return Ok(None);
        }
        reduced.set_parameter_offsets_and_index(blocks);
        Ok(Some(reduced))
    }

    fn remove_fixed_blocks(
        &mut self,
        blocks: &mut ProblemBlocks,
        removed_parameter_blocks: &mut Vec<ParameterBlockId>,
        fixed_cost: &mut f64,
        error: &mut String,
    ) -> Result<bool> {
        let mut scratch = vec![0.0; self.max_scratch_doubles_needed_for_evaluate(blocks)];
        *fixed_cost = 0.0;

        // Mark all the parameters as unused. Abuse the index member of the parameter blocks
        // for the marking.
        for &id in &self.parameter_blocks {
            blocks.parameter_blocks[id.0].index = -1;
        }

        // Filter out residual that have all-constant parameters, and mark all the parameter
        // blocks that appear in residuals.
        let mut num_active_residual_blocks = 0;
        for i in 0..self.residual_blocks.len() {
            let id = self.residual_blocks[i];
            let mut all_constant = true;
            for &pid in blocks.residual_blocks[id.0].parameter_blocks() {
                let parameter_block = &mut blocks.parameter_blocks[pid.0];
                if !parameter_block.is_constant() {
                    all_constant = false;
                    parameter_block.index = 1;
                }
            }

            if !all_constant {
                self.residual_blocks[num_active_residual_blocks] = id;
                num_active_residual_blocks += 1;
                continue;
            }

            // The residual is constant and will be removed, so its cost is added to the
            // variable fixed_cost.
            let values = ParameterValues {
                user: &blocks.values,
                state: None,
            };
            let residual_block = &blocks.residual_blocks[id.0];
            match residual_block.evaluate(
                &blocks.parameter_blocks,
                values,
                true,
                None,
                None,
                &mut scratch,
            )? {
                Some(cost) => *fixed_cost += cost,
                None => {
                    *error = format!(
                        "Evaluation of the residual {i} failed during removal of fixed residual blocks."
                    );
                    return Ok(false);
                }
            }
        }
        self.residual_blocks.truncate(num_active_residual_blocks);

        // Filter out unused or fixed parameter blocks.
        removed_parameter_blocks.clear();
        let mut num_active_parameter_blocks = 0;
        for i in 0..self.parameter_blocks.len() {
            let id = self.parameter_blocks[i];
            if blocks.parameter_blocks[id.0].index == -1 {
                removed_parameter_blocks.push(id);
            } else {
                self.parameter_blocks[num_active_parameter_blocks] = id;
                num_active_parameter_blocks += 1;
            }
        }
        self.parameter_blocks.truncate(num_active_parameter_blocks);

        if (self.num_residual_blocks() == 0) != (self.num_parameter_blocks() == 0) {
            *error = "Congratulations, you found a bug in Ceres. Please report it.".to_string();
            return Ok(false);
        }
        Ok(true)
    }

    /// Ceres' `IsParameterBlockSetIndependent`: no residual block touches two blocks of
    /// `independent_set`.
    pub fn is_parameter_block_set_independent(
        &self,
        blocks: &ProblemBlocks,
        independent_set: &BTreeSet<ParameterBlockId>,
    ) -> bool {
        // Loop over each residual block and ensure that no two parameter blocks in the same
        // residual block are part of the set, as that would violate the assumption that it is
        // an independent set in the Hessian matrix.
        self.residual_blocks.iter().all(|&id| {
            blocks
                .residual_block(id)
                .parameter_blocks()
                .iter()
                .filter(|pid| independent_set.contains(pid))
                .count()
                <= 1
        })
    }
}

// Ceres' AppendArrayToString: each value as "%12g ", unwritten ones as "Uninitialized ".
fn array_to_string(values: &[f64]) -> String {
    let mut text = String::new();
    for &value in values {
        if value == IMPOSSIBLE_VALUE {
            text.push_str("Uninitialized ");
        } else {
            text.push_str(&format!("{:>12} ", format_double(value, 6)));
        }
    }
    text
}
