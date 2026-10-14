//! Port of COLMAP's `RotationAveragingSolver` (`rotation_averaging_impl.h/.cc`): solves a
//! [`RotationAveragingProblem`] by L1 regression (ADMM,
//! [`crate::optim::least_absolute_deviations`]) followed by iteratively reweighted least
//! squares over the normal equations ([`crate::optim::sparse_cholesky`]). Port of
//! colmap-sharp's `Estimators/RotationAveragingSolver.cs`.
//!
//! Translation notes:
//! - COLMAP's L1 phase doubles `l1_solver_options.max_num_iterations` (capped at 100) after
//!   each non-converged iteration; its `LeastAbsoluteDeviationSolver` holds a reference to
//!   those options, so the next solve uses the raised cap. The Rust solver owns a copy, so
//!   the cap is raised through `set_max_num_iterations`.
//! - `SupernodalCholmodLLT` selects the LLT-with-LDLT-fallback solver, as does the IRLS phase
//!   (docs/CPP_DIVERGENCES.md entry 33).
//! - `LOG(ERROR)` paths return `Ok(false)`; `VLOG` diagnostics are not emitted.

use super::RotationWeightType;
use super::{RotationAveragingProblem, RotationConstraint, RotationEstimatorOptions};
use crate::linalg::{SparseMatrixCsc, VectorXd};
use crate::math::fns;
use crate::math::utils::deg_to_rad;
use crate::optim::least_absolute_deviations::{
    LeastAbsoluteDeviationSolver, LeastAbsoluteDeviationSolverOptions,
    LeastAbsoluteDeviationSolverType,
};
use crate::optim::SparseCholeskyWithFallbackSolver;
use crate::Result;

/// Solves the rotation averaging problem using L1 regression followed by IRLS. Port of
/// `colmap::RotationAveragingSolver`.
#[derive(Clone, Debug)]
pub struct RotationAveragingSolver {
    options: RotationEstimatorOptions,
}

fn has_nan(v: &VectorXd) -> bool {
    v.as_slice().iter().any(|x| x.is_nan())
}

impl RotationAveragingSolver {
    /// A solver with a copy of `options`.
    pub fn new(options: &RotationEstimatorOptions) -> Self {
        Self { options: *options }
    }

    /// `Solve`: false if a linear solve failed.
    pub fn solve(&self, problem: &mut RotationAveragingProblem) -> Result<bool> {
        if self.options.max_num_l1_iterations > 0 && !self.solve_l1_regression(problem)? {
            return Ok(false);
        }
        if self.options.max_num_irls_iterations > 0 && !self.solve_irls(problem)? {
            return Ok(false);
        }
        Ok(true)
    }

    /// `SolveL1Regression`: the L1 robust loss minimization phase.
    fn solve_l1_regression(&self, problem: &mut RotationAveragingProblem) -> Result<bool> {
        let l1_solver_options = LeastAbsoluteDeviationSolverOptions {
            max_num_iterations: 10,
            solver_type: LeastAbsoluteDeviationSolverType::SupernodalCholmodLlt,
            ridge_regularization: self.options.ridge_regularization,
            ..LeastAbsoluteDeviationSolverOptions::default()
        };

        // For weighted rotation averaging, the row-scaled constraint matrix and residuals
        // solve the weighted problem min ||D(Ax-b)||_1 = sum_i w_i |r_i|.
        let l1_matrix = problem.weighted_constraint_matrix();

        let mut l1_solver = LeastAbsoluteDeviationSolver::new(&l1_solver_options, &l1_matrix)?;
        if !l1_solver.valid() {
            // COLMAP: LOG(ERROR) "L1 regression linear solver factorization failed".
            return Ok(false);
        }

        let mut curr_norm = 0.0;
        for _iteration in 0..self.options.max_num_l1_iterations {
            problem.compute_residuals()?;

            let prev_norm = curr_norm;

            let mut step = VectorXd::zeros(problem.num_parameters());
            if !l1_solver.solve(&problem.weighted_residuals(), &mut step)? || has_nan(&step) {
                // COLMAP: LOG(ERROR) "L1 regression solve failed (iteration N)".
                return Ok(false);
            }

            curr_norm = step.norm();
            problem.update_state(&step)?;

            // Check convergence.
            const EPS: f64 = 1e-12;
            if problem.average_step_size(&step) < self.options.l1_step_convergence_threshold
                || (prev_norm - curr_norm).abs() < EPS
            {
                break;
            }
            l1_solver.set_max_num_iterations((l1_solver.max_num_iterations() * 2).min(100))?;
        }
        Ok(true)
    }

    /// `ComputeIRLSWeights`: the IRLS weight of every residual row; `None` if any weight is
    /// NaN.
    fn compute_irls_weights(
        &self,
        problem: &RotationAveragingProblem,
        sigma: f64,
    ) -> Option<VectorXd> {
        let mut weights = VectorXd::zeros(problem.num_residuals());
        let residuals = problem.residuals();

        for constraint in problem.pair_constraints().values() {
            let row = constraint.row_index;
            let err_squared = match constraint.constraint {
                RotationConstraint::GravityAligned1Dof { xz_error, .. } => {
                    // 1-DOF: Y-axis error plus xz_error.
                    let residual = residuals[row];
                    residual * residual + xz_error
                }
                RotationConstraint::Full3Dof { .. } => {
                    // 3-DOF: full rotation error.
                    residuals.segment(row, 3).squared_norm()
                }
            };

            // Compute the weight.
            let w = match self.options.weight_type {
                RotationWeightType::GemanMcClure => {
                    let tmp = err_squared + sigma * sigma;
                    sigma * sigma / (tmp * tmp)
                }
                RotationWeightType::HalfNorm => {
                    // Exponent for half-norm weight: (p - 2) / 2 where p = 0.5.
                    const HALF_NORM_EXPONENT: f64 = (0.5 - 2.0) / 2.0;
                    fns::pow(err_squared, HALF_NORM_EXPONENT)
                }
            };

            if w.is_nan() {
                // COLMAP: LOG(ERROR) "nan weight!".
                return None;
            }

            // Set weights for appropriate number of equations.
            for i in 0..constraint.constraint.num_rows() {
                weights[row + i] = w;
            }
        }

        // Set gauge-fixing weights to 1.
        let num_residuals = problem.num_residuals();
        for i in num_residuals - problem.num_gauge_fixing_residuals()..num_residuals {
            weights[i] = 1.0;
        }

        // Fold the residual-space reweighting W into the IRLS weights so the solver can
        // operate on the plain constraint matrix, scaling each robust weight by W
        // (gauge-fixing rows have W = 1 and are left unchanged).
        if let Some(reweighting) = problem.residual_reweighting() {
            for i in 0..weights.len() {
                weights[i] *= reweighting[i];
            }
        }
        Some(weights)
    }

    /// `SolveIRLS`: the iteratively reweighted least squares phase.
    fn solve_irls(&self, problem: &mut RotationAveragingProblem) -> Result<bool> {
        let mut solver = SparseCholeskyWithFallbackSolver::new();
        let mut pattern_analyzed = false;

        let sigma = deg_to_rad(self.options.irls_loss_parameter_sigma);

        // The constraint matrix A is constant across iterations; only the per-row weights
        // change. compute_irls_weights folds the optional residual-space reweighting W into
        // the robust weights, so we solve the normal equations A^T D A x = A^T D b directly
        // with the plain constraint matrix.
        let constraint_matrix = problem.constraint_matrix().clone();
        let constraint_matrix_t = constraint_matrix.transpose();

        for _iteration in 0..self.options.max_num_irls_iterations {
            problem.compute_residuals()?;

            let Some(weights_irls) = self.compute_irls_weights(problem, sigma) else {
                return Ok(false);
            };

            // A^T D: column i of A^T is row i of A, scaled by its weight.
            let mut at_weight = constraint_matrix_t.clone();
            scale_columns(&mut at_weight, &weights_irls);
            let mut at_weight_a = &at_weight * &constraint_matrix;

            // Optionally add a small Tikhonov ridge to stabilize poorly conditioned but
            // mathematically PD systems. When zero, no regularization is added.
            if self.options.ridge_regularization > 0.0 {
                at_weight_a = at_weight_a.add_to_diagonal(self.options.ridge_regularization);
            }

            if !pattern_analyzed {
                solver.analyze_pattern(&at_weight_a)?;
                pattern_analyzed = true;
            }
            if !solver.factorize(&at_weight_a)? {
                // COLMAP: LOG(ERROR) "IRLS Cholesky factorization failed (iteration N)".
                return Ok(false);
            }

            let at_weight_residuals = &at_weight * problem.residuals();
            let step = match solver.solve(&at_weight_residuals) {
                Some(step) if !has_nan(&step) => step,
                // COLMAP: LOG(ERROR) "IRLS solve failed (iteration N)".
                _ => return Ok(false),
            };
            problem.update_state(&step)?;

            if problem.average_step_size(&step) < self.options.irls_step_convergence_threshold {
                break;
            }
        }
        Ok(true)
    }
}

/// `A^T * weights.asDiagonal()`: scales column j of `matrix` by `scale[j]`.
fn scale_columns(matrix: &mut SparseMatrixCsc, scale: &VectorXd) {
    let col_ptr = matrix.col_ptr().to_vec();
    let values = matrix.values_mut();
    for j in 0..col_ptr.len() - 1 {
        for value in &mut values[col_ptr[j]..col_ptr[j + 1]] {
            *value *= scale[j];
        }
    }
}
