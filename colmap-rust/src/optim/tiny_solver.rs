//! Port of COLMAP's `colmap/optim/tiny_solver.h`, COLMAP's customized copy of
//! `ceres::TinySolver` (Ceres Solver, BSD-3-Clause, see THIRD_PARTY_NOTICES.md), via
//! colmap-sharp's `Optim/TinySolver.cs`. Tests: `tests/optim/tiny_solver.rs`
//! (`tiny_solver_test.cc`).
//!
//! A tiny dense Levenberg-Marquardt solver for small problems with low latency and low
//! overhead: the essential/fundamental-matrix and relative-pose-with-focal refinements of
//! estimators/solvers/ run it once per RANSAC model. COLMAP's copy differs from upstream
//! Ceres in three ways, all kept here: (1) an optional manifold decouples the ambient
//! parameter block from the tangent step (`plus` to step, `plus_jacobian` to project the
//! ambient Jacobian); (2) cost-function and linear-solver failures are handled (the step is
//! rejected, or `CostFunctionFailed` is reported at the start); (3) the parameter and tangent
//! sizes are fixed per problem, only the residual count varies.
//!
//! Algorithm: K. Madsen, H. Nielsen, O. Tingleff, "Methods for Non-linear Least Squares
//! Problems", 2004.
//!
//! Translation notes:
//! - C++'s compile-time sizes become [`TinySolverFunction::NUM_PARAMETERS`] and the
//!   manifold's ambient/tangent sizes. The residual count comes from the function instance
//!   ([`TinySolverFunction::num_residuals`], which C++ needs only when dynamic).
//! - The Jacobian pointer that C++ passes as `nullptr` when only residuals are wanted is
//!   `None`.
//! - The Eigen LDLT linear solver is [`Ldlt`], with its `info()` checked as C++ checks it.
//!   Buffers are allocated once per residual count and reused across solves.
//! - Only the Euclidean manifold lives here; the others (`tiny_manifold.h`) are in
//!   `estimators/cost_functions/tiny_manifold.rs`.
//!
//! Result tier: C (iterative). Scalar steps match the C++ line for line, but the Eigen
//! products and LDLT are replaced, so iterates agree to rounding, not bit for bit.

use crate::linalg::{maxi, mini, ComputationInfo, Ldlt, MatrixXd, VectorXd};
use crate::math::fns;
use crate::{check_eq, check_ge, check_gt, Result};

/// The cost function TinySolver minimizes: C++'s TinySolver "Function" concept (`Scalar =
/// double`, `NUM_PARAMETERS`, `NUM_RESIDUALS` / `NumResiduals()`, `operator()`).
pub trait TinySolverFunction {
    /// Size of the ambient parameter block, C++'s `NUM_PARAMETERS`.
    const NUM_PARAMETERS: usize;

    /// Number of residuals, C++'s `NUM_RESIDUALS` or `NumResiduals()`.
    fn num_residuals(&self) -> usize;

    /// Evaluates the residuals at `parameters` (`NUM_PARAMETERS` long) and, when `jacobian`
    /// is given, the column-major `num_residuals x NUM_PARAMETERS` Jacobian with respect to
    /// the ambient parameters. Returns false if the function cannot be evaluated there.
    fn evaluate(
        &self,
        parameters: &[f64],
        residuals: &mut [f64],
        jacobian: Option<&mut [f64]>,
    ) -> bool;
}

/// A manifold policy for TinySolver (the concept described in COLMAP's
/// `estimators/cost_functions/tiny_manifold.h`). Evaluated on doubles only.
pub trait TinyManifold {
    /// Size of the parameter block, C++'s `kAmbientSize`.
    fn ambient_size(&self) -> usize;
    /// Degrees of freedom, C++'s `kTangentSize`.
    fn tangent_size(&self) -> usize;
    /// True iff `plus` is plain addition, C++'s `kIsEuclidean`.
    fn is_euclidean(&self) -> bool;
    /// `x_plus_delta = Plus(x, delta)`, of sizes [ambient, tangent, ambient].
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]);
    /// The row-major `ambient x tangent` Jacobian of `plus` at delta = 0.
    fn plus_jacobian(&self, x: &[f64], jacobian: &mut [f64]);
}

/// Port of `colmap::EuclideanManifold<N>`: the identity manifold, tangent space equal to the
/// ambient space and `plus` plain addition. It reproduces plain `ceres::TinySolver`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TinyEuclideanManifold {
    size: usize,
}

impl TinyEuclideanManifold {
    /// The Euclidean manifold R^`size`; `size` must be positive.
    pub fn new(size: usize) -> Result<Self> {
        check_gt!(size, 0);
        Ok(Self { size })
    }
}

impl TinyManifold for TinyEuclideanManifold {
    fn ambient_size(&self) -> usize {
        self.size
    }
    fn tangent_size(&self) -> usize {
        self.size
    }
    fn is_euclidean(&self) -> bool {
        true
    }
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) {
        for i in 0..self.size {
            x_plus_delta[i] = x[i] + delta[i];
        }
    }
    /// The identity. Unused by the solver's Euclidean fast path, as in COLMAP.
    fn plus_jacobian(&self, _x: &[f64], jacobian: &mut [f64]) {
        jacobian[..self.size * self.size].fill(0.0);
        for i in 0..self.size {
            jacobian[i * self.size + i] = 1.0;
        }
    }
}

/// TinySolver's termination reason, C++'s `TinySolver::Status`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TinySolverStatus {
    /// max_norm |J'(x) * f(x)| < gradient_tolerance.
    GradientTooSmall,
    /// ||dx|| <= parameter_tolerance * (||x|| + parameter_tolerance).
    RelativeStepSizeTooSmall,
    /// cost_threshold > ||f(x)||^2 / 2.
    CostTooSmall,
    /// num_iterations >= max_num_iterations.
    HitMaxIterations,
    /// (new_cost - old_cost) < function_tolerance * old_cost.
    CostChangeTooSmall,
    /// The cost function failed to evaluate at the initial point (or, unexpectedly, at an
    /// accepted point), so no meaningful step could be taken.
    CostFunctionFailed,
}

/// TinySolver's options, C++'s `TinySolver::Options`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TinySolverOptions {
    /// Iteration cap.
    pub max_num_iterations: i32,
    /// max_norm |J'(x) * f(x)| < gradient_tolerance stops.
    pub gradient_tolerance: f64,
    /// ||dx|| <= parameter_tolerance * (||x|| + parameter_tolerance) stops.
    pub parameter_tolerance: f64,
    /// (new_cost - old_cost) < function_tolerance * old_cost stops.
    pub function_tolerance: f64,
    /// cost_threshold > ||f(x)||^2 / 2 stops. Default: machine epsilon.
    pub cost_threshold: f64,
    /// The first trust region radius; the LM damping starts at its inverse.
    pub initial_trust_region_radius: f64,
}

impl Default for TinySolverOptions {
    fn default() -> Self {
        Self {
            max_num_iterations: 50,
            gradient_tolerance: 1e-10,
            parameter_tolerance: 1e-8,
            function_tolerance: 1e-6,
            cost_threshold: f64::EPSILON,
            initial_trust_region_radius: 1e4,
        }
    }
}

impl TinySolverOptions {
    /// Validates the options (C++'s `Options::Check()`).
    pub fn check(&self) -> Result<()> {
        check_gt!(self.max_num_iterations, 0);
        check_ge!(self.gradient_tolerance, 0.0);
        check_ge!(self.parameter_tolerance, 0.0);
        check_ge!(self.function_tolerance, 0.0);
        check_ge!(self.cost_threshold, 0.0);
        check_gt!(self.initial_trust_region_radius, 0.0);
        Ok(())
    }
}

/// TinySolver's result, C++'s `TinySolver::Summary`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TinySolverSummary {
    /// 1/2 ||f(x0)||^2, or -1 when the function failed at x0.
    pub initial_cost: f64,
    /// 1/2 ||f(x)||^2 at the result, or -1 when the function failed at x0.
    pub final_cost: f64,
    /// max_norm |J'(x) f(x)| at the last accepted point (-1 if never evaluated).
    pub gradient_max_norm: f64,
    /// Iterations run.
    pub iterations: i32,
    /// Why the solver stopped.
    pub status: TinySolverStatus,
}

/// Port of `colmap::TinySolver`: small dense Levenberg-Marquardt over a manifold. One
/// instance can solve many problems; its buffers are reused.
#[derive(Clone, Debug)]
pub struct TinySolver<M: TinyManifold> {
    manifold: M,
    num_parameters: usize,
    num_tangent: usize,
    x_new: Vec<f64>,
    dx: Vec<f64>,
    g: Vec<f64>,
    jacobi_scaling: Vec<f64>,
    plus_jacobian: Vec<f64>,
    jtj: MatrixXd,
    residuals: Vec<f64>,
    fx_new: Vec<f64>,
    // Column-major num_residuals x num_parameters, written by the function when the
    // manifold is not Euclidean, then projected into `jacobian`.
    jacobian_ambient: Vec<f64>,
    // Column-major num_residuals x num_tangent: the tangent-space Jacobian.
    jacobian: Vec<f64>,
    cost: f64,
    iterations: i32,
    gradient_max_norm: f64,
}

impl TinySolver<TinyEuclideanManifold> {
    /// A solver on R^`num_parameters`: C++'s `TinySolver<Function>` with its default
    /// manifold.
    pub fn euclidean(num_parameters: usize) -> Result<Self> {
        Self::new(TinyEuclideanManifold::new(num_parameters)?)
    }
}

impl<M: TinyManifold> TinySolver<M> {
    /// Creates a solver for the given manifold (C++ default-constructs it).
    pub fn new(manifold: M) -> Result<Self> {
        let num_parameters = manifold.ambient_size();
        let num_tangent = manifold.tangent_size();
        check_gt!(num_tangent, 0);
        Ok(Self {
            manifold,
            num_parameters,
            num_tangent,
            x_new: vec![0.0; num_parameters],
            dx: vec![0.0; num_tangent],
            g: vec![0.0; num_tangent],
            jacobi_scaling: vec![0.0; num_tangent],
            plus_jacobian: vec![0.0; num_parameters * num_tangent],
            jtj: MatrixXd::zeros(num_tangent, num_tangent),
            residuals: Vec::new(),
            fx_new: Vec::new(),
            jacobian_ambient: Vec::new(),
            jacobian: Vec::new(),
            cost: 0.0,
            iterations: 0,
            gradient_max_norm: -1.0,
        })
    }

    /// Minimizes 1/2 ||f(x)||^2 starting from `x_and_min`, which receives the result.
    pub fn solve<F: TinySolverFunction>(
        &mut self,
        function: &F,
        x_and_min: &mut [f64],
        options: &TinySolverOptions,
    ) -> Result<TinySolverSummary> {
        check_eq!(
            F::NUM_PARAMETERS,
            self.num_parameters,
            "Function::NUM_PARAMETERS must match Manifold::kAmbientSize."
        );
        check_eq!(x_and_min.len(), self.num_parameters);
        options.check()?;
        self.initialize(function);
        let x = x_and_min;
        self.iterations = 0;
        self.gradient_max_norm = -1.0;
        let summary = |s: &Self, initial_cost, status| TinySolverSummary {
            initial_cost,
            final_cost: s.cost,
            gradient_max_norm: s.gradient_max_norm,
            iterations: s.iterations,
            status,
        };

        // Bail out cleanly if the cost function cannot be evaluated at the initial point;
        // there is nothing meaningful the solver can do in that case.
        if !self.update(function, x) {
            return Ok(TinySolverSummary {
                initial_cost: -1.0,
                final_cost: -1.0,
                gradient_max_norm: self.gradient_max_norm,
                iterations: self.iterations,
                status: TinySolverStatus::CostFunctionFailed,
            });
        }
        let initial_cost = self.cost;
        if self.gradient_max_norm < options.gradient_tolerance {
            return Ok(summary(
                self,
                initial_cost,
                TinySolverStatus::GradientTooSmall,
            ));
        }
        if self.cost < options.cost_threshold {
            return Ok(summary(self, initial_cost, TinySolverStatus::CostTooSmall));
        }

        let n_t = self.num_tangent;
        let mut status = TinySolverStatus::HitMaxIterations;
        let mut u = 1.0 / options.initial_trust_region_radius;
        let mut v = 2.0;
        let mut g_vector = VectorXd::zeros(n_t);

        self.iterations = 1;
        while self.iterations < options.max_num_iterations {
            const MIN_DIAGONAL: f64 = 1e-6;
            const MAX_DIAGONAL: f64 = 1e32;
            let mut jtj_regularized = self.jtj.clone();
            for i in 0..n_t {
                let jii = self.jtj[(i, i)];
                jtj_regularized[(i, i)] += u * mini(maxi(jii, MIN_DIAGONAL), MAX_DIAGONAL);
            }
            let linear_solver = Ldlt::new(&jtj_regularized);

            // If the factorization failed, the regularized normal equations were not
            // solvable. Treat this like a rejected step: shrink the trust region (which
            // increases the regularization) and try again.
            if linear_solver.info() != ComputationInfo::Success {
                u *= v;
                v *= 2.0;
                self.iterations += 1;
                continue;
            }

            g_vector.as_mut_slice().copy_from_slice(&self.g);
            let lm_step = linear_solver.solve(&g_vector);
            for i in 0..n_t {
                self.dx[i] = self.jacobi_scaling[i] * lm_step[i];
            }

            // Adding parameter_tolerance to x.norm() ensures that this works if x is near
            // zero.
            let parameter_tolerance =
                options.parameter_tolerance * (norm(x) + options.parameter_tolerance);
            if norm(&self.dx) < parameter_tolerance {
                status = TinySolverStatus::RelativeStepSizeTooSmall;
                break;
            }

            self.manifold.plus(x, &self.dx, &mut self.x_new);

            // If the cost function fails to evaluate at the trial point, reject the step and
            // shrink the trust region rather than acting on garbage.
            if !function.evaluate(&self.x_new, &mut self.fx_new, None) {
                u *= v;
                v *= 2.0;
                self.iterations += 1;
                continue;
            }

            let cost_change = 2.0 * self.cost - squared_norm(&self.fx_new);
            let model_cost_change = self.model_cost_change(lm_step.as_slice());

            // rho is the ratio of the actual reduction in error to the reduction in error
            // that would be obtained if the problem was linear. See [1] for details.
            let rho = cost_change / model_cost_change;
            if rho > 0.0 {
                // Accept the Levenberg-Marquardt step because the linear model fits well.
                x.copy_from_slice(&self.x_new);

                if cost_change.abs() < options.function_tolerance {
                    self.cost = squared_norm(&self.fx_new) / 2.0;
                    status = TinySolverStatus::CostChangeTooSmall;
                    break;
                }

                // The cost function already evaluated successfully at x_new == x above, so
                // re-evaluating (now also for the Jacobian) is not expected to fail; guard
                // against it regardless.
                if !self.update(function, x) {
                    status = TinySolverStatus::CostFunctionFailed;
                    break;
                }
                if self.gradient_max_norm < options.gradient_tolerance {
                    status = TinySolverStatus::GradientTooSmall;
                    break;
                }
                if self.cost < options.cost_threshold {
                    status = TinySolverStatus::CostTooSmall;
                    break;
                }

                let tmp = 2.0 * rho - 1.0;
                u *= maxi(1.0 / 3.0, 1.0 - tmp * tmp * tmp);
                v = 2.0;
            } else {
                // Reject the update because either the normal equations failed to solve or
                // the local linear model was not good (rho < 0).

                // Additionally if the cost change is too small, then terminate.
                if cost_change.abs() < options.function_tolerance {
                    status = TinySolverStatus::CostChangeTooSmall;
                    break;
                }

                // Reduce the size of the trust region.
                u *= v;
                v *= 2.0;
            }
            self.iterations += 1;
        }

        Ok(summary(self, initial_cost, status))
    }

    /// Evaluates the residuals and the Jacobian with respect to the ambient parameters, then
    /// projects it into the tangent space at x: J_tangent = J_ambient * dPlus(x, delta)/ddelta
    /// at delta = 0. Then forms the Jacobi-scaled normal equations.
    fn update<F: TinySolverFunction>(&mut self, function: &F, x: &[f64]) -> bool {
        let n = self.residuals.len();
        let n_t = self.num_tangent;
        if self.manifold.is_euclidean() {
            // The tangent space equals the ambient space, so the cost function writes its
            // Jacobian straight into `jacobian` (no projection, no copy).
            if !function.evaluate(x, &mut self.residuals, Some(&mut self.jacobian)) {
                return false;
            }
        } else {
            if !function.evaluate(x, &mut self.residuals, Some(&mut self.jacobian_ambient)) {
                return false;
            }
            self.manifold.plus_jacobian(x, &mut self.plus_jacobian);
            for t in 0..n_t {
                for r in 0..n {
                    let mut sum = 0.0;
                    for p in 0..self.num_parameters {
                        sum += self.jacobian_ambient[p * n + r] * self.plus_jacobian[p * n_t + t];
                    }
                    self.jacobian[t * n + r] = sum;
                }
            }
        }

        for r in self.residuals.iter_mut() {
            *r = -*r;
        }

        // On the first iteration, compute a diagonal (Jacobi) scaling matrix, which we store
        // as a vector: jacobi_scaling = 1 / (1 + ||column of J||). 1 is added to the
        // denominator to regularize small diagonal entries.
        if self.iterations == 0 {
            for t in 0..n_t {
                self.jacobi_scaling[t] = 1.0 / (1.0 + norm(&self.jacobian[t * n..(t + 1) * n]));
            }
        }

        // This explicitly computes the normal equations, which is numerically unstable.
        // Nevertheless, it is often good enough and is fast.
        for t in 0..n_t {
            let scale = self.jacobi_scaling[t];
            for value in &mut self.jacobian[t * n..(t + 1) * n] {
                *value *= scale;
            }
        }

        self.gradient_max_norm = 0.0;
        for a in 0..n_t {
            let column_a = &self.jacobian[a * n..(a + 1) * n];
            for b in 0..=a {
                let d = dot(column_a, &self.jacobian[b * n..(b + 1) * n]);
                self.jtj[(a, b)] = d;
                self.jtj[(b, a)] = d;
            }
            self.g[a] = dot(column_a, &self.residuals);
            self.gradient_max_norm = if a == 0 {
                self.g[a].abs()
            } else {
                maxi(self.gradient_max_norm, self.g[a].abs())
            };
        }

        self.cost = squared_norm(&self.residuals) / 2.0;
        true
    }

    /// lm_step . (2 g - jtj lm_step), the cost reduction the linear model predicts.
    fn model_cost_change(&self, lm_step: &[f64]) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.num_tangent {
            let mut jtj_step = 0.0;
            for (j, &step) in lm_step.iter().enumerate() {
                jtj_step += self.jtj[(i, j)] * step;
            }
            sum += lm_step[i] * (2.0 * self.g[i] - jtj_step);
        }
        sum
    }

    /// Only the number of residuals varies between problems; the parameter and tangent
    /// dimensions are fixed, so nothing else needs (re)allocation.
    fn initialize<F: TinySolverFunction>(&mut self, function: &F) {
        let num_residuals = function.num_residuals();
        if self.residuals.len() != num_residuals {
            self.residuals = vec![0.0; num_residuals];
            self.fx_new = vec![0.0; num_residuals];
            self.jacobian_ambient = if self.manifold.is_euclidean() {
                Vec::new()
            } else {
                vec![0.0; num_residuals * self.num_parameters]
            };
            self.jacobian = vec![0.0; num_residuals * self.num_tangent];
        }
    }
}

/// Left-to-right dot product from 0, as colmap-sharp's TinySolver sums.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        sum += x * y;
    }
    sum
}

fn squared_norm(a: &[f64]) -> f64 {
    dot(a, a)
}

fn norm(a: &[f64]) -> f64 {
    fns::sqrt(squared_norm(a))
}
