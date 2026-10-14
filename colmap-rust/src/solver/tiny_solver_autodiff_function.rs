//! [`TinySolverAutoDiffFunction`]: adapts an [`AutoDiffFunctor`] to
//! [`crate::optim::TinySolverFunction`]. Port of colmap-sharp's
//! `ColmapSharp/Solver/TinySolverAutoDiffFunction.cs` (from Ceres Solver 2.2.0
//! `include/ceres/tiny_solver_autodiff_function.h`, BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md).
//!
//! With no Jacobian requested it runs the functor on doubles; otherwise on [`Jet<N>`] with
//! every parameter a variable, one pass, and copies the derivative parts out into the
//! column-major Jacobian. Ceres' template arguments `<CostFunctor, kNumResiduals,
//! kNumParameters>` become `N = kNumParameters` (the Jet width is the parameter count) and a
//! constructor argument for the residual count (Ceres reads the functor's `NumResiduals()`
//! when it is `Eigen::Dynamic`).
//!
//! Like Ceres, whose residual Jets are a `mutable` member, the residual buffer is allocated
//! once and reused across calls (a [`RefCell`]), so one instance is not `Sync`; the
//! parameter Jets live on the stack.

use std::cell::RefCell;

use crate::optim::TinySolverFunction;

use super::autodiff_cost_function::{AutoDiffFunctor, IMPOSSIBLE_VALUE};
use super::jet::Jet;

/// Port of `ceres::TinySolverAutoDiffFunction<F, kNumResiduals, N>`: a
/// [`TinySolverFunction`] whose Jacobian comes from evaluating `F` on [`Jet<N>`].
#[derive(Debug)]
pub struct TinySolverAutoDiffFunction<F, const N: usize> {
    cost_functor: F,
    num_residuals: usize,
    jet_residuals: RefCell<Vec<Jet<N>>>,
}

impl<F: AutoDiffFunctor, const N: usize> TinySolverAutoDiffFunction<F, N> {
    /// Wraps `cost_functor`, which produces `num_residuals` residuals from `N` parameters.
    pub fn new(cost_functor: F, num_residuals: usize) -> Self {
        Self {
            cost_functor,
            num_residuals,
            jet_residuals: RefCell::new(vec![Jet::constant(0.0); num_residuals]),
        }
    }

    /// The wrapped functor.
    pub fn functor(&self) -> &F {
        &self.cost_functor
    }
}

impl<F: AutoDiffFunctor, const N: usize> TinySolverFunction for TinySolverAutoDiffFunction<F, N> {
    const NUM_PARAMETERS: usize = N;

    fn num_residuals(&self) -> usize {
        self.num_residuals
    }

    fn evaluate(
        &self,
        parameters: &[f64],
        residuals: &mut [f64],
        jacobian: Option<&mut [f64]>,
    ) -> bool {
        let n = self.num_residuals;
        let Some(jacobian) = jacobian else {
            // No Jacobian requested, so just directly call the cost function with doubles,
            // skipping jets and derivatives.
            return self
                .cost_functor
                .call(&parameters[..N], &mut residuals[..n]);
        };

        // Initialize the input jets with the passed parameters.
        let mut jet_parameters = [Jet::<N>::constant(0.0); N];
        for (i, jet) in jet_parameters.iter_mut().enumerate() {
            *jet = Jet::variable(parameters[i], i);
        }

        // Initialize the output jets such that we can detect user errors.
        let mut jet_residuals = self.jet_residuals.borrow_mut();
        jet_residuals.fill(Jet::new(IMPOSSIBLE_VALUE, [IMPOSSIBLE_VALUE; N]));

        // Execute the cost function, but with jets to find the derivative.
        if !self.cost_functor.call(&jet_parameters, &mut jet_residuals) {
            return false;
        }

        // Copy the jacobian out of the derivative part of the residual jets (column-major
        // n x N).
        for (r, jet) in jet_residuals.iter().enumerate() {
            residuals[r] = jet.a;
            for c in 0..N {
                jacobian[c * n + r] = jet.v[c];
            }
        }
        true
    }
}
