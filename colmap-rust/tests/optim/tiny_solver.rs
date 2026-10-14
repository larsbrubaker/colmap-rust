// Port of COLMAP's `src/colmap/optim/tiny_solver_test.cc` (same names, checks and
// tolerances). Tests `colmap_rust::optim::TinySolver` (Tier C: the convergence outcome,
// with COLMAP's bounds). `TinySolver.ManifoldConvergesAndStaysOnManifold` is not here yet:
// it needs `SphereManifold<3>` (estimators/cost_functions/tiny_manifold.h) and
// `ceres::TinySolverAutoDiffFunction` (Jets, the Ceres replacement of Phase 7); it lands
// with them.

use colmap_rust::linalg::{Ldlt, MatrixXd, VectorXd};
use colmap_rust::optim::{TinySolver, TinySolverFunction, TinySolverOptions, TinySolverStatus};

use crate::expect_near;

fn a_matrix() -> MatrixXd {
    MatrixXd::from_row_major(3, 2, &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0])
}

fn b_vector() -> VectorXd {
    VectorXd::from_slice(&[1.0, 2.0, 4.0])
}

/// Linear least squares: residual = A * p - b, with analytic Jacobian.
struct LinearResidual;

impl TinySolverFunction for LinearResidual {
    const NUM_PARAMETERS: usize = 2;

    fn num_residuals(&self) -> usize {
        3
    }

    fn evaluate(
        &self,
        parameters: &[f64],
        residuals: &mut [f64],
        jacobian: Option<&mut [f64]>,
    ) -> bool {
        let a = a_matrix();
        let res = &(&a * &VectorXd::from_slice(&parameters[..2])) - &b_vector();
        residuals.copy_from_slice(res.as_slice());
        if let Some(jacobian) = jacobian {
            // Column-major, NUM_RESIDUALS x NUM_PARAMETERS.
            jacobian.copy_from_slice(a.as_slice());
        }
        true
    }
}

#[test]
fn tiny_solver_euclidean_converges_to_normal_equations_solution() {
    let a = a_matrix();
    let at = a.transpose();
    let expected = Ldlt::new(&(&at * &a)).solve(&(&at * &b_vector()));

    let mut solver = TinySolver::euclidean(2).unwrap();
    let options = TinySolverOptions {
        gradient_tolerance: 0.0,
        parameter_tolerance: 0.0,
        function_tolerance: 0.0,
        max_num_iterations: 100,
        ..Default::default()
    };
    let mut x = [0.0, 0.0];
    let summary = solver.solve(&LinearResidual, &mut x, &options).unwrap();

    assert_ne!(summary.status, TinySolverStatus::CostFunctionFailed);
    assert!((&VectorXd::from_slice(&x) - &expected).norm() < 1e-9);
    // Started from x = 0, so initial_cost = 0.5 * ||b||^2 = 0.5 * 21.
    expect_near(summary.initial_cost, 10.5, 1e-9);
    // Cost must not increase, and at the least-squares optimum J'f(x) vanishes.
    assert!(summary.final_cost >= 0.0);
    assert!(summary.final_cost <= summary.initial_cost);
    expect_near(summary.gradient_max_norm, 0.0, 1e-6);
    assert!(summary.iterations >= 1);
    assert!(summary.iterations <= options.max_num_iterations);
}

struct FailingResidual;

impl TinySolverFunction for FailingResidual {
    const NUM_PARAMETERS: usize = 2;

    fn num_residuals(&self) -> usize {
        2
    }

    fn evaluate(
        &self,
        _parameters: &[f64],
        _residuals: &mut [f64],
        _jacobian: Option<&mut [f64]>,
    ) -> bool {
        false
    }
}

#[test]
fn tiny_solver_reports_cost_function_failure() {
    let mut solver = TinySolver::euclidean(2).unwrap();
    let mut x = [1.0, 2.0];
    let summary = solver
        .solve(&FailingResidual, &mut x, &TinySolverOptions::default())
        .unwrap();
    assert_eq!(summary.status, TinySolverStatus::CostFunctionFailed);
}
