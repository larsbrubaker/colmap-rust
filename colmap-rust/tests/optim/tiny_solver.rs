// Port of COLMAP's `src/colmap/optim/tiny_solver_test.cc` (same names, checks and
// tolerances). Tests `colmap_rust::optim::TinySolver` (Tier C: the convergence outcome,
// with COLMAP's bounds). `ManifoldConvergesAndStaysOnManifold` differentiates through
// `colmap_rust::solver::TinySolverAutoDiffFunction` (`ceres::TinySolverAutoDiffFunction`)
// over `TinySphereManifold` (`SphereManifold<3>`).

use colmap_rust::estimators::cost_functions::TinySphereManifold;
use colmap_rust::linalg::{Ldlt, MatrixXd, Vector3d, VectorXd};
use colmap_rust::optim::{TinySolver, TinySolverFunction, TinySolverOptions, TinySolverStatus};
use colmap_rust::solver::{AutoDiffFunctor, Scalar, TinySolverAutoDiffFunction};

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

// Autodiff functor minimizing ||x - target||^2; on the unit sphere the optimum is
// target.normalized().
struct SphereFitResidual {
    target: Vector3d,
}

impl AutoDiffFunctor for SphereFitResidual {
    fn call<T: Scalar>(&self, x: &[T], residuals: &mut [T]) -> bool {
        residuals[0] = x[0] - T::from_f64(self.target.x);
        residuals[1] = x[1] - T::from_f64(self.target.y);
        residuals[2] = x[2] - T::from_f64(self.target.z);
        true
    }
}

#[test]
fn tiny_solver_manifold_converges_and_stays_on_manifold() {
    let functor = SphereFitResidual {
        target: Vector3d::new(0.3, -0.7, 0.5),
    };
    let target = functor.target;

    // ceres::TinySolverAutoDiffFunction<SphereFitResidual, 3, 3>.
    let f = TinySolverAutoDiffFunction::<_, 3>::new(functor, 3);

    let mut solver = TinySolver::new(TinySphereManifold).unwrap();
    let options = TinySolverOptions {
        gradient_tolerance: 0.0,
        parameter_tolerance: 0.0,
        function_tolerance: 0.0,
        max_num_iterations: 100,
        ..Default::default()
    };
    let mut x = [1.0, 0.0, 0.0]; // Unit seed.
    solver.solve(&f, &mut x, &options).unwrap();

    let x = Vector3d::new(x[0], x[1], x[2]);
    expect_near(x.norm(), 1.0, 1e-9);
    assert!((x - target.normalized()).norm() < 1e-6);
}

// A functor that always fails to evaluate.
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
