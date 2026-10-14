// Rust-only tests of `colmap_rust::solver::ProgramEvaluator` and the loss handling of
// `Problem::evaluate`, which Ceres' problem_test.cc does not reach: Ceres' evaluator tests
// (evaluator_test.cc) run through its minimizer-side factory and linear solver types, ported
// with the evaluator slice. Pinned here: the dense and block-sparse Jacobians agree entry for
// entry, with the same cost, residuals and gradient (a gradient asked for alone goes through
// the evaluator's own Jacobian); `apply_loss_function` switches between the plain and the
// robust cost and residuals; a failing cost function fails the evaluation; and the evaluation
// counters split as Ceres' timers do.

use std::sync::Arc;

use colmap_rust::solver::{
    CauchyLoss, CostFunction, EvaluateOptions, EvaluatorJacobian, LossFunction, Problem,
    ProgramEvaluator, SubsetManifold,
};

// r = (x0 * y0 - 1, x1 + y1 * y1 - 2) over two blocks of two values.
struct Product;

impl CostFunction for Product {
    fn num_residuals(&self) -> usize {
        2
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &[2, 2]
    }
    fn evaluate(
        &self,
        p: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        let (x, y) = (p[0], p[1]);
        residuals[0] = x[0] * y[0] - 1.0;
        residuals[1] = x[1] + y[1] * y[1] - 2.0;
        if let Some(jacobians) = jacobians {
            if let Some(j) = jacobians[0].as_deref_mut() {
                j.copy_from_slice(&[y[0], 0.0, 0.0, 1.0]);
            }
            if let Some(j) = jacobians[1].as_deref_mut() {
                j.copy_from_slice(&[x[0], 0.0, 0.0, 2.0 * y[1]]);
            }
        }
        true
    }
}

// r = x0, failing whenever x0 is negative; leaves its residual unwritten when x1 is negative.
struct FailsWhenNegative;

impl CostFunction for FailsWhenNegative {
    fn num_residuals(&self) -> usize {
        1
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &[2]
    }
    fn evaluate(
        &self,
        p: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        if p[0][1] >= 0.0 {
            residuals[0] = p[0][0];
        }
        if let Some(j) = jacobians.and_then(|j| j[0].as_deref_mut()) {
            j.copy_from_slice(&[1.0, 0.0]);
        }
        p[0][0] >= 0.0
    }
}

fn problem() -> Problem {
    let mut problem = Problem::new();
    let a = problem.add_parameter_block(&[1.5, -0.5]);
    let b = problem.add_parameter_block(&[2.0, 3.0]);
    let c = problem.add_parameter_block(&[-1.0, 0.25]);
    let cost: Arc<dyn CostFunction> = Arc::new(Product);
    problem
        .add_residual_block(cost.clone(), None, &[a, b])
        .unwrap();
    problem
        .add_residual_block(cost.clone(), None, &[b, c])
        .unwrap();
    problem.add_residual_block(cost, None, &[c, a]).unwrap();
    problem
        .set_manifold(c, Some(Arc::new(SubsetManifold::new(2, &[1]).unwrap())))
        .unwrap();
    problem
}

// Every entry of a Jacobian, row-major.
fn dense_entries(jacobian: &EvaluatorJacobian) -> Vec<f64> {
    let matrix = jacobian.as_sparse_matrix();
    let (rows, cols) = (matrix.num_rows(), matrix.num_cols());
    let mut entries = vec![0.0; rows * cols];
    for c in 0..cols {
        let mut unit = vec![0.0; cols];
        unit[c] = 1.0;
        let mut column = vec![0.0; rows];
        matrix.right_multiply_and_accumulate(&unit, &mut column);
        for r in 0..rows {
            entries[r * cols + c] = column[r];
        }
    }
    entries
}

#[test]
fn rust_only_program_evaluator_dense_and_block_sparse_agree() {
    let mut problem = problem();
    let program = problem.program().clone();
    let (_, blocks) = problem.program_and_blocks_mut();
    program.set_parameter_offsets_and_index(blocks);
    let mut state = vec![0.0; program.num_parameters(blocks)];
    program.parameter_blocks_to_state_vector(blocks, &mut state);

    let mut results = Vec::new();
    for dense in [true, false] {
        let mut evaluator = ProgramEvaluator::new(program.clone(), blocks, dense, 0);
        let mut jacobian = evaluator.create_jacobian(blocks);
        let mut residuals = vec![0.0; evaluator.num_residuals()];
        let mut gradient = vec![0.0; evaluator.num_effective_parameters()];
        let cost = evaluator
            .evaluate(
                blocks,
                &state,
                true,
                Some(&mut residuals),
                Some(&mut gradient),
                Some(&mut jacobian),
            )
            .unwrap()
            .unwrap();

        // A gradient alone goes through the evaluator's own Jacobian and residuals.
        let mut gradient_only = vec![0.0; evaluator.num_effective_parameters()];
        let cost_only = evaluator
            .evaluate(blocks, &state, true, None, Some(&mut gradient_only), None)
            .unwrap()
            .unwrap();
        assert_eq!(cost_only, cost);
        assert_eq!(gradient_only, gradient);
        assert_eq!(evaluator.num_jacobian_evaluations(), 2);
        evaluator
            .evaluate(blocks, &state, true, None, None, None)
            .unwrap()
            .unwrap();
        assert_eq!(evaluator.num_residual_evaluations(), 1);

        results.push((cost, residuals, gradient, dense_entries(&jacobian)));
    }
    assert_eq!(results[0], results[1]);
    // Five tangent columns (c keeps one coordinate of two), six residuals.
    assert_eq!(results[0].3.len(), 6 * 5);
}

#[test]
fn rust_only_problem_evaluate_applies_the_loss_on_request() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[3.0, 4.0]);
    let y = problem.add_parameter_block(&[1.0, 1.0]);
    let loss: Arc<dyn LossFunction> = Arc::new(CauchyLoss::new(1.0));
    problem
        .add_residual_block(Arc::new(Product), Some(loss.clone()), &[x, y])
        .unwrap();

    // r = (3 * 1 - 1, 4 + 1 - 2) = (2, 3), s = 13.
    let mut residuals = Vec::new();
    let plain = EvaluateOptions {
        apply_loss_function: false,
        ..EvaluateOptions::default()
    };
    let cost = problem
        .evaluate(&plain, Some(&mut residuals), None, None)
        .unwrap()
        .unwrap();
    assert_eq!(cost, 6.5);
    assert_eq!(residuals, [2.0, 3.0]);

    let mut rho = [0.0; 3];
    loss.evaluate(13.0, &mut rho);
    let cost = problem
        .evaluate(
            &EvaluateOptions::default(),
            Some(&mut residuals),
            None,
            None,
        )
        .unwrap()
        .unwrap();
    assert_eq!(cost, 0.5 * rho[0]);
    // Cauchy's rho'' < 0 here, so the corrector only scales by sqrt(rho').
    let scale = rho[1].sqrt();
    assert_eq!(residuals, [2.0 * scale, 3.0 * scale]);
}

#[test]
fn rust_only_problem_evaluate_reports_a_failing_cost_function() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[-1.0, 0.0]);
    problem
        .add_residual_block(Arc::new(FailsWhenNegative), None, &[x])
        .unwrap();
    let mut jacobian = Default::default();
    assert_eq!(
        problem
            .evaluate(&EvaluateOptions::default(), None, None, Some(&mut jacobian))
            .unwrap(),
        None
    );
    problem.parameter_block_mut(x).unwrap()[0] = 2.0;
    assert_eq!(
        problem
            .evaluate(&EvaluateOptions::default(), None, None, Some(&mut jacobian))
            .unwrap(),
        Some(2.0)
    );
    assert_eq!(jacobian.values, [1.0, 0.0]);

    // A residual left unwritten fails the evaluation (Ceres' IsEvaluationValid).
    problem.parameter_block_mut(x).unwrap()[1] = -1.0;
    assert_eq!(
        problem
            .evaluate(&EvaluateOptions::default(), None, None, None)
            .unwrap(),
        None
    );
}
