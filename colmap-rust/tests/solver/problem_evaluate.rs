// Port of the ProblemEvaluateTest cases of Ceres Solver 2.2.0's internal/ceres/problem_test.cc
// and of CompareEvaluations (internal/ceres/evaluator_test_utils.cc) (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md): `colmap_rust::solver::Problem::evaluate`, same names, values and
// exact comparisons. Each case runs with every combination of residuals, gradient and
// Jacobian requested (CheckAllEvaluationCombinations); the CRS Jacobian is expanded to a dense
// row-major matrix first (CRSToDenseMatrix), and the fixture's TearDown check that the
// problem's program is still valid runs after every evaluation. The rest of problem_test.cc is
// in problem.rs.

use std::sync::Arc;

use colmap_rust::solver::{
    CostFunction, CrsMatrix, EvaluateOptions, ParameterBlockId, Problem, ResidualBlockId,
    SubsetManifold,
};

// r_i = i - sum_j (j + 1) * x_ji^2 over kNumParameterBlocks blocks of kNumResiduals values
// (QuadraticCostFunction<2, 2>).
struct QuadraticCostFunction {
    num_residuals: usize,
    sizes: Vec<usize>,
}

impl CostFunction for QuadraticCostFunction {
    fn num_residuals(&self) -> usize {
        self.num_residuals
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &self.sizes
    }
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        let n = self.num_residuals;
        for (i, r) in residuals[..n].iter_mut().enumerate() {
            *r = i as f64;
            for (j, p) in parameters.iter().enumerate() {
                *r -= (j as f64 + 1.0) * p[i] * p[i];
            }
        }
        let Some(jacobians) = jacobians else {
            return true;
        };
        for (j, jacobian) in jacobians.iter_mut().enumerate() {
            if let Some(jacobian) = jacobian {
                jacobian[..n * n].fill(0.0);
                for i in 0..n {
                    jacobian[i * n + i] = -2.0 * (j as f64 + 1.0) * parameters[j][i];
                }
            }
        }
        true
    }
}

struct Fixture {
    problem: Problem,
    parameter_blocks: Vec<ParameterBlockId>,
    residual_blocks: Vec<ResidualBlockId>,
}

fn set_up() -> Fixture {
    let mut problem = Problem::new();
    let parameter_blocks = vec![
        problem.add_parameter_block(&[1.0, 2.0]),
        problem.add_parameter_block(&[3.0, 4.0]),
        problem.add_parameter_block(&[5.0, 6.0]),
    ];
    let cost_function: Arc<dyn CostFunction> = Arc::new(QuadraticCostFunction {
        num_residuals: 2,
        sizes: vec![2, 2],
    });
    let p = &parameter_blocks;
    let residual_blocks = vec![
        // f(x, y)
        problem
            .add_residual_block(cost_function.clone(), None, &[p[0], p[1]])
            .unwrap(),
        // g(y, z)
        problem
            .add_residual_block(cost_function.clone(), None, &[p[1], p[2]])
            .unwrap(),
        // h(z, x)
        problem
            .add_residual_block(cost_function, None, &[p[2], p[0]])
            .unwrap(),
    ];
    Fixture {
        problem,
        parameter_blocks,
        residual_blocks,
    }
}

struct ExpectedEvaluation<'a> {
    num_rows: usize,
    num_cols: usize,
    cost: f64,
    residuals: &'a [f64],
    gradient: &'a [f64],
    jacobian: &'a [f64],
}

fn evaluate_and_compare(
    problem: &mut Problem,
    options: &EvaluateOptions,
    expected: &ExpectedEvaluation<'_>,
    want_residuals: bool,
    want_gradient: bool,
    want_jacobian: bool,
) {
    let mut residuals = Vec::new();
    let mut gradient = Vec::new();
    let mut jacobian = CrsMatrix::default();
    let cost = problem
        .evaluate(
            options,
            want_residuals.then_some(&mut residuals),
            want_gradient.then_some(&mut gradient),
            want_jacobian.then_some(&mut jacobian),
        )
        .unwrap()
        .expect("evaluation succeeds");

    // CompareEvaluations.
    assert_eq!(expected.cost, cost);
    if want_residuals {
        assert_eq!(residuals.len(), expected.num_rows);
        assert_eq!(residuals, expected.residuals);
    }
    if want_gradient {
        assert_eq!(gradient.len(), expected.num_cols);
        assert_eq!(gradient, expected.gradient);
    }
    if want_jacobian {
        assert_eq!(jacobian.num_rows, expected.num_rows);
        assert_eq!(jacobian.num_cols, expected.num_cols);

        // CRSToDenseMatrix, row-major.
        let mut dense = vec![0.0; jacobian.num_rows * jacobian.num_cols];
        for row in 0..jacobian.num_rows {
            for j in jacobian.rows[row]..jacobian.rows[row + 1] {
                dense[row * jacobian.num_cols + jacobian.cols[j]] = jacobian.values[j];
            }
        }
        assert_eq!(dense, expected.jacobian);
    }

    // TearDown: EXPECT_TRUE(problem_.program().IsValid()).
    assert!(problem.program().is_valid(problem.blocks()));
}

fn check_all_evaluation_combinations(
    problem: &mut Problem,
    options: &EvaluateOptions,
    expected: &ExpectedEvaluation<'_>,
) {
    for i in 0..8 {
        evaluate_and_compare(
            problem,
            options,
            expected,
            i & 1 != 0,
            i & 2 != 0,
            i & 4 != 0,
        );
    }
}

const ALL_RESIDUALS: [f64; 6] = [
    -19.0, -35.0, // f
    -59.0, -87.0, // g
    -27.0, -43.0, // h
];

const ALL_GRADIENT: [f64; 6] = [
    146.0, 484.0, // x
    582.0, 1256.0, // y
    1450.0, 2604.0, // z
];

#[rustfmt::skip]
const ALL_JACOBIAN: [f64; 36] = [
    //            x             y             z
    /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                   0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
    /* g(y, z) */  0.0,  0.0,  -6.0,   0.0, -20.0,   0.0,
                   0.0,  0.0,   0.0,  -8.0,   0.0, -24.0,
    /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                   0.0, -8.0,   0.0,   0.0,   0.0, -12.0,
];

const ALL: ExpectedEvaluation<'static> = ExpectedEvaluation {
    num_rows: 6,
    num_cols: 6,
    cost: 7607.0,
    residuals: &ALL_RESIDUALS,
    gradient: &ALL_GRADIENT,
    jacobian: &ALL_JACOBIAN,
};

#[test]
fn problem_evaluate_test_multiple_parameter_and_residual_blocks() {
    let mut f = set_up();
    check_all_evaluation_combinations(&mut f.problem, &EvaluateOptions::default(), &ALL);
}

#[test]
fn problem_evaluate_test_parameter_and_residual_blocks_passed_in_options() {
    let mut f = set_up();
    let options = EvaluateOptions {
        parameter_blocks: f.parameter_blocks.clone(),
        residual_blocks: f.residual_blocks.clone(),
        ..EvaluateOptions::default()
    };
    check_all_evaluation_combinations(&mut f.problem, &options, &ALL);
}

#[test]
fn problem_evaluate_test_reordered_residual_blocks() {
    let mut f = set_up();
    let r = &f.residual_blocks;
    let options = EvaluateOptions {
        parameter_blocks: f.parameter_blocks.clone(),
        // f, h, g
        residual_blocks: vec![r[0], r[2], r[1]],
        ..EvaluateOptions::default()
    };
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        num_rows: 6,
        num_cols: 6,
        cost: 7607.0,
        residuals: &[-19.0, -35.0, -27.0, -43.0, -59.0, -87.0],
        gradient: &ALL_GRADIENT,
        jacobian: &[
            /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
            /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0,   0.0,   0.0, -12.0,
            /* g(y, z) */  0.0,  0.0,  -6.0,   0.0, -20.0,   0.0,
                           0.0,  0.0,   0.0,  -8.0,   0.0, -24.0,
        ],
    };
    check_all_evaluation_combinations(&mut f.problem, &options, &expected);
}

#[test]
fn problem_evaluate_test_reordered_residual_blocks_and_reordered_parameter_blocks() {
    let mut f = set_up();
    let (p, r) = (&f.parameter_blocks, &f.residual_blocks);
    let options = EvaluateOptions {
        // z, y, x
        parameter_blocks: vec![p[2], p[1], p[0]],
        // f, h, g
        residual_blocks: vec![r[0], r[2], r[1]],
        ..EvaluateOptions::default()
    };
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        num_rows: 6,
        num_cols: 6,
        cost: 7607.0,
        residuals: &[-19.0, -35.0, -27.0, -43.0, -59.0, -87.0],
        gradient: &[1450.0, 2604.0, 582.0, 1256.0, 146.0, 484.0],
        jacobian: &[
            //             z             y             x
            /* f(x, y) */   0.0,   0.0, -12.0,   0.0, -2.0,  0.0,
                            0.0,   0.0,   0.0, -16.0,  0.0, -4.0,
            /* h(z, x) */ -10.0,   0.0,   0.0,   0.0, -4.0,  0.0,
                            0.0, -12.0,   0.0,   0.0,  0.0, -8.0,
            /* g(y, z) */ -20.0,   0.0,  -6.0,   0.0,  0.0,  0.0,
                            0.0, -24.0,   0.0,  -8.0,  0.0,  0.0,
        ],
    };
    check_all_evaluation_combinations(&mut f.problem, &options, &expected);
}

#[test]
fn problem_evaluate_test_constant_parameter_block() {
    let mut f = set_up();
    f.problem
        .set_parameter_block_constant(f.parameter_blocks[1])
        .unwrap();
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        num_rows: 6,
        num_cols: 6,
        cost: 7607.0,
        residuals: &ALL_RESIDUALS,
        gradient: &[146.0, 484.0, 0.0, 0.0, 1450.0, 2604.0],
        jacobian: &[
            /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0,   0.0,   0.0,   0.0,
            /* g(y, z) */  0.0,  0.0,   0.0,   0.0, -20.0,   0.0,
                           0.0,  0.0,   0.0,   0.0,   0.0, -24.0,
            /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0,   0.0,   0.0, -12.0,
        ],
    };
    check_all_evaluation_combinations(&mut f.problem, &EvaluateOptions::default(), &expected);
}

#[test]
fn problem_evaluate_test_excluded_a_residual_block() {
    let mut f = set_up();
    let r = &f.residual_blocks;
    let options = EvaluateOptions {
        residual_blocks: vec![r[0], r[2]],
        ..EvaluateOptions::default()
    };
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        num_rows: 4,
        num_cols: 6,
        cost: 2082.0,
        residuals: &[-19.0, -35.0, -27.0, -43.0],
        gradient: &[146.0, 484.0, 228.0, 560.0, 270.0, 516.0],
        jacobian: &[
            /* f(x, y) */ -2.0,  0.0, -12.0,   0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0, -16.0,   0.0,   0.0,
            /* h(z, x) */ -4.0,  0.0,   0.0,   0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0,   0.0,   0.0, -12.0,
        ],
    };
    check_all_evaluation_combinations(&mut f.problem, &options, &expected);
}

#[test]
fn problem_evaluate_test_excluded_parameter_block() {
    let mut f = set_up();
    let p = &f.parameter_blocks;
    let options = EvaluateOptions {
        // x, z
        parameter_blocks: vec![p[0], p[2]],
        residual_blocks: f.residual_blocks.clone(),
        ..EvaluateOptions::default()
    };
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        num_rows: 6,
        num_cols: 4,
        cost: 7607.0,
        residuals: &ALL_RESIDUALS,
        gradient: &[146.0, 484.0, 1450.0, 2604.0],
        jacobian: &[
            /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0,   0.0,
            /* g(y, z) */  0.0,  0.0, -20.0,   0.0,
                           0.0,  0.0,   0.0, -24.0,
            /* h(z, x) */ -4.0,  0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0, -12.0,
        ],
    };
    check_all_evaluation_combinations(&mut f.problem, &options, &expected);
}

#[test]
fn problem_evaluate_test_excluded_parameter_block_and_excluded_residual_block() {
    let mut f = set_up();
    let (p, r) = (&f.parameter_blocks, &f.residual_blocks);
    let options = EvaluateOptions {
        // x, z
        parameter_blocks: vec![p[0], p[2]],
        residual_blocks: vec![r[0], r[1]],
        ..EvaluateOptions::default()
    };
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        num_rows: 4,
        num_cols: 4,
        cost: 6318.0,
        residuals: &[-19.0, -35.0, -59.0, -87.0],
        gradient: &[38.0, 140.0, 1180.0, 2088.0],
        jacobian: &[
            /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,
                           0.0, -4.0,   0.0,   0.0,
            /* g(y, z) */  0.0,  0.0, -20.0,   0.0,
                           0.0,  0.0,   0.0, -24.0,
        ],
    };
    check_all_evaluation_combinations(&mut f.problem, &options, &expected);
}

#[test]
fn problem_evaluate_test_manifold() {
    let mut f = set_up();
    f.problem
        .set_manifold(
            f.parameter_blocks[1],
            Some(Arc::new(SubsetManifold::new(2, &[0]).unwrap())),
        )
        .unwrap();
    #[rustfmt::skip]
    let expected = ExpectedEvaluation {
        num_rows: 6,
        num_cols: 5,
        cost: 7607.0,
        residuals: &ALL_RESIDUALS,
        gradient: &[146.0, 484.0, 1256.0, 1450.0, 2604.0],
        jacobian: &[
            //             x      y             z
            /* f(x, y) */ -2.0,  0.0,   0.0,   0.0,   0.0,
                           0.0, -4.0, -16.0,   0.0,   0.0,
            /* g(y, z) */  0.0,  0.0,   0.0, -20.0,   0.0,
                           0.0,  0.0,  -8.0,   0.0, -24.0,
            /* h(z, x) */ -4.0,  0.0,   0.0, -10.0,   0.0,
                           0.0, -8.0,   0.0,   0.0, -12.0,
        ],
    };
    check_all_evaluation_combinations(&mut f.problem, &EvaluateOptions::default(), &expected);
}
