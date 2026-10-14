// Port of Ceres Solver 2.2.0's internal/ceres/program_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md): `colmap_rust::solver::Program`, same names and values. Ceres builds
// the problems over uninitialized doubles where the values do not matter; here they start at
// 0. Removed parameter blocks are ids where Ceres returns the user's pointers.
// Not ported: BlockJacobianTest.CreateJacobianBlockSparsityTranspose and
// Program.ReallocationInCreateJacobianBlockSparsityTranspose (the transposed sparsity pattern
// only feeds SuiteSparse's orderings, which are not ported).

use std::sync::Arc;

use colmap_rust::solver::{CostFunction, ParameterValues, Problem};

// A cost function that simply returns its argument.
struct UnaryIdentityCostFunction;

impl CostFunction for UnaryIdentityCostFunction {
    fn num_residuals(&self) -> usize {
        1
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &[1]
    }
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        residuals[0] = parameters[0][0];
        if let Some(jacobian) = jacobians.and_then(|j| j[0].as_deref_mut()) {
            jacobian[0] = 1.0;
        }
        true
    }
}

// MockCostFunctionBase<kNumResiduals, Ns...>: every residual is kNumResiduals + sum(Ns).
struct MockCostFunction {
    num_residuals: usize,
    sizes: Vec<usize>,
}

impl CostFunction for MockCostFunction {
    fn num_residuals(&self) -> usize {
        self.num_residuals
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &self.sizes
    }
    fn evaluate(
        &self,
        _parameters: &[&[f64]],
        residuals: &mut [f64],
        _jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        let num_parameters: usize = self.sizes.iter().sum();
        residuals[..self.num_residuals].fill((self.num_residuals + num_parameters) as f64);
        true
    }
}

fn mock(num_residuals: usize, sizes: &[usize]) -> Arc<dyn CostFunction> {
    Arc::new(MockCostFunction {
        num_residuals,
        sizes: sizes.to_vec(),
    })
}

fn unary() -> Arc<dyn CostFunction> {
    mock(2, &[1])
}

fn binary() -> Arc<dyn CostFunction> {
    mock(2, &[1, 1])
}

fn ternary() -> Arc<dyn CostFunction> {
    mock(2, &[1, 1, 1])
}

// Runs CreateReducedProgram on the problem's program: (num parameter blocks, num residual
// blocks, removed blocks, fixed cost).
fn reduce(
    problem: &mut Problem,
) -> (
    usize,
    usize,
    Vec<colmap_rust::solver::ParameterBlockId>,
    f64,
) {
    let (program, blocks) = problem.program_and_blocks_mut();
    let mut removed_parameter_blocks = Vec::new();
    let mut fixed_cost = 0.0;
    let mut message = String::new();
    let reduced_program = program
        .create_reduced_program(
            blocks,
            &mut removed_parameter_blocks,
            &mut fixed_cost,
            &mut message,
        )
        .unwrap()
        .unwrap();
    (
        reduced_program.num_parameter_blocks(),
        reduced_program.num_residual_blocks(),
        removed_parameter_blocks,
        fixed_cost,
    )
}

#[test]
fn program_remove_fixed_blocks_nothing_constant() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0]);
    let y = problem.add_parameter_block(&[0.0]);
    let z = problem.add_parameter_block(&[0.0]);
    problem.add_residual_block(unary(), None, &[x]).unwrap();
    problem.add_residual_block(binary(), None, &[x, y]).unwrap();
    problem
        .add_residual_block(ternary(), None, &[x, y, z])
        .unwrap();

    let (num_parameter_blocks, num_residual_blocks, removed, fixed_cost) = reduce(&mut problem);
    assert_eq!(num_parameter_blocks, 3);
    assert_eq!(num_residual_blocks, 3);
    assert_eq!(removed.len(), 0);
    assert_eq!(fixed_cost, 0.0);
}

#[test]
fn program_remove_fixed_blocks_all_parameter_blocks_constant() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0]);
    problem.add_residual_block(unary(), None, &[x]).unwrap();
    problem.set_parameter_block_constant(x).unwrap();

    let (num_parameter_blocks, num_residual_blocks, removed, fixed_cost) = reduce(&mut problem);
    assert_eq!(num_parameter_blocks, 0);
    assert_eq!(num_residual_blocks, 0);
    assert_eq!(removed.len(), 1);
    assert_eq!(removed[0], x);
    assert_eq!(fixed_cost, 9.0);
}

#[test]
fn program_remove_fixed_blocks_no_residual_blocks() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0]);
    problem.add_parameter_block(&[0.0]);
    problem.add_parameter_block(&[0.0]);

    let (num_parameter_blocks, num_residual_blocks, removed, fixed_cost) = reduce(&mut problem);
    assert_eq!(num_parameter_blocks, 0);
    assert_eq!(num_residual_blocks, 0);
    assert_eq!(removed.len(), 3);
    assert_eq!(fixed_cost, 0.0);
}

#[test]
fn program_remove_fixed_blocks_one_parameter_block_constant() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0]);
    let y = problem.add_parameter_block(&[0.0]);
    problem.add_parameter_block(&[0.0]);

    problem.add_residual_block(unary(), None, &[x]).unwrap();
    problem.add_residual_block(binary(), None, &[x, y]).unwrap();
    problem.set_parameter_block_constant(x).unwrap();

    let (num_parameter_blocks, num_residual_blocks, _, _) = reduce(&mut problem);
    assert_eq!(num_parameter_blocks, 1);
    assert_eq!(num_residual_blocks, 1);
}

#[test]
fn program_remove_fixed_blocks_num_eliminate_blocks() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0]);
    let y = problem.add_parameter_block(&[0.0]);
    let z = problem.add_parameter_block(&[0.0]);
    problem.add_residual_block(unary(), None, &[x]).unwrap();
    problem
        .add_residual_block(ternary(), None, &[x, y, z])
        .unwrap();
    problem.add_residual_block(binary(), None, &[x, y]).unwrap();
    problem.set_parameter_block_constant(x).unwrap();

    let (num_parameter_blocks, num_residual_blocks, _, _) = reduce(&mut problem);
    assert_eq!(num_parameter_blocks, 2);
    assert_eq!(num_residual_blocks, 2);
}

#[test]
fn program_remove_fixed_blocks_fixed_cost() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.23]);
    let y = problem.add_parameter_block(&[4.56]);
    let z = problem.add_parameter_block(&[7.89]);
    problem
        .add_residual_block(Arc::new(UnaryIdentityCostFunction), None, &[x])
        .unwrap();
    problem
        .add_residual_block(ternary(), None, &[x, y, z])
        .unwrap();
    problem.add_residual_block(binary(), None, &[x, y]).unwrap();
    problem.set_parameter_block_constant(x).unwrap();

    let expected_fixed_cost = {
        let blocks = problem.blocks();
        let expected_removed_block = blocks.residual_block(problem.program().residual_blocks[0]);
        let mut scratch = vec![
            0.0;
            expected_removed_block
                .num_scratch_doubles_for_evaluate(&blocks.parameter_blocks)
        ];
        let values = ParameterValues {
            user: &blocks.values,
            state: None,
        };
        expected_removed_block
            .evaluate(
                &blocks.parameter_blocks,
                values,
                true,
                None,
                None,
                &mut scratch,
            )
            .unwrap()
            .unwrap()
    };

    let (num_parameter_blocks, num_residual_blocks, _, fixed_cost) = reduce(&mut problem);
    assert_eq!(num_parameter_blocks, 2);
    assert_eq!(num_residual_blocks, 2);
    // EXPECT_DOUBLE_EQ: within 4 ulps.
    assert!(
        (fixed_cost - expected_fixed_cost).abs() <= 4.0 * f64::EPSILON * expected_fixed_cost.abs(),
        "{fixed_cost} vs {expected_fixed_cost}"
    );
}

#[test]
fn program_problem_has_nan_parameter_blocks() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, f64::NAN]);
    problem
        .add_residual_block(mock(1, &[2]), None, &[x])
        .unwrap();
    let error = problem
        .program()
        .parameter_blocks_are_finite(problem.blocks())
        .unwrap_err();
    assert!(error.contains("has at least one invalid value"), "{error}");
}

#[test]
fn program_infeasible_parameter_block() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0, 0.0]);
    problem
        .add_residual_block(mock(1, &[2]), None, &[x])
        .unwrap();
    problem.set_parameter_lower_bound(x, 0, 2.0).unwrap();
    problem.set_parameter_upper_bound(x, 0, 1.0).unwrap();
    let error = problem.program().is_feasible(problem.blocks()).unwrap_err();
    assert!(error.contains("infeasible bound"), "{error}");
}

#[test]
fn program_infeasible_constant_parameter_block() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0, 0.0]);
    problem
        .add_residual_block(mock(1, &[2]), None, &[x])
        .unwrap();
    problem.set_parameter_lower_bound(x, 0, 1.0).unwrap();
    problem.set_parameter_upper_bound(x, 0, 2.0).unwrap();
    problem.set_parameter_block_constant(x).unwrap();
    let error = problem.program().is_feasible(problem.blocks()).unwrap_err();
    assert!(error.contains("infeasible value"), "{error}");
}

// The message layouts of Ceres' StringPrintf calls (%e bounds, "%12g " values), with the block
// id where Ceres prints the user's pointer.
#[test]
fn rust_only_program_feasibility_messages_follow_ceres() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.5, 0.0]);
    problem
        .add_residual_block(mock(1, &[2]), None, &[x])
        .unwrap();
    problem.set_parameter_lower_bound(x, 0, 1.0).unwrap();
    problem.set_parameter_upper_bound(x, 0, 2.0).unwrap();
    problem.set_parameter_block_constant(x).unwrap();
    assert_eq!(
        problem.program().is_feasible(problem.blocks()).unwrap_err(),
        "ParameterBlock: 0 with size 2 has at least one infeasible value.\nFirst infeasible value is at index: 0.\nLower bound: 1.000000e+00, value: 5.000000e-01, upper bound: 2.000000e+00\nParameter block values:          0.5            0 "
    );
}
