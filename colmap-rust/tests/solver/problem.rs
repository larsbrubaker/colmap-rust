// Port of Ceres Solver 2.2.0's internal/ceres/problem_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md): `colmap_rust::solver::Problem` and `Problem::evaluate`, same names
// and values. The problem owns its parameter values and names blocks by id where Ceres takes
// the user's `double*` (divergence 306), so a test that adds "the same pointer" again uses the
// id it got back, an "unknown pointer" is an id the problem never handed out, and every death
// test becomes "returns Err". The ProblemEvaluateTest cases run every combination of residuals,
// gradient and Jacobian (problem_evaluate.rs).
//
// Not ported, with the reason:
// - AddResidualWithNullCostFunctionDies: an `Arc<dyn CostFunction>` cannot be null.
// - AddParameterWithDifferentSizesOnTheSameVariableDies, AddParameterWithAliasedParametersDies,
//   AddParameterIgnoresDuplicateCalls, RepeatedAddParameterBlockResetsManifold: each add makes
//   a new block with its own storage, so there is no second add of one block to check
//   (AddResidualAddsDuplicatedParametersOnlyOnce covers reusing a block; SetManifoldTwice and
//   SetManifoldAndThenClearItWithNull cover changing its manifold).
// - CostFunctionsAreDeletedEvenWithRemovals, RemoveParameterBlockWithUnknownPtrDies, the
//   DynamicProblem cases and ParameterBlockQueryTest's removal half: block removal is not
//   ported (COLMAP never removes blocks); the query half is ported.
// - ProblemEvaluateResidualBlockTest and the evaluation-callback tests: EvaluateResidualBlock
//   and EvaluationCallback are not ported (COLMAP uses neither).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use colmap_rust::solver::{
    CostFunction, EuclideanManifold, LossFunction, Manifold, ParameterBlockId, Problem,
    SubsetManifold, TrivialLoss,
};

// The unary, binary and ternary cost functions of Ceres' test: residuals all 1, 2 or 3.
struct ConstantCostFunction {
    value: f64,
    num_residuals: usize,
    sizes: Vec<usize>,
}

impl CostFunction for ConstantCostFunction {
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
        residuals[..self.num_residuals].fill(self.value);
        true
    }
}

fn unary(num_residuals: usize, size: usize) -> Arc<dyn CostFunction> {
    Arc::new(ConstantCostFunction {
        value: 1.0,
        num_residuals,
        sizes: vec![size],
    })
}

fn binary(num_residuals: usize, size1: usize, size2: usize) -> Arc<dyn CostFunction> {
    Arc::new(ConstantCostFunction {
        value: 2.0,
        num_residuals,
        sizes: vec![size1, size2],
    })
}

fn ternary(
    num_residuals: usize,
    size1: usize,
    size2: usize,
    size3: usize,
) -> Arc<dyn CostFunction> {
    Arc::new(ConstantCostFunction {
        value: 3.0,
        num_residuals,
        sizes: vec![size1, size2, size3],
    })
}

#[test]
fn problem_move_constructor() {
    let mut src = Problem::new();
    let x = src.add_parameter_block(&[0.0]);
    let dst = src;
    assert!(dst.has_parameter_block(x));
}

#[test]
fn problem_move_assignment() {
    let mut src = Problem::new();
    let x = src.add_parameter_block(&[0.0]);
    let mut dst = Problem::new();
    assert!(!dst.has_parameter_block(x));
    dst = src;
    assert!(dst.has_parameter_block(x));
}

#[test]
fn problem_add_residual_with_incorrect_number_of_parameter_blocks_dies() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 4]);
    problem.add_parameter_block(&[0.0; 5]);

    // UnaryCostFunction takes only one parameter, but two are passed.
    let error = problem
        .add_residual_block(unary(2, 3), None, &[x, y])
        .unwrap_err();
    assert!(error.message().contains("num_parameter_blocks"), "{error}");
}

#[test]
fn problem_add_residual_with_different_sizes_on_the_same_variable_dies() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    problem.add_residual_block(unary(2, 3), None, &[x]).unwrap();
    let error = problem
        .add_residual_block(unary(2, 4 /* 4 != 3 */), None, &[x])
        .unwrap_err();
    assert!(
        error
            .message()
            .contains("of size 4 but was given a block of size 3"),
        "{error}"
    );
}

#[test]
fn problem_add_residual_with_duplicate_parameters_dies() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let z = problem.add_parameter_block(&[0.0; 5]);
    let error = problem
        .add_residual_block(binary(2, 3, 3), None, &[x, x])
        .unwrap_err();
    assert!(
        error.message().contains("Duplicate parameter blocks"),
        "{error}"
    );
    let error = problem
        .add_residual_block(ternary(1, 5, 3, 5), None, &[z, x, z])
        .unwrap_err();
    assert!(
        error.message().contains("Duplicate parameter blocks"),
        "{error}"
    );
}

#[test]
fn problem_add_residual_with_incorrect_sizes_of_parameter_block_dies() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    problem.add_parameter_block(&[0.0; 4]);
    let z = problem.add_parameter_block(&[0.0; 5]);

    // The cost function expects the size of the second parameter, z, to be 4 instead of 5 as
    // declared above. This is fatal.
    let error = problem
        .add_residual_block(binary(2, 3, 4), None, &[x, z])
        .unwrap_err();
    assert!(
        error.message().contains("was given a block of size 5"),
        "{error}"
    );
}

#[test]
fn problem_add_residual_adds_duplicated_parameters_only_once() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 4]);
    let z = problem.add_parameter_block(&[0.0; 5]);
    problem.add_residual_block(unary(2, 3), None, &[x]).unwrap();
    problem.add_residual_block(unary(2, 3), None, &[x]).unwrap();
    problem.add_residual_block(unary(2, 4), None, &[y]).unwrap();
    problem.add_residual_block(unary(2, 5), None, &[z]).unwrap();

    assert_eq!(3, problem.num_parameter_blocks());
    assert_eq!(12, problem.num_parameters());
}

// Counts its drops, as DestructorCountingCostFunction counts its destructor calls.
struct DestructorCountingCostFunction {
    num_destructions: Arc<AtomicUsize>,
}

impl Drop for DestructorCountingCostFunction {
    fn drop(&mut self) {
        self.num_destructions.fetch_add(1, Ordering::Relaxed);
    }
}

impl CostFunction for DestructorCountingCostFunction {
    fn num_residuals(&self) -> usize {
        3
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &[4, 5]
    }
    fn evaluate(
        &self,
        _parameters: &[&[f64]],
        _residuals: &mut [f64],
        _jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        true
    }
}

#[test]
fn problem_reused_cost_functions_are_only_deleted_once() {
    let num_destructions = Arc::new(AtomicUsize::new(0));

    // Add a cost function multiple times and check to make sure that the destructor on the
    // cost function is only called once.
    {
        let mut problem = Problem::new();
        let y = problem.add_parameter_block(&[0.0; 4]);
        let z = problem.add_parameter_block(&[0.0; 5]);

        let cost: Arc<dyn CostFunction> = Arc::new(DestructorCountingCostFunction {
            num_destructions: num_destructions.clone(),
        });
        problem
            .add_residual_block(cost.clone(), None, &[y, z])
            .unwrap();
        problem
            .add_residual_block(cost.clone(), None, &[y, z])
            .unwrap();
        problem.add_residual_block(cost, None, &[y, z]).unwrap();
        assert_eq!(3, problem.num_residual_blocks());
    }

    // Check that the destructor was called only once.
    assert_eq!(num_destructions.load(Ordering::Relaxed), 1);
}

#[test]
fn problem_get_cost_function_for_residual_block() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let cost_function = unary(2, 3);
    let residual_block = problem
        .add_residual_block(cost_function.clone(), None, &[x])
        .unwrap();
    assert!(Arc::ptr_eq(
        problem
            .get_cost_function_for_residual_block(residual_block)
            .unwrap(),
        &cost_function
    ));
    assert!(problem
        .get_loss_function_for_residual_block(residual_block)
        .unwrap()
        .is_none());
}

#[test]
fn problem_get_loss_function_for_residual_block() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let cost_function = unary(2, 3);
    let loss_function: Arc<dyn LossFunction> = Arc::new(TrivialLoss);
    let residual_block = problem
        .add_residual_block(cost_function.clone(), Some(loss_function.clone()), &[x])
        .unwrap();
    assert!(Arc::ptr_eq(
        problem
            .get_cost_function_for_residual_block(residual_block)
            .unwrap(),
        &cost_function
    ));
    assert!(Arc::ptr_eq(
        problem
            .get_loss_function_for_residual_block(residual_block)
            .unwrap()
            .unwrap(),
        &loss_function
    ));
}

// A block id the problem with one block never handed out (Ceres: an unknown pointer).
const UNKNOWN: ParameterBlockId = ParameterBlockId(1);

#[test]
fn problem_set_parameter_block_constant_with_unknown_ptr_dies() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0; 3]);
    let error = problem.set_parameter_block_constant(UNKNOWN).unwrap_err();
    assert!(
        error.message().contains("Parameter block not found:"),
        "{error}"
    );
}

#[test]
fn problem_set_parameter_block_variable_with_unknown_ptr_dies() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0; 3]);
    let error = problem.set_parameter_block_variable(UNKNOWN).unwrap_err();
    assert!(
        error.message().contains("Parameter block not found:"),
        "{error}"
    );
}

#[test]
fn problem_is_parameter_block_constant() {
    let mut problem = Problem::new();
    let x1 = problem.add_parameter_block(&[0.0; 3]);
    let x2 = problem.add_parameter_block(&[0.0; 3]);

    assert!(!problem.is_parameter_block_constant(x1).unwrap());
    assert!(!problem.is_parameter_block_constant(x2).unwrap());

    problem.set_parameter_block_constant(x1).unwrap();
    assert!(problem.is_parameter_block_constant(x1).unwrap());
    assert!(!problem.is_parameter_block_constant(x2).unwrap());

    problem.set_parameter_block_constant(x2).unwrap();
    assert!(problem.is_parameter_block_constant(x1).unwrap());
    assert!(problem.is_parameter_block_constant(x2).unwrap());

    problem.set_parameter_block_variable(x1).unwrap();
    assert!(!problem.is_parameter_block_constant(x1).unwrap());
    assert!(problem.is_parameter_block_constant(x2).unwrap());
}

#[test]
fn problem_is_parameter_block_constant_with_unknown_ptr_dies() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0; 3]);
    let error = problem.is_parameter_block_constant(UNKNOWN).unwrap_err();
    assert!(
        error.message().contains("Parameter block not found:"),
        "{error}"
    );
}

#[test]
fn problem_set_manifold_with_unknown_ptr_dies() {
    let mut problem = Problem::new();
    problem.add_parameter_block(&[0.0; 3]);
    let error = problem
        .set_manifold(UNKNOWN, Some(Arc::new(EuclideanManifold::new(3))))
        .unwrap_err();
    assert!(
        error.message().contains("Parameter block not found:"),
        "{error}"
    );
}

#[test]
fn problem_get_manifold() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 2]);

    let manifold: Arc<dyn Manifold> = Arc::new(EuclideanManifold::new(3));
    problem.set_manifold(x, Some(manifold.clone())).unwrap();
    assert!(Arc::ptr_eq(
        problem.get_manifold(x).unwrap().unwrap(),
        &manifold
    ));
    assert!(problem.get_manifold(y).unwrap().is_none());
}

#[test]
fn problem_has_manifold() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 2]);

    problem
        .set_manifold(x, Some(Arc::new(EuclideanManifold::new(3))))
        .unwrap();
    assert!(problem.has_manifold(x).unwrap());
    assert!(!problem.has_manifold(y).unwrap());
}

fn subset(size: usize, constant: &[usize]) -> Option<Arc<dyn Manifold>> {
    Some(Arc::new(SubsetManifold::new(size, constant).unwrap()))
}

#[test]
fn problem_parameter_block_query_test_using_manifold() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 4]);

    problem.set_manifold(x, subset(3, &[0])).unwrap();
    assert_eq!(problem.parameter_block_size(x).unwrap(), 3);
    assert_eq!(problem.parameter_block_tangent_size(x).unwrap(), 2);
    assert_eq!(problem.parameter_block_tangent_size(y).unwrap(), 4);

    let parameter_blocks = problem.get_parameter_blocks();
    assert_eq!(parameter_blocks.len(), 2);
    assert_ne!(parameter_blocks[0], parameter_blocks[1]);
    assert!(parameter_blocks[0] == x || parameter_blocks[0] == y);
    assert!(parameter_blocks[1] == x || parameter_blocks[1] == y);

    assert!(problem.has_parameter_block(x));
}

#[test]
fn problem_parameter_block_query_test() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0; 3]);
    let y = problem.add_parameter_block(&[0.0; 4]);

    problem.set_manifold(x, subset(3, &[0])).unwrap();
    assert_eq!(problem.parameter_block_size(x).unwrap(), 3);
    assert_eq!(problem.parameter_block_tangent_size(x).unwrap(), 2);
    assert_eq!(problem.parameter_block_tangent_size(y).unwrap(), 4);

    let parameter_blocks = problem.get_parameter_blocks();
    assert_eq!(parameter_blocks.len(), 2);
    assert_ne!(parameter_blocks[0], parameter_blocks[1]);
    assert!(parameter_blocks[0] == x || parameter_blocks[0] == y);
    assert!(parameter_blocks[1] == x || parameter_blocks[1] == y);

    assert!(problem.has_parameter_block(x));
}

#[test]
fn problem_set_and_get_parameter_lower_bound() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0]);

    assert_eq!(-f64::MAX, problem.get_parameter_lower_bound(x, 0).unwrap());
    assert_eq!(-f64::MAX, problem.get_parameter_lower_bound(x, 1).unwrap());

    problem.set_parameter_lower_bound(x, 0, -1.0).unwrap();
    assert_eq!(-1.0, problem.get_parameter_lower_bound(x, 0).unwrap());
    assert_eq!(-f64::MAX, problem.get_parameter_lower_bound(x, 1).unwrap());

    problem.set_parameter_lower_bound(x, 0, -2.0).unwrap();
    assert_eq!(-2.0, problem.get_parameter_lower_bound(x, 0).unwrap());
    assert_eq!(-f64::MAX, problem.get_parameter_lower_bound(x, 1).unwrap());

    problem.set_parameter_lower_bound(x, 0, -f64::MAX).unwrap();
    assert_eq!(-f64::MAX, problem.get_parameter_lower_bound(x, 0).unwrap());
    assert_eq!(-f64::MAX, problem.get_parameter_lower_bound(x, 1).unwrap());
}

#[test]
fn problem_set_and_get_parameter_upper_bound() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0]);

    assert_eq!(f64::MAX, problem.get_parameter_upper_bound(x, 0).unwrap());
    assert_eq!(f64::MAX, problem.get_parameter_upper_bound(x, 1).unwrap());

    problem.set_parameter_upper_bound(x, 0, -1.0).unwrap();
    assert_eq!(-1.0, problem.get_parameter_upper_bound(x, 0).unwrap());
    assert_eq!(f64::MAX, problem.get_parameter_upper_bound(x, 1).unwrap());

    problem.set_parameter_upper_bound(x, 0, -2.0).unwrap();
    assert_eq!(-2.0, problem.get_parameter_upper_bound(x, 0).unwrap());
    assert_eq!(f64::MAX, problem.get_parameter_upper_bound(x, 1).unwrap());

    problem.set_parameter_upper_bound(x, 0, f64::MAX).unwrap();
    assert_eq!(f64::MAX, problem.get_parameter_upper_bound(x, 0).unwrap());
    assert_eq!(f64::MAX, problem.get_parameter_upper_bound(x, 1).unwrap());
}

#[test]
fn problem_set_manifold_twice() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0, 3.0]);
    problem.set_manifold(x, subset(3, &[1])).unwrap();
    assert_eq!(problem.get_manifold(x).unwrap().unwrap().ambient_size(), 3);
    assert_eq!(problem.get_manifold(x).unwrap().unwrap().tangent_size(), 2);

    problem.set_manifold(x, subset(3, &[0, 1])).unwrap();
    assert_eq!(problem.get_manifold(x).unwrap().unwrap().ambient_size(), 3);
    assert_eq!(problem.get_manifold(x).unwrap().unwrap().tangent_size(), 1);
}

#[test]
fn problem_set_manifold_and_then_clear_it_with_null() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[1.0, 2.0, 3.0]);
    problem.set_manifold(x, subset(3, &[1])).unwrap();
    assert_eq!(problem.get_manifold(x).unwrap().unwrap().ambient_size(), 3);
    assert_eq!(problem.get_manifold(x).unwrap().unwrap().tangent_size(), 2);

    problem.set_manifold(x, None).unwrap();
    assert!(problem.get_manifold(x).unwrap().is_none());
    assert_eq!(problem.parameter_block_tangent_size(x).unwrap(), 3);
    assert_eq!(problem.parameter_block_size(x).unwrap(), 3);
}

#[test]
fn solver_zero_tangent_sized_manifold_means_parameter_block_is_constant() {
    let mut problem = Problem::new();
    let x = problem.add_parameter_block(&[0.0]);
    let y = problem.add_parameter_block(&[1.0]);
    problem
        .add_residual_block(binary(1, 1, 1), None, &[x, y])
        .unwrap();
    problem.set_manifold(y, subset(1, &[0])).unwrap();
    assert!(problem.is_parameter_block_constant(y).unwrap());
}
