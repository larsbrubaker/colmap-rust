// Port of Ceres Solver 2.2.0's internal/ceres/parameter_block_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md): `colmap_rust::solver::ParameterBlock`, same names and values. A
// Ceres block wraps the user's `double*`; here it is built over an offset into a value arena
// and given its current values where Ceres reads them (`set_manifold`, `set_state`). The
// death test becomes "returns Err". Manifold identity (Ceres compares `Manifold*`) is
// `Arc::ptr_eq`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use colmap_rust::solver::{Manifold, ParameterBlock, SubsetManifold};

fn subset(size: usize, constant: &[usize]) -> Arc<dyn Manifold> {
    Arc::new(SubsetManifold::new(size, constant).unwrap())
}

fn same(a: Option<&Arc<dyn Manifold>>, b: &Arc<dyn Manifold>) -> bool {
    a.is_some_and(|a| Arc::ptr_eq(a, b))
}

#[test]
fn parameter_block_set_manifold_dies_on_size_mismatch() {
    let x = [1.0, 2.0, 3.0];
    let mut parameter_block = ParameterBlock::new(0, 3, -1);
    let subset_wrong_size = subset(4, &[1]);
    let error = parameter_block
        .set_manifold(Some(subset_wrong_size), &x)
        .unwrap_err();
    assert!(error.message().contains("ambient"), "{error}");
}

#[test]
fn parameter_block_set_manifold_with_same_existing_manifold() {
    let x = [1.0, 2.0, 3.0];
    let mut parameter_block = ParameterBlock::new(0, 3, -1);
    let subset = subset(3, &[1]);
    parameter_block
        .set_manifold(Some(subset.clone()), &x)
        .unwrap();
    parameter_block.set_manifold(Some(subset), &x).unwrap();
}

#[test]
fn parameter_block_set_manifold_allows_resetting_to_null() {
    let x = [1.0, 2.0, 3.0];
    let mut parameter_block = ParameterBlock::new(0, 3, -1);
    let subset = subset(3, &[1]);
    parameter_block
        .set_manifold(Some(subset.clone()), &x)
        .unwrap();
    assert!(same(parameter_block.manifold(), &subset));
    parameter_block.set_manifold(None, &x).unwrap();
    assert!(parameter_block.manifold().is_none());
    assert!(parameter_block.plus_jacobian().is_none());
}

#[test]
fn parameter_block_set_manifold_allows_resetting_to_different_manifold() {
    let x = [1.0, 2.0, 3.0];
    let mut parameter_block = ParameterBlock::new(0, 3, -1);
    let subset = subset(3, &[1]);
    parameter_block
        .set_manifold(Some(subset.clone()), &x)
        .unwrap();
    assert!(same(parameter_block.manifold(), &subset));

    let subset_different = self::subset(3, &[1]);
    parameter_block
        .set_manifold(Some(subset_different.clone()), &x)
        .unwrap();
    assert!(same(parameter_block.manifold(), &subset_different));
}

#[test]
fn parameter_block_set_manifold_and_normal_operation() {
    let x = [1.0, 2.0, 3.0];
    let mut parameter_block = ParameterBlock::new(0, 3, -1);
    parameter_block
        .set_manifold(Some(subset(3, &[1])), &x)
        .unwrap();

    // Ensure the manifold plus jacobian result is correctly computed (row-major 3 x 2).
    let manifold_jacobian = parameter_block.plus_jacobian().unwrap();
    assert_eq!(1.0, manifold_jacobian[0]);
    assert_eq!(0.0, manifold_jacobian[1]);
    assert_eq!(0.0, manifold_jacobian[2]);
    assert_eq!(0.0, manifold_jacobian[3]);
    assert_eq!(0.0, manifold_jacobian[4]);
    assert_eq!(1.0, manifold_jacobian[5]);

    // Check that updating works as expected.
    let mut x_plus_delta = [0.0; 3];
    let delta = [0.5, 0.3];
    parameter_block.plus(&x, &delta, &mut x_plus_delta);
    assert_eq!(1.5, x_plus_delta[0]);
    assert_eq!(2.0, x_plus_delta[1]);
    assert_eq!(3.3, x_plus_delta[2]);
}

struct TestManifold;

impl Manifold for TestManifold {
    fn ambient_size(&self) -> usize {
        1
    }
    fn tangent_size(&self) -> usize {
        1
    }
    fn plus(&self, _x: &[f64], _delta: &[f64], _x_plus_delta: &mut [f64]) -> bool {
        panic!("Shouldn't get called.");
    }
    fn plus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        jacobian[0] = x[0] * 2.0;
        true
    }
    fn minus(&self, _y: &[f64], _x: &[f64], _y_minus_x: &mut [f64]) -> bool {
        panic!("Shouldn't get called");
    }
    fn minus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        jacobian[0] = x[0] * 2.0;
        true
    }
}

#[test]
fn parameter_block_set_state_updates_plus_jacobian() {
    let mut x = [1.0];
    let mut parameter_block = ParameterBlock::new(0, 1, -1);
    parameter_block
        .set_manifold(Some(Arc::new(TestManifold)), &x)
        .unwrap();

    assert_eq!(2.0, parameter_block.plus_jacobian().unwrap()[0]);

    x[0] = 5.5;
    assert!(parameter_block.set_state(&x));
    assert_eq!(11.0, parameter_block.plus_jacobian().unwrap()[0]);
}

#[test]
fn parameter_block_plus_with_no_manifold() {
    let x = [1.0, 2.0];
    let parameter_block = ParameterBlock::new(0, 2, -1);

    let delta = [0.2, 0.3];
    let mut x_plus_delta = [0.0; 2];
    parameter_block.plus(&x, &delta, &mut x_plus_delta);
    assert_eq!(1.2, x_plus_delta[0]);
    assert_eq!(2.3, x_plus_delta[1]);
}

// Stops computing the plus_jacobian after the first time.
#[derive(Default)]
struct BadManifold {
    calls: AtomicUsize,
}

impl Manifold for BadManifold {
    fn ambient_size(&self) -> usize {
        1
    }
    fn tangent_size(&self) -> usize {
        1
    }
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        x_plus_delta[0] = x[0] + delta[0];
        true
    }
    fn plus_jacobian(&self, _x: &[f64], jacobian: &mut [f64]) -> bool {
        if self.calls.fetch_add(1, Ordering::Relaxed) == 0 {
            jacobian[0] = 0.0;
        }
        true
    }
    fn minus(&self, _y: &[f64], _x: &[f64], _y_minus_x: &mut [f64]) -> bool {
        panic!("Shouldn't get called");
    }
    fn minus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        jacobian[0] = x[0] * 2.0;
        true
    }
}

#[test]
fn parameter_block_detect_bad_manifold() {
    let x = [1.0];
    let mut parameter_block = ParameterBlock::new(0, 1, -1);
    parameter_block
        .set_manifold(Some(Arc::new(BadManifold::default())), &x)
        .unwrap();
    let y = [2.0];
    assert!(!parameter_block.set_state(&y));
}

#[test]
fn parameter_block_default_bounds() {
    let parameter_block = ParameterBlock::new(0, 2, -1);
    assert_eq!(parameter_block.upper_bound_for_parameter(0), f64::MAX);
    assert_eq!(parameter_block.upper_bound_for_parameter(1), f64::MAX);
    assert_eq!(parameter_block.lower_bound_for_parameter(0), -f64::MAX);
    assert_eq!(parameter_block.lower_bound_for_parameter(1), -f64::MAX);
}

#[test]
fn parameter_block_set_bounds() {
    let mut parameter_block = ParameterBlock::new(0, 2, -1);
    parameter_block.set_lower_bound(0, 1.0).unwrap();
    parameter_block.set_upper_bound(1, 1.0).unwrap();

    assert_eq!(parameter_block.lower_bound_for_parameter(0), 1.0);
    assert_eq!(parameter_block.lower_bound_for_parameter(1), -f64::MAX);

    assert_eq!(parameter_block.upper_bound_for_parameter(0), f64::MAX);
    assert_eq!(parameter_block.upper_bound_for_parameter(1), 1.0);
}

#[test]
fn parameter_block_plus_with_bounds_constraints() {
    let x = [1.0, 0.0];
    let delta = [2.0, -10.0];
    let mut parameter_block = ParameterBlock::new(0, 2, -1);
    parameter_block.set_upper_bound(0, 2.0).unwrap();
    parameter_block.set_lower_bound(1, -1.0).unwrap();
    let mut x_plus_delta = [0.0; 2];
    parameter_block.plus(&x, &delta, &mut x_plus_delta);
    assert_eq!(x_plus_delta[0], 2.0);
    assert_eq!(x_plus_delta[1], -1.0);
}

#[test]
fn parameter_block_reset_manifold_to_null() {
    let x = [1.0, 2.0, 3.0];
    let mut parameter_block = ParameterBlock::new(0, 3, -1);
    let subset = subset(3, &[1]);
    parameter_block
        .set_manifold(Some(subset.clone()), &x)
        .unwrap();
    assert!(same(parameter_block.manifold(), &subset));
    parameter_block.set_manifold(None, &x).unwrap();
    assert!(parameter_block.manifold().is_none());
}

#[test]
fn parameter_block_reset_manifold_to_not_null() {
    let x = [1.0, 2.0, 3.0];
    let mut parameter_block = ParameterBlock::new(0, 3, -1);
    let subset = subset(3, &[1]);
    parameter_block
        .set_manifold(Some(subset.clone()), &x)
        .unwrap();
    assert!(same(parameter_block.manifold(), &subset));

    let subset_different = self::subset(3, &[1]);
    parameter_block
        .set_manifold(Some(subset_different.clone()), &x)
        .unwrap();
    assert!(same(parameter_block.manifold(), &subset_different));
}

#[test]
fn parameter_block_set_null_manifold() {
    let x = [1.0, 2.0, 3.0];
    let mut parameter_block = ParameterBlock::new(0, 3, -1);
    assert!(parameter_block.manifold().is_none());

    parameter_block.set_manifold(None, &x).unwrap();
    assert!(parameter_block.manifold().is_none());
}
