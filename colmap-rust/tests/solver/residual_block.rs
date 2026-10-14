// Port of Ceres Solver 2.2.0's internal/ceres/residual_block_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md): `colmap_rust::solver::ResidualBlock`, same names and values. Ceres
// hands the residual block `ParameterBlock*`s over uninitialized user arrays; here the blocks
// are ids into a block list over a zeroed value arena (the cost functions never read the
// values). Getters compare `Arc`s by pointer, as Ceres compares raw pointers.

use std::sync::Arc;

use colmap_rust::solver::{
    CostFunction, Manifold, ParameterBlock, ParameterBlockId, ParameterValues, ResidualBlock,
    SubsetManifold,
};

// Trivial cost function that accepts three arguments.
struct TernaryCostFunction {
    num_residuals: usize,
    sizes: Vec<usize>,
}

impl TernaryCostFunction {
    fn new(num_residuals: usize, size1: usize, size2: usize, size3: usize) -> Self {
        Self {
            num_residuals,
            sizes: vec![size1, size2, size3],
        }
    }
}

impl CostFunction for TernaryCostFunction {
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
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        for (i, r) in residuals[..self.num_residuals].iter_mut().enumerate() {
            *r = i as f64;
        }
        if let Some(jacobians) = jacobians {
            for (k, jacobian) in jacobians.iter_mut().enumerate().take(3) {
                if let Some(jacobian) = jacobian {
                    jacobian[..self.num_residuals * self.sizes[k]].fill(k as f64);
                }
            }
        }
        true
    }
}

fn blocks_xyz() -> (Vec<ParameterBlock>, Vec<f64>, Vec<ParameterBlockId>) {
    let blocks = vec![
        ParameterBlock::new(0, 2, -1),
        ParameterBlock::new(2, 3, -1),
        ParameterBlock::new(5, 4, -1),
    ];
    let ids = vec![
        ParameterBlockId(0),
        ParameterBlockId(1),
        ParameterBlockId(2),
    ];
    (blocks, vec![0.0; 9], ids)
}

// Evaluates with residuals and the Jacobians (row-major) whose slots are Some.
fn evaluate(
    residual_block: &ResidualBlock,
    blocks: &[ParameterBlock],
    values: &[f64],
    residuals: Option<&mut [f64]>,
    jacobians: Option<&mut [Option<&mut [f64]>]>,
    scratch: &mut [f64],
) -> f64 {
    let values = ParameterValues {
        user: values,
        state: None,
    };
    residual_block
        .evaluate(blocks, values, true, residuals, jacobians, scratch)
        .unwrap()
        .unwrap()
}

#[test]
fn residual_block_evaluate_with_no_loss_function_or_manifolds() {
    let mut scratch = [0.0; 64];

    // Prepare the parameter blocks.
    let (blocks, values, parameters) = blocks_xyz();
    let cost_function: Arc<dyn CostFunction> = Arc::new(TernaryCostFunction::new(3, 2, 3, 4));

    // Create the object under tests.
    let residual_block = ResidualBlock::new(cost_function.clone(), None, parameters.clone());

    // Verify getters.
    assert!(Arc::ptr_eq(&cost_function, residual_block.cost_function()));
    assert!(residual_block.loss_function().is_none());
    assert_eq!(parameters[0], residual_block.parameter_blocks()[0]);
    assert_eq!(parameters[1], residual_block.parameter_blocks()[1]);
    assert_eq!(parameters[2], residual_block.parameter_blocks()[2]);
    assert_eq!(3, residual_block.num_scratch_doubles_for_evaluate(&blocks));

    // Verify cost-only evaluation.
    let cost = evaluate(&residual_block, &blocks, &values, None, None, &mut scratch);
    assert_eq!(0.5 * (0.0 * 0.0 + 1.0 * 1.0 + 2.0 * 2.0), cost);

    // Verify cost and residual evaluation.
    let mut residuals = [0.0; 3];
    let cost = evaluate(
        &residual_block,
        &blocks,
        &values,
        Some(&mut residuals),
        None,
        &mut scratch,
    );
    assert_eq!(0.5 * (0.0 * 0.0 + 1.0 * 1.0 + 2.0 * 2.0), cost);
    assert_eq!([0.0, 1.0, 2.0], residuals);

    // Verify cost, residual, and jacobian evaluation.
    residuals.fill(0.0);
    let mut jacobian_rx = [-1.0; 3 * 2];
    let mut jacobian_ry = [-1.0; 3 * 3];
    let mut jacobian_rz = [-1.0; 3 * 4];
    let cost = {
        let mut jacobian_ptrs = [
            Some(&mut jacobian_rx[..]),
            Some(&mut jacobian_ry[..]),
            Some(&mut jacobian_rz[..]),
        ];
        evaluate(
            &residual_block,
            &blocks,
            &values,
            Some(&mut residuals),
            Some(&mut jacobian_ptrs),
            &mut scratch,
        )
    };
    assert_eq!(0.5 * (0.0 * 0.0 + 1.0 * 1.0 + 2.0 * 2.0), cost);
    assert_eq!([0.0, 1.0, 2.0], residuals);
    assert!(jacobian_rx.iter().all(|&v| v == 0.0), "{jacobian_rx:?}");
    assert!(jacobian_ry.iter().all(|&v| v == 1.0), "{jacobian_ry:?}");
    assert!(jacobian_rz.iter().all(|&v| v == 2.0), "{jacobian_rz:?}");

    // Verify cost, residual, and partial jacobian evaluation.
    residuals.fill(0.0);
    jacobian_rx.fill(-1.0);
    jacobian_ry.fill(-1.0);
    jacobian_rz.fill(-1.0);
    let cost = {
        // Don't compute the jacobian for y.
        let mut jacobian_ptrs = [Some(&mut jacobian_rx[..]), None, Some(&mut jacobian_rz[..])];
        evaluate(
            &residual_block,
            &blocks,
            &values,
            Some(&mut residuals),
            Some(&mut jacobian_ptrs),
            &mut scratch,
        )
    };
    assert_eq!(0.5 * (0.0 * 0.0 + 1.0 * 1.0 + 2.0 * 2.0), cost);
    assert_eq!([0.0, 1.0, 2.0], residuals);
    assert!(jacobian_rx.iter().all(|&v| v == 0.0), "{jacobian_rx:?}");
    assert!(jacobian_ry.iter().all(|&v| v == -1.0), "{jacobian_ry:?}");
    assert!(jacobian_rz.iter().all(|&v| v == 2.0), "{jacobian_rz:?}");
}

// Trivial cost function that accepts three arguments (SizedCostFunction<3, 2, 3, 4>).
struct LocallyParameterizedCostFunction {
    sizes: [usize; 3],
}

impl CostFunction for LocallyParameterizedCostFunction {
    fn num_residuals(&self) -> usize {
        3
    }
    fn parameter_block_sizes(&self) -> &[usize] {
        &self.sizes
    }
    fn evaluate(
        &self,
        _parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        for (i, r) in residuals[..3].iter_mut().enumerate() {
            *r = i as f64;
        }
        if let Some(jacobians) = jacobians {
            for (k, jacobian) in jacobians.iter_mut().enumerate().take(3) {
                // The jacobians here are full sized, but they are transformed in the
                // evaluator into the "local" jacobian. In the tests, the "SubsetManifold" is
                // used, which should pick out columns from these jacobians. Put values in the
                // jacobian that make this obvious; in particular, make the jacobians like
                // this:
                //
                //   0 1 2 3 4 ...
                //   0 1 2 3 4 ...
                //   0 1 2 3 4 ...
                //
                if let Some(jacobian) = jacobian {
                    let cols = self.sizes[k];
                    for j in 0..k + 2 {
                        for r in 0..3 {
                            jacobian[r * cols + j] = j as f64;
                        }
                    }
                }
            }
        }
        true
    }
}

#[test]
fn residual_block_evaluate_with_manifolds() {
    let mut scratch = [0.0; 64];

    // Prepare the parameter blocks.
    let (mut blocks, values, parameters) = blocks_xyz();

    // Make x have the first component fixed.
    let x_manifold: Arc<dyn Manifold> = Arc::new(SubsetManifold::new(2, &[0]).unwrap());
    blocks[0]
        .set_manifold(Some(x_manifold), &values[0..2])
        .unwrap();

    // Make z have the last and last component fixed.
    let z_manifold: Arc<dyn Manifold> = Arc::new(SubsetManifold::new(4, &[2]).unwrap());
    blocks[2]
        .set_manifold(Some(z_manifold), &values[5..9])
        .unwrap();

    let cost_function: Arc<dyn CostFunction> =
        Arc::new(LocallyParameterizedCostFunction { sizes: [2, 3, 4] });

    // Create the object under tests.
    let residual_block = ResidualBlock::new(cost_function.clone(), None, parameters.clone());

    // Verify getters.
    assert!(Arc::ptr_eq(&cost_function, residual_block.cost_function()));
    assert!(residual_block.loss_function().is_none());
    assert_eq!(parameters[0], residual_block.parameter_blocks()[0]);
    assert_eq!(parameters[1], residual_block.parameter_blocks()[1]);
    assert_eq!(parameters[2], residual_block.parameter_blocks()[2]);
    assert_eq!(
        3 * (2 + 4) + 3,
        residual_block.num_scratch_doubles_for_evaluate(&blocks)
    );

    // Verify cost-only evaluation.
    let cost = evaluate(&residual_block, &blocks, &values, None, None, &mut scratch);
    assert_eq!(0.5 * (0.0 * 0.0 + 1.0 * 1.0 + 2.0 * 2.0), cost);

    // Verify cost and residual evaluation.
    let mut residuals = [0.0; 3];
    let cost = evaluate(
        &residual_block,
        &blocks,
        &values,
        Some(&mut residuals),
        None,
        &mut scratch,
    );
    assert_eq!(0.5 * (0.0 * 0.0 + 1.0 * 1.0 + 2.0 * 2.0), cost);
    assert_eq!([0.0, 1.0, 2.0], residuals);

    // Verify cost, residual, and jacobian evaluation.
    residuals.fill(0.0);
    let mut jacobian_rx = [-1.0; 3]; // Since the first element is fixed.
    let mut jacobian_ry = [-1.0; 3 * 3];
    let mut jacobian_rz = [-1.0; 3 * 3]; // Since the third element is fixed.
    let cost = {
        let mut jacobian_ptrs = [
            Some(&mut jacobian_rx[..]),
            Some(&mut jacobian_ry[..]),
            Some(&mut jacobian_rz[..]),
        ];
        evaluate(
            &residual_block,
            &blocks,
            &values,
            Some(&mut residuals),
            Some(&mut jacobian_ptrs),
            &mut scratch,
        )
    };
    assert_eq!(0.5 * (0.0 * 0.0 + 1.0 * 1.0 + 2.0 * 2.0), cost);
    assert_eq!([0.0, 1.0, 2.0], residuals);

    let expected_jacobian_rx = [1.0, 1.0, 1.0];
    #[rustfmt::skip]
    let expected_jacobian_ry = [
        0.0, 1.0, 2.0,
        0.0, 1.0, 2.0,
        0.0, 1.0, 2.0,
    ];
    #[rustfmt::skip]
    let expected_jacobian_rz = [
        0.0, 1.0, /* 2.0, */ 3.0, // 3rd parameter constant.
        0.0, 1.0, /* 2.0, */ 3.0,
        0.0, 1.0, /* 2.0, */ 3.0,
    ];
    assert_eq!(expected_jacobian_rx, jacobian_rx);
    assert_eq!(expected_jacobian_ry, jacobian_ry);
    assert_eq!(expected_jacobian_rz, jacobian_rz);

    // Verify cost, residual, and partial jacobian evaluation.
    residuals.fill(0.0);
    jacobian_rx.fill(-1.0);
    jacobian_ry.fill(-1.0);
    jacobian_rz.fill(-1.0);
    let cost = {
        // Don't compute the jacobian for y.
        let mut jacobian_ptrs = [Some(&mut jacobian_rx[..]), None, Some(&mut jacobian_rz[..])];
        evaluate(
            &residual_block,
            &blocks,
            &values,
            Some(&mut residuals),
            Some(&mut jacobian_ptrs),
            &mut scratch,
        )
    };
    assert_eq!(0.5 * (0.0 * 0.0 + 1.0 * 1.0 + 2.0 * 2.0), cost);
    assert_eq!([0.0, 1.0, 2.0], residuals);

    assert_eq!(expected_jacobian_rx, jacobian_rx);
    assert!(jacobian_ry.iter().all(|&v| v == -1.0), "{jacobian_ry:?}");
    assert_eq!(expected_jacobian_rz, jacobian_rz);
}
