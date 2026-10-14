// Rust-only (COLMAP's cost-function tests arrive with the cost functions in Phase 8): port of
// colmap-sharp's `ColmapSharp.Tests/Solver/AutoDiffCostFunctionTests.cs`, testing
// colmap_rust::solver::AutoDiffCostFunction on a reprojection functor shaped like COLMAP's
// ReprojErrorCostFunctor (point 3, pose 7 = Eigen quaternion x y z w + translation,
// SIMPLE_RADIAL camera 4, two residuals), plus solver::rotation.
// - Jacobians agree with central finite differences of the double evaluation.
// - Residuals without Jacobians come from the double path; a block whose Jacobian is not
//   requested is left untouched.
// - Parameter and Jacobian slices cut from shared buffers give the per-block bits.
// Not ported: `EveryGradientWidth_GivesIdenticalBits`, which pins colmap-sharp's chunked
// evaluation (narrower Jets in several passes). The Rust AutoDiffCostFunction has no chunks:
// its Jet width is the const-generic parameter count, one pass, as Ceres does.

use colmap_rust::math::fns;
use colmap_rust::sensor::models::{CameraModel, SimpleRadialCameraModel};
use colmap_rust::solver::rotation::{angle_axis_to_quaternion, quaternion_to_angle_axis};
use colmap_rust::solver::{AutoDiffCostFunction, AutoDiffFunctor, CostFunction, Jet, Scalar};

/// A test reprojection functor: point (3), cam_from_world (7), SIMPLE_RADIAL (4).
#[derive(Clone, Copy)]
struct TestReprojectionFunctor {
    observed_x: f64,
    observed_y: f64,
}

impl AutoDiffFunctor for TestReprojectionFunctor {
    fn call<T: Scalar>(&self, p: &[T], residuals: &mut [T]) -> bool {
        let point = &p[..3];
        let pose = &p[3..10];
        let camera = &p[10..14];

        // R(q) p = p + 2 w (v x p) + 2 v x (v x p), with q = (v, w) stored x, y, z, w.
        let (qx, qy, qz, qw) = (pose[0], pose[1], pose[2], pose[3]);
        let c0 = qy * point[2] - qz * point[1];
        let c1 = qz * point[0] - qx * point[2];
        let c2 = qx * point[1] - qy * point[0];
        let d0 = qy * c2 - qz * c1;
        let d1 = qz * c0 - qx * c2;
        let d2 = qx * c1 - qy * c0;
        let u = point[0] + (qw * c0 + d0) * 2.0 + pose[4];
        let v = point[1] + (qw * c1 + d1) * 2.0 + pose[5];
        let w = point[2] + (qw * c2 + d2) * 2.0 + pose[6];

        let (mut x, mut y) = (T::from_f64(0.0), T::from_f64(0.0));
        if SimpleRadialCameraModel::img_from_cam(camera, u, v, w, &mut x, &mut y, true) {
            residuals[0] = x - self.observed_x;
            residuals[1] = y - self.observed_y;
        } else {
            residuals[0] = T::from_f64(0.0);
            residuals[1] = T::from_f64(0.0);
        }
        true
    }
}

const BLOCK_SIZES: [usize; 3] = [3, 7, 4];

fn parameters() -> Vec<Vec<f64>> {
    let norm = fns::sqrt(0.1 * 0.1 + 0.2 * 0.2 + 0.05 * 0.05 + 0.97 * 0.97);
    vec![
        vec![0.4, -0.3, 4.0],
        vec![
            0.1 / norm,
            -0.2 / norm,
            0.05 / norm,
            0.97 / norm,
            0.1,
            0.2,
            0.5,
        ],
        vec![800.0, 320.0, 240.0, -0.05],
    ]
}

fn create() -> AutoDiffCostFunction<TestReprojectionFunctor, 2, 14> {
    let functor = TestReprojectionFunctor {
        observed_x: 300.0,
        observed_y: 260.0,
    };
    AutoDiffCostFunction::new(functor, &BLOCK_SIZES).unwrap()
}

fn views(parameters: &[Vec<f64>]) -> Vec<&[f64]> {
    parameters.iter().map(Vec::as_slice).collect()
}

/// Residuals and every block's Jacobian.
fn run(cost_function: &impl CostFunction, parameters: &[Vec<f64>]) -> ([f64; 2], Vec<Vec<f64>>) {
    let mut residuals = [0.0; 2];
    let mut jacobians: Vec<Vec<f64>> = BLOCK_SIZES.iter().map(|&s| vec![0.0; 2 * s]).collect();
    let mut jacobian_views: Vec<Option<&mut [f64]>> = jacobians
        .iter_mut()
        .map(|j| Some(j.as_mut_slice()))
        .collect();
    assert!(cost_function.evaluate(
        &views(parameters),
        &mut residuals,
        Some(&mut jacobian_views)
    ));
    (residuals, jacobians)
}

// The evaluator's layout: every parameter block and every Jacobian a slice of one shared
// buffer, at offsets that do not start at zero. Must give the same bits as per-block arrays.
#[test]
fn rust_only_shared_buffer_views_match_per_block_arrays() {
    let cost_function = create();
    let parameters = parameters();
    let (expected_residuals, expected_jacobians) = run(&cost_function, &parameters);

    let mut parameter_buffer = [0.0; 5 + 14];
    let mut offset = 5;
    for (block, &size) in parameters.iter().zip(&BLOCK_SIZES) {
        parameter_buffer[offset..offset + size].copy_from_slice(block);
        offset += size;
    }
    let parameter_views: Vec<&[f64]> = [5, 8, 15]
        .iter()
        .zip(&BLOCK_SIZES)
        .map(|(&start, &size)| &parameter_buffer[start..start + size])
        .collect();

    let mut jacobian_buffer = [0.0; 3 + 2 * 14];
    let (head, rest) = jacobian_buffer.split_at_mut(3);
    let (j0, rest) = rest.split_at_mut(2 * BLOCK_SIZES[0]);
    let (j1, j2) = rest.split_at_mut(2 * BLOCK_SIZES[1]);
    let mut jacobian_views = [Some(j0), Some(j1), Some(j2)];

    let mut residuals = [0.0; 2];
    assert!(cost_function.evaluate(&parameter_views, &mut residuals, Some(&mut jacobian_views)));
    assert_eq!(residuals, expected_residuals);
    for (view, expected) in jacobian_views.iter().zip(&expected_jacobians) {
        assert_eq!(view.as_deref().unwrap(), expected.as_slice());
    }
    assert_eq!(head, &[0.0, 0.0, 0.0]);
}

#[test]
fn rust_only_jacobians_match_finite_differences() {
    let cost_function = create();
    let mut parameters = parameters();
    let (_, jacobians) = run(&cost_function, &parameters);

    let mut forward = [0.0; 2];
    let mut backward = [0.0; 2];
    for b in 0..BLOCK_SIZES.len() {
        for c in 0..BLOCK_SIZES[b] {
            let original = parameters[b][c];
            let h = 1e-6 * 1.0_f64.max(original.abs());
            parameters[b][c] = original + h;
            cost_function.evaluate(&views(&parameters), &mut forward, None);
            parameters[b][c] = original - h;
            cost_function.evaluate(&views(&parameters), &mut backward, None);
            parameters[b][c] = original;
            for r in 0..2 {
                let fd = (forward[r] - backward[r]) / (2.0 * h);
                let analytic = jacobians[b][r * BLOCK_SIZES[b] + c];
                assert!(
                    (analytic - fd).abs() <= 1e-5 * 1.0_f64.max(fd.abs()),
                    "block {b} column {c} residual {r}: {analytic} vs {fd}"
                );
            }
        }
    }
}

#[test]
fn rust_only_residuals_only_use_double_path_and_partial_jacobians_match_full() {
    let cost_function = create();
    let parameters = parameters();
    let mut residuals = [0.0; 2];
    assert!(cost_function.evaluate(&views(&parameters), &mut residuals, None));

    let mut direct = [0.0; 2];
    let flat: Vec<f64> = parameters.concat();
    cost_function.functor().call(&flat, &mut direct);
    assert_eq!(residuals, direct);

    let mut point_jacobian = [7.0; 6];
    let mut camera_jacobian = [0.0; 8];
    let mut partial = [
        Some(&mut point_jacobian[..]),
        None,
        Some(&mut camera_jacobian[..]),
    ];
    assert!(cost_function.evaluate(&views(&parameters), &mut residuals, Some(&mut partial)));
    assert!(!point_jacobian.iter().all(|&value| value == 7.0));
    let (_, full) = run(&cost_function, &parameters);
    assert_eq!(&camera_jacobian[..], full[2].as_slice());

    let mut point_only = [0.0; 6];
    let mut only_point = [Some(&mut point_only[..]), None, None];
    cost_function.evaluate(&views(&parameters), &mut residuals, Some(&mut only_point));
    assert!(point_only.iter().any(|&value| value != 0.0));
}

#[test]
fn rust_only_angle_axis_quaternion_round_trip() {
    let angle_axis = [0.3, -0.5, 0.2];
    let mut quaternion = [0.0; 4];
    angle_axis_to_quaternion(&angle_axis, &mut quaternion);
    let theta = fns::sqrt(0.3 * 0.3 + 0.5 * 0.5 + 0.2 * 0.2);
    assert!((quaternion[0] - fns::cos(theta / 2.0)).abs() <= 1e-15);
    assert!((quaternion[1] - 0.3 / theta * fns::sin(theta / 2.0)).abs() <= 1e-15);

    let mut back = [0.0; 3];
    quaternion_to_angle_axis(&quaternion, &mut back);
    for i in 0..3 {
        assert!((back[i] - angle_axis[i]).abs() <= 1e-15);
    }

    // A quaternion with negative w (angle > pi) maps to the equivalent angle below pi.
    let flipped = quaternion.map(|q| -q);
    quaternion_to_angle_axis(&flipped, &mut back);
    for i in 0..3 {
        assert!((back[i] - angle_axis[i]).abs() <= 1e-15);
    }
}

// At zero rotation the Taylor branch gives the exact derivatives dq/da = I / 2 and
// da/dq = 2 I on the vector part, where the general formula would produce NaN.
#[test]
fn rust_only_angle_axis_quaternion_derivatives_at_zero() {
    let angle_axis = [
        Jet::<3>::variable(0.0, 0),
        Jet::variable(0.0, 1),
        Jet::variable(0.0, 2),
    ];
    let mut quaternion = [Jet::<3>::constant(0.0); 4];
    angle_axis_to_quaternion(&angle_axis, &mut quaternion);
    assert_eq!(quaternion[0].a, 1.0);
    for i in 0..3 {
        for k in 0..3 {
            assert_eq!(quaternion[i + 1].v[k], if i == k { 0.5 } else { 0.0 });
        }
    }

    let q = [
        Jet::<3>::constant(1.0),
        Jet::variable(0.0, 0),
        Jet::variable(0.0, 1),
        Jet::variable(0.0, 2),
    ];
    let mut back = [Jet::<3>::constant(0.0); 3];
    quaternion_to_angle_axis(&q, &mut back);
    for (i, jet) in back.iter().enumerate() {
        for k in 0..3 {
            assert_eq!(jet.v[k], if i == k { 2.0 } else { 0.0 });
        }
    }
}
