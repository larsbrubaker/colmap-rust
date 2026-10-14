// Rust-only (COLMAP has no test of its Ceres manifolds, and Ceres' manifold_test.cc is built
// on its gtest invariant matchers, which are not ported): port of colmap-sharp's C#-only
// `ColmapSharp.Tests/Solver/ManifoldTests.cs`, same cases and tolerances. It checks the
// manifold invariants Ceres' manifold_test_utils.h checks, for every manifold COLMAP uses
// (`colmap_rust::solver::{manifold, sphere_manifold, product_manifold}`):
// - Plus(x, 0) = x and Minus(x, x) = 0,
// - Minus(Plus(x, delta), x) = delta for a small tangent step,
// - PlusJacobian agrees with central finite differences of Plus, and MinusJacobian *
//   PlusJacobian is the tangent-space identity,
// - RightMultiplyByPlusJacobian equals the explicit product with PlusJacobian.
// colmap-sharp's data-driven cases become one loop per test that names the failing case.

use colmap_rust::solver::{
    EigenQuaternionManifold, EuclideanManifold, Manifold, ProductManifold, QuaternionManifold,
    SphereManifold, SubsetManifold,
};

const H: f64 = 1e-6;
const TOLERANCE: f64 = 1e-8;

fn sphere(size: usize) -> SphereManifold {
    SphereManifold::new(size).expect("size > 1")
}

fn cases() -> Vec<(&'static str, Box<dyn Manifold>, Vec<f64>)> {
    let inv_sqrt = 1.0 / (0.1_f64 * 0.1 + 0.2 * 0.2 + 0.3 * 0.3 + 0.9 * 0.9).sqrt();
    let unit_quaternion = vec![
        0.1 * inv_sqrt,
        0.2 * inv_sqrt,
        0.3 * inv_sqrt,
        0.9 * inv_sqrt,
    ];
    let with_tail = |tail: &[f64]| {
        let mut x = unit_quaternion.clone();
        x.extend_from_slice(tail);
        x
    };
    vec![
        (
            "Euclidean3",
            Box::new(EuclideanManifold::new(3)),
            vec![1.0, -2.0, 0.5],
        ),
        (
            "Subset5",
            Box::new(SubsetManifold::new(5, &[3, 1]).unwrap()),
            vec![1.0, 2.0, 3.0, 4.0, 5.0],
        ),
        (
            "EigenQuaternion",
            Box::new(EigenQuaternionManifold),
            unit_quaternion.clone(),
        ),
        (
            "CeresQuaternion",
            Box::new(QuaternionManifold),
            unit_quaternion.clone(),
        ),
        ("Sphere3", Box::new(sphere(3)), vec![0.3, -0.4, 1.2]),
        (
            "Sphere3NegativePivot",
            Box::new(sphere(3)),
            vec![0.3, -0.4, -1.2],
        ),
        ("Sphere4", Box::new(sphere(4)), vec![0.5, 0.5, -0.5, 0.5]),
        (
            "QuaternionTimesR3",
            Box::new(
                ProductManifold::new(vec![
                    Box::new(EigenQuaternionManifold),
                    Box::new(EuclideanManifold::new(3)),
                ])
                .unwrap(),
            ),
            with_tail(&[1.0, 2.0, 3.0]),
        ),
        (
            "QuaternionTimesSphere",
            Box::new(
                ProductManifold::new(vec![Box::new(EigenQuaternionManifold), Box::new(sphere(3))])
                    .unwrap(),
            ),
            with_tail(&[0.0, 0.6, 0.8]),
        ),
    ]
}

#[test]
fn rust_only_plus_minus_invariants_hold() {
    for (name, m, x) in cases() {
        check_plus_minus_invariants(name, m.as_ref(), &x);
    }
}

#[test]
fn rust_only_jacobians_match_finite_differences() {
    for (name, m, x) in cases() {
        check_jacobians(name, m.as_ref(), &x);
    }
}

// Rust-only beyond colmap-sharp's cases: at the poles of the sphere (x on the last axis) the
// Householder vector takes its degenerate branch, beta = 0 at the north pole and 2 at the
// south pole.
#[test]
fn rust_only_sphere_invariants_hold_at_the_poles() {
    for (name, x) in [
        ("NorthPole", [0.0, 0.0, 2.0]),
        ("SouthPole", [0.0, 0.0, -2.0]),
    ] {
        check_plus_minus_invariants(name, &sphere(3), &x);
        check_jacobians(name, &sphere(3), &x);
    }
}

fn check_plus_minus_invariants(name: &str, m: &dyn Manifold, x: &[f64]) {
    let n = m.ambient_size();
    let t = m.tangent_size();

    let mut plus_zero = vec![0.0; n];
    assert!(m.plus(x, &vec![0.0; t], &mut plus_zero));
    for i in 0..n {
        assert_eq!(plus_zero[i], x[i], "{name}: Plus(x, 0)[{i}]");
    }

    let mut minus_self = vec![0.0; t];
    assert!(m.minus(x, x, &mut minus_self));
    for (i, value) in minus_self.iter().enumerate() {
        assert!(value.abs() <= 1e-15, "{name}: Minus(x, x)[{i}] = {value}");
    }

    let delta: Vec<f64> = (0..t)
        .map(|i| 0.01 * (i + 1) as f64 * if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let mut y = vec![0.0; n];
    assert!(m.plus(x, &delta, &mut y));
    let mut recovered = vec![0.0; t];
    assert!(m.minus(&y, x, &mut recovered));
    for i in 0..t {
        assert!(
            (recovered[i] - delta[i]).abs() <= 1e-12,
            "{name}: Minus(Plus(x, delta), x)[{i}] = {} vs {}",
            recovered[i],
            delta[i]
        );
    }
}

fn check_jacobians(name: &str, m: &dyn Manifold, x: &[f64]) {
    let n = m.ambient_size();
    let t = m.tangent_size();

    let mut plus_jacobian = vec![0.0; n * t];
    assert!(m.plus_jacobian(x, &mut plus_jacobian));
    let mut minus_jacobian = vec![0.0; t * n];
    assert!(m.minus_jacobian(x, &mut minus_jacobian));

    let mut forward = vec![0.0; n];
    let mut backward = vec![0.0; n];
    for j in 0..t {
        let mut step = vec![0.0; t];
        step[j] = H;
        assert!(m.plus(x, &step, &mut forward));
        step[j] = -H;
        assert!(m.plus(x, &step, &mut backward));
        for i in 0..n {
            let fd = (forward[i] - backward[i]) / (2.0 * H);
            assert!(
                (plus_jacobian[i * t + j] - fd).abs() <= TOLERANCE,
                "{name}: PlusJacobian({i}, {j}) = {} vs finite difference {fd}",
                plus_jacobian[i * t + j]
            );
        }
    }

    // d Minus(y, x) / dy at y = x, stepping y along the ambient axes. Only the component
    // of the step tangent to the manifold is meaningful, so compare MinusJacobian *
    // PlusJacobian (the tangent-space identity) instead of raw ambient columns.
    for r in 0..t {
        for col in 0..t {
            let mut sum = 0.0;
            for k in 0..n {
                sum += minus_jacobian[r * n + k] * plus_jacobian[k * t + col];
            }
            let expected = if r == col { 1.0 } else { 0.0 };
            assert!(
                (sum - expected).abs() <= 1e-12,
                "{name}: (MinusJacobian * PlusJacobian)({r}, {col}) = {sum}"
            );
        }
    }

    let ambient: Vec<f64> = (0..2 * n).map(|i| 0.1 * i as f64 - 0.3).collect();
    let mut tangent = vec![0.0; 2 * t];
    assert!(m.right_multiply_by_plus_jacobian(x, 2, &ambient, &mut tangent));
    for r in 0..2 {
        for col in 0..t {
            let mut sum = 0.0;
            for k in 0..n {
                sum += ambient[r * n + k] * plus_jacobian[k * t + col];
            }
            assert!(
                (tangent[r * t + col] - sum).abs() <= 1e-14,
                "{name}: RightMultiplyByPlusJacobian({r}, {col}) = {} vs {sum}",
                tangent[r * t + col]
            );
        }
    }
}

// Ceres' SubsetManifold rejects duplicate and out-of-range constant indices.
#[test]
fn rust_only_subset_manifold_rejects_bad_indices() {
    assert!(SubsetManifold::new(3, &[1, 1]).is_err());
    assert!(SubsetManifold::new(3, &[3]).is_err());
}

// Ceres' SphereManifold needs at least two dimensions, and ProductManifold two components.
#[test]
fn rust_only_sphere_and_product_reject_degenerate_sizes() {
    assert!(SphereManifold::new(1).is_err());
    assert!(ProductManifold::new(vec![Box::new(EuclideanManifold::new(3))]).is_err());
}
