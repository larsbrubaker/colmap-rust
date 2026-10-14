// Port of COLMAP's src/colmap/estimators/cost_functions/tiny_manifold_test.cc
// (Suite_Name -> suite_name), testing colmap_rust::estimators::cost_functions::tiny_manifold.
// Retractions checked against Eigen-equivalent rotations and Plus Jacobians against central
// finite differences; COLMAP's tolerances are the bars. C++'s static_asserts on the product
// sizes become assertions on ambient_size()/tangent_size().

use crate::expect_near;
use colmap_rust::estimators::cost_functions::{
    TinyEigenQuaternionManifold, TinyProductManifold, TinySphereManifold,
};
use colmap_rust::linalg::{AngleAxisd, MatrixXd, Quaterniond, Vector3d};
use colmap_rust::optim::TinyManifold;

/// EigenMatrixNear: every coefficient within `tol`.
fn assert_matrix_near(actual: &MatrixXd, expected: &MatrixXd, tol: f64) {
    assert_eq!(
        (actual.rows(), actual.cols()),
        (expected.rows(), expected.cols())
    );
    for (k, (a, e)) in actual
        .as_slice()
        .iter()
        .zip(expected.as_slice())
        .enumerate()
    {
        assert!((a - e).abs() <= tol, "coefficient {k}: |{a} - {e}| > {tol}");
    }
}

fn assert_slice_near(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tol, "coefficient {k}: |{a} - {e}| > {tol}");
    }
}

/// Frobenius norm of a block.
fn block_norm(j: &MatrixXd, row: usize, col: usize, rows: usize, cols: usize) -> f64 {
    j.block(row, col, rows, cols)
        .as_slice()
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
}

/// Central finite-difference approximation of the (ambient x tangent) Jacobian of
/// Manifold::Plus at delta = 0.
fn numeric_plus_jacobian<M: TinyManifold>(manifold: &M, x: &[f64]) -> MatrixXd {
    let ambient = manifold.ambient_size();
    let tangent = manifold.tangent_size();
    const K_EPS: f64 = 1e-6;
    let mut j = MatrixXd::zeros(ambient, tangent);
    for c in 0..tangent {
        let mut delta_plus = vec![0.0; tangent];
        let mut delta_minus = vec![0.0; tangent];
        delta_plus[c] = K_EPS;
        delta_minus[c] = -K_EPS;
        let mut x_plus = vec![0.0; ambient];
        let mut x_minus = vec![0.0; ambient];
        manifold.plus(x, &delta_plus, &mut x_plus);
        manifold.plus(x, &delta_minus, &mut x_minus);
        for r in 0..ambient {
            j[(r, c)] = (x_plus[r] - x_minus[r]) / (2.0 * K_EPS);
        }
    }
    j
}

/// Row-major analytic Jacobian returned by Manifold::PlusJacobian, as a matrix.
fn analytic_plus_jacobian<M: TinyManifold>(manifold: &M, x: &[f64]) -> MatrixXd {
    let ambient = manifold.ambient_size();
    let tangent = manifold.tangent_size();
    let mut data = vec![0.0; ambient * tangent];
    manifold.plus_jacobian(x, &mut data);
    MatrixXd::from_row_major(ambient, tangent, &data)
}

fn quaternion(angle: f64, axis: Vector3d) -> Quaterniond {
    Quaterniond::from_angle_axis(AngleAxisd::new(angle, axis.normalized()))
}

fn coeffs(q: Quaterniond) -> [f64; 4] {
    [q.x, q.y, q.z, q.w]
}

#[test]
fn eigen_quaternion_manifold_plus_at_zero_is_identity() {
    let q = quaternion(0.7, Vector3d::new(1.0, 2.0, 3.0));
    let manifold = TinyEigenQuaternionManifold;
    let delta = [0.0; 3];
    let mut x_plus = [0.0; 4];
    manifold.plus(&coeffs(q), &delta, &mut x_plus);
    assert_slice_near(&x_plus, &coeffs(q), 1e-12);
}

#[test]
fn eigen_quaternion_manifold_plus_composes_rotation_on_the_right() {
    let q = quaternion(0.7, Vector3d::new(1.0, 2.0, 3.0));
    let delta = Vector3d::new(0.05, -0.1, 0.08);
    let manifold = TinyEigenQuaternionManifold;
    let mut x_plus = [0.0; 4];
    manifold.plus(&coeffs(q), &[delta.x, delta.y, delta.z], &mut x_plus);

    let q_plus = Quaterniond::new(x_plus[3], x_plus[0], x_plus[1], x_plus[2]);
    let expected = q.to_rotation_matrix()
        * AngleAxisd::new(delta.norm(), delta.normalized()).to_rotation_matrix();
    assert_slice_near(
        q_plus.to_rotation_matrix().as_slice(),
        expected.as_slice(),
        1e-10,
    );
}

#[test]
fn eigen_quaternion_manifold_plus_jacobian_matches_finite_diff() {
    let q = quaternion(0.7, Vector3d::new(1.0, 2.0, 3.0));
    let manifold = TinyEigenQuaternionManifold;
    assert_matrix_near(
        &analytic_plus_jacobian(&manifold, &coeffs(q)),
        &numeric_plus_jacobian(&manifold, &coeffs(q)),
        1e-6,
    );
}

#[test]
fn sphere_manifold_plus_stays_on_unit_sphere() {
    let x = Vector3d::new(0.5, -1.0, 2.0).normalized();
    let x_arr = [x.x, x.y, x.z];
    let manifold = TinySphereManifold;
    let delta = [0.15, -0.2];
    let mut x_plus = [0.0; 3];
    manifold.plus(&x_arr, &delta, &mut x_plus);
    expect_near(
        Vector3d::new(x_plus[0], x_plus[1], x_plus[2]).norm(),
        1.0,
        1e-12,
    );

    let zero = [0.0; 2];
    let mut x_plus_zero = [0.0; 3];
    manifold.plus(&x_arr, &zero, &mut x_plus_zero);
    assert_slice_near(&x_plus_zero, &x_arr, 1e-12);
}

#[test]
fn sphere_manifold_plus_jacobian_matches_finite_diff() {
    let x = Vector3d::new(0.5, -1.0, 2.0).normalized();
    let x_arr = [x.x, x.y, x.z];
    let manifold = TinySphereManifold;
    // The columns span the tangent plane, i.e. are orthogonal to x.
    let j = analytic_plus_jacobian(&manifold, &x_arr);
    let xt_j: f64 = (0..2)
        .map(|c| {
            let v = x.x * j[(0, c)] + x.y * j[(1, c)] + x.z * j[(2, c)];
            v * v
        })
        .sum::<f64>()
        .sqrt();
    assert!(xt_j < 1e-12, "{xt_j}");
    assert_matrix_near(&j, &numeric_plus_jacobian(&manifold, &x_arr), 1e-6);
}

#[test]
fn product_manifold_sizes_and_block_structure() {
    type RelativePoseManifold =
        TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold>;
    let manifold = RelativePoseManifold::default();
    assert_eq!(manifold.ambient_size(), 7);
    assert_eq!(manifold.tangent_size(), 5);

    let q = quaternion(0.9, Vector3d::new(-1.0, 0.5, 2.0));
    let t = Vector3d::new(1.0, -2.0, 0.5).normalized();
    let x = [q.x, q.y, q.z, q.w, t.x, t.y, t.z];

    // Plus at zero recovers the point.
    let zero = [0.0; 5];
    let mut x_plus = [0.0; 7];
    manifold.plus(&x, &zero, &mut x_plus);
    assert_slice_near(&x_plus, &x, 1e-12);

    // The analytic Jacobian is block-diagonal and matches finite differences.
    let j = analytic_plus_jacobian(&manifold, &x);
    assert_matrix_near(&j, &numeric_plus_jacobian(&manifold, &x), 1e-6);
    let top_right_norm = block_norm(&j, 0, 3, 4, 2);
    let bottom_left_norm = block_norm(&j, 4, 0, 3, 3);
    assert!(top_right_norm < 1e-12, "{top_right_norm}");
    assert!(bottom_left_norm < 1e-12, "{bottom_left_norm}");
}

// A three-way product exercises the nesting (C++'s variadic recursion, depth 3). The two
// manifolds already in this file suffice; the blocks are laid out in argument order and the
// Plus Jacobian stays block-diagonal.
#[test]
fn product_manifold_three_way_sizes_and_block_structure() {
    type ThreeWayManifold = TinyProductManifold<
        TinyEigenQuaternionManifold,
        TinyProductManifold<TinySphereManifold, TinyEigenQuaternionManifold>,
    >;
    let manifold = ThreeWayManifold::default();
    assert_eq!(manifold.ambient_size(), 4 + 3 + 4);
    assert_eq!(manifold.tangent_size(), 3 + 2 + 3);

    let q0 = quaternion(0.9, Vector3d::new(-1.0, 0.5, 2.0));
    let t = Vector3d::new(1.0, -2.0, 0.5).normalized();
    let q1 = quaternion(0.3, Vector3d::new(0.2, -1.0, 0.7));
    let x = [
        q0.x, q0.y, q0.z, q0.w, t.x, t.y, t.z, q1.x, q1.y, q1.z, q1.w,
    ];

    // Plus at zero recovers the point.
    let zero = [0.0; 8];
    let mut x_plus = [0.0; 11];
    manifold.plus(&x, &zero, &mut x_plus);
    assert_slice_near(&x_plus, &x, 1e-12);

    // The analytic Jacobian matches finite differences.
    let mut j = analytic_plus_jacobian(&manifold, &x);
    assert_matrix_near(&j, &numeric_plus_jacobian(&manifold, &x), 1e-6);

    // Everything outside the three diagonal blocks (4x3, 3x2, 4x3) is zero.
    for (row, col, rows, cols) in [(0, 0, 4, 3), (4, 3, 3, 2), (7, 5, 4, 3)] {
        j.set_block(row, col, &MatrixXd::zeros(rows, cols));
    }
    let norm = block_norm(&j, 0, 0, 11, 8);
    assert!(norm < 1e-12, "{norm}");
}
