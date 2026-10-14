// Port of COLMAP's src/colmap/estimators/cost_functions/quaternion_utils_test.cc
// (Suite_Name -> suite_name), testing colmap_rust::estimators::cost_functions::quaternion_utils.
// Scalar formulas checked against Eigen-equivalent quaternion products and central finite
// differences; COLMAP's tolerances are the bars.
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. Values COLMAP draws in
// one statement are drawn in clang's left-to-right order.

use std::ops::Index;

use crate::expect_near;
use colmap_rust::estimators::cost_functions::quaternion_utils::{
    angle_axis_from_eigen_quaternion, eigen_quaternion_from_angle_axis,
    quaternion_left_mult_matrix, quaternion_right_mult_matrix, quaternion_rotate_point_with_jac,
};
use colmap_rust::linalg::{Matrix4d, Quaterniond, Vector4d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};

/// EigenMatrixNear: every coefficient within `tol`.
fn assert_all_near(actual: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (k, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() <= tol, "coefficient {k}: |{a} - {e}| > {tol}");
    }
}

/// The coefficients of a fixed-size vector.
fn arr<T: Index<usize, Output = f64>, const N: usize>(v: &T) -> [f64; N] {
    std::array::from_fn(|i| v[i])
}

fn from_coeffs(v: Vector4d) -> Quaterniond {
    Quaterniond::new(v.w, v.x, v.y, v.z)
}

/// Central differences of `f` (4-vector -> 4-vector) at `v`, as a 4x4 matrix.
fn numeric_4x4(v: Vector4d, f: impl Fn(Vector4d) -> Vector4d) -> Matrix4d {
    const K_EPS: f64 = 1e-7;
    let mut j = Matrix4d::zeros();
    for k in 0..4 {
        let (mut plus, mut minus) = (v, v);
        plus[k] += K_EPS;
        minus[k] -= K_EPS;
        let col = (f(plus) - f(minus)) / (2.0 * K_EPS);
        for r in 0..4 {
            j[(r, k)] = col[r];
        }
    }
    j
}

#[test]
fn quaternion_left_mult_matrix_nominal() {
    set_prng_seed(0);
    for _ in 0..100 {
        let q = random_eigen_quaterniond();
        let p = random_eigen_quaterniond();
        let p_vec = p.coeffs();

        // L(q) * p = q * p.
        let result = quaternion_left_mult_matrix(q) * p_vec;
        assert_all_near(
            &arr::<_, 4>(&result),
            &arr::<_, 4>(&(q * p).coeffs()),
            1e-12,
        );

        // L(q) = d(q*p)/dp (Jacobian w.r.t. second argument).
        let j_numeric = numeric_4x4(p_vec, |pv| (q * from_coeffs(pv)).coeffs());
        assert_all_near(
            quaternion_left_mult_matrix(q).as_slice(),
            j_numeric.as_slice(),
            1e-5,
        );
    }
}

#[test]
fn quaternion_right_mult_matrix_nominal() {
    set_prng_seed(0);
    for _ in 0..100 {
        let q = random_eigen_quaterniond();
        let p = random_eigen_quaterniond();
        let q_vec = q.coeffs();

        // R(p) * q = q * p.
        let result = quaternion_right_mult_matrix(p) * q_vec;
        assert_all_near(
            &arr::<_, 4>(&result),
            &arr::<_, 4>(&(q * p).coeffs()),
            1e-12,
        );

        // R(p) = d(q*p)/dq (Jacobian w.r.t. first argument).
        let j_numeric = numeric_4x4(q_vec, |qv| (from_coeffs(qv) * p).coeffs());
        assert_all_near(
            quaternion_right_mult_matrix(p).as_slice(),
            j_numeric.as_slice(),
            1e-5,
        );
    }
}

#[test]
fn quaternion_rotate_point_with_jac_nominal() {
    const K_EPS: f64 = 1e-7;
    set_prng_seed(0);
    for _ in 0..100 {
        let q = random_eigen_quaterniond();
        let pt = random_eigen_vector3d();
        let q_arr = [q.x, q.y, q.z, q.w];
        let pt_arr = [pt.x, pt.y, pt.z];

        // R(q) * pt matches Eigen.
        let mut j_analytical = [0.0; 12];
        let result = quaternion_rotate_point_with_jac(&q_arr, &pt_arr, Some(&mut j_analytical));
        assert_all_near(&arr::<_, 3>(&result), &arr::<_, 3>(&(q * pt)), 1e-12);

        // Jacobian d(R(q)*pt)/dq matches numeric (row-major 3x4).
        let mut j_numeric = [0.0; 12];
        for k in 0..4 {
            let (mut q_plus, mut q_minus) = (q_arr, q_arr);
            q_plus[k] += K_EPS;
            q_minus[k] -= K_EPS;
            let col = (quaternion_rotate_point_with_jac(&q_plus, &pt_arr, None)
                - quaternion_rotate_point_with_jac(&q_minus, &pt_arr, None))
                / (2.0 * K_EPS);
            for r in 0..3 {
                j_numeric[r * 4 + k] = col[r];
            }
        }
        assert_all_near(&j_analytical, &j_numeric, 1e-5);
    }
}

#[test]
fn eigen_quaternion_angle_axis_roundtrip() {
    set_prng_seed(0);
    for _ in 0..100 {
        let q = random_eigen_quaterniond();
        let q_arr = [q.x, q.y, q.z, q.w];

        // quaternion -> angle-axis -> quaternion recovers the original rotation.
        let mut angle_axis = [0.0; 3];
        angle_axis_from_eigen_quaternion(&q_arr, &mut angle_axis);
        let mut q_out = [0.0; 4];
        eigen_quaternion_from_angle_axis(&angle_axis, &mut q_out);

        // Compare as rotations to avoid the quaternion double-cover sign ambiguity.
        let q_recovered = Quaterniond::new(q_out[3], q_out[0], q_out[1], q_out[2]);
        expect_near(q.angular_distance(q_recovered), 0.0, 1e-10);
    }
}
