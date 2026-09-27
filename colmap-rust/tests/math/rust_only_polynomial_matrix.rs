// Rust-only tests for math::polynomial, math::matrix and math::random_eigen: the Rust API
// edges COLMAP's tests don't reach (THROW_CHECK -> Err, the MatrixXd / Matrix3d RQ entry
// points, random_eigen's draw order across its fixed, dynamic and float variants). They
// never stand in for the ported polynomial.rs / matrix.rs / random_eigen.rs.

use colmap_rust::linalg::VectorXd;
use colmap_rust::linalg::{Matrix3d, Matrix4d, MatrixXd, DUMMY_PRECISION};
use colmap_rust::math::matrix::{decompose_matrix_rq, decompose_matrix_rq_3d};
use colmap_rust::math::polynomial::{
    find_cubic_polynomial_roots, find_linear_polynomial_roots,
    find_polynomial_roots_companion_matrix, find_polynomial_roots_durand_kerner,
    find_quadratic_polynomial_roots,
};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{
    random_eigen_matrix3d, random_eigen_matrix_xd, random_eigen_matrix_xf, random_eigen_matrixd,
    random_eigen_matrixf, random_eigen_vector3d, random_eigen_vector_xd, random_eigen_vector_xf,
};

#[test]
fn rust_only_polynomial_checks_return_err() {
    let three = VectorXd::from_slice(&[1.0, 2.0, 3.0]);
    let err = find_linear_polynomial_roots(&three).unwrap_err();
    assert!(err.to_string().contains("Check failed"), "{err}");
    assert!(find_quadratic_polynomial_roots(&VectorXd::zeros(2)).is_err());
    assert!(find_polynomial_roots_durand_kerner(&VectorXd::zeros(1)).is_err());
    assert!(find_polynomial_roots_companion_matrix(&VectorXd::zeros(1)).is_err());
}

#[test]
fn rust_only_polynomial_constant_has_no_roots() {
    // All-zero and nonzero-constant polynomials: C++ returns false (degree <= 0).
    for coeffs in [[0.0, 0.0, 0.0], [0.0, 0.0, 4.0]] {
        let coeffs = VectorXd::from_slice(&coeffs);
        assert!(find_polynomial_roots_durand_kerner(&coeffs)
            .unwrap()
            .is_none());
        assert!(find_polynomial_roots_companion_matrix(&coeffs)
            .unwrap()
            .is_none());
    }
}

#[test]
fn rust_only_companion_matrix_agrees_with_durand_kerner_on_real_roots() {
    // (x - 1)(x - 2)(x - 3)(x + 4) = x^4 - 2x^3 - 13x^2 + 38x - 24.
    let coeffs = VectorXd::from_slice(&[1.0, -2.0, -13.0, 38.0, -24.0]);
    for find in [
        find_polynomial_roots_companion_matrix,
        find_polynomial_roots_durand_kerner,
    ] {
        let roots = find(&coeffs).unwrap().unwrap();
        let mut real = roots.real.into_vec();
        real.sort_by(f64::total_cmp);
        for (r, e) in real.iter().zip([-4.0, 1.0, 2.0, 3.0]) {
            assert!((r - e).abs() <= 1e-8, "{r} vs {e}");
        }
        assert!(roots.imag.as_slice().iter().all(|x| x.abs() <= 1e-8));
    }
}

#[test]
fn rust_only_cubic_leaves_unused_entries_zero() {
    let (num_roots, real) = find_cubic_polynomial_roots(0.0, 1.0, 1.0);
    assert_eq!(num_roots, 1);
    assert_eq!((real.y, real.z), (0.0, 0.0));
}

#[test]
fn rust_only_decompose_matrix_rq_dynamic_and_3d() {
    set_prng_seed(7);
    for n in [2, 3, 5] {
        let a = random_eigen_matrix_xd(n, n);
        let (r, q) = decompose_matrix_rq(&a).unwrap();
        assert!(r.is_upper_triangular(DUMMY_PRECISION));
        assert!(q.is_unitary(DUMMY_PRECISION));
        assert!((q.determinant() - 1.0).abs() <= 1e-9);
        assert!(a.is_approx_with(&(&r * &q), 1e-9));
    }
    let a3: Matrix3d = random_eigen_matrix3d();
    let (r3, q3) = decompose_matrix_rq_3d(&a3);
    let (rx, qx) = decompose_matrix_rq(&MatrixXd::from(a3)).unwrap();
    assert_eq!(MatrixXd::from(r3), rx);
    assert_eq!(MatrixXd::from(q3), qx);
    assert!(decompose_matrix_rq(&MatrixXd::zeros(2, 3)).is_err());
}

#[test]
fn rust_only_random_eigen_fill_order_is_column_major_everywhere() {
    set_prng_seed(3);
    let fixed: Matrix4d = random_eigen_matrixd();
    set_prng_seed(3);
    let dynamic = random_eigen_matrix_xd(4, 4);
    assert_eq!(MatrixXd::from(fixed), dynamic);

    set_prng_seed(3);
    let v = random_eigen_vector3d();
    set_prng_seed(3);
    let vx = random_eigen_vector_xd(3);
    assert_eq!([v.x, v.y, v.z], vx.as_slice());

    // Float: the fixed array is column-major, the dynamic RowMajorMatrix holds (r, c).
    set_prng_seed(5);
    let columns: [f32; 6] = random_eigen_matrixf();
    set_prng_seed(5);
    let xf = random_eigen_matrix_xf(2, 3);
    for c in 0..3 {
        for r in 0..2 {
            assert_eq!(columns[c * 2 + r], xf.as_slice()[r * 3 + c]);
        }
    }
    set_prng_seed(5);
    assert_eq!(random_eigen_vector_xf(6), columns.to_vec());
}
