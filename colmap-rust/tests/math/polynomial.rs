// Port of COLMAP's src/colmap/math/polynomial_test.cc (1:1, same test names in snake_case,
// same values and tolerances). Tests colmap_rust::math::polynomial.
//
// Tiers: linear/quadratic roots are Tier A (exact comparisons where COLMAP uses EXPECT_EQ);
// the cubic, Durand-Kerner and companion-matrix roots are Tier B with COLMAP's tolerances
// (docs/CPP_DIVERGENCES.md entries 1, 50, 51). `EigenMatrixNear(v, tol)` is Eigen's
// `isApprox(v, tol)` after a shape check (`expect_near` below). The C++ output pointers
// become the returned `PolynomialRoots`; `None` stands for `false`. COLMAP's literals are
// kept digit for digit, hence the excessive_precision allowance.

#![allow(clippy::excessive_precision)]

use colmap_rust::linalg::{Complex, VectorXd};
use colmap_rust::math::polynomial::{
    evaluate_polynomial, find_cubic_polynomial_roots, find_linear_polynomial_roots,
    find_polynomial_roots_companion_matrix, find_polynomial_roots_durand_kerner,
    find_quadratic_polynomial_roots, PolynomialRoots,
};

type RootFinder = fn(&VectorXd) -> colmap_rust::Result<Option<PolynomialRoots>>;

fn v(values: &[f64]) -> VectorXd {
    VectorXd::from_slice(values)
}

/// `EXPECT_THAT(actual, EigenMatrixNear(expected, tol))`.
fn expect_near(actual: &VectorXd, expected: &[f64], tol: f64) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "shape: {actual} vs {expected:?}"
    );
    let expected = v(expected);
    if expected.as_slice().iter().all(|&x| x == 0.0) {
        // isApprox() is not well-defined for zero matrices.
        assert!(actual.norm() <= tol, "{actual} not near zero");
    } else {
        assert!(
            actual.is_approx_with(&expected, tol),
            "{actual} not near {expected}"
        );
    }
}

fn eval(coeffs: &VectorXd, re: f64, im: f64) -> Complex {
    evaluate_polynomial(coeffs, Complex::new(re, im))
}

/// COLMAP's CHECK_EQUAL_RESULT macro.
fn check_equal_result(f1: RootFinder, c1: &VectorXd, f2: RootFinder, c2: &VectorXd) {
    let r1 = f1(c1).unwrap();
    let r2 = f2(c2).unwrap();
    assert_eq!(r1.is_some(), r2.is_some());
    if let (Some(r1), Some(r2)) = (r1, r2) {
        assert_eq!(r1.real, r2.real);
        assert_eq!(r1.imag, r2.imag);
    }
}

#[test]
fn evaluate_polynomial_nominal() {
    // C++ deduces T = int for the literal 1; the value is exact either way.
    assert_eq!(
        evaluate_polynomial(&v(&[1.0, -3.0, 3.0, -5.0, 10.0]), 1.0),
        f64::from(1 - 3 + 3 - 5 + 10)
    );
    assert!(
        (evaluate_polynomial(&v(&[1.0, -3.0, 3.0, -5.0]), 2.0)
            - f64::from(2 * 2 * 2 - 3 * 2 * 2 + 3 * 2 - 5))
        .abs()
            <= 1e-6
    );
}

#[test]
fn find_linear_polynomial_roots_nominal() {
    let coeffs = v(&[3.0, -2.0]);
    let roots = find_linear_polynomial_roots(&coeffs).unwrap().unwrap();
    assert_eq!(roots.real[0], 2.0 / 3.0);
    assert_eq!(roots.imag[0], 0.0);
    let value = eval(&coeffs, roots.real[0], roots.imag[0]);
    assert!(value.re.abs() <= 1e-6);
    assert!(value.im.abs() <= 1e-6);

    assert!(find_linear_polynomial_roots(&v(&[0.0, 1.0]))
        .unwrap()
        .is_none());
}

#[test]
fn find_quadratic_polynomial_roots_real() {
    let coeffs = v(&[3.0, -2.0, -4.0]); // negative b
    let roots = find_quadratic_polynomial_roots(&coeffs).unwrap().unwrap();
    expect_near(&roots.real, &[-0.868517092, 1.535183758], 1e-6);
    assert_eq!(roots.imag, v(&[0.0, 0.0]));
    assert!(eval(&coeffs, roots.real[0], roots.imag[0]).re.abs() <= 1e-6);
    assert!(eval(&coeffs, roots.real[1], roots.imag[1]).im.abs() <= 1e-6);

    let coeffs = v(&[1.0, 5.0, 2.0]); // positive b
    let roots = find_quadratic_polynomial_roots(&coeffs).unwrap().unwrap();
    expect_near(
        &roots.real,
        &[-4.561552812808831, -0.4384471871911697],
        1e-6,
    );
    assert_eq!(roots.imag, v(&[0.0, 0.0]));
    assert!(eval(&coeffs, roots.real[0], roots.imag[0]).re.abs() <= 1e-6);
    assert!(eval(&coeffs, roots.real[1], roots.imag[1]).re.abs() <= 1e-6);
}

#[test]
fn find_quadratic_polynomial_roots_complex() {
    let coeffs = v(&[0.276025076998578, 0.679702676853675, 0.655098003973841]);
    let roots = find_quadratic_polynomial_roots(&coeffs).unwrap().unwrap();
    expect_near(&roots.real, &[-1.231233560813707, -1.231233560813707], 1e-6);
    expect_near(&roots.imag, &[0.925954520440279, -0.925954520440279], 1e-6);
    assert!(eval(&coeffs, roots.real[0], roots.imag[0]).re.abs() <= 1e-6);
    assert!(eval(&coeffs, roots.real[1], roots.imag[1]).im.abs() <= 1e-6);
}

#[test]
fn find_quadratic_polynomial_roots_zero_leading_coefficient() {
    let roots = find_quadratic_polynomial_roots(&v(&[0.0, 2.0, -4.0]))
        .unwrap()
        .unwrap();
    assert_eq!(roots.real.len(), 1);
    assert_eq!(roots.imag.len(), 1);
    assert_eq!(roots.real[0], 2.0);
    assert_eq!(roots.imag[0], 0.0);
}

#[test]
fn find_quadratic_polynomial_roots_only_zero_solution() {
    let roots = find_quadratic_polynomial_roots(&v(&[0.0, 2.0, 0.0]))
        .unwrap()
        .unwrap();
    assert_eq!(roots.real.len(), 1);
    assert_eq!(roots.imag.len(), 1);
    assert_eq!(roots.real[0], 0.0);
    assert_eq!(roots.imag[0], 0.0);
}

#[test]
fn find_quadratic_polynomial_roots_only_leading_coefficient_non_zero() {
    let roots = find_quadratic_polynomial_roots(&v(&[5.0, 0.0, 0.0]))
        .unwrap()
        .unwrap();
    assert_eq!(roots.real.len(), 1);
    assert_eq!(roots.imag.len(), 1);
    assert_eq!(roots.real[0], 0.0);
    assert_eq!(roots.imag[0], 0.0);
}

#[test]
fn find_cubic_polynomial_roots_single_root() {
    let coeffs = v(&[1.0, 0.276025076998578, 0.679702676853675, 0.655098003973841]);
    let (num_roots, real) = find_cubic_polynomial_roots(coeffs[1], coeffs[2], coeffs[3]);
    assert_eq!(num_roots, 1);
    assert!((real.x - -0.68359403879256575).abs() <= 1e-6);
    assert!(eval(&coeffs, real.x, 0.0).re.abs() <= 1e-6);
}

#[test]
fn find_cubic_polynomial_roots_multi_root() {
    let coeffs = v(&[1.0, -3.0, -3.0, 5.0]);
    let (num_roots, real) = find_cubic_polynomial_roots(coeffs[1], coeffs[2], coeffs[3]);
    assert_eq!(num_roots, 3);
    let mut real = [real.x, real.y, real.z];
    real.sort_by(f64::total_cmp);
    let real = v(&real);
    expect_near(&real, &[-1.4494897427831781, 1.0, 3.4494897427831783], 1e-6);
    for i in 0..3 {
        assert!(eval(&coeffs, real[i], 0.0).re.abs() <= 1e-6);
    }
    let roots_durand_kerner = find_polynomial_roots_durand_kerner(&coeffs)
        .unwrap()
        .unwrap();
    let mut real_durand_kerner = roots_durand_kerner.real.into_vec();
    real_durand_kerner.sort_by(f64::total_cmp);
    expect_near(&real, &real_durand_kerner, 1e-4);
}

#[test]
fn find_polynomial_roots_durand_kerner_nominal() {
    let coeffs = v(&[10.0, -5.0, 3.0, -3.0, 1.0]);
    let roots = find_polynomial_roots_durand_kerner(&coeffs)
        .unwrap()
        .unwrap();
    // Reference values generated with OpenCV/Matlab.
    expect_near(
        &roots.real,
        &[-0.201826, -0.201826, 0.451826, 0.451826],
        1e-6,
    );
    expect_near(
        &roots.imag,
        &[-0.627696, 0.627696, 0.160867, -0.160867],
        1e-6,
    );
}

#[test]
fn find_polynomial_roots_durand_kerner_linear_quadratic() {
    check_equal_result(
        find_polynomial_roots_durand_kerner,
        &v(&[1.0, 2.0]),
        find_linear_polynomial_roots,
        &v(&[1.0, 2.0]),
    );
    check_equal_result(
        find_polynomial_roots_durand_kerner,
        &v(&[0.0, 0.0, 1.0, 2.0]),
        find_linear_polynomial_roots,
        &v(&[1.0, 2.0]),
    );
    check_equal_result(
        find_polynomial_roots_durand_kerner,
        &v(&[1.0, 2.0, 3.0]),
        find_quadratic_polynomial_roots,
        &v(&[1.0, 2.0, 3.0]),
    );
    check_equal_result(
        find_polynomial_roots_durand_kerner,
        &v(&[0.0, 0.0, 1.0, 2.0, 3.0]),
        find_quadratic_polynomial_roots,
        &v(&[1.0, 2.0, 3.0]),
    );
}

#[test]
fn find_polynomial_roots_companion_matrix_nominal() {
    let coeffs = v(&[10.0, -5.0, 3.0, -3.0, 1.0]);
    let roots = find_polynomial_roots_companion_matrix(&coeffs)
        .unwrap()
        .unwrap();
    // Reference values generated with OpenCV/Matlab.
    expect_near(
        &roots.real,
        &[-0.201826, -0.201826, 0.451826, 0.451826],
        1e-6,
    );
    expect_near(
        &roots.imag,
        &[0.627696, -0.627696, 0.160867, -0.160867],
        1e-6,
    );
}

#[test]
fn find_polynomial_roots_companion_matrix_linear_quadratic() {
    check_equal_result(
        find_polynomial_roots_companion_matrix,
        &v(&[1.0, 2.0]),
        find_linear_polynomial_roots,
        &v(&[1.0, 2.0]),
    );
    check_equal_result(
        find_polynomial_roots_companion_matrix,
        &v(&[0.0, 0.0, 1.0, 2.0]),
        find_linear_polynomial_roots,
        &v(&[1.0, 2.0]),
    );
    check_equal_result(
        find_polynomial_roots_companion_matrix,
        &v(&[1.0, 2.0, 3.0]),
        find_quadratic_polynomial_roots,
        &v(&[1.0, 2.0, 3.0]),
    );
    check_equal_result(
        find_polynomial_roots_companion_matrix,
        &v(&[0.0, 0.0, 1.0, 2.0, 3.0]),
        find_quadratic_polynomial_roots,
        &v(&[1.0, 2.0, 3.0]),
    );
}

#[test]
fn find_polynomial_roots_companion_matrix_zero_solution() {
    let coeffs = v(&[10.0, -5.0, 3.0, -3.0, 0.0]);
    let roots = find_polynomial_roots_companion_matrix(&coeffs)
        .unwrap()
        .unwrap();
    // Reference values generated with Matlab.
    expect_near(&roots.real, &[0.692438, -0.0962191, -0.0962191, 0.0], 1e-6);
    expect_near(&roots.imag, &[0.0, 0.651148, -0.651148, 0.0], 1e-6);
}

#[test]
fn find_polynomial_roots_companion_matrix_only_zero_solution() {
    let coeffs = v(&[0.0, 0.0, 5.0, 0.0, 0.0, 0.0]);
    let roots = find_polynomial_roots_companion_matrix(&coeffs)
        .unwrap()
        .unwrap();
    assert_eq!(roots.real.len(), 1);
    assert_eq!(roots.imag.len(), 1);
    assert_eq!(roots.real[0], 0.0);
    assert_eq!(roots.imag[0], 0.0);
}
