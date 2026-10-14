// Port of Ceres Solver 2.2.0's internal/ceres/polynomial_test.cc and the ExpectClose helper of
// internal/ceres/test_util.cc (BSD-3-Clause, see THIRD_PARTY_NOTICES.md), via colmap-sharp's
// `ColmapSharp.Tests/Solver/CeresPolynomialTests.cs`: `colmap_rust::solver::ceres_polynomial`,
// same 26 cases, values and tolerances. Ceres' suite is `Polynomial` (one case is spelled
// `Polymomial.ConstantInterpolatingPolynomial`; the typo is dropped here). Tier B: the roots
// of degree three and up come from `linalg::EigenSolver`.
// - `find_polynomial_roots` always returns both parts; the NullPointerAs*Part /
//   BothOutputArgumentsNull cases check only the parts Ceres asks for, so they still exercise
//   the same root finding.
// Not ported: line_search_minimizer_test.cc and line_search_preprocessor_test.cc test Ceres'
// line search minimizer, which is not ported.

use colmap_rust::linalg::VectorXd;
use colmap_rust::solver::ceres_polynomial::{
    differentiate_polynomial, evaluate_polynomial, find_interpolating_polynomial,
    find_polynomial_roots, minimize_polynomial, FunctionSample,
};

// For IEEE-754 doubles, machine precision is about 2e-16.
const K_EPSILON: f64 = 1e-13;
const K_EPSILON_LOOSE: f64 = 1e-9;

// Return the constant polynomial p(x) = 1.23.
fn constant_polynomial(value: f64) -> VectorXd {
    VectorXd::from_slice(&[value])
}

// Return the polynomial p(x) = poly(x) * (x - root).
fn add_real_root(poly: &VectorXd, root: f64) -> VectorXd {
    let mut poly2 = VectorXd::zeros(poly.len() + 1);
    for i in 0..poly.len() {
        poly2[i] += poly[i];
    }
    for i in 0..poly.len() {
        poly2[i + 1] -= root * poly[i];
    }
    poly2
}

// Return the polynomial p(x) = poly(x) * (x - real - imag*i) * (x - real + imag*i).
fn add_complex_root_pair(poly: &VectorXd, real: f64, imag: f64) -> VectorXd {
    let mut poly2 = VectorXd::zeros(poly.len() + 2);
    // Multiply poly by x^2 - 2real + abs(real,imag)^2
    for i in 0..poly.len() {
        poly2[i] += poly[i];
    }
    for i in 0..poly.len() {
        poly2[i + 1] -= 2.0 * real * poly[i];
    }
    for i in 0..poly.len() {
        poly2[i + 2] += (real * real + imag * imag) * poly[i];
    }
    poly2
}

// test_util.cc ExpectClose: relative difference, absolute when either value is zero.
fn expect_close(x: f64, y: f64, max_abs_relative_difference: f64) {
    if x.is_infinite() && y.is_infinite() {
        assert_eq!(x.is_sign_negative(), y.is_sign_negative());
        return;
    }
    if x.is_nan() && y.is_nan() {
        return;
    }

    let absolute_difference = (x - y).abs();
    let mut relative_difference = absolute_difference / x.abs().max(y.abs());
    if x == 0.0 || y == 0.0 {
        // If x or y is exactly zero, then relative difference doesn't have any meaning. Take
        // the absolute difference instead.
        relative_difference = absolute_difference;
    }
    assert!(
        relative_difference.abs() <= max_abs_relative_difference,
        "x = {x}, y = {y}, relative difference {relative_difference} > {max_abs_relative_difference}"
    );
}

// Run a test with the polynomial defined by the N real roots in roots_real. If use_real is
// false the real parts are not checked; if use_imaginary is false the imaginary parts are not.
fn run_polynomial_test_real_roots(
    real_roots: &[f64],
    use_real: bool,
    use_imaginary: bool,
    epsilon: f64,
) {
    let mut poly = constant_polynomial(1.23);
    for &root in real_roots {
        poly = add_real_root(&poly, root);
    }

    let roots = find_polynomial_roots(&poly);
    assert!(roots.is_some());
    let (real, imaginary) = roots.unwrap();
    let n = real_roots.len();
    if use_real {
        assert_eq!(real.len(), n);
        // Needed because the roots are not returned in sorted order.
        let mut sorted = real.as_slice().to_vec();
        sorted.sort_by(f64::total_cmp);
        for i in 0..n {
            expect_close(sorted[i], real_roots[i], epsilon);
        }
    }
    if use_imaginary {
        assert_eq!(imaginary.len(), n);
        for i in 0..n {
            expect_close(imaginary[i], 0.0, epsilon);
        }
    }
}

fn diff_norm(a: &VectorXd, b: &VectorXd) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut sum = 0.0;
    for i in 0..a.len() {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    sum.sqrt()
}

#[test]
fn polynomial_invalid_polynomial_of_zero_length_is_rejected() {
    let poly = VectorXd::zeros(0);
    assert!(find_polynomial_roots(&poly).is_none());
}

#[test]
fn polynomial_constant_polynomial_returns_no_roots() {
    let poly = constant_polynomial(1.23);
    let (real, imag) = find_polynomial_roots(&poly).expect("success");
    assert_eq!(real.len(), 0);
    assert_eq!(imag.len(), 0);
}

#[test]
fn polynomial_linear_polynomial_with_positive_root_works() {
    run_polynomial_test_real_roots(&[42.42], true, true, K_EPSILON);
}

#[test]
fn polynomial_linear_polynomial_with_negative_root_works() {
    run_polynomial_test_real_roots(&[-42.42], true, true, K_EPSILON);
}

#[test]
fn polynomial_quadratic_polynomial_with_positive_roots_works() {
    run_polynomial_test_real_roots(&[1.0, 42.42], true, true, K_EPSILON);
}

#[test]
fn polynomial_quadratic_polynomial_with_one_negative_root_works() {
    run_polynomial_test_real_roots(&[-42.42, 1.0], true, true, K_EPSILON);
}

#[test]
fn polynomial_quadratic_polynomial_with_two_negative_roots_works() {
    run_polynomial_test_real_roots(&[-42.42, -1.0], true, true, K_EPSILON);
}

#[test]
fn polynomial_quadratic_polynomial_with_close_roots_works() {
    run_polynomial_test_real_roots(&[42.42, 42.43], true, false, K_EPSILON_LOOSE);
}

#[test]
fn polynomial_quadratic_polynomial_with_complex_roots_works() {
    let poly = add_complex_root_pair(&constant_polynomial(1.23), 42.42, 4.2);
    let (real, imag) = find_polynomial_roots(&poly).expect("success");

    assert_eq!(real.len(), 2);
    assert_eq!(imag.len(), 2);
    expect_close(real[0], 42.42, K_EPSILON);
    expect_close(real[1], 42.42, K_EPSILON);
    expect_close(imag[0].abs(), 4.2, K_EPSILON);
    expect_close(imag[1].abs(), 4.2, K_EPSILON);
    expect_close((imag[0] + imag[1]).abs(), 0.0, K_EPSILON);
}

#[test]
fn polynomial_quartic_polynomial_works() {
    run_polynomial_test_real_roots(&[1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5], true, true, K_EPSILON);
}

#[test]
fn polynomial_quartic_polynomial_with_two_clusters_of_close_roots_works() {
    run_polynomial_test_real_roots(
        &[1.23e-1, 2.46e-1, 1.23e+5, 2.46e+5],
        true,
        true,
        K_EPSILON_LOOSE,
    );
}

#[test]
fn polynomial_quartic_polynomial_with_two_zero_roots_works() {
    run_polynomial_test_real_roots(
        &[-42.42, 0.0, 0.0, 42.42],
        true,
        true,
        2.0 * K_EPSILON_LOOSE,
    );
}

#[test]
fn polynomial_quartic_monomial_works() {
    run_polynomial_test_real_roots(&[0.0, 0.0, 0.0, 0.0], true, true, K_EPSILON);
}

#[test]
fn polynomial_null_pointer_as_imaginary_part_works() {
    run_polynomial_test_real_roots(
        &[1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5],
        true,
        false,
        K_EPSILON,
    );
}

#[test]
fn polynomial_null_pointer_as_real_part_works() {
    run_polynomial_test_real_roots(
        &[1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5],
        false,
        true,
        K_EPSILON,
    );
}

#[test]
fn polynomial_both_output_arguments_null_works() {
    run_polynomial_test_real_roots(
        &[1.23e-4, 1.23e-1, 1.23e+2, 1.23e+5],
        false,
        false,
        K_EPSILON,
    );
}

#[test]
fn polynomial_differentiate_constant_polynomial() {
    // p(x) = 1;
    let polynomial = VectorXd::from_slice(&[1.0]);
    let derivative = differentiate_polynomial(&polynomial).unwrap();
    assert_eq!(derivative.len(), 1);
    assert_eq!(derivative[0], 0.0);
}

#[test]
fn polynomial_differentiate_quadratic_polynomial() {
    // p(x) = x^2 + 2x + 3;
    let polynomial = VectorXd::from_slice(&[1.0, 2.0, 3.0]);
    let derivative = differentiate_polynomial(&polynomial).unwrap();
    assert_eq!(derivative.len(), 2);
    assert_eq!(derivative[0], 2.0);
    assert_eq!(derivative[1], 2.0);
}

#[test]
fn polynomial_minimize_constant_polynomial() {
    // p(x) = 1;
    let polynomial = VectorXd::from_slice(&[1.0]);
    let min_x = 0.0;
    let max_x = 1.0;
    let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, min_x, max_x);
    assert_eq!(optimal_value, 1.0);
    assert!(optimal_x <= max_x);
    assert!(optimal_x >= min_x);
}

#[test]
fn polynomial_minimize_linear_polynomial() {
    // p(x) = x - 2
    let polynomial = VectorXd::from_slice(&[1.0, 2.0]);
    let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, 0.0, 1.0);
    assert_eq!(optimal_x, 0.0);
    assert_eq!(optimal_value, 2.0);
}

#[test]
fn polynomial_minimize_quadratic_polynomial() {
    // p(x) = x^2 - 3 x + 2
    // min_x = 3/2
    // min_value = -1/4;
    let polynomial = VectorXd::from_slice(&[1.0, -3.0, 2.0]);

    let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, -2.0, 2.0);
    assert_eq!(optimal_x, 3.0 / 2.0);
    assert_eq!(optimal_value, -1.0 / 4.0);

    let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, -2.0, 1.0);
    assert_eq!(optimal_x, 1.0);
    assert_eq!(optimal_value, 0.0);

    let (optimal_x, optimal_value) = minimize_polynomial(&polynomial, 2.0, 3.0);
    assert_eq!(optimal_x, 2.0);
    assert_eq!(optimal_value, 0.0);
}

#[test]
fn polynomial_constant_interpolating_polynomial() {
    // p(x) = 1.0
    let true_polynomial = VectorXd::from_slice(&[1.0]);
    let samples = [FunctionSample::with_value(1.0, 1.0)];
    let polynomial = find_interpolating_polynomial(&samples);
    assert!(diff_norm(&true_polynomial, &polynomial) <= 1e-15);
}

#[test]
fn polynomial_linear_interpolating_polynomial() {
    // p(x) = 2x - 1
    let true_polynomial = VectorXd::from_slice(&[2.0, -1.0]);
    let samples = [FunctionSample::with_value_and_gradient(1.0, 1.0, 2.0)];
    let polynomial = find_interpolating_polynomial(&samples);
    assert!(diff_norm(&true_polynomial, &polynomial) <= 1e-15);
}

#[test]
fn polynomial_quadratic_interpolating_polynomial() {
    // p(x) = 2x^2 + 3x + 2
    let true_polynomial = VectorXd::from_slice(&[2.0, 3.0, 2.0]);
    let samples = [
        FunctionSample::with_value_and_gradient(1.0, 7.0, 7.0),
        FunctionSample::with_value(-3.0, 11.0),
    ];
    let polynomial = find_interpolating_polynomial(&samples);
    assert!(diff_norm(&true_polynomial, &polynomial) <= 1e-15);
}

#[test]
fn polynomial_deficient_cubic_interpolating_polynomial() {
    // p(x) = 2x^2 + 3x + 2
    let true_polynomial = VectorXd::from_slice(&[0.0, 2.0, 3.0, 2.0]);
    let samples = [
        FunctionSample::with_value_and_gradient(1.0, 7.0, 7.0),
        FunctionSample::with_value_and_gradient(-3.0, 11.0, -9.0),
    ];
    let polynomial = find_interpolating_polynomial(&samples);
    assert!(diff_norm(&true_polynomial, &polynomial) <= 1e-14);
}

#[test]
fn polynomial_cubic_interpolating_polynomial_from_values() {
    // p(x) = x^3 + 2x^2 + 3x + 2
    let true_polynomial = VectorXd::from_slice(&[1.0, 2.0, 3.0, 2.0]);
    let samples: Vec<FunctionSample> = [1.0, -3.0, 2.0, 0.0]
        .iter()
        .map(|&x| FunctionSample::with_value(x, evaluate_polynomial(&true_polynomial, x)))
        .collect();
    let polynomial = find_interpolating_polynomial(&samples);
    assert!(diff_norm(&true_polynomial, &polynomial) <= 1e-14);
}

#[test]
fn polynomial_cubic_interpolating_polynomial_from_values_and_one_gradient() {
    // p(x) = x^3 + 2x^2 + 3x + 2
    let true_polynomial = VectorXd::from_slice(&[1.0, 2.0, 3.0, 2.0]);
    let true_gradient_polynomial = differentiate_polynomial(&true_polynomial).unwrap();
    let samples = [
        FunctionSample::with_value(1.0, evaluate_polynomial(&true_polynomial, 1.0)),
        FunctionSample::with_value(-3.0, evaluate_polynomial(&true_polynomial, -3.0)),
        FunctionSample::with_value_and_gradient(
            2.0,
            evaluate_polynomial(&true_polynomial, 2.0),
            evaluate_polynomial(&true_gradient_polynomial, 2.0),
        ),
    ];
    let polynomial = find_interpolating_polynomial(&samples);
    assert!(diff_norm(&true_polynomial, &polynomial) <= 1e-14);
}

#[test]
fn polynomial_cubic_interpolating_polynomial_from_values_and_gradients() {
    // p(x) = x^3 + 2x^2 + 3x + 2
    let true_polynomial = VectorXd::from_slice(&[1.0, 2.0, 3.0, 2.0]);
    let true_gradient_polynomial = differentiate_polynomial(&true_polynomial).unwrap();
    let samples = [
        FunctionSample::with_value_and_gradient(
            -3.0,
            evaluate_polynomial(&true_polynomial, -3.0),
            evaluate_polynomial(&true_gradient_polynomial, -3.0),
        ),
        FunctionSample::with_value_and_gradient(
            2.0,
            evaluate_polynomial(&true_polynomial, 2.0),
            evaluate_polynomial(&true_gradient_polynomial, 2.0),
        ),
    ];
    let polynomial = find_interpolating_polynomial(&samples);
    assert!(diff_norm(&true_polynomial, &polynomial) <= 1e-14);
}

// Rust-only: Ceres CHECK-fails differentiating an empty polynomial; the port returns the
// check's error.
#[test]
fn rust_only_differentiate_empty_polynomial_is_an_error() {
    assert!(differentiate_polynomial(&VectorXd::zeros(0)).is_err());
}
