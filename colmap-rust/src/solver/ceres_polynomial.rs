//! Ceres' own polynomial helpers, which its line search uses to pick a step size: fit the
//! polynomial through function samples (values and derivatives), then minimize it on an
//! interval by checking the ends, the middle and the real parts of the derivative's roots.
//! Port of colmap-sharp's `ColmapSharp/Solver/CeresPolynomial.cs`, which ports Ceres Solver
//! 2.2.0's `internal/ceres/polynomial.cc`, `polynomial.h` and `function_sample.h`
//! (BSD-3-Clause, see THIRD_PARTY_NOTICES.md). The trust-region line search (a later Phase 7
//! slice) is the caller.
//!
//! These differ from COLMAP's `math/polynomial.rs`: the companion matrix is balanced by powers
//! of two before the eigenvalues are taken, and the interpolation system is solved with
//! full-pivoting LU at a zero threshold. The eigenvalues come from
//! [`crate::linalg::EigenSolver`] and the LU from [`crate::linalg::FullPivLu`], both written
//! from the published algorithms, so results are Tier B.
//!
//! The balancing's 1-norms are left-to-right sums in coefficient order, a deliberate
//! divergence: `BalanceCompanionMatrix` takes them with Eigen's `lpNorm<1>()`, which may
//! vectorize and reduce in packet lanes whose order depends on the target's SIMD width and the
//! compiler. Eigen is MPL-2.0 and not ported (contract rule 2), there is no single order to
//! match, and a sequential order gives the same bits on every target, native and wasm. Results
//! can differ from COLMAP's Ceres in the last bits. Evidence: `tests/solver/ceres_polynomial.rs`
//! (Ceres' `polynomial_test.cc`) passes at Ceres' own tolerances. Same as colmap-sharp entry
//! 115.
//!
//! Polynomials are coefficient vectors, highest degree first.
//!
//! Tests: `tests/solver/ceres_polynomial.rs` (Ceres' `polynomial_test.cc`).

use crate::linalg::{ComputationInfo, EigenSolver, FullPivLu, MatrixXd, VectorXd};
use crate::math::fns;
use crate::{check_ge, Result};

/// Port of `ceres::internal::FunctionSample`: a sample of a univariate function (the line
/// search's cost along the step) with its value and derivative, each possibly invalid.
/// Ceres' vector-valued members (`vector_x`, `vector_gradient`) belong to its line search's
/// bookkeeping, not to the polynomial code, and come with that slice if it needs them.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FunctionSample {
    /// The argument.
    pub x: f64,
    /// The function value.
    pub value: f64,
    /// Whether `value` is valid.
    pub value_is_valid: bool,
    /// The derivative.
    pub gradient: f64,
    /// Whether `gradient` is valid.
    pub gradient_is_valid: bool,
}

impl FunctionSample {
    /// A sample with a valid value.
    pub fn with_value(x: f64, value: f64) -> Self {
        Self {
            x,
            value,
            value_is_valid: true,
            ..Self::default()
        }
    }

    /// A sample with a valid value and derivative.
    pub fn with_value_and_gradient(x: f64, value: f64, gradient: f64) -> Self {
        Self {
            x,
            value,
            value_is_valid: true,
            gradient,
            gradient_is_valid: true,
        }
    }
}

/// Port of `ceres::internal::EvaluatePolynomial`: Horner evaluation, highest degree first.
pub fn evaluate_polynomial(polynomial: &VectorXd, x: f64) -> f64 {
    let mut v = 0.0;
    for &coefficient in polynomial.as_slice() {
        v = v * x + coefficient;
    }
    v
}

/// Port of `ceres::internal::FindPolynomialRoots`: the (complex) roots as (real parts,
/// imaginary parts) after dropping leading zeros; closed forms up to degree 2, the balanced
/// companion matrix's eigenvalues above. A constant polynomial has no roots (and succeeds).
/// `None` for an empty polynomial or when the eigenvalue iteration fails, where Ceres logs an
/// error and returns false.
///
/// Ceres fills only the outputs the caller asks for; both are always returned here.
pub fn find_polynomial_roots(polynomial_in: &VectorXd) -> Option<(VectorXd, VectorXd)> {
    if polynomial_in.is_empty() {
        return None;
    }

    let mut polynomial = remove_leading_zeros(polynomial_in);
    let degree = polynomial.len() - 1;

    // Is the polynomial constant? We return true with no roots, not false, as if the
    // polynomial is constant it is correct that there are no roots. It is not the case that
    // they were there, but that we have failed to extract them.
    if degree == 0 {
        return Some((VectorXd::zeros(0), VectorXd::zeros(0)));
    }

    if degree == 1 {
        let real = VectorXd::from_slice(&[-polynomial[1] / polynomial[0]]);
        return Some((real, VectorXd::zeros(1)));
    }

    if degree == 2 {
        return Some(find_quadratic_polynomial_roots(&polynomial));
    }

    // The degree is now known to be at least 3. For cubic or higher roots we use the method
    // of companion matrices. Divide by leading term.
    let leading_term = polynomial[0];
    for coefficient in polynomial.as_mut_slice() {
        *coefficient /= leading_term;
    }

    let mut companion = build_companion_matrix(&polynomial);
    balance_companion_matrix(&mut companion);
    let solver = EigenSolver::new(&companion, false);
    if solver.info() != ComputationInfo::Success {
        return None;
    }

    let eigenvalues = solver.eigenvalues();
    let real = eigenvalues.iter().map(|value| value.re).collect();
    let imaginary = eigenvalues.iter().map(|value| value.im).collect();
    Some((VectorXd::from_vec(real), VectorXd::from_vec(imaginary)))
}

/// Port of `ceres::internal::DifferentiatePolynomial`: the derivative; a constant's is the
/// zero constant. Ceres CHECK-fails on an empty polynomial; the port returns the check's
/// error.
pub fn differentiate_polynomial(polynomial: &VectorXd) -> Result<VectorXd> {
    // Ceres' int degree is -1 for an empty polynomial.
    check_ge!(polynomial.len() as i64 - 1, 0);
    Ok(derivative_of(polynomial))
}

/// The body of [`differentiate_polynomial`] for a non-empty polynomial.
fn derivative_of(polynomial: &VectorXd) -> VectorXd {
    let degree = polynomial.len() - 1;

    // Degree zero polynomials are constants, and their derivative does not result in a
    // smaller degree polynomial, just a degree zero polynomial with value zero.
    if degree == 0 {
        return VectorXd::zeros(1);
    }

    let mut derivative = VectorXd::zeros(degree);
    for i in 0..degree {
        derivative[i] = (degree - i) as f64 * polynomial[i];
    }
    derivative
}

/// Port of `ceres::internal::MinimizePolynomial`: the (argument, value) of the minimum of the
/// polynomial on [`x_min`, `x_max`], checking the middle, the ends and the real parts of the
/// critical points, in that order (strictly smaller wins, so the first of equal values is
/// kept).
pub fn minimize_polynomial(polynomial: &VectorXd, x_min: f64, x_max: f64) -> (f64, f64) {
    // We start by inspecting the middle of the interval. Technically this is not needed, but
    // we do this to make this code as close to the minFunc package as possible.
    let mut optimal_x = (x_min + x_max) / 2.0;
    let mut optimal_value = evaluate_polynomial(polynomial, optimal_x);

    let x_min_value = evaluate_polynomial(polynomial, x_min);
    if x_min_value < optimal_value {
        optimal_value = x_min_value;
        optimal_x = x_min;
    }

    let x_max_value = evaluate_polynomial(polynomial, x_max);
    if x_max_value < optimal_value {
        optimal_value = x_max_value;
        optimal_x = x_max;
    }

    // If the polynomial is linear or constant, we are done.
    if polynomial.len() <= 2 {
        return (optimal_x, optimal_value);
    }

    let derivative = derivative_of(polynomial);
    let Some((roots_real, _)) = find_polynomial_roots(&derivative) else {
        // Ceres logs "Unable to find the critical points of the interpolating polynomial."
        return (optimal_x, optimal_value);
    };

    // This is a bit of an overkill, as some of the roots may actually have a complex part,
    // but its simpler to just check these values.
    for &root in roots_real.as_slice() {
        if root < x_min || root > x_max {
            continue;
        }

        let value = evaluate_polynomial(polynomial, root);
        if value < optimal_value {
            optimal_value = value;
            optimal_x = root;
        }
    }

    (optimal_x, optimal_value)
}

/// Port of `ceres::internal::FindInterpolatingPolynomial`: the polynomial of degree
/// (constraints - 1) matching every valid value and derivative of the samples.
pub fn find_interpolating_polynomial(samples: &[FunctionSample]) -> VectorXd {
    let num_constraints: usize = samples
        .iter()
        .map(|sample| usize::from(sample.value_is_valid) + usize::from(sample.gradient_is_valid))
        .sum();

    // Ceres' int degree; -1 (no constraints) gives an empty system and an empty solution.
    let degree = num_constraints as i32 - 1;
    let mut lhs = MatrixXd::zeros(num_constraints, num_constraints);
    let mut rhs = VectorXd::zeros(num_constraints);
    let mut row = 0;
    for sample in samples {
        if sample.value_is_valid {
            for j in 0..=degree {
                lhs[(row, j as usize)] = fns::pow(sample.x, f64::from(degree - j));
            }
            rhs[row] = sample.value;
            row += 1;
        }

        if sample.gradient_is_valid {
            for j in 0..degree {
                lhs[(row, j as usize)] =
                    f64::from(degree - j) * fns::pow(sample.x, f64::from(degree - j - 1));
            }
            rhs[row] = sample.gradient;
            row += 1;
        }
    }

    // Ceres: "This is a hack" (ceres-solver issue 248): full-pivoting LU with a zero
    // threshold, so a singular system still gives an answer.
    FullPivLu::new(&lhs).solve(&rhs, 0.0)
}

/// Port of `ceres::internal::MinimizeInterpolatingPolynomial`: minimizes the interpolating
/// polynomial on [`x_min`, `x_max`], also trying the samples inside the interval. Returns
/// (argument, value).
pub fn minimize_interpolating_polynomial(
    samples: &[FunctionSample],
    x_min: f64,
    x_max: f64,
) -> (f64, f64) {
    let polynomial = find_interpolating_polynomial(samples);
    let (mut optimal_x, mut optimal_value) = minimize_polynomial(&polynomial, x_min, x_max);
    for sample in samples {
        if sample.x < x_min || sample.x > x_max {
            continue;
        }

        let value = evaluate_polynomial(&polynomial, sample.x);
        if value < optimal_value {
            optimal_x = sample.x;
            optimal_value = value;
        }
    }
    (optimal_x, optimal_value)
}

/// The polynomial without its leading zero coefficients (the last one is always kept).
fn remove_leading_zeros(polynomial_in: &VectorXd) -> VectorXd {
    let mut i = 0;
    while i < polynomial_in.len() - 1 && polynomial_in[i] == 0.0 {
        i += 1;
    }
    polynomial_in.tail(polynomial_in.len() - i)
}

/// Ceres' `FindQuadraticPolynomialRoots`.
fn find_quadratic_polynomial_roots(polynomial: &VectorXd) -> (VectorXd, VectorXd) {
    let a = polynomial[0];
    let b = polynomial[1];
    let c = polynomial[2];
    let d = b * b - 4.0 * a * c;
    let sqrt_d = fns::sqrt(d.abs());
    let mut real = VectorXd::zeros(2);
    let mut imaginary = VectorXd::zeros(2);

    // Real roots.
    if d >= 0.0 {
        // Stable quadratic roots according to BKP Horn.
        // http://people.csail.mit.edu/bkph/articles/Quadratics.pdf
        if b >= 0.0 {
            real[0] = (-b - sqrt_d) / (2.0 * a);
            real[1] = (2.0 * c) / (-b - sqrt_d);
        } else {
            real[0] = (2.0 * c) / (-b + sqrt_d);
            real[1] = (-b + sqrt_d) / (2.0 * a);
        }
        return (real, imaginary);
    }

    // Use the normal quadratic formula for the complex case.
    real[0] = -b / (2.0 * a);
    real[1] = -b / (2.0 * a);
    imaginary[0] = sqrt_d / (2.0 * a);
    imaginary[1] = -sqrt_d / (2.0 * a);
    (real, imaginary)
}

/// Ceres' `BuildCompanionMatrix`: ones on the subdiagonal, the negated (monic) coefficients,
/// lowest degree first, in the last column.
fn build_companion_matrix(polynomial: &VectorXd) -> MatrixXd {
    let degree = polynomial.len() - 1;
    let mut companion = MatrixXd::zeros(degree, degree);
    for i in 1..degree {
        companion[(i, i - 1)] = 1.0;
    }
    for i in 0..degree {
        companion[(i, degree - 1)] = -polynomial[degree - i];
    }
    companion
}

/// Ceres' `BalanceCompanionMatrix`: greedily scales row/column pairs of the off-diagonal part
/// by powers of two until the 1-norms stop improving; the diagonal is left alone.
fn balance_companion_matrix(companion: &mut MatrixXd) {
    let degree = companion.rows();
    let mut off_diagonal = companion.clone();
    for i in 0..degree {
        off_diagonal[(i, i)] = 0.0;
    }

    // gamma <= 1 controls how much a change in the scaling has to lower the 1-norm of the
    // companion matrix to be accepted. gamma = 1 seems to lead to cycles (numerical issues?),
    // so we set it slightly lower.
    const GAMMA: f64 = 0.9;

    // Greedily scale row/column pairs until there is no change.
    loop {
        let mut scaling_has_changed = false;
        for i in 0..degree {
            let mut row_norm = 0.0;
            let mut col_norm = 0.0;
            for j in 0..degree {
                row_norm += off_diagonal[(i, j)].abs();
                col_norm += off_diagonal[(j, i)].abs();
            }

            // Decompose row_norm/col_norm into mantissa * 2^exponent, where
            // 0.5 <= mantissa < 1. Discard mantissa (return value of frexp), as only the
            // exponent is needed. (C++'s int division truncates toward zero, as Rust's does.)
            let exponent = frexp_exponent(row_norm / col_norm) / 2;
            if exponent != 0 {
                let scaled_col_norm = col_norm * pow2(exponent);
                let scaled_row_norm = row_norm * pow2(-exponent);
                if scaled_col_norm + scaled_row_norm < GAMMA * (col_norm + row_norm) {
                    // Accept the new scaling. (Multiplication by powers of 2 should not
                    // introduce rounding errors (ignoring non-normalized numbers and over- or
                    // underflow))
                    scaling_has_changed = true;
                    let row_scale = pow2(-exponent);
                    let col_scale = pow2(exponent);
                    for j in 0..degree {
                        off_diagonal[(i, j)] *= row_scale;
                    }
                    for j in 0..degree {
                        off_diagonal[(j, i)] *= col_scale;
                    }
                }
            }
        }

        if !scaling_has_changed {
            break;
        }
    }

    for i in 0..degree {
        for j in 0..degree {
            if i != j {
                companion[(i, j)] = off_diagonal[(i, j)];
            }
        }
    }
}

/// `std::frexp`'s exponent: x = m 2^e with 0.5 <= |m| < 1; 0 for zero and (as libc++ on macOS
/// and glibc return) for infinity and NaN. Subnormals get their true exponent.
fn frexp_exponent(x: f64) -> i32 {
    if x == 0.0 || !x.is_finite() {
        return 0;
    }

    let biased = ((x.to_bits() >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: scale by 2^54 (exact) into the normal range first.
        let scaled = x * pow2(54);
        return ((scaled.to_bits() >> 52) & 0x7ff) as i32 - 1022 - 54;
    }
    biased - 1022
}

/// 2^`e` for |e| <= 1022, built from its bits (exact), so `x * pow2(e)` is `std::ldexp(x, e)`:
/// one correctly rounded multiplication by a normal power of two. The balancing's exponents
/// are at most 512 in magnitude (half of frexp's range).
fn pow2(e: i32) -> f64 {
    f64::from_bits(((1023 + e) as u64) << 52)
}
