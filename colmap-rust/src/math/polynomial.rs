//! Port of COLMAP's `src/colmap/math/polynomial.h` / `polynomial.cc`: evaluation and
//! real/complex roots of polynomials given by their coefficients in decreasing order of
//! degree (`coeffs[0] x^N + ... + coeffs[N]`): closed forms for degree 1, 2 and 3
//! (depressed-cubic Cardano / trigonometric form plus one Newton step), Durand-Kerner
//! iteration, and the companion-matrix eigenvalues through [`crate::linalg::EigenSolver`].
//! Mirrors colmap-sharp's `Mathematics/Polynomial.cs`. Tests:
//! `colmap-rust/tests/math/polynomial.rs` (`polynomial_test.cc` 1:1).
//!
//! Tiers: [`evaluate_polynomial`] at a real point and the linear/quadratic roots are plain
//! arithmetic in COLMAP's order, Tier A. The cubic goes through `cbrt`/`acos`/`cos`
//! (`math::fns`, `docs/CPP_DIVERGENCES.md` entry 1), Durand-Kerner through complex division
//! (Smith's algorithm, not libc++'s `logb`/`scalbn` scaling: entry 50), and the companion
//! matrix through our `EigenSolver` (Francis QR, not Eigen's: entries 32 and 51), so those are
//! Tier B; `polynomial_test.cc` compares them with tolerances.
//!
//! C++'s nullable `real`/`imag` output pointers become one returned [`PolynomialRoots`]
//! (`None` where C++ returns `false`); see entry 52. `THROW_CHECK`s return `Err`.

use crate::linalg::{Complex, ComputationInfo, EigenSolver, MatrixXd, Vector3d, VectorXd};
use crate::math::fns;
use crate::{check_eq, check_ge};

/// The roots of a polynomial: `real[i] + imag[i] i`, both of the same length.
#[derive(Clone, Debug, PartialEq)]
pub struct PolynomialRoots {
    /// Real parts.
    pub real: VectorXd,
    /// Imaginary parts.
    pub imag: VectorXd,
}

impl PolynomialRoots {
    fn single_zero() -> Self {
        Self {
            real: VectorXd::zeros(1),
            imag: VectorXd::zeros(1),
        }
    }
}

/// A point type [`evaluate_polynomial`] can evaluate at: `f64` or [`Complex`]
/// (`EvaluatePolynomial<T>` with `T = double` or `std::complex<double>`).
pub trait PolynomialPoint: Copy {
    /// `0` of this type.
    fn zero() -> Self;
    /// `value * x + coeff`, one Horner step. For a complex point the real coefficient is
    /// added to the real part only, as `std::complex<double> + double` does.
    fn horner_step(value: Self, x: Self, coeff: f64) -> Self;
}

impl PolynomialPoint for f64 {
    fn zero() -> Self {
        0.0
    }

    fn horner_step(value: Self, x: Self, coeff: f64) -> Self {
        value * x + coeff
    }
}

impl PolynomialPoint for Complex {
    fn zero() -> Self {
        Complex::ZERO
    }

    fn horner_step(value: Self, x: Self, coeff: f64) -> Self {
        let product = value * x;
        Complex::new(product.re + coeff, product.im)
    }
}

/// Evaluates the polynomial at `x` with the Horner scheme. Port of
/// `colmap::EvaluatePolynomial`.
pub fn evaluate_polynomial<T: PolynomialPoint>(coeffs: &VectorXd, x: T) -> T {
    let mut value = T::zero();
    for &coeff in coeffs.as_slice() {
        value = T::horner_step(value, x, coeff);
    }
    value
}

/// The root of `coeffs[0] x + coeffs[1] = 0`; `None` when `coeffs[0] == 0`.
/// Port of `colmap::FindLinearPolynomialRoots`.
pub fn find_linear_polynomial_roots(coeffs: &VectorXd) -> crate::Result<Option<PolynomialRoots>> {
    check_eq!(coeffs.len(), 2);

    if coeffs[0] == 0.0 {
        return Ok(None);
    }

    Ok(Some(PolynomialRoots {
        real: VectorXd::from_slice(&[-coeffs[1] / coeffs[0]]),
        imag: VectorXd::zeros(1),
    }))
}

/// The roots of `coeffs[0] x^2 + coeffs[1] x + coeffs[2] = 0` by the cancellation-free
/// quadratic formula; a zero leading coefficient falls back to the linear case.
/// Port of `colmap::FindQuadraticPolynomialRoots`.
pub fn find_quadratic_polynomial_roots(
    coeffs: &VectorXd,
) -> crate::Result<Option<PolynomialRoots>> {
    check_eq!(coeffs.len(), 3);

    let a = coeffs[0];
    if a == 0.0 {
        return find_linear_polynomial_roots(&coeffs.tail(2));
    }

    let b = coeffs[1];
    let c = coeffs[2];
    if b == 0.0 && c == 0.0 {
        return Ok(Some(PolynomialRoots::single_zero()));
    }

    let d = b * b - 4.0 * a * c;

    let mut real = VectorXd::zeros(2);
    let mut imag = VectorXd::zeros(2);
    if d >= 0.0 {
        let sqrt_d = fns::sqrt(d);
        if b >= 0.0 {
            real[0] = (-b - sqrt_d) / (2.0 * a);
            real[1] = (2.0 * c) / (-b - sqrt_d);
        } else {
            real[0] = (2.0 * c) / (-b + sqrt_d);
            real[1] = (-b + sqrt_d) / (2.0 * a);
        }
    } else {
        real = VectorXd::constant(2, -b / (2.0 * a));
        imag[0] = fns::sqrt(-d) / (2.0 * a);
        imag[1] = -imag[0];
    }

    Ok(Some(PolynomialRoots { real, imag }))
}

/// The real roots of the monic cubic `x^3 + c2 x^2 + c1 x + c0 = 0`, each refined by one
/// Newton step. Returns the number of roots (1 or 3) and the roots in the leading entries;
/// the unused entries are zero (C++ leaves the caller's values there, entry 52).
/// Port of `colmap::FindCubicPolynomialRoots`.
pub fn find_cubic_polynomial_roots(c2: f64, c1: f64, c0: f64) -> (usize, Vector3d) {
    // COLMAP's literals, kept digit for digit.
    #[allow(clippy::excessive_precision)]
    const K2_PI_OVER_3: f64 = 2.09439510239319526263557236234192;
    #[allow(clippy::excessive_precision)]
    const K4_PI_OVER_3: f64 = 4.18879020478639052527114472468384;
    let c2_over_3 = c2 / 3.0;
    let a = c1 - c2 * c2_over_3;
    let mut b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    let mut c = b * b / 4.0 + a * a * a / 27.0;
    let mut roots = [0.0; 3];
    let num_roots;
    if c > 0.0 {
        c = fns::sqrt(c);
        b *= -0.5;
        roots[0] = fns::cbrt(b + c) + fns::cbrt(b - c) - c2_over_3;
        num_roots = 1;
    } else {
        c = 3.0 * b / (2.0 * a) * fns::sqrt(-3.0 / a);
        let d = 2.0 * fns::sqrt(-a / 3.0);
        let acos_over_3 = fns::acos(c) / 3.0;
        roots[0] = d * fns::cos(acos_over_3) - c2_over_3;
        roots[1] = d * fns::cos(acos_over_3 - K2_PI_OVER_3) - c2_over_3;
        roots[2] = d * fns::cos(acos_over_3 - K4_PI_OVER_3) - c2_over_3;
        num_roots = 3;
    }

    // Single Newton iteration.
    for root in roots.iter_mut().take(num_roots) {
        let x = *root;
        let x2 = x * x;
        let x3 = x * x2;
        let dx = -(x3 + c2 * x2 + c1 * x + c0) / (3.0 * x2 + 2.0 * c2 * x + c1);
        *root += dx;
    }

    (num_roots, Vector3d::new(roots[0], roots[1], roots[2]))
}

/// All complex roots by Durand-Kerner (Weierstrass) iteration, based on
/// <https://en.wikipedia.org/wiki/Durand%E2%80%93Kerner_method>. Comparatively fast but
/// often unstable/inaccurate. Degrees 1 and 2 use the closed forms; `None` for a constant.
/// Port of `colmap::FindPolynomialRootsDurandKerner`.
pub fn find_polynomial_roots_durand_kerner(
    coeffs_all: &VectorXd,
) -> crate::Result<Option<PolynomialRoots>> {
    check_ge!(coeffs_all.len(), 2);

    let coeffs = remove_leading_zeros(coeffs_all);

    // `coeffs.size() - 1` as a signed int in C++; an all-zero input gives -1.
    let degree = coeffs.len() as isize - 1;

    if degree <= 0 {
        return Ok(None);
    } else if degree == 1 {
        return find_linear_polynomial_roots(&coeffs);
    } else if degree == 2 {
        return find_quadratic_polynomial_roots(&coeffs);
    }
    let degree = degree as usize;

    // Initialize roots.
    let mut roots = vec![Complex::ZERO; degree];
    roots[degree - 1] = Complex::new(1.0, 0.0);
    for i in (0..degree - 1).rev() {
        roots[i] = roots[i + 1] * Complex::new(1.0, 1.0);
    }

    // Iterative solver.
    const K_MAX_NUM_ITERATIONS: usize = 100;
    const K_MAX_ROOT_CHANGE: f64 = 1e-10;
    for _ in 0..K_MAX_NUM_ITERATIONS {
        let mut max_root_change: f64 = 0.0;
        for i in 0..degree {
            let root_i = roots[i];
            let mut numerator = Complex::from_real(coeffs[0]);
            let mut denominator = Complex::from_real(coeffs[0]);
            for j in 0..degree {
                numerator = Complex::horner_step(numerator, root_i, coeffs[j + 1]);
                if i != j {
                    denominator = denominator * (root_i - roots[j]);
                }
            }
            let root_i_change = numerator / denominator;
            roots[i] = root_i - root_i_change;
            // std::max(a, b) returns a unless a < b.
            max_root_change = std_max(max_root_change, root_i_change.re.abs());
            max_root_change = std_max(max_root_change, root_i_change.im.abs());
        }

        // Break, if roots do not change anymore.
        if max_root_change < K_MAX_ROOT_CHANGE {
            break;
        }
    }

    Ok(Some(PolynomialRoots {
        real: VectorXd::from_vec(roots.iter().map(|r| r.re).collect()),
        imag: VectorXd::from_vec(roots.iter().map(|r| r.im).collect()),
    }))
}

/// All complex roots as the eigenvalues of the companion matrix, based on R. A. Horn &
/// C. R. Johnson, Matrix Analysis (1999), pp. 146-7, and NumPy's `roots`. Slower than
/// Durand-Kerner but more stable/accurate. Degrees 1 and 2 use the closed forms, trailing
/// zero coefficients add a root at zero (listed last); `None` for a constant or when the
/// eigen solver fails. Port of `colmap::FindPolynomialRootsCompanionMatrix`.
pub fn find_polynomial_roots_companion_matrix(
    coeffs_all: &VectorXd,
) -> crate::Result<Option<PolynomialRoots>> {
    check_ge!(coeffs_all.len(), 2);

    let mut coeffs = remove_leading_zeros(coeffs_all);

    let degree = coeffs.len() as isize - 1;

    if degree <= 0 {
        return Ok(None);
    } else if degree == 1 {
        return find_linear_polynomial_roots(&coeffs);
    } else if degree == 2 {
        return find_quadratic_polynomial_roots(&coeffs);
    }
    let degree = degree as usize;

    // Remove the coefficients where zero is a solution.
    coeffs = remove_trailing_zeros(&coeffs);

    // Check if only zero is a solution.
    if coeffs.len() == 1 {
        return Ok(Some(PolynomialRoots::single_zero()));
    }

    // Fill the companion matrix.
    let size = coeffs.len() - 1;
    let mut companion = MatrixXd::zeros(size, size);
    for i in 1..size {
        companion[(i, i - 1)] = 1.0;
    }
    for j in 0..size {
        companion[(0, j)] = -coeffs[j + 1] / coeffs[0];
    }

    // Solve for the roots of the polynomial.
    let solver = EigenSolver::new(&companion, false);
    if solver.info() != ComputationInfo::Success {
        return Ok(None);
    }
    let eigenvalues = solver.eigenvalues();

    // If there are trailing zeros, we must add zero as a solution (the zero-filled last
    // entry).
    let effective_degree = if size < degree { coeffs.len() } else { size };

    let mut real = VectorXd::zeros(effective_degree);
    let mut imag = VectorXd::zeros(effective_degree);
    for (i, eigenvalue) in eigenvalues.iter().enumerate() {
        real[i] = eigenvalue.re;
        imag[i] = eigenvalue.im;
    }

    Ok(Some(PolynomialRoots { real, imag }))
}

/// `std::max(a, b)`: `b` only when `a < b`.
fn std_max(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// Removes leading zero coefficients.
fn remove_leading_zeros(coeffs: &VectorXd) -> VectorXd {
    let num_zeros = coeffs.as_slice().iter().take_while(|&&c| c == 0.0).count();
    coeffs.tail(coeffs.len() - num_zeros)
}

/// Removes trailing zero coefficients.
fn remove_trailing_zeros(coeffs: &VectorXd) -> VectorXd {
    let num_zeros = coeffs
        .as_slice()
        .iter()
        .rev()
        .take_while(|&&c| c == 0.0)
        .count();
    coeffs.head(coeffs.len() - num_zeros)
}
