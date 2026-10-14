// Rust-only (COLMAP has no test of ceres::Jet, and Ceres' jet_test.cc is not part of this
// port): the elementary-function rules of colmap_rust::solver::Jet that the camera model
// tests do not reach, because the iterative undistortion only runs +, -, * and / on Jets.
// Port of colmap-sharp's `ColmapSharp.Tests/Solver/JetTests.cs`, same cases and tolerances;
// the reference values come from `math::fns` (the crate's libm), as the Jet's do. Each rule
// is checked two ways:
// - against its closed-form derivative, written out here, bit for bit where the rule's
//   formula is the closed form's own grouping (Ceres 2.2.0's), and
// - against a central finite difference of the value part, (f(a + h) - f(a - h)) / 2h,
//   which checks the calculus independently of how the formula is grouped.
// Both derivative slots are checked, with a seed that differs per slot, so a rule that
// mixes up v0 and v1 fails.

use colmap_rust::math::fns;
use colmap_rust::solver::{fmax, fmin, Jet, Scalar};

type Jet2 = Jet<2>;

// Finite-difference step and tolerance: central differences have O(h^2) truncation and
// O(eps / h) rounding error, so h = 1e-5 leaves about 1e-10 relative; 1e-8 is loose enough
// for every function and point below and tight enough to catch a wrong rule.
const H: f64 = 1e-5;
const FINITE_DIFFERENCE_TOLERANCE: f64 = 1e-8;

// Seeds for the two derivative slots: d/dt of the input a + t*(V0_SEED, V1_SEED).
const V0_SEED: f64 = 1.0;
const V1_SEED: f64 = -0.5;

fn input(a: f64) -> Jet2 {
    Jet2::new(a, [V0_SEED, V1_SEED])
}

fn check_finite_difference(f: impl Fn(Jet2) -> Jet2, g: impl Fn(f64) -> f64, a: f64) {
    let result = f(input(a));
    let slope = (g(a + H) - g(a - H)) / (2.0 * H);
    let scale = 1.0_f64.max(slope.abs());
    assert!((result.v[0] - slope * V0_SEED).abs() <= FINITE_DIFFERENCE_TOLERANCE * scale);
    assert!((result.v[1] - slope * V1_SEED).abs() <= FINITE_DIFFERENCE_TOLERANCE * scale);
    assert_eq!(result.a, g(a));
}

#[test]
fn rust_only_sqrt_matches_closed_form_and_finite_difference() {
    for a in [0.25, 2.0, 9.5] {
        let r = input(a).sqrt();
        let two_a_inverse = 1.0 / (2.0 * fns::sqrt(a));
        assert_eq!(r.a, fns::sqrt(a));
        assert_eq!(r.v[0], V0_SEED * two_a_inverse);
        assert_eq!(r.v[1], V1_SEED * two_a_inverse);
        check_finite_difference(Scalar::sqrt, fns::sqrt, a);
    }
}

#[test]
fn rust_only_abs_matches_closed_form_and_finite_difference() {
    for a in [-1.3, 0.7] {
        let r = input(a).abs();
        let sign = a.signum();
        assert_eq!(r.a, a.abs());
        assert_eq!(r.v[0], sign * V0_SEED);
        assert_eq!(r.v[1], sign * V1_SEED);
        check_finite_difference(Scalar::abs, f64::abs, a);
    }
}

// Ceres 2.1+ (pycolmap 4.2.0 links 2.2.0) uses copysign(1, a), so -0.0 takes the negative
// branch: value +0.0, derivative -v. The older `a < 0 ? -f : f` would give +v here.
#[test]
fn rust_only_abs_negative_zero_uses_copysign() {
    let r = input(-0.0).abs();
    assert!(!r.a.is_sign_negative());
    assert_eq!(r.a, 0.0);
    assert_eq!(r.v[0], -V0_SEED);
    assert_eq!(r.v[1], -V1_SEED);

    let positive = input(0.0).abs();
    assert_eq!(positive.v[0], V0_SEED);
    assert_eq!(positive.v[1], V1_SEED);
}

#[test]
fn rust_only_sin_matches_closed_form_and_finite_difference() {
    for a in [-2.0, 0.3, 1.4] {
        let r = input(a).sin();
        assert_eq!(r.a, fns::sin(a));
        assert_eq!(r.v[0], fns::cos(a) * V0_SEED);
        assert_eq!(r.v[1], fns::cos(a) * V1_SEED);
        check_finite_difference(Scalar::sin, fns::sin, a);
    }
}

#[test]
fn rust_only_cos_matches_closed_form_and_finite_difference() {
    for a in [-2.0, 0.3, 1.4] {
        let r = input(a).cos();
        assert_eq!(r.a, fns::cos(a));
        assert_eq!(r.v[0], -fns::sin(a) * V0_SEED);
        assert_eq!(r.v[1], -fns::sin(a) * V1_SEED);
        check_finite_difference(Scalar::cos, fns::cos, a);
    }
}

#[test]
fn rust_only_tan_matches_closed_form_and_finite_difference() {
    for a in [-1.2, 0.3, 0.9] {
        let r = input(a).tan();
        let tan_a = fns::tan(a);
        let sec_squared = 1.0 + tan_a * tan_a;
        assert_eq!(r.a, tan_a);
        assert_eq!(r.v[0], sec_squared * V0_SEED);
        assert_eq!(r.v[1], sec_squared * V1_SEED);
        check_finite_difference(Scalar::tan, fns::tan, a);
    }
}

#[test]
fn rust_only_atan_matches_closed_form_and_finite_difference() {
    for a in [-3.0, 0.0, 0.8] {
        let r = input(a).atan();
        let derivative = 1.0 / (1.0 + a * a);
        assert_eq!(r.a, fns::atan(a));
        assert_eq!(r.v[0], derivative * V0_SEED);
        assert_eq!(r.v[1], derivative * V1_SEED);
        check_finite_difference(Scalar::atan, fns::atan, a);
    }
}

// atan2(y, x) with y and x depending on different variables: y = y0 + t0, x = x0 + t1, so
// d/dt0 = x / (x^2 + y^2) and d/dt1 = -y / (x^2 + y^2).
#[test]
fn rust_only_atan2_matches_closed_form_and_finite_difference() {
    for (y, x) in [(0.6, 1.1), (-0.4, -2.0), (1.5, -0.3)] {
        let r = Jet2::variable(y, 0).atan2(Jet2::variable(x, 1));
        let inverse_norm_squared = 1.0 / (x * x + y * y);
        assert_eq!(r.a, fns::atan2(y, x));
        assert_eq!(r.v[0], inverse_norm_squared * (-y * 0.0 + x * 1.0));
        assert_eq!(r.v[1], inverse_norm_squared * (-y * 1.0 + x * 0.0));

        let dy = (fns::atan2(y + H, x) - fns::atan2(y - H, x)) / (2.0 * H);
        let dx = (fns::atan2(y, x + H) - fns::atan2(y, x - H)) / (2.0 * H);
        assert!((r.v[0] - dy).abs() <= FINITE_DIFFERENCE_TOLERANCE);
        assert!((r.v[1] - dx).abs() <= FINITE_DIFFERENCE_TOLERANCE);
    }
}

// C++ `1.0 - alpha` with alpha a Jet: Ceres' operator-(T, Jet) gives (s - a, -v), with a true
// negation (so a +0.0 derivative becomes -0.0, unlike Jet(s) - alpha's 0 - v).
#[test]
fn rust_only_double_minus_jet_negates_derivative() {
    let r = 1.0 - input(0.3);
    assert_eq!(r.a, 1.0 - 0.3);
    assert_eq!(r.v[0], -V0_SEED);
    assert_eq!(r.v[1], -V1_SEED);
    assert_eq!(input(0.3).sub_from_f64(1.0).v, r.v);

    let zero_slope = 1.0 - Jet2::constant(0.3);
    assert!(zero_slope.v[0].is_sign_negative());
    check_finite_difference(|j| 1.0 - j, |a| 1.0 - a, 0.3);
}

// The rules for exp, log, the inverse trig functions and pow: value, closed form in Ceres'
// grouping, and finite difference.
#[test]
fn rust_only_exp_matches_closed_form_and_finite_difference() {
    for a in [-1.5, 0.4, 2.2] {
        let r = input(a).exp();
        assert_eq!(r.v[0], fns::exp(a) * V0_SEED);
        check_finite_difference(Scalar::exp, fns::exp, a);
    }
}

#[test]
fn rust_only_log_matches_closed_form_and_finite_difference() {
    for a in [0.3, 2.5] {
        let r = input(a).log();
        assert_eq!(r.v[1], V1_SEED * (1.0 / a));
        check_finite_difference(Scalar::log, fns::ln, a);
    }
}

#[test]
fn rust_only_asin_matches_closed_form_and_finite_difference() {
    for a in [-0.7, 0.2] {
        let r = input(a).asin();
        assert_eq!(r.v[0], 1.0 / fns::sqrt(1.0 - a * a) * V0_SEED);
        check_finite_difference(Scalar::asin, fns::asin, a);
    }
}

#[test]
fn rust_only_acos_matches_closed_form_and_finite_difference() {
    for a in [-0.7, 0.2] {
        let r = input(a).acos();
        assert_eq!(r.v[0], -1.0 / fns::sqrt(1.0 - a * a) * V0_SEED);
        check_finite_difference(Scalar::acos, fns::acos, a);
    }
}

#[test]
fn rust_only_pow_constant_exponent_matches_closed_form_and_finite_difference() {
    for (a, p) in [(0.6, 2.5), (3.0, -1.5)] {
        let r = input(a).pow_f64(p);
        assert_eq!(r.v[0], p * fns::pow(a, p - 1.0) * V0_SEED);
        check_finite_difference(|j| j.pow_f64(p), |x| fns::pow(x, p), a);
    }
}

// pow(f, g) with f = f0 + t0 and g = g0 + t1: d/dt0 = g f^(g-1), d/dt1 = f^g log f.
#[test]
fn rust_only_pow_jet_exponent_matches_finite_difference() {
    for (f, g) in [(1.7, 0.8), (0.4, 2.3)] {
        let r = Jet2::variable(f, 0).pow(Jet2::variable(g, 1));
        let df = (fns::pow(f + H, g) - fns::pow(f - H, g)) / (2.0 * H);
        let dg = (fns::pow(f, g + H) - fns::pow(f, g - H)) / (2.0 * H);
        assert_eq!(r.a, fns::pow(f, g));
        assert!((r.v[0] - df).abs() <= FINITE_DIFFERENCE_TOLERANCE);
        assert!((r.v[1] - dg).abs() <= FINITE_DIFFERENCE_TOLERANCE);
    }
}

// Ceres' special cases: 0^g for g > 1 is a constant zero; a negative base with an integral
// exponent differentiates the base and is NaN in the exponent's slots.
#[test]
fn rust_only_pow_jet_exponent_special_cases() {
    let zero = Jet2::variable(0.0, 0).pow(Jet2::variable(2.0, 1));
    assert_eq!(zero.a, 0.0);
    assert_eq!(zero.v[0], 0.0);
    assert_eq!(zero.v[1], 0.0);

    let negative = Jet2::variable(-2.0, 0).pow(Jet2::variable(3.0, 1));
    assert_eq!(negative.a, -8.0);
    assert_eq!(negative.v[0], 12.0);
    assert!(negative.v[1].is_nan());
}

#[test]
fn rust_only_hypot_matches_closed_form() {
    let (x, y, z) = (0.3, -1.2, 0.7);
    let two = Jet2::variable(x, 0).hypot(Jet2::variable(y, 1));
    assert_eq!(two.a, fns::hypot(x, y));
    assert!((two.v[0] - x / fns::hypot(x, y)).abs() <= 1e-15);
    assert!((two.v[1] - y / fns::hypot(x, y)).abs() <= 1e-15);

    let three = Jet2::hypot3(
        Jet2::variable(x, 0),
        Jet2::constant(y),
        Jet2::variable(z, 1),
    );
    let norm = fns::sqrt(x * x + y * y + z * z);
    assert_eq!(three.a, norm);
    assert!((three.v[0] - x / norm).abs() <= 1e-15);
    assert!((three.v[1] - z / norm).abs() <= 1e-15);
}

// libc++'s three-argument hypot rescales out-of-range arguments instead of overflowing.
#[test]
fn rust_only_hypot3_does_not_overflow_or_underflow() {
    // Multiples of a power of two, so 3-4-5 is exact and only over- or underflow could break it.
    let big = 2.0_f64.powi(1000);
    let tiny = 2.0_f64.powi(-1000);
    assert_eq!(
        <f64 as Scalar>::hypot3(3.0 * big, 4.0 * big, 0.0),
        5.0 * big
    );
    assert_eq!(
        <f64 as Scalar>::hypot3(3.0 * tiny, 0.0, 4.0 * tiny),
        5.0 * tiny
    );
    assert_eq!(<f64 as Scalar>::hypot3(1.0, 2.0, 2.0), 3.0);
}

// ceres::fmin/fmax: NaN is missing data, and a value tie averages the two Jets.
#[test]
fn rust_only_min_max_handle_nan_and_ties() {
    let small = Jet2::variable(1.0, 0);
    let large = Jet2::variable(2.0, 1);
    assert_eq!(small.fmin(large).v[0], 1.0);
    assert_eq!(small.fmax(large).v[1], 1.0);

    let nan = Jet2::constant(f64::NAN);
    assert_eq!(nan.fmin(large).a, 2.0);
    assert_eq!(small.fmax(nan).a, 1.0);

    let tie = Jet2::variable(1.0, 0).fmax(Jet2::variable(1.0, 1));
    assert_eq!(tie.a, 1.0);
    assert_eq!(tie.v[0], 0.5);
    assert_eq!(tie.v[1], 0.5);
    assert_eq!(fmax(f64::NAN, 3.0), 3.0);
    assert_eq!(fmin(3.0, f64::NAN), 3.0);
}

// The mixed operators have Ceres' own rules, which are not Jet(s) op x: s / g is s / g.a
// with slope -s / g.a^2 (not s * (1 / g.a)), f / s goes through 1 / s.
#[test]
fn rust_only_mixed_operators_follow_ceres_rules() {
    let s = 0.7;
    let g = input(1.3);
    let quotient = s / g;
    assert_eq!(quotient.a, s / 1.3);
    assert_eq!(quotient.v[0], V0_SEED * (-s / (1.3 * 1.3)));
    assert_eq!(g.div_from_f64(s).v, quotient.v);

    let by_scalar = g / s;
    let s_inverse = 1.0 / s;
    assert_eq!(by_scalar.a, 1.3 * s_inverse);
    assert_eq!(by_scalar.v[1], V1_SEED * s_inverse);

    let product = s * g;
    assert_eq!(product.v[1], V1_SEED * s);
    assert_eq!((g + s).v[0], V0_SEED);
    assert_eq!((g - s).a, 1.3 - s);
    assert_eq!(input(1.7).floor().a, 1.0);
    assert_eq!(input(1.7).floor().v[0], 0.0);
}

// Variable(value, k) seeds exactly slot k, for any width.
#[test]
fn rust_only_variable_seeds_one_slot() {
    let jet = Jet::<15>::variable(2.5, 11);
    assert_eq!(jet.a, 2.5);
    for (i, &d) in jet.v.iter().enumerate() {
        assert_eq!(d, if i == 11 { 1.0 } else { 0.0 });
    }
}
