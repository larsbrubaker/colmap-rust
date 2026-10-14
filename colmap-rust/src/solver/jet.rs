//! [`Jet`]: `ceres::Jet<double, N>`, a dual number `a + sum_i v_i e_i` with `e_i e_j = 0`, for
//! forward-mode automatic differentiation in `N` variables. Port of colmap-sharp's
//! `ColmapSharp/Solver/Jet.cs` (from Ceres Solver 2.2.0 `include/ceres/jet.h`,
//! BSD-3-Clause, see THIRD_PARTY_NOTICES.md). It implements [`Scalar`], so the camera models
//! (`sensor/models`, whose iterative undistortion runs on `Jet<2>`), the rotation conversions
//! ([`super::rotation`]) and every cost functor evaluated by [`super::AutoDiffCostFunction`]
//! differentiate through it.
//!
//! The rules follow Ceres 2.2.0, the version the pycolmap 4.2.0 wheel links, compiled as
//! C++17 (COLMAP's `CMAKE_CXX_STANDARD`), which picks fmax/fmin's `(x + y) * 0.5` tie rule
//! over C++20's midpoint. Each rule keeps Ceres' own grouping, because Jets feed Tier A code
//! (the iterative undistortion's Newton steps): the quotient is `f.a * (1 / g.a)`, sqrt
//! divides by `2 sqrt(a)` through a reciprocal, and so on. Every derivative component is
//! computed from the value parts and that component alone, with no FMA (CLAUDE.md).
//!
//! `N` is a const generic, so `Jet<N>` is `Copy`, lives inline (`[f64; N]`) and never
//! allocates; COLMAP's widths run from 2 (undistortion) to 33 (the rig reprojection functor
//! with the 16-parameter camera). Rules, `f = (a, v)`, `g = (b, w)`, `s` a plain double:
//! - `f + g = (a + b, v + w)`, `f - g = (a - b, v - w)`, `-f = (-a, -v)`
//! - `f * g = (a b, a w + v b)`; `f / g`: `ib = 1 / b`, `q = a ib`, `(q, (v - q w) ib)`
//! - `f + s = s + f = (a + s, v)`, `f - s = (a - s, v)`, `s - f = (s - a, -v)`
//! - `f * s = s * f = (a s, v s)`, `f / s = (a is, v is)` with `is = 1 / s`,
//!   `s / g = (s / b, w (-s / (b b)))`
//! - the elementary functions, documented on each `Scalar` method below
//! - comparisons and `==` compare `a` only.

// Indexed loops mirror Ceres' lane-by-lane expressions over several arrays at once.
#![allow(clippy::needless_range_loop)]

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::math::fns;

use super::scalar::Scalar;

/// Port of `ceres::Jet<double, N>`: a value `a` and its derivative `v` with respect to `N`
/// variables.
#[derive(Clone, Copy, Debug)]
pub struct Jet<const N: usize> {
    /// The value part, Ceres' `a`.
    pub a: f64,
    /// The derivative (infinitesimal) part, Ceres' `v`.
    pub v: [f64; N],
}

impl<const N: usize> Default for Jet<N> {
    /// Ceres' `Jet()`: zero value and zero derivative.
    fn default() -> Self {
        Self::constant(0.0)
    }
}

impl<const N: usize> Jet<N> {
    /// A value and explicit derivative components.
    pub const fn new(a: f64, v: [f64; N]) -> Self {
        Self { a, v }
    }

    /// `Jet(value)`: a constant.
    pub const fn constant(a: f64) -> Self {
        Self { a, v: [0.0; N] }
    }

    /// `Jet(value, k)`: the `k`-th independent variable (`v = e_k`). Panics if `k >= N`,
    /// as an out-of-range Eigen index is an internal invariant violation.
    pub fn variable(a: f64, k: usize) -> Self {
        let mut v = [0.0; N];
        v[k] = 1.0;
        Self { a, v }
    }

    /// `(a, s * v)`, Ceres' `v * s` lane by lane: the shape of most unary rules.
    #[inline]
    fn scaled(a: f64, s: f64, v: &[f64; N]) -> Self {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = s * v[i];
        }
        Self { a, v: out }
    }

    /// `(a, -v)`: a true negation of every slot, as Eigen's unary minus.
    #[inline]
    fn negated(a: f64, v: &[f64; N]) -> Self {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = -v[i];
        }
        Self { a, v: out }
    }

    /// `(a, s0 * v0 + s1 * v1)`.
    #[inline]
    fn combined(a: f64, s0: f64, v0: &[f64; N], s1: f64, v1: &[f64; N]) -> Self {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = s0 * v0[i] + s1 * v1[i];
        }
        Self { a, v: out }
    }
}

impl<const N: usize> PartialEq for Jet<N> {
    /// Ceres' `==` compares the value parts.
    fn eq(&self, other: &Self) -> bool {
        self.a == other.a
    }
}

impl<const N: usize> PartialOrd for Jet<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.a.partial_cmp(&other.a)
    }
}

impl<const N: usize> Add for Jet<N> {
    type Output = Self;
    fn add(self, g: Self) -> Self {
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = self.v[i] + g.v[i];
        }
        Self { a: self.a + g.a, v }
    }
}

impl<const N: usize> Sub for Jet<N> {
    type Output = Self;
    fn sub(self, g: Self) -> Self {
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = self.v[i] - g.v[i];
        }
        Self { a: self.a - g.a, v }
    }
}

impl<const N: usize> Neg for Jet<N> {
    type Output = Self;
    fn neg(self) -> Self {
        Self::negated(-self.a, &self.v)
    }
}

impl<const N: usize> Mul for Jet<N> {
    type Output = Self;
    /// `(f.a g.a, f.a g.v + f.v g.a)`.
    fn mul(self, g: Self) -> Self {
        Self::combined(self.a * g.a, self.a, &g.v, g.a, &self.v)
    }
}

impl<const N: usize> Div for Jet<N> {
    type Output = Self;
    /// Ceres uses `(a + u) / (b + v) = (a + u)(b - v) / b^2`, which holds because `v v = 0`,
    /// and evaluates it through the reciprocal of `g.a`.
    fn div(self, g: Self) -> Self {
        let g_a_inverse = 1.0 / g.a;
        let f_a_by_g_a = self.a * g_a_inverse;
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = (self.v[i] - f_a_by_g_a * g.v[i]) * g_a_inverse;
        }
        Self { a: f_a_by_g_a, v }
    }
}

impl<const N: usize> Add<f64> for Jet<N> {
    type Output = Self;
    fn add(self, s: f64) -> Self {
        Self::new(self.a + s, self.v)
    }
}

impl<const N: usize> Add<Jet<N>> for f64 {
    type Output = Jet<N>;
    fn add(self, f: Jet<N>) -> Jet<N> {
        Jet::new(f.a + self, f.v)
    }
}

impl<const N: usize> Sub<f64> for Jet<N> {
    type Output = Self;
    fn sub(self, s: f64) -> Self {
        Self::new(self.a - s, self.v)
    }
}

impl<const N: usize> Sub<Jet<N>> for f64 {
    type Output = Jet<N>;
    /// C++ `1.0 - alpha`: Ceres gives `(s - a, -v)`, a true negation, so a `+0.0` derivative
    /// becomes `-0.0` (`Jet(s) - alpha` would compute `0 - v = +0.0`).
    fn sub(self, f: Jet<N>) -> Jet<N> {
        Jet::negated(self - f.a, &f.v)
    }
}

impl<const N: usize> Mul<f64> for Jet<N> {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::scaled(self.a * s, s, &self.v)
    }
}

impl<const N: usize> Mul<Jet<N>> for f64 {
    type Output = Jet<N>;
    fn mul(self, f: Jet<N>) -> Jet<N> {
        Jet::scaled(f.a * self, self, &f.v)
    }
}

impl<const N: usize> Div<f64> for Jet<N> {
    type Output = Self;
    /// Through the reciprocal: `(a / s, v / s) = (a s^-1, v s^-1)`.
    fn div(self, s: f64) -> Self {
        let s_inverse = 1.0 / s;
        Self::scaled(self.a * s_inverse, s_inverse, &self.v)
    }
}

impl<const N: usize> Div<Jet<N>> for f64 {
    type Output = Jet<N>;
    /// `(s / g.a, g.v * (-s / g.a^2))`.
    fn div(self, g: Jet<N>) -> Jet<N> {
        g.div_from_f64(self)
    }
}

impl<const N: usize> Scalar for Jet<N> {
    fn from_f64(value: f64) -> Self {
        Self::constant(value)
    }

    fn value(self) -> f64 {
        self.a
    }

    fn sub_from_f64(self, s: f64) -> Self {
        s - self
    }

    fn div_from_f64(self, s: f64) -> Self {
        let minus_s_g_a_inverse2 = -s / (self.a * self.a);
        Self::scaled(s / self.a, minus_s_g_a_inverse2, &self.v)
    }

    /// `sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a))`.
    fn sqrt(self) -> Self {
        let tmp = fns::sqrt(self.a);
        let two_a_inverse = 1.0 / (2.0 * tmp);
        Self::scaled(tmp, two_a_inverse, &self.v)
    }

    /// `abs(a + h) ~= abs(a) + sgn(a) h`, with Ceres 2.1+'s `sgn = copysign(1, a)`: at
    /// `a = -0.0` the value is `+0.0` and the derivative is `-h`.
    fn abs(self) -> Self {
        Self::scaled(self.a.abs(), 1.0_f64.copysign(self.a), &self.v)
    }

    /// `sin(a + h) ~= sin(a) + cos(a) h`.
    fn sin(self) -> Self {
        Self::scaled(fns::sin(self.a), fns::cos(self.a), &self.v)
    }

    /// `cos(a + h) ~= cos(a) - sin(a) h`.
    fn cos(self) -> Self {
        Self::scaled(fns::cos(self.a), -fns::sin(self.a), &self.v)
    }

    /// `tan(a + h) ~= tan(a) + (1 + tan(a)^2) h`.
    fn tan(self) -> Self {
        let tan_a = fns::tan(self.a);
        let tmp = 1.0 + tan_a * tan_a;
        Self::scaled(tan_a, tmp, &self.v)
    }

    /// `asin(a + h) ~= asin(a) + 1 / sqrt(1 - a^2) h`.
    fn asin(self) -> Self {
        let tmp = 1.0 / fns::sqrt(1.0 - self.a * self.a);
        Self::scaled(fns::asin(self.a), tmp, &self.v)
    }

    /// `acos(a + h) ~= acos(a) - 1 / sqrt(1 - a^2) h`.
    fn acos(self) -> Self {
        let tmp = -1.0 / fns::sqrt(1.0 - self.a * self.a);
        Self::scaled(fns::acos(self.a), tmp, &self.v)
    }

    /// `atan(a + h) ~= atan(a) + 1 / (1 + a^2) h`.
    fn atan(self) -> Self {
        let tmp = 1.0 / (1.0 + self.a * self.a);
        Self::scaled(fns::atan(self.a), tmp, &self.v)
    }

    /// `atan2(g + dg, f + df) ~= atan2(g, f) + (f dg - g df) / (f^2 + g^2)`, with `self` as
    /// `g` (y) and `x` as `f`; Ceres evaluates `tmp * (-g.a * f.v + f.a * g.v)`.
    fn atan2(self, x: Self) -> Self {
        let g = self;
        let f = x;
        let tmp = 1.0 / (f.a * f.a + g.a * g.a);
        let minus_g_a = -g.a;
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = tmp * (minus_g_a * f.v[i] + f.a * g.v[i]);
        }
        Self {
            a: fns::atan2(g.a, f.a),
            v,
        }
    }

    /// `exp(a + h) ~= exp(a) + exp(a) h`.
    fn exp(self) -> Self {
        let tmp = fns::exp(self.a);
        Self::scaled(tmp, tmp, &self.v)
    }

    /// `log(a + h) ~= log(a) + h / a`, through the reciprocal of `a`.
    fn log(self) -> Self {
        let a_inverse = 1.0 / self.a;
        Self::scaled(fns::ln(self.a), a_inverse, &self.v)
    }

    /// `(a + da)^p ~= a^p + p a^(p - 1) da`, for a constant exponent.
    fn pow_f64(self, g: f64) -> Self {
        let tmp = g * fns::pow(self.a, g - 1.0);
        Self::scaled(fns::pow(self.a, g), tmp, &self.v)
    }

    /// pow with both base and exponent differentiable, with Ceres' special cases: `f == 0`
    /// and `g > 1` gives zero; `f == 0` and `g == 1` gives `f`; `f < 0` with an integral `g`
    /// differentiates the base only and makes every slot where `g` has a derivative NaN
    /// (moving `g` leaves the reals); everything else uses
    /// `f^g + g f^(g - 1) df + f^g log(f) dg` and lets log produce the non-finite cases.
    fn pow(self, g: Self) -> Self {
        let f = self;
        if f.a == 0.0 && g.a >= 1.0 {
            return if g.a > 1.0 { Self::constant(0.0) } else { f };
        }
        if f.a < 0.0 && g.a == g.a.floor() {
            let tmp = g.a * fns::pow(f.a, g.a - 1.0);
            let mut result = Self::scaled(fns::pow(f.a, g.a), tmp, &f.v);
            for i in 0..N {
                if g.v[i] != 0.0 {
                    result.v[i] = f64::NAN;
                }
            }
            return result;
        }
        let tmp1 = fns::pow(f.a, g.a);
        let tmp2 = g.a * fns::pow(f.a, g.a - 1.0);
        let tmp3 = tmp1 * fns::ln(f.a);
        Self::combined(tmp1, tmp2, &f.v, tmp3, &g.v)
    }

    /// floor: piecewise constant, so the derivative is zero.
    fn floor(self) -> Self {
        Self::constant(self.a.floor())
    }

    /// `hypot(x, y) = sqrt(x^2 + y^2)` without intermediate overflow; derivative
    /// `x / hypot dx + y / hypot dy`.
    fn hypot(self, y: Self) -> Self {
        let tmp = fns::hypot(self.a, y.a);
        Self::combined(tmp, self.a / tmp, &self.v, y.a / tmp, &y.v)
    }

    /// `hypot(x, y, z)`; derivative `x / hypot dx + y / hypot dy + z / hypot dz`, summed
    /// left to right as Eigen's expression evaluates it.
    fn hypot3(x: Self, y: Self, z: Self) -> Self {
        let tmp = fns::hypot3(x.a, y.a, z.a);
        let (sx, sy, sz) = (x.a / tmp, y.a / tmp, z.a / tmp);
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = sx * x.v[i] + sy * y.v[i] + sz * z.v[i];
        }
        Self { a: tmp, v }
    }

    /// `ceres::fmin`: the smaller by value, the other one when either is NaN, and on a value
    /// tie the average `(x + y) * 0.5` of the two Jets, so the result keeps a derivative and
    /// does not depend on argument order.
    fn fmin(self, y: Self) -> Self {
        let x = self;
        if x.a.is_nan() || y.a.is_nan() || x.a < y.a || x.a > y.a {
            return if x.a.is_nan() || x.a > y.a { y } else { x };
        }
        (x + y) * 0.5
    }

    /// `ceres::fmax`, with the same NaN and tie rules as `fmin`.
    fn fmax(self, y: Self) -> Self {
        let x = self;
        if x.a.is_nan() || y.a.is_nan() || x.a < y.a || x.a > y.a {
            return if x.a.is_nan() || x.a < y.a { y } else { x };
        }
        (x + y) * 0.5
    }

    fn is_finite(self) -> bool {
        self.a.is_finite()
    }

    fn is_nan(self) -> bool {
        self.a.is_nan()
    }
}
