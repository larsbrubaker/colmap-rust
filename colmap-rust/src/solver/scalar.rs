//! [`Scalar`]: the numeric abstraction COLMAP's `template <typename T>` code is written
//! against, so one body serves plain `double` evaluation and `ceres::Jet` automatic
//! differentiation. The camera models (`sensor/models.h` instantiates `ImgFromCam` and
//! `Distortion` with both), the rotation conversions ([`super::rotation`]) and every cost
//! functor evaluated by [`super::AutoDiffCostFunction`] or
//! [`super::TinySolverAutoDiffFunction`] are generic over it. Port of colmap-sharp's
//! `IScalar<T>` / `Real` / `ScalarMath` (`ColmapSharp/Solver/Scalar.cs`).
//!
//! Implemented for `f64` here (every transcendental through [`crate::math::fns`], so the
//! double path rounds like the rest of the crate) and for [`super::Jet`]. The trait holds what
//! `ceres::Jet` offers COLMAP's templates: the four binary operators on two scalars,
//! negation, the mixed operators with a plain `f64` on the right (`t * 2.0`, `t - s`), the
//! two non-commutative mixed operators with the `f64` on the left as methods
//! ([`Scalar::sub_from_f64`] for C++ `s - t`, [`Scalar::div_from_f64`] for `s / t`),
//! comparisons on the value part, `T(double)` construction ([`Scalar::from_f64`]), and the
//! elementary functions COLMAP's templates call.
//!
//! The mixed operators are not the same as `T::from_f64(s) op x` for a Jet: Ceres gives them
//! their own rules (`s / g` is `s / g.a` with derivative `-s / g.a^2`, `1.0 - alpha` negates
//! every derivative slot where `Jet(1) - alpha` computes `0 - v`), and those reach the last
//! bit, so they are separate operations here. `s + t` and `s * t` equal `t + s` and `t * s`
//! bit for bit in both IEEE and Ceres, so generic code writes them with the scalar on the
//! right; the concrete [`super::Jet`] also implements the `f64`-on-the-left operators.
//!
//! Method names follow Ceres (`log`, not `ln`): on a concrete `f64`, `x.log()` does not
//! compile (std's `f64::log` takes a base), so a call can never silently bypass
//! [`crate::math::fns`]. Inside generic code every call resolves to the trait.

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::math::fns;

/// A scalar type COLMAP's templated numeric code runs on: `f64` for plain evaluation,
/// [`super::Jet`] for forward-mode automatic differentiation. Comparisons (`PartialOrd`)
/// compare the value part, like Ceres' Jet.
pub trait Scalar:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// C++ `T(value)`: a constant (a Jet with zero derivative).
    fn from_f64(value: f64) -> Self;

    /// The value part: the double itself, or a Jet's `a`.
    fn value(self) -> f64;

    /// C++ `s - self` with `s` a plain double (Ceres' mixed operator: `(s - a, -v)`).
    fn sub_from_f64(self, s: f64) -> Self;

    /// C++ `s / self` with `s` a plain double (Ceres: `(s / a, v * (-s / (a * a)))`).
    fn div_from_f64(self, s: f64) -> Self;

    /// `ceres::sqrt`.
    fn sqrt(self) -> Self;

    /// `ceres::abs`.
    fn abs(self) -> Self;

    /// `ceres::sin`.
    fn sin(self) -> Self;

    /// `ceres::cos`.
    fn cos(self) -> Self;

    /// `ceres::tan`.
    fn tan(self) -> Self;

    /// `ceres::asin`.
    fn asin(self) -> Self;

    /// `ceres::acos`.
    fn acos(self) -> Self;

    /// `ceres::atan`.
    fn atan(self) -> Self;

    /// `ceres::atan2(self, x)`: `self` is y, as in C.
    fn atan2(self, x: Self) -> Self;

    /// `ceres::exp`.
    fn exp(self) -> Self;

    /// `ceres::log` (natural logarithm).
    fn log(self) -> Self;

    /// `ceres::pow` with a constant exponent.
    fn pow_f64(self, exponent: f64) -> Self;

    /// `ceres::pow` with a differentiable exponent.
    fn pow(self, exponent: Self) -> Self;

    /// `ceres::floor` (a Jet's derivative is zero).
    fn floor(self) -> Self;

    /// `ceres::hypot(self, y)`.
    fn hypot(self, y: Self) -> Self;

    /// C++17's three-argument `hypot(x, y, z)`, as `ceres/rotation.h` calls it.
    fn hypot3(x: Self, y: Self, z: Self) -> Self;

    /// `ceres::fmin`: NaN is missing data; a Jet tie averages the two Jets.
    fn fmin(self, y: Self) -> Self;

    /// `ceres::fmax`: NaN is missing data; a Jet tie averages the two Jets.
    fn fmax(self, y: Self) -> Self;

    /// `ceres::isfinite`, on the value part.
    fn is_finite(self) -> bool;

    /// `ceres::isnan`, on the value part.
    fn is_nan(self) -> bool;
}

/// `std::fmin`: a NaN argument counts as missing, so the other one is returned.
pub fn fmin(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return y;
    }
    if y.is_nan() {
        return x;
    }
    if y < x {
        y
    } else {
        x
    }
}

/// `std::fmax`: a NaN argument counts as missing, so the other one is returned.
pub fn fmax(x: f64, y: f64) -> f64 {
    if x.is_nan() {
        return y;
    }
    if y.is_nan() {
        return x;
    }
    if x < y {
        y
    } else {
        x
    }
}

impl Scalar for f64 {
    #[inline]
    fn from_f64(value: f64) -> Self {
        value
    }

    #[inline]
    fn value(self) -> f64 {
        self
    }

    #[inline]
    fn sub_from_f64(self, s: f64) -> Self {
        s - self
    }

    #[inline]
    fn div_from_f64(self, s: f64) -> Self {
        s / self
    }

    #[inline]
    fn sqrt(self) -> Self {
        fns::sqrt(self)
    }

    #[inline]
    fn abs(self) -> Self {
        f64::abs(self)
    }

    #[inline]
    fn sin(self) -> Self {
        fns::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        fns::cos(self)
    }

    #[inline]
    fn tan(self) -> Self {
        fns::tan(self)
    }

    #[inline]
    fn asin(self) -> Self {
        fns::asin(self)
    }

    #[inline]
    fn acos(self) -> Self {
        fns::acos(self)
    }

    #[inline]
    fn atan(self) -> Self {
        fns::atan(self)
    }

    #[inline]
    fn atan2(self, x: Self) -> Self {
        fns::atan2(self, x)
    }

    #[inline]
    fn exp(self) -> Self {
        fns::exp(self)
    }

    #[inline]
    fn log(self) -> Self {
        fns::ln(self)
    }

    #[inline]
    fn pow_f64(self, exponent: f64) -> Self {
        fns::pow(self, exponent)
    }

    #[inline]
    fn pow(self, exponent: Self) -> Self {
        fns::pow(self, exponent)
    }

    #[inline]
    fn floor(self) -> Self {
        f64::floor(self)
    }

    #[inline]
    fn hypot(self, y: Self) -> Self {
        fns::hypot(self, y)
    }

    #[inline]
    fn hypot3(x: Self, y: Self, z: Self) -> Self {
        fns::hypot3(x, y, z)
    }

    #[inline]
    fn fmin(self, y: Self) -> Self {
        fmin(self, y)
    }

    #[inline]
    fn fmax(self, y: Self) -> Self {
        fmax(self, y)
    }

    #[inline]
    fn is_finite(self) -> bool {
        f64::is_finite(self)
    }

    #[inline]
    fn is_nan(self) -> bool {
        f64::is_nan(self)
    }
}
