//! [`Scalar`]: the numeric abstraction COLMAP's `template <typename T>` camera model code is
//! written against, so one body serves plain `double` evaluation and `ceres::Jet` automatic
//! differentiation (`colmap/sensor/models.h` instantiates `ImgFromCam` and `Distortion` with
//! both).
//!
//! The trait holds what `ceres::Jet` offers COLMAP's templates: the four binary operators on
//! two scalars, negation, comparisons on the value part, `T(double)` construction
//! ([`Scalar::from_f64`]), and the elementary functions the models call. Implemented for `f64`
//! here (every transcendental through [`crate::math::fns`]) and for the forward-mode
//! [`super::jet::Jet`] used by the iterative undistortion. A later solver phase implements it
//! for its own Jet type so the cost functions reuse the model code unchanged. Port of the
//! design of colmap-sharp's `IScalar<T>` (`ColmapSharp/Solver/Scalar.cs`).
//!
//! Ceres gives the *mixed* scalar/Jet operators (`1.0 - alpha` with `alpha` a Jet) their own
//! rules, which can differ from `T(1.0) - alpha` in the sign of a zero derivative; the one
//! mixed operation the camera models use has its own member, [`Scalar::sub_from_f64`].

use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::math::fns;

/// A scalar type the camera models run on: `f64` for plain evaluation, a Jet for automatic
/// differentiation. Comparisons (`PartialOrd`) compare the value part, like Ceres' Jet.
pub trait Scalar:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// C++ `T(value)`: a constant (a Jet with zero derivative).
    fn from_f64(value: f64) -> Self;

    /// The value part: the double itself, or a Jet's `a`.
    fn value(self) -> f64;

    /// C++ `s - self` with `s` a plain double (Ceres' mixed operator: `(s - a, -v)`).
    fn sub_from_f64(self, s: f64) -> Self;

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

    /// `ceres::atan`.
    fn atan(self) -> Self;

    /// `ceres::atan2(self, x)`: `self` is y, as in C.
    fn atan2(self, x: Self) -> Self;
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
    fn atan(self) -> Self {
        fns::atan(self)
    }

    #[inline]
    fn atan2(self, x: Self) -> Self {
        fns::atan2(self, x)
    }
}
