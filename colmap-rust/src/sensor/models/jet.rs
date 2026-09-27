//! [`Jet`]: forward-mode dual numbers with `N` derivative components, following the rules of
//! `ceres::Jet<double, N>` (Ceres Solver, BSD-3-Clause) operation for operation, so a
//! derivative computed here rounds exactly like Ceres'. Ported from colmap-sharp's
//! `ColmapSharp/Solver/Jet.cs`, restricted to what [`super::scalar::Scalar`] requires.
//!
//! Used by the iterative undistortion (`undistortion.rs`), which, like COLMAP, evaluates a
//! model's `Distortion` on `Jet<2>` to get its 2x2 Jacobian. Crate-private: the solver phase
//! owns the general Jet (and its mixed Jet/double operators) that bundle adjustment uses.
//!
//! Rules (Ceres `jet.h`), `f = (a, v)`, `g = (b, w)`:
//! - `f + g = (a + b, v + w)`, `f - g = (a - b, v - w)`, `-f = (-a, -v)`
//! - `f * g = (a b, a w + v b)`
//! - `f / g`: `ib = 1 / b`, `q = a ib`, `(q, (v - q w) ib)`
//! - `sqrt(f) = (s, v * (1 / (2 s)))` with `s = sqrt(a)`
//! - `abs(f) = (|a|, copysign(1, a) v)`, `sin(f) = (sin a, cos a v)`,
//!   `cos(f) = (cos a, -sin a v)`, `tan(f) = (t, (1 + t t) v)`,
//!   `atan(f) = (atan a, (1 / (1 + a a)) v)`,
//!   `atan2(g, f) = (atan2(b, a), (1 / (a a + b b)) (-b v + a w))`
//! - `s - f = (s - a, -v)` (Ceres' mixed operator)
//! - comparisons compare `a` only.

// Indexed loops mirror Ceres' lane-by-lane expressions over several arrays at once.
#![allow(clippy::needless_range_loop)]

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::math::fns;

use super::scalar::Scalar;

/// Port of `ceres::Jet<double, N>`: a value `a` and its derivative `v`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Jet<const N: usize> {
    /// The value part.
    pub(crate) a: f64,
    /// The derivative (infinitesimal) part.
    pub(crate) v: [f64; N],
}

impl<const N: usize> Jet<N> {
    /// `Jet(value)`: a constant.
    pub(crate) fn constant(a: f64) -> Self {
        Self { a, v: [0.0; N] }
    }

    /// `Jet(value, k)`: the `k`-th independent variable (`v = e_k`).
    pub(crate) fn variable(a: f64, k: usize) -> Self {
        let mut v = [0.0; N];
        v[k] = 1.0;
        Self { a, v }
    }

    /// `(a, s * v)`, Ceres' `v * s` lane by lane.
    fn scaled(a: f64, s: f64, v: [f64; N]) -> Self {
        let mut out = [0.0; N];
        for i in 0..N {
            out[i] = s * v[i];
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
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = -self.v[i];
        }
        Self { a: -self.a, v }
    }
}

impl<const N: usize> Mul for Jet<N> {
    type Output = Self;
    fn mul(self, g: Self) -> Self {
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = self.a * g.v[i] + self.v[i] * g.a;
        }
        Self { a: self.a * g.a, v }
    }
}

impl<const N: usize> Div for Jet<N> {
    type Output = Self;
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

impl<const N: usize> Scalar for Jet<N> {
    fn from_f64(value: f64) -> Self {
        Self::constant(value)
    }

    fn value(self) -> f64 {
        self.a
    }

    fn sub_from_f64(self, s: f64) -> Self {
        let mut v = [0.0; N];
        for i in 0..N {
            v[i] = -self.v[i];
        }
        Self { a: s - self.a, v }
    }

    fn sqrt(self) -> Self {
        let tmp = fns::sqrt(self.a);
        let two_a_inverse = 1.0 / (2.0 * tmp);
        Self::scaled(tmp, two_a_inverse, self.v)
    }

    fn abs(self) -> Self {
        Self::scaled(self.a.abs(), 1.0_f64.copysign(self.a), self.v)
    }

    fn sin(self) -> Self {
        Self::scaled(fns::sin(self.a), fns::cos(self.a), self.v)
    }

    fn cos(self) -> Self {
        Self::scaled(fns::cos(self.a), -fns::sin(self.a), self.v)
    }

    fn tan(self) -> Self {
        let tan_a = fns::tan(self.a);
        let tmp = 1.0 + tan_a * tan_a;
        Self::scaled(tan_a, tmp, self.v)
    }

    fn atan(self) -> Self {
        let tmp = 1.0 / (1.0 + self.a * self.a);
        Self::scaled(fns::atan(self.a), tmp, self.v)
    }

    fn atan2(self, x: Self) -> Self {
        // Ceres' atan2(g, f) with g = self (y) and f = x.
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
}
