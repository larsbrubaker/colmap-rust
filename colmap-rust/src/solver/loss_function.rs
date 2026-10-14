//! Robust loss functions rho(s), applied to a residual block's squared norm s = |f(x)|^2 so the
//! solver minimizes 1/2 sum_i rho(|f_i|^2). Port of colmap-sharp's
//! `ColmapSharp/Solver/LossFunctions.cs`, which ports Ceres Solver 2.2.0's
//! `include/ceres/loss_function.h` and `internal/ceres/loss_function.cc` (BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md).
//!
//! COLMAP builds Trivial, SoftLOne, Cauchy and Huber for bundle adjustment
//! (`estimators/bundle_adjustment_ceres.cc` `CreateLossFunction`), Cauchy for view-graph
//! calibration, Arctan for gravity refinement and a Scaled Huber/Cauchy for global
//! positioning. Tolerant, Tukey and Composed are Ceres' remaining losses and come along because
//! they are a few lines each and share the test. The residual-block corrector
//! (`solver/corrector.rs`) consumes `evaluate`'s rho[0..2].
//!
//! Translation notes:
//! - [`LossFunction`] is object safe and `Send + Sync` (every loss is immutable data), so a
//!   problem can hold `Box<dyn LossFunction>` and evaluate it from any thread.
//! - Ceres' ownership flags (TAKE_OWNERSHIP / DO_NOT_TAKE_OWNERSHIP) have no Rust meaning: the
//!   composed and scaled losses own their inner losses as boxes.
//! - `LossFunctionWrapper` (a mutable indirection for changing a loss mid-solve) is not used
//!   by COLMAP and is not ported.
//!
//! Tests: `tests/solver/loss_function.rs` (Ceres' `loss_function_test.cc`).

use crate::math::fns;
use crate::{check_ge, check_gt, Result};

/// `std::numeric_limits<double>::min()`, the smallest positive normal double (not
/// `f64::MIN`, the most negative, nor the smallest subnormal). Ceres floors rho' at it so the
/// corrector never divides by zero.
const MIN_NORMAL: f64 = f64::MIN_POSITIVE;

/// Port of `ceres::LossFunction`: a robustifier rho(s) of the squared residual norm s.
pub trait LossFunction: Send + Sync {
    /// Writes rho(s), rho'(s) and rho''(s) to `rho`. s is non-negative; rho(0) = 0,
    /// rho'(0) = 1 and rho'(s) > 0 for every loss here except Tukey's (whose outlier region
    /// has rho' = 0).
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]);
}

/// Port of `ceres::TrivialLoss`: rho(s) = s, plain least squares.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TrivialLoss;

impl LossFunction for TrivialLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        rho[0] = s;
        rho[1] = 1.0;
        rho[2] = 0.0;
    }
}

/// Port of `ceres::HuberLoss`: rho(s) = s for s <= a^2, 2 a sqrt(s) - a^2 beyond; quadratic
/// near zero and linear in the residual for outliers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HuberLoss {
    a: f64,
    b: f64,
}

impl HuberLoss {
    /// The loss with scale `a` (the residual size where it turns linear).
    pub fn new(a: f64) -> Self {
        Self { a, b: a * a }
    }
}

impl LossFunction for HuberLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        if s > self.b {
            // Outlier region. 'r' is always positive.
            let r = fns::sqrt(s);
            rho[0] = 2.0 * self.a * r - self.b;
            rho[1] = MIN_NORMAL.max(self.a / r);
            rho[2] = -rho[1] / (2.0 * s);
        } else {
            // Inlier region.
            rho[0] = s;
            rho[1] = 1.0;
            rho[2] = 0.0;
        }
    }
}

/// Port of `ceres::SoftLOneLoss`: rho(s) = 2 a^2 (sqrt(1 + s / a^2) - 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SoftLOneLoss {
    b: f64,
    c: f64,
}

impl SoftLOneLoss {
    /// The loss with scale `a`.
    pub fn new(a: f64) -> Self {
        let b = a * a;
        Self { b, c: 1.0 / b }
    }
}

impl LossFunction for SoftLOneLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let sum = 1.0 + s * self.c;
        let tmp = fns::sqrt(sum);
        // 'sum' and 'tmp' are always positive, assuming that 's' is.
        rho[0] = 2.0 * self.b * (tmp - 1.0);
        rho[1] = MIN_NORMAL.max(1.0 / tmp);
        rho[2] = -(self.c * rho[1]) / (2.0 * sum);
    }
}

/// Port of `ceres::CauchyLoss`: rho(s) = a^2 log(1 + s / a^2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CauchyLoss {
    b: f64,
    c: f64,
}

impl CauchyLoss {
    /// The loss with scale `a`.
    pub fn new(a: f64) -> Self {
        let b = a * a;
        Self { b, c: 1.0 / b }
    }
}

impl LossFunction for CauchyLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let sum = 1.0 + s * self.c;
        let inv = 1.0 / sum;
        // 'sum' and 'inv' are always positive, assuming that 's' is.
        rho[0] = self.b * fns::ln(sum);
        rho[1] = MIN_NORMAL.max(inv);
        rho[2] = -self.c * (inv * inv);
    }
}

/// Port of `ceres::ArctanLoss`: rho(s) = a atan2(s, a), which caps the cost of any residual
/// at a pi / 2.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ArctanLoss {
    a: f64,
    b: f64,
}

impl ArctanLoss {
    /// The loss with scale `a`.
    pub fn new(a: f64) -> Self {
        Self {
            a,
            b: 1.0 / (a * a),
        }
    }
}

impl LossFunction for ArctanLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let sum = 1.0 + s * s * self.b;
        let inv = 1.0 / sum;
        // 'sum' and 'inv' are always positive.
        rho[0] = self.a * fns::atan2(s, self.a);
        rho[1] = MIN_NORMAL.max(inv);
        rho[2] = -2.0 * s * self.b * (inv * inv);
    }
}

/// Port of `ceres::TolerantLoss`: rho(s) = b log(1 + e^((s - a) / b)) - b log(1 + e^(-a / b)),
/// which costs almost nothing below a and grows linearly above it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TolerantLoss {
    a: f64,
    b: f64,
    c: f64,
}

impl TolerantLoss {
    /// The loss with tolerance `a` (>= 0) and transition width `b` (> 0). Ceres CHECK-fails
    /// on either bound; the port returns the check's error.
    pub fn new(a: f64, b: f64) -> Result<Self> {
        check_ge!(a, 0.0);
        check_gt!(b, 0.0);
        Ok(Self {
            a,
            b,
            c: b * fns::ln(1.0 + fns::exp(-a / b)),
        })
    }
}

impl LossFunction for TolerantLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let x = (s - self.a) / self.b;
        // The basic equation is rho[0] = b ln(1 + e^x). However, if e^x is too large, it will
        // overflow. Since numerically 1 + e^x == e^x when x is greater than about ln(2^53)
        // for doubles, beyond this threshold x substitutes for ln(1 + e^x) as a numerically
        // equivalent approximation.
        const LOG2_POW53: f64 = 36.7;
        if x > LOG2_POW53 {
            rho[0] = s - self.a - self.c;
            rho[1] = 1.0;
            rho[2] = 0.0;
        } else {
            let e_x = fns::exp(x);
            rho[0] = self.b * fns::ln(1.0 + e_x) - self.c;
            rho[1] = MIN_NORMAL.max(e_x / (1.0 + e_x));
            rho[2] = 0.5 / (self.b * (1.0 + fns::cosh(x)));
        }
    }
}

/// Port of `ceres::TukeyLoss`: Tukey's biweight, rho(s) = a^2 / 3 (1 - (1 - s / a^2)^3) for
/// s <= a^2 and the constant a^2 / 3 beyond, so outliers contribute no gradient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TukeyLoss {
    a_squared: f64,
}

impl TukeyLoss {
    /// The loss with scale `a`.
    pub fn new(a: f64) -> Self {
        Self { a_squared: a * a }
    }
}

impl LossFunction for TukeyLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        if s <= self.a_squared {
            // Inlier region.
            let value = 1.0 - s / self.a_squared;
            let value_sq = value * value;
            rho[0] = self.a_squared / 3.0 * (1.0 - value_sq * value);
            rho[1] = value_sq;
            rho[2] = -2.0 / self.a_squared * value;
        } else {
            // Outlier region.
            rho[0] = self.a_squared / 3.0;
            rho[1] = 0.0;
            rho[2] = 0.0;
        }
    }
}

/// Port of `ceres::ComposedLoss`: rho(s) = f(g(s)).
pub struct ComposedLoss {
    f: Box<dyn LossFunction>,
    g: Box<dyn LossFunction>,
}

impl ComposedLoss {
    /// f(g(s)).
    pub fn new(f: Box<dyn LossFunction>, g: Box<dyn LossFunction>) -> Self {
        Self { f, g }
    }
}

impl LossFunction for ComposedLoss {
    fn evaluate(&self, s: f64, rho: &mut [f64; 3]) {
        let mut rho_f = [0.0; 3];
        let mut rho_g = [0.0; 3];
        self.g.evaluate(s, &mut rho_g);
        self.f.evaluate(rho_g[0], &mut rho_f);
        rho[0] = rho_f[0];
        // f'(g(s)) * g'(s).
        rho[1] = rho_f[1] * rho_g[1];
        // f''(g(s)) * g'(s) * g'(s) + f'(g(s)) * g''(s).
        rho[2] = rho_f[2] * rho_g[1] * rho_g[1] + rho_f[1] * rho_g[2];
    }
}

/// Port of `ceres::ScaledLoss`: a * rho(s), or a * s when no inner loss is given. COLMAP
/// weights residual groups against each other with it (global positioning).
pub struct ScaledLoss {
    rho: Option<Box<dyn LossFunction>>,
    a: f64,
}

impl ScaledLoss {
    /// a * rho(s); `None` for `rho` means the trivial loss (Ceres' NULL).
    pub fn new(rho: Option<Box<dyn LossFunction>>, a: f64) -> Self {
        Self { rho, a }
    }
}

impl LossFunction for ScaledLoss {
    fn evaluate(&self, s: f64, result: &mut [f64; 3]) {
        match &self.rho {
            None => {
                result[0] = self.a * s;
                result[1] = self.a;
                result[2] = 0.0;
            }
            Some(rho) => {
                rho.evaluate(s, result);
                result[0] *= self.a;
                result[1] *= self.a;
                result[2] *= self.a;
            }
        }
    }
}
