//! Port of the scalar helpers of COLMAP's `colmap/math/math.h` and `math.cc` (re-exported at
//! [`crate::math`]): sign, clamp, degree/radian conversion, percentile/median/MAD,
//! mean/variance/stddev, `NextCombination`, sigmoid scaling, `NChooseK` and `TruncateCast`.
//! Port of colmap-sharp's `Mathematics/MathUtils.cs`. Tests: `tests/math/math.rs`
//! (math_test.cc 1:1).
//!
//! Tier A (exact) for the integer and plain-arithmetic helpers. Two exceptions:
//! - `sigmoid`/`scale_sigmoid` call `exp` through [`crate::math::fns`], which can differ
//!   from Apple libm in the last ulp (docs/CPP_DIVERGENCES.md entry 1); math_test.cc compares
//!   them with a tolerance.
//! - `percentile`/`median`/`median_absolute_deviation` are exact for ordinary input, but with
//!   NaN in the data, or -0.0 and +0.0 tied at the selected rank, the selection may pick a
//!   different element than libc++'s `nth_element` (entry 41).
//!
//! Translation notes:
//! - C++ templates over arithmetic `T` become generics over [`AsF64`] (the
//!   `static_cast<double>` every helper applies) and `PartialOrd`.
//! - `Percentile`/`Median` reorder the slice in place like COLMAP's `std::nth_element` does,
//!   via `select_nth_unstable_by`. The order left behind differs from libc++'s (it is
//!   unspecified there too); the returned value does not, because it depends only on the
//!   order statistics.

use crate::math::fns;
use crate::{check, check_ge, check_le};
use std::cmp::Ordering;

/// 95% quantile of chi-square distribution with 3 degrees of freedom.
/// Port of `colmap::kChiSquare95ThreeDof`.
pub const CHI_SQUARE_95_THREE_DOF: f64 = 7.814727903251179;

/// `M_PI` as COLMAP's math.h defines it.
pub const M_PI: f64 = std::f64::consts::PI;

/// A numeric type COLMAP's statistics helpers accept: `static_cast<double>(value)`.
pub trait AsF64: Copy + PartialOrd {
    /// `static_cast<double>(self)`.
    fn as_f64(self) -> f64;
}

macro_rules! impl_as_f64 {
    ($($t:ty),*) => {$(
        impl AsF64 for $t {
            #[inline]
            fn as_f64(self) -> f64 { self as f64 }
        }
    )*};
}
impl_as_f64!(i8, u8, i16, u16, i32, u32, i64, u64, isize, usize, f32, f64);

/// `SignOfNumber`: 1 if the number is positive (including 0), -1 if negative. NaN returns
/// -1, as in COLMAP (`val >= 0` is false for NaN).
pub fn sign_of_number<T: PartialOrd + Default>(val: T) -> i32 {
    if val >= T::default() {
        1
    } else {
        -1
    }
}

/// `Clamp`: evaluated exactly as `std::max(low, std::min(value, high))`, so NaN and inverted
/// bounds behave as in COLMAP.
pub fn clamp<T: PartialOrd + Copy>(value: T, low: T, high: T) -> T {
    // std::min(a, b) is (b < a) ? b : a; std::max(a, b) is (a < b) ? b : a.
    let min = if high < value { high } else { value };
    if low < min {
        min
    } else {
        low
    }
}

/// `DegToRad(float)`.
// COLMAP's exact literal, kept digit for digit.
#[allow(clippy::excessive_precision)]
pub fn deg_to_rad_f32(deg: f32) -> f32 {
    deg * 0.0174532925199432954743716805978692718781530857086181640625_f32
}

/// `DegToRad(double)`.
// COLMAP's exact literal, kept digit for digit.
#[allow(clippy::excessive_precision)]
pub fn deg_to_rad(deg: f64) -> f64 {
    deg * 0.0174532925199432954743716805978692718781530857086181640625
}

/// `RadToDeg(float)`.
// COLMAP's exact literal, kept digit for digit.
#[allow(clippy::excessive_precision)]
pub fn rad_to_deg_f32(rad: f32) -> f32 {
    rad * 57.29577951308232286464772187173366546630859375_f32
}

/// `RadToDeg(double)`.
// COLMAP's exact literal, kept digit for digit.
#[allow(clippy::excessive_precision)]
pub fn rad_to_deg(rad: f64) -> f64 {
    rad * 57.29577951308232286464772187173366546630859375
}

// Total order used for selection; NaN compares equal to everything (entry 41).
fn partial_order<T: PartialOrd>(a: &T, b: &T) -> Ordering {
    a.partial_cmp(b).unwrap_or(Ordering::Equal)
}

/// `Percentile`: the p-th percentile (p in [0, 100]) with linear interpolation between
/// values. Reorders the elements in place.
pub fn percentile<T: AsF64>(elems: &mut [T], p: f64) -> crate::Result<f64> {
    check!(!elems.is_empty());
    check_ge!(p, 0.0);
    check_le!(p, 100.0);
    let idx_double = p / 100. * (elems.len() - 1) as f64;
    let left_idx_double = idx_double.floor();
    let left_idx = left_idx_double as usize;
    let right_idx_double = idx_double.ceil();
    let right_idx = right_idx_double as usize;
    elems.select_nth_unstable_by(right_idx, partial_order);
    let right = elems[right_idx].as_f64();
    if left_idx == right_idx {
        return Ok(right);
    }
    // After the selection everything before right_idx is <= it, so the largest of those is
    // the (right_idx - 1)-th order statistic. std::max_element keeps the first maximum.
    let mut left_value = elems[0];
    for &e in &elems[1..right_idx] {
        if left_value < e {
            left_value = e;
        }
    }
    let left = left_value.as_f64();
    Ok((right_idx_double - idx_double) * left + (idx_double - left_idx_double) * right)
}

/// `Median`: the median with linear interpolation between the mid values. Reorders the
/// elements in place.
pub fn median<T: AsF64>(elems: &mut [T]) -> crate::Result<f64> {
    percentile(elems, 50.0)
}

/// `MedianAbsoluteDeviation`: returns `(median, MAD)`. Reorders the elements in place.
pub fn median_absolute_deviation<T: AsF64>(elems: &mut [T]) -> crate::Result<(f64, f64)> {
    let median_value = median(elems)?;
    let mut abs_deviations: Vec<f64> = elems
        .iter()
        .map(|e| (e.as_f64() - median_value).abs())
        .collect();
    Ok((median_value, median(&mut abs_deviations)?))
}

/// `Mean`: the mean value of the elements.
pub fn mean<T: AsF64>(elems: &[T]) -> crate::Result<f64> {
    check!(!elems.is_empty());
    let mut sum = 0.0;
    for &el in elems {
        sum += el.as_f64();
    }
    Ok(sum / elems.len() as f64)
}

/// `Variance`: the sample variance. A single element divides by zero and gives NaN, as in
/// COLMAP.
pub fn variance<T: AsF64>(elems: &[T]) -> crate::Result<f64> {
    let mean_value = mean(elems)?;
    let mut var = 0.0;
    for &el in elems {
        let diff = el.as_f64() - mean_value;
        var += diff * diff;
    }
    Ok(var / (elems.len() - 1) as f64)
}

/// `StdDev`: the sample standard deviation.
pub fn std_dev<T: AsF64>(elems: &[T]) -> crate::Result<f64> {
    Ok(fns::sqrt(variance(elems)?))
}

/// `NextCombination(first, middle, last)`: generate N-choose-K combinations. The current
/// combination is `elems[..middle]`; elements must start in sorted order. Returns false once
/// the sequence wraps back to the first combination.
pub fn next_combination<T: PartialOrd>(elems: &mut [T], middle: usize) -> bool {
    next_combination_impl(elems, 0, middle, middle, elems.len())
}

// Port of colmap::internal::NextCombination, with iterators as indices into one slice.
fn next_combination_impl<T: PartialOrd>(
    s: &mut [T],
    mut first1: usize,
    last1: usize,
    mut first2: usize,
    last2: usize,
) -> bool {
    if first1 == last1 || first2 == last2 {
        return false;
    }
    let mut m1 = last1 - 1;
    let mut m2 = last2 - 1;
    while m1 != first1 && s[m1] >= s[m2] {
        m1 -= 1;
    }
    let result = m1 == first1 && s[first1] >= s[m2];
    if !result {
        while first2 != m2 && s[m1] >= s[first2] {
            first2 += 1;
        }
        first1 = m1;
        s.swap(first1, first2);
        first1 += 1;
        first2 += 1;
    }
    if first1 != last1 && first2 != last2 {
        m1 = last1;
        m2 = first2;
        while m1 != first1 && m2 != last2 {
            m1 -= 1;
            s.swap(m1, m2);
            m2 += 1;
        }
        s[first1..m1].reverse();
        s[first1..last1].reverse();
        s[m2..last2].reverse();
        s[first2..last2].reverse();
    }
    !result
}

/// A floating-point type for [`sigmoid`] and [`scale_sigmoid`]: `f32` or `f64`.
pub trait SigmoidFloat:
    Copy
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::Div<Output = Self>
    + std::ops::Neg<Output = Self>
{
    /// `T(1)`.
    const ONE: Self;
    /// `T(2)`.
    const TWO: Self;
    /// `T(10)`.
    const TEN: Self;
    /// `std::exp` (via [`crate::math::fns`]).
    fn exp(self) -> Self;
}

impl SigmoidFloat for f64 {
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const TEN: Self = 10.0;
    fn exp(self) -> Self {
        fns::exp(self)
    }
}

impl SigmoidFloat for f32 {
    const ONE: Self = 1.0;
    const TWO: Self = 2.0;
    const TEN: Self = 10.0;
    fn exp(self) -> Self {
        fns::expf(self)
    }
}

/// `Sigmoid(x, alpha)`; COLMAP's default `alpha` is 1.
pub fn sigmoid<T: SigmoidFloat>(x: T, alpha: T) -> T {
    T::ONE / (T::ONE + (-x * alpha).exp())
}

/// `ScaleSigmoid(x, alpha, x0)`: x in [0, 1] -> [-x0, x0] -> sigmoid(x, alpha) -> [0, 1].
/// COLMAP's defaults are `alpha = 1`, `x0 = 10` ([`scale_sigmoid_default`]).
pub fn scale_sigmoid<T: SigmoidFloat>(x: T, alpha: T, x0: T) -> T {
    let t0 = sigmoid(-x0, alpha);
    let t1 = sigmoid(x0, alpha);
    (sigmoid(T::TWO * x0 * x - x0, alpha) - t0) / (t1 - t0)
}

/// `ScaleSigmoid(x)` with COLMAP's defaults `alpha = 1`, `x0 = 10`.
pub fn scale_sigmoid_default<T: SigmoidFloat>(x: T) -> T {
    scale_sigmoid(x, T::ONE, T::TEN)
}

/// `NChooseK`: the binomial coefficient n! / ((n - k)! k!). Implementation based on
/// <https://blog.plover.com/math/choose.html>. `uint64_t` arithmetic wraps as in C++.
pub fn n_choose_k(mut n: u64, k: u64) -> u64 {
    if n == 0 || n < k {
        return 0;
    }
    let mut r: u64 = 1;
    for d in 1..=k {
        r = r.wrapping_mul(n);
        n -= 1;
        r /= d;
    }
    r
}

/// `TruncateCast<T1, T2>`: cast and saturate instead of overflowing. Integer targets only:
/// C++ clamps against `std::numeric_limits<T2>::min()`, which for a floating-point T2 is the
/// smallest positive normal, and every COLMAP call site has an integer target.
pub trait TruncateCast<T2> {
    /// `TruncateCast<Self, T2>(self)`.
    fn truncate_cast(self) -> T2;
}

macro_rules! impl_truncate_cast {
    ($t1:ty => $($t2:ty),*) => {$(
        impl TruncateCast<$t2> for $t1 {
            #[allow(clippy::unnecessary_cast)]
            fn truncate_cast(self) -> $t2 {
                // static_cast<T1>(numeric_limits<T2>::max()/min()), C++'s modular / rounding
                // conversions (`as`).
                let max = <$t2>::MAX as $t1;
                let min = <$t2>::MIN as $t1;
                // std::max(min, value), then std::min(max, lower): NaN ends at min.
                let lower = if min < self { self } else { min };
                let clamped = if lower < max { lower } else { max };
                clamped as $t2
            }
        }
    )*};
}
impl_truncate_cast!(i32 => i8, u8, i16, u16, i32, u32);
impl_truncate_cast!(i64 => i8, u8, i16, u16, i32, u32);
impl_truncate_cast!(f32 => i8, u8, i16, u16, i32, u32);
impl_truncate_cast!(f64 => i8, u8, i16, u16, i32, u32);

/// `TruncateCast<T1, T2>(value)` as a free function.
pub fn truncate_cast<T1: TruncateCast<T2>, T2>(value: T1) -> T2 {
    value.truncate_cast()
}
