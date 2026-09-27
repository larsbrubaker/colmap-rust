//! Port of the `BitmapColor<T>` template and `internal::BitmapColorCast` of
//! `colmap/sensor/bitmap.h`: the pixel value type of [`super::Bitmap`]. COLMAP instantiates
//! it for `uint8_t` (stored pixels) and `float` (interpolated pixels). Port of colmap-sharp's
//! `ColmapSharp/Sensor/BitmapColor.cs`. Tests: the `BitmapColor` cases of `bitmap_test.cc`
//! in `tests/sensor/bitmap.rs`.
//!
//! Tier A. Translation notes:
//! - The template over arithmetic `T` is generic over [`BitmapColorScalar`] (`u8`, `u16`,
//!   `i32`, `f32`, `f64`).
//! - `==` compares with `T`'s `==`, like C++ (so NaN != NaN).
//! - [`BitmapColor::cast`] reproduces `BitmapColorCast` exactly: `std::round` (half away from
//!   zero), then a clamp to `[numeric_limits<D>::min(), numeric_limits<D>::max()]` converted
//!   to `T` and evaluated with `std::min`/`std::max`'s comparison order. For a floating `D`,
//!   `numeric_limits::min()` is the smallest positive normal, not `-max` (so COLMAP maps 0 to
//!   `FLT_MIN`); that quirk is kept.
//! - `Display` matches `operator<<`: `RGB(r, g, b)`, integers as integers and floating point
//!   in an ostream's default format (6 significant digits).

use std::fmt;

use crate::util::stream_format::{format_double, DEFAULT_PRECISION};

/// A component type of [`BitmapColor`]: the arithmetic types COLMAP instantiates it with.
pub trait BitmapColorScalar: Copy + PartialEq + fmt::Debug {
    /// `std::numeric_limits<T>::min()` (the smallest positive normal for floating point).
    const LIMITS_MIN: f64;
    /// `std::numeric_limits<T>::max()`.
    const LIMITS_MAX: f64;
    /// Whether `T` is floating point (for the ostream format).
    const IS_FLOAT: bool;
    /// The value as a double (exact for every implementor).
    fn to_f64(self) -> f64;
    /// `static_cast<T>(value)`; Rust's `as` saturates where C++ is undefined.
    fn from_f64(value: f64) -> Self;
}

macro_rules! impl_bitmap_color_scalar {
    ($t:ty, $min:expr, $is_float:expr) => {
        impl BitmapColorScalar for $t {
            const LIMITS_MIN: f64 = $min as f64;
            const LIMITS_MAX: f64 = <$t>::MAX as f64;
            const IS_FLOAT: bool = $is_float;
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_f64(value: f64) -> Self {
                value as $t
            }
        }
    };
}

impl_bitmap_color_scalar!(u8, u8::MIN, false);
impl_bitmap_color_scalar!(u16, u16::MIN, false);
impl_bitmap_color_scalar!(i32, i32::MIN, false);
impl_bitmap_color_scalar!(f32, f32::MIN_POSITIVE, true);
impl_bitmap_color_scalar!(f64, f64::MIN_POSITIVE, true);

/// Port of `colmap::BitmapColor<T>`: an RGB triple. Grey pixels use `r` (and set all three).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BitmapColor<T> {
    /// Red (or grey) component.
    pub r: T,
    /// Green component.
    pub g: T,
    /// Blue component.
    pub b: T,
}

impl<T: Copy> BitmapColor<T> {
    /// `BitmapColor(r, g, b)`.
    pub const fn new(r: T, g: T, b: T) -> Self {
        BitmapColor { r, g, b }
    }

    /// `BitmapColor(gray)`: all three components set to `gray`.
    pub const fn gray(gray: T) -> Self {
        BitmapColor {
            r: gray,
            g: gray,
            b: gray,
        }
    }
}

impl<T: BitmapColorScalar> BitmapColor<T> {
    /// Port of `BitmapColor::Cast<D>`: rounds each component and clamps it to `D`'s range.
    pub fn cast<D: BitmapColorScalar>(&self) -> BitmapColor<D> {
        BitmapColor {
            r: bitmap_color_cast::<T, D>(self.r),
            g: bitmap_color_cast::<T, D>(self.g),
            b: bitmap_color_cast::<T, D>(self.b),
        }
    }
}

/// Port of `internal::BitmapColorCast<T1, T2>`.
fn bitmap_color_cast<T: BitmapColorScalar, D: BitmapColorScalar>(value: T) -> D {
    // std::round(value) for an integer T converts to double first; for float it is roundf.
    // Rounding a float in double is exact, so one double path serves both. f64::round is
    // half away from zero, as std::round.
    let rounded = value.to_f64().round();
    // The limits are converted to T first, as the C++ static_cast<T1> does.
    let low = T::from_f64(D::LIMITS_MIN).to_f64();
    let high = T::from_f64(D::LIMITS_MAX).to_f64();
    // std::max(a, b) is (a < b) ? b : a and std::min(a, b) is (b < a) ? b : a, which decides
    // what a NaN turns into (the lower limit).
    let lowered = if low < rounded { rounded } else { low };
    let clamped = if lowered < high { lowered } else { high };
    D::from_f64(clamped)
}

impl<T: BitmapColorScalar> fmt::Display for BitmapColor<T> {
    /// Port of `operator<<(std::ostream&, const BitmapColor<T>&)`: `RGB(r, g, b)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let format = |value: T| {
            if T::IS_FLOAT {
                format_double(value.to_f64(), DEFAULT_PRECISION)
            } else {
                // Integers print exactly; to_f64 is exact for every integer implementor.
                format!("{}", value.to_f64() as i64)
            }
        };
        write!(
            f,
            "RGB({}, {}, {})",
            format(self.r),
            format(self.g),
            format(self.b)
        )
    }
}
