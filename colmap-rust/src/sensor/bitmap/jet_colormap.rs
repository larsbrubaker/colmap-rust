//! Port of the `JetColormap` class of `colmap/sensor/bitmap.h` / `bitmap.cc` (via
//! colmap-sharp's `ColmapSharp/Sensor/JetColormap.cs`), used by the MVS depth and normal map
//! visualizations. Next to [`super::Bitmap`] because COLMAP declares it there.
//! `bitmap_test.cc` has no case for it; `tests/sensor/bitmap_extra.rs` has a Rust-only one.
//!
//! Tier A: single-precision arithmetic in COLMAP's order.

/// Port of `colmap::JetColormap`: Jet colormap inspired by Matlab. Grayvalues are expected in
/// the range [0, 1] and are converted to RGB values in the same range.
pub struct JetColormap;

impl JetColormap {
    /// `JetColormap::Red`.
    pub fn red(gray: f32) -> f32 {
        Self::base(gray - 0.25)
    }

    /// `JetColormap::Green`.
    pub fn green(gray: f32) -> f32 {
        Self::base(gray)
    }

    /// `JetColormap::Blue`.
    pub fn blue(gray: f32) -> f32 {
        Self::base(gray + 0.25)
    }

    fn base(val: f32) -> f32 {
        if val <= 0.125 {
            0.0
        } else if val <= 0.375 {
            Self::interpolate(2.0 * val - 1.0, 0.0, -0.75, 1.0, -0.25)
        } else if val <= 0.625 {
            1.0
        } else if val <= 0.87 {
            Self::interpolate(2.0 * val - 1.0, 1.0, 0.25, 0.0, 0.75)
        } else {
            0.0
        }
    }

    fn interpolate(val: f32, y0: f32, x0: f32, y1: f32, x1: f32) -> f32 {
        (val - x0) * (y1 - y0) / (x1 - x0) + y0
    }
}
