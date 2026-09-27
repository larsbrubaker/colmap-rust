//! Port of `colmap/sensor/bitmap.h` / `bitmap.cc`: [`Bitmap`], the 8-bit grey or RGB pixel
//! buffer (row-major, interleaved) that feature extraction, undistortion and MVS read images
//! into. Port of colmap-sharp's `ColmapSharp/Sensor/Bitmap*.cs`, split by responsibility:
//! - `mod.rs` (this file): storage, pixel access, interpolation, rotation, grey/RGB clones;
//! - [`color`]: the pixel value type [`BitmapColor`];
//! - `exif.rs`: the metadata store and the EXIF getters (focal length, GPS);
//! - `resize.rs`: `Rescale` / `Thumbnail`'s resampling filters;
//! - [`jet_colormap`]: the `JetColormap` declared in `bitmap.h`;
//! - `tests.rs`: unit tests of the resampler against a full-intermediate reference.
//!
//! Tests: `tests/sensor/bitmap.rs` (`bitmap_test.cc`; the file I/O cases are skipped, see
//! `PORTING_PLAN.md`).
//!
//! No file decoding or encoding: COLMAP's `Read`/`Write` go through OpenImageIO, which is
//! native and not ported (`docs/LICENSE_AUDIT.md`, `docs/CPP_DIVERGENCES.md` entry 124). The
//! host decodes an image into [`Bitmap::row_major_data_mut`] and feeds its EXIF block to
//! [`super::exif_reader`], which fills the metadata the EXIF getters read under OpenImageIO's
//! attribute names.
//!
//! Tier A (exact) for everything here except `Rescale` (`resize.rs`, entry 121). Translation
//! notes:
//! - Dimensions stay C++ `int` (`i32`), since pixel coordinates can be negative.
//! - The C++ copy constructor/assignment is [`Clone`]: a copy of an empty bitmap has no
//!   metadata, like COLMAP's null `meta_data_`. C++ move construction/assignment, which leave
//!   the source empty, is `std::mem::take`.
//! - The linear-colorspace flag only matters to OIIO's `Read`/`Write` color conversion; it is
//!   kept (and recorded as "oiio:ColorSpace" metadata, as COLMAP's constructor does) so a
//!   host can round-trip it.

pub mod color;
mod exif;
pub mod jet_colormap;
mod resize;
#[cfg(test)]
mod tests;

use std::fmt;

pub use color::{BitmapColor, BitmapColorScalar};
pub use exif::MetaDataValue;
pub use jet_colormap::JetColormap;
pub use resize::RescaleFilter;

use exif::BitmapMetaData;

use crate::util::check::{ColmapError, ErrorKind};
use crate::{check_eq, Result};

/// Port of `colmap::Bitmap`: an 8-bit grey (1 channel) or RGB (3 channels) image, stored
/// row-major with interleaved channels.
#[derive(Debug, Default)]
pub struct Bitmap {
    width: i32,
    height: i32,
    channels: i32,
    linear_colorspace: bool,
    data: Vec<u8>,
    meta_data: BitmapMetaData,
}

impl Clone for Bitmap {
    /// Port of the copy constructor (`Bitmap::Clone`): a deep copy whose metadata is copied
    /// only when the source is not empty.
    fn clone(&self) -> Self {
        Bitmap {
            width: self.width,
            height: self.height,
            channels: self.channels,
            linear_colorspace: self.linear_colorspace,
            data: self.data.clone(),
            meta_data: if self.is_empty() {
                BitmapMetaData::default()
            } else {
                self.meta_data.clone()
            },
        }
    }
}

/// `width * height * channels` as a buffer length, or an error for negative dimensions, a
/// product that overflows `usize`, or a scan line (`Pitch`, a C++ `int`) beyond `i32`. COLMAP
/// resizes its vector with the `int` product, where such sizes are undefined behavior or a
/// huge allocation.
fn checked_buffer_len(width: i32, height: i32, channels: i32) -> Result<usize> {
    let invalid = || {
        ColmapError::new(
            ErrorKind::InvalidArgument,
            format!("Invalid bitmap dimensions: {width}x{height}x{channels}"),
        )
    };
    let dim = |v: i32| usize::try_from(v).map_err(|_| invalid());
    width.checked_mul(channels).ok_or_else(invalid)?;
    dim(width)?
        .checked_mul(dim(height)?)
        .and_then(|n| n.checked_mul(channels as usize))
        .filter(|&n| n <= isize::MAX as usize)
        .ok_or_else(invalid)
}

/// [`checked_buffer_len`] for sizes the caller guarantees; panics on invalid ones.
fn buffer_len(width: i32, height: i32, channels: i32) -> usize {
    match checked_buffer_len(width, height, channels) {
        Ok(len) => len,
        Err(error) => panic!("{}", error.message()),
    }
}

impl Bitmap {
    /// `Bitmap()`: an empty bitmap (width, height and channels 0), without metadata.
    pub fn empty() -> Self {
        Self::default()
    }

    /// `Bitmap(width, height, as_rgb)`: a zero-filled bitmap in the sRGB colorspace.
    ///
    /// # Panics
    ///
    /// On negative or overflowing dimensions. Use [`Bitmap::try_new`] for sizes from
    /// untrusted input (e.g. a decoder's image header).
    pub fn new(width: i32, height: i32, as_rgb: bool) -> Self {
        Self::with_colorspace(width, height, as_rgb, false)
    }

    /// [`Bitmap::new`] for untrusted sizes: an error (`InvalidArgument`) for negative
    /// dimensions or a buffer size that overflows.
    pub fn try_new(width: i32, height: i32, as_rgb: bool) -> Result<Self> {
        Self::try_with_colorspace(width, height, as_rgb, false)
    }

    /// `Bitmap(width, height, as_rgb, linear_colorspace)`.
    ///
    /// # Panics
    ///
    /// On negative or overflowing dimensions; see [`Bitmap::try_with_colorspace`].
    pub fn with_colorspace(width: i32, height: i32, as_rgb: bool, linear_colorspace: bool) -> Self {
        match Self::try_with_colorspace(width, height, as_rgb, linear_colorspace) {
            Ok(bitmap) => bitmap,
            Err(error) => panic!("{}", error.message()),
        }
    }

    /// [`Bitmap::with_colorspace`] for untrusted sizes: an error for negative dimensions or
    /// a buffer size that overflows.
    pub fn try_with_colorspace(
        width: i32,
        height: i32,
        as_rgb: bool,
        linear_colorspace: bool,
    ) -> Result<Self> {
        let channels = if as_rgb { 3 } else { 1 };
        let len = checked_buffer_len(width, height, channels)?;
        let mut meta_data = BitmapMetaData::default();
        meta_data.set(
            "oiio:ColorSpace",
            MetaDataValue::String(if linear_colorspace { "linear" } else { "sRGB" }.to_string()),
        );
        Ok(Bitmap {
            width,
            height,
            channels,
            linear_colorspace,
            data: vec![0; len],
            meta_data,
        })
    }

    /// `Width`.
    pub fn width(&self) -> i32 {
        self.width
    }

    /// `Height`.
    pub fn height(&self) -> i32 {
        self.height
    }

    /// `Channels`: 1 for grey, 3 for RGB, 0 when empty.
    pub fn channels(&self) -> i32 {
        self.channels
    }

    /// `BitsPerPixel`: 8 for grey and 24 for RGB images.
    pub fn bits_per_pixel(&self) -> i32 {
        self.channels * 8
    }

    /// `NumBytes`: number of bytes required to store the image.
    pub fn num_bytes(&self) -> usize {
        self.data.len()
    }

    /// `Pitch`: scan line size in bytes, also known as stride.
    pub fn pitch(&self) -> i32 {
        self.width * self.channels
    }

    /// `IsEmpty`: whether the image is empty (i.e., width/height=0).
    pub fn is_empty(&self) -> bool {
        self.num_bytes() == 0
    }

    /// `IsRGB`.
    pub fn is_rgb(&self) -> bool {
        self.channels == 3
    }

    /// `IsGrey`.
    pub fn is_grey(&self) -> bool {
        self.channels == 1
    }

    /// Whether the pixel values are in a linear (not sRGB) colorspace.
    pub fn is_linear_colorspace(&self) -> bool {
        self.linear_colorspace
    }

    /// `RowMajorData() const`: the raw pixels, row-major with interleaved channels.
    pub fn row_major_data(&self) -> &[u8] {
        &self.data
    }

    /// `RowMajorData()`: the host writes decoded pixels here. The length is fixed by the
    /// dimensions (a slice, not the vector, so it cannot drift from them).
    pub fn row_major_data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Port of `Bitmap::GetPixel`: `None` outside the image. For grey images all three
    /// components hold the grey value.
    pub fn get_pixel(&self, x: i32, y: i32) -> Option<BitmapColor<u8>> {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return None;
        }
        let index = (y as usize * self.width as usize + x as usize) * self.channels as usize;
        if self.is_grey() {
            Some(BitmapColor::gray(self.data[index]))
        } else if self.is_rgb() {
            Some(BitmapColor::new(
                self.data[index],
                self.data[index + 1],
                self.data[index + 2],
            ))
        } else {
            None
        }
    }

    /// Port of `Bitmap::SetPixel`: false outside the image. For grayscale images, only the
    /// red element of the color is used.
    pub fn set_pixel(&mut self, x: i32, y: i32, color: BitmapColor<u8>) -> bool {
        if x < 0 || x >= self.width || y < 0 || y >= self.height {
            return false;
        }
        let index = (y as usize * self.width as usize + x as usize) * self.channels as usize;
        if self.is_grey() {
            self.data[index] = color.r;
            true
        } else if self.is_rgb() {
            self.data[index] = color.r;
            self.data[index + 1] = color.g;
            self.data[index + 2] = color.b;
            true
        } else {
            false
        }
    }

    /// Port of `Bitmap::Fill`: fill the entire bitmap with a uniform color. For grayscale
    /// images, the red element is used.
    pub fn fill(&mut self, color: BitmapColor<u8>) -> Result<()> {
        if self.is_grey() {
            self.data.fill(color.r);
        } else {
            check_eq!(self.data.len() % 3, 0);
            for pixel in self.data.chunks_exact_mut(3) {
                pixel[0] = color.r;
                pixel[1] = color.g;
                pixel[2] = color.b;
            }
        }
        Ok(())
    }

    /// Port of `Bitmap::InterpolateNearestNeighbor`: the pixel at the rounded position
    /// (`std::round`, half away from zero), or `None` outside the image.
    pub fn interpolate_nearest_neighbor(&self, x: f64, y: f64) -> Option<BitmapColor<u8>> {
        // `as i32` maps NaN to 0, which would return pixel (0, 0); a NaN point is outside the
        // image (docs/CPP_DIVERGENCES.md, entry 123). Infinities and values beyond int range
        // saturate and get_pixel rejects them.
        if x.is_nan() || y.is_nan() {
            return None;
        }
        self.get_pixel(x.round() as i32, y.round() as i32)
    }

    /// Port of `Bitmap::InterpolateBilinear`: `None` unless all four neighbors are inside
    /// the image. Accumulates in double and rounds to float at the end, as COLMAP does.
    pub fn interpolate_bilinear(&self, x: f64, y: f64) -> Option<BitmapColor<f32>> {
        // COLMAP's check is `x0 < 0 || x1 >= width_ || ...` on x0 = (int)floor(x),
        // x1 = x0 + 1. For finite x, floor(x) >= 0 <=> x >= 0 and floor(x) + 1 < width <=>
        // x < width - 1, so testing the doubles first is the same check; it also keeps points
        // beyond int range and NaN out, where the C++ cast is undefined (entry 123).
        let in_range =
            x >= 0.0 && x < f64::from(self.width - 1) && y >= 0.0 && y < f64::from(self.height - 1);
        if !in_range {
            return None;
        }

        let x0 = x.floor() as usize;
        let x1 = x0 + 1;
        let y0 = y.floor() as usize;
        let y1 = y0 + 1;

        let dx = x - x0 as f64;
        let dy = y - y0 as f64;
        let dx_1 = 1.0 - dx;
        let dy_1 = 1.0 - dy;

        let pitch = self.pitch() as usize;
        let line0 = &self.data[y0 * pitch..];
        let line1 = &self.data[y1 * pitch..];
        let v = |line: &[u8], i: usize| f64::from(line[i]);

        if self.is_grey() {
            // Top row, column-wise linear interpolation.
            let v0 = dx_1 * v(line0, x0) + dx * v(line0, x1);
            // Bottom row, column-wise linear interpolation.
            let v1 = dx_1 * v(line1, x0) + dx * v(line1, x1);
            // Row-wise linear interpolation.
            let r = (dy_1 * v0 + dy * v1) as f32;
            Some(BitmapColor::gray(r))
        } else if self.is_rgb() {
            let (p00, p01) = (3 * x0, 3 * x1);
            // Top row, column-wise linear interpolation.
            let v0_r = dx_1 * v(line0, p00) + dx * v(line0, p01);
            let v0_g = dx_1 * v(line0, p00 + 1) + dx * v(line0, p01 + 1);
            let v0_b = dx_1 * v(line0, p00 + 2) + dx * v(line0, p01 + 2);
            // Bottom row, column-wise linear interpolation.
            let v1_r = dx_1 * v(line1, p00) + dx * v(line1, p01);
            let v1_g = dx_1 * v(line1, p00 + 1) + dx * v(line1, p01 + 1);
            let v1_b = dx_1 * v(line1, p00 + 2) + dx * v(line1, p01 + 2);
            // Row-wise linear interpolation.
            Some(BitmapColor::new(
                (dy_1 * v0_r + dy * v1_r) as f32,
                (dy_1 * v0_g + dy * v1_g) as f32,
                (dy_1 * v0_b + dy * v1_b) as f32,
            ))
        } else {
            None
        }
    }

    /// Port of `Bitmap::Rot90`: rotate the image by `k * 90` degrees counter-clockwise (`k`
    /// may be negative). COLMAP delegates to OIIO's rotate90/180/270, which are exact pixel
    /// moves.
    pub fn rot90(&mut self, k: i32) {
        if self.is_empty() {
            return;
        }
        let k = k.rem_euclid(4);
        if k == 0 {
            return;
        }

        let (width, height, channels) = (
            self.width as usize,
            self.height as usize,
            self.channels as usize,
        );
        let swap_dims = k == 1 || k == 3;
        let new_width = if swap_dims { height } else { width };
        let new_height = if swap_dims { width } else { height };
        let mut new_data = vec![0u8; new_width * new_height * channels];

        for y in 0..height {
            for x in 0..width {
                // Where source pixel (x, y) lands after the counter-clockwise rotation.
                let (nx, ny) = match k {
                    1 => (y, width - 1 - x),
                    2 => (width - 1 - x, height - 1 - y),
                    _ => (height - 1 - y, x),
                };
                let src = (y * width + x) * channels;
                let dst = (ny * new_width + nx) * channels;
                new_data[dst..dst + channels].copy_from_slice(&self.data[src..src + channels]);
            }
        }

        self.width = new_width as i32;
        self.height = new_height as i32;
        self.data = new_data;
    }

    /// Port of `Bitmap::CloneAsGrey`. Uses COLMAP's Rec. 709 luma weights in float and rounds
    /// by adding 0.5 before truncating (the weighted sum is non-negative).
    pub fn clone_as_grey(&self) -> Bitmap {
        if self.is_grey() {
            return self.clone();
        }
        let data = self
            .data
            .chunks_exact(3)
            .map(|p| {
                // Each product and sum is rounded to float, as the C++ float expression is.
                (0.2126f32 * f32::from(p[0])
                    + 0.7152f32 * f32::from(p[1])
                    + 0.0722f32 * f32::from(p[2])
                    + 0.5f32) as u8
            })
            .collect();
        Bitmap {
            width: self.width,
            height: self.height,
            channels: 1,
            linear_colorspace: self.linear_colorspace,
            data,
            meta_data: self.meta_data.clone(),
        }
    }

    /// Port of `Bitmap::CloneAsRGB`: replicates the grey value into all channels.
    pub fn clone_as_rgb(&self) -> Result<Bitmap> {
        if self.is_rgb() {
            return Ok(self.clone());
        }
        check_eq!(self.channels, 1);
        let data = self.data.iter().flat_map(|&v| [v, v, v]).collect();
        Ok(Bitmap {
            width: self.width,
            height: self.height,
            channels: 3,
            linear_colorspace: self.linear_colorspace,
            data,
            meta_data: self.meta_data.clone(),
        })
    }
}

impl fmt::Display for Bitmap {
    /// Port of `operator<<(std::ostream&, const Bitmap&)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Bitmap(width={}, height={}, channels={})",
            self.width, self.height, self.channels
        )
    }
}
