//! `Bitmap::Rescale` and `Bitmap::Thumbnail` of `colmap/sensor/bitmap.cc`. Part of
//! [`super::Bitmap`]; port of colmap-sharp's `ColmapSharp/Sensor/BitmapResize.cs`.
//!
//! COLMAP resamples with OpenImageIO's `ImageBufAlgo::resize` (a "triangle" filter for
//! `kBilinear`, "box" for `kBox`), which is native and not ported. This is a resampler written
//! (by colmap-sharp) to the model OIIO's output follows, found by probing pycolmap 4.2.0
//! (`oracle/fixture_bitmap_rescale.py`): each destination pixel center maps to source
//! coordinates by the size ratio; the filter (triangle of radius 1, box of radius 1/2, a box
//! including samples exactly on its edge) is widened by the ratio when downsampling, so
//! downsampling antialiases; source samples beyond the border repeat the edge pixel (clamp)
//! and still count toward the weight total; the filter is separable; the result is rounded
//! half away from zero to a byte. Accumulation is in double.
//!
//! Not Tier A (`docs/CPP_DIVERGENCES.md`, entry 121). Bilinear (the default, used by
//! `Thumbnail`) is Tier B: OIIO accumulates in float, so a sum within float rounding of a
//! half-integer can round to the neighboring gray level; `tests/sensor/rust_only_bitmap_rescale_oracle.rs`
//! pins it against pycolmap within one gray level. Box matches OIIO on every probe without a
//! tie, but when a source pixel center lies exactly on the box edge at a non-integer ratio
//! (23 -> 10 pixels), OIIO includes or excludes it by rules not reproduced here.
//! `bitmap_test.cc` pins only the output dimensions and that the two filters differ.
//! `Thumbnail`'s size arithmetic is exact.

use std::collections::VecDeque;

use super::{buffer_len, Bitmap};
use crate::{check_gt, Result};

/// Port of `Bitmap::RescaleFilter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RescaleFilter {
    /// `kBilinear`: triangle (tent) filter; OIIO's "triangle".
    #[default]
    Bilinear,
    /// `kBox`: box filter; OIIO's "box".
    Box,
}

/// Normalized filter weights of one destination index along an axis: the first contributing
/// source index and the weights of the consecutive source indices from it.
pub(super) struct AxisWeights {
    pub(super) first: usize,
    pub(super) weights: Vec<f64>,
}

impl Bitmap {
    /// Port of `Bitmap::Rescale`: resample the image to the new dimensions in place.
    ///
    /// Two separable passes. The horizontal pass is done one source row at a time and only
    /// for the rows the vertical filter of the current output row needs: rows are kept in a
    /// window that slides down the image (the filters' first rows never decrease), so the
    /// double-precision intermediate is a few rows, not `height x new_width`. The arithmetic
    /// is the same as a full intermediate's (`src/sensor/bitmap/tests.rs` pins it bit for bit
    /// and `tests/sensor_alloc.rs` bounds the memory).
    ///
    /// # Panics
    ///
    /// On negative or overflowing new dimensions (as [`Bitmap::new`]).
    pub fn rescale(&mut self, new_width: i32, new_height: i32, filter: RescaleFilter) {
        let channels = self.channels as usize;
        let mut new_data = vec![0u8; buffer_len(new_width, new_height, self.channels)];
        if !new_data.is_empty() && !self.data.is_empty() {
            let (width, height) = (self.width as usize, self.height as usize);
            let new_width_u = new_width as usize;
            let column_weights = compute_resample_weights(width, new_width_u, filter);
            let row_weights = compute_resample_weights(height, new_height as usize, filter);
            let row_len = new_width_u * channels;

            // Horizontally filtered source rows [window_first, window_first + window.len()).
            let mut window: VecDeque<Vec<f64>> = VecDeque::new();
            let mut window_first = 0usize;
            // Row buffers dropped from the front, reused for new rows.
            let mut spare: Vec<Vec<f64>> = Vec::new();

            for (y, row) in row_weights.iter().enumerate() {
                // Drop rows above this output row's filter.
                while window_first < row.first && !window.is_empty() {
                    spare.extend(window.pop_front());
                    window_first += 1;
                }
                if window.is_empty() {
                    window_first = row.first;
                }
                // Horizontal pass for the rows below the window that the filter reaches.
                while window_first + window.len() < row.first + row.weights.len() {
                    let source_y = window_first + window.len();
                    let mut buffer = spare.pop().unwrap_or_default();
                    buffer.resize(row_len, 0.0);
                    let source_row = &self.data[source_y * width * channels..];
                    for (x, column) in column_weights.iter().enumerate() {
                        for c in 0..channels {
                            let mut sum = 0.0;
                            for (i, &weight) in column.weights.iter().enumerate() {
                                let source = (column.first + i) * channels + c;
                                sum += weight * f64::from(source_row[source]);
                            }
                            buffer[x * channels + c] = sum;
                        }
                    }
                    window.push_back(buffer);
                }

                // Vertical pass.
                let offset = row.first - window_first;
                for x in 0..new_width_u {
                    for c in 0..channels {
                        let mut sum = 0.0;
                        for (i, &weight) in row.weights.iter().enumerate() {
                            sum += weight * window[offset + i][x * channels + c];
                        }
                        // f64::round is half away from zero; the clamp keeps the byte range.
                        new_data[(y * new_width_u + x) * channels + c] =
                            sum.round().clamp(0.0, 255.0) as u8;
                    }
                }
            }
        }

        self.width = new_width;
        self.height = new_height;
        self.data = new_data;
    }

    /// Port of `Bitmap::Thumbnail`: downscale in place so that neither dimension exceeds
    /// `max_image_size`, preserving the aspect ratio. Returns the scale factor applied (1 if
    /// the image already fits).
    pub fn thumbnail(&mut self, max_image_size: i32, filter: RescaleFilter) -> Result<f64> {
        check_gt!(max_image_size, 0);
        if self.width <= max_image_size && self.height <= max_image_size {
            return Ok(1.0);
        }
        // Fit the down-sampled version exactly into the max dimensions.
        let scale = f64::from(max_image_size) / f64::from(self.width.max(self.height));
        self.rescale(
            (f64::from(self.width) * scale).round() as i32,
            (f64::from(self.height) * scale).round() as i32,
            filter,
        );
        Ok(scale)
    }
}

pub(super) fn compute_resample_weights(
    source_size: usize,
    dest_size: usize,
    filter: RescaleFilter,
) -> Vec<AxisWeights> {
    let ratio = source_size as f64 / dest_size as f64;
    // Filter widths in destination pixels (triangle 2, box 1), widened to source pixels when
    // downsampling so every source pixel contributes.
    let scale = ratio.max(1.0);
    let radius = match filter {
        RescaleFilter::Bilinear => 1.0,
        RescaleFilter::Box => 0.5,
    } * scale;
    let last_index = source_size as i64 - 1;

    (0..dest_size)
        .map(|d| {
            let center = (d as f64 + 0.5) * ratio;
            // Source pixels whose centers can fall inside the filter, including virtual ones
            // beyond the border, which fold onto the edge pixel.
            let first = (center - radius - 0.5).floor() as i64;
            let last = (center + radius - 0.5).ceil() as i64;
            let folded_first = first.clamp(0, last_index);
            let folded_last = last.clamp(0, last_index);
            let mut weights = vec![0.0f64; (folded_last - folded_first + 1) as usize];
            let mut total = 0.0;
            for s in first..=last {
                let distance = (s as f64 + 0.5 - center).abs() / scale;
                let weight = match filter {
                    RescaleFilter::Bilinear => (1.0 - distance).max(0.0),
                    RescaleFilter::Box => {
                        if distance <= 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                };
                weights[(s.clamp(0, last_index) - folded_first) as usize] += weight;
                total += weight;
            }
            for weight in &mut weights {
                *weight /= total;
            }
            AxisWeights {
                first: folded_first as usize,
                weights,
            }
        })
        .collect()
}
