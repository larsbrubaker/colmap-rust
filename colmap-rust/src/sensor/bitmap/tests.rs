//! Unit tests of `bitmap/` that need private access: the row-window resampler of `resize.rs`
//! against the straightforward two-pass version (a full `height x new_width` intermediate),
//! which is kept here as the reference. The two must agree bit for bit: the window only
//! changes which horizontal rows are held in memory, not any arithmetic.

use super::resize::{compute_resample_weights, RescaleFilter};
use super::Bitmap;

/// The full-intermediate two-pass resampler, the form colmap-sharp's `BitmapResize.cs` uses.
fn reference_rescale(
    data: &[u8],
    (width, height, channels): (usize, usize, usize),
    (new_width, new_height): (usize, usize),
    filter: RescaleFilter,
) -> Vec<u8> {
    let column_weights = compute_resample_weights(width, new_width, filter);
    let row_weights = compute_resample_weights(height, new_height, filter);
    let mut horizontal = vec![0.0f64; height * new_width * channels];
    for y in 0..height {
        for (x, column) in column_weights.iter().enumerate() {
            for c in 0..channels {
                let mut sum = 0.0;
                for (i, &weight) in column.weights.iter().enumerate() {
                    sum += weight * f64::from(data[(y * width + column.first + i) * channels + c]);
                }
                horizontal[(y * new_width + x) * channels + c] = sum;
            }
        }
    }
    let mut out = vec![0u8; new_width * new_height * channels];
    for (y, row) in row_weights.iter().enumerate() {
        for x in 0..new_width {
            for c in 0..channels {
                let mut sum = 0.0;
                for (i, &weight) in row.weights.iter().enumerate() {
                    sum += weight * horizontal[((row.first + i) * new_width + x) * channels + c];
                }
                out[(y * new_width + x) * channels + c] = sum.round().clamp(0.0, 255.0) as u8;
            }
        }
    }
    out
}

#[test]
fn rust_only_row_window_rescale_matches_full_intermediate_bit_for_bit() {
    let shapes = [
        (23, 17, 10, 7),
        (17, 11, 40, 29),
        (30, 20, 13, 20),
        (9, 7, 4, 11),
        (4, 4, 1, 1),
        (1, 1, 5, 3),
        (64, 48, 7, 5),
    ];
    for as_rgb in [false, true] {
        for filter in [RescaleFilter::Bilinear, RescaleFilter::Box] {
            for &(width, height, new_width, new_height) in &shapes {
                let mut bitmap = Bitmap::new(width, height, as_rgb);
                for (i, v) in bitmap.row_major_data_mut().iter_mut().enumerate() {
                    *v = (i * 97 % 256) as u8;
                }
                let channels = bitmap.channels() as usize;
                let expected = reference_rescale(
                    bitmap.row_major_data(),
                    (width as usize, height as usize, channels),
                    (new_width as usize, new_height as usize),
                    filter,
                );
                bitmap.rescale(new_width, new_height, filter);
                assert_eq!(
                    bitmap.row_major_data(),
                    expected.as_slice(),
                    "{width}x{height} -> {new_width}x{new_height} {filter:?} rgb={as_rgb}"
                );
            }
        }
    }
}
