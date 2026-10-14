//! Port of `vl_imsmooth_f` (with `_vl_new_gaussian_fitler_f`), `vl_imgradient_f` and
//! `vl_imgradient_polar_f` from VLFeat's `imopv.c`, as ported in colmap-sharp's
//! `Feature/VLFeat/VlImOpv.Smooth.cs`. The covariant detector's scale space
//! ([`super::scalespace`]) smooths its levels with the first; the detector's second-moment and
//! orientation steps and COLMAP's covariant extractor take the gradients. Builds on
//! [`super::imopv::vl_imconvcol_vf`] (the separable column convolution).
//!
//! Tier A (exact). Pixels and filter taps are float; the filter's mass accumulates in float from
//! double taps, as the C's `T mass; mass += g + g` does. `exp` goes through
//! [`crate::math::fns`] (docs/CPP_DIVERGENCES.md entry 1). The gradients are VLFeat's
//! per-pixel one-sided (border) / central (inside) differences; each output pixel depends only
//! on its input neighbours, so the traversal is a plain double loop.
//!
//! Translation note: VLFeat's `vl_imsmooth_f` is called with `smoothed == image` (in place) and
//! with two levels of one octave buffer. Both cases are [`vl_imsmooth_f_within`], which reads
//! the source region of a buffer and writes the destination region of the same buffer; the
//! C's intermediate transposed buffer makes that safe in either case.

use super::imopv::{vl_imconvcol_vf, ConvColParams, VL_PAD_BY_CONTINUITY, VL_TRANSPOSE};
use super::mathop::{vl_fast_atan2_f, vl_fast_sqrt_f, vl_mod_2pi_f, VL_PI};
use crate::math::fns;

/// The region of a buffer an image occupies: `width` x `height` pixels, rows `stride` apart,
/// starting at `offset`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageRegion {
    /// Where pixel (0, 0) is.
    pub offset: usize,
    /// Pixels per row.
    pub width: usize,
    /// Rows.
    pub height: usize,
    /// Elements between the starts of consecutive rows.
    pub stride: usize,
}

/// Port of `_vl_new_gaussian_fitler_f`: a normalized float Gaussian of half-width
/// `ceil(3 * sigma)`, `2 * half_width + 1` taps.
pub fn vl_new_gaussian_filter_f(sigma: f64) -> Vec<f32> {
    // `vl_size width = vl_ceil_d(sigma * 3.0)`: a double-to-unsigned conversion.
    let width = (sigma * 3.0).ceil() as usize;
    let mut filter = vec![0.0f32; 2 * width + 1];
    let mut mass = 1.0f32;
    filter[width] = 1.0;
    for i in 1..=width {
        let x = i as f64 / sigma;
        let g = fns::exp(-0.5 * x * x);
        // `mass += g + g`: float mass promoted to double, the sum rounded back to float.
        mass = (f64::from(mass) + (g + g)) as f32;
        filter[width - i] = g as f32;
        filter[width + i] = g as f32;
    }
    for tap in &mut filter {
        *tap /= mass;
    }
    filter
}

/// The two filters of `vl_imsmooth_f` (the y filter is the x filter when the sigmas are equal).
fn smoothing_filters(sigmax: f64, sigmay: f64) -> (Vec<f32>, Vec<f32>) {
    let filterx = vl_new_gaussian_filter_f(sigmax);
    let filtery = if sigmax == sigmay {
        filterx.clone()
    } else {
        vl_new_gaussian_filter_f(sigmay)
    };
    (filterx, filtery)
}

/// `vl_imconvcol_vf`'s support arguments for a filter of `size` taps.
fn support(size: usize) -> (isize, isize) {
    let half = (size as isize - 1) / 2;
    (-half, half)
}

/// The first pass of `vl_imsmooth_f`: the y filter over `src`, transposed into a fresh buffer.
fn smooth_columns(src: &[f32], region: ImageRegion, filtery: &[f32]) -> Vec<f32> {
    let mut buffer = vec![0.0f32; region.width * region.height];
    let (filt_begin, filt_end) = support(filtery.len());
    vl_imconvcol_vf(
        &mut buffer,
        src,
        filtery,
        &ConvColParams {
            dst_offset: 0,
            dst_stride: region.height,
            src_offset: region.offset,
            src_width: region.width,
            src_height: region.height,
            src_stride: region.stride,
            filt_begin,
            filt_end,
            step: 1,
            flags: VL_PAD_BY_CONTINUITY | VL_TRANSPOSE,
        },
    );
    buffer
}

/// The second pass of `vl_imsmooth_f`: the x filter over the transposed buffer, transposed
/// back into `dst`.
fn smooth_rows(
    dst: &mut [f32],
    dst_offset: usize,
    dst_stride: usize,
    buffer: &[f32],
    width: usize,
    height: usize,
    filterx: &[f32],
) {
    let (filt_begin, filt_end) = support(filterx.len());
    vl_imconvcol_vf(
        dst,
        buffer,
        filterx,
        &ConvColParams {
            dst_offset,
            dst_stride,
            src_offset: 0,
            src_width: height,
            src_height: width,
            src_stride: height,
            filt_begin,
            filt_end,
            step: 1,
            flags: VL_PAD_BY_CONTINUITY | VL_TRANSPOSE,
        },
    );
}

/// Port of `vl_imsmooth_f`: separable Gaussian smoothing (standard deviations `sigmax`,
/// `sigmay`) with padding by continuity, from the `src` region into `dst` (rows `dst_stride`
/// apart from `dst_offset`).
pub fn vl_imsmooth_f(
    dst: &mut [f32],
    dst_offset: usize,
    dst_stride: usize,
    src: &[f32],
    region: ImageRegion,
    sigmax: f64,
    sigmay: f64,
) {
    let (filterx, filtery) = smoothing_filters(sigmax, sigmay);
    let buffer = smooth_columns(src, region, &filtery);
    smooth_rows(
        dst,
        dst_offset,
        dst_stride,
        &buffer,
        region.width,
        region.height,
        &filterx,
    );
}

/// [`vl_imsmooth_f`] with source and destination in one buffer (VLFeat's in-place call, or two
/// levels of one octave): smooths the `src` region of `data` into `data` at `dst_offset`.
pub fn vl_imsmooth_f_within(
    data: &mut [f32],
    dst_offset: usize,
    dst_stride: usize,
    src: ImageRegion,
    sigmax: f64,
    sigmay: f64,
) {
    let (filterx, filtery) = smoothing_filters(sigmax, sigmay);
    let buffer = smooth_columns(data, src, &filtery);
    smooth_rows(
        data, dst_offset, dst_stride, &buffer, src.width, src.height, &filterx,
    );
}

/// The difference VLFeat's gradient functions take at (x, y): `(next - previous) / 2` inside,
/// one-sided on the first and last row/column. `0.5 * (a - b)` promotes the float difference
/// to double; halving is exact, so the float result is the same.
fn gradient(image: &[f32], r: ImageRegion, x: usize, y: usize) -> (f32, f32) {
    let p = r.offset + y * r.stride + x;
    let gx = if x == 0 {
        image[p + 1] - image[p]
    } else if x == r.width - 1 {
        image[p] - image[p - 1]
    } else {
        (0.5 * f64::from(image[p + 1] - image[p - 1])) as f32
    };
    let gy = if y == 0 {
        image[p + r.stride] - image[p]
    } else if y == r.height - 1 {
        image[p] - image[p - r.stride]
    } else {
        (0.5 * f64::from(image[p + r.stride] - image[p - r.stride])) as f32
    };
    (gx, gy)
}

/// Where a gradient image puts pixel (x, y): `offset + y * height_stride + x * width_stride`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GradientLayout {
    /// Offset of the first output in the first output buffer.
    pub first_offset: usize,
    /// Offset of the first output in the second output buffer.
    pub second_offset: usize,
    /// `gradWidthStride`: elements between horizontally adjacent outputs.
    pub width_stride: usize,
    /// `gradHeightStride`: elements between vertically adjacent outputs.
    pub height_stride: usize,
}

/// Port of `vl_imgradient_f`: x and y derivatives of the `image` region, one-sided on the
/// border and central inside. Requires an image at least 2 pixels wide and high.
pub fn vl_imgradient_f(
    x_gradient: &mut [f32],
    y_gradient: &mut [f32],
    layout: GradientLayout,
    image: &[f32],
    region: ImageRegion,
) {
    for y in 0..region.height {
        for x in 0..region.width {
            let (gx, gy) = gradient(image, region, x, y);
            let o = y * layout.height_stride + x * layout.width_stride;
            x_gradient[layout.first_offset + o] = gx;
            y_gradient[layout.second_offset + o] = gy;
        }
    }
}

/// Port of `vl_imgradient_polar_f`: the gradient's modulus (`vl_fast_sqrt_f`) and angle
/// (`vl_fast_atan2_f`, wrapped to `[0, 2*pi]`), laid out like [`vl_imgradient_f`].
pub fn vl_imgradient_polar_f(
    modulus: &mut [f32],
    angle: &mut [f32],
    layout: GradientLayout,
    image: &[f32],
    region: ImageRegion,
) {
    for y in 0..region.height {
        for x in 0..region.width {
            let (gx, gy) = gradient(image, region, x, y);
            let o = y * layout.height_stride + x * layout.width_stride;
            modulus[layout.first_offset + o] = vl_fast_sqrt_f(gx * gx + gy * gy);
            // `vl_fast_atan2_f(gy, gx) + 2*VL_PI` is a double, rounded to float by the call.
            angle[layout.second_offset + o] =
                vl_mod_2pi_f((f64::from(vl_fast_atan2_f(gy, gx)) + 2.0 * VL_PI) as f32);
        }
    }
}
