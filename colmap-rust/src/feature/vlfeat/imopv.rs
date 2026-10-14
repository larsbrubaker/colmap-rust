//! Port of `vl_imconvcol_vf` from VLFeat's `imopv.c`: the separable column convolution SIFT's
//! Gaussian smoothing runs twice per level ([`super::sift_filter`]). Only the float
//! instantiation is ported; it is the only one COLMAP's SIFT calls. Port of colmap-sharp's
//! `Feature/VLFeat/VlImOpv.cs` (its `VlImOpv.Smooth.cs`, used by the covariant detector only,
//! is not ported yet).
//!
//! Tier A (exact). This is the scalar (non-SSE2) path, which is what COLMAP compiles on arm64.
//! Each output column is one sequential float accumulation in VLFeat's order. The columns run
//! in order (colmap-sharp runs them in parallel; the result is the same).

/// `VL_PAD_BY_ZERO`.
pub const VL_PAD_BY_ZERO: u32 = 0x0;
/// `VL_PAD_BY_CONTINUITY`.
pub const VL_PAD_BY_CONTINUITY: u32 = 0x1;
/// `VL_PAD_MASK`.
pub const VL_PAD_MASK: u32 = 0x3;
/// `VL_TRANSPOSE`.
pub const VL_TRANSPOSE: u32 = 0x1 << 2;

/// The geometry of one [`vl_imconvcol_vf`] call: VLFeat's pointer arguments become slices
/// plus element offsets and strides.
#[derive(Debug, Clone, Copy)]
pub struct ConvColParams {
    /// Where the output starts in `dst`.
    pub dst_offset: usize,
    /// `dst_stride`.
    pub dst_stride: usize,
    /// Where the input starts in `src`.
    pub src_offset: usize,
    /// `src_width`.
    pub src_width: usize,
    /// `src_height`.
    pub src_height: usize,
    /// `src_stride`.
    pub src_stride: usize,
    /// `filt_begin`: the filter's support starts here (usually `-half_width`).
    pub filt_begin: isize,
    /// `filt_end`: the filter's support ends here, inclusive.
    pub filt_end: isize,
    /// `step`: the row subsampling.
    pub step: usize,
    /// `flags`: padding mode and [`VL_TRANSPOSE`].
    pub flags: u32,
}

/// Port of `vl_imconvcol_vf`: convolves each column of the input with `filt` (support
/// `[filt_begin, filt_end]`, `filt[0]` is the tap at `filt_begin`), subsampling rows by
/// `step`. With [`VL_TRANSPOSE`] column x of the input becomes row x of the output.
pub fn vl_imconvcol_vf(dst: &mut [f32], src: &[f32], filt: &[f32], p: &ConvColParams) {
    let transp = p.flags & VL_TRANSPOSE != 0;
    let zeropad = p.flags & VL_PAD_MASK == VL_PAD_BY_ZERO;
    let src_height = p.src_height as isize;
    let src_stride = p.src_stride as isize;
    let src_offset = p.src_offset as isize;
    let filt_begin = p.filt_begin;
    let filt_end = p.filt_end;

    // Where column x starts in dst: VLFeat walks dst with a pointer and rewinds it after each
    // column; the closed form gives the same positions.
    let dst_column_step = if transp { p.dst_stride } else { 1 };
    let dst_row_step = if transp { 1 } else { p.dst_stride };

    // Let filt point to the last sample of the filter.
    let filt_last = filt_end - filt_begin;

    for x in 0..p.src_width {
        let xi = x as isize;
        let mut d = p.dst_offset + x * dst_column_step;
        let mut y: isize = 0;
        while y < src_height {
            let mut acc = 0.0f32;
            let mut v = 0.0f32;
            let mut filti = filt_last;
            let mut stop = filt_end - y;
            let mut srci = src_offset + xi - stop * src_stride;

            // CHUNK_A: samples above the image, padded.
            if stop > 0 {
                v = if zeropad {
                    0.0
                } else {
                    src[(src_offset + xi) as usize]
                };
                while filti > filt_last - stop {
                    let c = filt[filti as usize];
                    filti -= 1;
                    acc += v * c;
                    srci += src_stride;
                }
            }

            // CHUNK_B: samples inside the image.
            stop = filt_end - filt_begin.max(y - src_height + 1) + 1;
            while filti > filt_last - stop {
                v = src[srci as usize];
                let c = filt[filti as usize];
                filti -= 1;
                acc += v * c;
                srci += src_stride;
            }

            // CHUNK_C: samples below the image, padded with the last value (or zero).
            if zeropad {
                v = 0.0;
            }
            stop = filt_end - filt_begin + 1;
            while filti > filt_last - stop {
                let c = filt[filti as usize];
                filti -= 1;
                acc += v * c;
            }

            dst[d] = acc;
            d += dst_row_step;
            y += p.step as isize;
        }
    }
}
