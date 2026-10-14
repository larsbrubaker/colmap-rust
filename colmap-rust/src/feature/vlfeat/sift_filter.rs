//! Port of VLFeat's SIFT filter (`thirdparty/VLFeat/sift.h` / `sift.c`), the CPU SIFT COLMAP
//! runs: the filter state ([`VlSiftFilter`], `VlSiftFilt`) and the Gaussian scale space
//! (`vl_sift_new`, `vl_sift_process_first_octave` / `_next_octave` and their helpers).
//! [`super::sift_detect`] has the DoG extremum detector, [`super::sift_descriptor`] the
//! gradients, orientations and descriptors. Port of colmap-sharp's
//! `Feature/VLFeat/VlSiftFilter.cs`. Tests: `tests/feature/rust_only_vl_sift_filter.rs`.
//!
//! Tier A (exact) by construction: pixels are float (`vl_sift_pix`), filter parameters double,
//! and every mixed expression keeps C's promotion points. `pow` / `powf` / `exp` / `log` /
//! `sqrt` go through [`crate::math::fns`] (docs/CPP_DIVERGENCES.md entry 1).
//!
//! Translation notes:
//! - VLFeat's pointers into the octave / temp buffers become slices plus offsets; the buffers
//!   keep VLFeat's sizes and layouts, including the in-place smoothing and the downsample that
//!   writes the next octave's base over the current octave's first level. The gradient buffer
//!   holds only the `S` levels `update_gradient` fills (VLFeat allocates `S + 2`; the extra
//!   two are never written or read).
//! - VLFeat returns `VL_ERR_EOF` when there is no further octave; here the process methods
//!   return `Ok(false)`.
//! - Cancellation (not in VLFeat): the process methods and `detect` take a [`CancelToken`]
//!   and check it between scale levels, returning `Err` with [`ErrorKind::Cancelled`]. The
//!   checks never touch the numbers.
//! - `vl_sift_process_first_octave` resets the gradient cache's octave (`grad_o`), which
//!   VLFeat resets only in `vl_sift_new`, so a filter reused for a new image never takes the
//!   previous image's gradient (docs/CPP_DIVERGENCES.md entry 201).
//!
//! [`ErrorKind::Cancelled`]: crate::util::check::ErrorKind::Cancelled

use super::imopv::{vl_imconvcol_vf, ConvColParams, VL_PAD_BY_CONTINUITY, VL_TRANSPOSE};
use super::mathop::VL_LOG_OF_2;
use crate::math::fns;
use crate::util::cancellation::CancelToken;
use crate::util::check::Result;

/// `NBO`: number of orientation bins of the descriptor.
pub(super) const NBO: i32 = 8;
/// `NBP`: number of spatial bins per side of the descriptor.
pub(super) const NBP: i32 = 4;

/// Port of `VlSiftKeypoint`: a detected SIFT keypoint.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VlSiftKeypoint {
    /// Octave index.
    pub o: i32,
    /// Integer x in the octave.
    pub ix: i32,
    /// Integer y in the octave.
    pub iy: i32,
    /// Integer scale level.
    pub is: i32,
    /// x in image coordinates (pixel centers at integers).
    pub x: f32,
    /// y in image coordinates.
    pub y: f32,
    /// Fractional scale level.
    pub s: f32,
    /// Scale (sigma) in image pixels.
    pub sigma: f32,
}

/// Port of `VlSiftFilt` and the `vl_sift_*` functions.
#[derive(Debug, Clone)]
pub struct VlSiftFilter {
    pub(super) sigman: f64,
    pub(super) sigma0: f64,
    pub(super) sigmak: f64,
    pub(super) dsigma0: f64,

    pub(super) width: i32,
    pub(super) height: i32,
    pub(super) o_num: i32,
    pub(super) s_num: i32,
    pub(super) o_min: i32,
    pub(super) s_min: i32,
    pub(super) s_max: i32,
    pub(super) o_cur: i32,
    pub(super) octave_width: i32,
    pub(super) octave_height: i32,

    pub(super) temp: Vec<f32>,
    pub(super) octave: Vec<f32>,
    pub(super) dog: Vec<f32>,
    pub(super) grad: Vec<f32>,
    pub(super) grad_o: i32,

    gauss_filter: Vec<f32>,
    gauss_filter_sigma: f64,
    gauss_filter_width: i32,

    pub(super) keys: Vec<VlSiftKeypoint>,

    /// DoG peak threshold (`vl_sift_set_peak_thresh`).
    pub peak_thresh: f64,
    /// Edge threshold (`vl_sift_set_edge_thresh`).
    pub edge_thresh: f64,
    /// Descriptor norm threshold (`vl_sift_set_norm_thresh`); zero disables it.
    pub norm_thresh: f64,
    /// Descriptor magnification factor (`vl_sift_set_magnif`).
    pub magnif: f64,
    /// Gaussian window size of the descriptor, in spatial bins (`vl_sift_set_window_size`).
    pub window_size: f64,
}

/// `VL_SHIFT_LEFT`: a left shift for `n >= 0`, an arithmetic right shift otherwise.
fn shift_left(x: i32, n: i32) -> i32 {
    if n >= 0 {
        x << n
    } else {
        x >> -n
    }
}

/// `sift.c` defines `log2(x)` as `log(x) / VL_LOG_OF_2`.
fn log2(x: f64) -> f64 {
    fns::ln(x) / VL_LOG_OF_2
}

impl VlSiftFilter {
    /// Port of `vl_sift_new`. A negative `noctaves` selects the most octaves the image size
    /// allows.
    pub fn new(width: i32, height: i32, mut noctaves: i32, nlevels: i32, o_min: i32) -> Self {
        let w = shift_left(width, -o_min);
        let h = shift_left(height, -o_min);
        let nel = (w * h) as usize;

        if noctaves < 0 {
            noctaves = (log2(f64::from(width.min(height))).floor() - f64::from(o_min) - 3.0)
                .max(1.0) as i32;
        }

        let s_min = -1;
        let s_max = nlevels + 1;
        let sigmak = fns::pow(2.0, 1.0 / f64::from(nlevels));
        let sigma0 = 1.6 * sigmak;
        VlSiftFilter {
            sigman: 0.5,
            sigma0,
            sigmak,
            dsigma0: sigma0 * fns::sqrt(1.0 - 1.0 / (sigmak * sigmak)),
            width,
            height,
            o_num: noctaves,
            s_num: nlevels,
            o_min,
            s_min,
            s_max,
            o_cur: o_min,
            octave_width: 0,
            octave_height: 0,
            temp: vec![0.0; nel],
            octave: vec![0.0; nel * (s_max - s_min + 1) as usize],
            dog: vec![0.0; nel * (s_max - s_min) as usize],
            grad: vec![0.0; nel * 2 * (s_max - 2 - s_min) as usize],
            grad_o: o_min - 1,
            gauss_filter: Vec::new(),
            gauss_filter_sigma: 0.0,
            gauss_filter_width: 0,
            keys: Vec::new(),
            peak_thresh: 0.0,
            edge_thresh: 10.0,
            norm_thresh: 0.0,
            magnif: 3.0,
            window_size: f64::from(NBP / 2),
        }
    }

    /// Image width.
    pub fn width(&self) -> i32 {
        self.width
    }

    /// Image height.
    pub fn height(&self) -> i32 {
        self.height
    }

    /// Number of octaves (`O`).
    pub fn num_octaves(&self) -> i32 {
        self.o_num
    }

    /// Number of levels per octave (`S`).
    pub fn num_levels(&self) -> i32 {
        self.s_num
    }

    /// First octave index (`o_min`).
    pub fn octave_min(&self) -> i32 {
        self.o_min
    }

    /// The octave the filter currently holds (`o_cur`).
    pub fn current_octave(&self) -> i32 {
        self.o_cur
    }

    /// Width of the current octave.
    pub fn octave_width(&self) -> i32 {
        self.octave_width
    }

    /// Height of the current octave.
    pub fn octave_height(&self) -> i32 {
        self.octave_height
    }

    /// Smoothing of the pyramid base (`sigma0`).
    pub fn sigma0(&self) -> f64 {
        self.sigma0
    }

    /// The keypoints the last `detect` found (`vl_sift_get_keypoints`).
    pub fn keypoints(&self) -> &[VlSiftKeypoint] {
        &self.keys
    }

    /// Port of `vl_sift_process_first_octave`: builds the Gaussian scale space of the first
    /// octave of `im` (row-major, `width` x `height`). `Ok(false)` (`VL_ERR_EOF`) when there
    /// are no octaves.
    pub fn process_first_octave(&mut self, im: &[f32], cancel: &CancelToken) -> Result<bool> {
        cancel.check_cancelled()?;

        let width = self.width;
        let height = self.height;
        let o_min = self.o_min;

        self.o_cur = o_min;
        self.keys.clear();

        // Not in VLFeat, which resets grad_o only in vl_sift_new (entry 201).
        self.grad_o = o_min - 1;

        let w = shift_left(width, -self.o_cur);
        let h = shift_left(height, -self.o_cur);
        self.octave_width = w;
        self.octave_height = h;

        if self.o_num == 0 {
            return Ok(false);
        }

        // If the first octave has negative index, we upscale the image; if the first octave
        // has positive index, we downscale the image; if the first octave has index zero, we
        // just copy the image.
        let base = self.octave_offset(self.s_min);
        let (wu, hu) = (width as usize, height as usize);
        if o_min < 0 {
            // Double once.
            copy_and_upsample_rows(&mut self.temp, 0, &im[..wu * hu], wu, hu);
            copy_and_upsample_rows(
                &mut self.octave,
                base,
                &self.temp[..2 * wu * hu],
                hu,
                2 * wu,
            );

            // Double more.
            let mut o = -1;
            while o > o_min {
                let (ws, hs) = ((width << -o) as usize, (height << -o) as usize);
                let n = ws * hs;
                copy_and_upsample_rows(&mut self.temp, 0, &self.octave[base..base + n], ws, hs);
                copy_and_upsample_rows(&mut self.octave, base, &self.temp[..2 * n], ws, 2 * hs);
                cancel.check_cancelled()?;
                o -= 1;
            }
        } else if o_min > 0 {
            copy_and_downsample(&mut self.octave[base..], &im[..wu * hu], wu, hu, o_min);
        } else {
            self.octave[base..base + wu * hu].copy_from_slice(&im[..wu * hu]);
        }

        // Here we adjust the smoothing of the first level of the octave. The input image is
        // assumed to have nominal smoothing equal to sigman.
        let sa = self.sigma0 * fns::pow(self.sigmak, f64::from(self.s_min));
        let sb = self.sigman * fns::pow(2.0, f64::from(-o_min));

        if sa > sb {
            let sd = fns::sqrt(sa * sa - sb * sb);
            self.smooth(base, base, w, h, sd);
        }

        self.fill_octave(w, h, cancel)?;
        Ok(true)
    }

    /// Port of `vl_sift_process_next_octave`: downsamples to the next octave and builds its
    /// scale space. Clears the keypoints of the previous octave. `Ok(false)` (`VL_ERR_EOF`)
    /// after the last octave.
    pub fn process_next_octave(&mut self, cancel: &CancelToken) -> Result<bool> {
        if self.o_cur == self.o_min + self.o_num - 1 {
            return Ok(false);
        }

        // Retrieve base.
        let s_best = (self.s_min + self.s_num).min(self.s_max);
        let w = self.octave_width;
        let h = self.octave_height;
        let pt = self.octave_offset(s_best);
        let base = self.octave_offset(self.s_min);

        cancel.check_cancelled()?;

        // Next octave. The destination (the first level) lies before the source level and
        // advances no faster than it, so the in-place copy reads every value before it is
        // overwritten, as VLFeat's does.
        copy_and_downsample_in_place(&mut self.octave, base, pt, w as usize, h as usize, 1);

        self.o_cur += 1;
        self.keys.clear();
        let w = shift_left(self.width, -self.o_cur);
        let h = shift_left(self.height, -self.o_cur);
        self.octave_width = w;
        self.octave_height = h;

        // VLFeat uses powf here (unlike the first octave): sigmak and the level are rounded
        // to float and the power is computed in single precision.
        let sigmak_f = self.sigmak as f32;
        let sa = self.sigma0 * f64::from(fns::powf(sigmak_f, self.s_min as f32));
        let sb = self.sigma0 * f64::from(fns::powf(sigmak_f, (s_best - self.s_num) as f32));

        if sa > sb {
            let sd = fns::sqrt(sa * sa - sb * sb);
            let base = self.octave_offset(self.s_min);
            self.smooth(base, base, w, h, sd);
        }

        self.fill_octave(w, h, cancel)?;
        Ok(true)
    }

    /// Start of scale level `s` of the current octave (`vl_sift_get_octave`).
    pub(super) fn octave_offset(&self, s: i32) -> usize {
        (self.octave_width * self.octave_height * (s - self.s_min)) as usize
    }

    /// Smooths each level from the one below it (the "Fill octave" loop of both process
    /// calls).
    fn fill_octave(&mut self, w: i32, h: i32, cancel: &CancelToken) -> Result<()> {
        for s in self.s_min + 1..=self.s_max {
            cancel.check_cancelled()?;
            let sd = self.dsigma0 * fns::pow(self.sigmak, f64::from(s));
            let output = self.octave_offset(s);
            let input = self.octave_offset(s - 1);
            self.smooth(output, input, w, h, sd);
        }
        Ok(())
    }

    /// Port of `_vl_sift_smooth`: separable Gaussian smoothing of the octave level at `input`
    /// into the one at `output` (which may be the same), through the temp buffer.
    fn smooth(&mut self, output: usize, input: usize, width: i32, height: i32, sigma: f64) {
        // Prepare Gaussian filter.
        if self.gauss_filter_sigma != sigma {
            let mut acc = 0.0f32;
            self.gauss_filter_width = (4.0 * sigma).ceil().max(1.0) as i32;
            self.gauss_filter_sigma = sigma;
            let n = (2 * self.gauss_filter_width + 1) as usize;
            self.gauss_filter = vec![0.0; n];
            for j in 0..n {
                let d = (j as i32 - self.gauss_filter_width) as f32 / sigma as f32;
                self.gauss_filter[j] = fns::exp(-0.5 * f64::from(d * d)) as f32;
                acc += self.gauss_filter[j];
            }
            for g in self.gauss_filter.iter_mut() {
                *g /= acc;
            }
        }

        let (wu, hu) = (width as usize, height as usize);
        if self.gauss_filter_width == 0 {
            self.octave.copy_within(input..input + wu * hu, output);
            return;
        }

        let half = self.gauss_filter_width as isize;
        let first = ConvColParams {
            dst_offset: 0,
            dst_stride: hu,
            src_offset: input,
            src_width: wu,
            src_height: hu,
            src_stride: wu,
            filt_begin: -half,
            filt_end: half,
            step: 1,
            flags: VL_PAD_BY_CONTINUITY | VL_TRANSPOSE,
        };
        vl_imconvcol_vf(&mut self.temp, &self.octave, &self.gauss_filter, &first);

        let second = ConvColParams {
            dst_offset: output,
            dst_stride: wu,
            src_offset: 0,
            src_width: hu,
            src_height: wu,
            src_stride: hu,
            ..first
        };
        vl_imconvcol_vf(&mut self.octave, &self.temp, &self.gauss_filter, &second);
    }
}

/// Port of `copy_and_upsample_rows`: upsamples the rows of `src` (`width` x `height`) by
/// linear interpolation and transposes, writing `2 * width` rows of `height` at `dst_offset`.
fn copy_and_upsample_rows(
    dst: &mut [f32],
    dst_offset: usize,
    src: &[f32],
    width: usize,
    height: usize,
) {
    let mut s = 0usize;
    let mut d = dst_offset;
    for _ in 0..height {
        let mut a = src[s];
        let mut b = a;
        s += 1;
        for _ in 0..width.saturating_sub(1) {
            b = src[s];
            s += 1;
            dst[d] = a;
            d += height;
            // 0.5 * (a + b) promotes the float sum to double; halving is exact.
            dst[d] = (0.5 * f64::from(a + b)) as f32;
            d += height;
            a = b;
        }
        dst[d] = b;
        d += height;
        dst[d] = b;
        d += height;
        // d += 1 - width * 2 * height, rewinding to the next column.
        d = d + 1 - width * 2 * height;
    }
}

/// Port of `copy_and_downsample` between two buffers: keeps every `2^d`-th pixel of every
/// `2^d`-th row of `src` (`width` x `height`).
fn copy_and_downsample(dst: &mut [f32], src: &[f32], width: usize, height: usize, d: i32) {
    let step = 1usize << d;
    let mut di = 0usize;
    for y in (0..height).step_by(step) {
        let mut srcrow = y * width;
        let mut x = 0usize;
        while x + (step - 1) < width {
            dst[di] = src[srcrow];
            di += 1;
            srcrow += step;
            x += step;
        }
    }
}

/// [`copy_and_downsample`] within one buffer, from `src_offset` to `dst_offset`
/// (`dst_offset <= src_offset`), as `vl_sift_process_next_octave` calls it on the octave.
fn copy_and_downsample_in_place(
    buf: &mut [f32],
    dst_offset: usize,
    src_offset: usize,
    width: usize,
    height: usize,
    d: i32,
) {
    let step = 1usize << d;
    let mut di = dst_offset;
    for y in (0..height).step_by(step) {
        let mut srcrow = src_offset + y * width;
        let mut x = 0usize;
        while x + (step - 1) < width {
            buf[di] = buf[srcrow];
            di += 1;
            srcrow += step;
            x += step;
        }
    }
}
