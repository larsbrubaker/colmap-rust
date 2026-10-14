//! Port of `update_gradient`, `vl_sift_calc_keypoint_orientations`, `normalize_histogram` and
//! `vl_sift_calc_keypoint_descriptor` from VLFeat's `sift.c`. Part of [`VlSiftFilter`] (scale
//! space in [`super::sift_filter`], detection in [`super::sift_detect`]). Port of
//! colmap-sharp's `Feature/VLFeat/VlSiftFilter.Descriptor.cs`; its `vl_sift_calc_raw_descriptor`
//! (used only by the covariant extractor) is [`super::sift_raw_descriptor`], and
//! `vl_sift_keypoint_init` is not ported because nothing in COLMAP calls it.
//!
//! Tier A (exact). Gradients and descriptor bins are float, orientation histograms double;
//! every conversion between them is written out where the C converts implicitly (a float
//! local assigned a double expression rounds once, at the assignment). `sin` / `cos` / `sqrt`
//! go through [`crate::math::fns`].

use super::mathop::{
    fast_expn, vl_fast_atan2_f, vl_fast_sqrt_f, vl_floor_d, vl_floor_f, vl_mod_2pi_f, VL_EPSILON_D,
    VL_EPSILON_F, VL_PI,
};
use super::sift_filter::{VlSiftFilter, VlSiftKeypoint, NBO, NBP};
use crate::math::fns;

/// Length of a SIFT descriptor (`NBO * NBP * NBP`).
pub const VL_SIFT_DESCRIPTOR_LENGTH: usize = (NBO * NBP * NBP) as usize;

impl VlSiftFilter {
    /// Port of `vl_sift_calc_keypoint_orientations`: up to four dominant gradient
    /// orientations of `k`, written to `angles`. Returns how many were found; zero when the
    /// keypoint is not in the current octave or out of bounds.
    pub fn calc_keypoint_orientations(
        &mut self,
        angles: &mut [f64; 4],
        k: &VlSiftKeypoint,
    ) -> usize {
        const WINF: f64 = 1.5;
        let xper = fns::pow(2.0, f64::from(self.o_cur));

        let w = self.octave_width;
        let h = self.octave_height;
        const XO: i32 = 2;
        let yo = 2 * w;
        let so = 2 * w * h;
        let x = f64::from(k.x) / xper;
        let y = f64::from(k.y) / xper;
        let sigma = f64::from(k.sigma) / xper;

        // (int)(x + 0.5): truncation; `as` saturates where C is undefined.
        let xi = (x + 0.5) as i32;
        let yi = (y + 0.5) as i32;
        let si = k.is;

        let sigmaw = WINF * sigma;
        let big_w = (3.0 * sigmaw).floor().max(1.0) as i32;

        const NBINS: i32 = 36;
        let mut hist = [0.0f64; NBINS as usize];

        // Skip if the keypoint octave is not current.
        if k.o != self.o_cur {
            return 0;
        }

        // Skip the keypoint if it is out of bounds.
        if xi < 0
            || xi > w - 1
            || yi < 0
            || yi > h - 1
            || si < self.s_min + 1
            || si > self.s_max - 2
        {
            return 0;
        }

        self.update_gradient();

        // Compute orientation histogram.
        let pt = XO * xi + yo * yi + so * (si - self.s_min - 1);

        for ys in (-big_w).max(-yi)..=big_w.min(h - 1 - yi) {
            for xs in (-big_w).max(-xi)..=big_w.min(w - 1 - xi) {
                let dx = f64::from(xi + xs) - x;
                let dy = f64::from(yi + ys) - y;
                let r2 = dx * dx + dy * dy;

                // Limit to a circular window.
                if r2 >= f64::from(big_w * big_w) + 0.6 {
                    continue;
                }

                let wgt = fast_expn(r2 / (2.0 * sigmaw * sigmaw));
                let g = (pt + xs * XO + ys * yo) as usize;
                let m = f64::from(self.grad[g]);
                let ang = f64::from(self.grad[g + 1]);
                let fbin = f64::from(NBINS) * ang / (2.0 * VL_PI);

                // VL_SIFT_BILINEAR_ORIENTATIONS is defined: split the vote between two bins.
                let bin = vl_floor_d(fbin - 0.5) as i32;
                let rbin = fbin - f64::from(bin) - 0.5;
                hist[((bin + NBINS) % NBINS) as usize] += (1.0 - rbin) * m * wgt;
                hist[((bin + 1) % NBINS) as usize] += rbin * m * wgt;
            }
        }

        // Smooth histogram.
        let n = NBINS as usize;
        for _ in 0..6 {
            let mut prev = hist[n - 1];
            let first = hist[0];
            for i in 0..n - 1 {
                let newh = (prev + hist[i] + hist[(i + 1) % n]) / 3.0;
                prev = hist[i];
                hist[i] = newh;
            }
            hist[n - 1] = (prev + hist[n - 1] + first) / 3.0;
        }

        // Find the histogram maximum (VL_MAX: `a > b ? a : b`).
        let mut maxh = 0.0f64;
        for &v in &hist {
            maxh = if maxh > v { maxh } else { v };
        }

        // Find peaks within 80% from max.
        let mut nangles = 0usize;
        for i in 0..n {
            let h0 = hist[i];
            let hm = hist[(i + n - 1) % n];
            let hp = hist[(i + 1) % n];

            // Is this a peak?
            if h0 > 0.8 * maxh && h0 > hm && h0 > hp {
                // Quadratic interpolation.
                let di = -0.5 * (hp - hm) / (hp + hm - 2.0 * h0);
                let th = 2.0 * VL_PI * (i as f64 + di + 0.5) / f64::from(NBINS);
                angles[nangles] = th;
                nangles += 1;
                if nangles == 4 {
                    break;
                }
            }
        }
        nangles
    }

    /// Port of `vl_sift_calc_keypoint_descriptor`: the 128-bin SIFT descriptor of `k` at
    /// orientation `angle0`, in VLFeat's bin order. Leaves `descr` untouched when the
    /// keypoint is not usable, like VLFeat.
    pub fn calc_keypoint_descriptor(
        &mut self,
        descr: &mut [f32; VL_SIFT_DESCRIPTOR_LENGTH],
        k: &VlSiftKeypoint,
        angle0: f64,
    ) {
        let magnif = self.magnif;
        let xper = fns::pow(2.0, f64::from(self.o_cur));

        let w = self.octave_width;
        let h = self.octave_height;
        const XO: i32 = 2;
        let yo = 2 * w;
        let so = 2 * w * h;
        let x = f64::from(k.x) / xper;
        let y = f64::from(k.y) / xper;
        let sigma = f64::from(k.sigma) / xper;

        let xi = (x + 0.5) as i32;
        let yi = (y + 0.5) as i32;
        let si = k.is;

        let st0 = fns::sin(angle0);
        let ct0 = fns::cos(angle0);
        let sbp = magnif * sigma + VL_EPSILON_D;
        let big_w = (fns::sqrt(2.0) * sbp * f64::from(NBP + 1) / 2.0 + 0.5).floor() as i32;

        const BINTO: i32 = 1;
        const BINYO: i32 = NBO * NBP;
        const BINXO: i32 = NBO;

        // Check bounds.
        if k.o != self.o_cur
            || xi < 0
            || xi >= w
            || yi < 0
            || yi >= h - 1
            || si < self.s_min + 1
            || si > self.s_max - 2
        {
            return;
        }

        self.update_gradient();

        descr.fill(0.0);

        // Center the scale space and the descriptor on the current keypoint. Note that dpt is
        // pointing to the bin of center (SBP/2, SBP/2, 0).
        let pt = xi * XO + yi * yo + (si - self.s_min - 1) * so;
        let dpt = (NBP / 2) * BINYO + (NBP / 2) * BINXO;

        let wsigma = self.window_size as f32;

        // Process pixels in the intersection of the image rectangle (1,1)-(M-1,N-1) and the
        // keypoint bounding box.
        for dyi in (-big_w).max(1 - yi)..=big_w.min(h - yi - 2) {
            for dxi in (-big_w).max(1 - xi)..=big_w.min(w - xi - 2) {
                let g = (pt + dxi * XO + dyi * yo) as usize;
                let m = self.grad[g];
                let angle = self.grad[g + 1];
                let theta = vl_mod_2pi_f((f64::from(angle) - angle0) as f32);

                // Fractional displacement.
                let dx = (f64::from(xi + dxi) - x) as f32;
                let dy = (f64::from(yi + dyi) - y) as f32;

                // Get the displacement normalized w.r.t. the keypoint orientation and
                // extension.
                let nx = ((ct0 * f64::from(dx) + st0 * f64::from(dy)) / sbp) as f32;
                let ny = ((-st0 * f64::from(dx) + ct0 * f64::from(dy)) / sbp) as f32;
                // NBO * theta is int * float, a float product; the division is double.
                let nt = (f64::from(NBO as f32 * theta) / (2.0 * VL_PI)) as f32;

                // Get the Gaussian weight of the sample. The Gaussian window has a standard
                // deviation equal to NBP/2. Note that dx and dy are in the normalized frame,
                // so that -NBP/2 <= dx <= NBP/2.
                let win = fast_expn(
                    f64::from(nx * nx + ny * ny) / (2.0 * f64::from(wsigma) * f64::from(wsigma)),
                ) as f32;

                // The sample will be distributed in 8 adjacent bins. We start from the
                // "lower-left" bin. nx - 0.5 is a double rounded to float for vl_floor_f.
                let binx = vl_floor_f((f64::from(nx) - 0.5) as f32) as i32;
                let biny = vl_floor_f((f64::from(ny) - 0.5) as f32) as i32;
                let bint = vl_floor_f(nt) as i32;
                let rbinx = (f64::from(nx) - (f64::from(binx) + 0.5)) as f32;
                let rbiny = (f64::from(ny) - (f64::from(biny) + 0.5)) as f32;
                let rbint = nt - bint as f32;

                // Distribute the current sample into the 8 adjacent bins.
                for dbinx in 0..2 {
                    for dbiny in 0..2 {
                        for dbint in 0..2 {
                            if binx + dbinx >= -(NBP / 2)
                                && binx + dbinx < NBP / 2
                                && biny + dbiny >= -(NBP / 2)
                                && biny + dbiny < NBP / 2
                            {
                                let weight = win
                                    * m
                                    * ((1 - dbinx) as f32 - rbinx).abs()
                                    * ((1 - dbiny) as f32 - rbiny).abs()
                                    * ((1 - dbint) as f32 - rbint).abs();
                                let index = dpt
                                    + ((bint + dbint) % NBO) * BINTO
                                    + (biny + dbiny) * BINYO
                                    + (binx + dbinx) * BINXO;
                                descr[index as usize] += weight;
                            }
                        }
                    }
                }
            }
        }

        // Standard SIFT descriptors are normalized, truncated and normalized again.

        // Normalize the histogram to L2 unit length.
        let norm = normalize_histogram(descr);

        // Set the descriptor to zero if it is lower than our norm_threshold.
        if self.norm_thresh != 0.0 && f64::from(norm) < self.norm_thresh {
            descr.fill(0.0);
        } else {
            // Truncate at 0.2.
            for d in descr.iter_mut() {
                if f64::from(*d) > 0.2 {
                    *d = 0.2f64 as f32;
                }
            }
            // Normalize again.
            normalize_histogram(descr);
        }
    }

    /// Port of `update_gradient`: polar gradients (modulus, angle in [0, 2*pi]) of the levels
    /// `s_min + 1 ..= s_max - 2` of the current octave, computed once per octave.
    fn update_gradient(&mut self) {
        if self.grad_o == self.o_cur {
            return;
        }
        let w = self.octave_width as usize;
        let h = self.octave_height as usize;
        let so = w * h;
        for s in self.s_min + 1..=self.s_max - 2 {
            let src = self.octave_offset(s);
            let g = 2 * so * (s - self.s_min - 1) as usize;
            gradient_level(
                &self.octave[src..src + so],
                &mut self.grad[g..g + 2 * so],
                w,
                h,
            );
        }
        self.grad_o = self.o_cur;
    }
}

/// Port of `normalize_histogram`: divides by the fast L2 norm (+ `VL_EPSILON_F`), returns it.
pub(super) fn normalize_histogram(hist: &mut [f32]) -> f32 {
    let mut norm = 0.0f32;
    for &v in hist.iter() {
        norm += v * v;
    }
    norm = vl_fast_sqrt_f(norm) + VL_EPSILON_F;
    for v in hist.iter_mut() {
        *v /= norm;
    }
    norm
}

/// One level of `update_gradient`: one-sided differences on the border, central inside,
/// written as interleaved (modulus, angle) pairs. Every pixel's value depends only on its
/// neighbors, so the closed-form traversal matches VLFeat's pointer walk for every level of
/// at least 2 x 2 pixels. Only an octave with a keypoint has its gradient computed, and a
/// keypoint needs a 3 x 3 neighborhood, so smaller levels never get here.
fn gradient_level(level: &[f32], grad: &mut [f32], w: usize, h: usize) {
    // "0.5 * (a - b)" promotes the float difference to double; halving is exact, so the value
    // is the same as the float product.
    let half = |v: f32| (0.5 * f64::from(v)) as f32;
    for y in 0..h {
        for x in 0..w {
            let p = y * w + x;
            let gx = if x == 0 {
                level[p + 1] - level[p]
            } else if x == w - 1 {
                level[p] - level[p - 1]
            } else {
                half(level[p + 1] - level[p - 1])
            };
            let gy = if y == 0 {
                level[p + w] - level[p]
            } else if y == h - 1 {
                level[p] - level[p - w]
            } else {
                half(level[p + w] - level[p - w])
            };
            grad[2 * p] = vl_fast_sqrt_f(gx * gx + gy * gy);
            // `vl_fast_atan2_f(gy, gx) + 2*VL_PI` is a double, rounded to float by the call.
            grad[2 * p + 1] =
                vl_mod_2pi_f((f64::from(vl_fast_atan2_f(gy, gx)) + 2.0 * VL_PI) as f32);
        }
    }
}
