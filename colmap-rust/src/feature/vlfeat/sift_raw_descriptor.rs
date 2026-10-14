//! Port of `vl_sift_calc_raw_descriptor` from VLFeat's `sift.c`, as ported in colmap-sharp's
//! `Feature/VLFeat/VlSiftFilter.Descriptor.cs`: the SIFT descriptor of a frame computed from a
//! caller-supplied polar gradient image, which COLMAP's covariant extractor runs on each
//! normalized patch. Part of [`VlSiftFilter`]; it uses only the filter's `magnif`,
//! `window_size` and `norm_thresh`. The scale-space descriptor is in [`super::sift_descriptor`],
//! whose `normalize_histogram` this shares.
//!
//! Tier A (exact). Descriptor bins are float; every conversion is written out where the C
//! converts implicitly (a float local assigned a double expression rounds once, at the
//! assignment). `sin` / `cos` / `sqrt` go through [`crate::math::fns`].

use super::mathop::{fast_expn, vl_floor_f, vl_mod_2pi_f, VL_EPSILON_D, VL_PI};
use super::sift_descriptor::{normalize_histogram, VL_SIFT_DESCRIPTOR_LENGTH};
use super::sift_filter::{VlSiftFilter, NBO, NBP};
use crate::math::fns;

/// The frame `vl_sift_calc_raw_descriptor` describes, in gradient-image pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RawDescriptorFrame {
    /// Keypoint x.
    pub x: f64,
    /// Keypoint y.
    pub y: f64,
    /// Keypoint scale.
    pub sigma: f64,
    /// Keypoint orientation (radians).
    pub angle0: f64,
}

impl VlSiftFilter {
    /// Port of `vl_sift_calc_raw_descriptor`: the 128-bin SIFT descriptor of `frame` from
    /// `grad`, a `width` x `height` gradient image of interleaved (modulus, angle) pairs (what
    /// [`super::imopv_smooth::vl_imgradient_polar_f`] writes with a width stride of 2). Leaves
    /// `descr` untouched when the keypoint is outside the image, like VLFeat. Unlike the
    /// scale-space descriptor, border pixels are included, and the norm threshold is scaled
    /// by the number of samples (the Gaussian window is not normalized).
    pub fn calc_raw_descriptor(
        &self,
        grad: &[f32],
        descr: &mut [f32; VL_SIFT_DESCRIPTOR_LENGTH],
        width: i32,
        height: i32,
        frame: RawDescriptorFrame,
    ) {
        let RawDescriptorFrame {
            x,
            y,
            sigma,
            angle0,
        } = frame;
        let magnif = self.magnif;
        let w = width;
        let h = height;
        const XO: i32 = 2;
        let yo = 2 * w;

        // `(int)(x + 0.5)` truncates toward zero.
        let xi = (x + 0.5) as i32;
        let yi = (y + 0.5) as i32;

        let st0 = fns::sin(angle0);
        let ct0 = fns::cos(angle0);
        let sbp = magnif * sigma + VL_EPSILON_D;
        let big_w = (fns::sqrt(2.0) * sbp * f64::from(NBP + 1) / 2.0 + 0.5).floor() as i32;

        const BINTO: i32 = 1;
        const BINYO: i32 = NBO * NBP;
        const BINXO: i32 = NBO;

        // Check bounds.
        if xi < 0 || xi >= w || yi < 0 || yi >= h - 1 {
            return;
        }

        descr.fill(0.0);

        // Center the scale space and the descriptor on the current keypoint. Note that dpt is
        // pointing to the bin of center (SBP/2, SBP/2, 0).
        let pt = xi * XO + yi * yo;
        let dpt = (NBP / 2) * BINYO + (NBP / 2) * BINXO;
        let wsigma = self.window_size as f32;

        for dyi in (-big_w).max(-yi)..=big_w.min(h - yi - 1) {
            for dxi in (-big_w).max(-xi)..=big_w.min(w - xi - 1) {
                let g = (pt + dxi * XO + dyi * yo) as usize;
                let m = grad[g];
                let angle = grad[g + 1];
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
        let norm = normalize_histogram(descr);

        // Set the descriptor to zero if it is lower than our norm_threshold. We divide by the
        // number of samples in the descriptor region because the Gaussian window used in the
        // calculation of the descriptor is not normalized.
        let num_samples = (big_w.min(w - xi - 1) - (-big_w).max(-xi) + 1)
            * (big_w.min(h - yi - 1) - (-big_w).max(-yi) + 1);
        if self.norm_thresh != 0.0 && f64::from(norm) < self.norm_thresh * f64::from(num_samples) {
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
}
