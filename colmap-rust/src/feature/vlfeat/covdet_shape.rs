//! The frame-dependent parts of VLFeat's `covdet.c`, as ported in colmap-sharp's
//! `Feature/VLFeat/VlCovDet.Shape.cs`: `vl_covdet_extract_patch_helper`,
//! `vl_covdet_extract_patch_for_frame`, `vl_covdet_extract_affine_shape(_for_frame)` and
//! `vl_covdet_extract_orientations(_for_frame)`. Part of [`VlCovDet`] (detection in
//! [`super::covdet`]); patches are warped from its Gaussian scale space ([`super::scalespace`]),
//! smoothed and differentiated with [`super::imopv_smooth`], and decomposed with
//! [`super::linear`].
//!
//! Tier A (exact); see [`super::covdet`]. `exp`, `log2`, `pow`, `atan2`, `cos` and `sin` go
//! through [`crate::math::fns`]. Kept VLFeat quirks, since they decide the numbers:
//! - The patch helper picks its scale-space level with `floor(log2(sigma / (factor * base)) -
//!   o)`, which ignores the octave resolution.
//! - When the warped patch leaves the image, the padded copy stops copying two pixels early on
//!   each row and repeats the next source pixel instead (`patchWidth - padx1 - 2`).
//! - A feature whose orientation histogram has no peak keeps its frame unrotated (it is not
//!   dropped), and affine adaptation never drops a feature (its only failure is out of
//!   memory).
//! - The orientations are sorted by descending score with `qsort`; macOS's `qsort`
//!   insertion-sorts arrays this short (at most 4), so ties keep peak order. Here: the same
//!   insertion sort.
//!
//! Cancellation (not in VLFeat): the per-feature passes poll every 64 features.

use super::covdet::{
    VlCovDet, VlFrameOrientedEllipse, AA_CONVERGENCE_THRESHOLD, AA_MAX_ANISOTROPY,
    AA_MAX_NUM_ITERATIONS, AA_PATCH_EXTENT, AA_PATCH_RESOLUTION, AA_PATCH_SIDE,
    AA_RELATIVE_DERIVATIVE_SIGMA, MAX_NUM_ORIENTATIONS, OR_ADDITIONAL_PEAKS_RELATIVE_SIZE,
    OR_NUM_ORIENTATION_HISTOGRAM_BINS,
};
use super::imopv_smooth::{
    vl_imgradient_f, vl_imgradient_polar_f, vl_imsmooth_f_within, GradientLayout, ImageRegion,
};
use super::linear::{vl_solve_linear_system_2, vl_svd2};
use super::mathop::{vl_floor_d, VL_PI};
use super::scalespace::VlScaleSpace;
use crate::math::fns;
use crate::util::cancellation::CancelToken;
use crate::util::check::Result;
use crate::{check, check_notnull};

/// `VL_MIN`: a comparison, so NaN picks the second argument.
fn vl_min(a: f64, b: f64) -> f64 {
    if a < b {
        a
    } else {
        b
    }
}

/// `VL_MAX`: a comparison, so NaN picks the second argument.
fn vl_max(a: f64, b: f64) -> f64 {
    if a > b {
        a
    } else {
        b
    }
}

/// The frame's `A` in VLFeat's column-major order and its translation.
fn frame_matrices(frame: &VlFrameOrientedEllipse) -> ([f64; 4], [f64; 2]) {
    (
        [
            f64::from(frame.a11),
            f64::from(frame.a21),
            f64::from(frame.a12),
            f64::from(frame.a22),
        ],
        [f64::from(frame.x), f64::from(frame.y)],
    )
}

/// The orientations `vl_covdet_extract_orientations_for_frame` found, by decreasing score.
struct Orientations {
    angles: [f64; MAX_NUM_ORIENTATIONS],
    scores: [f64; MAX_NUM_ORIENTATIONS],
    count: usize,
}

impl VlCovDet {
    /// Port of `vl_covdet_extract_patch_for_frame`: samples a `(2 * resolution + 1)^2` patch
    /// covering `[-extent, extent]^2` in the frame's coordinates from the scale-space level
    /// whose smoothing best approximates `sigma` (in patch units) from below. `Err` before
    /// `put_image` or when `patch` is too small.
    pub fn extract_patch_for_frame(
        &mut self,
        patch: &mut [f32],
        resolution: usize,
        extent: f64,
        sigma: f64,
        frame: &VlFrameOrientedEllipse,
    ) -> Result<()> {
        let side = 2 * resolution + 1;
        check!(patch.len() >= side * side);
        let gss = check_notnull!(self.gss.as_ref());
        let (a, t) = frame_matrices(frame);
        let (d, _, _) = vl_svd2(&a);
        patch_helper(
            gss,
            &mut self.patch,
            patch,
            resolution,
            extent,
            sigma,
            &a,
            &t,
            d[0],
            d[3],
        );
        Ok(())
    }

    /// Port of `vl_covdet_extract_affine_shape`: replaces each feature's frame by its
    /// shape-adapted, upright frame (Baumberg / Mikolajczyk-Schmid iteration on the windowed
    /// second-moment matrix). `Err` when cancelled, or before `put_image` with features.
    pub fn extract_affine_shape(&mut self, cancel: &CancelToken) -> Result<()> {
        for i in 0..self.features.len() {
            if i & 63 == 0 {
                cancel.check_cancelled()?;
            }
            let frame = self.features[i].frame;
            self.features[i].frame = self.extract_affine_shape_for_frame(&frame)?;
        }
        Ok(())
    }

    /// Port of `vl_covdet_extract_orientations`: rotates each feature to its dominant gradient
    /// orientation, appending a copy of the feature for each further orientation (up to
    /// four). `Err` when cancelled, or before `put_image` with features.
    pub fn extract_orientations(&mut self, cancel: &CancelToken) -> Result<()> {
        let num_features = self.features.len();
        for i in 0..num_features {
            if i & 63 == 0 {
                cancel.check_cancelled()?;
            }
            let feature = self.features[i];
            let orientations = self.extract_orientations_for_frame(&feature.frame)?;
            let (a, _) = frame_matrices(&feature.frame);
            for j in 0..orientations.count {
                let r1 = fns::cos(orientations.angles[j]);
                let r2 = fns::sin(orientations.angles[j]);
                let oriented = if j == 0 {
                    i
                } else {
                    self.features.push(feature);
                    self.features.len() - 1
                };
                let f = &mut self.features[oriented];
                f.orientation_score = orientations.scores[j] as f32;
                f.frame.a11 = (a[0] * r1 + a[2] * r2) as f32;
                f.frame.a21 = (a[1] * r1 + a[3] * r2) as f32;
                f.frame.a12 = (-a[0] * r2 + a[2] * r1) as f32;
                f.frame.a22 = (-a[1] * r2 + a[3] * r1) as f32;
            }
        }
        Ok(())
    }

    /// Port of `vl_covdet_extract_affine_shape_for_frame`. Always succeeds in VLFeat
    /// (`VL_ERR_OK`); `Err` only before `put_image`.
    fn extract_affine_shape_for_frame(
        &mut self,
        frame: &VlFrameOrientedEllipse,
    ) -> Result<VlFrameOrientedEllipse> {
        let gss = check_notnull!(self.gss.as_ref());
        let (mut a, t) = frame_matrices(frame);
        let sigma_d = AA_RELATIVE_DERIVATIVE_SIGMA;
        let mut reference_scale = 0.0;
        let side = AA_PATCH_SIDE;
        let region = ImageRegion {
            offset: 0,
            width: side,
            height: side,
            stride: side,
        };
        let mut iter = 0;
        let mut adapted = *frame;

        loop {
            // A = U D V'
            let (mut d, u, _) = vl_svd2(&a);
            let anisotropy = vl_max(d[0] / d[3], d[3] / d[0]);
            if anisotropy > AA_MAX_ANISOTROPY {
                // Diverged, give up with the current solution.
                break;
            }

            // Make sure that the smallest singular value stays fixed after the first
            // iteration.
            let factor = if iter == 0 {
                reference_scale = vl_min(d[0], d[3]);
                1.0
            } else {
                reference_scale / vl_min(d[0], d[3])
            };
            d[0] *= factor;
            d[3] *= factor;

            a = [u[0] * d[0], u[1] * d[0], u[2] * d[3], u[3] * d[3]];
            adapted.a11 = a[0] as f32;
            adapted.a21 = a[1] as f32;
            adapted.a12 = a[2] as f32;
            adapted.a22 = a[3] as f32;

            iter += 1;
            if iter >= AA_MAX_NUM_ITERATIONS {
                break;
            }

            patch_helper(
                gss,
                &mut self.patch,
                &mut self.aa_patch,
                AA_PATCH_RESOLUTION,
                AA_PATCH_EXTENT,
                sigma_d,
                &a,
                &t,
                d[0],
                d[3],
            );

            // VL_COVDET_AA_ACCURATE_SMOOTHING is off: no extra smoothing of the patch here.

            // Compute the second moment matrix.
            vl_imgradient_f(
                &mut self.aa_patch_x,
                &mut self.aa_patch_y,
                GradientLayout {
                    first_offset: 0,
                    second_offset: 0,
                    width_stride: 1,
                    height_stride: side,
                },
                &self.aa_patch,
                region,
            );
            let (mut lxx, mut lxy, mut lyy) = (0.0f64, 0.0f64, 0.0f64);
            for k in 0..side * side {
                let lx = f64::from(self.aa_patch_x[k]);
                let ly = f64::from(self.aa_patch_y[k]);
                let mask = f64::from(self.aa_mask[k]);
                lxx += lx * lx * mask;
                lyy += ly * ly * mask;
                lxy += lx * ly * mask;
            }

            if lxx == 0.0 || lyy == 0.0 {
                adapted = *frame;
                break;
            }

            // Decompose M = P * Q * P'.
            let (q, p, _) = vl_svd2(&[lxx, lxy, lxy, lyy]);

            // Setting A <- A * dA changes M approximately to dA' M dA = dA' P Q P' dA. To
            // make this proportional to the identity, dA ~= P Q^-1/2, keeping the smallest
            // singular value of A unchanged.
            if q[3] / q[0] < AA_CONVERGENCE_THRESHOLD && q[0] / q[3] < AA_CONVERGENCE_THRESHOLD {
                break;
            }

            let q0 = q[0].sqrt();
            let q1 = q[3].sqrt();
            a = [
                (a[0] * p[0] + a[2] * p[1]) / q0,
                (a[1] * p[0] + a[3] * p[1]) / q0,
                (a[0] * p[2] + a[2] * p[3]) / q1,
                (a[1] * p[2] + a[3] * p[3]) / q1,
            ];
        }

        // Make upright. Shape adaptation does not estimate rotation, so the frame is rotated
        // such that the vertical axis (the non-transposed convention) is not rotated at all.
        let (au, _) = frame_matrices(&adapted);
        let refv = [0.0, 1.0];
        let mut ref_ = [0.0f64; 2];
        // VLFeat ignores the solver's status; on a singular A, ref_ is what it leaves.
        let _ = vl_solve_linear_system_2(&mut ref_, &au, &refv);
        let angle = fns::atan2(refv[1], refv[0]);
        let angle_ = fns::atan2(ref_[1], ref_[0]);
        let dangle = angle_ - angle;
        let r1 = fns::cos(dangle);
        let r2 = fns::sin(dangle);
        adapted.a11 = (au[0] * r1 + au[2] * r2) as f32;
        adapted.a21 = (au[1] * r1 + au[3] * r2) as f32;
        adapted.a12 = (-au[0] * r2 + au[2] * r1) as f32;
        adapted.a22 = (-au[1] * r2 + au[3] * r1) as f32;
        Ok(adapted)
    }

    /// Port of `vl_covdet_extract_orientations_for_frame`: the dominant gradient orientations
    /// of the frame's patch, by decreasing score. `Err` only before `put_image`.
    fn extract_orientations_for_frame(
        &mut self,
        frame: &VlFrameOrientedEllipse,
    ) -> Result<Orientations> {
        const NUM_BINS: usize = OR_NUM_ORIENTATION_HISTOGRAM_BINS;
        let gss = check_notnull!(self.gss.as_ref());
        let extent = AA_PATCH_EXTENT;
        let resolution = AA_PATCH_RESOLUTION;
        let side = AA_PATCH_SIDE;
        let bin_extent = 2.0 * VL_PI / NUM_BINS as f64;
        let peak_relative_size = OR_ADDITIONAL_PEAKS_RELATIVE_SIZE;
        let sigma_d = 1.0;
        let region = ImageRegion {
            offset: 0,
            width: side,
            height: side,
            stride: side,
        };

        // The goal is a rotation R(theta) such that the patch given by A R(theta) has the
        // strongest average gradient pointing right. To compensate for the anisotropic
        // smoothing due to warping, A = U D V' is warped by U D only and R_(theta) is
        // estimated instead, where A R(theta) = U D R_(theta): an extra rotation of theta0.
        let (a, t) = frame_matrices(frame);
        let (d, u, v) = vl_svd2(&a);
        let a = [u[0] * d[0], u[1] * d[0], u[2] * d[3], u[3] * d[3]];
        let theta0 = fns::atan2(v[1], v[0]);

        let (sigma1, sigma2) = patch_helper(
            gss,
            &mut self.patch,
            &mut self.aa_patch,
            resolution,
            extent,
            sigma_d,
            &a,
            &t,
            d[0],
            d[3],
        );

        let delta_sigma1 = vl_max(sigma_d * sigma_d - sigma1 * sigma1, 0.0).sqrt();
        let delta_sigma2 = vl_max(sigma_d * sigma_d - sigma2 * sigma2, 0.0).sqrt();
        let stephat = extent / resolution as f64;
        vl_imsmooth_f_within(
            &mut self.aa_patch,
            0,
            side,
            region,
            delta_sigma1 / stephat,
            delta_sigma2 / stephat,
        );

        // Histogram of oriented gradients.
        vl_imgradient_polar_f(
            &mut self.aa_patch_x,
            &mut self.aa_patch_y,
            GradientLayout {
                first_offset: 0,
                second_offset: 0,
                width_stride: 1,
                height_stride: side,
            },
            &self.aa_patch,
            region,
        );

        let mut hist = [0.0f64; NUM_BINS];
        for k in 0..side * side {
            let modulus = f64::from(self.aa_patch_x[k]);
            let angle = f64::from(self.aa_patch_y[k]);
            let weight = f64::from(self.aa_mask[k]);

            let x = angle / bin_extent;
            let bin = vl_floor_d(x);
            let w2 = x - bin as f64;
            let w1 = 1.0 - w2;

            // `(bin + numBins) % numBins` in VLFeat's unsigned (vl_size) arithmetic.
            let n = NUM_BINS as u64;
            hist[((bin as u64).wrapping_add(n) % n) as usize] += w1 * (modulus * weight);
            hist[((bin as u64).wrapping_add(n + 1) % n) as usize] += w2 * (modulus * weight);
        }

        // Smooth the histogram.
        for _ in 0..6 {
            let mut prev = hist[NUM_BINS - 1];
            let first = hist[0];
            for i in 0..NUM_BINS - 1 {
                let curr = (prev + hist[i] + hist[(i + 1) % NUM_BINS]) / 3.0;
                prev = hist[i];
                hist[i] = curr;
            }
            hist[NUM_BINS - 1] = (prev + hist[NUM_BINS - 1] + first) / 3.0;
        }

        // Find the histogram maximum.
        let max_peak_value = hist.iter().fold(0.0, |m, &h| vl_max(m, h));

        // Find the peaks within 80% of the maximum.
        let mut out = Orientations {
            angles: [0.0; MAX_NUM_ORIENTATIONS],
            scores: [0.0; MAX_NUM_ORIENTATIONS],
            count: 0,
        };
        for i in 0..NUM_BINS {
            let h0 = hist[i];
            let hm = hist[(i + NUM_BINS - 1) % NUM_BINS];
            let hp = hist[(i + 1) % NUM_BINS];

            // Is this a peak?
            if h0 > peak_relative_size * max_peak_value && h0 > hm && h0 > hp {
                // Quadratic interpolation.
                let di = -0.5 * (hp - hm) / (hp + hm - 2.0 * h0);
                let th = bin_extent * (i as f64 + di) + theta0;
                out.angles[out.count] = th;
                out.scores[out.count] = h0;
                out.count += 1;
                if out.count >= MAX_NUM_ORIENTATIONS {
                    break;
                }
            }
        }

        // Sort the orientations by decreasing scores (qsort: an insertion sort at this size).
        for i in 1..out.count {
            let mut k = i;
            while k > 0 && out.scores[k - 1] < out.scores[k] {
                out.scores.swap(k - 1, k);
                out.angles.swap(k - 1, k);
                k -= 1;
            }
        }
        Ok(out)
    }
}

/// Port of `vl_covdet_extract_patch_helper`: warps `[-extent, extent]^2` through
/// `x -> A x + T` from the scale-space level `(o, s)` whose smoothing `sigma_` satisfies
/// `sigma_ * factor <= sigma` as closely as possible, `factor = 1 / min(d1, d2)`, into
/// `patch_out`; returns the resulting smoothing `(sigma1, sigma2)` along the two axes.
/// Starting from a level smoothed by `sigma_`, the mapping `A = U D V'` makes the warped
/// smoothing `sigma_^2 V D^-2 V'`, so with `A` rotated to `U D` it is `sigma_^2 D^-2`,
/// axis-aligned. `pad` is the detector's padded-copy buffer. Always succeeds in VLFeat.
#[allow(clippy::too_many_arguments)]
fn patch_helper(
    gss: &VlScaleSpace,
    pad: &mut Vec<f32>,
    patch_out: &mut [f32],
    resolution: usize,
    extent: f64,
    sigma: f64,
    a_: &[f64; 4],
    t_: &[f64; 2],
    d1: f64,
    d2: f64,
) -> (f64, f64) {
    let mut a = *a_;
    let mut t = *t_;
    let geom = gss.geometry();
    let first_sub = i64::from(geom.octave_first_subdivision);
    let last_sub = i64::from(geom.octave_last_subdivision);
    let resolution_d = f64::from(geom.octave_resolution);

    // Determine the best level (o, s) such that sigma_(o, s) factor <= sigma, scanning
    // octaves from the smallest and stopping when no level in the octave satisfies it.
    let factor = 1.0 / vl_min(d1, d2);
    let level_for = |o: i64| -> (i64, f64) {
        let s = vl_floor_d(fns::log2(sigma / (factor * geom.base_scale)) - o as f64);
        let s = s.max(first_sub).min(last_sub);
        let sigma_ = geom.base_scale * fns::pow(2.0, o as f64 + s as f64 / resolution_d);
        (s, sigma_)
    };
    let mut o = i64::from(geom.first_octave) + 1;
    while o <= i64::from(geom.last_octave) {
        let (_, sigma_) = level_for(o);
        if factor * sigma_ > sigma {
            o -= 1;
            break;
        }
        o += 1;
    }
    let o = o.min(i64::from(geom.last_octave));
    let (s, sigma_) = level_for(o);
    let sigmas = (sigma_ / d1, sigma_ / d2);

    // Now the level to warp from is known. If the patch is partially or completely out of
    // the image, create a padded copy of the required region first.
    let oct = gss.octave_geometry(o as i32);
    let mut level = gss.level(o as i32, s as i32);
    let mut width = oct.width as i64;
    // VLFeat also replaces the height by the padded copy's, but reads it only in asserts.
    let height = oct.height as i64;
    let step = oct.step;
    for v in &mut a {
        *v /= step;
    }
    t[0] /= step;
    t[1] /= step;

    // Warp the patch domain to the image domain, take the enclosing box, and grow it by one
    // pixel on every side for the bilinear interpolation.
    let (mut x0, mut x1) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY);
    let boxx = [extent, extent, -extent, -extent];
    let boxy = [-extent, extent, extent, -extent];
    for i in 0..4 {
        let x = a[0] * boxx[i] + a[2] * boxy[i] + t[0];
        let y = a[1] * boxx[i] + a[3] * boxy[i] + t[1];
        x0 = vl_min(x0, x);
        x1 = vl_max(x1, x);
        y0 = vl_min(y0, y);
        y1 = vl_max(y1, y);
    }
    // `floor(x0) - 1` is a double stored in a vl_index.
    let x0i = (x0.floor() - 1.0) as i64;
    let y0i = (y0.floor() - 1.0) as i64;
    let x1i = (x1.ceil() + 1.0) as i64;
    let y1i = (y1.ceil() + 1.0) as i64;

    // If the box is not inside the image domain, copy the region, extending the image by
    // continuity.
    if x0i < 0 || x1i > width - 1 || y0i < 0 || y1i > height - 1 {
        // The amount of left, right, top and bottom padding needed.
        let padx0 = 0.max(-x0i);
        let pady0 = 0.max(-y0i);
        let padx1 = 0.max(x1i - (width - 1));
        let pady1 = 0.max(y1i - (height - 1));

        // Make enough room for the patch (the buffer only grows, like VLFeat's).
        let patch_width = x1i - x0i + 1;
        let patch_height = y1i - y0i + 1;
        let patch_size = (patch_width * patch_height) as usize;
        if patch_size > pad.len() {
            pad.resize(patch_size, 0.0);
        }

        if pady0 < patch_height - pady1 {
            // Start by filling the central horizontal band. VLFeat stops the copy two pixels
            // early and repeats the next source pixel for the rest; kept as is.
            for yi in y0i + pady0..y0i + patch_height - pady1 {
                let mut dst = ((yi - y0i) * patch_width) as usize;
                let mut src = (yi * width + x0i.max(0).min(width - 1)) as usize;
                let mut xi = x0i;
                while xi < x0i + padx0 {
                    pad[dst] = level[src];
                    dst += 1;
                    xi += 1;
                }
                while xi < x0i + patch_width - padx1 - 2 {
                    pad[dst] = level[src];
                    dst += 1;
                    src += 1;
                    xi += 1;
                }
                while xi < x0i + patch_width {
                    pad[dst] = level[src];
                    dst += 1;
                    xi += 1;
                }
            }

            // Now extend the central band up and down.
            let row = patch_width as usize;
            for yi in 0..pady0 as usize {
                let from = pady0 as usize * row;
                pad.copy_within(from..from + row, yi * row);
            }
            for yi in (patch_height - pady1) as usize..patch_height as usize {
                let from = (patch_height - pady1 - 1) as usize * row;
                pad.copy_within(from..from + row, yi * row);
            }
        } else {
            // "should be handled better!"
            pad.fill(0.0);
        }

        level = pad.as_slice();
        width = patch_width;
        t[0] -= x0i as f64;
        t[1] -= y0i as f64;
    }
    // Resample by bilinear interpolation. The box above keeps every read inside the level.
    let mut pt = 0;
    let mut yhat = -extent;
    let stephat = extent / resolution as f64;
    let at = |yi: i64, xi: i64| f64::from(level[(yi * width + xi) as usize]);
    for _ in 0..2 * resolution + 1 {
        let mut xhat = -extent;
        let rx = a[2] * yhat + t[0];
        let ry = a[3] * yhat + t[1];
        for _ in 0..2 * resolution + 1 {
            let x = a[0] * xhat + rx;
            let y = a[1] * xhat + ry;
            let xi = vl_floor_d(x);
            let yi = vl_floor_d(y);
            let i00 = at(yi, xi);
            let i10 = at(yi, xi + 1);
            let i01 = at(yi + 1, xi);
            let i11 = at(yi + 1, xi + 1);
            let wx = x - xi as f64;
            let wy = y - yi as f64;

            patch_out[pt] = ((1.0 - wy) * ((1.0 - wx) * i00 + wx * i10)
                + wy * ((1.0 - wx) * i01 + wx * i11)) as f32;
            pt += 1;

            xhat += stephat;
        }
        yhat += stephat;
    }
    sigmas
}
