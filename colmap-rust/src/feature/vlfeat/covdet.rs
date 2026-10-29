//! Port of VLFeat's covariant feature detector (`thirdparty/VLFeat/covdet.h` / `covdet.c`) with
//! the DoG method, the one COLMAP's `CovariantSiftCPUFeatureExtractor` creates, as ported in
//! colmap-sharp's `Feature/VLFeat/VlCovDet.cs`. This file holds the detector state,
//! `vl_covdet_new`, `vl_covdet_put_image`, the DoG response, `vl_find_local_extrema_3`,
//! `vl_refine_local_extreum_3` and `vl_covdet_detect` with its non-extrema suppression.
//! [`super::covdet_shape`] has patch extraction, affine adaptation and orientation assignment.
//! The Gaussian scale space is [`super::scalespace`].
//!
//! Not ported: the Hessian, Harris and Laplacian methods (and the Laplacian filter bank
//! `vl_covdet_new` precomputes for them), the 2D extremum search they use, feature dropping by
//! margin, and the transposed convention. COLMAP uses none of them.
//!
//! Tier A (exact). Feature frames and scores are `f32` (`VlFrameOrientedEllipse`,
//! `VlCovDetFeature` and `VlCovDetExtremum3` all store float), computation is `f64`; every point
//! where the C rounds to float, or computes in float before promoting, is written out. `VL_MAX`
//! / `VL_MIN` are comparisons, not `f64::max` / `min`, because they differ on NaN. `pow` and
//! `log2` go through [`crate::math::fns`]. Pinned against
//! VLFeat compiled without FMA contraction by `tests/feature/rust_only_vl_covdet.rs`
//! (`oracle/sift_harness.c`'s covdet mode).
//!
//! Translation notes:
//! - Cancellation (not in VLFeat): `put_image` polls before each octave, `detect` before each
//!   octave's extremum scan and every 256 features of the suppression pass. The checks never
//!   touch the numbers.
//! - VLFeat's asserts (an image of at least 1x1, `detect` after `put_image`) are `Err`s; so
//!   are an image too small for the octaves and an image buffer of the wrong length, where
//!   VLFeat's behaviour is undefined (docs/CPP_DIVERGENCES.md entry 295).
//! - The feature buffer is a `Vec`; VLFeat's growth policy (1000 spare features) never reaches
//!   an output.

use super::linear::vl_solve_linear_system_3;
use super::mathop::vl_floor_d;
use super::scalespace::{VlScaleSpace, VlScaleSpaceGeometry};
use crate::math::fns;
use crate::util::cancellation::CancelToken;
use crate::util::check::Result;
use crate::{check, check_eq, check_notnull};

/// `VL_COVDET_MAX_NUM_ORIENTATIONS`.
pub(super) const MAX_NUM_ORIENTATIONS: usize = 4;
/// `VL_COVDET_AA_PATCH_RESOLUTION`.
pub(super) const AA_PATCH_RESOLUTION: usize = 20;
/// `VL_COVDET_AA_MAX_NUM_ITERATIONS`.
pub(super) const AA_MAX_NUM_ITERATIONS: i32 = 15;
/// `VL_COVDET_OR_NUM_ORIENTATION_HISTOGAM_BINS`.
pub(super) const OR_NUM_ORIENTATION_HISTOGRAM_BINS: usize = 36;
/// `VL_COVDET_AA_RELATIVE_INTEGRATION_SIGMA`.
pub(super) const AA_RELATIVE_INTEGRATION_SIGMA: f64 = 3.0;
/// `VL_COVDET_AA_RELATIVE_DERIVATIVE_SIGMA`.
pub(super) const AA_RELATIVE_DERIVATIVE_SIGMA: f64 = 1.0;
/// `VL_COVDET_AA_MAX_ANISOTROPY`.
pub(super) const AA_MAX_ANISOTROPY: f64 = 5.0;
/// `VL_COVDET_AA_CONVERGENCE_THRESHOLD`.
pub(super) const AA_CONVERGENCE_THRESHOLD: f64 = 1.001;
/// `VL_COVDET_AA_PATCH_EXTENT`.
pub(super) const AA_PATCH_EXTENT: f64 = 3.0 * AA_RELATIVE_INTEGRATION_SIGMA;
/// `VL_COVDET_OR_ADDITIONAL_PEAKS_RELATIVE_SIZE`.
pub(super) const OR_ADDITIONAL_PEAKS_RELATIVE_SIZE: f64 = 0.8;
/// Side of the affine-adaptation and orientation patches.
pub(super) const AA_PATCH_SIDE: usize = 2 * AA_PATCH_RESOLUTION + 1;

/// Port of `VlFrameOrientedEllipse`: an affine frame mapping the unit circle to the feature,
/// `x -> A x + (x, y)` with `A = [a11 a12; a21 a22]`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VlFrameOrientedEllipse {
    /// Center x-coordinate.
    pub x: f32,
    /// Center y-coordinate.
    pub y: f32,
    /// `A(1,1)`.
    pub a11: f32,
    /// `A(1,2)`.
    pub a12: f32,
    /// `A(2,1)`.
    pub a21: f32,
    /// `A(2,2)`.
    pub a22: f32,
}

/// Port of `VlCovDetFeature`.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct VlCovDetFeature {
    /// Feature frame.
    pub frame: VlFrameOrientedEllipse,
    /// Detected octave.
    pub o: i32,
    /// Octave subdivision (the rounded level index within the DoG octave).
    pub s: i32,
    /// Peak score.
    pub peak_score: f32,
    /// Edge score.
    pub edge_score: f32,
    /// Orientation score.
    pub orientation_score: f32,
    /// Laplacian scale score (unused by the DoG method).
    pub laplacian_scale_score: f32,
}

/// Port of `VlCovDet` with `VL_COVDET_METHOD_DOG`.
#[derive(Debug, Clone)]
pub struct VlCovDet {
    /// Index of the first octave (`vl_covdet_set_first_octave`).
    pub first_octave: i32,
    /// Number of levels per octave (`vl_covdet_set_octave_resolution`).
    pub octave_resolution: i32,
    /// Peak threshold (`vl_covdet_set_peak_threshold`).
    pub peak_threshold: f64,
    /// Edge threshold (`vl_covdet_set_edge_threshold`).
    pub edge_threshold: f64,
    /// Non-extrema suppression tolerance (`vl_covdet_set_non_extrema_suppression_threshold`);
    /// zero disables it.
    pub non_extrema_suppression: f64,
    num_non_extrema_suppressed: usize,
    pub(super) features: Vec<VlCovDetFeature>,
    pub(super) gss: Option<VlScaleSpace>,
    css: Option<VlScaleSpace>,
    /// `self->patch`: the padded copy of a level region when a warped patch leaves the image.
    /// It only grows, like VLFeat's.
    pub(super) patch: Vec<f32>,
    pub(super) aa_patch: Vec<f32>,
    pub(super) aa_patch_x: Vec<f32>,
    pub(super) aa_patch_y: Vec<f32>,
    pub(super) aa_mask: Vec<f32>,
}

impl Default for VlCovDet {
    fn default() -> Self {
        Self::new()
    }
}

impl VlCovDet {
    /// Port of `vl_covdet_new(VL_COVDET_METHOD_DOG)`.
    pub fn new() -> Self {
        const W: i64 = AA_PATCH_RESOLUTION as i64;
        let side = AA_PATCH_SIDE;
        let mut aa_mask = vec![0.0f32; side * side];
        let step = (2.0 * AA_PATCH_EXTENT) / (2 * W + 1) as f64;
        let sigma = AA_RELATIVE_INTEGRATION_SIGMA;
        for j in -W..=W {
            for i in -W..=W {
                let dx = i as f64 * step / sigma;
                let dy = j as f64 * step / sigma;
                aa_mask[((i + W) + (2 * W + 1) * (j + W)) as usize] =
                    fns::exp(-0.5 * (dx * dx + dy * dy)) as f32;
            }
        }
        Self {
            first_octave: -1,
            octave_resolution: 3,
            peak_threshold: 0.01, // VL_COVDET_DOG_DEF_PEAK_THRESHOLD
            edge_threshold: 10.0, // VL_COVDET_DOG_DEF_EDGE_THRESHOLD
            non_extrema_suppression: 0.5,
            num_non_extrema_suppressed: 0,
            features: Vec::new(),
            gss: None,
            css: None,
            patch: Vec::new(),
            aa_patch: vec![0.0; side * side],
            aa_patch_x: vec![0.0; side * side],
            aa_patch_y: vec![0.0; side * side],
            aa_mask,
        }
    }

    /// Port of `vl_covdet_get_num_non_extrema_suppressed`: features dropped by the last
    /// `detect`'s non-extrema suppression.
    pub fn num_non_extrema_suppressed(&self) -> usize {
        self.num_non_extrema_suppressed
    }

    /// Port of `vl_covdet_get_num_features`.
    pub fn num_features(&self) -> usize {
        self.features.len()
    }

    /// Port of `vl_covdet_get_features`.
    pub fn features(&self) -> &[VlCovDetFeature] {
        &self.features
    }

    /// Port of `vl_covdet_get_features` used for editing the features in place.
    pub fn features_mut(&mut self) -> &mut [VlCovDetFeature] {
        &mut self.features
    }

    /// Port of `vl_covdet_get_gss`: the Gaussian scale space of the last image.
    pub fn gss(&self) -> Option<&VlScaleSpace> {
        self.gss.as_ref()
    }

    /// Port of `vl_covdet_get_css`: the DoG cornerness scale space of the last `detect`.
    pub fn css(&self) -> Option<&VlScaleSpace> {
        self.css.as_ref()
    }

    /// Port of `vl_covdet_append_feature`.
    pub fn append_feature(&mut self, feature: VlCovDetFeature) {
        self.features.push(feature);
    }

    /// Port of `vl_covdet_put_image`: builds the Gaussian scale space of a row-major
    /// `width` x `height` float image, down to octaves of at least 16 pixels on the shorter
    /// side. `Err` for an empty image, a buffer of another size, or an image too small for
    /// the octaves (entry 295).
    pub fn put_image(
        &mut self,
        image: &[f32],
        width: usize,
        height: usize,
        cancel: &CancelToken,
    ) -> Result<()> {
        const MIN_OCTAVE_SIZE: usize = 16;
        check!(width >= 1);
        check!(height >= 1);
        check_eq!(image.len(), width * height);
        let mut geom = VlScaleSpaceGeometry::default_geometry(width, height)?;

        // (minOctaveSize - 1) 2^lastOctave <= min(width,height) - 1
        let w1 = width as f64 - 1.0;
        let h1 = height as f64 - 1.0;
        let log_size = fns::log2((if w1 < h1 { w1 } else { h1 }) / (MIN_OCTAVE_SIZE - 1) as f64);
        // A 1-pixel side gives log2(0) = -inf, which VLFeat converts to an integer (undefined
        // behaviour): no octave fits such an image.
        check!(
            log_size.is_finite(),
            "image too small for the covariant detector"
        );
        let last_octave = vl_floor_d(log_size) as i32;

        geom.width = width;
        geom.height = height;
        geom.first_octave = self.first_octave;
        geom.last_octave = last_octave;
        geom.octave_resolution = self.octave_resolution;
        geom.octave_first_subdivision = -1;
        geom.octave_last_subdivision = self.octave_resolution + 1;

        let mut gss = VlScaleSpace::new(geom)?;
        gss.put_image(image, cancel)?;
        self.gss = Some(gss);
        Ok(())
    }

    /// Port of `vl_covdet_detect`: DoG scale-space extrema, refined and thresholded, from the
    /// coarsest octave down, stopping after the octave in which the count reaches
    /// `max_num_features`; then non-extrema suppression. `Err` before `put_image`.
    pub fn detect(&mut self, max_num_features: usize, cancel: &CancelToken) -> Result<()> {
        let gss = check_notnull!(self.gss.as_ref());
        let geom = gss.geometry();

        // Clear previous detections if any.
        self.features.clear();

        // Prepare buffers.
        let mut cgeom = geom;
        cgeom.octave_last_subdivision -= 1;
        let mut css = VlScaleSpace::new(cgeom)?;

        // Compute cornerness: `_vl_dog_response(clevel, level(s + 1), level(s))`, which
        // writes level(s) - level(s + 1).
        for o in cgeom.first_octave..=cgeom.last_octave {
            for s in cgeom.octave_first_subdivision..=cgeom.octave_last_subdivision {
                let level = gss.level(o, s);
                let next = gss.level(o, s + 1);
                for ((c, &l), &n) in css.level_mut(o, s).iter_mut().zip(level).zip(next) {
                    *c = l - n;
                }
            }
        }

        // Find and refine local maxima.
        let mut extrema = Vec::new();
        let depth = (cgeom.octave_last_subdivision - cgeom.octave_first_subdivision + 1) as usize;
        let mut o = cgeom.last_octave;
        while o >= cgeom.first_octave {
            cancel.check_cancelled()?;
            let octgeom = css.octave_geometry(o);
            let step = octgeom.step;
            let (width, height) = (octgeom.width, octgeom.height);
            let octave = css.octave(o);
            find_local_extrema_3(
                &mut extrema,
                octave,
                width,
                height,
                depth,
                0.8 * self.peak_threshold,
            );
            for e in &extrema {
                let (ok, refined) = refine_local_extremum_3(octave, width, height, depth, *e);
                let ok = ok
                    && f64::from(refined.peak_score).abs() > self.peak_threshold
                    && f64::from(refined.edge_score) < self.edge_threshold;
                if ok {
                    // refined.z + octaveFirstSubdivision is float, and so is its quotient by
                    // the (unsigned) resolution and the sum with o: only pow is double.
                    let exponent = o as f32
                        + (refined.z + cgeom.octave_first_subdivision as f32)
                            / cgeom.octave_resolution as f32;
                    let sigma = cgeom.base_scale * fns::pow(2.0, f64::from(exponent));
                    self.features.push(VlCovDetFeature {
                        frame: VlFrameOrientedEllipse {
                            x: (f64::from(refined.x) * step) as f32,
                            y: (f64::from(refined.y) * step) as f32,
                            a11: sigma as f32,
                            a12: 0.0,
                            a21: 0.0,
                            a22: sigma as f32,
                        },
                        o,
                        // C `round`: half away from zero, as `f64::round`.
                        s: f64::from(refined.z).round() as i32,
                        peak_score: refined.peak_score,
                        edge_score: refined.edge_score,
                        orientation_score: 0.0,
                        laplacian_scale_score: 0.0,
                    });
                }
            }

            if self.features.len() >= max_num_features {
                break;
            }
            o -= 1;
        }
        self.css = Some(css);

        if self.non_extrema_suppression != 0.0 {
            self.suppress_non_extrema(cancel)?;
        }
        Ok(())
    }

    /// The non-extrema suppression of `vl_covdet_detect`: a feature is dropped when a
    /// stronger one sits within `tol * sigma` of it at a scale within a factor `1 + tol`.
    fn suppress_non_extrema(&mut self, cancel: &CancelToken) -> Result<()> {
        let tol = self.non_extrema_suppression;
        let features = &mut self.features;
        self.num_non_extrema_suppressed = 0;
        for i in 0..features.len() {
            if i & 255 == 0 {
                cancel.check_cancelled()?;
            }
            let x = f64::from(features[i].frame.x);
            let y = f64::from(features[i].frame.y);
            let sigma = f64::from(features[i].frame.a11);
            let score = f64::from(features[i].peak_score);
            if score == 0.0 {
                continue;
            }

            // Feature i itself never passes `score.abs() > score_.abs()`.
            for other in features.iter_mut() {
                let dx_ = f64::from(other.frame.x) - x;
                let dy_ = f64::from(other.frame.y) - y;
                let sigma_ = f64::from(other.frame.a11);
                let score_ = f64::from(other.peak_score);
                if score_ == 0.0 {
                    continue;
                }
                if sigma < (1.0 + tol) * sigma_
                    && sigma_ < (1.0 + tol) * sigma
                    && dx_.abs() < tol * sigma
                    && dy_.abs() < tol * sigma
                    && score.abs() > score_.abs()
                {
                    other.peak_score = 0.0;
                    self.num_non_extrema_suppressed += 1;
                }
            }
        }
        features.retain(|f| f.peak_score != 0.0);
        Ok(())
    }
}

/// An `(x, y, z)` sample of a DoG octave volume.
#[derive(Debug, Clone, Copy)]
struct Extremum {
    x: usize,
    y: usize,
    z: usize,
}

/// Port of `VlCovDetExtremum3`'s float part (the integer location is not read by `detect`).
#[derive(Debug, Clone, Copy)]
struct Refined {
    x: f32,
    y: f32,
    z: f32,
    peak_score: f32,
    edge_score: f32,
}

/// Port of `vl_find_local_extrema_3`: the interior samples that are strictly above all 26
/// neighbours and `>= threshold`, or strictly below and `<= -threshold`, in x-fastest order.
fn find_local_extrema_3(
    extrema: &mut Vec<Extremum>,
    map: &[f32],
    width: usize,
    height: usize,
    depth: usize,
    threshold: f64,
) {
    extrema.clear();
    let yo = width as isize;
    let zo = (width * height) as isize;
    let neighbours = |pt: usize, cmp: fn(f32, f32) -> bool| {
        let v = map[pt];
        (-1isize..=1).all(|dz| {
            (-1isize..=1).all(|dy| {
                (-1isize..=1).all(|dx| {
                    (dx == 0 && dy == 0 && dz == 0)
                        || cmp(v, map[(pt as isize + dx + dy * yo + dz * zo) as usize])
                })
            })
        })
    };
    for z in 1..depth.saturating_sub(1) {
        for y in 1..height.saturating_sub(1) {
            for x in 1..width.saturating_sub(1) {
                let pt = x + y * width + z * width * height;
                let value = f64::from(map[pt]);
                if (value >= threshold && neighbours(pt, |v, n| v > n))
                    || (value <= -threshold && neighbours(pt, |v, n| v < n))
                {
                    extrema.push(Extremum { x, y, z });
                }
            }
        }
    }
}

/// Port of `vl_refine_local_extreum_3`: up to five Newton steps on the 3D quadratic fit,
/// moving the integer x/y location while the offset exceeds 0.6. Returns whether the
/// refinement was stable and inside the map, with the refined extremum.
fn refine_local_extremum_3(
    map: &[f32],
    width: usize,
    height: usize,
    depth: usize,
    start: Extremum,
) -> (bool, Refined) {
    let yo = width as isize;
    let zo = (width * height) as isize;
    let (mut x, mut y, z) = (start.x as isize, start.y as isize, start.z as isize);
    let (w, h) = (width as isize, height as isize);
    let (mut d_x, mut d_y, mut d_z) = (0.0f64, 0.0f64, 0.0f64);
    let (mut d_xx, mut d_yy, mut d_xy) = (0.0f64, 0.0f64, 0.0f64);
    let mut b = [0.0f64; 3];
    let mut ok = true;
    let mut dx = 0isize;
    let mut dy = 0isize;
    let mut pt = 0isize;
    let at = |pt: isize, ddx: isize, ddy: isize, ddz: isize| -> f32 {
        map[(pt + ddx + ddy * yo + ddz * zo) as usize]
    };

    for _ in 0..5 {
        x += dx;
        y += dy;
        pt = x + y * yo + z * zo;

        // Compute the gradient: float differences, promoted.
        d_x = 0.5 * f64::from(at(pt, 1, 0, 0) - at(pt, -1, 0, 0));
        d_y = 0.5 * f64::from(at(pt, 0, 1, 0) - at(pt, 0, -1, 0));
        d_z = 0.5 * f64::from(at(pt, 0, 0, 1) - at(pt, 0, 0, -1));

        // Compute the Hessian. The first two samples add in float.
        let c2 = 2.0 * f64::from(at(pt, 0, 0, 0));
        d_xx = f64::from(at(pt, 1, 0, 0) + at(pt, -1, 0, 0)) - c2;
        d_yy = f64::from(at(pt, 0, 1, 0) + at(pt, 0, -1, 0)) - c2;
        let d_zz = f64::from(at(pt, 0, 0, 1) + at(pt, 0, 0, -1)) - c2;

        d_xy = 0.25
            * f64::from(at(pt, 1, 1, 0) + at(pt, -1, -1, 0) - at(pt, -1, 1, 0) - at(pt, 1, -1, 0));
        let d_xz = 0.25
            * f64::from(at(pt, 1, 0, 1) + at(pt, -1, 0, -1) - at(pt, -1, 0, 1) - at(pt, 1, 0, -1));
        let d_yz = 0.25
            * f64::from(at(pt, 0, 1, 1) + at(pt, 0, -1, -1) - at(pt, 0, -1, 1) - at(pt, 0, 1, -1));

        // Solve the linear system (column-major A).
        let a = [d_xx, d_xy, d_xz, d_xy, d_yy, d_yz, d_xz, d_yz, d_zz];
        let rhs = [-d_x, -d_y, -d_z];
        if vl_solve_linear_system_3(&mut b, &a, &rhs).is_err() {
            ok = false;
            b = [0.0; 3];
            break;
        }

        // Keep going if there is sufficient translation.
        dx = isize::from(b[0] > 0.6 && x < w - 2) - isize::from(b[0] < -0.6 && x > 1);
        dy = isize::from(b[1] > 0.6 && y < h - 2) - isize::from(b[1] < -0.6 && y > 1);

        if dx == 0 && dy == 0 {
            break;
        }
    }

    // Check threshold and other conditions.
    let peak_score = f64::from(at(pt, 0, 0, 0)) + 0.5 * (d_x * b[0] + d_y * b[1] + d_z * b[2]);
    let alpha = (d_xx + d_yy) * (d_xx + d_yy) / (d_xx * d_yy - d_xy * d_xy);
    let edge_score = if alpha < 0.0 {
        f64::INFINITY // not an extremum
    } else {
        let m = 0.25 * alpha - 1.0;
        (0.5 * alpha - 1.0) + ((if m > 0.0 { m } else { 0.0 }) * alpha).sqrt()
    };

    let refined = Refined {
        x: (x as f64 + b[0]) as f32,
        y: (y as f64 + b[1]) as f32,
        z: (z as f64 + b[2]) as f32,
        peak_score: peak_score as f32,
        edge_score: edge_score as f32,
    };
    let ok = ok
        && b[0].abs() < 1.5
        && b[1].abs() < 1.5
        && b[2].abs() < 1.5
        && refined.x >= 0.0
        && refined.x <= (width - 1) as f32
        && refined.y >= 0.0
        && refined.y <= (height - 1) as f32
        && refined.z >= 0.0
        && refined.z <= (depth - 1) as f32;
    (ok, refined)
}
