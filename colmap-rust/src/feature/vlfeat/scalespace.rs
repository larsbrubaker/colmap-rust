//! Port of VLFeat's Gaussian scale space (`thirdparty/VLFeat/scalespace.h` / `.c`) as ported in
//! colmap-sharp's `Feature/VLFeat/VlScaleSpace.cs`: the geometry, the per-octave level storage
//! and `vl_scalespace_put_image` with its upsample / downsample / fill helpers. The covariant
//! detector keeps its Gaussian and cornerness scale spaces in it. VLFeat's SIFT filter
//! ([`super::sift_filter`]) has its own, different, scale space and does not use this.
//! Smoothing goes through [`super::imopv_smooth`].
//!
//! Tier A (exact). Kept quirks: `_vl_scalespace_fill_octave` takes the incremental sigma with
//! `sqrtf` (a float square root of the double difference rounded to float); the default
//! geometry's base scale uses octave resolution 3 whatever resolution the caller then sets.
//! `pow` and `log2` go through [`crate::math::fns`] (docs/CPP_DIVERGENCES.md entry 1).
//!
//! Translation notes:
//! - VLFeat's octaves are separately allocated float arrays (uninitialized `vl_malloc`); here
//!   each is a zeroed `Vec<f32>`, and a level is a slice of its octave. Every level is written
//!   by `put_image` before it is read, so the initial contents never reach an output.
//! - `vl_scalespace_new_with_geometry`'s `assert(is_valid_geometry(geom))` and the image-size
//!   asserts of `vl_scalespace_get_default_geometry` are `Err`s; `vl_scalespace_get_level`'s
//!   range asserts are panics (an internal invariant of the callers). `put_image` with a
//!   negative first octave and no octave 0, where VLFeat writes out of bounds, is an `Err`
//!   (docs/CPP_DIVERGENCES.md entry 295).
//! - `vl_scalespacegeometry_is_equal` (which compares `octaveFirstSubdivision` against
//!   `octaveLastSubdivision`, an upstream typo) is not ported: the covariant detector builds a
//!   new scale space per image, so a scale space is never reused.

use super::imopv_smooth::{vl_imsmooth_f_within, ImageRegion};
use crate::math::fns;
use crate::util::cancellation::CancelToken;
use crate::util::check::Result;
use crate::{check, check_ge};

/// Port of `VlScaleSpaceGeometry`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VlScaleSpaceGeometry {
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
    /// Index of the first octave.
    pub first_octave: i32,
    /// Index of the last octave.
    pub last_octave: i32,
    /// Number of octave subdivisions.
    pub octave_resolution: i32,
    /// Index of the first octave subdivision.
    pub octave_first_subdivision: i32,
    /// Index of the last octave subdivision.
    pub octave_last_subdivision: i32,
    /// Base smoothing (smoothing of octave 0, level 0).
    pub base_scale: f64,
    /// Nominal smoothing of the original image.
    pub nominal_scale: f64,
}

impl VlScaleSpaceGeometry {
    /// Port of `vl_scalespace_get_default_geometry`. `Err` for an empty image.
    pub fn default_geometry(width: usize, height: usize) -> Result<Self> {
        check!(width >= 1);
        check!(height >= 1);
        // `VL_MAX(floor(vl_log2_d(VL_MIN(width, height))) - 3, 0)`, stored in a vl_index.
        let last_octave = (fns::log2(width.min(height) as f64).floor() - 3.0).max(0.0) as i32;
        let octave_resolution = 3;
        Ok(Self {
            width,
            height,
            first_octave: 0,
            last_octave,
            octave_resolution,
            octave_first_subdivision: 0,
            octave_last_subdivision: octave_resolution - 1,
            base_scale: 1.6 * fns::pow(2.0, 1.0 / f64::from(octave_resolution)),
            nominal_scale: 0.5,
        })
    }
}

/// Port of `VlScaleSpaceOctaveGeometry`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VlScaleSpaceOctaveGeometry {
    /// Octave width.
    pub width: usize,
    /// Octave height.
    pub height: usize,
    /// Sampling step (size of a pixel in the original image).
    pub step: f64,
}

/// `VL_SHIFT_LEFT(x, n)`: `x << n`, or `x >> -n` for negative `n`.
pub(crate) fn vl_shift_left(x: usize, n: i32) -> usize {
    if n >= 0 {
        x << n
    } else {
        x >> -n
    }
}

/// Port of `VlScaleSpace` and the `vl_scalespace_*` functions.
#[derive(Debug, Clone)]
pub struct VlScaleSpace {
    geom: VlScaleSpaceGeometry,
    /// One buffer per octave, its levels one after the other.
    octaves: Vec<Vec<f32>>,
}

impl VlScaleSpace {
    /// Port of `vl_scalespace_new_with_geometry`. `Err` when the geometry is not valid
    /// (`is_valid_geometry`).
    pub fn new(geom: VlScaleSpaceGeometry) -> Result<Self> {
        check!(geom.first_octave <= geom.last_octave);
        check!(geom.octave_resolution >= 1);
        check!(geom.octave_first_subdivision <= geom.octave_last_subdivision);
        check_ge!(geom.base_scale, 0.0);
        check_ge!(geom.nominal_scale, 0.0);
        let num_sublevels =
            (geom.octave_last_subdivision - geom.octave_first_subdivision + 1) as usize;
        let mut space = Self {
            geom,
            octaves: Vec::new(),
        };
        space.octaves = (geom.first_octave..=geom.last_octave)
            .map(|o| {
                let ogeom = space.octave_geometry(o);
                vec![0.0f32; ogeom.width * ogeom.height * num_sublevels]
            })
            .collect();
        Ok(space)
    }

    /// Port of `vl_scalespace_new`: the scale space of the default geometry.
    pub fn with_default_geometry(width: usize, height: usize) -> Result<Self> {
        Self::new(VlScaleSpaceGeometry::default_geometry(width, height)?)
    }

    /// Port of `vl_scalespace_get_geometry`.
    pub fn geometry(&self) -> VlScaleSpaceGeometry {
        self.geom
    }

    /// Port of `vl_scalespace_get_octave_geometry`.
    pub fn octave_geometry(&self, o: i32) -> VlScaleSpaceOctaveGeometry {
        VlScaleSpaceOctaveGeometry {
            width: vl_shift_left(self.geom.width, -o),
            height: vl_shift_left(self.geom.height, -o),
            step: fns::pow(2.0, f64::from(o)),
        }
    }

    /// Where level `(o, s)` starts in its octave's buffer, and that buffer's index. Panics when
    /// `(o, s)` is outside the geometry (VLFeat's asserts).
    fn level_location(&self, o: i32, s: i32) -> (usize, usize) {
        let ogeom = self.octave_geometry(o);
        assert!(o >= self.geom.first_octave && o <= self.geom.last_octave);
        assert!(s >= self.geom.octave_first_subdivision && s <= self.geom.octave_last_subdivision);
        (
            (o - self.geom.first_octave) as usize,
            ogeom.width * ogeom.height * (s - self.geom.octave_first_subdivision) as usize,
        )
    }

    /// Port of `vl_scalespace_get_level`: the pixels of level `(o, s)`, row-major at the
    /// octave's size. Panics outside the geometry.
    pub fn level(&self, o: i32, s: i32) -> &[f32] {
        let (index, offset) = self.level_location(o, s);
        let ogeom = self.octave_geometry(o);
        &self.octaves[index][offset..offset + ogeom.width * ogeom.height]
    }

    /// Every level of octave `o`, one after the other from the first subdivision: what
    /// `vl_scalespace_get_level(o, octaveFirstSubdivision)` points into when the covariant
    /// detector scans the octave as a 3D volume. Panics outside the geometry.
    pub fn octave(&self, o: i32) -> &[f32] {
        let (index, _) = self.level_location(o, self.geom.octave_first_subdivision);
        &self.octaves[index]
    }

    /// The writable pixels of level `(o, s)` (`vl_scalespace_get_level` used as an output).
    pub fn level_mut(&mut self, o: i32, s: i32) -> &mut [f32] {
        let (index, offset) = self.level_location(o, s);
        let ogeom = self.octave_geometry(o);
        &mut self.octaves[index][offset..offset + ogeom.width * ogeom.height]
    }

    /// Port of `vl_scalespace_get_level_sigma`.
    pub fn level_sigma(&self, o: i32, s: i32) -> f64 {
        self.geom.base_scale
            * fns::pow(
                2.0,
                f64::from(o) + f64::from(s) / f64::from(self.geom.octave_resolution),
            )
    }

    /// Port of `vl_scalespace_put_image`: builds every level from `image` (row-major floats of
    /// the geometry's size). `cancel` is polled before each octave. `Err` when the first
    /// octave is negative and octave 0 is not in the geometry.
    pub fn put_image(&mut self, image: &[f32], cancel: &CancelToken) -> Result<()> {
        // A negative first octave is upsampled from octave 0, which VLFeat writes whether or
        // not the geometry has it (out of bounds when it does not); here that is an error.
        check!(self.geom.first_octave >= 0 || self.geom.last_octave >= 0);
        cancel.check_cancelled()?;
        self.start_octave_from_image(image, self.geom.first_octave);
        self.fill_octave(self.geom.first_octave);
        for o in self.geom.first_octave + 1..=self.geom.last_octave {
            cancel.check_cancelled()?;
            self.start_octave_from_previous_octave(o);
            self.fill_octave(o);
        }
        Ok(())
    }

    /// Smooths level `(o, src_s)` into level `(o, dst_s)` (possibly the same level) by `sigma`
    /// in octave pixels.
    fn smooth_level(&mut self, o: i32, dst_s: i32, src_s: i32, sigma: f64) {
        let ogeom = self.octave_geometry(o);
        let (index, dst_offset) = self.level_location(o, dst_s);
        let (_, src_offset) = self.level_location(o, src_s);
        vl_imsmooth_f_within(
            &mut self.octaves[index],
            dst_offset,
            ogeom.width,
            ImageRegion {
                offset: src_offset,
                width: ogeom.width,
                height: ogeom.height,
                stride: ogeom.width,
            },
            sigma,
            sigma,
        );
    }

    /// Port of `_vl_scalespace_fill_octave`: each level after the first is the previous one
    /// smoothed by the missing sigma.
    fn fill_octave(&mut self, o: i32) {
        let step = self.octave_geometry(o).step;
        for s in self.geom.octave_first_subdivision + 1..=self.geom.octave_last_subdivision {
            let sigma = self.level_sigma(o, s);
            let previous_sigma = self.level_sigma(o, s - 1);
            // `sqrtf`: the double difference rounded to float, square-rooted in float.
            let delta_sigma =
                f64::from(((sigma * sigma - previous_sigma * previous_sigma) as f32).sqrt());
            self.smooth_level(o, s, s - 1, delta_sigma / step);
        }
    }

    /// Port of `_vl_scalespace_start_octave_from_image`.
    fn start_octave_from_image(&mut self, image: &[f32], o: i32) {
        let first = self.geom.octave_first_subdivision;

        // Copy the image to the first subdivision of octave o, upscaling or downscaling as
        // needed. For o < 0 the image first lands in octave 0, which therefore must exist.
        let (width, height) = (self.geom.width, self.geom.height);
        copy_and_downsample(
            self.level_mut(o.max(0), first),
            image,
            width,
            height,
            o.max(0),
        );

        let mut op = -1;
        while op >= o {
            let succ_geom = self.octave_geometry(op + 1);
            let (succ_index, succ_offset) = self.level_location(op + 1, first);
            let (index, offset) = self.level_location(op, first);
            // op's buffer comes before op + 1's.
            let (lower, upper) = self.octaves.split_at_mut(succ_index);
            copy_and_upsample(
                &mut lower[index][offset..],
                &upper[0][succ_offset..],
                succ_geom.width,
                succ_geom.height,
            );
            op -= 1;
        }

        // Adjust the smoothing of the first level just initialised, accounting for the fact
        // that the input image is assumed to be a nominal scale level.
        let sigma = self.level_sigma(o, first);
        let image_sigma = self.geom.nominal_scale;
        if sigma > image_sigma {
            let step = self.octave_geometry(o).step;
            let delta_sigma = (sigma * sigma - image_sigma * image_sigma).sqrt();
            self.smooth_level(o, first, first, delta_sigma / step);
        }
    }

    /// Port of `_vl_scalespace_start_octave_from_previous_octave`.
    fn start_octave_from_previous_octave(&mut self, o: i32) {
        let first = self.geom.octave_first_subdivision;

        // From the previous octave pick the level closest to this octave's first subdivision.
        let prev_level_index =
            (first + self.geom.octave_resolution).min(self.geom.octave_last_subdivision);
        let prev_geom = self.octave_geometry(o - 1);
        let (prev_index, prev_offset) = self.level_location(o - 1, prev_level_index);
        let (index, offset) = self.level_location(o, first);
        let (lower, upper) = self.octaves.split_at_mut(index);
        copy_and_downsample(
            &mut upper[0][offset..],
            &lower[prev_index][prev_offset..],
            prev_geom.width,
            prev_geom.height,
            1,
        );

        // Add remaining smoothing, if any.
        let sigma = self.level_sigma(o, first);
        let prev_sigma = self.level_sigma(o - 1, prev_level_index);
        if sigma > prev_sigma {
            let step = self.octave_geometry(o).step;
            let delta_sigma = (sigma * sigma - prev_sigma * prev_sigma).sqrt();
            self.smooth_level(o, first, first, delta_sigma / step);
        }
    }
}

/// Port of `copy_and_upsample`: doubles the resolution of the `width` x `height` `source` by
/// bilinear interpolation, reading the last row/column twice at the far border.
fn copy_and_upsample(destination: &mut [f32], source: &[f32], width: usize, height: usize) {
    let mut d = 0;
    let mut s = 0;
    for y in 0..height {
        let oy = if y + 1 < height { width } else { 0 };
        let mut v10 = source[s];
        let mut v11 = source[s + oy];
        for x in 0..width {
            let ox = usize::from(x + 1 < width);
            let v00 = v10;
            let v01 = v11;
            v10 = source[s + ox];
            v11 = source[s + ox + oy];
            destination[d] = v00;
            destination[d + 1] = 0.5 * (v00 + v10);
            destination[d + 2 * width] = 0.5 * (v00 + v01);
            destination[d + 2 * width + 1] = 0.25 * (v00 + v01 + v10 + v11);
            d += 2;
            s += 1;
        }
        d += 2 * width;
    }
}

/// Port of `copy_and_downsample`: keeps every `2^num_octaves`-th pixel of every
/// `2^num_octaves`-th row of the `width` x `height` `source`.
fn copy_and_downsample(
    destination: &mut [f32],
    source: &[f32],
    width: usize,
    height: usize,
    num_octaves: i32,
) {
    if num_octaves == 0 {
        destination[..width * height].copy_from_slice(&source[..width * height]);
        return;
    }
    let step = 1usize << num_octaves;
    let mut d = 0;
    for y in (0..height).step_by(step) {
        let mut p = y * width;
        let mut x = 0;
        // `x < (signed)width - ((signed)step - 1)`, in signed arithmetic.
        while (x as isize) < width as isize - (step as isize - 1) {
            destination[d] = source[p];
            d += 1;
            p += step;
            x += step;
        }
    }
}
