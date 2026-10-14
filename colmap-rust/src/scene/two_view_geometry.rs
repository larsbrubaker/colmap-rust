//! Port of `colmap/scene/two_view_geometry.h` / `.cc`: [`TwoViewGeometry`], the geometric
//! relation verified between two images — its configuration, the E/F/H matrices, the
//! relative pose, the intrinsics a two-view solver recovered, and the inlier feature matches
//! ([`crate::feature::types::FeatureMatches`]). The two-view estimator (Phase 6) produces it,
//! the database stores it and the correspondence graph holds one per image pair. Port of
//! colmap-sharp's `Scene/TwoViewGeometry.cs`.
//! Tests: `tests/scene/two_view_geometry.rs` (`two_view_geometry_test.cc` 1:1).
//!
//! Tier B for [`TwoViewGeometry::invert`] (`H` goes through a 3x3 inverse); everything else
//! is Tier A bookkeeping.
//!
//! COLMAP's `int config` holding a `ConfigurationType` is the enum [`ConfigurationType`]
//! itself, with COLMAP's numeric values ([`ConfigurationType::as_i32`] /
//! [`ConfigurationType::from_i32`] convert for storage), as in colmap-sharp. The
//! out-of-line copy/move members exist in C++ only to silence a GCC warning; here they are
//! the derived `Clone`.

use crate::feature::types::FeatureMatches;
use crate::geometry::rigid3::Rigid3d;
use crate::linalg::Matrix3d;
use crate::scene::camera::Camera;

/// Port of `TwoViewGeometry::ConfigurationType`, with COLMAP's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum ConfigurationType {
    /// `UNDEFINED` (0).
    #[default]
    Undefined = 0,
    /// `DEGENERATE` (1): degenerate configuration (e.g., no overlap or not enough inliers).
    Degenerate = 1,
    /// `CALIBRATED` (2): essential matrix.
    Calibrated = 2,
    /// `UNCALIBRATED` (3): fundamental matrix. A pair whose focal length(s) were recovered by
    /// the two-view solver (e.g. the shared-focal essential-matrix solver) is also
    /// `UNCALIBRATED`: it carries F built from the estimated focal, and exposes the estimated
    /// intrinsics via `camera1` / `camera2` so consumers can seed them without trusting the
    /// camera's placeholder focal.
    Uncalibrated = 3,
    /// `PLANAR` (4): homography, planar scene with baseline.
    Planar = 4,
    /// `PANORAMIC` (5): homography, pure rotation without baseline.
    Panoramic = 5,
    /// `PLANAR_OR_PANORAMIC` (6): homography, planar or panoramic.
    PlanarOrPanoramic = 6,
    /// `WATERMARK` (7): pure 2D translation in image borders.
    Watermark = 7,
    /// `MULTIPLE` (8): multi-model configuration, i.e. the inlier matches result from
    /// multiple individual, non-degenerate configurations.
    Multiple = 8,
    /// `CALIBRATED_RIG` (9): relative pose (metric) from a calibrated (non-panoramic) rig.
    CalibratedRig = 9,
}

impl ConfigurationType {
    /// The `int` COLMAP stores for this configuration.
    pub const fn as_i32(self) -> i32 {
        self as i32
    }

    /// The configuration for a stored `int`, `None` for a value COLMAP does not define.
    pub const fn from_i32(value: i32) -> Option<ConfigurationType> {
        Some(match value {
            0 => ConfigurationType::Undefined,
            1 => ConfigurationType::Degenerate,
            2 => ConfigurationType::Calibrated,
            3 => ConfigurationType::Uncalibrated,
            4 => ConfigurationType::Planar,
            5 => ConfigurationType::Panoramic,
            6 => ConfigurationType::PlanarOrPanoramic,
            7 => ConfigurationType::Watermark,
            8 => ConfigurationType::Multiple,
            9 => ConfigurationType::CalibratedRig,
            _ => return None,
        })
    }
}

/// Port of `colmap::TwoViewGeometry`: two-view geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct TwoViewGeometry {
    /// The configuration of the two-view geometry.
    pub config: ConfigurationType,
    /// `E`: essential matrix.
    pub e: Option<Matrix3d>,
    /// `F`: fundamental matrix.
    pub f: Option<Matrix3d>,
    /// `H`: homography matrix.
    pub h: Option<Matrix3d>,
    /// Relative pose.
    pub cam2_from_cam1: Option<Rigid3d>,
    /// Side 1's intrinsics recovered by the two-view solver, `None` if that side was not
    /// estimated.
    pub camera1: Option<Camera>,
    /// Side 2's intrinsics recovered by the two-view solver, `None` if that side was not
    /// estimated.
    pub camera2: Option<Camera>,
    /// Inlier matches of the configuration.
    pub inlier_matches: FeatureMatches,
    /// Median triangulation angle; -1 when not computed.
    pub tri_angle: f64,
}

impl Default for TwoViewGeometry {
    /// `TwoViewGeometry()`: `UNDEFINED`, nothing estimated, no matches, `tri_angle` -1.
    fn default() -> Self {
        Self {
            config: ConfigurationType::Undefined,
            e: None,
            f: None,
            h: None,
            cam2_from_cam1: None,
            camera1: None,
            camera2: None,
            inlier_matches: FeatureMatches::new(),
            tri_angle: -1.0,
        }
    }
}

impl TwoViewGeometry {
    /// `Invert`: inverts the geometry to match swapped cameras — F and E transposed, H
    /// inverted, the relative pose inverted, the two cameras swapped and each inlier match's
    /// indices swapped.
    pub fn invert(&mut self) {
        if let Some(f) = self.f.as_mut() {
            *f = f.transpose();
        }
        if let Some(e) = self.e.as_mut() {
            *e = e.transpose();
        }
        if let Some(h) = self.h.as_mut() {
            *h = h.inverse();
        }
        if let Some(cam2_from_cam1) = self.cam2_from_cam1.as_mut() {
            *cam2_from_cam1 = cam2_from_cam1.inverse();
        }
        std::mem::swap(&mut self.camera1, &mut self.camera2);
        for feature_match in &mut self.inlier_matches {
            std::mem::swap(
                &mut feature_match.point2d_idx1,
                &mut feature_match.point2d_idx2,
            );
        }
    }
}
