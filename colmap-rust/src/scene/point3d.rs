//! Port of `colmap/scene/point3d.h` / `point3d.cc`: [`Point3D`], a triangulated 3D point
//! with its mean reprojection error, color and [`Track`] of observations. Neighbors:
//! [`super::point2d`], [`super::track`]. Tests: `tests/scene/point3d.rs` (`point3d_test.cc`
//! 1:1).
//!
//! Tier A (plain data).

use std::fmt;

use super::track::Track;
use crate::linalg::{Vector3d, Vector3ub};
use crate::util::stream_format::{format_double, DEFAULT_PRECISION};

/// Port of `colmap::Point3D`: 3D point class that holds information about triangulated 2D
/// points.
#[derive(Debug, Clone, PartialEq)]
pub struct Point3D {
    /// The 3D position of the point.
    pub xyz: Vector3d,
    /// The mean reprojection error in pixels; -1 when unset.
    pub error: f64,
    /// The track of the point as a list of image observations.
    pub track: Track,
    /// The color of the point in the range [0, 255].
    pub color: Vector3ub,
}

impl Default for Point3D {
    fn default() -> Self {
        Self {
            xyz: Vector3d::zeros(),
            error: -1.0,
            track: Track::new(),
            color: Vector3ub::default(),
        }
    }
}

impl Point3D {
    /// `HasError`: whether a reprojection error was set (anything but -1).
    pub fn has_error(&self) -> bool {
        self.error != -1.0
    }
}

impl fmt::Display for Point3D {
    /// Port of `operator<<(std::ostream&, const Point3D&)`:
    /// `Point3D(xyz=[x, y, z], track_len=n)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Point3D(xyz=[{}, {}, {}], track_len={})",
            format_double(self.xyz.x, DEFAULT_PRECISION),
            format_double(self.xyz.y, DEFAULT_PRECISION),
            format_double(self.xyz.z, DEFAULT_PRECISION),
            self.track.length()
        )
    }
}
