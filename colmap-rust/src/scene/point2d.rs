//! Port of `colmap/scene/point2d.h` / `point2d.cc`: [`Point2D`], a feature in an image and
//! the 3D point it observes, if any. Neighbors: [`super::point3d`] (the triangulated
//! points), [`super::track`]. Tests: `tests/scene/point2d.rs` (`point2d_test.cc` 1:1).
//!
//! Tier A (plain data).

use std::fmt;

use crate::linalg::Vector2d;
use crate::util::stream_format::{format_double, DEFAULT_PRECISION};
use crate::util::types::{Point3DId, INVALID_POINT3D_ID};

/// Port of `colmap::Point2D`. 2D point class corresponds to a feature in an image. It may or
/// may not have a corresponding 3D point if it is part of a triangulated track.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point2D {
    /// The image coordinates in pixels, starting at upper left corner with 0.
    pub xy: Vector2d,
    /// The identifier of the 3D point. If the 2D point is not part of a 3D point track the
    /// identifier is [`INVALID_POINT3D_ID`] and [`Point2D::has_point3d`] is false.
    pub point3d_id: Point3DId,
}

impl Default for Point2D {
    fn default() -> Self {
        Self {
            xy: Vector2d::zeros(),
            point3d_id: INVALID_POINT3D_ID,
        }
    }
}

impl Point2D {
    /// `HasPoint3D`: whether the 2D point observes a 3D point.
    pub fn has_point3d(&self) -> bool {
        self.point3d_id != INVALID_POINT3D_ID
    }
}

impl fmt::Display for Point2D {
    /// Port of `operator<<(std::ostream&, const Point2D&)`:
    /// `Point2D(xy=[x, y], point3D_id=<id|-1>)`, the coordinates at the stream's default
    /// precision.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let point3d_id = if self.has_point3d() {
            self.point3d_id.to_string()
        } else {
            "-1".to_string()
        };
        write!(
            f,
            "Point2D(xy=[{}, {}], point3D_id={point3d_id})",
            format_double(self.xy.x, DEFAULT_PRECISION),
            format_double(self.xy.y, DEFAULT_PRECISION)
        )
    }
}
