//! Port of COLMAP's `colmap/geometry/pose_prior.h` and `pose_prior.cc`: [`PosePrior`] (a
//! position and gravity prior for one sensor measurement), its NaN-aware equality and
//! `operator<<`, and the EXIF-orientation gravity helpers. Port of colmap-sharp's
//! `Geometry/PosePrior.cs`. Tests: `tests/geometry/pose_prior.rs` (pose_prior_test.cc).
//! Tier A.

use std::fmt;

use super::rigid3::format_list;
use crate::linalg::{Matrix3d, Vector3d};
use crate::math::fns;
use crate::util::types::{DataId, PosePriorId, INVALID_DATA_ID, INVALID_POSE_PRIOR_ID};

/// `PosePrior::CoordinateSystem` (`MAKE_ENUM_CLASS(CoordinateSystem, -1, UNDEFINED, WGS84,
/// CARTESIAN)`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum CoordinateSystem {
    /// `UNDEFINED` (-1), the default.
    #[default]
    Undefined = -1,
    /// `WGS84` (0).
    Wgs84 = 0,
    /// `CARTESIAN` (1).
    Cartesian = 1,
}

impl CoordinateSystem {
    /// `CoordinateSystemToString`: "UNDEFINED", "WGS84" or "CARTESIAN".
    pub fn as_str(self) -> &'static str {
        match self {
            CoordinateSystem::Undefined => "UNDEFINED",
            CoordinateSystem::Wgs84 => "WGS84",
            CoordinateSystem::Cartesian => "CARTESIAN",
        }
    }
}

/// A pose prior for one sensor measurement. Port of `colmap::PosePrior`.
#[derive(Clone, Copy, Debug)]
pub struct PosePrior {
    /// The unique identifier of the pose prior.
    pub pose_prior_id: PosePriorId,
    /// The measurement (e.g. an image of a camera, or an IMU sample) whose pose this prior
    /// defines.
    pub corr_data_id: DataId,
    /// The position of the associated sensor in the world coordinate system (NaN if unset).
    pub position: Vector3d,
    /// The position covariance in the Cartesian world coordinate system (NaN if unset).
    pub position_covariance: Matrix3d,
    /// The coordinate system of the position in the world.
    pub coordinate_system: CoordinateSystem,
    /// The gravity (down) in the sensor coordinate system (NaN if unset).
    pub gravity: Vector3d,
}

impl Default for PosePrior {
    /// COLMAP's member initializers: invalid ids, NaN position, covariance and gravity, and
    /// an undefined coordinate system.
    fn default() -> Self {
        Self {
            pose_prior_id: INVALID_POSE_PRIOR_ID,
            corr_data_id: INVALID_DATA_ID,
            position: Vector3d::new(f64::NAN, f64::NAN, f64::NAN),
            position_covariance: Matrix3d::from_column_major([f64::NAN; 9]),
            coordinate_system: CoordinateSystem::Undefined,
            gravity: Vector3d::new(f64::NAN, f64::NAN, f64::NAN),
        }
    }
}

impl PosePrior {
    /// `HasPosition()`: every position coefficient is finite.
    pub fn has_position(&self) -> bool {
        self.position.to_array().iter().all(|v| v.is_finite())
    }

    /// `HasPositionCov()`: every covariance coefficient is finite.
    pub fn has_position_cov(&self) -> bool {
        self.position_covariance
            .as_slice()
            .iter()
            .all(|v| v.is_finite())
    }

    /// `HasGravity()`: every gravity coefficient is finite.
    pub fn has_gravity(&self) -> bool {
        self.gravity.to_array().iter().all(|v| v.is_finite())
    }
}

/// COLMAP's `IsNaNEqual`: coefficient-wise equality where NaN equals NaN (the default C++
/// comparison returns false for NaN == NaN).
fn is_nan_equal(left: &[f64], right: &[f64]) -> bool {
    left.iter()
        .zip(right)
        .all(|(&l, &r)| l.is_nan() == r.is_nan() && (l.is_nan() || l == r))
}

/// COLMAP's `operator==`: ids and coordinate system equal, and position, covariance and
/// gravity equal with NaN treated as equal to NaN.
impl PartialEq for PosePrior {
    fn eq(&self, other: &Self) -> bool {
        self.pose_prior_id == other.pose_prior_id
            && self.corr_data_id == other.corr_data_id
            && self.coordinate_system == other.coordinate_system
            && is_nan_equal(&self.position.to_array(), &other.position.to_array())
            && is_nan_equal(
                self.position_covariance.as_slice(),
                other.position_covariance.as_slice(),
            )
            && is_nan_equal(&self.gravity.to_array(), &other.gravity.to_array())
    }
}

/// COLMAP's `operator<<`, e.g. `PosePrior(pose_prior_id=0, corr_data_id=(CAMERA, 1, 2),
/// position=[0, 0, 0], position_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1],
/// coordinate_system=CARTESIAN, gravity=[0, 0, 1])`. The covariance prints row by row, as
/// Eigen's `IOFormat` with `", "` row and coefficient separators does.
impl fmt::Display for PosePrior {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let c = &self.position_covariance;
        let covariance_rows: Vec<f64> = (0..3)
            .flat_map(|row| (0..3).map(move |col| c[(row, col)]))
            .collect();
        write!(
            f,
            "PosePrior(pose_prior_id={}, corr_data_id=({}, {}, {}), position=[{}], \
             position_covariance=[{}], coordinate_system={}, gravity=[{}])",
            self.pose_prior_id,
            self.corr_data_id.sensor_id.sensor_type,
            self.corr_data_id.sensor_id.id,
            self.corr_data_id.id,
            format_list(&self.position.to_array()),
            format_list(&covariance_rows),
            self.coordinate_system.as_str(),
            format_list(&self.gravity.to_array())
        )
    }
}

/// Port of `colmap::GravityFromExifOrientation`: the image-space gravity of an upright EXIF
/// orientation (1, 3, 6 or 8), `None` otherwise. COLMAP also logs a warning for the mirrored
/// orientations (2, 4, 5, 7) and an error for unknown values; the port has no logger yet, so
/// only the `None` remains (docs/CPP_DIVERGENCES.md entry 84).
pub fn gravity_from_exif_orientation(orientation: i32) -> Option<Vector3d> {
    match orientation {
        1 => Some(Vector3d::new(0.0, 1.0, 0.0)),  // Normal
        3 => Some(Vector3d::new(0.0, -1.0, 0.0)), // Rotate 180
        6 => Some(Vector3d::new(1.0, 0.0, 0.0)),  // Rotate 90 CW
        8 => Some(Vector3d::new(-1.0, 0.0, 0.0)), // Rotate 270 CW
        _ => None,
    }
}

/// Port of `colmap::ComputeRot90FromGravity`: the number of 90 degree counter-clockwise
/// rotations (0..=3) that make the sensor upright, i.e. bring the image-space gravity angle
/// to pi/2.
pub fn compute_rot90_from_gravity(gravity: Vector3d) -> i32 {
    let angle = fns::atan2(gravity.y, gravity.x);
    const HALF_PI: f64 = std::f64::consts::PI / 2.0;
    // `std::round` rounds half away from zero, like `f64::round`; the value is in [-3, 1].
    let mut rot90_ccw = (((angle - HALF_PI) / HALF_PI).round() as i32) % 4;
    if rot90_ccw < 0 {
        rot90_ccw += 4;
    }
    rot90_ccw
}
