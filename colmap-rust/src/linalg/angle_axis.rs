//! [`AngleAxisd`]: a rotation as an angle (radians) about a unit axis, the replacement for
//! `Eigen::AngleAxisd`. COLMAP uses it in geometry/pose.cc (`RotationMatrixToAngleAxis`,
//! `AngleAxisToRotationMatrix`, `EulerAnglesToRotationMatrix`). Port of colmap-sharp's
//! `LinearAlgebra/AngleAxisd.cs`, written to Eigen's documented semantics from the standard
//! formulas; Eigen is not ported (docs/LICENSE_AUDIT.md). Siblings: [`Quaterniond`] and
//! [`Matrix3d`].
//!
//! Conventions, as Eigen documents them:
//! - The axis is expected to be unit length; nothing normalizes it.
//! - From a quaternion: angle = 2 atan2(|vec|, |w|) in [0, pi], axis = vec / |vec|, flipped
//!   when w < 0 so the angle stays in [0, pi]. Only an exactly zero vector part gives angle 0
//!   about the x axis; tiny ones keep their angle. Oracle-pinned: bit-identical except for
//!   `fns::atan2`'s last bit (docs/CPP_DIVERGENCES.md, entry 1).
//! - From a rotation matrix: through `Quaterniond::from_rotation_matrix`, then as above.
//! - `to_rotation_matrix` is Rodrigues' formula, R = c I + s [a]x + (1 - c) a a^T. pycolmap
//!   exposes no Eigen AngleAxis -> matrix call, so its bits are not oracle-pinned (Tier B).

use super::{maxi, mini, Matrix3d, Quaterniond, Vector3d, DUMMY_PRECISION, MACHINE_EPSILON};
use crate::math::fns;

/// Angle-axis rotation. Replacement for `Eigen::AngleAxisd`.
///
/// Deliberately no `Default`: Eigen's default constructor leaves it uninitialized, and a
/// derived default would carry a zero (non-unit) axis. Construct it with [`AngleAxisd::new`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AngleAxisd {
    /// Rotation angle in radians.
    pub angle: f64,
    /// Rotation axis; expected unit length.
    pub axis: Vector3d,
}

impl AngleAxisd {
    /// The rotation by `angle` (radians) about `axis` (expected unit length).
    pub const fn new(angle: f64, axis: Vector3d) -> Self {
        Self { angle, axis }
    }

    /// The angle-axis form of a quaternion (see the file header). The quaternion need not be
    /// normalized; the angle comes from atan2, which is scale-invariant.
    pub fn from_quaternion(q: Quaterniond) -> Self {
        let vec = q.vec();

        // Only an exactly zero vector part has no axis. Anything else, however small, is a
        // real (tiny) rotation: the oracle gives angle 2e-17 for (x=1e-17, w=1) and 2e-200 for
        // (x=1e-200, w=1), so the norm must not underflow either.
        if vec.x == 0.0 && vec.y == 0.0 && vec.z == 0.0 {
            return Self::new(0.0, Vector3d::unit_x());
        }

        let n = vector_part_norm(vec);
        let angle = 2.0 * fns::atan2(n, q.w.abs());
        let axis = if q.w < 0.0 { -vec / n } else { vec / n };
        Self::new(angle, axis)
    }

    /// The angle-axis form of a rotation matrix, through its quaternion.
    pub fn from_rotation_matrix(rotation: Matrix3d) -> Self {
        Self::from_quaternion(Quaterniond::from_rotation_matrix(rotation))
    }

    /// The unit quaternion `(cos(a/2), sin(a/2) axis)`.
    pub fn to_quaternion(self) -> Quaterniond {
        Quaterniond::from_angle_axis(self)
    }

    /// The rotation matrix by Rodrigues' formula in its textbook element form, with
    /// c = cos(angle), s = sin(angle), t = 1 - c and unit axis (x, y, z):
    /// R = [t x x + c, t x y - s z, t x z + s y; t x y + s z, t y y + c, t y z - s x;
    /// t x z - s y, t y z + s x, t z z + c]. Products evaluate left to right, (t x) y.
    pub fn to_rotation_matrix(self) -> Matrix3d {
        let c = fns::cos(self.angle);
        let s = fns::sin(self.angle);
        let t = 1.0 - c;
        let Vector3d { x, y, z } = self.axis;

        Matrix3d::new(
            t * x * x + c,
            t * x * y - s * z,
            t * x * z + s * y,
            t * x * y + s * z,
            t * y * y + c,
            t * y * z - s * x,
            t * x * z - s * y,
            t * y * z + s * x,
            t * z * z + c,
        )
    }

    /// Eigen's AngleAxis `isApprox` at [`DUMMY_PRECISION`].
    pub fn is_approx(self, other: Self) -> bool {
        self.is_approx_with(other, DUMMY_PRECISION)
    }

    /// Eigen's AngleAxis `isApprox`: the axes are approximately equal and the angles are
    /// within `precision * min(|a|, |b|)` of each other.
    pub fn is_approx_with(self, other: Self, precision: f64) -> bool {
        self.axis.is_approx_with(other.axis, precision)
            && (self.angle - other.angle).abs()
                <= precision * mini(self.angle.abs(), other.angle.abs())
    }
}

// The norm of a nonzero vector part. Found by colmap-sharp with
// oracle/linear_algebra_rotations.py, whose 138 cases this reproduces bit for bit:
// - normally the plain Vector3d::norm, sqrt(x*x + y*y + z*z);
// - below machine epsilon, the norm of the vector divided by its largest magnitude, times that
//   magnitude, so tiny squares cannot underflow. The plain norm is wrong there even without
//   underflow: for (1e-17, -2e-17, 0) it is 1 ulp off the oracle.
// Scaling by a power of two, or by the reciprocal of the largest magnitude, each miss at least
// one fixture case.
fn vector_part_norm(v: Vector3d) -> f64 {
    let n = v.norm();
    if n >= MACHINE_EPSILON {
        return n;
    }
    // C#'s Math.Max(a, Math.Max(b, c)); the magnitudes are never NaN here.
    let largest = maxi(v.x.abs(), maxi(v.y.abs(), v.z.abs()));
    largest * (v / largest).norm()
}
