//! [`AlignedBox3d`]: axis-aligned 3D box of doubles, the replacement for
//! `Eigen::AlignedBox3d` (COLMAP's bounding-box queries). Port of colmap-sharp's
//! `LinearAlgebra/AlignedBox3d.cs`, written to Eigen's documented semantics; Eigen is not
//! ported (docs/LICENSE_AUDIT.md).
//!
//! Semantics, as Eigen documents them: the default box is empty, with min at `f64::MAX` and
//! max at `f64::MIN` (Eigen's `setEmpty` uses `highest()` and `lowest()`), so extending it by
//! any box gives that box. `extend` takes the coefficient-wise min and max (`std::min` /
//! `std::max`), and `diagonal` is max - min. Tier A: no arithmetic beyond one subtraction per
//! coefficient.

use super::{maxi, mini, Vector3d};

/// Axis-aligned 3D box. Replacement for `Eigen::AlignedBox3d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AlignedBox3d {
    /// The minimum corner, Eigen's `min()`.
    pub min: Vector3d,
    /// The maximum corner, Eigen's `max()`.
    pub max: Vector3d,
}

impl Default for AlignedBox3d {
    /// The empty box, like Eigen's default constructor.
    fn default() -> Self {
        Self::empty()
    }
}

impl AlignedBox3d {
    /// The box with the given corners.
    pub const fn new(min: Vector3d, max: Vector3d) -> Self {
        Self { min, max }
    }

    /// The empty box, Eigen's `setEmpty()`.
    pub const fn empty() -> Self {
        Self::new(
            Vector3d::new(f64::MAX, f64::MAX, f64::MAX),
            Vector3d::new(f64::MIN, f64::MIN, f64::MIN),
        )
    }

    /// Whether the point lies in the box, borders included: Eigen's `contains(p)`,
    /// min <= p and p <= max coefficient-wise (so a NaN coordinate is never contained).
    pub fn contains(self, p: Vector3d) -> bool {
        self.min.x <= p.x
            && self.min.y <= p.y
            && self.min.z <= p.z
            && p.x <= self.max.x
            && p.y <= self.max.y
            && p.z <= self.max.z
    }

    /// `max - min`, Eigen's `diagonal()`.
    pub fn diagonal(self) -> Vector3d {
        self.max - self.min
    }

    /// The smallest box containing this box and `other`, Eigen's `extend(box)`: the
    /// coefficient-wise `std::min` of the min corners and `std::max` of the max corners.
    pub fn extend(self, other: Self) -> Self {
        Self::new(
            Vector3d::new(
                mini(self.min.x, other.min.x),
                mini(self.min.y, other.min.y),
                mini(self.min.z, other.min.z),
            ),
            Vector3d::new(
                maxi(self.max.x, other.max.x),
                maxi(self.max.y, other.max.y),
                maxi(self.max.z, other.max.z),
            ),
        )
    }
}
