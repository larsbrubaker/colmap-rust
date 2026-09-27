//! Fixed-size vectors: [`Vector2d`], [`Vector3d`], [`Vector4d`] (doubles), [`Vector3f`]
//! (floats) and [`Vector3ub`] (bytes, an RGB color), the replacements for the Eigen types of
//! the same names. Port of colmap-sharp's `LinearAlgebra/Vector{2d,3d,4d,3f,3ub}.cs`, written
//! to Eigen's documented semantics; Eigen is not ported (docs/LICENSE_AUDIT.md).
//!
//! Arithmetic order (the module contract is in `linalg/mod.rs`): 2- and 3-vector reductions
//! are left to right. [`Vector4d`]'s two reductions are NOT: Eigen vectorizes a 4-coefficient
//! reduction as two 2-lane packets, so it sums lanes {0, 2} and {1, 3} first,
//! `(x*x + z*z) + (y*y + w*w)`. `oracle/linear_algebra_rotations.py` pins this through the
//! quaternion norm and inverse (bit-identical on all fixture cases with this pairing; 14 and
//! 35 of the first 130 miss with the left-to-right sum). `dot` uses the same pairing on the
//! assumption that it is the same reduction; pycolmap exposes no Eigen dot to check it.
//!
//! The macOS arm64 pycolmap wheel contracts some `a*b - c*d` into an FMA (`cross`, for one);
//! we do not (docs/CPP_DIVERGENCES.md, entry 2).

use super::{mini, DUMMY_PRECISION};
use crate::math::fns;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Sub, SubAssign,
};

/// Element-wise operators, indexing and array conversion shared by the double vectors.
macro_rules! vector_common {
    ($t:ident, $n:literal, $($f:ident => $i:literal),+) => {
        impl $t {
            /// The zero vector, Eigen's `Zero()`.
            pub const fn zeros() -> Self {
                Self { $($f: 0.0),+ }
            }

            /// Eigen's `Ones()`.
            pub const fn ones() -> Self {
                Self { $($f: 1.0),+ }
            }

            /// The coefficients as an array.
            pub const fn to_array(self) -> [f64; $n] {
                [$(self.$f),+]
            }

            /// The vector with these coefficients.
            pub const fn from_array(a: [f64; $n]) -> Self {
                Self { $($f: a[$i]),+ }
            }

            /// Euclidean norm, `sqrt(squared_norm())`.
            pub fn norm(self) -> f64 {
                fns::sqrt(self.squared_norm())
            }

            /// This vector divided by its norm. Like Eigen, a zero vector is returned
            /// unchanged rather than turned into NaN.
            pub fn normalized(self) -> Self {
                let squared_norm = self.squared_norm();
                if squared_norm > 0.0 {
                    self / fns::sqrt(squared_norm)
                } else {
                    self
                }
            }

            /// Coefficient-wise product, Eigen's `cwiseProduct`.
            pub fn cwise_product(self, other: Self) -> Self {
                Self { $($f: self.$f * other.$f),+ }
            }

            /// Coefficient-wise absolute value, Eigen's `cwiseAbs`.
            pub fn cwise_abs(self) -> Self {
                Self { $($f: self.$f.abs()),+ }
            }

            /// Eigen's `isApprox` at [`DUMMY_PRECISION`].
            pub fn is_approx(self, other: Self) -> bool {
                self.is_approx_with(other, DUMMY_PRECISION)
            }

            /// Eigen's `isApprox`: `||a - b|| <= precision * min(||a||, ||b||)`. Relative, so
            /// nothing but an exact zero is approximately equal to zero.
            pub fn is_approx_with(self, other: Self, precision: f64) -> bool {
                (self - other).norm() <= precision * mini(self.norm(), other.norm())
            }
        }

        impl Add for $t {
            type Output = Self;
            fn add(self, b: Self) -> Self {
                Self { $($f: self.$f + b.$f),+ }
            }
        }

        impl Sub for $t {
            type Output = Self;
            fn sub(self, b: Self) -> Self {
                Self { $($f: self.$f - b.$f),+ }
            }
        }

        impl Neg for $t {
            type Output = Self;
            fn neg(self) -> Self {
                Self { $($f: -self.$f),+ }
            }
        }

        impl Mul<f64> for $t {
            type Output = Self;
            fn mul(self, s: f64) -> Self {
                Self { $($f: self.$f * s),+ }
            }
        }

        impl Mul<$t> for f64 {
            type Output = $t;
            fn mul(self, a: $t) -> $t {
                $t { $($f: self * a.$f),+ }
            }
        }

        /// Scalar division divides each coefficient, as Eigen does (no reciprocal).
        impl Div<f64> for $t {
            type Output = Self;
            fn div(self, s: f64) -> Self {
                Self { $($f: self.$f / s),+ }
            }
        }

        impl AddAssign for $t {
            fn add_assign(&mut self, b: Self) {
                *self = *self + b;
            }
        }

        impl SubAssign for $t {
            fn sub_assign(&mut self, b: Self) {
                *self = *self - b;
            }
        }

        impl MulAssign<f64> for $t {
            fn mul_assign(&mut self, s: f64) {
                *self = *self * s;
            }
        }

        impl DivAssign<f64> for $t {
            fn div_assign(&mut self, s: f64) {
                *self = *self / s;
            }
        }

        /// Coefficient `i`; panics when `i` is out of range.
        impl Index<usize> for $t {
            type Output = f64;
            fn index(&self, i: usize) -> &f64 {
                match i {
                    $($i => &self.$f,)+
                    _ => panic!(concat!(stringify!($t), " index {} out of range"), i),
                }
            }
        }

        impl IndexMut<usize> for $t {
            fn index_mut(&mut self, i: usize) -> &mut f64 {
                match i {
                    $($i => &mut self.$f,)+
                    _ => panic!(concat!(stringify!($t), " index {} out of range"), i),
                }
            }
        }
    };
}

/// Fixed-size 2-vector of doubles. Replacement for `Eigen::Vector2d`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector2d {
    /// Eigen's `x()` / `[0]`.
    pub x: f64,
    /// Eigen's `y()` / `[1]`.
    pub y: f64,
}

vector_common!(Vector2d, 2, x => 0, y => 1);

impl Vector2d {
    /// The vector `(x, y)`.
    pub const fn new(x: f64, y: f64) -> Self {
        Self { x, y }
    }

    /// Squared Euclidean norm, `x*x + y*y`.
    pub fn squared_norm(self) -> f64 {
        self.x * self.x + self.y * self.y
    }

    /// Dot product, `x*o.x + y*o.y`.
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y
    }

    /// `(x, y, 1)`, Eigen's `homogeneous()`.
    pub const fn homogeneous(self) -> Vector3d {
        Vector3d::new(self.x, self.y, 1.0)
    }
}

/// Fixed-size 3-vector of doubles. Replacement for `Eigen::Vector3d`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3d {
    /// Eigen's `x()` / `[0]`.
    pub x: f64,
    /// Eigen's `y()` / `[1]`.
    pub y: f64,
    /// Eigen's `z()` / `[2]`.
    pub z: f64,
}

vector_common!(Vector3d, 3, x => 0, y => 1, z => 2);

impl Vector3d {
    /// The vector `(x, y, z)`.
    pub const fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    /// `(1, 0, 0)`, Eigen's `UnitX()`.
    pub const fn unit_x() -> Self {
        Self::new(1.0, 0.0, 0.0)
    }

    /// `(0, 1, 0)`, Eigen's `UnitY()`.
    pub const fn unit_y() -> Self {
        Self::new(0.0, 1.0, 0.0)
    }

    /// `(0, 0, 1)`, Eigen's `UnitZ()`.
    pub const fn unit_z() -> Self {
        Self::new(0.0, 0.0, 1.0)
    }

    /// Squared Euclidean norm, `x*x + y*y + z*z` left to right (the oracle confirms it
    /// through `AngleAxisd`'s angle).
    pub fn squared_norm(self) -> f64 {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Dot product, `x*o.x + y*o.y + z*o.z`.
    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    /// Cross product `self x o`, each coefficient `a*b - c*d` without FMA.
    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    /// `(x, y, z, 1)`, Eigen's `homogeneous()`.
    pub const fn homogeneous(self) -> Vector4d {
        Vector4d::new(self.x, self.y, self.z, 1.0)
    }

    /// `(x / z, y / z)`, Eigen's `hnormalized()`.
    pub fn hnormalized(self) -> Vector2d {
        Vector2d::new(self.x / self.z, self.y / self.z)
    }

    /// `(x, y)`, Eigen's `head<2>()`.
    pub const fn head2(self) -> Vector2d {
        Vector2d::new(self.x, self.y)
    }
}

/// Fixed-size 4-vector of doubles. Replacement for `Eigen::Vector4d`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector4d {
    /// Eigen's `x()` / `[0]`.
    pub x: f64,
    /// Eigen's `y()` / `[1]`.
    pub y: f64,
    /// Eigen's `z()` / `[2]`.
    pub z: f64,
    /// Eigen's `w()` / `[3]`.
    pub w: f64,
}

vector_common!(Vector4d, 4, x => 0, y => 1, z => 2, w => 3);

impl Vector4d {
    /// The vector `(x, y, z, w)`.
    pub const fn new(x: f64, y: f64, z: f64, w: f64) -> Self {
        Self { x, y, z, w }
    }

    /// Squared Euclidean norm, paired as Eigen's packet reduction:
    /// `(x*x + z*z) + (y*y + w*w)` (see the file header).
    pub fn squared_norm(self) -> f64 {
        (self.x * self.x + self.z * self.z) + (self.y * self.y + self.w * self.w)
    }

    /// Dot product, paired as Eigen's packet reduction (see the file header).
    pub fn dot(self, o: Self) -> f64 {
        (self.x * o.x + self.z * o.z) + (self.y * o.y + self.w * o.w)
    }

    /// `(x / w, y / w, z / w)`, Eigen's `hnormalized()`.
    pub fn hnormalized(self) -> Vector3d {
        Vector3d::new(self.x / self.w, self.y / self.w, self.z / self.w)
    }

    /// `(x, y, z)`, Eigen's `head<3>()`.
    pub const fn head3(self) -> Vector3d {
        Vector3d::new(self.x, self.y, self.z)
    }
}

/// Fixed-size 3-vector of floats. Replacement for `Eigen::Vector3f` where COLMAP keeps
/// single-precision positions (Delaunay meshing's input points). Storage and equality only:
/// the float arithmetic that uses it is written out at the call site with its order stated.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vector3f {
    /// Eigen's `x()`.
    pub x: f32,
    /// Eigen's `y()`.
    pub y: f32,
    /// Eigen's `z()`.
    pub z: f32,
}

impl Vector3f {
    /// The vector `(x, y, z)`.
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// Fixed-size 3-vector of bytes. Replacement for `Eigen::Vector3ub` (colmap/util/types.h),
/// which COLMAP uses for RGB colors. `Default` is Eigen's `Zero()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Vector3ub {
    /// First coefficient (red, for a color).
    pub x: u8,
    /// Second coefficient (green).
    pub y: u8,
    /// Third coefficient (blue).
    pub z: u8,
}

impl Vector3ub {
    /// The vector `(x, y, z)`.
    pub const fn new(x: u8, y: u8, z: u8) -> Self {
        Self { x, y, z }
    }

    /// Eigen's `Vector3ub::Zero()`.
    pub const fn zeros() -> Self {
        Self::new(0, 0, 0)
    }
}
