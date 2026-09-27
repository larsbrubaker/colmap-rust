//! [`Quaterniond`]: rotation quaternion of doubles, the replacement for `Eigen::Quaterniond`,
//! the rotation half of COLMAP's `Rigid3d` and `Sim3d`. Port of colmap-sharp's
//! `LinearAlgebra/Quaterniond.cs`, written from the published algorithms cited below to
//! Eigen's documented semantics; Eigen is not ported (docs/LICENSE_AUDIT.md). Siblings:
//! [`AngleAxisd`] (the other rotation representation) and [`Matrix3d`].
//!
//! Conventions, as Eigen documents them:
//! - `new(w, x, y, z)`; the coefficients are stored (`#[repr(C)]` field order) and returned by
//!   `coeffs()` as `(x, y, z, w)`, Eigen's memory order, which is how COLMAP's `Rigid3d` packs
//!   its params. COLMAP's file formats write `qw qx qy qz`; that belongs to the I/O code.
//! - Hamilton product, rotating a vector v as q v q*. The product composes rotations:
//!   `(a * b) * v == a * (b * v)`.
//! - Norms and dot products run over `coeffs()` with [`Vector4d`]'s paired reduction.
//!
//! Algorithms:
//! - Rotating a vector uses the 15-multiply form of q v q* for a unit q: t = 2 (u x v),
//!   v' = v + w t + u x t, with u = (x, y, z) (F. Giesen, "Rotating a vector by a unit
//!   quaternion", 2015). Like Eigen it assumes q is normalized.
//! - `to_rotation_matrix` is the standard unit-quaternion rotation matrix (K. Shoemake,
//!   "Animating rotation with quaternion curves", SIGGRAPH 1985).
//! - `from_rotation_matrix` is Shoemake's branch algorithm ("Quaternion Calculus and Fast
//!   Animation", SIGGRAPH 1987 course notes): if the trace is positive, w comes from the trace
//!   and is positive; otherwise the largest diagonal element picks the component computed
//!   first, and that component is positive (Eigen's documented sign convention).
//!
//! Tiers, checked bit for bit against Eigen through pycolmap
//! (`oracle/linear_algebra_rotations.py`, `tests/linalg/rust_only_rotation_oracle.rs`):
//! - Tier A, bit-identical: the product (with Eigen's SIMD pairing), `norm` and `inverse`
//!   (Vector4d's paired reduction), `to_rotation_matrix`, `from_rotation_matrix`
//!   (`Matrix3d::trace`'s grouping).
//! - Tier A up to docs/CPP_DIVERGENCES.md entry 1: `angular_distance` and
//!   `AngleAxisd::from_quaternion`'s angle are bit-identical with Apple libm's atan2 and within
//!   1 ulp with `fns::atan2` (the `libm` crate).
//! - Tier B: `q * v`, which the macOS wheel computes with FMA-contracted cross products
//!   (docs/CPP_DIVERGENCES.md, entry 2), and `from_angle_axis`, whose `sin` can differ from the
//!   wheel's by 1 ulp (entry 3). `from_two_vectors` (entry 4) and `slerp` are Tier B.

use super::{AngleAxisd, Matrix3d, Vector3d, Vector4d, DUMMY_PRECISION};
use crate::math::fns;
use std::ops::Mul;

/// Rotation quaternion of doubles. Replacement for `Eigen::Quaterniond`. `Default` is the
/// zero quaternion (all coefficients 0), not the identity.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Quaterniond {
    /// The x (i) coefficient.
    pub x: f64,
    /// The y (j) coefficient.
    pub y: f64,
    /// The z (k) coefficient.
    pub z: f64,
    /// The scalar (real) coefficient.
    pub w: f64,
}

// Below this 1 + c, from_two_vectors composes a half turn with a short arc (see its doc).
const NEARLY_OPPOSITE_THRESHOLD: f64 = 1e-8;

impl Quaterniond {
    /// The quaternion w + xi + yj + zk. Argument order `(w, x, y, z)` like Eigen's constructor.
    pub const fn new(w: f64, x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z, w }
    }

    /// The identity rotation `(1, 0, 0, 0)`.
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0)
    }

    /// Coefficients in Eigen's memory order `(x, y, z, w)`, Eigen's `coeffs()`.
    pub const fn coeffs(self) -> Vector4d {
        Vector4d::new(self.x, self.y, self.z, self.w)
    }

    /// The vector part `(x, y, z)`, Eigen's `vec()`.
    pub const fn vec(self) -> Vector3d {
        Vector3d::new(self.x, self.y, self.z)
    }

    /// The quaternion with coefficients in Eigen's memory order `(x, y, z, w)`.
    pub const fn from_coeffs(xyzw: Vector4d) -> Self {
        Self::new(xyzw.w, xyzw.x, xyzw.y, xyzw.z)
    }

    /// Squared norm over the coefficients.
    pub fn squared_norm(self) -> f64 {
        self.coeffs().squared_norm()
    }

    /// Norm, `sqrt(squared_norm())`.
    pub fn norm(self) -> f64 {
        self.coeffs().norm()
    }

    /// Dot product of the coefficients, in memory order.
    pub fn dot(self, other: Self) -> f64 {
        self.coeffs().dot(other.coeffs())
    }

    /// The unit quaternion; a zero quaternion is returned unchanged, like Eigen.
    pub fn normalized(self) -> Self {
        Self::from_coeffs(self.coeffs().normalized())
    }

    /// The conjugate `(w, -x, -y, -z)`; the inverse rotation for a unit quaternion.
    pub fn conjugate(self) -> Self {
        Self::new(self.w, -self.x, -self.y, -self.z)
    }

    /// The multiplicative inverse, conjugate / squared norm. A zero quaternion has none and
    /// yields the zero quaternion. For unit quaternions `conjugate` is cheaper and equal.
    pub fn inverse(self) -> Self {
        let squared_norm = self.squared_norm();
        if squared_norm > 0.0 {
            Self::new(
                self.w / squared_norm,
                -self.x / squared_norm,
                -self.y / squared_norm,
                -self.z / squared_norm,
            )
        } else {
            Self::default()
        }
    }

    /// The angle in radians of the rotation taking `other` to `self`,
    /// `2 atan2(|d.vec|, |d.w|)` with `d = self * other.conjugate()`; in [0, pi]. Eigen's
    /// `angularDistance`.
    pub fn angular_distance(self, other: Self) -> f64 {
        let d = self * other.conjugate();
        2.0 * fns::atan2(d.vec().norm(), d.w.abs())
    }

    /// The 3x3 rotation matrix of this (unit) quaternion, Shoemake's (1985)
    /// R = I + 2 w [v]x + 2 [v]x^2 written out per element: diagonal 1 - 2(b^2 + c^2),
    /// off-diagonals 2(ab -/+ wc).
    pub fn to_rotation_matrix(self) -> Matrix3d {
        // Grouping (the oracle pins it bit for bit): each off-diagonal is a sum or difference
        // of two doubled products, and each diagonal subtracts a sum of two doubled squares
        // from 1. The 2 multiplies a coordinate before the product is formed, so each doubled
        // product rounds once: 2 * (x * y) rounds x * y first, which differs in the subnormal
        // range (fixture case q = (3e-162, 4e-162, 0, 0.5): 2.5e-323 vs 2e-323). Which factor
        // carries the 2 does not matter; (2a) * b is round(2ab) either way.
        let (x2, y2, z2) = (2.0 * self.x, 2.0 * self.y, 2.0 * self.z);
        let (xx, yy, zz) = (x2 * self.x, y2 * self.y, z2 * self.z);
        let (xy, xz, yz) = (x2 * self.y, x2 * self.z, y2 * self.z);
        let (wx, wy, wz) = (x2 * self.w, y2 * self.w, z2 * self.w);

        Matrix3d::new(
            1.0 - (yy + zz),
            xy - wz,
            xz + wy,
            xy + wz,
            1.0 - (xx + zz),
            yz - wx,
            xz - wy,
            yz + wx,
            1.0 - (xx + yy),
        )
    }

    /// The quaternion of a rotation matrix by Shoemake's branch algorithm (see the file header
    /// for the sign convention). Eigen's `Quaternion(const Matrix3&)`. The input is not
    /// orthogonalized; COLMAP normalizes the result where it needs a unit quaternion.
    pub fn from_rotation_matrix(m: Matrix3d) -> Self {
        // Shoemake: 4 w^2 = 1 + trace. The test is strictly positive because the oracle
        // requires it: on the exactly-zero-trace matrices in the fixture (trace_zero_cases)
        // Eigen takes the diagonal branch, and >= 0 gives different bits.
        let trace = m.trace();
        if trace > 0.0 {
            let root = fns::sqrt(trace + 1.0); // 2|w|
            let scale = 0.5 / root; // 1 / (4w)
            return Self::new(
                root * 0.5,
                (m[(2, 1)] - m[(1, 2)]) * scale,
                (m[(0, 2)] - m[(2, 0)]) * scale,
                (m[(1, 0)] - m[(0, 1)]) * scale,
            );
        }

        // Otherwise solve first for the component whose diagonal entry is largest (ties go to
        // the earlier axis), which keeps the square root well away from zero. For that axis a
        // with cyclic successors b, c: 4 a^2 = R_aa - R_bb - R_cc + 1, summed in exactly that
        // order (the oracle distinguishes the orders).
        if m[(0, 0)] >= m[(1, 1)] && m[(0, 0)] >= m[(2, 2)] {
            let root = fns::sqrt(m[(0, 0)] - m[(1, 1)] - m[(2, 2)] + 1.0); // 2|x|
            let scale = 0.5 / root;
            return Self::new(
                (m[(2, 1)] - m[(1, 2)]) * scale,
                root * 0.5,
                (m[(1, 0)] + m[(0, 1)]) * scale,
                (m[(2, 0)] + m[(0, 2)]) * scale,
            );
        }

        if m[(1, 1)] >= m[(2, 2)] {
            let root = fns::sqrt(m[(1, 1)] - m[(2, 2)] - m[(0, 0)] + 1.0); // 2|y|
            let scale = 0.5 / root;
            return Self::new(
                (m[(0, 2)] - m[(2, 0)]) * scale,
                (m[(0, 1)] + m[(1, 0)]) * scale,
                root * 0.5,
                (m[(2, 1)] + m[(1, 2)]) * scale,
            );
        }

        let root = fns::sqrt(m[(2, 2)] - m[(0, 0)] - m[(1, 1)] + 1.0); // 2|z|
        let scale = 0.5 / root;
        Self::new(
            (m[(1, 0)] - m[(0, 1)]) * scale,
            (m[(0, 2)] + m[(2, 0)]) * scale,
            (m[(1, 2)] + m[(2, 1)]) * scale,
            root * 0.5,
        )
    }

    /// The quaternion of an angle-axis rotation: `(cos(a/2), sin(a/2) axis)`.
    pub fn from_angle_axis(angle_axis: AngleAxisd) -> Self {
        let half_angle = 0.5 * angle_axis.angle;
        let sine = fns::sin(half_angle);
        let axis = angle_axis.axis;
        Self::new(
            fns::cos(half_angle),
            sine * axis.x,
            sine * axis.y,
            sine * axis.z,
        )
    }

    /// The shortest-arc rotation that maps the direction of `a` onto the direction of `b`
    /// (the documented behavior of Eigen's `Quaternion::FromTwoVectors`, which COLMAP's
    /// `SynthesizeDataset` aims its synthetic frames with). Zero-length inputs are normalized
    /// like Eigen's `normalized()` (left as zero).
    /// - General case, from S. Melax, "The Shortest Arc Quaternion", Game Programming Gems 1
    ///   (2000): with unit u, v and c = u . v, s = sqrt(2 (1 + c)), q = (s / 2, (u x v) / s).
    ///   The vector part multiplies by the reciprocal 1 / s; with that rounding colmap-sharp's
    ///   synthesized frame rotations match pycolmap bit for bit.
    /// - Nearly opposite vectors, 1 + c < 1e-8: s would come from 1 + c, whose absolute
    ///   rounding error (~1e-16) becomes a relative error above ~5e-9 in s, and at c = -1 the
    ///   axis u x v vanishes. Instead the rotation is split into a half turn about an axis p
    ///   perpendicular to u (maps u to -u; quaternion (0, p)), then the short arc from -u to v
    ///   by the formula above, where (-u) . v = -c is close to 1 and well conditioned.
    ///   p = normalize(u x e), with e the coordinate axis least aligned with u (|u_k|
    ///   smallest), so |u x e| >= sqrt(2/3). Results can differ from Eigen's in this branch
    ///   (docs/CPP_DIVERGENCES.md, entry 4).
    pub fn from_two_vectors(a: Vector3d, b: Vector3d) -> Self {
        let v0 = a.normalized();
        let v1 = b.normalized();
        let c = v1.dot(v0);

        if 1.0 + c < NEARLY_OPPOSITE_THRESHOLD {
            let half_turn = Self::half_turn_perpendicular_to(v0);
            return Self::shortest_arc_unit(-v0, v1, -c) * half_turn;
        }

        Self::shortest_arc_unit(v0, v1, c)
    }

    // Melax's shortest-arc quaternion for unit u, v with c = u . v, valid away from c = -1.
    fn shortest_arc_unit(u: Vector3d, v: Vector3d, c: f64) -> Self {
        let axis = u.cross(v);
        let s = fns::sqrt((1.0 + c) * 2.0);
        let inv_s = 1.0 / s;
        Self::new(s * 0.5, axis.x * inv_s, axis.y * inv_s, axis.z * inv_s)
    }

    // A rotation by pi about a unit axis perpendicular to the unit vector u.
    fn half_turn_perpendicular_to(u: Vector3d) -> Self {
        let (ax, ay, az) = (u.x.abs(), u.y.abs(), u.z.abs());
        let least_aligned = if ax <= ay && ax <= az {
            Vector3d::unit_x()
        } else if ay <= az {
            Vector3d::unit_y()
        } else {
            Vector3d::unit_z()
        };
        let p = u.cross(least_aligned).normalized();
        Self::new(0.0, p.x, p.y, p.z)
    }

    /// Spherical linear interpolation from `self` (t = 0) to `other` (t = 1) along the shorter
    /// arc (the documented behavior of Eigen's `Quaternion::slerp`, which COLMAP's
    /// `InterpolateCameraPoses` calls). From the textbook form of Shoemake's slerp (SIGGRAPH
    /// 1985): slerp(a, b; t) = [sin((1 - t) w) a + sin(t w) b] / sin(w), w the angle between a
    /// and b on the unit 3-sphere. Tier B.
    /// - Shortest arc: q and -q are the same rotation, so when a . b < 0 the target is
    ///   replaced by -b before interpolating.
    /// - The angle is w = 2 atan2(|a - b|, |a + b|) rather than acos(a . b): for unit a, b,
    ///   |a - b| = 2 sin(w/2) and |a + b| = 2 cos(w/2), and this form keeps full relative
    ///   accuracy for tiny w (W. Kahan, "How Futile are Mindless Assessments of Roundoff in
    ///   Floating-Point Computation?", 2006).
    /// - Small angles: sin(k w) / sin(w) = k (1 - (k^2 - 1) w^2 / 6 + O(w^4)), so below
    ///   w = 1e-8 the weights equal 1 - t and t to under half an ulp; plain linear
    ///   interpolation is then the correctly rounded slerp, and it avoids 0 / 0 at w = 0.
    pub fn slerp(self, t: f64, other: Self) -> Self {
        const LINEAR_BELOW_ANGLE: f64 = 1e-8;

        let from = self.coeffs();
        let to = if self.dot(other) < 0.0 {
            -other.coeffs()
        } else {
            other.coeffs()
        };

        let angle = 2.0 * fns::atan2((from - to).norm(), (from + to).norm());
        if angle < LINEAR_BELOW_ANGLE {
            return Self::from_coeffs((1.0 - t) * from + t * to);
        }

        let sin_angle = fns::sin(angle);
        let weight_from = fns::sin((1.0 - t) * angle) / sin_angle;
        let weight_to = fns::sin(t * angle) / sin_angle;
        Self::from_coeffs(weight_from * from + weight_to * to)
    }

    /// Eigen's `isApprox` on the coefficients at [`DUMMY_PRECISION`]. q and -q are the same
    /// rotation but are not approximately equal here, as in Eigen.
    pub fn is_approx(self, other: Self) -> bool {
        self.is_approx_with(other, DUMMY_PRECISION)
    }

    /// Eigen's `isApprox` on the coefficients: `||a - b|| <= precision * min(||a||, ||b||)`.
    pub fn is_approx_with(self, other: Self, precision: f64) -> bool {
        self.coeffs().is_approx_with(other.coeffs(), precision)
    }
}

/// Hamilton product: the rotation `b` followed by `a`. Each coefficient is the textbook
/// four-term sum, grouped in two pairs the way Eigen's packet (SIMD) quaternion product groups
/// them. Found with `oracle/linear_algebra_rotations.py`: of all 24 left-to-right orders, 3
/// pairings and their FMA variants, only this one is bit-identical on all fixture cases, for
/// every coefficient.
impl Mul for Quaterniond {
    type Output = Self;
    fn mul(self, b: Self) -> Self {
        let a = self;
        Self::new(
            (a.w * b.w - a.y * b.y) + (-(a.x * b.x) - a.z * b.z),
            (a.w * b.x + a.y * b.z) + (a.x * b.w - a.z * b.y),
            (a.w * b.y + a.y * b.w) + (a.z * b.x - a.x * b.z),
            (a.w * b.z - a.y * b.x) + (a.z * b.w + a.x * b.y),
        )
    }
}

/// Rotates `v` by the unit quaternion (q v q*), as t = 2 (u x v), v + w t + u x t. Like Eigen,
/// q is assumed normalized.
impl Mul<Vector3d> for Quaterniond {
    type Output = Vector3d;
    fn mul(self, v: Vector3d) -> Vector3d {
        let u = self.vec();
        let mut uv = u.cross(v);
        uv += uv;
        v + self.w * uv + u.cross(uv)
    }
}
