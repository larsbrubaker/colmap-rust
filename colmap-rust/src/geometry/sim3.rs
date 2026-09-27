//! Port of COLMAP's `colmap/geometry/sim3.h` and `sim3.cc`: [`Sim3d`], the 7-DoF similarity
//! transform `x_in_b = s * R * x_in_a + t`, with its inverse, composition, matrix
//! conversions, text file round trip and `operator<<`. Port of colmap-sharp's
//! `Geometry/Sim3d.cs`. Neighbors: [`super::rigid3`]. Tests: `tests/geometry/sim3.rs`
//! (sim3_test.cc) and `tests/geometry/rust_only_transforms_oracle.rs`.
//!
//! Tiers: scale and rotation of composition and `inverse`, `to_matrix`, `from_matrix`,
//! `Display` and `to_file`'s text are Tier A (bit-identical to the pycolmap oracle); applying
//! the transform and the translations of composition and `inverse` rotate a vector with
//! `q * v` and are Tier B (docs/CPP_DIVERGENCES.md entries 2 and 80).
//!
//! Translation notes: the parameters `[qx, qy, qz, qw, tx, ty, tz, s]` are public fields;
//! `Default` is hand-written as the identity (COLMAP's default constructor); COLMAP's free
//! `Inverse(Sim3d)` is [`Sim3d::inverse`]. Estimating a `Sim3d` from point correspondences
//! lives in COLMAP's `estimators/similarity_transform` and needs an SVD, so it is not here.

use std::fmt;
use std::ops::Mul;
use std::path::Path;

use super::rigid3::format_list;
use crate::linalg::{Matrix3x4d, Quaterniond, Vector3d};
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::stream_format::{format_double, DEFAULT_PRECISION};
use crate::util::string::string_to_double;
use crate::Result;

/// 3D similarity transform with 7 degrees of freedom, `x_in_b = s * R * x_in_a + t`. Port of
/// `colmap::Sim3d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sim3d {
    /// The scale `s`.
    pub scale: f64,
    /// The rotation `R` (a unit quaternion).
    pub rotation: Quaterniond,
    /// The translation `t`.
    pub translation: Vector3d,
}

impl Default for Sim3d {
    /// COLMAP's default constructor: the identity transform.
    fn default() -> Self {
        Self::identity()
    }
}

impl Sim3d {
    /// `Sim3d(scale, rotation, translation)`.
    pub const fn new(scale: f64, rotation: Quaterniond, translation: Vector3d) -> Self {
        Self {
            scale,
            rotation,
            translation,
        }
    }

    /// The identity transform, `params = (0, 0, 0, 1, 0, 0, 0, 1)`.
    pub const fn identity() -> Self {
        Self::new(1.0, Quaterniond::identity(), Vector3d::new(0.0, 0.0, 0.0))
    }

    /// `ToMatrix()`: `[s * R | t]`.
    pub fn to_matrix(&self) -> Matrix3x4d {
        Matrix3x4d::from_blocks(
            self.scale * self.rotation.to_rotation_matrix(),
            self.translation,
        )
    }

    /// `FromMatrix(matrix)`: the scale is the norm of the first column, and the rotation is
    /// `Quaterniond(matrix.leftCols<3>() / scale).normalized()`.
    pub fn from_matrix(matrix: &Matrix3x4d) -> Self {
        let scale = matrix.col(0).norm();
        Self::new(
            scale,
            Quaterniond::from_rotation_matrix(matrix.left_cols3() / scale).normalized(),
            matrix.col(3),
        )
    }

    /// Port of `colmap::Inverse(const Sim3d&)`: `a_from_b` from `b_from_a`.
    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        Self::new(
            1.0 / self.scale,
            rotation,
            (rotation * self.translation) / -self.scale,
        )
    }

    /// `ToFile(path)`: writes `s qw qx qy qz tx ty tz` at precision 17 (lossless) with a
    /// trailing newline, truncating the file.
    pub fn to_file(&self, path: &Path) -> Result<()> {
        let q = self.rotation;
        let t = self.translation;
        let text = [self.scale, q.w, q.x, q.y, q.z, t.x, t.y, t.z]
            .iter()
            .map(|&v| format_double(v, 17))
            .collect::<Vec<_>>()
            .join(" ");
        std::fs::write(path, text + "\n").map_err(|_| file_error(path))
    }

    /// `FromFile(path)`: reads the eight whitespace-separated values `ToFile` writes. Fails
    /// like COLMAP's `THROW_CHECK` when the file cannot be opened or a value is missing or
    /// not a number (docs/CPP_DIVERGENCES.md entry 60 for the number syntax).
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path).map_err(|_| file_error(path))?;
        let mut values = [0.0; 8];
        let mut tokens = text.split_ascii_whitespace();
        for value in &mut values {
            let token = tokens.next().ok_or_else(|| parse_error(path))?;
            *value = string_to_double(token).map_err(|_| parse_error(path))?;
        }
        let [s, qw, qx, qy, qz, tx, ty, tz] = values;
        Ok(Self::new(
            s,
            Quaterniond::new(qw, qx, qy, qz),
            Vector3d::new(tx, ty, tz),
        ))
    }
}

fn file_error(path: &Path) -> ColmapError {
    ColmapError::new(
        ErrorKind::InvalidArgument,
        format!("Check failed: file.good() {}", path.display()),
    )
}

fn parse_error(path: &Path) -> ColmapError {
    ColmapError::new(
        ErrorKind::InvalidArgument,
        format!(
            "Check failed: file >> t.scale() >> ... >> t.translation().z() {}",
            path.display()
        ),
    )
}

/// `x_in_b = b_from_a * x_in_a`: `s * (R * x) + t`.
impl Mul<Vector3d> for Sim3d {
    type Output = Vector3d;
    fn mul(self, x: Vector3d) -> Vector3d {
        self.scale * (self.rotation * x) + self.translation
    }
}

/// `c_from_a = c_from_b * b_from_a`; the composed rotation is re-normalized as in COLMAP.
impl Mul for Sim3d {
    type Output = Sim3d;
    fn mul(self, b_from_a: Sim3d) -> Sim3d {
        Sim3d::new(
            self.scale * b_from_a.scale,
            (self.rotation * b_from_a.rotation).normalized(),
            self.translation + self.scale * (self.rotation * b_from_a.translation),
        )
    }
}

/// COLMAP's `operator<<`:
/// `Sim3d(scale=s, rotation_xyzw=[x, y, z, w], translation=[x, y, z])`.
impl fmt::Display for Sim3d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let q = self.rotation;
        let t = self.translation;
        write!(
            f,
            "Sim3d(scale={}, rotation_xyzw=[{}], translation=[{}])",
            format_double(self.scale, DEFAULT_PRECISION),
            format_list(&[q.x, q.y, q.z, q.w]),
            format_list(&[t.x, t.y, t.z])
        )
    }
}
