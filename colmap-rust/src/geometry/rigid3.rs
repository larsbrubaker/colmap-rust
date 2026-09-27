//! Port of COLMAP's `colmap/geometry/rigid3.h` and `rigid3.cc`: [`Rigid3d`], the 6-DoF rigid
//! transform `x_in_b = R * x_in_a + t`, with its inverse, composition, matrix conversions,
//! adjoints and the covariance helpers (inverse, composed, relative), plus
//! [`cross_product_matrix`].
//! Port of colmap-sharp's `Geometry/Rigid3d.cs`. Neighbors: [`super::sim3`] (the 7-DoF
//! similarity) and [`super::pose`] (functions on poses). Tests: `tests/geometry/rigid3.rs`
//! (rigid3_test.cc) and `tests/geometry/rust_only_transforms_oracle.rs`.
//!
//! Tiers (pinned by the pycolmap oracle):
//! - Tier A: composition's and `inverse`'s rotation, `to_matrix`, `from_matrix`, `adjoint`,
//!   `==` and `Display`.
//! - Tier B: everything that rotates a vector with `q * v` (applying the transform, the
//!   translations of `inverse` and composition, `tgt_origin_in_src`), `adjoint_inverse` and
//!   [`get_covariance_for_rigid3d_inverse`], because the macOS wheel contracts their products
//!   into FMAs and this port never does (docs/CPP_DIVERGENCES.md entries 2 and 80).
//!
//! Translation notes:
//! - COLMAP stores `params = [qx, qy, qz, qw, tx, ty, tz]` and hands out mutable `Eigen::Map`
//!   views; here the rotation and translation are plain public fields.
//! - COLMAP's default constructor is the identity, so `Default` is hand-written as
//!   [`Rigid3d::identity`] (never a derived all-zero rotation).
//! - COLMAP's free `Inverse(Rigid3d)` is [`Rigid3d::inverse`].
//! - `GetCovarianceForComposedRigid3d` and `GetCovarianceForRelativeRigid3d` take the 12x12
//!   joint covariance as a [`MatrixXd`] (checked 12x12) and build their 6x12 Jacobians as
//!   `MatrixXd`, so they are Tier B through its left-to-right products
//!   (docs/CPP_DIVERGENCES.md entries 20 and 80), pinned by the oracle's `cov_composed` and
//!   `cov_relative` fields.

use std::fmt;
use std::ops::Mul;

use crate::linalg::{Matrix3d, Matrix3x4d, Matrix6d, MatrixXd, Quaterniond, Vector3d};
use crate::util::stream_format::{format_double, DEFAULT_PRECISION};

/// Port of `colmap::CrossProductMatrix`: the skew-symmetric matrix `[v]_x` with
/// `[v]_x * u == v.cross(u)`.
pub fn cross_product_matrix(vector: Vector3d) -> Matrix3d {
    Matrix3d::new(
        0.0, -vector.z, vector.y, //
        vector.z, 0.0, -vector.x, //
        -vector.y, vector.x, 0.0,
    )
}

/// 3D rigid transform with 6 degrees of freedom, `x_in_b = R * x_in_a + t`. Port of
/// `colmap::Rigid3d`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rigid3d {
    /// The rotation `R` (a unit quaternion).
    pub rotation: Quaterniond,
    /// The translation `t`.
    pub translation: Vector3d,
}

impl Default for Rigid3d {
    /// COLMAP's default constructor: the identity transform.
    fn default() -> Self {
        Self::identity()
    }
}

impl Rigid3d {
    /// `Rigid3d(rotation, translation)`.
    pub const fn new(rotation: Quaterniond, translation: Vector3d) -> Self {
        Self {
            rotation,
            translation,
        }
    }

    /// The identity transform, `params = (0, 0, 0, 1, 0, 0, 0)`.
    pub const fn identity() -> Self {
        Self::new(Quaterniond::identity(), Vector3d::new(0.0, 0.0, 0.0))
    }

    /// `ToMatrix()`: `[R | t]`.
    pub fn to_matrix(&self) -> Matrix3x4d {
        Matrix3x4d::from_blocks(self.rotation.to_rotation_matrix(), self.translation)
    }

    /// `FromMatrix(matrix)`: the rotation is `Quaterniond(matrix.leftCols<3>()).normalized()`.
    pub fn from_matrix(matrix: &Matrix3x4d) -> Self {
        Self::new(
            Quaterniond::from_rotation_matrix(matrix.left_cols3()).normalized(),
            matrix.col(3),
        )
    }

    /// `Adjoint()`: `[R, 0; [t]_x R, R]`, used to propagate uncertainty.
    /// Reference: <https://gtsam.org/2021/02/23/uncertainties-part3.html>.
    pub fn adjoint(&self) -> Matrix6d {
        let rotation = self.rotation.to_rotation_matrix();
        // Eigen's R.colwise().cross(-t): each column c becomes c x (-t), which is t x c.
        let minus_t = -self.translation;
        let t_cross_r = Matrix3d::from_columns(
            rotation.col(0).cross(minus_t),
            rotation.col(1).cross(minus_t),
            rotation.col(2).cross(minus_t),
        );
        Matrix6d::from_blocks(rotation, Matrix3d::zeros(), t_cross_r, rotation)
    }

    /// `AdjointInverse()`: `[R^T, 0; -R^T [t]_x, R^T]`.
    pub fn adjoint_inverse(&self) -> Matrix6d {
        let rotation_t = self.rotation.to_rotation_matrix().transpose();
        Matrix6d::from_blocks(
            rotation_t,
            Matrix3d::zeros(),
            -rotation_t * cross_product_matrix(self.translation),
            rotation_t,
        )
    }

    /// `TgtOriginInSrc()`: the origin of the target frame in the source frame, `R^-1 * -t`.
    pub fn tgt_origin_in_src(&self) -> Vector3d {
        self.rotation.inverse() * -self.translation
    }

    /// Port of `colmap::Inverse(const Rigid3d&)`: `a_from_b` from `b_from_a`.
    pub fn inverse(&self) -> Self {
        let rotation = self.rotation.inverse();
        Self::new(rotation, rotation * -self.translation)
    }
}

/// Port of `colmap::GetCovarianceForRigid3dInverse`: the 6x6 covariance of
/// `rigid3.inverse()` from the covariance of `rigid3`, `Ad^-1 * covar * Ad^-T`.
///
/// COLMAP follows the left convention (unlike the reference paper and GTSAM), where the
/// Jacobian of the inverse is `-Ad(X^-1)`; see Solà, Deray, Atchuthan, "A micro Lie theory for
/// state estimation in robotics", 2018, Eqs. (62) and (57).
pub fn get_covariance_for_rigid3d_inverse(rigid3: &Rigid3d, covar: &Matrix6d) -> Matrix6d {
    let adjoint_inv = rigid3.adjoint_inverse();
    adjoint_inv * *covar * adjoint_inv.transpose()
}

/// Port of `colmap::GetCovarianceForComposedRigid3d`: the 6x6 covariance of the composed
/// `a_from_c = a_from_b * b_from_c` from the 12x12 joint covariance of
/// `(a_from_b, b_from_c)`, `J * covar * J^T` with `J = [I, Ad(a_from_b)]`. `b_from_c` does
/// not contribute and is not needed.
///
/// # Panics
/// When `covar` is not 12x12 (a compile-time shape in COLMAP).
pub fn get_covariance_for_composed_rigid3d(a_from_b: &Rigid3d, covar: &MatrixXd) -> Matrix6d {
    check_joint_covariance(covar);
    let mut j = MatrixXd::zeros(6, 12);
    j.set_block(0, 0, &MatrixXd::identity(6));
    j.set_block(0, 6, &MatrixXd::from(a_from_b.adjoint()));
    (&(&j * covar) * &j.transpose()).to_matrix6d()
}

/// Port of `colmap::GetCovarianceForRelativeRigid3d`: the 6x6 covariance of the relative
/// `b_from_a = b_from_c * inverse(a_from_c)` from the 12x12 joint covariance of
/// `(a_from_c, b_from_c)`, `J * covar * J^T` with `J = [-Ad(b_from_c) Ad(a_from_c)^-1, I]`.
///
/// # Panics
/// When `covar` is not 12x12 (a compile-time shape in COLMAP).
pub fn get_covariance_for_relative_rigid3d(
    a_from_c: &Rigid3d,
    b_from_c: &Rigid3d,
    covar: &MatrixXd,
) -> Matrix6d {
    check_joint_covariance(covar);
    let mut j = MatrixXd::zeros(6, 12);
    j.set_block(
        0,
        0,
        &MatrixXd::from(-b_from_c.adjoint() * a_from_c.adjoint_inverse()),
    );
    j.set_block(0, 6, &MatrixXd::identity(6));
    (&(&j * covar) * &j.transpose()).to_matrix6d()
}

// COLMAP's parameter is a fixed Eigen::Matrix<double, 12, 12>; the shape is part of the type
// there and a runtime check here.
fn check_joint_covariance(covar: &MatrixXd) {
    assert!(
        covar.rows() == 12 && covar.cols() == 12,
        "Expected a 12x12 covariance, got {}x{}.",
        covar.rows(),
        covar.cols()
    );
}

/// `x_in_b = b_from_a * x_in_a`: `R * x + t`.
///
/// As in COLMAP, `d_from_c * c_from_b * b_from_a * x` evaluates left to right (composing the
/// transforms first); write `d_from_c * (c_from_b * (b_from_a * x))` to chain on the point.
impl Mul<Vector3d> for Rigid3d {
    type Output = Vector3d;
    fn mul(self, x: Vector3d) -> Vector3d {
        self.rotation * x + self.translation
    }
}

/// `c_from_a = c_from_b * b_from_a`; the composed rotation is re-normalized as in COLMAP.
impl Mul for Rigid3d {
    type Output = Rigid3d;
    fn mul(self, b_from_a: Rigid3d) -> Rigid3d {
        Rigid3d::new(
            (self.rotation * b_from_a.rotation).normalized(),
            self.translation + self.rotation * b_from_a.translation,
        )
    }
}

/// Eigen's `IOFormat(StreamPrecision, DontAlignCols, ", ", ", ")` at the default stream
/// precision: the values joined by ", ".
pub(crate) fn format_list(values: &[f64]) -> String {
    values
        .iter()
        .map(|&v| format_double(v, DEFAULT_PRECISION))
        .collect::<Vec<_>>()
        .join(", ")
}

/// COLMAP's `operator<<`: `Rigid3d(rotation_xyzw=[x, y, z, w], translation=[x, y, z])`.
impl fmt::Display for Rigid3d {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let q = self.rotation;
        let t = self.translation;
        write!(
            f,
            "Rigid3d(rotation_xyzw=[{}], translation=[{}])",
            format_list(&[q.x, q.y, q.z, q.w]),
            format_list(&[t.x, t.y, t.z])
        )
    }
}
