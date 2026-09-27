//! Port of COLMAP's `colmap/math/random_eigen.h`: random Eigen-type values drawn from COLMAP's
//! seeded PRNG ([`super::random`]) instead of Eigen's `Random()` / `UnitRandom()` (which use
//! the platform `rand()`). Only the fixed-size helpers the port needs so far are here; the
//! dynamic-size variants (`RandomEigenMatrixXd`, `RandomEigenVectorXd`) arrive with the
//! dynamic-size matrices.
//!
//! Tier A (exact): each coefficient is one `RandomUniformReal<double>(-1, 1)` draw, filled in
//! Eigen's linear (column-major) order, and the quaternion is Shoemake's method exactly as
//! COLMAP writes it (`sin`/`cos` through [`super::fns`], docs/CPP_DIVERGENCES.md entry 1).
//! Used by the ported geometry tests (`tests/geometry/`).

use super::fns;
use super::random::random_uniform_real;
use crate::linalg::{Matrix3d, Matrix6d, Quaterniond, Vector2d, Vector3d, Vector4d};

/// `EIGEN_PI` as a double.
const EIGEN_PI: f64 = std::f64::consts::PI;

fn random_coefficients<const N: usize>() -> [f64; N] {
    let mut values = [0.0; N];
    for value in &mut values {
        *value = random_uniform_real::<f64>(-1.0, 1.0);
    }
    values
}

/// `RandomEigenVectord<2>()`: each entry uniform in [-1, 1].
pub fn random_eigen_vector2d() -> Vector2d {
    Vector2d::from_array(random_coefficients())
}

/// `RandomEigenVectord<3>()`: each entry uniform in [-1, 1].
pub fn random_eigen_vector3d() -> Vector3d {
    Vector3d::from_array(random_coefficients())
}

/// `RandomEigenVectord<4>()`: each entry uniform in [-1, 1].
pub fn random_eigen_vector4d() -> Vector4d {
    Vector4d::from_array(random_coefficients())
}

/// `RandomEigenMatrixd<3, 3>()`: each entry uniform in [-1, 1], column-major fill order.
pub fn random_eigen_matrix3d() -> Matrix3d {
    Matrix3d::from_column_major(random_coefficients())
}

/// `RandomEigenMatrixd<6, 6>()`: each entry uniform in [-1, 1], column-major fill order.
pub fn random_eigen_matrix6d() -> Matrix6d {
    Matrix6d::from_column_major(random_coefficients())
}

/// `RandomEigenQuaterniond()`: a uniformly distributed unit quaternion (Shoemake's method,
/// matching `Eigen::Quaterniond::UnitRandom()`). The three draws are made in COLMAP's order.
pub fn random_eigen_quaterniond() -> Quaterniond {
    let u1 = random_uniform_real::<f64>(0.0, 1.0);
    let u2 = random_uniform_real::<f64>(0.0, 2.0 * EIGEN_PI);
    let u3 = random_uniform_real::<f64>(0.0, 2.0 * EIGEN_PI);
    let a = fns::sqrt(1.0 - u1);
    let b = fns::sqrt(u1);
    Quaterniond::new(
        a * fns::sin(u2),
        a * fns::cos(u2),
        b * fns::sin(u3),
        b * fns::cos(u3),
    )
}
