//! Port of COLMAP's `src/colmap/math/random_eigen.h`: random vectors, matrices and unit
//! quaternions drawn from COLMAP's deterministic, thread-local PRNG ([`super::random`])
//! instead of Eigen's `rand()`-based `Random()` / `UnitRandom()`, so seeded results are
//! reproducible across platforms. Mirrors colmap-sharp's `Mathematics/RandomEigen.cs`.
//! Tests: `colmap-rust/tests/math/random_eigen.rs` (`random_eigen_test.cc` 1:1).
//!
//! The templates `RandomEigenMatrixd<Rows, Cols>` / `RandomEigenVectord<N>` (and the `f`
//! variants) become [`random_eigen_matrixd`] / [`random_eigen_vectord`] (and
//! [`random_eigen_matrixf`] / [`random_eigen_vectorf`]) generic over [`RandomEigenFixed`],
//! which every fixed-size `linalg` type implements, as do plain arrays `[f64; N]` /
//! `[f32; N]` for the sizes `linalg` has no type for (there are no float matrix types). The
//! dynamic variants return [`MatrixXd`] / [`VectorXd`], and for single precision a
//! [`RowMajorMatrix<f32>`] / `Vec<f32>`. The named shorthands [`random_eigen_vector2d`] ..
//! [`random_eigen_matrix6d`] are the shapes the geometry code and tests call most.
//!
//! Tier A (exact): each coefficient is one `RandomUniformReal(-1, 1)` draw, taken in Eigen's
//! linear index order (column-major for matrices, whatever the returned storage), so the
//! same seed gives the same values as COLMAP. [`random_eigen_quaterniond`] goes through
//! `sqrt`, `sin` and `cos` (`math::fns`, `docs/CPP_DIVERGENCES.md` entry 1).

use super::fns;
use super::random::{random_uniform_real, CanonicalFloat};
use crate::linalg::{
    Matrix2d, Matrix2x3d, Matrix3d, Matrix3x2d, Matrix3x4d, Matrix4d, Matrix4x3d, Matrix6d,
    MatrixXd, Quaterniond, RowMajorMatrix, Vector2d, Vector3d, Vector3f, Vector4d, VectorXd,
};

/// A fixed-size Eigen-like quantity whose coefficients [`random_eigen_matrixd`] and friends
/// fill, in Eigen's linear index order.
pub trait RandomEigenFixed: Sized {
    /// The coefficient type (`f64` for the `d` functions, `f32` for the `f` ones).
    type Scalar: CanonicalFloat;
    /// Builds the value from `draw()` called once per coefficient in linear index order
    /// (column-major for a matrix).
    fn from_draws(draw: impl FnMut() -> Self::Scalar) -> Self;
}

impl<T: CanonicalFloat, const N: usize> RandomEigenFixed for [T; N] {
    type Scalar = T;
    fn from_draws(mut draw: impl FnMut() -> T) -> Self {
        std::array::from_fn(|_| draw())
    }
}

macro_rules! impl_random_matrix {
    ($($t:ident),*) => {$(
        impl RandomEigenFixed for $t {
            type Scalar = f64;
            fn from_draws(draw: impl FnMut() -> f64) -> Self {
                $t::from_column_major(RandomEigenFixed::from_draws(draw))
            }
        }
    )*};
}

impl_random_matrix!(
    Matrix2d, Matrix2x3d, Matrix3x2d, Matrix3d, Matrix3x4d, Matrix4x3d, Matrix4d, Matrix6d
);

impl RandomEigenFixed for Vector2d {
    type Scalar = f64;
    fn from_draws(draw: impl FnMut() -> f64) -> Self {
        let [x, y] = RandomEigenFixed::from_draws(draw);
        Vector2d::new(x, y)
    }
}

impl RandomEigenFixed for Vector3d {
    type Scalar = f64;
    fn from_draws(draw: impl FnMut() -> f64) -> Self {
        let [x, y, z] = RandomEigenFixed::from_draws(draw);
        Vector3d::new(x, y, z)
    }
}

impl RandomEigenFixed for Vector4d {
    type Scalar = f64;
    fn from_draws(draw: impl FnMut() -> f64) -> Self {
        let [x, y, z, w] = RandomEigenFixed::from_draws(draw);
        Vector4d::new(x, y, z, w)
    }
}

impl RandomEigenFixed for Vector3f {
    type Scalar = f32;
    fn from_draws(draw: impl FnMut() -> f32) -> Self {
        let [x, y, z] = RandomEigenFixed::from_draws(draw);
        Vector3f::new(x, y, z)
    }
}

/// `internal::SetRandomEigen`: one `RandomUniformReal(-1, 1)` per coefficient.
fn set_random_eigen<M: RandomEigenFixed>() -> M {
    let one = <M::Scalar as CanonicalFloat>::ONE;
    M::from_draws(|| random_uniform_real(-one, one))
}

/// Random matrix with each entry uniformly distributed in [-1, 1], matching the value range
/// of `Eigen::Matrix<...>::Random()`. Port of `colmap::RandomEigenMatrixd<Rows, Cols>`.
pub fn random_eigen_matrixd<M: RandomEigenFixed<Scalar = f64>>() -> M {
    set_random_eigen()
}

/// Single-precision [`random_eigen_matrixd`]. Port of `colmap::RandomEigenMatrixf<Rows,
/// Cols>`; with an array type the coefficients are column-major.
pub fn random_eigen_matrixf<M: RandomEigenFixed<Scalar = f32>>() -> M {
    set_random_eigen()
}

/// Random column vector with each entry uniformly distributed in [-1, 1].
/// Port of `colmap::RandomEigenVectord<N>`.
pub fn random_eigen_vectord<V: RandomEigenFixed<Scalar = f64>>() -> V {
    set_random_eigen()
}

/// Single-precision [`random_eigen_vectord`]. Port of `colmap::RandomEigenVectorf<N>`.
pub fn random_eigen_vectorf<V: RandomEigenFixed<Scalar = f32>>() -> V {
    set_random_eigen()
}

/// `RandomEigenVectord<2>()`, [`random_eigen_vectord`] for [`Vector2d`].
pub fn random_eigen_vector2d() -> Vector2d {
    random_eigen_vectord()
}

/// `RandomEigenVectord<3>()`, [`random_eigen_vectord`] for [`Vector3d`].
pub fn random_eigen_vector3d() -> Vector3d {
    random_eigen_vectord()
}

/// `RandomEigenVectord<4>()`, [`random_eigen_vectord`] for [`Vector4d`].
pub fn random_eigen_vector4d() -> Vector4d {
    random_eigen_vectord()
}

/// `RandomEigenMatrixd<3, 3>()`, [`random_eigen_matrixd`] for [`Matrix3d`].
pub fn random_eigen_matrix3d() -> Matrix3d {
    random_eigen_matrixd()
}

/// `RandomEigenMatrixd<6, 6>()`, [`random_eigen_matrixd`] for [`Matrix6d`].
pub fn random_eigen_matrix6d() -> Matrix6d {
    random_eigen_matrixd()
}

/// Dynamically sized [`random_eigen_matrixd`]. Port of `colmap::RandomEigenMatrixXd`.
pub fn random_eigen_matrix_xd(rows: usize, cols: usize) -> MatrixXd {
    let values: Vec<f64> = (0..rows * cols)
        .map(|_| random_uniform_real(-1.0, 1.0))
        .collect();
    MatrixXd::from_column_major_vec(rows, cols, values)
}

/// Dynamically sized [`random_eigen_matrixf`]. Port of `colmap::RandomEigenMatrixXf`. The
/// draws fill the matrix column by column (Eigen's linear order for its column-major
/// `MatrixXf`), although the returned storage is row-major.
pub fn random_eigen_matrix_xf(rows: usize, cols: usize) -> RowMajorMatrix<f32> {
    let mut matrix = RowMajorMatrix::new(rows, cols);
    let data = matrix.as_mut_slice();
    for col in 0..cols {
        for row in 0..rows {
            data[row * cols + col] = random_uniform_real(-1.0f32, 1.0f32);
        }
    }
    matrix
}

/// Dynamically sized [`random_eigen_vectord`]. Port of `colmap::RandomEigenVectorXd`.
pub fn random_eigen_vector_xd(size: usize) -> VectorXd {
    VectorXd::from_vec((0..size).map(|_| random_uniform_real(-1.0, 1.0)).collect())
}

/// Dynamically sized [`random_eigen_vectorf`]. Port of `colmap::RandomEigenVectorXf`.
pub fn random_eigen_vector_xf(size: usize) -> Vec<f32> {
    (0..size)
        .map(|_| random_uniform_real(-1.0f32, 1.0f32))
        .collect()
}

/// Uniformly distributed random unit quaternion, matching
/// `Eigen::Quaterniond::UnitRandom()` (Shoemake's method).
/// Port of `colmap::RandomEigenQuaterniond`.
pub fn random_eigen_quaterniond() -> Quaterniond {
    let u1 = random_uniform_real(0.0, 1.0);
    let u2 = random_uniform_real(0.0, 2.0 * std::f64::consts::PI);
    let u3 = random_uniform_real(0.0, 2.0 * std::f64::consts::PI);
    let a = fns::sqrt(1.0 - u1);
    let b = fns::sqrt(u1);
    Quaterniond::new(
        a * fns::sin(u2),
        a * fns::cos(u2),
        b * fns::sin(u3),
        b * fns::cos(u3),
    )
}
