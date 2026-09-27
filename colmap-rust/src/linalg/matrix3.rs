//! [`Matrix3d`]: fixed-size 3x3 matrix of doubles, the replacement for `Eigen::Matrix3d`
//! (rotations, calibration matrices, essential/fundamental/homography matrices). Port of
//! colmap-sharp's `LinearAlgebra/Matrix3d.cs`, written to Eigen's documented semantics; Eigen
//! is not ported (docs/LICENSE_AUDIT.md). Storage and operators come from `matrix_macros.rs`;
//! [`super::Quaterniond`] and [`super::AngleAxisd`] convert to and from it.
//!
//! `determinant` and `inverse` use the closed-form cofactor expansion along the first row.
//! Tier B and unpinned: Eigen's closed form may order the terms differently, and no pycolmap
//! 4.2.0 binding reaches Eigen's 3x3 inverse or determinant cleanly, so there is no oracle
//! fixture for them. Like Eigen, `inverse` does not check for singularity; a singular matrix
//! yields infinities or NaN.

use super::{Matrix3x2d, Vector3d};

matrix_common!(
    /// Fixed-size 3x3 matrix of doubles. Replacement for `Eigen::Matrix3d`.
    Matrix3d, Matrix3d, 3, 3
);

matrix_vector_product!(Matrix3d, Vector3d, Vector3d, 3, 3);
matrix_product!(Matrix3d, Matrix3d, Matrix3d, 3, 3, 3);
matrix_product!(Matrix3d, Matrix3x2d, Matrix3x2d, 3, 3, 2);

impl Matrix3d {
    /// The matrix from its coefficients in row-major reading order, like Eigen's comma
    /// initializer.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        m00: f64,
        m01: f64,
        m02: f64,
        m10: f64,
        m11: f64,
        m12: f64,
        m20: f64,
        m21: f64,
        m22: f64,
    ) -> Self {
        Self::from_column_major([m00, m10, m20, m01, m11, m21, m02, m12, m22])
    }

    /// The identity matrix.
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)
    }

    /// Eigen's `Matrix3d::Ones()`.
    pub const fn ones() -> Self {
        Self::from_column_major([1.0; 9])
    }

    /// The matrix whose rows are the given vectors.
    pub const fn from_rows(r0: Vector3d, r1: Vector3d, r2: Vector3d) -> Self {
        Self::new(r0.x, r0.y, r0.z, r1.x, r1.y, r1.z, r2.x, r2.y, r2.z)
    }

    /// The matrix whose columns are the given vectors.
    pub const fn from_columns(c0: Vector3d, c1: Vector3d, c2: Vector3d) -> Self {
        Self::from_column_major([c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z])
    }

    /// The diagonal matrix, Eigen's `v.asDiagonal()`.
    pub const fn from_diagonal(d: Vector3d) -> Self {
        Self::new(d.x, 0.0, 0.0, 0.0, d.y, 0.0, 0.0, 0.0, d.z)
    }

    /// Row `i`.
    pub fn row(self, i: usize) -> Vector3d {
        Vector3d::new(self[(i, 0)], self[(i, 1)], self[(i, 2)])
    }

    /// Column `j`.
    pub fn col(self, j: usize) -> Vector3d {
        Vector3d::new(self[(0, j)], self[(1, j)], self[(2, j)])
    }

    /// The diagonal `(m00, m11, m22)`.
    pub const fn diagonal(self) -> Vector3d {
        Vector3d::new(self.data[0], self.data[4], self.data[8])
    }

    /// Sum of the diagonal, grouped `m00 + (m11 + m22)`. That is how Eigen's unrolled
    /// reduction over the (strided, so not vectorized) diagonal splits it, as
    /// `oracle/linear_algebra_rotations.py` shows through `Quaterniond::from_rotation_matrix`:
    /// bit-identical on all fixture cases with this grouping, 4 mismatches left to right.
    pub fn trace(self) -> f64 {
        self.data[0] + (self.data[4] + self.data[8])
    }

    /// Determinant by cofactor expansion along the first row.
    pub fn determinant(self) -> f64 {
        let [m00, m10, m20, m01, m11, m21, m02, m12, m22] = self.data;
        let c00 = m11 * m22 - m12 * m21;
        let c01 = m12 * m20 - m10 * m22;
        let c02 = m10 * m21 - m11 * m20;
        m00 * c00 + m01 * c01 + m02 * c02
    }

    /// The inverse, the adjugate (transposed cofactor matrix) divided by the determinant. No
    /// singularity check, like Eigen's `inverse()`.
    pub fn inverse(self) -> Self {
        let [m00, m10, m20, m01, m11, m21, m02, m12, m22] = self.data;

        // Cofactors c(i, j) of element (i, j).
        let c00 = m11 * m22 - m12 * m21;
        let c01 = m12 * m20 - m10 * m22;
        let c02 = m10 * m21 - m11 * m20;
        let c10 = m02 * m21 - m01 * m22;
        let c11 = m00 * m22 - m02 * m20;
        let c12 = m01 * m20 - m00 * m21;
        let c20 = m01 * m12 - m02 * m11;
        let c21 = m02 * m10 - m00 * m12;
        let c22 = m00 * m11 - m01 * m10;
        let det = m00 * c00 + m01 * c01 + m02 * c02;

        // inverse(i, j) = c(j, i) / det.
        Self::new(
            c00 / det,
            c10 / det,
            c20 / det,
            c01 / det,
            c11 / det,
            c21 / det,
            c02 / det,
            c12 / det,
            c22 / det,
        )
    }
}
