//! The small matrices: [`Matrix2d`] (2D Jacobians and covariances), [`Matrix2x3d`] (the
//! projection Jacobian d(x, y) / d(u, v, w)) and [`Matrix3x2d`] (the unprojection Jacobian and
//! the unit-ray Jacobian). Port of colmap-sharp's `LinearAlgebra/Matrix2d.cs`,
//! `Matrix2x3d.cs` and `Matrix3x2d.cs`, written to Eigen's documented semantics; Eigen is not
//! ported (docs/LICENSE_AUDIT.md). Storage, operators and products come from
//! `matrix_macros.rs`; `Matrix3d * Matrix3x2d` is in `matrix3.rs`.

use super::{Matrix3d, Vector2d, Vector3d};

matrix_common!(
    /// Fixed-size 2x2 matrix of doubles. Replacement for `Eigen::Matrix2d`.
    Matrix2d, Matrix2d, 2, 2
);
matrix_common!(
    /// Fixed-size 2x3 matrix of doubles. Replacement for `Eigen::Matrix2x3d`.
    Matrix2x3d, Matrix3x2d, 2, 3
);
matrix_common!(
    /// Fixed-size 3x2 matrix of doubles. Replacement for `Eigen::Matrix3x2d`.
    Matrix3x2d, Matrix2x3d, 3, 2
);

matrix_vector_product!(Matrix2d, Vector2d, Vector2d, 2, 2);
matrix_vector_product!(Matrix2x3d, Vector3d, Vector2d, 2, 3);
matrix_vector_product!(Matrix3x2d, Vector2d, Vector3d, 3, 2);
matrix_product!(Matrix2d, Matrix2d, Matrix2d, 2, 2, 2);
matrix_product!(Matrix2x3d, Matrix3x2d, Matrix2d, 2, 3, 2);
matrix_product!(Matrix3x2d, Matrix2x3d, Matrix3d, 3, 2, 3);
matrix_product!(Matrix3x2d, Matrix2d, Matrix3x2d, 3, 2, 2);
matrix_product!(Matrix2x3d, Matrix3d, Matrix2x3d, 2, 3, 3);

impl Matrix2d {
    /// The matrix from its coefficients in row-major reading order, like Eigen's comma
    /// initializer.
    pub const fn new(m00: f64, m01: f64, m10: f64, m11: f64) -> Self {
        Self::from_column_major([m00, m10, m01, m11])
    }

    /// The identity matrix.
    pub const fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0)
    }

    /// The matrix whose columns are the given vectors.
    pub const fn from_columns(c0: Vector2d, c1: Vector2d) -> Self {
        Self::from_column_major([c0.x, c0.y, c1.x, c1.y])
    }

    /// The matrix whose rows are the given vectors.
    pub const fn from_rows(r0: Vector2d, r1: Vector2d) -> Self {
        Self::new(r0.x, r0.y, r1.x, r1.y)
    }

    /// Row `i`.
    pub fn row(self, i: usize) -> Vector2d {
        Vector2d::new(self[(i, 0)], self[(i, 1)])
    }

    /// Column `j`.
    pub fn col(self, j: usize) -> Vector2d {
        Vector2d::new(self[(0, j)], self[(1, j)])
    }

    /// Sum of the diagonal.
    pub fn trace(self) -> f64 {
        self.data[0] + self.data[3]
    }

    /// Determinant, `m00*m11 - m01*m10`.
    pub fn determinant(self) -> f64 {
        self.data[0] * self.data[3] - self.data[2] * self.data[1]
    }

    /// The inverse, `[m11 -m01; -m10 m00] / det`. No singularity check, like Eigen.
    pub fn inverse(self) -> Self {
        let det = self.determinant();
        let [m00, m10, m01, m11] = self.data;
        Self::new(m11 / det, -m01 / det, -m10 / det, m00 / det)
    }
}

impl Matrix2x3d {
    /// The matrix from its coefficients in row-major reading order, like Eigen's comma
    /// initializer.
    pub const fn new(m00: f64, m01: f64, m02: f64, m10: f64, m11: f64, m12: f64) -> Self {
        Self::from_column_major([m00, m10, m01, m11, m02, m12])
    }

    /// The matrix from 6 values in row-major order, like
    /// `Eigen::Map<const Eigen::Matrix<double, 2, 3, Eigen::RowMajor>>`.
    pub const fn from_row_major(v: [f64; 6]) -> Self {
        Self::new(v[0], v[1], v[2], v[3], v[4], v[5])
    }

    /// Row `i`.
    pub fn row(self, i: usize) -> Vector3d {
        Vector3d::new(self[(i, 0)], self[(i, 1)], self[(i, 2)])
    }

    /// Column `j`.
    pub fn col(self, j: usize) -> Vector2d {
        Vector2d::new(self[(0, j)], self[(1, j)])
    }
}

impl Matrix3x2d {
    /// The matrix from its coefficients in row-major reading order, like Eigen's comma
    /// initializer.
    pub const fn new(m00: f64, m01: f64, m10: f64, m11: f64, m20: f64, m21: f64) -> Self {
        Self::from_column_major([m00, m10, m20, m01, m11, m21])
    }

    /// The matrix whose columns are the given vectors.
    pub const fn from_columns(c0: Vector3d, c1: Vector3d) -> Self {
        Self::from_column_major([c0.x, c0.y, c0.z, c1.x, c1.y, c1.z])
    }

    /// Row `i`.
    pub fn row(self, i: usize) -> Vector2d {
        Vector2d::new(self[(i, 0)], self[(i, 1)])
    }

    /// Column `j`.
    pub fn col(self, j: usize) -> Vector3d {
        Vector3d::new(self[(0, j)], self[(1, j)], self[(2, j)])
    }
}
