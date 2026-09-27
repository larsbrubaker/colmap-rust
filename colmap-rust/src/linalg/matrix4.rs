//! [`Matrix3x4d`] (camera projection matrices `[R | t]`, `Rigid3d`/`Sim3d::ToMatrix`) and
//! [`Matrix4d`] (homogeneous transforms, quaternion-averaging accumulators). Port of
//! colmap-sharp's `LinearAlgebra/Matrix3x4d.cs` and `Matrix4d.cs`, written to Eigen's
//! documented semantics; Eigen is not ported (docs/LICENSE_AUDIT.md). Storage and operators
//! come from `matrix_macros.rs`.
//!
//! COLMAP's `P * X.homogeneous()` is `p * x.homogeneous()` here, a full four-term sum per row
//! whose last term multiplies by exactly 1.
//!
//! `Matrix4d::determinant` and `inverse` use the Laplace expansion by complementary 2x2 minors
//! of the top two and bottom two rows (D. Eberly, "The Laplace Expansion Theorem: Computing
//! the Determinants and Inverses of Matrices", Geometric Tools, 2008), written from the
//! published formulas. Tier B: Eigen's closed form may order the terms differently. No
//! singularity check, like Eigen.

use super::{Matrix3d, Vector3d, Vector4d};

matrix_common!(
    /// Fixed-size 3x4 matrix of doubles. Replacement for `Eigen::Matrix3x4d`.
    Matrix3x4d, Matrix4x3d, 3, 4
);
matrix_common!(
    /// Fixed-size 4x4 matrix of doubles. Replacement for `Eigen::Matrix4d`.
    Matrix4d, Matrix4d, 4, 4
);
matrix_common!(
    /// Fixed-size 4x3 matrix of doubles: only the transpose of a [`Matrix3x4d`].
    Matrix4x3d, Matrix3x4d, 4, 3
);

matrix_vector_product!(Matrix3x4d, Vector4d, Vector3d, 3, 4);
matrix_vector_product!(Matrix4d, Vector4d, Vector4d, 4, 4);
matrix_product!(Matrix3d, Matrix3x4d, Matrix3x4d, 3, 3, 4);
matrix_product!(Matrix3x4d, Matrix4d, Matrix3x4d, 3, 4, 4);
matrix_product!(Matrix4d, Matrix4d, Matrix4d, 4, 4, 4);

impl Matrix3x4d {
    /// The matrix from its coefficients in row-major reading order, like Eigen's comma
    /// initializer.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        m00: f64,
        m01: f64,
        m02: f64,
        m03: f64,
        m10: f64,
        m11: f64,
        m12: f64,
        m13: f64,
        m20: f64,
        m21: f64,
        m22: f64,
        m23: f64,
    ) -> Self {
        Self::from_column_major([m00, m10, m20, m01, m11, m21, m02, m12, m22, m03, m13, m23])
    }

    /// Eigen's `Identity()` for a 3x4 matrix: `[I | 0]`.
    pub const fn identity() -> Self {
        Self::from_blocks(Matrix3d::identity(), Vector3d::zeros())
    }

    /// `[left | col3]`, as COLMAP builds `matrix.leftCols<3>() = R; matrix.col(3) = t`.
    pub const fn from_blocks(left: Matrix3d, col3: Vector3d) -> Self {
        let l = left.to_column_major();
        Self::from_column_major([
            l[0], l[1], l[2], l[3], l[4], l[5], l[6], l[7], l[8], col3.x, col3.y, col3.z,
        ])
    }

    /// The matrix whose columns are the given vectors.
    pub const fn from_columns(c0: Vector3d, c1: Vector3d, c2: Vector3d, c3: Vector3d) -> Self {
        Self::from_column_major([
            c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z, c3.x, c3.y, c3.z,
        ])
    }

    /// Row `i`.
    pub fn row(self, i: usize) -> Vector4d {
        Vector4d::new(self[(i, 0)], self[(i, 1)], self[(i, 2)], self[(i, 3)])
    }

    /// Column `j` (`col(3)` is Eigen's `rightCols<1>()`).
    pub fn col(self, j: usize) -> Vector3d {
        Vector3d::new(self[(0, j)], self[(1, j)], self[(2, j)])
    }

    /// The left 3x3 block, Eigen's `leftCols<3>()`.
    pub fn left_cols3(self) -> Matrix3d {
        let mut m = [0.0; 9];
        m.copy_from_slice(&self.data[..9]);
        Matrix3d::from_column_major(m)
    }
}

impl Matrix4d {
    /// The matrix from its coefficients in row-major reading order, like Eigen's comma
    /// initializer.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        m00: f64,
        m01: f64,
        m02: f64,
        m03: f64,
        m10: f64,
        m11: f64,
        m12: f64,
        m13: f64,
        m20: f64,
        m21: f64,
        m22: f64,
        m23: f64,
        m30: f64,
        m31: f64,
        m32: f64,
        m33: f64,
    ) -> Self {
        Self::from_column_major([
            m00, m10, m20, m30, m01, m11, m21, m31, m02, m12, m22, m32, m03, m13, m23, m33,
        ])
    }

    /// The identity matrix.
    pub const fn identity() -> Self {
        Self::new(
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        )
    }

    /// The homogeneous transform with top rows `p` and bottom row `(0, 0, 0, 1)`, as COLMAP
    /// builds it with `Matrix4d::Identity()` then `topRows<3>() = P`.
    pub const fn from_top_rows(p: Matrix3x4d) -> Self {
        let d = p.to_column_major();
        Self::from_column_major([
            d[0], d[1], d[2], 0.0, d[3], d[4], d[5], 0.0, d[6], d[7], d[8], 0.0, d[9], d[10],
            d[11], 1.0,
        ])
    }

    /// The matrix whose rows are the given vectors.
    pub const fn from_rows(r0: Vector4d, r1: Vector4d, r2: Vector4d, r3: Vector4d) -> Self {
        Self::new(
            r0.x, r0.y, r0.z, r0.w, r1.x, r1.y, r1.z, r1.w, r2.x, r2.y, r2.z, r2.w, r3.x, r3.y,
            r3.z, r3.w,
        )
    }

    /// Row `i`.
    pub fn row(self, i: usize) -> Vector4d {
        Vector4d::new(self[(i, 0)], self[(i, 1)], self[(i, 2)], self[(i, 3)])
    }

    /// Column `j`.
    pub fn col(self, j: usize) -> Vector4d {
        Vector4d::new(self[(0, j)], self[(1, j)], self[(2, j)], self[(3, j)])
    }

    /// The top three rows, Eigen's `topRows<3>()`.
    pub fn top_rows3(self) -> Matrix3x4d {
        Matrix3x4d::from_columns(
            self.col(0).head3(),
            self.col(1).head3(),
            self.col(2).head3(),
            self.col(3).head3(),
        )
    }

    /// The top-left 3x3 block, Eigen's `topLeftCorner<3, 3>()`.
    pub fn top_left3x3(self) -> Matrix3d {
        Matrix3d::from_columns(
            self.col(0).head3(),
            self.col(1).head3(),
            self.col(2).head3(),
        )
    }

    /// Sum of the diagonal, left to right. Unverified: `Matrix3d::trace` showed Eigen groups a
    /// diagonal sum its own way, so this grouping may not match Eigen bit for bit.
    pub fn trace(self) -> f64 {
        self.data[0] + self.data[5] + self.data[10] + self.data[15]
    }

    /// Determinant by Laplace expansion over 2x2 minors of rows 0-1 and rows 2-3.
    pub fn determinant(self) -> f64 {
        let (s, c) = self.minors();
        s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0]
    }

    /// The inverse, the adjugate divided by the determinant, both from the Laplace expansion
    /// over 2x2 minors. No singularity check, like Eigen's `inverse()`.
    pub fn inverse(self) -> Self {
        let (s, c) = self.minors();
        let det = s[0] * c[5] - s[1] * c[4] + s[2] * c[3] + s[3] * c[2] - s[4] * c[1] + s[5] * c[0];
        let [a00, a10, a20, a30, a01, a11, a21, a31, a02, a12, a22, a32, a03, a13, a23, a33] =
            self.data;

        Self::new(
            (a11 * c[5] - a12 * c[4] + a13 * c[3]) / det,
            (-a01 * c[5] + a02 * c[4] - a03 * c[3]) / det,
            (a31 * s[5] - a32 * s[4] + a33 * s[3]) / det,
            (-a21 * s[5] + a22 * s[4] - a23 * s[3]) / det,
            (-a10 * c[5] + a12 * c[2] - a13 * c[1]) / det,
            (a00 * c[5] - a02 * c[2] + a03 * c[1]) / det,
            (-a30 * s[5] + a32 * s[2] - a33 * s[1]) / det,
            (a20 * s[5] - a22 * s[2] + a23 * s[1]) / det,
            (a10 * c[4] - a11 * c[2] + a13 * c[0]) / det,
            (-a00 * c[4] + a01 * c[2] - a03 * c[0]) / det,
            (a30 * s[4] - a31 * s[2] + a33 * s[0]) / det,
            (-a20 * s[4] + a21 * s[2] - a23 * s[0]) / det,
            (-a10 * c[3] + a11 * c[1] - a12 * c[0]) / det,
            (a00 * c[3] - a01 * c[1] + a02 * c[0]) / det,
            (-a30 * s[3] + a31 * s[1] - a32 * s[0]) / det,
            (a20 * s[3] - a21 * s[1] + a22 * s[0]) / det,
        )
    }

    // The six 2x2 minors of rows 0-1 (s) and of rows 2-3 (c), indexed as in Eberly's paper:
    // s0 = cols 01, s1 = 02, s2 = 03, s3 = 12, s4 = 13, s5 = 23; c5 pairs with s0, etc.
    fn minors(self) -> ([f64; 6], [f64; 6]) {
        let [a00, a10, a20, a30, a01, a11, a21, a31, a02, a12, a22, a32, a03, a13, a23, a33] =
            self.data;
        let s = [
            a00 * a11 - a10 * a01,
            a00 * a12 - a10 * a02,
            a00 * a13 - a10 * a03,
            a01 * a12 - a11 * a02,
            a01 * a13 - a11 * a03,
            a02 * a13 - a12 * a03,
        ];
        let c = [
            a20 * a31 - a30 * a21,
            a20 * a32 - a30 * a22,
            a20 * a33 - a30 * a23,
            a21 * a32 - a31 * a22,
            a21 * a33 - a31 * a23,
            a22 * a33 - a32 * a23,
        ];
        (s, c)
    }
}
