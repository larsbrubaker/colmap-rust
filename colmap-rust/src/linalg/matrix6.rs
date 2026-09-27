//! [`Matrix6d`]: fixed-size 6x6 matrix of doubles, the replacement for COLMAP's
//! `Eigen::Matrix6d` (the `Rigid3d` adjoint and 6-DoF pose covariances). Port of
//! colmap-sharp's `LinearAlgebra/Matrix6d.cs`, written to Eigen's documented semantics; Eigen
//! is not ported (docs/LICENSE_AUDIT.md). Storage and operators come from `matrix_macros.rs`;
//! its 3x3 blocks are [`Matrix3d`]s.
//!
//! Tier B for products: Eigen evaluates a 6x6 product with its vectorized kernel, which the
//! pycolmap wheel may contract into FMAs, and no pycolmap binding exposes a 6x6 product to pin
//! it. The product here is the module's left-to-right sum (docs/CPP_DIVERGENCES.md, entry 5).

use super::Matrix3d;

matrix_common!(
    /// Fixed-size 6x6 matrix of doubles. Replacement for `Eigen::Matrix6d`.
    Matrix6d, Matrix6d, 6, 6
);

matrix_product!(Matrix6d, Matrix6d, Matrix6d, 6, 6, 6);

impl Matrix6d {
    /// The identity matrix.
    pub const fn identity() -> Self {
        let mut data = [0.0; 36];
        let mut i = 0;
        while i < 6 {
            data[i * 6 + i] = 1.0;
            i += 1;
        }
        Self::from_column_major(data)
    }

    /// `[top_left, top_right; bottom_left, bottom_right]`, Eigen's `block<3, 3>(r, c)`
    /// assignments.
    pub fn from_blocks(
        top_left: Matrix3d,
        top_right: Matrix3d,
        bottom_left: Matrix3d,
        bottom_right: Matrix3d,
    ) -> Self {
        let mut m = Self::zeros();
        for col in 0..3 {
            for row in 0..3 {
                m[(row, col)] = top_left[(row, col)];
                m[(row, col + 3)] = top_right[(row, col)];
                m[(row + 3, col)] = bottom_left[(row, col)];
                m[(row + 3, col + 3)] = bottom_right[(row, col)];
            }
        }
        m
    }

    /// The 3x3 block starting at `(row, col)`, Eigen's `block<3, 3>(row, col)`.
    pub fn block3(self, row: usize, col: usize) -> Matrix3d {
        let mut m = Matrix3d::zeros();
        for c in 0..3 {
            for r in 0..3 {
                m[(r, c)] = self[(row + r, col + c)];
            }
        }
        m
    }
}
