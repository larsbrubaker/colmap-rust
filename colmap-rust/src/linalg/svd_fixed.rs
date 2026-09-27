//! Svd3d / Svd4d: allocation-free SVDs of [`Matrix3d`] and [`Matrix4d`], the replacement for
//! `Eigen::JacobiSVD<Matrix3d>` / `JacobiSVD<Matrix4d>` with `ComputeFullU | ComputeFullV`,
//! which COLMAP runs in hot paths (ComputeClosestRotationMatrix, essential/fundamental matrix
//! decomposition and rank-2 enforcement, the P3P/EPnP/triangulation null spaces). Port of
//! colmap-sharp's `ColmapSharp/LinearAlgebra/SvdFixed.cs`; Eigen (MPL-2.0) is not ported.
//!
//! They run the same kernel as [`JacobiSvd`](super::JacobiSvd) (`jacobi_svd_kernel.rs`, two-sided
//! Jacobi after Golub & Van Loan §8.6.3 and Brent-Luk-Van Loan) on stack arrays, so their
//! results are bit-identical to `JacobiSvd` on the same matrix. Semantics and sign
//! conventions: `jacobi_svd_kernel.rs`. Tier B.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_spectral_svd.rs` and
//! `rust_only_spectral_oracle.rs` (bit-identity to `JacobiSvd`).

use super::jacobi_svd_kernel;
use super::{ComputationInfo, Matrix3d, Matrix4d, Vector3d, Vector4d};

/// SVD of a Matrix3d, `M = U diag(singular_values) V^T`, with full U and V.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Svd3d {
    /// Left singular vectors (columns).
    pub matrix_u: Matrix3d,
    /// Right singular vectors (columns).
    pub matrix_v: Matrix3d,
    /// Singular values, non-negative and decreasing.
    pub singular_values: Vector3d,
    /// Success, or InvalidInput for a non-finite entry.
    pub info: ComputationInfo,
}

impl Svd3d {
    /// Decomposes `m`.
    pub fn compute(m: &Matrix3d) -> Self {
        let mut a = m.to_column_major();
        let mut u = [0.0; 9];
        let mut v = [0.0; 9];
        let mut s = [0.0; 3];
        let (info, _) = jacobi_svd_kernel::decompose(&mut a, 3, &mut u, &mut v, &mut s);
        Self {
            matrix_u: Matrix3d::from_column_major(u),
            matrix_v: Matrix3d::from_column_major(v),
            singular_values: Vector3d::new(s[0], s[1], s[2]),
            info,
        }
    }

    /// Numerical rank, JacobiSVD's rule (`max(1, 3) * eps *` the largest singular value).
    pub fn rank(&self) -> usize {
        let s = self.singular_values;
        jacobi_svd_kernel::rank(&[s.x, s.y, s.z])
    }
}

/// SVD of a Matrix4d, `M = U diag(singular_values) V^T`, with full U and V.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Svd4d {
    /// Left singular vectors (columns).
    pub matrix_u: Matrix4d,
    /// Right singular vectors (columns).
    pub matrix_v: Matrix4d,
    /// Singular values, non-negative and decreasing.
    pub singular_values: Vector4d,
    /// Success, or InvalidInput for a non-finite entry.
    pub info: ComputationInfo,
}

impl Svd4d {
    /// Decomposes `m`.
    pub fn compute(m: &Matrix4d) -> Self {
        let mut a = m.to_column_major();
        let mut u = [0.0; 16];
        let mut v = [0.0; 16];
        let mut s = [0.0; 4];
        let (info, _) = jacobi_svd_kernel::decompose(&mut a, 4, &mut u, &mut v, &mut s);
        Self {
            matrix_u: Matrix4d::from_column_major(u),
            matrix_v: Matrix4d::from_column_major(v),
            singular_values: Vector4d::new(s[0], s[1], s[2], s[3]),
            info,
        }
    }

    /// Numerical rank, JacobiSVD's rule (`max(1, 4) * eps *` the largest singular value).
    pub fn rank(&self) -> usize {
        let s = self.singular_values;
        jacobi_svd_kernel::rank(&[s.x, s.y, s.z, s.w])
    }
}
