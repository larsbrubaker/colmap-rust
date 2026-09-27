//! Fixed-size linear algebra: the replacement for the parts of Eigen COLMAP uses as value
//! types (`Eigen::Vector3d`, `Matrix3d`, `Quaterniond`, `AngleAxisd`, `AlignedBox3d`, ...).
//!
//! Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md). This module is a port of
//! colmap-sharp's own replacement (`ColmapSharp/LinearAlgebra/`, MIT), written to Eigen's
//! documented semantics:
//! - Matrices store their coefficients column-major, like Eigen's default, so `as_slice()` /
//!   `from_column_major` match `matrix.data()`. The `new` constructors take the values in
//!   row-major reading order, like Eigen's comma initializer (`m << a, b, c, ...`). A
//!   coefficient is `m[(row, col)]`.
//! - `Quaterniond::new(w, x, y, z)`; the fields are stored in Eigen's memory order
//!   `(x, y, z, w)`, which is also `coeffs()`.
//! - `normalized()` of a zero vector (or quaternion) returns it unchanged, not NaN.
//! - `is_approx` is Eigen's relative rule, `||a - b|| <= precision * min(||a||, ||b||)`, so
//!   only an exact zero is approximately zero. `is_approx_with` takes the precision.
//! - `==` is Eigen's coefficient-wise `==` (NaN never equal, `-0.0 == 0.0`).
//!
//! Arithmetic-order contract for the whole module (Tier A code downstream, pose and transform
//! algebra and the camera models, needs every sum to round the same way on every target):
//! - Dot products, norms and matrix products are plain left-to-right sums starting from the
//!   first term, `a0*b0 + a1*b1 + a2*b2`, never seeded with `0.0` (that would turn a `-0.0`
//!   result into `+0.0`).
//! - No FMA (`mul_add`) anywhere.
//! - Exceptions, each pinned by `oracle/linear_algebra_rotations.py`: where Eigen vectorizes a
//!   reduction or a product it pairs terms differently, and this module copies what the
//!   pycolmap oracle shows: [`Vector4d`]'s reductions, [`Quaterniond`]'s product and
//!   [`Matrix3d::trace`]. Elsewhere Eigen's SIMD order is not reproduced
//!   (docs/CPP_DIVERGENCES.md, entry 5).
//! - Transcendentals go through [`crate::math::fns`].
//!
//! Dynamic-size matrices ([`MatrixXd`], [`VectorXd`]) and the row-major feature-data container
//! ([`RowMajorMatrix`]) live here too, with the dense decompositions COLMAP calls (QR, LU,
//! Cholesky, SVD, eigen), each ported from colmap-sharp's own implementation (written from
//! Golub & Van Loan and the other published algorithms its file headers cite). Sparse
//! matrices are not here. Tests: `colmap-rust/tests/linalg.rs`.

#[macro_use]
mod matrix_macros;
mod aligned_box;
mod angle_axis;
mod matrix3;
mod matrix4;
mod matrix6;
mod matrix_small;
mod matrix_x;
mod matrix_x_ops;
mod quaternion;
mod row_major_matrix;
mod vector;
mod vector_x;

pub use aligned_box::AlignedBox3d;
pub use angle_axis::AngleAxisd;
pub use matrix3::Matrix3d;
pub use matrix4::{Matrix3x4d, Matrix4d, Matrix4x3d};
pub use matrix6::Matrix6d;
pub use matrix_small::{Matrix2d, Matrix2x3d, Matrix3x2d};
pub use matrix_x::MatrixXd;
pub use quaternion::Quaterniond;
pub use row_major_matrix::RowMajorMatrix;
pub use vector::{Vector2d, Vector3d, Vector3f, Vector3ub, Vector4d};
pub use vector_x::VectorXd;

use crate::math::fns;

/// Eigen's `NumTraits<double>::dummy_precision()`, the default precision of `isApprox`.
pub const DUMMY_PRECISION: f64 = 1e-12;

/// `std::numeric_limits<double>::epsilon()`.
pub const MACHINE_EPSILON: f64 = f64::EPSILON;

/// Outcome of a decomposition, Eigen's `ComputationInfo`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ComputationInfo {
    /// The decomposition succeeded.
    Success,
    /// The input did not have the required properties (e.g. not positive definite).
    NumericalIssue,
    /// An iterative decomposition (`EigenSolver`) did not converge.
    NoConvergence,
    /// The input contained a non-finite value (NaN or infinity).
    InvalidInput,
}

/// `std::min` on doubles, `(b < a) ? b : a` (Eigen's `numext::mini`). Unlike `f64::min` it
/// returns `a` when a NaN makes the comparison false.
#[inline]
pub(crate) fn mini(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `std::max` on doubles, `(a < b) ? b : a`.
#[inline]
pub(crate) fn maxi(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

/// Eigen's `isApprox` on flat coefficient slices: `||a - b|| <= precision * min(||a||, ||b||)`
/// with Frobenius (l2) norms, each summed left to right.
pub(crate) fn is_approx_slices(a: &[f64], b: &[f64], precision: f64) -> bool {
    debug_assert_eq!(a.len(), b.len());
    let mut difference_squared = 0.0;
    let mut a_squared = 0.0;
    let mut b_squared = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        let difference = x - y;
        difference_squared += difference * difference;
        a_squared += x * x;
        b_squared += y * y;
    }
    fns::sqrt(difference_squared) <= precision * mini(fns::sqrt(a_squared), fns::sqrt(b_squared))
}

/// Dot product of two equal-length slices, `a0*b0 + a1*b1 + ...` summed left to right from
/// the first term (0 when empty), no FMA. colmap-sharp's `VectorXd.Dot(span, span)`.
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    if a.is_empty() {
        return 0.0;
    }
    let mut sum = a[0] * b[0];
    for (&x, &y) in a[1..].iter().zip(&b[1..]) {
        sum += x * y;
    }
    sum
}

/// Frobenius norm of a coefficient slice, `sqrt(a0*a0 + a1*a1 + ...)`, left to right from the
/// first term.
pub(crate) fn frobenius_norm(a: &[f64]) -> f64 {
    let mut sum = a[0] * a[0];
    for &x in &a[1..] {
        sum += x * x;
    }
    fns::sqrt(sum)
}

/// Column-major product `out = a * b` of an `r x k` and a `k x c` matrix (a vector is a
/// `k x 1` matrix). Each coefficient is `a(i,0)*b(0,j) + a(i,1)*b(1,j) + ...`, summed left to
/// right from the first term, no FMA.
pub(crate) fn product(a: &[f64], b: &[f64], r: usize, k: usize, c: usize, out: &mut [f64]) {
    for j in 0..c {
        for i in 0..r {
            let mut sum = a[i] * b[j * k];
            for l in 1..k {
                sum += a[l * r + i] * b[j * k + l];
            }
            out[j * r + i] = sum;
        }
    }
}

// Dense decompositions (decomposition agent; the lead may move these up).
mod col_piv_householder_qr;
mod full_piv_lu;
pub mod householder;
mod householder_qr;
mod ldlt;
mod llt;
mod partial_piv_lu;
pub use col_piv_householder_qr::ColPivHouseholderQr;
pub use full_piv_lu::FullPivLu;
pub use householder_qr::HouseholderQr;
pub use ldlt::Ldlt;
pub use llt::Llt;
pub use partial_piv_lu::PartialPivLu;

mod complex;
mod eigen_solver;
mod eigen_solver_vectors;
mod jacobi_svd;
mod jacobi_svd_kernel;
mod self_adjoint_eigen_solver;
mod svd_fixed;
pub use complex::{Complex, ComplexMatrixXd};
pub use eigen_solver::EigenSolver;
pub use jacobi_svd::{JacobiSvd, SvdFactor, SvdOptions};
pub use self_adjoint_eigen_solver::SelfAdjointEigenSolver;
pub use svd_fixed::{Svd3d, Svd4d};
