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
//! Dynamic-size matrices, decompositions (SVD, eigen, QR, LU, Cholesky) and sparse matrices are
//! not here. Tests: `colmap-rust/tests/linalg.rs`.

#[macro_use]
mod matrix_macros;
mod aligned_box;
mod angle_axis;
mod matrix3;
mod matrix4;
mod matrix6;
mod matrix_small;
mod quaternion;
mod vector;

pub use aligned_box::AlignedBox3d;
pub use angle_axis::AngleAxisd;
pub use matrix3::Matrix3d;
pub use matrix4::{Matrix3x4d, Matrix4d, Matrix4x3d};
pub use matrix6::Matrix6d;
pub use matrix_small::{Matrix2d, Matrix2x3d, Matrix3x2d};
pub use quaternion::Quaterniond;
pub use vector::{Vector2d, Vector3d, Vector3f, Vector3ub, Vector4d};

use crate::math::fns;

/// Eigen's `NumTraits<double>::dummy_precision()`, the default precision of `isApprox`.
pub const DUMMY_PRECISION: f64 = 1e-12;

/// `std::numeric_limits<double>::epsilon()`.
pub const MACHINE_EPSILON: f64 = f64::EPSILON;

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
