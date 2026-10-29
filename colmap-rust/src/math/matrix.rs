//! Port of COLMAP's `src/colmap/math/matrix.h`: `DecomposeMatrixRQ`, the RQ decomposition
//! of a square matrix through a Householder QR of its flipped transpose (used by
//! `geometry/pose.cc`'s `DecomposeProjectionMatrix` with a 3x3 matrix). Mirrors colmap-sharp's
//! `Mathematics/MatrixUtils.cs`. Builds on [`crate::linalg::HouseholderQr`]; the neighbors
//! are `polynomial.rs` and `random_eigen.rs`. Tests: `colmap-rust/tests/math/matrix.rs`
//! (`matrix_test.cc` 1:1).
//!
//! Tier B, a deliberate divergence: COLMAP factors the flipped transpose with
//! `Eigen::HouseholderQR`; this uses [`HouseholderQr`] (colmap-sharp's port, not Eigen's
//! blocked implementation, because Eigen is MPL-2.0 and not ported), so R and Q can differ
//! from COLMAP's in the last bits and match within the test's 1e-6 tolerance, not bit for bit.
//! The algorithm around the QR (flips, zeroing loop, `det(Q) > 0` normalization) is COLMAP's.
//! COLMAP's template takes any square Eigen matrix type; here there are three entry points,
//! [`decompose_matrix_rq`] ([`MatrixXd`], `Err` unless square), [`decompose_matrix_rq_3d`]
//! ([`Matrix3d`]) and [`decompose_matrix_rq_4d`] ([`Matrix4d`]). The `det(Q) > 0` test uses
//! the determinant of the caller's type, as Eigen's template does: Laplace/cofactor expansion
//! for the fixed sizes, partial-pivot LU for `MatrixXd`. `decompose_matrix_rq_nominal`
//! (COLMAP's 1e-6 tolerances, upper-triangular and unitary at Eigen's default precision)
//! passes 1:1, and `rust_only_decompose_matrix_rq_dynamic_and_3d`
//! (`tests/math/rust_only_polynomial_matrix.rs`) covers 2x2/3x3/5x5 at 1e-9 and pins the 3x3
//! entry point to the dynamic one bit for bit. Same as colmap-sharp (its
//! `Mathematics/MatrixUtils.cs` header).

use crate::check_eq;
use crate::linalg::{HouseholderQr, Matrix3d, Matrix4d, MatrixXd};

/// Port of `colmap::DecomposeMatrixRQ` for a dynamic square matrix: `A = R * Q` with `R`
/// upper triangular and `Q` orthogonal, made unique by requiring `det(Q) > 0`. Returns
/// `(R, Q)`; `Err` unless `A` is square.
pub fn decompose_matrix_rq(a: &MatrixXd) -> crate::Result<(MatrixXd, MatrixXd)> {
    check_eq!(a.rows(), a.cols());
    let (mut r, mut q) = rq_without_sign_fix(a);
    if q.determinant() < 0.0 {
        flip_sign(&mut r, &mut q);
    }
    Ok((r, q))
}

/// Port of `colmap::DecomposeMatrixRQ` for `Eigen::Matrix3d`. Returns `(R, Q)`.
pub fn decompose_matrix_rq_3d(a: &Matrix3d) -> (Matrix3d, Matrix3d) {
    let (mut r, mut q) = rq_without_sign_fix(&MatrixXd::from(*a));
    if q.to_matrix3d().determinant() < 0.0 {
        flip_sign(&mut r, &mut q);
    }
    (r.to_matrix3d(), q.to_matrix3d())
}

/// Port of `colmap::DecomposeMatrixRQ` for `Eigen::Matrix4d`. Returns `(R, Q)`.
pub fn decompose_matrix_rq_4d(a: &Matrix4d) -> (Matrix4d, Matrix4d) {
    let (mut r, mut q) = rq_without_sign_fix(&MatrixXd::from(*a));
    if q.to_matrix4d().determinant() < 0.0 {
        flip_sign(&mut r, &mut q);
    }
    (r.to_matrix4d(), q.to_matrix4d())
}

/// Everything in `DecomposeMatrixRQ` before the `det(Q) > 0` normalization.
fn rq_without_sign_fix(a: &MatrixXd) -> (MatrixXd, MatrixXd) {
    // A.transpose().rowwise().reverse(): reverses the order of the columns.
    let a_flipud_transpose = a.transpose().reverse_cols();

    let qr = HouseholderQr::new(&a_flipud_transpose);
    let q0 = qr.householder_q();
    // COLMAP reads matrixQR() whole, Householder essentials below the diagonal included, and
    // relies on the zeroing loop below to clear them.
    let r0 = qr.matrix_qr();

    // R0.transpose().colwise().reverse(), then rowwise().reverse().
    let mut r = r0.transpose().reverse_rows().reverse_cols();
    let (rows, cols) = (r.rows(), r.cols());
    for i in 0..rows {
        let mut j = 0;
        while j < cols && (cols - j) > (rows - i) {
            r[(i, j)] = 0.0;
            j += 1;
        }
    }

    // Q0.transpose().colwise().reverse()
    let q = q0.transpose().reverse_rows();
    (r, q)
}

/// `Q->row(1) *= -1.0; R->col(1) *= -1.0;`
fn flip_sign(r: &mut MatrixXd, q: &mut MatrixXd) {
    for c in 0..q.cols() {
        q[(1, c)] *= -1.0;
    }
    for row in 0..r.rows() {
        r[(row, 1)] *= -1.0;
    }
}
