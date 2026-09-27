//! Port of COLMAP's `colmap/math/matrix.h`: [`decompose_matrix_rq`], the RQ decomposition of a
//! square matrix through a Householder QR of its flipped transpose. Port of colmap-sharp's
//! `Mathematics/MatrixUtils.cs`. Built on [`crate::linalg::HouseholderQr`]; used by
//! `geometry::pose::decompose_projection_matrix`. Tests: `tests/math/matrix.rs`
//! (matrix_test.cc).
//!
//! Tier B: the result goes through a Householder QR (Eigen's is not ported), so it matches
//! COLMAP within the test's 1e-6 tolerance, not bit for bit. COLMAP's template takes any
//! square Eigen matrix type; here there is a [`MatrixXd`] version and a [`Matrix3d`] wrapper
//! (the only fixed shape COLMAP instantiates outside tests).

use crate::linalg::{HouseholderQr, Matrix3d, MatrixXd};

/// Port of `colmap::DecomposeMatrixRQ`: `A = R * Q` with `R` upper triangular and `Q`
/// orthogonal, made unique by requiring `det(Q) > 0`. Returns `(R, Q)`.
///
/// # Panics
/// When `a` is not square (a compile-time shape in COLMAP).
pub fn decompose_matrix_rq(a: &MatrixXd) -> (MatrixXd, MatrixXd) {
    assert!(
        a.rows() == a.cols(),
        "DecomposeMatrixRQ needs a square matrix, got {}x{}.",
        a.rows(),
        a.cols()
    );

    // A.transpose().rowwise().reverse()
    let a_flipud_transpose = a.transpose().reverse_cols();

    let qr = HouseholderQr::new(&a_flipud_transpose);
    let q0 = qr.householder_q();
    // COLMAP reads matrixQR() whole, the reflector essentials below the diagonal included,
    // and relies on the zeroing loop below to clear them.
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
    let mut q = q0.transpose().reverse_rows();

    // Make the decomposition unique by requiring that det(Q) > 0.
    if q.determinant() < 0.0 {
        for c in 0..q.cols() {
            q[(1, c)] *= -1.0;
        }
        for row in 0..r.rows() {
            r[(row, 1)] *= -1.0;
        }
    }
    (r, q)
}

/// [`decompose_matrix_rq`] for a [`Matrix3d`] (the shape `DecomposeProjectionMatrix` uses).
pub fn decompose_matrix_rq_3d(a: &Matrix3d) -> (Matrix3d, Matrix3d) {
    let (r, q) = decompose_matrix_rq(&MatrixXd::from(*a));
    (r.to_matrix3d(), q.to_matrix3d())
}
