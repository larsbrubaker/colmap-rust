//! HouseholderQr: `A = Q R` by Householder reflections, the replacement for
//! `Eigen::HouseholderQR` / `matrix.householderQr()`. COLMAP uses `householderQ()` for an
//! orthonormal null-space basis (the 8-point fundamental and essential solvers,
//! geometry/pose.cc `GravityAlignedRotation`) and math/matrix.h's `DecomposeMatrixRQ` builds
//! on it. Port of colmap-sharp's `ColmapSharp/LinearAlgebra/HouseholderQR.cs`, written from
//! Golub & Van Loan / colmap-sharp; Eigen (MPL-2.0) is not ported.
//!
//! Algorithm: unblocked Householder QR, Golub & Van Loan, "Matrix Computations", 4th ed.,
//! Algorithm 5.2.1, with the reflector convention in [`super::householder`] (LAPACK
//! `dgeqr2`'s). Any shape is accepted; min(m, n) reflectors are formed, the last one of a
//! square matrix being the identity (tau = 0). Tier B: Eigen switches to a blocked update for
//! large matrices, so results agree within rounding; the signs of Q's columns and R's rows
//! follow the documented convention and are pinned against numpy (LAPACK).
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_decomposition.rs` and
//! `rust_only_decomposition_oracle.rs`.

use super::householder;
use super::{MatrixXd, VectorXd};

/// Householder QR, `A = Q R`. Replacement for `Eigen::HouseholderQR<MatrixXd>`.
#[derive(Clone, Debug)]
pub struct HouseholderQr {
    qr: MatrixXd,
    h_coeffs: Vec<f64>,
}

impl HouseholderQr {
    /// Factorizes a matrix of any shape. The input is not modified.
    pub fn new(a: &MatrixXd) -> Self {
        let mut qr = a.clone();
        let mut h_coeffs = vec![0.0; a.rows().min(a.cols())];
        householder::factor_in_place(qr.as_mut_slice(), a.rows(), a.cols(), &mut h_coeffs);
        Self { qr, h_coeffs }
    }

    /// The packed factor (Eigen's `matrixQR()`): R on and above the diagonal, the essential
    /// parts of the reflectors below it.
    pub fn matrix_qr(&self) -> &MatrixXd {
        &self.qr
    }

    /// The reflector coefficients tau (Eigen's `hCoeffs()`).
    pub fn h_coeffs(&self) -> VectorXd {
        VectorXd::from_slice(&self.h_coeffs)
    }

    /// The upper-triangular (trapezoidal) factor R, m x n.
    pub fn matrix_r(&self) -> MatrixXd {
        householder::upper_part(&self.qr)
    }

    /// The full orthogonal factor Q (m x m), Eigen's `householderQ()` evaluated.
    pub fn householder_q(&self) -> MatrixXd {
        householder::accumulate_q(&self.qr, &self.h_coeffs)
    }

    /// Least-squares solution of `A x = b` for a matrix with at least as many rows as columns
    /// and full column rank (the exact solution when A is square).
    ///
    /// # Panics
    /// When A has fewer rows than columns or `b` has the wrong length.
    pub fn solve(&self, b: &VectorXd) -> VectorXd {
        let (m, n) = (self.qr.rows(), self.qr.cols());
        assert!(m >= n, "HouseholderQr::solve needs rows >= cols.");
        assert!(
            b.len() == m,
            "Right-hand side has {} rows, expected {m}.",
            b.len()
        );

        let mut y = b.clone();
        householder::apply_q_transpose(self.qr.as_slice(), m, &self.h_coeffs, y.as_mut_slice());
        householder::back_substitute(self.qr.as_slice(), m, n, y.as_mut_slice());
        y.head(n)
    }
}
