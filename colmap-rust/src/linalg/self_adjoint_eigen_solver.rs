//! SelfAdjointEigenSolver: eigen-decomposition `A = V diag(lambda) V^T` of a real symmetric
//! matrix, the replacement for `Eigen::SelfAdjointEigenSolver`. COLMAP uses it on the 4x4
//! normal matrix in geometry/triangulation.cc (`eigenvectors().col(0)`, the eigenvector of the
//! smallest eigenvalue). Port of colmap-sharp's
//! `ColmapSharp/LinearAlgebra/SelfAdjointEigenSolver.cs`; Eigen (MPL-2.0) is not ported.
//!
//! Algorithm: cyclic Jacobi, Golub & Van Loan, "Matrix Computations", 4th ed., Algorithm
//! 8.5.3, with the symmetric Schur rotation of Algorithm 8.5.1. Rotations are accumulated into
//! V, so the eigenvectors are orthonormal to working precision. Sweeps run until the
//! off-diagonal Frobenius norm is at most epsilon times the matrix's Frobenius norm (or a sweep
//! rotates nothing), on the input divided by its largest |entry| so the norms cannot under- or
//! overflow; Info is NoConvergence if the sweep cap is reached first. Jacobi is chosen over
//! tridiagonal QL for its high relative accuracy on the small matrices COLMAP decomposes.
//!
//! Semantics follow Eigen's documentation: only the lower triangle of the input is read;
//! eigenvalues are returned in increasing order (ties keep their diagonal order); the
//! eigenvector columns are normalized. Eigenvector signs are arbitrary, as in Eigen
//! (docs/CPP_DIVERGENCES.md). Tier B.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_spectral_eigen.rs` and
//! `rust_only_spectral_oracle.rs`.

use super::jacobi_svd_kernel::swap_columns;
use super::{ComputationInfo, MatrixXd, VectorXd, MACHINE_EPSILON};
use crate::math::fns;

const MAX_SWEEPS: usize = 100;

/// Eigenvalues and eigenvectors of a real symmetric matrix by cyclic Jacobi. Replacement for
/// `Eigen::SelfAdjointEigenSolver`.
#[derive(Clone, Debug)]
pub struct SelfAdjointEigenSolver {
    eigenvalues: Vec<f64>,
    eigenvectors: Option<MatrixXd>,
    info: ComputationInfo,
}

impl SelfAdjointEigenSolver {
    /// Decomposes the symmetric matrix whose lower triangle is `a`'s (not modified).
    ///
    /// # Panics
    /// When `a` is not square (programmer error, colmap-sharp's ArgumentException).
    pub fn new(a: &MatrixXd, compute_eigenvectors: bool) -> Self {
        assert!(
            a.rows() == a.cols(),
            "Matrix must be square, got {}x{}.",
            a.rows(),
            a.cols()
        );

        let n = a.rows();
        let mut w = MatrixXd::zeros(n, n);
        let mut finite = true;
        for j in 0..n {
            for i in j..n {
                w[(i, j)] = a[(i, j)];
                w[(j, i)] = a[(i, j)];
                finite &= a[(i, j)].is_finite();
            }
        }

        let mut eigenvalues = vec![0.0; n];
        let mut v = MatrixXd::identity(n);
        if !finite {
            eigenvalues.fill(f64::NAN);
            return Self {
                eigenvalues,
                eigenvectors: None,
                info: ComputationInfo::InvalidInput,
            };
        }

        // Work on A / max|a_ij| so the squared norms below neither underflow (entries near
        // 1e-170) nor overflow (near 1e160); the eigenvalues are scaled back at the end.
        let mut scale = 0.0_f64;
        for &value in w.as_slice() {
            scale = scale.max(value.abs());
        }
        if scale == 0.0 {
            scale = 1.0;
        }
        for value in w.as_mut_slice() {
            *value /= scale;
        }

        let mut info = ComputationInfo::NoConvergence;
        let tolerance = MACHINE_EPSILON * w.norm();
        for _ in 0..MAX_SWEEPS {
            if off_diagonal_norm(&w) <= tolerance {
                info = ComputationInfo::Success;
                break;
            }

            let mut rotated = false;
            for p in 0..n.saturating_sub(1) {
                for q in p + 1..n {
                    if w[(p, q)] != 0.0 {
                        rotated = true;
                        rotate(&mut w, &mut v, p, q);
                    }
                }
            }

            if !rotated {
                info = ComputationInfo::Success;
                break;
            }
        }

        for (i, value) in eigenvalues.iter_mut().enumerate() {
            *value = w[(i, i)] * scale;
        }

        // Stable insertion sort into increasing order, moving the eigenvector columns.
        for i in 1..n {
            let mut k = i;
            while k > 0 && eigenvalues[k] < eigenvalues[k - 1] {
                eigenvalues.swap(k, k - 1);
                swap_columns(v.as_mut_slice(), n, k, k - 1);
                k -= 1;
            }
        }

        Self {
            eigenvalues,
            eigenvectors: if compute_eigenvectors { Some(v) } else { None },
            info,
        }
    }

    /// Success, NoConvergence, or InvalidInput when the lower triangle had a non-finite entry.
    pub fn info(&self) -> ComputationInfo {
        self.info
    }

    /// The eigenvalues in increasing order. A copy.
    pub fn eigenvalues(&self) -> VectorXd {
        VectorXd::from_slice(&self.eigenvalues)
    }

    /// The normalized eigenvectors as columns, in eigenvalue order. A copy.
    ///
    /// # Panics
    /// When eigenvectors were not requested (or the input was non-finite).
    pub fn eigenvectors(&self) -> MatrixXd {
        self.eigenvectors
            .clone()
            .expect("Eigenvectors were not requested.")
    }
}

/// Frobenius norm of the off-diagonal part, column by column, seeded with `0.0` like
/// colmap-sharp (the terms are squares, so the seed does not change the result).
fn off_diagonal_norm(w: &MatrixXd) -> f64 {
    let mut sum = 0.0;
    for j in 0..w.cols() {
        for i in 0..w.rows() {
            if i != j {
                sum += w[(i, j)] * w[(i, j)];
            }
        }
    }
    fns::sqrt(sum)
}

/// G&VL Algorithm 8.5.1: the rotation `J = [c s; -s c]` in the (p, q) plane with
/// `(J^T W J)(p, q) = 0`, applied as `W <- J^T W J` and `V <- V J`.
fn rotate(w: &mut MatrixXd, v: &mut MatrixXd, p: usize, q: usize) {
    let tau = (w[(q, q)] - w[(p, p)]) / (2.0 * w[(p, q)]);
    let t = (if tau >= 0.0 { 1.0 } else { -1.0 }) / (tau.abs() + fns::sqrt(1.0 + tau * tau));
    let c = 1.0 / fns::sqrt(1.0 + t * t);
    let s = t * c;
    let n = w.rows();
    for k in 0..n {
        let x = w[(k, p)];
        let y = w[(k, q)];
        w[(k, p)] = c * x - s * y;
        w[(k, q)] = s * x + c * y;
    }

    for k in 0..n {
        let x = w[(p, k)];
        let y = w[(q, k)];
        w[(p, k)] = c * x - s * y;
        w[(q, k)] = s * x + c * y;
    }

    w[(p, q)] = 0.0;
    w[(q, p)] = 0.0;
    for k in 0..n {
        let x = v[(k, p)];
        let y = v[(k, q)];
        v[(k, p)] = c * x - s * y;
        v[(k, q)] = s * x + c * y;
    }
}
