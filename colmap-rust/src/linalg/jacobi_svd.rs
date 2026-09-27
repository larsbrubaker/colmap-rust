//! JacobiSvd: singular value decomposition `A = U diag(s) V^T` of a matrix of any shape, the
//! replacement for `Eigen::JacobiSVD<MatrixXd>` and the fixed/partly-dynamic shapes COLMAP
//! uses (Dynamic x 9 and x 6 for the DLT systems, 6x3..6x5 and 12x12 in EPnP, the 4 x N
//! quaternion average in geometry/pose.cc). 3x3 and 4x4 callers in hot paths use the
//! allocation-free [`Svd3d`](super::Svd3d) / [`Svd4d`](super::Svd4d) (`svd_fixed.rs`)
//! instead; both run `jacobi_svd_kernel.rs`. Port of colmap-sharp's
//! `ColmapSharp/LinearAlgebra/JacobiSVD.cs`; Eigen (MPL-2.0) is not ported.
//!
//! Rectangular input is reduced first (Golub & Van Loan, "Matrix Computations", 4th ed.,
//! §8.6.3, the R-SVD idea): for rows > cols, `A P = Q R` with [`ColPivHouseholderQr`], the
//! kernel decomposes the square top of `R = Ur S Vr^T`, `U = Q blockdiag(Ur, I)` and
//! `V = P Vr`, so the full U falls out of Q without any basis completion. The column pivoting
//! is Eigen's documented default preconditioner (ColPivHouseholderQRPreconditioner) and it
//! matters: it makes R graded (decreasing diagonal), which two-sided Jacobi resolves to high
//! relative accuracy (Demmel and Veselić 1992, cited in `jacobi_svd_kernel.rs`). Unpivoted QR
//! lost the null vector of badly column-scaled DLT systems (homography_matrix_test's
//! NumericalStability, pixel coordinates of 1e6 next to a column of ones). Q is never formed:
//! the stored reflectors are applied to just the requested columns of `blockdiag(Ur, I)`, so a
//! thin U of an N x 3 system costs O(N) memory, not O(N^2). rows < cols decomposes `A^T` and
//! swaps the factors.
//!
//! Semantics follow Eigen's documentation: singular values non-negative and decreasing; U and
//! V computed only when requested, thin (`min(rows, cols)` columns) or full; `rank()` and
//! `solve()` use the threshold `max(1, diagSize) * epsilon` relative to the largest singular
//! value; `solve()` returns the minimum-norm least-squares solution with the singular values
//! at or below the threshold treated as zero. Singular vector signs are arbitrary (see
//! `jacobi_svd_kernel.rs`, docs/CPP_DIVERGENCES.md). Tier B.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_spectral_svd.rs` and
//! `rust_only_spectral_oracle.rs`.

use super::householder;
use super::jacobi_svd_kernel;
use super::{ColPivHouseholderQr, ComputationInfo, MatrixXd, VectorXd};

/// Which singular vectors [`JacobiSvd`] computes (Eigen's `DecompositionOptions`,
/// colmap-sharp's `SvdOptions` flags). `Default` is singular values only.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SvdOptions {
    /// The first `min(rows, cols)` columns of U (`ComputeThinU`).
    pub thin_u: bool,
    /// All rows x rows of U (`ComputeFullU`).
    pub full_u: bool,
    /// The first `min(rows, cols)` columns of V (`ComputeThinV`).
    pub thin_v: bool,
    /// All cols x cols of V (`ComputeFullV`).
    pub full_v: bool,
}

impl SvdOptions {
    /// Singular values only.
    pub const NONE: SvdOptions = SvdOptions {
        thin_u: false,
        full_u: false,
        thin_v: false,
        full_v: false,
    };

    /// `ComputeThinU | ComputeThinV`.
    pub const THIN_UV: SvdOptions = SvdOptions {
        thin_u: true,
        full_u: false,
        thin_v: true,
        full_v: false,
    };

    /// `ComputeFullU | ComputeFullV`.
    pub const FULL_UV: SvdOptions = SvdOptions {
        thin_u: false,
        full_u: true,
        thin_v: false,
        full_v: true,
    };

    /// The options with the roles of U and V exchanged (for `A^T`).
    fn swap_uv(self) -> Self {
        SvdOptions {
            thin_u: self.thin_v,
            full_u: self.full_v,
            thin_v: self.thin_u,
            full_v: self.full_u,
        }
    }
}

/// Singular value decomposition `A = U diag(s) V^T` by two-sided Jacobi. Replacement for
/// `Eigen::JacobiSVD`.
#[derive(Clone, Debug)]
pub struct JacobiSvd {
    u: Option<MatrixXd>,
    v: Option<MatrixXd>,
    singular_values: Vec<f64>,
    rows: usize,
    cols: usize,
    info: ComputationInfo,
    sweeps: usize,
}

impl JacobiSvd {
    /// Decomposes `a` (not modified).
    ///
    /// # Panics
    /// When both the thin and the full form of one factor are requested (programmer error,
    /// colmap-sharp's ArgumentException).
    pub fn new(a: &MatrixXd, options: SvdOptions) -> Self {
        assert!(
            !(options.thin_u && options.full_u || options.thin_v && options.full_v),
            "Request either the thin or the full factor, not both."
        );

        let rows = a.rows();
        let cols = a.cols();
        if rows < cols {
            let transposed = JacobiSvd::new(&a.transpose(), options.swap_uv());
            return JacobiSvd {
                u: transposed.v,
                v: transposed.u,
                singular_values: transposed.singular_values,
                rows,
                cols,
                info: transposed.info,
                sweeps: transposed.sweeps,
            };
        }

        let n = cols;
        let m = rows;
        let want_u = options.thin_u || options.full_u;
        let want_v = options.thin_v || options.full_v;
        let mut singular_values = vec![0.0; n];

        let qr = if m > n {
            Some(ColPivHouseholderQr::new(a))
        } else {
            None
        };
        let mut square = match &qr {
            Some(qr) => qr.matrix_r().top_rows(n),
            None => a.clone(),
        };

        let mut ur = if want_u {
            Some(MatrixXd::zeros(n, n))
        } else {
            None
        };
        let mut v = if want_v {
            Some(MatrixXd::zeros(n, n))
        } else {
            None
        };
        let (info, sweeps) = jacobi_svd_kernel::decompose(
            square.as_mut_slice(),
            n,
            ur.as_mut().map_or(&mut [][..], |m| m.as_mut_slice()),
            v.as_mut().map_or(&mut [][..], |m| m.as_mut_slice()),
            &mut singular_values,
        );

        if let (Some(vr), Some(qr)) = (&v, &qr) {
            // V = P Vr: row j of Vr belongs to column cols_permutation_indices()[j] of A.
            let permutation = qr.cols_permutation_indices();
            let mut permuted = MatrixXd::zeros(n, n);
            for j in 0..n {
                for c in 0..n {
                    permuted[(permutation[j], c)] = vr[(j, c)];
                }
            }
            v = Some(permuted);
        }

        let u = ur.map(|ur| match &qr {
            None => ur,
            Some(qr) => {
                // U = Q blockdiag(Ur, I). Column c is Q applied to [Ur(:, c); 0] for c < n
                // and to the unit vector e_c after that, so the reflectors are applied to
                // just the requested columns and the thin U never forms the m x m Q.
                let u_cols = if options.full_u { m } else { n };
                let packed = qr.matrix_qr();
                let tau = qr.h_coeffs();
                let mut u = MatrixXd::zeros(m, u_cols);
                for c in 0..u_cols {
                    let column = u.column_mut(c);
                    if c < n {
                        column[..n].copy_from_slice(ur.column(c));
                    } else {
                        column[c] = 1.0;
                    }
                    householder::apply_q(packed.as_slice(), m, tau.as_slice(), column);
                }
                u
            }
        });

        JacobiSvd {
            u,
            v,
            singular_values,
            rows,
            cols,
            info,
            sweeps,
        }
    }

    /// Rows of the decomposed matrix.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// Columns of the decomposed matrix.
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Jacobi sweeps the decomposition ran, the last of which rotated nothing (0 for a zero
    /// or non-finite matrix). A diagnostic for convergence tests.
    pub fn sweeps(&self) -> usize {
        self.sweeps
    }

    /// Success, NoConvergence, or InvalidInput when the matrix had a non-finite entry.
    pub fn info(&self) -> ComputationInfo {
        self.info
    }

    /// The `min(rows, cols)` singular values, non-negative and decreasing. A copy.
    pub fn singular_values(&self) -> VectorXd {
        VectorXd::from_slice(&self.singular_values)
    }

    /// U (rows x rows when full, rows x `min(rows, cols)` when thin). A copy.
    ///
    /// # Panics
    /// When U was not requested (programmer error).
    pub fn matrix_u(&self) -> MatrixXd {
        self.u.clone().expect("U was not requested (SvdOptions).")
    }

    /// V (cols x cols when full, cols x `min(rows, cols)` when thin). A copy.
    ///
    /// # Panics
    /// When V was not requested (programmer error).
    pub fn matrix_v(&self) -> MatrixXd {
        self.v.clone().expect("V was not requested (SvdOptions).")
    }

    /// Numerical rank: the number of singular values strictly greater than
    /// `max(1, min(rows, cols)) * epsilon *` the largest singular value.
    pub fn rank(&self) -> usize {
        jacobi_svd_kernel::rank(&self.singular_values)
    }

    /// Minimum-norm least-squares solution of `A x = b`; singular values at or below the rank
    /// threshold are treated as zero. Needs U and V (thin or full).
    ///
    /// # Panics
    /// When U or V was not requested, or `b` has the wrong length (programmer errors).
    pub fn solve(&self, b: &VectorXd) -> VectorXd {
        let (Some(u), Some(v)) = (&self.u, &self.v) else {
            panic!("Solve needs U and V (SvdOptions).");
        };
        assert!(
            b.len() == self.rows,
            "Right-hand side has {} rows, expected {}.",
            b.len(),
            self.rows
        );

        let rank = self.rank();
        let mut x = VectorXd::zeros(self.cols);
        for i in 0..rank {
            // Seeded with 0.0 like colmap-sharp's loop.
            let mut coefficient = 0.0;
            for r in 0..self.rows {
                coefficient += u[(r, i)] * b[r];
            }
            coefficient /= self.singular_values[i];
            for c in 0..self.cols {
                x[c] += v[(c, i)] * coefficient;
            }
        }
        x
    }
}
