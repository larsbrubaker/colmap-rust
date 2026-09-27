//! EigenSolver: eigenvalues (and optionally eigenvectors) of a general real square matrix, the
//! replacement for `Eigen::EigenSolver`. COLMAP uses it for polynomial roots
//! (math/polynomial.cc, the companion matrix, eigenvalues only) and for the 4x4 in
//! estimators/solvers/generalized_relative_pose.cc (eigenvalues and complex eigenvectors).
//! Port of colmap-sharp's `ColmapSharp/LinearAlgebra/EigenSolver.cs`; the eigenvector
//! back-substitution (`EigenSolver.Vectors.cs`) is `eigen_solver_vectors.rs`. Eigen (MPL-2.0)
//! is not ported.
//!
//! Algorithm (Golub & Van Loan, "Matrix Computations", 4th ed.):
//! 1. Householder reduction to upper Hessenberg form `H = Z^T A Z` (Algorithm 7.4.2).
//! 2. Real Schur form `T = Z^T A Z` by the shifted Francis double-shift QR step (Algorithm
//!    7.5.1) with the deflation of §7.5.1 (a subdiagonal entry is negligible when it is at
//!    most epsilon times the sum of its two diagonal neighbors) and Wilkinson's ad hoc
//!    exceptional shift after 10 and 20 iterations without deflation (Wilkinson and Reinsch,
//!    "Handbook for Automatic Computation II", 1971, procedure hqr). A deflated 2x2 block
//!    with real eigenvalues is split by a rotation (its first column rotated onto an
//!    eigenvector), so T is quasi-triangular with 1x1 real blocks and 2x2 blocks for complex
//!    pairs only.
//!
//! An eigenvalues-only solve skips Z and confines the QR sweeps to the active window
//! (LAPACK's wantt = wantz = false). That does not change a single rounding: every entry the
//! eigenvalues depend on gets the same operations in the same order (GR6P's 64 x 64 action
//! matrix is the hot caller). colmap-sharp unrolls its reflector kernels (four columns at a
//! time, and a 3-element special case); each unrolled form performs the plain loop's
//! operations in the same order per entry, so the plain loops here give the same bits.
//!
//! Semantics follow Eigen's documentation: eigenvalues come in the order of the diagonal
//! blocks of T (not sorted); a complex pair appears as `(re + i im, re - i im)` with `im > 0`
//! first; eigenvectors are the columns of a complex matrix, each normalized to unit norm. The
//! eigenvector of a real eigenvalue is real; the complex phase of a complex eigenvector (and
//! every sign) is this implementation's, like Eigen's arbitrary (docs/CPP_DIVERGENCES.md).
//! Info is NoConvergence when the QR iteration exceeds 30 iterations per eigenvalue. The input
//! is divided by its largest |entry| first and the eigenvalues multiplied back. Tier B.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_spectral_eigen.rs` and
//! `rust_only_spectral_oracle.rs`.

use super::complex::{Complex, ComplexMatrixXd};
use super::eigen_solver_vectors;
use super::householder;
use super::{dot, ComputationInfo, MatrixXd, MACHINE_EPSILON};
use crate::math::fns;

const MAX_ITERATIONS_PER_EIGENVALUE: usize = 30;

/// Eigenvalues and eigenvectors of a general real matrix via Hessenberg reduction and the
/// Francis double-shift QR algorithm. Replacement for `Eigen::EigenSolver`.
#[derive(Clone, Debug)]
pub struct EigenSolver {
    eigenvalues: Vec<Complex>,
    eigenvectors: Option<ComplexMatrixXd>,
    info: ComputationInfo,
}

impl EigenSolver {
    /// Decomposes the square matrix `a` (not modified).
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
        let nan = Complex::new(f64::NAN, f64::NAN);
        if a.as_slice().iter().any(|value| !value.is_finite()) {
            return Self {
                eigenvalues: vec![nan; n],
                // Requested eigenvectors come back NaN-filled, like JacobiSvd's U/V, so
                // `eigenvectors()` never panics on bad data (docs/CPP_DIVERGENCES.md entry 32).
                eigenvectors: compute_eigenvectors.then(|| ComplexMatrixXd::filled(n, n, nan)),
                info: ComputationInfo::InvalidInput,
            };
        }

        // Work on A / max|a_ij| so the Francis step's products cannot overflow (or
        // underflow); eigenvalues scale back, eigenvectors are unaffected.
        let mut t = a.clone();
        let mut scale = 0.0_f64;
        for &value in t.as_slice() {
            scale = scale.max(value.abs());
        }
        if scale == 0.0 {
            scale = 1.0;
        }
        for value in t.as_mut_slice() {
            *value /= scale;
        }

        // Eigenvalues only: Z is never needed, and the Schur form only has to be right on the
        // diagonal blocks, so the QR sweeps touch just the active window (LAPACK dhseqr's
        // wantt = wantz = false). Every entry inside the window gets the same operations in
        // the same order either way, so the eigenvalues are bit-identical to the full run.
        let mut z = if compute_eigenvectors {
            Some(MatrixXd::identity(n))
        } else {
            None
        };
        {
            let z_data: &mut [f64] = z.as_mut().map_or(&mut [][..], |z| z.as_mut_slice());
            reduce_to_hessenberg(t.as_mut_slice(), n, z_data);
        }
        let converged = {
            let z_data: &mut [f64] = z.as_mut().map_or(&mut [][..], |z| z.as_mut_slice());
            compute_real_schur(t.as_mut_slice(), n, z_data, compute_eigenvectors)
        };
        if !converged {
            return Self {
                eigenvalues: vec![nan; n],
                eigenvectors: compute_eigenvectors.then(|| ComplexMatrixXd::filled(n, n, nan)),
                info: ComputationInfo::NoConvergence,
            };
        }

        let t_data = t.as_slice();
        let mut eigenvalues = vec![Complex::ZERO; n];
        let mut i = 0;
        while i < n {
            if i == n - 1 || t_data[i * n + i + 1] == 0.0 {
                eigenvalues[i] = Complex::new(t_data[i * n + i], 0.0);
            } else {
                let (re, im) = complex_block_eigenvalue(t_data, n, i);
                eigenvalues[i] = Complex::new(re, im);
                eigenvalues[i + 1] = Complex::new(re, -im);
                i += 1;
            }
            i += 1;
        }

        let eigenvectors =
            z.map(|z| eigen_solver_vectors::compute_eigenvectors(&t, &z, &eigenvalues));

        for value in &mut eigenvalues {
            *value *= scale;
        }

        Self {
            eigenvalues,
            eigenvectors,
            info: ComputationInfo::Success,
        }
    }

    /// Success, InvalidInput (non-finite entry) or NoConvergence.
    pub fn info(&self) -> ComputationInfo {
        self.info
    }

    /// The eigenvalues in Schur-block order. A copy.
    pub fn eigenvalues(&self) -> Vec<Complex> {
        self.eigenvalues.clone()
    }

    /// The unit-norm eigenvectors as columns (`m[(row, column)]`). A copy.
    ///
    /// # Panics
    /// When eigenvectors were not requested. On non-finite input or no convergence they are
    /// NaN-filled.
    pub fn eigenvectors(&self) -> ComplexMatrixXd {
        self.eigenvectors
            .clone()
            .expect("Eigenvectors were not requested (compute_eigenvectors was false).")
    }
}

/// The eigenvalue `re + i im` (`im > 0`) of the complex 2x2 block at (i, i): with
/// `p = (a - d) / 2`, the eigenvalues are `d + p +- sqrt(p^2 + b c)`.
fn complex_block_eigenvalue(t: &[f64], n: usize, i: usize) -> (f64, f64) {
    let a = t[i * n + i];
    let b = t[(i + 1) * n + i];
    let c = t[i * n + i + 1];
    let d = t[(i + 1) * n + i + 1];
    let p = 0.5 * (a - d);
    let q = p * p + b * c;
    (d + p, fns::sqrt(q.abs()))
}

/// G&VL Algorithm 7.4.2: `H = Z^T A Z` upper Hessenberg on the column-major n x n `h`, Z
/// accumulated unless `z` is empty.
fn reduce_to_hessenberg(h: &mut [f64], n: usize, z: &mut [f64]) {
    if n <= 2 {
        return;
    }
    let mut x = vec![0.0; n];
    let mut work = vec![0.0; n];
    for k in 0..n - 2 {
        let length = n - k - 1;
        let v = &mut x[..length];
        v.copy_from_slice(&h[k * n + k + 1..k * n + k + 1 + length]);

        let tau = householder::make_in_place(v);
        if tau == 0.0 {
            continue;
        }

        let beta = v[0];
        v[0] = 1.0;
        apply_reflector_left(h, n, v, tau, k + 1, k, n);
        apply_reflector_right(h, n, v, tau, k + 1, 0, n, &mut work);
        if !z.is_empty() {
            apply_reflector_right(z, n, v, tau, k + 1, 0, n, &mut work);
        }

        h[k * n + k + 1] = beta;
        h[k * n + k + 2..k * n + n].fill(0.0);
    }
}

/// `(I - tau v v^T)` applied to rows `first..` of columns `first_col..col_end` of the
/// column-major n-row `m`. Each column's `v^T c` is summed left to right from `0.0`.
fn apply_reflector_left(
    m: &mut [f64],
    n: usize,
    v: &[f64],
    tau: f64,
    first: usize,
    first_col: usize,
    col_end: usize,
) {
    let length = v.len();
    for c in first_col..col_end {
        let segment = &mut m[c * n + first..c * n + first + length];
        let mut dot = 0.0;
        for i in 0..length {
            dot += v[i] * segment[i];
        }
        dot *= tau;
        for i in 0..length {
            segment[i] -= dot * v[i];
        }
    }
}

/// `(I - tau v v^T)` applied from the right to columns `first..` of rows
/// `row_start..row_end` of the column-major n-row `m`. Each row's dot product is the
/// left-to-right sum over v from `0.0`; `work` (at least `row_end - row_start` long) holds
/// the running sums.
#[allow(clippy::too_many_arguments)]
fn apply_reflector_right(
    m: &mut [f64],
    n: usize,
    v: &[f64],
    tau: f64,
    first: usize,
    row_start: usize,
    row_end: usize,
    work: &mut [f64],
) {
    if row_end <= row_start {
        return;
    }
    let rows = row_end - row_start;
    let dots = &mut work[..rows];
    dots.fill(0.0);
    for (i, &vi) in v.iter().enumerate() {
        let column = &m[(first + i) * n + row_start..(first + i) * n + row_start + rows];
        for r in 0..rows {
            dots[r] += column[r] * vi;
        }
    }
    for value in dots.iter_mut() {
        *value *= tau;
    }
    for (i, &vi) in v.iter().enumerate() {
        let column = &mut m[(first + i) * n + row_start..(first + i) * n + row_start + rows];
        for r in 0..rows {
            column[r] -= dots[r] * vi;
        }
    }
}

/// Francis QR iteration on the column-major Hessenberg matrix `t`, accumulating into `z`,
/// until `t` is quasi-triangular with standardized real 2x2 blocks split. When `full` is
/// false only the active window is updated (`z` is empty), which leaves the diagonal blocks,
/// all the eigenvalues read, exactly as in the full run. False on no convergence.
fn compute_real_schur(t: &mut [f64], n: usize, z: &mut [f64], full: bool) -> bool {
    let norm = fns::sqrt(if t.is_empty() { 0.0 } else { dot(t, t) });
    let mut hi = n as isize - 1;
    let mut iterations = 0;
    let mut total_iterations = 0;
    let mut work = vec![0.0; n];
    while hi >= 0 {
        let h = hi as usize;
        let mut lo = h;
        while lo > 0 {
            let mut s = t[(lo - 1) * n + lo - 1].abs() + t[lo * n + lo].abs();
            if s == 0.0 {
                s = norm;
            }
            if t[(lo - 1) * n + lo].abs() <= MACHINE_EPSILON * s {
                t[(lo - 1) * n + lo] = 0.0;
                break;
            }
            lo -= 1;
        }

        if lo == h {
            hi -= 1;
            iterations = 0;
        } else if lo + 1 == h {
            split_real_block(t, n, z, lo, full);
            hi -= 2;
            iterations = 0;
        } else {
            iterations += 1;
            total_iterations += 1;
            if total_iterations > MAX_ITERATIONS_PER_EIGENVALUE * n {
                return false;
            }
            francis_step(t, n, z, lo, h, iterations, full, &mut work);
        }
    }
    true
}

/// If the 2x2 block at (m, m) has real eigenvalues, rotates it to upper triangular form: the
/// rotation's first column is the eigenvector `(lambda - d, c)` of the eigenvalue
/// `lambda = d + p + sign(p) sqrt(p^2 + b c)`, the one farther from d. When `full` is false
/// only the block itself is rotated.
fn split_real_block(t: &mut [f64], n: usize, z: &mut [f64], m: usize, full: bool) {
    let c0_index = m * n;
    let c1_index = (m + 1) * n;
    let p = 0.5 * (t[c0_index + m] - t[c1_index + m + 1]);
    let q = p * p + t[c1_index + m] * t[c0_index + m + 1];
    if q < 0.0 {
        return;
    }

    let root = fns::sqrt(q);
    let shifted = if p >= 0.0 { p + root } else { p - root };
    let c0 = shifted;
    let c1 = t[c0_index + m + 1];
    let length = fns::sqrt(c0 * c0 + c1 * c1);
    if length == 0.0 {
        return;
    }

    let c = c0 / length;
    let s = c1 / length;

    // T <- G^T T G, Z <- Z G with G = [c -s; s c].
    let col_end = if full { n } else { m + 2 };
    for j in m..col_end {
        let x = t[j * n + m];
        let y = t[j * n + m + 1];
        t[j * n + m] = c * x + s * y;
        t[j * n + m + 1] = -s * x + c * y;
    }

    for i in (if full { 0 } else { m })..=m + 1 {
        let x = t[c0_index + i];
        let y = t[c1_index + i];
        t[c0_index + i] = c * x + s * y;
        t[c1_index + i] = -s * x + c * y;
    }

    if !z.is_empty() {
        for i in 0..n {
            let x = z[c0_index + i];
            let y = z[c1_index + i];
            z[c0_index + i] = c * x + s * y;
            z[c1_index + i] = -s * x + c * y;
        }
    }

    t[c0_index + m + 1] = 0.0;
}

/// G&VL Algorithm 7.5.1 on the unreduced window `lo..=hi` (at least 3 x 3): one implicit
/// double-shift QR step by bulge chasing with 3-element Householder reflectors. When `full`
/// is false the reflectors are applied inside the window only.
#[allow(clippy::too_many_arguments)]
fn francis_step(
    t: &mut [f64],
    n: usize,
    z: &mut [f64],
    lo: usize,
    hi: usize,
    iterations: usize,
    full: bool,
    work: &mut [f64],
) {
    // t(r, c) is t[c * n + r].
    let (shift_sum, shift_product) = if iterations == 10 || iterations == 20 {
        // Wilkinson's exceptional shift breaks cycles of the standard shift.
        let s = t[(hi - 1) * n + hi].abs() + t[(hi - 2) * n + hi - 1].abs();
        (1.5 * s, s * s)
    } else {
        (
            t[(hi - 1) * n + hi - 1] + t[hi * n + hi],
            t[(hi - 1) * n + hi - 1] * t[hi * n + hi] - t[hi * n + hi - 1] * t[(hi - 1) * n + hi],
        )
    };

    let t_lo_lo = t[lo * n + lo];
    let mut x = t_lo_lo * t_lo_lo + t[(lo + 1) * n + lo] * t[lo * n + lo + 1] - shift_sum * t_lo_lo
        + shift_product;
    let mut y = t[lo * n + lo + 1] * (t_lo_lo + t[(lo + 1) * n + lo + 1] - shift_sum);
    let mut w = t[lo * n + lo + 1] * t[(lo + 1) * n + lo + 2];
    let col_end = if full { n } else { hi + 1 };
    let row_start = if full { 0 } else { lo };
    let mut buffer = [0.0; 3];
    for k in lo..hi {
        let length = if k < hi - 1 { 3 } else { 2 };
        let reflector = &mut buffer[..length];
        reflector[0] = x;
        reflector[1] = y;
        if length == 3 {
            reflector[2] = w;
        }

        let tau = householder::make_in_place(reflector);
        if tau != 0.0 {
            let beta = reflector[0];
            reflector[0] = 1.0;
            let first_col = lo.max(k.saturating_sub(1));
            let row_end = (k + 3).min(hi) + 1;
            apply_reflector_left(t, n, reflector, tau, k, first_col, col_end);
            apply_reflector_right(t, n, reflector, tau, k, row_start, row_end, work);
            if !z.is_empty() {
                apply_reflector_right(z, n, reflector, tau, k, 0, n, work);
            }

            if k > lo {
                t[(k - 1) * n + k] = beta;
                t[(k - 1) * n + k + 1] = 0.0;
                if length == 3 {
                    t[(k - 1) * n + k + 2] = 0.0;
                }
            }
        }

        if k < hi - 1 {
            x = t[k * n + k + 1];
            y = t[k * n + k + 2];
            if k < hi - 2 {
                w = t[k * n + k + 3];
            }
        }
    }
}
