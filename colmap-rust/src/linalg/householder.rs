//! Householder reflector helpers: the elementary reflector `H = I - tau * v * v^T`
//! (`v[0] = 1`) shared by [`HouseholderQr`](super::HouseholderQr) and
//! [`ColPivHouseholderQr`](super::ColPivHouseholderQr), plus the routines that apply a
//! sequence of them and accumulate Q. Port of colmap-sharp's
//! `ColmapSharp/LinearAlgebra/Householder.cs`, written from Golub & Van Loan / colmap-sharp;
//! Eigen (MPL-2.0) is not ported.
//!
//! Golub & Van Loan, "Matrix Computations", 4th ed., §5.1 (Algorithm 5.1.1) and §5.2
//! (factored-form representation), with the sign convention of LAPACK's `dlarfg` (only its
//! published convention, no code): for `x = (alpha, tail)`,
//! `beta = -sign(alpha) * ||x||`, `tau = (beta - alpha) / beta`,
//! `v = (1, tail / (alpha - beta))`, so `H x = (beta, 0, ..., 0)`. A zero tail gives
//! `tau = 0` (`H = I`) and `beta = alpha`. numpy's qr (LAPACK `dgeqrf`) gives the same signs,
//! which the decomposition oracle test pins.
//!
//! These are free functions over column-major slices, so the minimal solvers can run them
//! allocation-free. Arithmetic: the tail's squared norm and every `v^T c` are left-to-right
//! sums seeded with the first term; no FMA.
//!
//! Tests: `colmap-rust/tests/linalg/rust_only_decomposition.rs` and
//! `rust_only_decomposition_oracle.rs`.

use super::{dot, MatrixXd};
use crate::math::fns;

/// Turns `x = (alpha, tail)` into a reflector in place: `x[0]` becomes beta, the tail becomes
/// the essential part of v. Returns tau.
pub fn make_in_place(x: &mut [f64]) -> f64 {
    let alpha = x[0];
    let tail_squared_norm = dot(&x[1..], &x[1..]);
    if tail_squared_norm == 0.0 {
        return 0.0;
    }

    let norm = fns::sqrt(alpha * alpha + tail_squared_norm);
    let beta = if alpha >= 0.0 { -norm } else { norm };
    let scale = alpha - beta;
    for value in &mut x[1..] {
        *value /= scale;
    }

    x[0] = beta;
    (beta - alpha) / beta
}

/// Applies `H = I - tau * v v^T` (`v = (1, essential)`) from the left to one column segment
/// `c` of the same length as v.
pub fn apply_left(essential: &[f64], tau: f64, c: &mut [f64]) {
    if tau == 0.0 {
        return;
    }

    let mut w = c[0];
    for (i, &e) in essential.iter().enumerate() {
        w += e * c[i + 1];
    }

    let tw = tau * w;
    c[0] -= tw;
    for (i, &e) in essential.iter().enumerate() {
        c[i + 1] -= e * tw;
    }
}

/// `Q = H_0 H_1 ... H_{k-1}` (m x m) from packed reflectors: reflector j has its essential
/// part in rows j+1.. of column j of `packed` and acts on rows j..m-1. Built backwards from
/// the identity, so each `H_j` only touches rows and columns j..m-1.
pub fn accumulate_q(packed: &MatrixXd, tau: &[f64]) -> MatrixXd {
    let m = packed.rows();
    let mut q = MatrixXd::identity(m);
    for j in (0..tau.len()).rev() {
        let essential = &packed.column(j)[j + 1..];
        for c in j..m {
            apply_left(essential, tau[j], &mut q.column_mut(c)[j..]);
        }
    }

    q
}

/// Unblocked Householder QR of a column-major m x n slice in place (HouseholderQr's
/// factorization, the same operations in the same order): afterwards R is on and above the
/// diagonal and reflector j's essential part below it; `tau` (length min(m, n)) receives the
/// coefficients. Allocation-free.
pub fn factor_in_place(a: &mut [f64], m: usize, n: usize, tau: &mut [f64]) {
    let k = m.min(n);
    for j in 0..k {
        // Column j and the columns after it, borrowed separately.
        let (head, rest) = a.split_at_mut((j + 1) * m);
        let column = &mut head[j * m + j..];
        let t = make_in_place(column);
        tau[j] = t;
        let essential = &column[1..];
        for c in j + 1..n {
            let offset = (c - j - 1) * m + j;
            apply_left(essential, t, &mut rest[offset..offset + m - j]);
        }
    }
}

/// Applies `Q = H_0 H_1 ... H_{k-1}` to a vector in place (the last reflector first), for a
/// factor packed by [`factor_in_place`] with m rows; `tau.len()` is the reflector count.
/// `Q e_j` is column j of `householderQ()`. Allocation-free.
pub fn apply_q(packed: &[f64], m: usize, tau: &[f64], x: &mut [f64]) {
    for j in (0..tau.len()).rev() {
        let start = j * m + j + 1;
        apply_left(&packed[start..start + m - j - 1], tau[j], &mut x[j..]);
    }
}

/// Applies `Q^T = H_{k-1} ... H_0` to a vector in place, for a factor packed by
/// [`factor_in_place`] with m rows. Allocation-free.
pub fn apply_q_transpose(packed: &[f64], m: usize, tau: &[f64], x: &mut [f64]) {
    for (j, &t) in tau.iter().enumerate() {
        let start = j * m + j + 1;
        apply_left(&packed[start..start + m - j - 1], t, &mut x[j..]);
    }
}

/// The upper-triangular (trapezoidal) part of the packed factor (Eigen's
/// `matrixQR().triangularView<Upper>()`).
pub fn upper_part(packed: &MatrixXd) -> MatrixXd {
    let mut r = MatrixXd::zeros(packed.rows(), packed.cols());
    for c in 0..packed.cols() {
        for row in 0..(c + 1).min(packed.rows()) {
            r[(row, c)] = packed[(row, c)];
        }
    }

    r
}

/// Back substitution `R(0:n, 0:n) z = y(0:n)` with the upper triangle of a column-major
/// packed factor with m rows. Allocation-free.
pub fn back_substitute(packed: &[f64], m: usize, n: usize, y: &mut [f64]) {
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in i + 1..n {
            sum -= packed[k * m + i] * y[k];
        }

        y[i] = sum / packed[i * m + i];
    }
}
