//! JacobiSvdKernel: the square two-sided Jacobi SVD that [`JacobiSvd`](super::JacobiSvd) (any
//! shape, `MatrixXd`) and the allocation-free fixed-size wrappers
//! [`Svd3d`](super::Svd3d) / [`Svd4d`](super::Svd4d) share. It works in place on column-major
//! slices so the fixed-size callers can run it on stack arrays. Port of colmap-sharp's
//! `ColmapSharp/LinearAlgebra/JacobiSvdKernel.cs`; Eigen (MPL-2.0) is not ported.
//!
//! Algorithm: two-sided (Kogbetliantz) cyclic Jacobi, Golub & Van Loan, "Matrix
//! Computations", 4th ed., §8.6.3. Each 2x2 step follows Brent, Luk and Van Loan,
//! "Computation of the singular value decomposition using mesh-connected processors",
//! J. VLSI Comput. Syst. 1 (1985), which diagonalizes `B = [w x; y z]` by a left rotation
//! through theta and a right rotation through phi found from their sum and difference.
//! Derivation used here: split B into a scaled rotation plus a scaled reflection,
//!   `B = r1 Rot(a1) + r2 Ref(a2)`, `Rot(a) = [cos -sin; sin cos]`, `Ref(a) = [cos sin; sin -cos]`,
//! with `r1 Rot(a1) = [(w+z)/2, (x-y)/2; (y-x)/2, (w+z)/2]` and
//!      `r2 Ref(a2) = [(w-z)/2, (x+y)/2; (x+y)/2, (z-w)/2]`.
//! Since `Rot(theta)^T Rot(a) Rot(phi) = Rot(a - theta + phi)` and
//! `Rot(theta)^T Ref(a) Rot(phi) = Ref(a - theta - phi)`, choosing `theta = (a1 + a2) / 2` and
//! `phi = (a2 - a1) / 2` gives `Rot(theta)^T B Rot(phi) = diag(r1 + r2, r1 - r2)`. The angles
//! come from atan2, which neither overflows nor underflows. Every rotation is accumulated
//! into U and V, so both are orthogonal to working precision whatever the rank, which is
//! what COLMAP's null-space reads (the last column of V) rely on. A 2x2 block with an exactly
//! zero column (row) is instead diagonalized by the left (right) rotation alone, the right
//! (left) one being exactly the identity, so exact null vectors from zero columns stay exact
//! (`try_one_sided_rotation`).
//!
//! Convergence: the relative test of Demmel and Veselić, "Jacobi's method is more accurate
//! than QR", SIAM J. Matrix Anal. Appl. 13 (1992): the pair (i, j) is rotated while
//! `max(|a(i,j)|, |a(j,i)|) > max(eps * sqrt(|a(i,i)|) * sqrt(|a(j,j)|), eps * ||A||_F)`,
//! and the iteration stops after a sweep that rotates nothing. The absolute floor
//! `eps * ||A||_F` is Golub & Van Loan's stopping level `off(A) <= eps ||A||_F` (§8.5.3)
//! applied per pair, in the spirit of the tolerances of Drmač and Veselić, "New fast and
//! accurate Jacobi SVD algorithm" I/II (LAPACK Working Notes 169/170, 2007): without it a
//! zero diagonal (rank-deficient input: essential matrices, noise-free DLT systems, zero
//! columns) would keep rotating on roundoff. Exactly rank-deficient inputs converge in a
//! few sweeps (the spectral tests pin the counts; Info NoConvergence if a sweep cap is
//! reached first, which does not happen on finite input in practice). The input is scaled by
//! the power of two nearest its Frobenius norm (computed without overflow), which is exact,
//! and the singular values scaled back.
//!
//! Conventions (Eigen's documented JacobiSVD semantics): singular values non-negative and
//! sorted in decreasing order. A negative diagonal entry flips the sign of the matching U
//! column. Ties keep their original diagonal position (stable insertion sort). The signs of
//! singular vector pairs are whatever the rotations produce; like Eigen's, they are
//! arbitrary, and COLMAP only uses them up to sign (docs/CPP_DIVERGENCES.md). Tier B.
//!
//! `ilogb` / `scalbn` (C#'s `Math.ILogB` / `Math.ScaleB`) are exact operations and come
//! straight from the `libm` crate. Tests: `colmap-rust/tests/linalg/rust_only_spectral_svd.rs`
//! and `rust_only_spectral_oracle.rs`.

use super::{ComputationInfo, MACHINE_EPSILON};
use crate::math::fns;

const MAX_SWEEPS: usize = 100;

/// Decomposes the n x n column-major matrix in `a` (destroyed) as `U diag(s) V^T`. `u` and
/// `v` receive n x n column-major factors when non-empty; `s` receives the n singular
/// values, descending. Returns the info and the number of sweeps run (the last one rotates
/// nothing; 0 for a zero or non-finite matrix). InvalidInput for a non-finite entry
/// (outputs are then NaN).
pub(crate) fn decompose(
    a: &mut [f64],
    n: usize,
    u: &mut [f64],
    v: &mut [f64],
    s: &mut [f64],
) -> (ComputationInfo, usize) {
    let mut sweeps = 0;
    if !u.is_empty() {
        set_identity(u, n);
    }
    if !v.is_empty() {
        set_identity(v, n);
    }

    if a.iter().any(|value| !value.is_finite()) {
        s.fill(f64::NAN);
        u.fill(f64::NAN);
        v.fill(f64::NAN);
        return (ComputationInfo::InvalidInput, 0);
    }

    let norm = frobenius_norm(a);
    if norm == 0.0 {
        s[..n].fill(0.0);
        return (ComputationInfo::Success, 0);
    }

    // Scale by the power of two nearest the norm: exact, so an input that needs no
    // rotation (a diagonal matrix) returns its |entries| bit for bit.
    let exponent = libm::ilogb(norm);
    for value in a.iter_mut() {
        *value = libm::scalbn(*value, -exponent);
    }

    let floor = MACHINE_EPSILON * frobenius_norm(a);
    let mut info = ComputationInfo::NoConvergence;
    for sweep in 0..MAX_SWEEPS {
        sweeps = sweep + 1;
        let mut rotated = false;
        for i in 0..n.saturating_sub(1) {
            for j in i + 1..n {
                let off = a[j * n + i].abs().max(a[i * n + j].abs());
                let diagonal_scale = fns::sqrt(a[i * n + i].abs()) * fns::sqrt(a[j * n + j].abs());
                if off > (MACHINE_EPSILON * diagonal_scale).max(floor) {
                    rotated = true;
                    rotate_pair(a, n, i, j, u, v);
                }
            }
        }

        if !rotated {
            info = ComputationInfo::Success;
            break;
        }
    }

    for i in 0..n {
        let d = a[i * n + i];
        s[i] = libm::scalbn(d.abs(), exponent);
        if d < 0.0 && !u.is_empty() {
            for value in &mut u[i * n..(i + 1) * n] {
                *value = -*value;
            }
        }
    }

    sort_descending(n, u, v, s);
    (info, sweeps)
}

/// The Brent-Luk-Van Loan step on rows/columns `i < j` (derivation in the file header):
/// `A <- Rot(theta)^T A Rot(phi)`, `U <- U Rot(theta)`, `V <- V Rot(phi)`.
fn rotate_pair(a: &mut [f64], n: usize, i: usize, j: usize, u: &mut [f64], v: &mut [f64]) {
    let w = a[i * n + i];
    let x = a[j * n + i];
    let y = a[i * n + j];
    let z = a[j * n + j];
    let (ct, st, cp, sp) = match try_one_sided_rotation(w, x, y, z) {
        Some(rotation) => rotation,
        None => {
            let rotation_angle = fns::atan2(0.5 * y - 0.5 * x, 0.5 * w + 0.5 * z);
            let reflection_angle = fns::atan2(0.5 * x + 0.5 * y, 0.5 * w - 0.5 * z);
            let theta = 0.5 * (rotation_angle + reflection_angle);
            let phi = 0.5 * (reflection_angle - rotation_angle);
            (
                fns::cos(theta),
                fns::sin(theta),
                fns::cos(phi),
                fns::sin(phi),
            )
        }
    };

    // Rows i, j <- Rot(theta)^T rows.
    for k in 0..n {
        let ri = a[k * n + i];
        let rj = a[k * n + j];
        a[k * n + i] = ct * ri + st * rj;
        a[k * n + j] = -st * ri + ct * rj;
    }

    // Columns i, j <- columns Rot(phi).
    rotate_columns(a, n, i, j, cp, sp);

    // The pair is diagonal in exact arithmetic; drop the rounding residue.
    a[j * n + i] = 0.0;
    a[i * n + j] = 0.0;

    // A = Rot(theta) A' Rot(phi)^T.
    if !u.is_empty() {
        rotate_columns(u, n, i, j, ct, st);
    }
    if !v.is_empty() {
        rotate_columns(v, n, i, j, cp, sp);
    }
}

/// The 2x2 block `B = [w x; y z]` with an exactly zero column (or row) is diagonalized by a
/// left (right) rotation alone: the rotation that folds the other column (row) onto one
/// axis. The right (left) rotation is then exactly the identity, so a zero column of A is
/// never mixed into V (a zero row never into U), and the null vector it stands for stays an
/// exact unit vector. The general two-angle step cannot promise that: `cos(pi/2)` is
/// 6.1e-17, not 0, and that residue lands in coordinates COLMAP tests for exact zero
/// (TriangulatePoint rejects parallel rays by `V(3,3) == 0`, triangulation_test.cc
/// TriangulatePoint.ParallelRays). Returns `(ct, st, cp, sp)`, or None when neither case
/// applies.
fn try_one_sided_rotation(w: f64, x: f64, y: f64, z: f64) -> Option<(f64, f64, f64, f64)> {
    if x == 0.0 && z == 0.0 {
        // Column j is zero: Rot(theta)^T (w, y) = (r, 0).
        let r = hypot(w, y);
        return Some((w / r, y / r, 1.0, 0.0));
    }

    if w == 0.0 && y == 0.0 {
        // Column i is zero: Rot(theta)^T (x, z) = (0, r).
        let r = hypot(x, z);
        return Some((z / r, -x / r, 1.0, 0.0));
    }

    if y == 0.0 && z == 0.0 {
        // Row j is zero: (w, x) Rot(phi) = (r, 0).
        let r = hypot(w, x);
        return Some((1.0, 0.0, w / r, x / r));
    }

    if w == 0.0 && x == 0.0 {
        // Row i is zero: (y, z) Rot(phi) = (0, r).
        let r = hypot(y, z);
        return Some((1.0, 0.0, z / r, -y / r));
    }

    None
}

/// `sqrt(a^2 + b^2)` of a nonzero pair without overflow or underflow (colmap-sharp's own
/// formula, not libm's hypot, so the bits match colmap-sharp).
fn hypot(a: f64, b: f64) -> f64 {
    let largest = a.abs().max(b.abs());
    let sa = a / largest;
    let sb = b / largest;
    largest * fns::sqrt(sa * sa + sb * sb)
}

/// Columns i, j of an n x n matrix times `Rot = [c -s; s c]`.
fn rotate_columns(m: &mut [f64], n: usize, i: usize, j: usize, c: f64, s: f64) {
    for k in 0..n {
        let p = m[i * n + k];
        let q = m[j * n + k];
        m[i * n + k] = c * p + s * q;
        m[j * n + k] = -s * p + c * q;
    }
}

/// Frobenius norm without overflow or underflow: the entries are divided by the largest
/// magnitude before squaring. The sum starts from `0.0` as in colmap-sharp (the terms are
/// squares, so this equals seeding with the first term).
pub(crate) fn frobenius_norm(values: &[f64]) -> f64 {
    let mut largest = 0.0_f64;
    for &value in values {
        largest = largest.max(value.abs());
    }
    if largest == 0.0 {
        return 0.0;
    }

    let mut sum = 0.0;
    for &value in values {
        let scaled = value / largest;
        sum += scaled * scaled;
    }
    largest * fns::sqrt(sum)
}

/// Insertion sort of the singular values into decreasing order, moving the matching U and V
/// columns. Insertion sort is stable, so equal values keep their order.
fn sort_descending(n: usize, u: &mut [f64], v: &mut [f64], s: &mut [f64]) {
    for i in 1..n {
        let mut k = i;
        while k > 0 && s[k] > s[k - 1] {
            s.swap(k, k - 1);
            swap_columns(u, n, k, k - 1);
            swap_columns(v, n, k, k - 1);
            k -= 1;
        }
    }
}

/// Swaps columns `a` and `b` of the n-row column-major `m`; no-op for an empty `m`.
pub(crate) fn swap_columns(m: &mut [f64], n: usize, a: usize, b: usize) {
    if m.is_empty() {
        return;
    }
    for k in 0..n {
        m.swap(a * n + k, b * n + k);
    }
}

fn set_identity(m: &mut [f64], n: usize) {
    m[..n * n].fill(0.0);
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
}

/// Eigen's documented default rank rule: a singular value counts as nonzero when it is
/// strictly greater than `max(1, diagSize) * epsilon *` the largest singular value.
pub(crate) fn rank(s: &[f64]) -> usize {
    if s.is_empty() || s[0] == 0.0 {
        return 0;
    }
    let threshold = s[0] * (s.len().max(1) as f64) * MACHINE_EPSILON;
    s.iter().filter(|&&value| value > threshold).count()
}
