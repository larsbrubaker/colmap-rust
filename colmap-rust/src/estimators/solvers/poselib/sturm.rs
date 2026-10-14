//! Port of PoseLib's `PoseLib/misc/sturm.h` (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson;
//! see THIRD_PARTY_NOTICES.md; commit fa7280fee27f97aff31ae7f98bab7f583fac7d08, COLMAP
//! 4.2.0's pin): real-root isolation of a univariate polynomial by Sturm sequences and
//! bisection, finished with Ridders' method and Newton steps ([`bisect_sturm`]), and the
//! characteristic polynomial by Danilevsky's method with pivoting
//! ([`charpoly_danilevsky_piv`]). Port of colmap-sharp's
//! `Estimators/Solvers/PoseLib/Sturm.cs`. The PoseLib minimal solvers whose problem reduces
//! to one polynomial call it: re3q3 (degree 8, for P4Pf and GP3P), the five-point relative
//! pose (degree 10) and the shared-focal six-point relative pose (degree 15, through the
//! characteristic polynomial of its action matrix). Tests:
//! `tests/estimators/rust_only_sturm.rs`.
//!
//! Tier B: the roots are within `tol` of the true roots, and the arithmetic follows PoseLib's
//! operation order.
//!
//! Translation notes:
//! - PoseLib makes the degree N a template parameter so the buffers live on the stack. Rust
//!   cannot size an array `2N + 1` from a const generic on stable, so N is an argument and
//!   the buffers are stack arrays sized for [`MAX_DEGREE`]; there is still no heap
//!   allocation. A larger degree is a caller bug (PoseLib's degrees are compile-time
//!   constants) and panics.
//! - `signchanges<N>` counts sign changes with a popcount over a bit mask for N < 32; that is
//!   the same count as the plain loop PoseLib uses for N >= 32, which is what is ported.
//! - `MAX_STURM_RECURSION_DEPTH_LIMIT` keeps PoseLib's default, 300.

use crate::math::fns;

/// The largest polynomial degree (and matrix size for [`charpoly_danilevsky_piv`]) the stack
/// buffers hold. PoseLib's solvers use at most 15.
pub const MAX_DEGREE: usize = 32;

/// PoseLib's `MAX_STURM_RECURSION_DEPTH_LIMIT`.
const MAX_RECURSION_DEPTH: usize = 300;

/// The tolerance PoseLib's `bisect_sturm` defaults to.
pub const DEFAULT_TOL: f64 = 1e-10;

/// Find the real roots of the degree-`n` polynomial `sum coeffs[i] x^i` (`coeffs` holds
/// `n + 1` values, constant first) and write them to `roots` (which must hold `n` values) in
/// ascending order. Returns the number of roots written, or 0 when the leading coefficient is
/// exactly zero. Port of `poselib::sturm::bisect_sturm<N>`.
///
/// # Panics
///
/// When `n > MAX_DEGREE`, or `coeffs` / `roots` are shorter than `n + 1` / `n`.
pub fn bisect_sturm(n: usize, coeffs: &[f64], roots: &mut [f64], tol: f64) -> usize {
    assert!(n <= MAX_DEGREE, "bisect_sturm: degree {n} > {MAX_DEGREE}");
    if n == 0 {
        return 0;
    }

    if n == 1 {
        if coeffs[1] == 0.0 {
            return 0;
        }
        roots[0] = -coeffs[0] / coeffs[1];
        return 1;
    }

    if coeffs[n] == 0.0 {
        return 0;
    }

    let mut fvec = [0.0; 2 * MAX_DEGREE + 1];
    let mut svec = [0.0; 3 * MAX_DEGREE];

    // fvec is the polynomial and its first derivative.
    fvec[..n + 1].copy_from_slice(&coeffs[..n + 1]);

    // Normalize w.r.t. leading coeff
    let c_inv = 1.0 / fvec[n];
    for value in &mut fvec[..n] {
        *value *= c_inv;
    }
    fvec[n] = 1.0;

    // Compute the derivative with normalized coefficients
    for i in 0..n - 1 {
        fvec[n + 1 + i] = fvec[i + 1] * ((i + 1) as f64 / n as f64);
    }
    fvec[2 * n] = 1.0;

    // Compute sturm sequences
    build_sturm_seq(n, &fvec, &mut svec);

    // All real roots are in the interval [-r0, r0]
    let r0 = get_bounds(n, &fvec);
    let a = -r0;
    let b = r0;

    let sa = sign_changes(n, &svec, a);
    let sb = sign_changes(n, &svec, b);

    // PoseLib's `int n_roots = sa - sb; if (n_roots == 0) return 0;`. Rounding can leave
    // sa < sb (the counts are unsigned here), and then PoseLib's negative count isolates no
    // root below, so comparing the counts keeps its result without the unsigned underflow.
    if sa == sb {
        return 0;
    }

    let mut n_roots = 0;
    isolate_roots(n, &fvec, &svec, a, b, sa, sb, roots, &mut n_roots, tol, 0);
    n_roots
}

/// Constructs the quotients needed for evaluating the sturm sequence.
fn build_sturm_seq(n: usize, fvec: &[f64], svec: &mut [f64]) {
    let mut f = [0.0; 3 * MAX_DEGREE];
    f[..2 * n + 1].copy_from_slice(&fvec[..2 * n + 1]);

    // f1, f2, f3 are offsets into f, juggled as PoseLib juggles its pointers.
    let mut f1 = 0;
    let mut f2 = n + 1;
    let mut f3 = f2 + n;

    for i in 0..n - 1 {
        let q1 = f[f1 + n - i] * f[f2 + n - 1 - i];
        let q0 = f[f1 + n - 1 - i] * f[f2 + n - 1 - i] - f[f1 + n - i] * f[f2 + n - 2 - i];

        f[f3] = f[f1] - q0 * f[f2];
        for j in 1..n - 1 - i {
            f[f3 + j] = f[f1 + j] - q1 * f[f2 + j - 1] - q0 * f[f2 + j];
        }

        let c = -f[f3 + n - 2 - i].abs();
        let ci = 1.0 / c;
        for j in 0..n - 1 - i {
            f[f3 + j] *= ci;
        }

        // juggle pointers (f1,f2,f3) -> (f2,f3,f1)
        let tmp = f1;
        f1 = f2;
        f2 = f3;
        f3 = tmp;

        svec[3 * i] = q0;
        svec[3 * i + 1] = q1;
        svec[3 * i + 2] = c;
    }

    svec[3 * n - 3] = f[f1];
    svec[3 * n - 2] = f[f1 + 1];
    svec[3 * n - 1] = f[f2];
}

/// Evaluates polynomial using Horner's method. Assumes that `f[n] = 1.0`.
fn polyval(n: usize, f: &[f64], x: f64) -> f64 {
    let mut fx = x + f[n - 1];
    for i in (0..n - 1).rev() {
        fx = x * fx + f[i];
    }
    fx
}

/// Evaluates the sturm sequence and counts the number of sign changes.
fn sign_changes(n: usize, svec: &[f64], x: f64) -> usize {
    let mut f = [0.0; MAX_DEGREE + 1];
    f[n] = svec[3 * n - 1];
    f[n - 1] = svec[3 * n - 3] + x * svec[3 * n - 2];

    for i in (0..n - 1).rev() {
        f[i] = (svec[3 * i] + x * svec[3 * i + 1]) * f[i + 1] + svec[3 * i + 2] * f[i + 2];
    }

    let mut count = 0;
    let mut neg1 = f[0] < 0.0;
    for value in &f[1..n + 1] {
        let neg2 = *value < 0.0;
        if neg1 ^ neg2 {
            count += 1;
        }
        neg1 = neg2;
    }
    count
}

/// Computes the Cauchy bound on the real roots. Experiments with more complicated
/// (expensive) bounds did not seem to have a good trade-off.
fn get_bounds(n: usize, fvec: &[f64]) -> f64 {
    let mut max = 0.0f64;
    for value in &fvec[..n] {
        // PoseLib's std::max(max, std::abs(v)) keeps `max` for a NaN coefficient
        // (colmap-sharp's Math.Max returns NaN there); a NaN coefficient gives no usable
        // roots either way.
        let abs = value.abs();
        if max < abs {
            max = abs;
        }
    }
    1.0 + max
}

/// Applies Ridder's bracketing method until we get close to root, followed by newton
/// iterations.
fn ridders_method_newton(
    n: usize,
    fvec: &[f64],
    mut a: f64,
    mut b: f64,
    roots: &mut [f64],
    n_roots: &mut usize,
    tol: f64,
) {
    let mut fa = polyval(n, fvec, a);
    let mut fb = polyval(n, fvec, b);

    if !((fa < 0.0) ^ (fb < 0.0)) {
        return;
    }

    const TOL_NEWTON: f64 = 1e-3;

    for _ in 0..30 {
        if (a - b).abs() < TOL_NEWTON {
            break;
        }

        let c = (a + b) * 0.5;
        let fc = polyval(n, fvec, c);
        let s = fns::sqrt(fc * fc - fa * fb);
        // C++ `if (!s)`: true only for s == 0 (a NaN s is truthy and carries on).
        if s == 0.0 {
            break;
        }

        let d = if fa < fb {
            c + (a - c) * fc / s
        } else {
            c + (c - a) * fc / s
        };
        let fd = polyval(n, fvec, d);

        if if fd >= 0.0 { fc < 0.0 } else { fc > 0.0 } {
            a = c;
            fa = fc;
            b = d;
            fb = fd;
        } else if if fd >= 0.0 { fa < 0.0 } else { fa > 0.0 } {
            b = d;
            fb = fd;
        } else {
            a = d;
            fa = fd;
        }
    }

    // We switch to Newton's method once we are close to the root
    let mut x = (a + b) * 0.5;

    let fpvec = &fvec[n + 1..];
    for _ in 0..10 {
        let fx = polyval(n, fvec, x);
        if fx.abs() < tol {
            break;
        }
        let fpx = n as f64 * polyval(n - 1, fpvec, x);
        let dx = fx / fpx;
        x -= dx;
        if dx.abs() < tol {
            break;
        }
    }

    roots[*n_roots] = x;
    *n_roots += 1;
}

#[allow(clippy::too_many_arguments)] // PoseLib's recursive signature, kept as is.
fn isolate_roots(
    n: usize,
    fvec: &[f64],
    svec: &[f64],
    a: f64,
    b: f64,
    sa: usize,
    sb: usize,
    roots: &mut [f64],
    n_roots: &mut usize,
    tol: f64,
    depth: usize,
) {
    if depth > MAX_RECURSION_DEPTH {
        return;
    }

    if b - a < tol {
        roots[*n_roots] = b;
        *n_roots += 1;
        return;
    }

    // Sturm's theorem makes sa >= sb on a < b; wrapping keeps a degenerate (NaN-driven)
    // count from panicking and lands in neither branch below, as PoseLib's negative int does.
    let n_rts = sa.wrapping_sub(sb) as isize;

    if n_rts > 1 {
        let c = (a + b) * 0.5;
        let sc = sign_changes(n, svec, c);
        isolate_roots(n, fvec, svec, a, c, sa, sc, roots, n_roots, tol, depth + 1);
        isolate_roots(n, fvec, svec, c, b, sc, sb, roots, n_roots, tol, depth + 1);
    } else if n_rts == 1 {
        ridders_method_newton(n, fvec, a, b, roots, n_roots, tol);
    }
}

/// Writes the monic characteristic polynomial of the `n x n` column-major matrix `a`
/// (destroyed) to `p` (`n + 1` coefficients, constant first), by Danilevsky's reduction to
/// Frobenius form with row/column pivoting. Port of
/// `poselib::sturm::charpoly_danilevsky_piv`.
///
/// # Panics
///
/// When `n > MAX_DEGREE`, or `a` / `p` are shorter than `n * n` / `n + 1`.
pub fn charpoly_danilevsky_piv(a: &mut [f64], n: usize, p: &mut [f64]) {
    assert!(
        n <= MAX_DEGREE,
        "charpoly_danilevsky_piv: size {n} > {MAX_DEGREE}"
    );
    let mut v = [0.0; MAX_DEGREE];
    let mut vinv = [0.0; MAX_DEGREE];
    let mut a_col = [0.0; MAX_DEGREE];
    let mut new_row = [0.0; MAX_DEGREE];
    for i in (1..n).rev() {
        let mut piv_ind = i - 1;
        let mut piv = a[i + n * (i - 1)].abs();

        // Find largest pivot
        for j in 0..i - 1 {
            if a[i + n * j].abs() > piv {
                piv = a[i + n * j].abs();
                piv_ind = j;
            }
        }

        if piv_ind != i - 1 {
            // Perform permutation
            for c in 0..n {
                a.swap((i - 1) + n * c, piv_ind + n * c);
            }
            for r in 0..n {
                a.swap(r + n * (i - 1), r + n * piv_ind);
            }
        }

        piv = a[i + n * (i - 1)];

        // A.row(i - 1) = v^T * A with v = A.row(i).
        for c in 0..n {
            v[c] = a[i + n * c];
        }
        for c in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += v[k] * a[k + n * c];
            }
            new_row[c] = sum;
        }
        for c in 0..n {
            a[(i - 1) + n * c] = new_row[c];
        }

        for k in 0..n {
            vinv[k] = -v[k]; // PoseLib: -1.0 * v (bit-identical)
        }
        vinv[i - 1] = 1.0;
        for value in &mut vinv[..n] {
            *value /= piv;
        }
        vinv[i - 1] -= 1.0;
        for r in 0..n {
            a_col[r] = a[r + n * (i - 1)];
        }

        for j in 0..=i {
            for c in 0..n {
                a[j + n * c] += a_col[j] * vinv[c];
            }
        }

        for c in 0..n {
            a[i + n * c] = 0.0;
        }
        a[i + n * (i - 1)] = 1.0;
    }

    p[n] = 1.0;
    for i in 0..n {
        p[i] = -a[n * (n - i - 1)];
    }
}
