//! Port of the small dense linear algebra of VLFeat's `mathop.c` that the covariant detector
//! uses: `vl_svd2` with its `vl_lapack_dlasv2` (Bai and Demmel's 2x2 triangular SVD, as written
//! in VLFeat's own C, not LAPACK's Fortran), `vl_gaussian_elimination` and
//! `vl_solve_linear_system_2` / `_3`. Port of colmap-sharp's `Feature/VLFeat/VlMathOp.Linear.cs`.
//! Neighbor: [`super::mathop`] (the SIFT approximations).
//!
//! Tier A (exact): double throughout, same operation order as the C, no FMA. Matrices are
//! VLFeat's column-major 2x2 / 3x3 arrays. VLFeat's `VL_ERR_OVERFLOW` return of a singular
//! elimination is an `Err(`[`SingularSystem`]`)`; the matrix (and the solution, for the
//! `solve` functions) is left exactly as VLFeat leaves it.

use super::mathop::VL_EPSILON_D;

/// VLFeat's `VL_ERR_OVERFLOW` from [`vl_gaussian_elimination`]: a pivot below `1e-10`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SingularSystem;

/// `mathop.c`'s `sign(x)` macro: -1 for negative, +1 otherwise (so +1 for zero and NaN).
fn sign(x: f64) -> i32 {
    if x < 0.0 {
        -1
    } else {
        1
    }
}

/// `mathop.c`'s `isign(i)` macro.
fn isign(i: i32) -> f64 {
    if i < 0 {
        -1.0
    } else {
        1.0
    }
}

/// Port of `vl_svd2`: `M = U * S * V'` for a column-major 2x2 matrix. Returns `(S, U, V)`;
/// `S[0]` and `S[3]` are the signed singular values (largest modulus first), `S[1] = S[2] = 0`.
pub fn vl_svd2(m: &[f64; 4]) -> ([f64; 4], [f64; 4], [f64; 4]) {
    let m11 = m[0];
    let m21 = m[1];
    let m12 = m[2];
    let m22 = m[3];
    let mut cu1 = m11;
    let mut su1 = m21;
    let norm = (cu1 * cu1 + su1 * su1).sqrt();
    cu1 /= norm;
    su1 /= norm;

    let f = cu1 * m11 + su1 * m21;
    let g = cu1 * m12 + su1 * m22;
    let h = -su1 * m12 + cu1 * m22;

    let r = vl_lapack_dlasv2(f, g, h);
    let (cu2, su2, cv2, sv2) = (r.cu, r.su, r.cv, r.sv);

    let s = [r.smax, 0.0, 0.0, r.smin];
    let u = [
        cu2 * cu1 - su2 * su1,
        su2 * cu1 + cu2 * su1,
        -cu2 * su1 - su2 * cu1,
        -su2 * su1 + cu2 * cu1,
    ];
    let v = [cv2, sv2, -sv2, cv2];
    (s, u, v)
}

/// The outputs of [`vl_lapack_dlasv2`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dlasv2 {
    /// Smallest (in modulus) singular value.
    pub smin: f64,
    /// Largest (in modulus) singular value.
    pub smax: f64,
    /// Second component of the right singular vector of `smax`.
    pub sv: f64,
    /// First component of the right singular vector of `smax`.
    pub cv: f64,
    /// Second component of the left singular vector of `smax`.
    pub su: f64,
    /// First component of the left singular vector of `smax`.
    pub cu: f64,
}

/// Port of `vl_lapack_dlasv2`: SVD of the upper triangular `[f g; 0 h]` (Z. Bai and J. Demmel,
/// "Computing the Generalized Singular Value Decomposition", SIAM J. Sci. Comput. 14(6), 1993),
/// in VLFeat's formulation.
pub fn vl_lapack_dlasv2(f: f64, g: f64, h: f64) -> Dlasv2 {
    let (mut svt, mut cvt, mut sut, mut cut) = (0.0, 0.0, 0.0, 0.0);
    let (mut ft, gt, mut ht) = (f, g, h);
    let (mut fa, ga, mut ha) = (f.abs(), g.abs(), h.abs());
    let mut pmax = 1;
    let mut swap = false;
    let mut glarge = false;
    let mut smin = 0.0;
    let mut smax = 0.0;

    // Make fa >= ha.
    if fa < ha {
        pmax = 3;
        std::mem::swap(&mut ft, &mut ht);
        std::mem::swap(&mut fa, &mut ha);
        swap = true;
    }

    if ga == 0.0 {
        // Diagonal.
        smin = ha;
        smax = fa;
        cut = 1.0;
        sut = 0.0;
        cvt = 1.0;
        svt = 0.0;
    } else {
        if ga > fa {
            // g is the largest entry.
            pmax = 2;
            if fa / ga < VL_EPSILON_D {
                // g is very large.
                glarge = true;
                smax = ga;
                smin = if ha > 1.0 {
                    fa / (ga / ha)
                } else {
                    (fa / ga) * ha
                };
                cut = 1.0;
                sut = ht / gt;
                cvt = 1.0;
                svt = ft / gt;
            }
        }

        if !glarge {
            // Normal case.
            let fmh = fa - ha;
            // Cope with infinite f or h.
            let d = if fmh == fa { 1.0 } else { fmh / fa };
            let q = gt / ft;
            let s = 2.0 - d;
            let dd = d * d;
            let qq = q * q;
            let ss = s * s;
            let spq = (ss + qq).sqrt();
            let dpq = if d == 0.0 { q.abs() } else { (dd + qq).sqrt() };
            let a = 0.5 * (spq + dpq);
            smin = ha / a;
            smax = fa * a;
            let tmp = if qq == 0.0 {
                // qq underflow.
                if d == 0.0 {
                    f64::from(sign(ft) * 2 * sign(gt))
                } else {
                    gt / (f64::from(sign(ft)) * fmh) + q / s
                }
            } else {
                (q / (spq + s) + q / (dpq + d)) * (1.0 + a)
            };
            let tt = (tmp * tmp + 4.0).sqrt();
            cvt = 2.0 / tt;
            svt = tmp / tt;
            cut = (cvt + svt * q) / a;
            sut = (ht / ft) * svt / a;
        }
    }

    let (cu, su, cv, sv) = if swap {
        (svt, cvt, sut, cut)
    } else {
        (cut, sut, cvt, svt)
    };

    // Correct the signs of smax and smin.
    let tsign = match pmax {
        1 => sign(cv) * sign(cu) * sign(f),
        2 => sign(sv) * sign(cu) * sign(g),
        _ => sign(sv) * sign(su) * sign(h),
    };
    Dlasv2 {
        smax: isign(tsign) * smax,
        smin: isign(tsign * sign(f) * sign(h)) * smin,
        sv,
        cv,
        su,
        cu,
    }
}

/// Port of `vl_gaussian_elimination`: Gaussian elimination with partial pivoting of the
/// column-major `num_rows` x `num_columns` matrix `[A, b]` in place, leaving the solution in
/// the last columns. `Err` (`VL_ERR_OVERFLOW`), with the matrix half-processed as VLFeat leaves
/// it, when a pivot is below `1e-10`.
pub fn vl_gaussian_elimination(
    a: &mut [f64],
    num_rows: usize,
    num_columns: usize,
) -> Result<(), SingularSystem> {
    let at = |i: usize, j: usize| i + j * num_rows;
    for j in 0..num_rows {
        let mut maxa = 0.0;
        let mut maxabsa = 0.0;
        let mut maxi = None;

        // Look for the maximally stable pivot.
        for i in j..num_rows {
            let v = a[at(i, j)];
            let absa = f64::abs(v);
            if absa > maxabsa {
                maxa = v;
                maxabsa = absa;
                maxi = Some(i);
            }
        }

        // If singular give up.
        let maxi = match maxi {
            Some(maxi) if maxabsa >= 1e-10 => maxi,
            _ => return Err(SingularSystem),
        };

        // Swap the j-th row with the i-th row and normalize the j-th row.
        for jj in j..num_columns {
            a.swap(at(maxi, jj), at(j, jj));
            a[at(j, jj)] /= maxa;
        }

        // Elimination.
        for ii in j + 1..num_rows {
            let x = a[at(ii, j)];
            for jj in j..num_columns {
                a[at(ii, jj)] -= x * a[at(j, jj)];
            }
        }
    }

    // Backward substitution.
    for i in (1..num_rows).rev() {
        // Substitute in all rows above.
        for ii in (0..i).rev() {
            let x = a[at(ii, i)];
            for j in num_rows..num_columns {
                a[at(ii, j)] -= x * a[at(i, j)];
            }
        }
    }

    Ok(())
}

/// Port of `vl_solve_linear_system_3`: `x = A \ b` for a column-major 3x3 `A`. On `Err`, `x` is
/// what VLFeat writes from the half-processed elimination.
pub fn vl_solve_linear_system_3(
    x: &mut [f64; 3],
    a: &[f64; 9],
    b: &[f64; 3],
) -> Result<(), SingularSystem> {
    let mut m = [0.0f64; 12];
    m[..9].copy_from_slice(a);
    m[9..].copy_from_slice(b);
    let result = vl_gaussian_elimination(&mut m, 3, 4);
    x.copy_from_slice(&m[9..]);
    result
}

/// Port of `vl_solve_linear_system_2`: `x = A \ b` for a column-major 2x2 `A`. On `Err`, `x` is
/// what VLFeat writes from the half-processed elimination.
pub fn vl_solve_linear_system_2(
    x: &mut [f64; 2],
    a: &[f64; 4],
    b: &[f64; 2],
) -> Result<(), SingularSystem> {
    let mut m = [0.0f64; 6];
    m[..4].copy_from_slice(a);
    m[4..].copy_from_slice(b);
    let result = vl_gaussian_elimination(&mut m, 2, 3);
    x.copy_from_slice(&m[4..]);
    result
}
