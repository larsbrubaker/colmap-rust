//! Port of PoseLib's `PoseLib/solvers/relpose_5pt.cc` (`poselib::relpose_5pt`, the
//! essential-matrix overload COLMAP calls; BSD-3-Clause, Copyright (c) 2020 Viktor Larsson;
//! see THIRD_PARTY_NOTICES.md; commit fa7280fee27f97aff31ae7f98bab7f583fac7d08), through
//! colmap-sharp's `Estimators/Solvers/PoseLib/Relpose5pt.cs`: the minimal five-point relative
//! pose solver of D. Nister, "An Efficient Solution to the Five-Point Relative Pose Problem",
//! PAMI 2004. The minimal case of COLMAP's `EssentialMatrixFivePointEstimator`. The
//! `CameraPose` overload (which decomposes each E with `misc/essential.cc`) is not ported:
//! COLMAP does its own decomposition.
//!
//! Steps: the 4D null space of the five epipolar constraints; the ten cubic constraints
//! (determinant plus the trace constraint) in its coefficients
//! (`compute_trace_constraints`); Gauss-Jordan elimination with a 10 x 10 LU solve; the
//! degree-10 determinant polynomial of the resulting 3 x 3 polynomial matrix (the sibling
//! `relpose_5pt_det`), whose real roots [`super::sturm`] brackets; back substitution for each
//! root.
//!
//! The null space comes from an unpivoted Householder QR instead of PoseLib's
//! `fullPivHouseholderQr`: same space, different basis, same solution set. Tier B. Tests:
//! `tests/estimators/rust_only_relpose_5pt.rs`.
//!
//! That is a deliberate divergence. PoseLib's `relpose_5pt` (the minimal case of COLMAP's
//! `EssentialMatrixFivePointEstimator`) takes the 4D null space of its five epipolar
//! constraints from the last four columns of the full Q of
//! `epipolar_constraints.fullPivHouseholderQr()` (a 9 x 5 matrix). This port takes the same
//! four columns from an unpivoted Householder QR (`linalg::householder::factor_in_place` /
//! `apply_q`). Both are orthonormal bases of the same space; only the basis inside it differs,
//! and with it the intermediate polynomial coefficients (`compute_trace_constraints`, the
//! degree-10 determinant polynomial and its roots, which parametrize the solutions in basis
//! coordinates). Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no
//! published convention precise enough to reproduce Eigen's Q column for column. The solution
//! set does not depend on the basis: the solver returns every unit-norm
//! E = x N0 + y N1 + z N2 + N3 in the null space that meets the cubic constraints, and a change
//! of orthonormal basis maps that set onto itself (up to the sign of each E). What can differ is
//! numerical conditioning near the back substitution's 1e-6 switch to the three-row QR solve and
//! the Sturm bracketing tolerance, where a near-degenerate root can be kept on one side and
//! dropped on the other. `rust_only_recovers_true_essential_matrix` checks that on 100 random
//! minimal problems every solution has unit norm and satisfies the five epipolar constraints to
//! 1e-8, and the true E is among the solutions up to sign; colmap-sharp's 1:1
//! `EssentialMatrixFivePointEstimatorTests_Nominal` cases pass on the same construction. Not
//! compared against C++ PoseLib intermediates. Same as colmap-sharp entry 27.

use super::relpose_5pt_det::determinant_polynomial;
use super::sturm::{bisect_sturm, DEFAULT_TOL};
use crate::linalg::householder::{apply_q, factor_in_place};
use crate::linalg::{
    ColPivHouseholderQr, Matrix2d, Matrix3d, MatrixXd, PartialPivLu, Vector3d, VectorXd,
};
use crate::math::fns;

/// Computes the essential matrices from five point correspondences (bearings `x1`, `x2`,
/// with `x2^T E x1 = 0`), appends them to `essential_matrices` (not cleared, as in PoseLib)
/// and returns their count. Every E has unit Frobenius norm. Port of `poselib::relpose_5pt`.
pub fn relpose_5pt(
    x1: &[Vector3d],
    x2: &[Vector3d],
    essential_matrices: &mut Vec<Matrix3d>,
) -> usize {
    // Compute nullspace to epipolar constraints. Column i holds x1[i](k) * x2[i](r) at
    // 3k + r, the coefficient of E(r, k) in column-major order.
    let mut epipolar_constraints = [0.0; 9 * 5];
    for i in 0..5 {
        let col = &mut epipolar_constraints[9 * i..9 * i + 9];
        for k in 0..3 {
            for r in 0..3 {
                col[3 * k + r] = x1[i][k] * x2[i][r];
            }
        }
    }

    // The last four columns of the full Q of the 9 x 5 system are orthogonal to every
    // constraint. N (4 x 9, column-major) holds them as rows: n[4k + j] is basis vector j's
    // coefficient of E element k.
    let mut tau = [0.0; 5];
    factor_in_place(&mut epipolar_constraints, 9, 5, &mut tau);
    let mut n = [0.0; 4 * 9];
    for j in 0..4 {
        let mut q = [0.0; 9];
        q[5 + j] = 1.0;
        apply_q(&epipolar_constraints, 9, &tau, &mut q);
        for k in 0..9 {
            n[4 * k + j] = q[k];
        }
    }

    // Compute equation coefficients for the trace constraints + determinant.
    let mut coeffs = [0.0; 10 * 20];
    compute_trace_constraints(&n, &mut coeffs);

    // coeffs.block<10, 10>(0, 10) = coeffs.block<10, 10>(0, 0).partialPivLu().solve(
    //     coeffs.block<10, 10>(0, 10)); both blocks are contiguous column-major.
    let lu = PartialPivLu::new(&MatrixXd::from_column_major(10, 10, &coeffs[..100]));
    let solved = lu.solve_matrix(&MatrixXd::from_column_major(10, 10, &coeffs[100..]));
    coeffs[100..].copy_from_slice(solved.as_slice());

    // Perform eliminations using the 6 bottom rows. A(r, c) = a[r + 3 c].
    let mut a = [0.0; 3 * 13];
    for i in 0..3 {
        a[i] = 0.0;
        a[i + 3 * 4] = 0.0;
        a[i + 3 * 8] = 0.0;
        for k in 0..3 {
            a[i + 3 * (1 + k)] = coeff(&coeffs, 4 + 2 * i, 10 + k);
            a[i + 3 * (5 + k)] = coeff(&coeffs, 4 + 2 * i, 13 + k);
        }
        for k in 0..4 {
            a[i + 3 * (9 + k)] = coeff(&coeffs, 4 + 2 * i, 16 + k);
        }
        for k in 0..3 {
            a[i + 3 * k] -= coeff(&coeffs, 5 + 2 * i, 10 + k);
            a[i + 3 * (4 + k)] -= coeff(&coeffs, 5 + 2 * i, 13 + k);
        }
        for k in 0..4 {
            a[i + 3 * (8 + k)] -= coeff(&coeffs, 5 + 2 * i, 16 + k);
        }
    }

    // Compute degree 10 poly representing determinant (equation 14 in the paper).
    let c = determinant_polynomial(&a);

    // Solve for the roots using sturm bracketing.
    let mut roots = [0.0; 10];
    let n_sols = bisect_sturm(10, &c, &mut roots, DEFAULT_TOL);

    // Back substitution to recover essential matrices.
    let am = |r: usize, col: usize| a[r + 3 * col];
    for &z in &roots[..n_sols] {
        let z2 = z * z;
        let z3 = z2 * z;
        let z4 = z2 * z2;

        let mut b_col0 = [0.0; 3];
        let mut b_col1 = [0.0; 3];
        let mut b = [0.0; 3];
        for r in 0..3 {
            b_col0[r] = am(r, 0) * z3 + am(r, 1) * z2 + am(r, 2) * z + am(r, 3);
            b_col1[r] = am(r, 4) * z3 + am(r, 5) * z2 + am(r, 6) * z + am(r, 7);
            b[r] = am(r, 8) * z4 + am(r, 9) * z3 + am(r, 10) * z2 + am(r, 11) * z + am(r, 12);
        }

        // We try to solve using top two rows.
        let top = Matrix2d::new(b_col0[0], b_col1[0], b_col0[1], b_col1[1]).inverse();
        let mut xz0 = top[(0, 0)] * b[0] + top[(0, 1)] * b[1];
        let mut xz1 = top[(1, 0)] * b[0] + top[(1, 1)] * b[1];

        // If this fails we revert to more expensive QR solver using all three rows.
        if (b_col0[2] * xz0 + b_col1[2] * xz1 - b[2]).abs() > 1e-6 {
            let mut big_b = MatrixXd::zeros(3, 2);
            let mut rhs_b = VectorXd::zeros(3);
            for r in 0..3 {
                big_b[(r, 0)] = b_col0[r];
                big_b[(r, 1)] = b_col1[r];
                rhs_b[r] = b[r];
            }

            let solution = ColPivHouseholderQr::new(&big_b).solve(&rhs_b);
            xz0 = solution[0];
            xz1 = solution[1];
        }

        let x = -xz0;
        let y = -xz1;

        // Since the rows of N are orthogonal unit vectors, we can normalize the
        // coefficients instead.
        let inv_norm = 1.0 / fns::sqrt(x * x + y * y + z * z + 1.0);
        let mut e = [0.0; 9];
        for k in 0..9 {
            e[k] = (n[4 * k] * x + n[4 * k + 1] * y + n[4 * k + 2] * z + n[4 * k + 3]) * inv_norm;
        }

        // Eigen::Map<Matrix<double, 1, 9>>(E.data()): e is E column-major.
        essential_matrices.push(Matrix3d::from_column_major(e));
    }

    n_sols
}

/// Entry (row, col) of the column-major 10 x 20 coefficient matrix.
fn coeff(coeffs: &[f64; 200], row: usize, col: usize) -> f64 {
    coeffs[row + 10 * col]
}

/// PoseLib's `EE(i, j)` macro: the four null-space coefficients `[x, y, z, 1]` of `E(i, j)`.
fn ee(n: &[f64; 36], i: usize, j: usize) -> &[f64] {
    let start = 4 * (3 * j + i);
    &n[start..start + 4]
}

fn set_row(coeffs: &mut [f64; 200], r: usize, row: &[f64; 20]) {
    for (col, &value) in row.iter().enumerate() {
        coeffs[r + 10 * col] = value;
    }
}

/// The ten cubic constraints on the null-space coefficients `[x, y, z, 1]`: rows 0-8 the
/// trace constraint `2 E E^T E - tr(E E^T) E = 0` (equation 22 of the paper, with the factor
/// 1/2 folded in), row 9 `det(E) = 0`. Each row holds the 20 cubic monomials in Nister's
/// order. Port of `compute_trace_constraints`.
fn compute_trace_constraints(n: &[f64; 36], coeffs: &mut [f64; 200]) {
    let mut d = [0.0; 60];

    // Determinant constraint.
    let mut row = [0.0; 20];

    o1(ee(n, 0, 1), ee(n, 1, 2), &mut d);
    o1m(ee(n, 0, 2), ee(n, 1, 1), &mut d);
    o2(&d, ee(n, 2, 0), &mut row);

    o1(ee(n, 0, 2), ee(n, 1, 0), &mut d);
    o1m(ee(n, 0, 0), ee(n, 1, 2), &mut d);
    o2p(&d, ee(n, 2, 1), &mut row);

    o1(ee(n, 0, 0), ee(n, 1, 1), &mut d);
    o1m(ee(n, 0, 1), ee(n, 1, 0), &mut d);
    o2p(&d, ee(n, 2, 2), &mut row);

    set_row(coeffs, 9, &row);

    // EET[i][j] is the offset into d of the (symmetric) entry (i, j) of E E^T:
    // {{d, d + 10, d + 20}, {d + 10, d + 40, d + 30}, {d + 20, d + 30, d + 50}}.
    const EET: [usize; 9] = [0, 10, 20, 10, 40, 30, 20, 30, 50];

    // Compute EE^T (equation 20 in paper).
    for i in 0..3 {
        for j in i..3 {
            let off = EET[3 * i + j];
            let target = &mut d[off..off + 10];
            o1(ee(n, i, 0), ee(n, j, 0), target);
            o1p(ee(n, i, 1), ee(n, j, 1), target);
            o1p(ee(n, i, 2), ee(n, j, 2), target);
        }
    }

    // Subtract trace (equation 22 in paper).
    for i in 0..10 {
        let t = 0.5 * (d[EET[0] + i] + d[EET[4] + i] + d[EET[8] + i]);
        d[EET[0] + i] -= t;
        d[EET[4] + i] -= t;
        d[EET[8] + i] -= t;
    }

    let mut cnt = 0;
    for i in 0..3 {
        for j in 0..3 {
            let slice = |k: usize| &d[EET[3 * i + k]..EET[3 * i + k] + 10];
            o2(slice(0), ee(n, 0, j), &mut row);
            o2p(slice(1), ee(n, 1, j), &mut row);
            o2p(slice(2), ee(n, 2, j), &mut row);
            set_row(coeffs, cnt, &row);
            cnt += 1;
        }
    }
}

// a, b are first order polys [x, y, z, 1]; c is degree 2 poly with order
// [x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1].
fn o1(a: &[f64], b: &[f64], c: &mut [f64]) {
    c[0] = a[0] * b[0];
    c[1] = a[0] * b[1] + a[1] * b[0];
    c[2] = a[0] * b[2] + a[2] * b[0];
    c[3] = a[0] * b[3] + a[3] * b[0];
    c[4] = a[1] * b[1];
    c[5] = a[1] * b[2] + a[2] * b[1];
    c[6] = a[1] * b[3] + a[3] * b[1];
    c[7] = a[2] * b[2];
    c[8] = a[2] * b[3] + a[3] * b[2];
    c[9] = a[3] * b[3];
}

fn o1p(a: &[f64], b: &[f64], c: &mut [f64]) {
    c[0] += a[0] * b[0];
    c[1] += a[0] * b[1] + a[1] * b[0];
    c[2] += a[0] * b[2] + a[2] * b[0];
    c[3] += a[0] * b[3] + a[3] * b[0];
    c[4] += a[1] * b[1];
    c[5] += a[1] * b[2] + a[2] * b[1];
    c[6] += a[1] * b[3] + a[3] * b[1];
    c[7] += a[2] * b[2];
    c[8] += a[2] * b[3] + a[3] * b[2];
    c[9] += a[3] * b[3];
}

fn o1m(a: &[f64], b: &[f64], c: &mut [f64]) {
    c[0] -= a[0] * b[0];
    c[1] -= a[0] * b[1] + a[1] * b[0];
    c[2] -= a[0] * b[2] + a[2] * b[0];
    c[3] -= a[0] * b[3] + a[3] * b[0];
    c[4] -= a[1] * b[1];
    c[5] -= a[1] * b[2] + a[2] * b[1];
    c[6] -= a[1] * b[3] + a[3] * b[1];
    c[7] -= a[2] * b[2];
    c[8] -= a[2] * b[3] + a[3] * b[2];
    c[9] -= a[3] * b[3];
}

// a is second degree poly with order [x^2, x*y, x*z, x, y^2, y*z, y, z^2, z, 1]; b is
// first degree with order [x y z 1]; c is third degree with order (same as Nister's
// paper) [x^3, y^3, x^2*y, x*y^2, x^2*z, x^2, y^2*z, y^2, x*y*z, x*y, x*z^2, x*z, x,
// y*z^2, y*z, y, z^3, z^2, z, 1].
fn o2(a: &[f64], b: &[f64], c: &mut [f64; 20]) {
    c[0] = a[0] * b[0];
    c[1] = a[4] * b[1];
    c[2] = a[0] * b[1] + a[1] * b[0];
    c[3] = a[1] * b[1] + a[4] * b[0];
    c[4] = a[0] * b[2] + a[2] * b[0];
    c[5] = a[0] * b[3] + a[3] * b[0];
    c[6] = a[4] * b[2] + a[5] * b[1];
    c[7] = a[4] * b[3] + a[6] * b[1];
    c[8] = a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
    c[9] = a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
    c[10] = a[2] * b[2] + a[7] * b[0];
    c[11] = a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
    c[12] = a[3] * b[3] + a[9] * b[0];
    c[13] = a[5] * b[2] + a[7] * b[1];
    c[14] = a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
    c[15] = a[6] * b[3] + a[9] * b[1];
    c[16] = a[7] * b[2];
    c[17] = a[7] * b[3] + a[8] * b[2];
    c[18] = a[8] * b[3] + a[9] * b[2];
    c[19] = a[9] * b[3];
}

fn o2p(a: &[f64], b: &[f64], c: &mut [f64; 20]) {
    c[0] += a[0] * b[0];
    c[1] += a[4] * b[1];
    c[2] += a[0] * b[1] + a[1] * b[0];
    c[3] += a[1] * b[1] + a[4] * b[0];
    c[4] += a[0] * b[2] + a[2] * b[0];
    c[5] += a[0] * b[3] + a[3] * b[0];
    c[6] += a[4] * b[2] + a[5] * b[1];
    c[7] += a[4] * b[3] + a[6] * b[1];
    c[8] += a[1] * b[2] + a[2] * b[1] + a[5] * b[0];
    c[9] += a[1] * b[3] + a[3] * b[1] + a[6] * b[0];
    c[10] += a[2] * b[2] + a[7] * b[0];
    c[11] += a[2] * b[3] + a[3] * b[2] + a[8] * b[0];
    c[12] += a[3] * b[3] + a[9] * b[0];
    c[13] += a[5] * b[2] + a[7] * b[1];
    c[14] += a[5] * b[3] + a[6] * b[2] + a[8] * b[1];
    c[15] += a[6] * b[3] + a[9] * b[1];
    c[16] += a[7] * b[2];
    c[17] += a[7] * b[3] + a[8] * b[2];
    c[18] += a[8] * b[3] + a[9] * b[2];
    c[19] += a[9] * b[3];
}
