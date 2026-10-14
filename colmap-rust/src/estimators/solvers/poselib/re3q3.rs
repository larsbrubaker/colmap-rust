//! Port of PoseLib's `PoseLib/misc/re3q3.cc` (`re3q3`, `refine_3q3`, and the inhomogeneous
//! 3 x 10 overloads of `rotation_to_3q3` and `re3q3_rotation`; BSD-3-Clause, Copyright (c)
//! 2020 Viktor Larsson; see THIRD_PARTY_NOTICES.md; commit
//! fa7280fee27f97aff31ae7f98bab7f583fac7d08, COLMAP 4.2.0's pin), through colmap-sharp's
//! `Estimators/Solvers/PoseLib/Re3q3.cs`. Solves three quadratic equations in three unknowns:
//! PoseLib's re-implementation of Kukelova's E3Q3 (adapted from Jan Heller's implementation)
//! with the elimination-variable choice of Zhou et al., "A Stable Algebraic Camera Pose
//! Estimation for Minimal Configurations of 2D/3D Point and Line Correspondences", ACCV 2018.
//! Used by P4Pf and [`super::gp3p`]; the univariate step goes through [`super::sturm`], the
//! generated elimination expressions live in `re3q3_coeffs.rs` and the random change of
//! variables in `re3q3_var_change.rs`. Tests: `tests/estimators/rust_only_re3q3.rs`.
//!
//! Translation notes:
//! - The 3 x 10 coefficient matrix and the 3 x 8 solution matrix are column-major slices
//!   (`coeffs(k, j) = coeffs[k + 3 j]`), so a call allocates nothing on the heap.
//! - When all three elimination matrices are near-singular, PoseLib retries after a random
//!   affine change of variables drawn from `std::rand()`; here it comes from a fresh
//!   fixed-seed mt19937 (docs/CPP_DIVERGENCES.md entry 220). `re3q3_rotation`'s random
//!   pre-rotation likewise uses a fixed seed (entry 221).
//! - Upstream behavior, kept: a system with a purely linear equation (for example x^2 = 1,
//!   y^2 = 4, z = x + y, which has four finite solutions) returns 0 solutions. An affine
//!   change of variables cannot give a linear equation quadratic terms, so every elimination
//!   matrix keeps a zero row, the retry runs with det = 0, and the inverse yields non-finite
//!   coefficients that Sturm bisection rejects. PoseLib does the same for every `std::srand`
//!   seed colmap-sharp tried; `rust_only_linear_equation_returns_no_solutions_like_poselib`
//!   pins it. Callers (P4Pf) never pass a linear row.
//! - The homogeneous (3 x 9) overloads and `cayley_param` are not ported: no solver COLMAP
//!   calls uses them.
//!
//! Tier B.

use super::camera_pose::{quat_multiply, quat_to_rotmat};
use super::re3q3_coeffs::eliminate;
use super::re3q3_var_change::{solve_with_random_var_change, unit_random_quaternion};
use super::sturm::{bisect_sturm, DEFAULT_TOL};
use crate::linalg::{Matrix3d, Vector3d, Vector4d};
use crate::math::random::Mt19937;

/// The fixed seed of `re3q3_rotation`'s random pre-rotation (docs/CPP_DIVERGENCES.md entry
/// 221). Distinct from the change-of-variables seed so the two draws are not the same
/// rotation.
const ROTATION_PRECONDITION_SEED: u32 = 1;

/// Solve the system `coeffs * [x^2, xy, xz, y^2, yz, z^2, x, y, z, 1]^T = 0` (3 x 10,
/// column-major, 30 values). Writes up to eight solutions as the columns of `solutions`
/// (3 x 8, column-major, 24 values) and returns their count. Port of `poselib::re3q3::re3q3`.
pub fn solve(coeffs: &[f64], solutions: &mut [f64], try_random_var_change: bool) -> usize {
    let ax = Matrix3d::from_columns(col(coeffs, 3), col(coeffs, 5), col(coeffs, 4)); // y^2, z^2, yz
    let ay = Matrix3d::from_columns(col(coeffs, 0), col(coeffs, 5), col(coeffs, 2)); // x^2, z^2, xz
    let az = Matrix3d::from_columns(col(coeffs, 3), col(coeffs, 0), col(coeffs, 1)); // y^2, x^2, yx

    // We check det(A) as a cheaper proxy for condition number
    let mut elim_var = 0;
    let detx = ax.determinant().abs();
    let dety = ay.determinant().abs();
    let detz = az.determinant().abs();
    let mut det = detx;
    if det < dety {
        det = dety;
        elim_var = 1;
    }

    if det < detz {
        det = detz;
        elim_var = 2;
    }

    if try_random_var_change && det < 1e-10 {
        return solve_with_random_var_change(coeffs, solutions);
    }

    // P is 3 x 7, column-major: P(r, c) = p[r + 3 c].
    let (order, elim): ([usize; 7], Matrix3d) = match elim_var {
        // re-order columns to eliminate x (target: y^2 z^2 yz x^2 xy xz x y z 1)
        0 => ([0, 1, 2, 6, 7, 8, 9], ax),
        // re-order columns to eliminate y (target: x^2 z^2 xz y^2 xy yz y x z 1)
        1 => ([3, 1, 4, 7, 6, 8, 9], ay),
        // re-order columns to eliminate z (target: y^2 x^2 yx z^2 zy z y x 1)
        _ => ([5, 4, 2, 8, 7, 6, 9], az),
    };
    let neg_inv = -elim.inverse();
    let mut p = [0.0; 21];
    for (k, &j) in order.iter().enumerate() {
        let c = neg_inv * col(coeffs, j);
        p[3 * k] = c.x;
        p[3 * k + 1] = c.y;
        p[3 * k + 2] = c.z;
    }

    let e = eliminate(&p);
    let [a11, a12, a13, a14, a15, a16, a17, a18, a19, a110] = e.a1;
    let [a21, a22, a23, a24, a25, a26, a27, a28, a29, a210] = e.a2;

    let mut roots = [0.0; 8];
    let n_roots = bisect_sturm(8, &e.c, &mut roots, DEFAULT_TOL);

    for (i, &xs1) in roots.iter().enumerate().take(n_roots) {
        let xs2 = xs1 * xs1;
        let xs3 = xs1 * xs2;

        let m00 = a11 * xs2 + a12 * xs1 + a13;
        let m01 = a14 * xs2 + a15 * xs1 + a16;
        let m02 = a17 * xs3 + a18 * xs2 + a19 * xs1 + a110;
        let m10 = a21 * xs2 + a22 * xs1 + a23;
        let m11 = a24 * xs2 + a25 * xs1 + a26;
        let m12 = a27 * xs3 + a28 * xs2 + a29 * xs1 + a210;
        // The third row of PoseLib's A is filled but never read.

        solutions[3 * i] = xs1;
        solutions[3 * i + 1] = (m12 * m01 - m02 * m11) / (m00 * m11 - m10 * m01);
        solutions[3 * i + 2] = (m12 * m00 - m02 * m10) / (m01 * m10 - m11 * m00);
    }

    if elim_var == 1 {
        swap_rows(solutions, 0, 1);
    } else if elim_var == 2 {
        swap_rows(solutions, 0, 2);
    }

    refine_3q3(coeffs, solutions, n_roots);

    n_roots
}

/// Inhomogeneous linear constraints on a rotation matrix, `rcoeffs * [R(:); 1] = 0` (3 x 10,
/// column-major, `R(:)` column-major), converted into a 3q3 problem in the Cayley
/// parameters, written to `coeffs` (3 x 10). Port of the 3 x 10 overload of
/// `poselib::re3q3::rotation_to_3q3`.
pub fn rotation_to_3q3(rcoeffs: &[f64], coeffs: &mut [f64]) {
    for k in 0..3 {
        let r0 = rcoeffs[k];
        let r1 = rcoeffs[k + 3];
        let r2 = rcoeffs[k + 6];
        let r3 = rcoeffs[k + 9];
        let r4 = rcoeffs[k + 12];
        let r5 = rcoeffs[k + 15];
        let r6 = rcoeffs[k + 18];
        let r7 = rcoeffs[k + 21];
        let r8 = rcoeffs[k + 24];
        let r9 = rcoeffs[k + 27];
        coeffs[k] = r0 - r4 - r8 + r9;
        coeffs[k + 3] = 2.0 * r1 + 2.0 * r3;
        coeffs[k + 6] = 2.0 * r2 + 2.0 * r6;
        coeffs[k + 9] = r4 - r0 - r8 + r9;
        coeffs[k + 12] = 2.0 * r5 + 2.0 * r7;
        coeffs[k + 15] = r8 - r4 - r0 + r9;
        coeffs[k + 18] = 2.0 * r5 - 2.0 * r7;
        coeffs[k + 21] = 2.0 * r6 - 2.0 * r2;
        coeffs[k + 24] = 2.0 * r1 - 2.0 * r3;
        coeffs[k + 27] = r0 + r4 + r8 + r9;
    }
}

/// Solve `rcoeffs * [R(:); 1] = 0` (3 x 10, column-major) for rotations R. Writes up to eight
/// unit quaternions (PoseLib order, real part first) to `solutions` and returns their count.
/// PoseLib first rotates the problem by a random rotation R0 so that no solution sits at the
/// Cayley transform's singularity (a rotation by pi), solves in Cayley parameters, and
/// composes R0 back. Port of the 3 x 10 overload of `poselib::re3q3::re3q3_rotation`
/// (`re3q3_rotation_impl`). R0 comes from a fixed-seed mt19937 rather than `std::rand()`
/// (docs/CPP_DIVERGENCES.md entry 221).
pub fn solve_rotation(
    rcoeffs: &[f64],
    solutions: &mut [Vector4d],
    try_random_var_change: bool,
) -> usize {
    // PoseLib reads UnitRandom().coeffs(), which is Eigen's (x, y, z, w) memory order, as a
    // (w, x, y, z) vector. The draw is uniform on the unit sphere either way, so the
    // component order does not change the distribution; the port keeps PoseLib's reading.
    let draw = unit_random_quaternion(&mut Mt19937::new(ROTATION_PRECONDITION_SEED));
    let q0 = Vector4d::new(draw.x, draw.y, draw.z, draw.w);
    let r0 = quat_to_rotmat(q0);

    // Rcoeffs.block<3, 3>(0, 3 b) = Rcoeffs.block<3, 3>(0, 3 b) * R0 for b = 0, 1, 2.
    let mut rotated = [0.0; 30];
    rotated.copy_from_slice(&rcoeffs[..30]);
    for b in 0..3 {
        for row in 0..3 {
            for c in 0..3 {
                rotated[row + 3 * (3 * b + c)] = rcoeffs[row + 3 * (3 * b)] * r0[(0, c)]
                    + rcoeffs[row + 3 * (3 * b + 1)] * r0[(1, c)]
                    + rcoeffs[row + 3 * (3 * b + 2)] * r0[(2, c)];
            }
        }
    }

    let mut coeffs = [0.0; 30];
    rotation_to_3q3(&rotated, &mut coeffs);

    let mut solutions_cayley = [0.0; 24];
    let n_sols = solve(&coeffs, &mut solutions_cayley, try_random_var_change);

    for (i, sol) in solutions.iter_mut().enumerate().take(n_sols) {
        let q = Vector4d::new(
            1.0,
            solutions_cayley[3 * i],
            solutions_cayley[3 * i + 1],
            solutions_cayley[3 * i + 2],
        )
        .normalized();
        *sol = quat_multiply(q0, q);
    }

    n_sols
}

/// Column `j` of a 3 x N column-major matrix.
pub(super) fn col(m: &[f64], j: usize) -> Vector3d {
    Vector3d::new(m[3 * j], m[3 * j + 1], m[3 * j + 2])
}

/// Eigen's `row(a).swap(row(b))` over all 8 columns of the 3 x 8 solution matrix.
fn swap_rows(solutions: &mut [f64], r0: usize, r1: usize) {
    for j in 0..8 {
        solutions.swap(r0 + 3 * j, r1 + 3 * j);
    }
}

/// Port of `poselib::re3q3::refine_3q3`: up to five Newton steps per solution.
pub(super) fn refine_3q3(coeffs: &[f64], solutions: &mut [f64], n_sols: usize) {
    for i in 0..n_sols {
        let mut x = solutions[3 * i];
        let mut y = solutions[3 * i + 1];
        let mut z = solutions[3 * i + 2];

        // [x^2, x*y, x*z, y^2, y*z, z^2, x, y, z, 1.0]
        for _ in 0..5 {
            let r = col(coeffs, 0) * x * x
                + col(coeffs, 1) * x * y
                + col(coeffs, 2) * x * z
                + col(coeffs, 3) * y * y
                + col(coeffs, 4) * y * z
                + col(coeffs, 5) * z * z
                + col(coeffs, 6) * x
                + col(coeffs, 7) * y
                + col(coeffs, 8) * z
                + col(coeffs, 9);

            let r_abs = r.cwise_abs();
            // maxCoeff() < 1e-8, written so that a NaN residual never stops the iteration
            // (as with C#'s NaN-propagating Math.Max).
            if r_abs.x < 1e-8 && r_abs.y < 1e-8 && r_abs.z < 1e-8 {
                break;
            }

            let j = Matrix3d::from_columns(
                2.0 * col(coeffs, 0) * x + col(coeffs, 1) * y + col(coeffs, 2) * z + col(coeffs, 6),
                col(coeffs, 1) * x + 2.0 * col(coeffs, 3) * y + col(coeffs, 4) * z + col(coeffs, 7),
                col(coeffs, 2) * x + col(coeffs, 4) * y + 2.0 * col(coeffs, 5) * z + col(coeffs, 8),
            );

            let dx = j.inverse() * r;

            x -= dx.x;
            y -= dx.y;
            z -= dx.z;
        }

        solutions[3 * i] = x;
        solutions[3 * i + 1] = y;
        solutions[3 * i + 2] = z;
    }
}
