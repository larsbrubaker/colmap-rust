//! Port of PoseLib's `PoseLib/solvers/relpose_6pt_focal.cc` (`poselib::relpose_6pt_shared_focal`;
//! BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see THIRD_PARTY_NOTICES.md; commit
//! fa7280fee27f97aff31ae7f98bab7f583fac7d08), through colmap-sharp's
//! `Estimators/Solvers/PoseLib/Relpose6ptSharedFocal.cs`: the minimal relative pose solver for
//! two views sharing one unknown focal length, from 6 correspondences, as an automatically
//! generated elimination-template (action matrix) solver. COLMAP's
//! `RelativePoseSharedFocalEstimator` (`estimators/solvers/relpose_shared_focal.cc`) is its
//! only caller. The generated parts are split out: `relpose_6pt_shared_focal_coeffs1..4.rs`
//! (the 280 coefficients) and `relpose_6pt_shared_focal_template.rs` (the elimination template
//! indices).
//!
//! Pipeline, as in PoseLib: the 3D null space N of the six epipolar constraints parametrizes
//! `F = N0 + x N1 + y N2`; the coefficients are scattered into the template `[C0 | C1]`,
//! `C12 = C0^-1 C1` (partial-pivot LU), the 15 x 15 action matrix is read off C12, its
//! characteristic polynomial (Danilevsky, [`super::sturm`]) has its real roots bracketed by
//! Sturm sequences, and the other unknowns come from the structured back-substitution of
//! `shared_focal_relpose_fast_eigenvector_solver` (a 7 x 7 Householder QR solve per root). Each
//! F with a positive `q = 1/f^2` becomes `E = K F K` and is decomposed by
//! [`super::essential::motion_from_essential`].
//!
//! The null space comes from an unpivoted Householder QR instead of PoseLib's
//! `fullPivHouseholderQr`: same space, different basis, the same solution set in exact
//! arithmetic, though spurious real roots of the ill-conditioned degree-15 polynomial can
//! appear or vanish with the basis. Tier B. Tests:
//! `tests/estimators/rust_only_relpose_6pt_focal.rs`.
//!
//! That is a deliberate divergence, shared with [`super::relpose_6pt_onesided_focal`]. PoseLib's
//! `relpose_6pt_shared_focal` and `relpose_6pt_onesided_focal` (the minimal solvers of COLMAP's
//! `RelativePoseSharedFocalEstimator` and `RelativePoseOneSidedFocalEstimator`) take the 3D null
//! space of their six epipolar constraints from the last three columns of the full Q of
//! `epipolar_constraints.fullPivHouseholderQr()` (a 9 x 6 matrix). `null_space3`, used by both,
//! takes the same three columns from an unpivoted Householder QR
//! (`linalg::householder::factor_in_place` / `apply_q`). Both are orthonormal bases of the same
//! space; only the basis inside it differs, and with it the 280 (shared) or 190 (one-sided)
//! template coefficients, the action matrix and its eigenvalues, which parametrize
//! F = N0 + x N1 + y N2 in basis coordinates. The one-sided solver also orders its nine
//! solutions by the eigenvalue order of `linalg::EigenSolver` (Schur order), which may differ
//! from Eigen's. Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no
//! published convention precise enough to reproduce Eigen's Q column for column (the
//! five-point solver makes the same choice, [`super::relpose_5pt`]). In exact arithmetic the
//! solution set does not depend on the basis (apart from solutions whose F has no N0
//! component, which the affine parametrization cannot represent; a measure-zero configuration
//! that moves with the basis). In floating point it can: this solver brackets the real roots
//! of an ill-conditioned degree-15 characteristic polynomial with Sturm sequences, and which
//! spurious (non-physical) real roots survive that bracketing depends on the basis, so extra
//! models can appear or vanish on either side. The output order is not part of the contract:
//! COLMAP treats the models as a set. What can differ is numerical conditioning near the
//! solvers' thresholds (the Sturm bracketing tolerance, the 1e-8 tests on imaginary parts and on
//! q = 1/f^2). In `tests/estimators/rust_only_relpose_6pt_focal.rs`, on 50 random minimal
//! problems each, both solvers return the true pose and focal (relative error below 1e-4;
//! well-conditioned problems agree to 1e-12, one shared-focal problem was 5e-6 off).
//! colmap-sharp's differential run against the real PoseLib (its entry 30) found the one-sided
//! models identical and the shared-focal models matching except for spurious roots. Not
//! compared against C++ PoseLib here. Same as colmap-sharp entry 30.

use super::camera_pose::{CameraPose, ImagePair};
use super::essential::motion_from_essential;
use super::poselib_camera::PoseLibCamera;
use super::relpose_6pt_shared_focal_coeffs1::compute_coefficients1;
use super::relpose_6pt_shared_focal_coeffs2::compute_coefficients2;
use super::relpose_6pt_shared_focal_coeffs3::compute_coefficients3;
use super::relpose_6pt_shared_focal_coeffs4::compute_coefficients4;
use super::relpose_6pt_shared_focal_template::{C0_IND, C1_IND, COEFFS0_IND, COEFFS1_IND};
use super::sturm::{bisect_sturm, charpoly_danilevsky_piv};
use crate::linalg::householder::{apply_q, factor_in_place};
use crate::linalg::{HouseholderQr, Matrix3d, MatrixXd, PartialPivLu, Vector3d, VectorXd};
use crate::math::fns;

/// PoseLib's `AM_ind`: the rows of `RR = [-C12.bottomRows(8); I]` forming the action matrix.
const ACTION_MATRIX_ROWS: [usize; 15] = [15, 11, 0, 1, 2, 12, 3, 16, 4, 5, 17, 6, 18, 19, 7];

/// The non-trivial rows of the action matrix (the fast eigenvector solver's `ind`).
const NON_TRIVIAL_ROWS: [usize; 8] = [2, 3, 4, 6, 8, 9, 11, 14];

/// Estimates the relative pose and shared focal from the first 6 correspondences
/// (`x2^T F x1 = 0`, homogeneous image points of both views with the principal point at the
/// origin). Clears `out_image_pairs`, appends one [`ImagePair`] per cheirality-consistent pose
/// (both cameras SIMPLE_PINHOLE with the recovered focal) and returns their count. Port of
/// `poselib::relpose_6pt_shared_focal`.
pub fn relpose_6pt_shared_focal(
    x1: &[Vector3d],
    x2: &[Vector3d],
    out_image_pairs: &mut Vec<ImagePair>,
) -> usize {
    // Compute nullspace to epipolar constraints. Column i holds x1[i](k) * x2[i](r) at
    // 3k + r, the coefficient of F(r, k) in column-major order.
    let mut epipolar_constraints = [0.0; 9 * 6];
    for i in 0..6 {
        let col = &mut epipolar_constraints[9 * i..9 * i + 9];
        for k in 0..3 {
            for r in 0..3 {
                col[3 * k + r] = x1[i][k] * x2[i][r];
            }
        }
    }

    // B = N (9 x 3, column-major): the last three columns of the full Q.
    let b = null_space3(&mut epipolar_constraints);

    let mut sols = [0.0; 3 * 15];
    let n_sols = solve_template(&b, &mut sols);

    out_image_pairs.clear();

    let mut poses: Vec<CameraPose> = Vec::with_capacity(4);
    let mut x1u = [Vector3d::zeros(); 6];
    let mut x2u = [Vector3d::zeros(); 6];
    for i in 0..n_sols {
        // The real-root path only produces real solutions, so PoseLib's imaginary-part test
        // (sols.col(i).imag().norm() > 1e-8) never rejects one.
        if sols[3 * i + 2] < 1e-8 {
            continue;
        }

        let focal = fns::sqrt(1.0 / sols[3 * i + 2]);
        let f = fundamental_from_null_space(&b, sols[3 * i], sols[3 * i + 1]);

        let k = Matrix3d::from_diagonal(Vector3d::new(focal, focal, 1.0));
        let e = k * (f * k);

        for j in 0..6 {
            x1u[j] = Vector3d::new(x1[j].x / focal, x1[j].y / focal, x1[j].z).normalized();
            x2u[j] = Vector3d::new(x2[j].x / focal, x2[j].y / focal, x2[j].z).normalized();
        }

        poses.clear();
        motion_from_essential(&e, &x1u, &x2u, &mut poses);

        for pose in &poses {
            let camera = PoseLibCamera::new(0, &[focal, 0.0, 0.0], -1, -1);
            out_image_pairs.push(ImagePair::new(*pose, camera.clone(), camera));
        }
    }

    out_image_pairs.len()
}

/// `F = N0 + x N1 + y N2` from the column-major 9 x 3 null space `b`, normalized to unit
/// Frobenius norm (unless zero), read column-major like `Eigen::Matrix3d(F_vector.data())`.
pub(super) fn fundamental_from_null_space(b: &[f64; 27], x: f64, y: f64) -> Matrix3d {
    let mut f_vector = [0.0; 9];
    let mut squared_norm = 0.0;
    for k in 0..9 {
        f_vector[k] = b[k] + x * b[9 + k] + y * b[18 + k];
        squared_norm += f_vector[k] * f_vector[k];
    }

    if squared_norm > 0.0 {
        let norm = fns::sqrt(squared_norm);
        for v in &mut f_vector {
            *v /= norm;
        }
    }

    Matrix3d::from_columns(
        Vector3d::new(f_vector[0], f_vector[1], f_vector[2]),
        Vector3d::new(f_vector[3], f_vector[4], f_vector[5]),
        Vector3d::new(f_vector[6], f_vector[7], f_vector[8]),
    )
}

/// The last three columns of the full Q of the 9 x 6 column-major matrix `a` (destroyed),
/// column-major: an orthonormal basis of the space orthogonal to a's columns.
pub(super) fn null_space3(a: &mut [f64; 54]) -> [f64; 27] {
    let mut tau = [0.0; 6];
    factor_in_place(a, 9, 6, &mut tau);
    let mut n = [0.0; 27];
    for j in 0..3 {
        let q = &mut n[9 * j..9 * j + 9];
        q[6 + j] = 1.0;
        apply_q(a, 9, &tau, q);
    }

    n
}

/// PoseLib's `solver_shared_focal_relpose_6pt`: fills `sols` (3 x 15, column-major; real
/// parts only, see [`relpose_6pt_shared_focal`]) and returns the number of real roots.
fn solve_template(data: &[f64; 27], sols: &mut [f64; 45]) -> usize {
    // Compute coefficients
    let mut coeffs = [0.0; 280];
    compute_coefficients1(data, &mut coeffs);
    compute_coefficients2(data, &mut coeffs);
    compute_coefficients3(data, &mut coeffs);
    compute_coefficients4(data, &mut coeffs);

    // Setup elimination template
    let mut c0 = MatrixXd::zeros(31, 31);
    let mut c1 = MatrixXd::zeros(31, 15);
    for (&dst, &src) in C0_IND.iter().zip(&COEFFS0_IND) {
        c0.as_mut_slice()[dst] = coeffs[src];
    }
    for (&dst, &src) in C1_IND.iter().zip(&COEFFS1_IND) {
        c1.as_mut_slice()[dst] = coeffs[src];
    }

    let c12 = PartialPivLu::new(&c0).solve_matrix(&c1);

    // Setup action matrix: RR = [-C12.bottomRows(8); I15] (23 x 15), AM.row(i) =
    // RR.row(AM_ind[i]). am is column-major 15 x 15.
    let mut am = [0.0; 15 * 15];
    for (i, &rr) in ACTION_MATRIX_ROWS.iter().enumerate() {
        for c in 0..15 {
            am[i + 15 * c] = if rr < 8 {
                -c12[(23 + rr, c)]
            } else if rr - 8 == c {
                1.0
            } else {
                0.0
            };
        }
    }

    *sols = [0.0; 45];

    // Solve eigenvalue problem: the real eigenvalues are the real roots of the
    // characteristic polynomial.
    let mut p = [0.0; 1 + 15];
    let mut amp = am;
    charpoly_danilevsky_piv(&mut amp, 15, &mut p);
    let mut eigv = [0.0; 15];
    let nroots = bisect_sturm(15, &p, &mut eigv, 1e-12);

    fast_eigenvector_solver(&eigv[..nroots], &am, sols);

    nroots
}

/// PoseLib's `shared_focal_relpose_fast_eigenvector_solver`.
fn fast_eigenvector_solver(eigv: &[f64], am: &[f64; 225], sols: &mut [f64; 45]) {
    // Truncated action matrix containing non-trivial rows (8 x 15, row accessor).
    let at = |r: usize, c: usize| am[NON_TRIVIAL_ROWS[r] + 15 * c];

    let mut aa = MatrixXd::zeros(7, 7);
    let mut rhs = VectorXd::zeros(7);
    let mut col = [0.0; 8];
    for (i, &ev) in eigv.iter().enumerate() {
        let mut zi = [ev, 0.0, 0.0];
        for j in 1..3 {
            zi[j] = zi[j - 1] * ev;
        }

        // AA (8 x 8) column by column; only its top 7 rows are used.
        for c in 0..8 {
            for (r, v) in col.iter_mut().enumerate() {
                *v = match c {
                    0 => at(r, 2),
                    1 => at(r, 6),
                    2 => zi[0] * at(r, 4) + at(r, 5),
                    3 => at(r, 1) + zi[0] * at(r, 3),
                    4 => at(r, 14),
                    5 => zi[0] * at(r, 11) + at(r, 13),
                    6 => zi[1] * at(r, 9) + zi[0] * at(r, 10) + at(r, 12),
                    _ => at(r, 0) + zi[0] * at(r, 7) + zi[1] * at(r, 8),
                };
            }

            match c {
                0 => col[0] -= zi[0],
                1 => col[3] -= zi[0],
                2 => col[2] -= zi[1],
                3 => col[1] -= zi[1],
                4 => col[7] -= zi[0],
                5 => col[6] -= zi[1],
                6 => col[5] -= zi[2],
                _ => col[4] -= zi[2],
            }

            for r in 0..7 {
                if c < 7 {
                    aa[(r, c)] = col[r];
                } else {
                    rhs[r] = -col[r];
                }
            }
        }

        // Using column pivoting leads to unstable numerics
        let s = HouseholderQr::new(&aa).solve(&rhs);

        sols[3 * i] = s[3];
        sols[3 * i + 1] = zi[0];
        sols[3 * i + 2] = s[6];
    }
}
