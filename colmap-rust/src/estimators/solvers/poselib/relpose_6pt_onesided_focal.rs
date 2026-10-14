//! Port of PoseLib's `PoseLib/solvers/relpose_6pt_onesided_focal.cc`
//! (`poselib::relpose_6pt_onesided_focal` with `use_elim = false`; BSD-3-Clause, Copyright (c)
//! 2020 Viktor Larsson; see THIRD_PARTY_NOTICES.md; commit
//! fa7280fee27f97aff31ae7f98bab7f583fac7d08), through colmap-sharp's
//! `Estimators/Solvers/PoseLib/Relpose6ptOnesidedFocal.cs`: the minimal relative pose solver
//! from 6 correspondences when the first view has one unknown focal length and the second
//! view is calibrated, as an automatically generated elimination-template (action matrix)
//! solver. COLMAP's `RelativePoseOneSidedFocalEstimator`
//! (`estimators/solvers/relpose_one_sided_focal.cc`) is its only caller. The generated parts
//! are split out: `relpose_6pt_onesided_focal_coeffs1..4.rs` (the 190 coefficients) and
//! `relpose_6pt_onesided_focal_template.rs` (the elimination template fill).
//!
//! Only the full template (`use_elim = false`) is ported: it is the only one COLMAP calls, as
//! it solves for the focal directly. The compact template (`solver_..._elim`) and the Kruppa
//! focal recovery it needs (`onesided_focal_from_fundamental`) are left for the caller that
//! first needs them.
//!
//! Pipeline, as in PoseLib: the view-1 points are rescaled by their mean magnitude; the 3D
//! null space N of the six epipolar constraints parametrizes `F = N0 + x N1 + y N2`; the
//! coefficients fill the template `[C0 | C1]`, `C2 = C0^-1 C1` (partial-pivot LU), and the
//! 9 x 9 action matrix is read off C2. Its eigenvalues give x; y and `q = 1/f^2` are
//! eigenvector ratios, so the eigenvector phase freedom of [`crate::linalg::EigenSolver`]
//! cancels. Each real solution becomes `E = F diag(1, 1, 1/f)` and is decomposed by
//! [`super::essential::motion_from_essential`].
//!
//! The null space comes from an unpivoted Householder QR instead of PoseLib's
//! `fullPivHouseholderQr` (docs/CPP_DIVERGENCES.md entry 255). The order of the eigenvalues
//! (Schur order) sets the order of the returned poses, which may differ from Eigen's; COLMAP
//! treats the output as a set. Tier B. Tests: `tests/estimators/rust_only_relpose_6pt_focal.rs`.

use super::camera_pose::{CameraPose, ImagePair};
use super::essential::motion_from_essential;
use super::poselib_camera::PoseLibCamera;
use super::relpose_6pt_onesided_focal_coeffs1::compute_coefficients1;
use super::relpose_6pt_onesided_focal_coeffs2::compute_coefficients2;
use super::relpose_6pt_onesided_focal_coeffs3::compute_coefficients3;
use super::relpose_6pt_onesided_focal_coeffs4::compute_coefficients4;
use super::relpose_6pt_onesided_focal_template::{C0_COEFFS, C0_IND, C1_COEFFS, C1_IND};
use super::relpose_6pt_shared_focal::{fundamental_from_null_space, null_space3};
use crate::linalg::{Complex, EigenSolver, Matrix3d, MatrixXd, PartialPivLu, Vector3d};
use crate::math::fns;

/// PoseLib's `AM_rows`: the rows of `RR = [-C2.bottomRows(5); I9]` forming the action matrix.
const ACTION_MATRIX_ROWS: [usize; 9] = [9, 7, 1, 0, 10, 4, 3, 11, 2];

/// Estimates the relative pose and the first view's focal from the first 6 correspondences:
/// `x1` homogeneous, principal-point-centered pixels of the uncalibrated view, `x2` bearings
/// of the calibrated one. Clears `out_image_pairs`, appends one [`ImagePair`] per
/// cheirality-consistent pose (camera 1 SIMPLE_PINHOLE with the focal in the pixel units of
/// `x1`, camera 2 the unit SIMPLE_PINHOLE) and returns their count. Port of
/// `poselib::relpose_6pt_onesided_focal`.
pub fn relpose_6pt_onesided_focal(
    x1: &[Vector3d],
    x2: &[Vector3d],
    out_image_pairs: &mut Vec<ImagePair>,
) -> usize {
    // The solver works on rescaled pixel coordinates in the uncalibrated image: dividing by
    // the average magnitude brings the effective focal length close to one, which keeps the
    // polynomial system well-conditioned for pixel-scale input. The focal is scaled back when
    // constructing the output cameras.
    let mut scale = 0.0;
    let mut x1s = [Vector3d::zeros(); 6];
    for i in 0..6 {
        x1s[i] = x1[i] / x1[i].z;
        scale += fns::sqrt(x1s[i].x * x1s[i].x + x1s[i].y * x1s[i].y);
    }

    scale /= 6.0;
    for v in &mut x1s {
        *v = Vector3d::new(v.x / scale, v.y / scale, v.z);
    }

    // Compute nullspace to the epipolar constraints (3-dimensional). The fundamental matrix
    // is parameterized as F = N.col(0) + x1 * N.col(1) + x2 * N.col(2).
    let mut epipolar_constraints = [0.0; 9 * 6];
    for i in 0..6 {
        let col = &mut epipolar_constraints[9 * i..9 * i + 9];
        for k in 0..3 {
            for r in 0..3 {
                col[3 * k + r] = x1s[i][k] * x2[i][r];
            }
        }
    }

    let b = null_space3(&mut epipolar_constraints);

    let sols = solve_full_template(&b);

    out_image_pairs.clear();

    let mut poses: Vec<CameraPose> = Vec::with_capacity(4);
    let mut x1u = [Vector3d::zeros(); 6];
    let mut x2u = [Vector3d::zeros(); 6];
    for sol in &sols {
        // x1 and x2 (rows 0 and 1) must be real.
        if sol[0].im.abs() > 1e-8 || sol[1].im.abs() > 1e-8 {
            continue;
        }

        let f = fundamental_from_null_space(&b, sol[0].re, sol[1].re);

        // The full template solves for q = 1/f^2 (monomial halving) as the third unknown.
        if sol[2].im.abs() > 1e-8 || sol[2].re <= 0.0 {
            continue;
        }

        let focal = 1.0 / fns::sqrt(sol[2].re);
        if !focal.is_finite() || focal <= 0.0 {
            continue;
        }

        // Camera 1 is the uncalibrated camera (K1 = diag(f, f, 1), x1 are pixel points, so
        // diag(1, 1, f) * x1 is the calibrated bearing); camera 2 is calibrated. The solved
        // focal lives in the rescaled coordinates; undo the rescaling.
        let focal_px = focal * scale;

        let e = f * Matrix3d::from_diagonal(Vector3d::new(1.0, 1.0, 1.0 / focal));

        for j in 0..6 {
            x1u[j] = Vector3d::new(x1s[j].x, x1s[j].y, x1s[j].z * focal).normalized();
            x2u[j] = x2[j].normalized();
        }

        poses.clear();
        motion_from_essential(&e, &x1u, &x2u, &mut poses);

        for pose in &poses {
            out_image_pairs.push(ImagePair::new(
                *pose,
                PoseLibCamera::new(0, &[focal_px, 0.0, 0.0], -1, -1),
                PoseLibCamera::new(0, &[1.0, 0.0, 0.0], -1, -1),
            ));
        }
    }

    out_image_pairs.len()
}

/// PoseLib's `solver_relpose_6pt_onesided_focal_full`: the full action-matrix template (190
/// coefficients, 19 x 19 elimination, 9 solutions). Returns the 9 solutions `(x, y, q)`.
fn solve_full_template(data: &[f64; 27]) -> [[Complex; 3]; 9] {
    let mut coeffs = [0.0; 190];
    compute_coefficients1(data, &mut coeffs);
    compute_coefficients2(data, &mut coeffs);
    compute_coefficients3(data, &mut coeffs);
    compute_coefficients4(data, &mut coeffs);

    let mut c0 = MatrixXd::zeros(19, 19);
    let mut c1 = MatrixXd::zeros(19, 9);
    for (&dst, &src) in C0_IND.iter().zip(&C0_COEFFS) {
        c0.as_mut_slice()[dst] = coeffs[src];
    }
    for (&dst, &src) in C1_IND.iter().zip(&C1_COEFFS) {
        c1.as_mut_slice()[dst] = coeffs[src];
    }

    let c2 = PartialPivLu::new(&c0).solve_matrix(&c1);

    // RR = [-C2.bottomRows(5); I9] (14 x 9), AM.row(i) = RR.row(AM_rows[i]).
    let mut am = MatrixXd::zeros(9, 9);
    for (i, &rr) in ACTION_MATRIX_ROWS.iter().enumerate() {
        for c in 0..9 {
            am[(i, c)] = if rr < 5 {
                -c2[(14 + rr, c)]
            } else if rr - 5 == c {
                1.0
            } else {
                0.0
            };
        }
    }

    let es = EigenSolver::new(&am, true);
    let eigs = es.eigenvalues();
    let v = es.eigenvectors();

    let mut sols = [[Complex::ZERO; 3]; 9];
    for (j, sol) in sols.iter_mut().enumerate() {
        let scale = v[(0, j)];
        sol[0] = eigs[j]; // x1
        sol[1] = v[(7, j)] / scale; // x2
        sol[2] = v[(1, j)] / scale; // focal variable
    }

    sols
}
