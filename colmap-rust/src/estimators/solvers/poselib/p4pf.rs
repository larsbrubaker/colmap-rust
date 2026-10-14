//! Port of PoseLib's `PoseLib/solvers/p4pf.cc` (both `poselib::p4pf` overloads; BSD-3-Clause,
//! Copyright (c) 2020 Viktor Larsson; see THIRD_PARTY_NOTICES.md; commit
//! fa7280fee27f97aff31ae7f98bab7f583fac7d08), through colmap-sharp's
//! `Estimators/Solvers/PoseLib/P4pf.cs`: absolute pose with unknown focal length from four
//! 2D-3D correspondences, via a nullspace of the projection constraints and the
//! three-quadratics solver [`super::re3q3`]. Behind COLMAP's `P4PFEstimator`.
//!
//! The method: V. Larsson, Z. Kukelova, Y. Zheng, "Making Minimal Solvers for Absolute Pose
//! Estimation Compact and Robust", ICCV 2017.
//!
//! Translation note: the 8x8 Householder Q is formed through [`HouseholderQr`] on a
//! [`MatrixXd`], the one heap allocation per call besides the output lists.
//!
//! Tier B. Tests: `tests/estimators/rust_only_p4pf.rs` (COLMAP tests it through
//! `P4PFEstimator`).

use super::camera_pose::CameraPose;
use super::re3q3;
use crate::linalg::{HouseholderQr, Matrix3x4d, Matrix4d, MatrixXd, Vector2d, Vector3d, Vector4d};

/// Shared-focal variant: solves with separate fx, fy and reports `(fx + fy) / 2`. With
/// `filter_solutions`, keeps only the solution with the aspect ratio closest to 1 (and only
/// when it is within a factor of 2). Returns the number of poses in `output`. Port of the
/// `poselib::p4pf` overload with one focal output.
///
/// Unlike the overload below, PoseLib does not clear the outputs when filtering; COLMAP
/// always passes fresh vectors. That is kept: with `filter_solutions` the best solution is
/// appended.
pub fn p4pf(
    x: &[Vector2d],
    big_x: &[Vector3d],
    output: &mut Vec<CameraPose>,
    output_focal: &mut Vec<f64>,
    filter_solutions: bool,
) -> usize {
    let mut poses = Vec::with_capacity(8);
    let mut fx = Vec::with_capacity(8);
    let mut fy = Vec::with_capacity(8);
    let n = p4pf_fx_fy(x, big_x, &mut poses, &mut fx, &mut fy, filter_solutions);

    if filter_solutions {
        let mut best_ind = None;
        let mut best_err = 1.0;

        for i in 0..n {
            let a = fx[i] / fy[i];
            // Math.Max / std::max of two non-negative values; a NaN aspect ratio never wins.
            let err = f64::max((a - 1.0).abs(), (1.0 / a - 1.0).abs());
            if err < best_err {
                best_err = err;
                best_ind = Some(i);
            }
        }

        if let Some(best) = best_ind.filter(|_| best_err < 1.0) {
            let focal = (fx[best] + fy[best]) / 2.0;
            output_focal.push(focal);
            output.push(poses[best]);
        }
    } else {
        output.clear();
        output.extend_from_slice(&poses);
        output_focal.clear();
        for i in 0..n {
            output_focal.push((fx[i] + fy[i]) / 2.0);
        }
    }

    output.len()
}

/// Separate-focal variant: estimates the pose and the focal lengths fx, fy. With
/// `filter_solutions`, drops solutions with a point behind the camera. Clears the outputs and
/// returns the number of poses. Port of the `poselib::p4pf` overload with fx and fy outputs.
pub fn p4pf_fx_fy(
    x: &[Vector2d],
    big_x: &[Vector3d],
    output: &mut Vec<CameraPose>,
    output_fx: &mut Vec<f64>,
    output_fy: &mut Vec<f64>,
    filter_solutions: bool,
) -> usize {
    let f0 = (x[0].norm() + x[1].norm() + x[2].norm() + x[3].norm()) / 4.0;
    let mut points2d = [Vector2d::zeros(); 4];
    for i in 0..4 {
        points2d[i] = x[i] / f0;
    }

    // Setup nullspace
    let mut m = MatrixXd::zeros(8, 4);
    for i in 0..4 {
        m[(0, i)] = -points2d[i].y * big_x[i].x;
        m[(2, i)] = -points2d[i].y * big_x[i].y;
        m[(4, i)] = -points2d[i].y * big_x[i].z;
        m[(6, i)] = -points2d[i].y;
        m[(1, i)] = points2d[i].x * big_x[i].x;
        m[(3, i)] = points2d[i].x * big_x[i].y;
        m[(5, i)] = points2d[i].x * big_x[i].z;
        m[(7, i)] = points2d[i].x;
    }

    // d[0..32] is N (8 x 4) and d[32..48] is B (4 x 4), both column-major, as in PoseLib.
    let mut d = [0.0; 48];

    // Compute nullspace using QR
    let q = HouseholderQr::new(&m).householder_q();
    for c in 0..4 {
        for r in 0..8 {
            d[r + 8 * c] = q[(r, 4 + c)];
        }
    }

    // Setup matrices A and B (see paper for definition)
    let mut a = [0.0; 16];
    let mut b = [0.0; 16];
    for i in 0..4 {
        let xi = big_x[i];
        // Row offset 1 reads the y constraint rows of N (1, 3, 5, 7), offset 0 the x rows.
        let use_y = points2d[i].x.abs() < points2d[i].y.abs();
        let s = if use_y { points2d[i].y } else { points2d[i].x };
        let o = usize::from(use_y);
        a[i] = s * xi.x;
        a[i + 4] = s * xi.y;
        a[i + 8] = s * xi.z;
        a[i + 12] = s;
        for k in 0..4 {
            // alpha1, alpha2, alpha3, 1
            b[i + 4 * k] = xi.x * d[o + 8 * k]
                + xi.y * d[o + 2 + 8 * k]
                + xi.z * d[o + 4 + 8 * k]
                + d[o + 6 + 8 * k];
        }
    }

    // [p31,p32,p33,p34] = B * [alpha; 1]
    let b_mat = Matrix4d::from_column_major(a).inverse() * Matrix4d::from_column_major(b);
    d[32..].copy_from_slice(&b_mat.to_column_major());

    let mut coeffs = [0.0; 30];
    let mut solutions = [0.0; 24];
    setup_coefficients(&d, &mut coeffs);

    let n_sols = re3q3::solve(&coeffs, &mut solutions, true);

    output.clear();
    output_fx.clear();
    output_fy.clear();

    let mut pm = [0.0; 12];
    for i in 0..n_sols {
        // P is 3 x 4, column-major: P(r, c) = pm[r + 3 c].
        let alpha = Vector4d::new(
            solutions[3 * i],
            solutions[3 * i + 1],
            solutions[3 * i + 2],
            1.0,
        );
        for c in 0..4 {
            // P.block<2,4>(0,0) = Map<2x4>(N * alpha).
            pm[3 * c] = n_times_alpha(&d, 2 * c, alpha);
            pm[3 * c + 1] = n_times_alpha(&d, 2 * c + 1, alpha);
            // P.row(2) = B * alpha.
            pm[3 * c + 2] = b_mat[(c, 0)] * alpha.x
                + b_mat[(c, 1)] * alpha.y
                + b_mat[(c, 2)] * alpha.z
                + b_mat[(c, 3)] * alpha.w;
        }

        let mut p = Matrix3x4d::from_column_major(pm);
        if p.left_cols3().determinant() < 0.0 {
            p = -p;
        }

        p /= Vector3d::new(p[(2, 0)], p[(2, 1)], p[(2, 2)]).norm();
        let mut fx = Vector3d::new(p[(0, 0)], p[(0, 1)], p[(0, 2)]).norm();
        let mut fy = Vector3d::new(p[(1, 0)], p[(1, 1)], p[(1, 2)]).norm();
        pm = p.to_column_major();
        for c in 0..4 {
            pm[3 * c] /= fx;
            pm[3 * c + 1] /= fy;
        }

        let p = Matrix3x4d::from_column_major(pm);
        let r = p.left_cols3();
        let t = p.col(3);
        fx *= f0;
        fy *= f0;

        let pose = CameraPose::from_rt(r, t);

        if filter_solutions {
            // Check cheirality
            let behind = big_x[..4].iter().any(|&xk| r.row(2).dot(xk) + t.z < 0.0);
            if behind {
                continue;
            }
        }

        output.push(pose);
        output_fx.push(fx);
        output_fy.push(fy);
    }

    output.len()
}

/// Row `row` of N (8 x 4, column-major in `d[0..32]`) times alpha.
fn n_times_alpha(d: &[f64], row: usize, alpha: Vector4d) -> f64 {
    d[row] * alpha.x + d[row + 8] * alpha.y + d[row + 16] * alpha.z + d[row + 24] * alpha.w
}

/// Orthogonality constraints
fn setup_coefficients(d: &[f64; 48], coeffs: &mut [f64; 30]) {
    coeffs[0] = d[0] * d[1] + d[2] * d[3] + d[4] * d[5];
    coeffs[3] =
        d[0] * d[9] + d[1] * d[8] + d[2] * d[11] + d[3] * d[10] + d[4] * d[13] + d[5] * d[12];
    coeffs[6] =
        d[0] * d[17] + d[1] * d[16] + d[2] * d[19] + d[3] * d[18] + d[4] * d[21] + d[5] * d[20];
    coeffs[9] = d[8] * d[9] + d[10] * d[11] + d[12] * d[13];
    coeffs[12] =
        d[8] * d[17] + d[9] * d[16] + d[10] * d[19] + d[11] * d[18] + d[12] * d[21] + d[13] * d[20];
    coeffs[15] = d[16] * d[17] + d[18] * d[19] + d[20] * d[21];
    coeffs[18] =
        d[0] * d[25] + d[1] * d[24] + d[2] * d[27] + d[3] * d[26] + d[4] * d[29] + d[5] * d[28];
    coeffs[21] =
        d[8] * d[25] + d[9] * d[24] + d[10] * d[27] + d[11] * d[26] + d[12] * d[29] + d[13] * d[28];
    coeffs[24] = d[16] * d[25]
        + d[17] * d[24]
        + d[18] * d[27]
        + d[19] * d[26]
        + d[20] * d[29]
        + d[21] * d[28];
    coeffs[27] = d[24] * d[25] + d[26] * d[27] + d[28] * d[29];
    coeffs[1] = d[0] * d[32] + d[2] * d[33] + d[4] * d[34];
    coeffs[4] =
        d[0] * d[36] + d[2] * d[37] + d[8] * d[32] + d[4] * d[38] + d[10] * d[33] + d[12] * d[34];
    coeffs[7] =
        d[0] * d[40] + d[2] * d[41] + d[4] * d[42] + d[16] * d[32] + d[18] * d[33] + d[20] * d[34];
    coeffs[10] = d[8] * d[36] + d[10] * d[37] + d[12] * d[38];
    coeffs[13] = d[8] * d[40]
        + d[10] * d[41]
        + d[16] * d[36]
        + d[12] * d[42]
        + d[18] * d[37]
        + d[20] * d[38];
    coeffs[16] = d[16] * d[40] + d[18] * d[41] + d[20] * d[42];
    coeffs[19] =
        d[0] * d[44] + d[2] * d[45] + d[4] * d[46] + d[24] * d[32] + d[26] * d[33] + d[28] * d[34];
    coeffs[22] = d[8] * d[44]
        + d[10] * d[45]
        + d[12] * d[46]
        + d[24] * d[36]
        + d[26] * d[37]
        + d[28] * d[38];
    coeffs[25] = d[16] * d[44]
        + d[18] * d[45]
        + d[24] * d[40]
        + d[20] * d[46]
        + d[26] * d[41]
        + d[28] * d[42];
    coeffs[28] = d[24] * d[44] + d[26] * d[45] + d[28] * d[46];
    coeffs[2] = d[1] * d[32] + d[3] * d[33] + d[5] * d[34];
    coeffs[5] =
        d[1] * d[36] + d[3] * d[37] + d[9] * d[32] + d[5] * d[38] + d[11] * d[33] + d[13] * d[34];
    coeffs[8] =
        d[1] * d[40] + d[3] * d[41] + d[5] * d[42] + d[17] * d[32] + d[19] * d[33] + d[21] * d[34];
    coeffs[11] = d[9] * d[36] + d[11] * d[37] + d[13] * d[38];
    coeffs[14] = d[9] * d[40]
        + d[11] * d[41]
        + d[17] * d[36]
        + d[13] * d[42]
        + d[19] * d[37]
        + d[21] * d[38];
    coeffs[17] = d[17] * d[40] + d[19] * d[41] + d[21] * d[42];
    coeffs[20] =
        d[1] * d[44] + d[3] * d[45] + d[5] * d[46] + d[25] * d[32] + d[27] * d[33] + d[29] * d[34];
    coeffs[23] = d[9] * d[44]
        + d[11] * d[45]
        + d[13] * d[46]
        + d[25] * d[36]
        + d[27] * d[37]
        + d[29] * d[38];
    coeffs[26] = d[17] * d[44]
        + d[19] * d[45]
        + d[25] * d[40]
        + d[21] * d[46]
        + d[27] * d[41]
        + d[29] * d[42];
    coeffs[29] = d[25] * d[44] + d[27] * d[45] + d[29] * d[46];
}
