//! Port of COLMAP's `colmap/geometry/homography_matrix.h` and `homography_matrix.cc`: the
//! analytical decomposition of a calibrated homography into its (up to four) rotation,
//! translation and plane-normal candidates (Malis and Vargas, "Deeper understanding of the
//! homography decomposition for vision-based control", INRIA RR-6303, 2007), picking the
//! candidate by cheirality and reprojection error, building H from a pose and plane, and the
//! homography transfer error. Port of colmap-sharp's `Geometry/HomographyMatrix.cs`. Builds
//! on [`super::triangulation`] (`triangulate_mid_point`), [`super::rigid3`] and
//! [`crate::math::sign_of_number`]. Tests: `tests/geometry/homography_matrix.rs`
//! (homography_matrix_test.cc) and `tests/geometry/rust_only_two_view_oracle.rs`.
//!
//! Tiers: `homography_matrix_from_pose` and `compute_squared_homography_error` are scalar
//! products. `decompose_homography_matrix` scales H by its middle singular value
//! (sign-free: singular values are non-negative) and is otherwise closed form, so it matches
//! COLMAP to rounding, Tier B. `pose_from_homography_matrix` adds `triangulate_mid_point`
//! (Tier B, its SVD null vector enters only through a ratio).
//!
//! For noise-free correspondences of a plane, two of the four candidates are usually both
//! physically valid: every point triangulates in front of both cameras with a reprojection
//! sum at rounding level (~1e-16). `pose_from_homography_matrix` then picks between them by
//! rounding, in COLMAP as here, so the two can disagree on such exact data
//! (docs/CPP_DIVERGENCES.md entry 86); with any noise the choice is well defined and matches
//! (the oracle test).

use super::rigid3::Rigid3d;
use super::triangulation::triangulate_mid_point;
use crate::linalg::{Matrix3d, Quaterniond, Svd3d, Vector2d, Vector3d};
use crate::math::{fns, sign_of_number};
use crate::{check_eq, check_gt, Result};

fn compute_opposite_of_minor(matrix: &Matrix3d, row: usize, col: usize) -> f64 {
    let col1 = if col == 0 { 1 } else { 0 };
    let col2 = if col == 2 { 1 } else { 2 };
    let row1 = if row == 0 { 1 } else { 0 };
    let row2 = if row == 2 { 1 } else { 2 };
    matrix[(row1, col2)] * matrix[(row2, col1)] - matrix[(row1, col1)] * matrix[(row2, col2)]
}

fn compute_homography_rotation(
    h_normalized: &Matrix3d,
    tstar: Vector3d,
    n: Vector3d,
    v: f64,
) -> Matrix3d {
    // (2.0 / v) * tstar * n.transpose(): the scaled vector's outer product with n.
    let scaled = (2.0 / v) * tstar;
    let outer = Matrix3d::from_columns(scaled * n.x, scaled * n.y, scaled * n.z);
    *h_normalized * (Matrix3d::identity() - outer)
}

// Eigen's lpNorm<Infinity>(): the largest absolute coefficient.
fn max_abs_coefficient(m: &Matrix3d) -> f64 {
    m.as_slice().iter().fold(0.0, |max, &v| {
        let a = v.abs();
        if a > max {
            a
        } else {
            max
        }
    })
}

/// Port of `colmap::DecomposeHomographyMatrix`: the possible relative poses and plane
/// normals of `h` (between cameras with calibrations `k1` and `k2`), `(cams2_from_cams1,
/// normals)`: one (rotation, zero translation, zero normal) for a pure rotation, otherwise
/// four candidates.
pub fn decompose_homography_matrix(
    h: &Matrix3d,
    k1: &Matrix3d,
    k2: &Matrix3d,
) -> (Vec<Rigid3d>, Vec<Vector3d>) {
    // Remove calibration from homography.
    let mut h_normalized = k2.inverse() * *h * *k1;

    // Remove scale from normalized homography.
    let hmatrix_norm_svd = Svd3d::compute(&h_normalized);
    h_normalized /= hmatrix_norm_svd.singular_values.y;

    // Ensure that we always return rotations, and never reflections.
    //
    // It's enough to take det(H_normalized) > 0.
    //
    // To see this:
    // - In the paper: R := H_normalized * (Id + x y^t)^{-1} (page 32).
    // - Can check that this implies that R is orthogonal: RR^t = Id.
    // - To return a rotation, we also need det(R) > 0.
    // - By Sylvester's idenitity: det(Id + x y^t) = (1 + x^t y), which
    //   is positive by choice of x and y (page 24).
    // - So det(R) and det(H_normalized) have the same sign.
    if h_normalized.determinant() < 0.0 {
        h_normalized *= -1.0;
    }

    let s = h_normalized.transpose() * h_normalized - Matrix3d::identity();

    // Check if H is rotation matrix.
    const MIN_INFINITY_NORM: f64 = 1e-3;
    if max_abs_coefficient(&s) < MIN_INFINITY_NORM {
        return (
            vec![Rigid3d::new(
                Quaterniond::from_rotation_matrix(h_normalized),
                Vector3d::zeros(),
            )],
            vec![Vector3d::zeros()],
        );
    }

    let m00 = compute_opposite_of_minor(&s, 0, 0);
    let m11 = compute_opposite_of_minor(&s, 1, 1);
    let m22 = compute_opposite_of_minor(&s, 2, 2);

    let rt_m00 = fns::sqrt(maxd(m00, 0.0));
    let rt_m11 = fns::sqrt(maxd(m11, 0.0));
    let rt_m22 = fns::sqrt(maxd(m22, 0.0));

    let m01 = compute_opposite_of_minor(&s, 0, 1);
    let m12 = compute_opposite_of_minor(&s, 1, 2);
    let m02 = compute_opposite_of_minor(&s, 0, 2);

    let e12 = f64::from(sign_of_number(m12));
    let e02 = f64::from(sign_of_number(m02));
    let e01 = f64::from(sign_of_number(m01));

    let n_s00 = s[(0, 0)].abs();
    let n_s11 = s[(1, 1)].abs();
    let n_s22 = s[(2, 2)].abs();

    // std::max_element: the first of equal maxima.
    let mut idx = 0;
    if n_s11 > n_s00 {
        idx = 1;
    }
    if n_s22 > if idx == 0 { n_s00 } else { n_s11 } {
        idx = 2;
    }

    let (np1, np2) = match idx {
        0 => (
            Vector3d::new(s[(0, 0)], s[(0, 1)] + rt_m22, s[(0, 2)] + e12 * rt_m11),
            Vector3d::new(s[(0, 0)], s[(0, 1)] - rt_m22, s[(0, 2)] - e12 * rt_m11),
        ),
        1 => (
            Vector3d::new(s[(0, 1)] + rt_m22, s[(1, 1)], s[(1, 2)] - e02 * rt_m00),
            Vector3d::new(s[(0, 1)] - rt_m22, s[(1, 1)], s[(1, 2)] + e02 * rt_m00),
        ),
        _ => (
            Vector3d::new(s[(0, 2)] + e01 * rt_m11, s[(1, 2)] + rt_m00, s[(2, 2)]),
            Vector3d::new(s[(0, 2)] - e01 * rt_m11, s[(1, 2)] - rt_m00, s[(2, 2)]),
        ),
    };

    let trace_s = s.trace();
    let v = 2.0 * fns::sqrt(maxd(1.0 + trace_s - m00 - m11 - m22, 0.0));

    let e_sii = f64::from(sign_of_number(s[(idx, idx)]));
    let r_2 = 2.0 + trace_s + v;
    let nt_2 = 2.0 + trace_s - v;

    let r = fns::sqrt(maxd(r_2, 0.0));
    let n_t = fns::sqrt(maxd(nt_2, 0.0));

    let n1 = np1.normalized();
    let n2 = np2.normalized();

    let half_nt = 0.5 * n_t;
    let esii_t_r = e_sii * r;

    let t1_star = half_nt * (esii_t_r * n2 - n_t * n1);
    let t2_star = half_nt * (esii_t_r * n1 - n_t * n2);

    let r1 = compute_homography_rotation(&h_normalized, t1_star, n1, v);
    let t1 = r1 * t1_star;

    let r2 = compute_homography_rotation(&h_normalized, t2_star, n2, v);
    let t2 = r2 * t2_star;

    let q1 = Quaterniond::from_rotation_matrix(r1);
    let q2 = Quaterniond::from_rotation_matrix(r2);
    (
        vec![
            Rigid3d::new(q1, t1),
            Rigid3d::new(q1, -t1),
            Rigid3d::new(q2, t2),
            Rigid3d::new(q2, -t2),
        ],
        vec![-n1, n1, -n2, n2],
    )
}

/// `std::max(a, b)` on doubles, `(a < b) ? b : a`.
fn maxd(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

fn check_cheirality_and_reproj_error_sum(
    cam2_from_cam1: &Rigid3d,
    cam_rays1: &[Vector3d],
    cam_rays2: &[Vector3d],
    points3d: &mut Vec<Vector3d>,
) -> f64 {
    let mut reproj_residual_sum = 0.0;
    points3d.clear();
    for (&ray1, &ray2) in cam_rays1.iter().zip(cam_rays2) {
        let Some(point3d_in_cam1) = triangulate_mid_point(cam2_from_cam1, ray1, ray2) else {
            continue;
        };
        let point3d_in_cam2 = *cam2_from_cam1 * point3d_in_cam1;
        let error1 = 1.0 - ray1.dot(point3d_in_cam1.normalized()).clamp(-1.0, 1.0);
        let error2 = 1.0 - ray2.dot(point3d_in_cam2.normalized()).clamp(-1.0, 1.0);
        reproj_residual_sum += error1 + error2;
        points3d.push(point3d_in_cam1);
    }
    reproj_residual_sum
}

/// Port of `colmap::PoseFromHomographyMatrix`: the decomposition candidate that
/// triangulates the most ray pairs in front of both cameras (ties broken by the smaller
/// angular reprojection error), its plane normal, and those points in camera 1's frame,
/// `(cam2_from_cam1, normal, points3D)`. The ray lists must have equal length.
///
/// The first candidate is always accepted (with no triangulated points its residual sum is
/// 0 < `f64::MAX`; with any points it beats the empty initial set), so the identity / zero
/// normal the outputs start from is never returned.
pub fn pose_from_homography_matrix(
    h: &Matrix3d,
    k1: &Matrix3d,
    k2: &Matrix3d,
    cam_rays1: &[Vector3d],
    cam_rays2: &[Vector3d],
) -> Result<(Rigid3d, Vector3d, Vec<Vector3d>)> {
    check_eq!(cam_rays1.len(), cam_rays2.len());

    let (cams2_from_cams1, normals) = decompose_homography_matrix(h, k1, k2);
    check_eq!(cams2_from_cams1.len(), normals.len());

    let mut cam2_from_cam1 = Rigid3d::identity();
    let mut normal = Vector3d::zeros();
    let mut points3d: Vec<Vector3d> = Vec::new();
    let mut tentative_points3d: Vec<Vector3d> = Vec::new();
    let mut best_reproj_residual_sum = f64::MAX;
    for (candidate, candidate_normal) in cams2_from_cams1.iter().zip(&normals) {
        // Note that we can typically eliminate 2 of the 4 solutions using the
        // cheirality check. We can then typically narrow it down to 1 solution by
        // picking the solution with minimal overall reprojection error.
        let reproj_residual_sum = check_cheirality_and_reproj_error_sum(
            candidate,
            cam_rays1,
            cam_rays2,
            &mut tentative_points3d,
        );
        if tentative_points3d.len() > points3d.len()
            || (tentative_points3d.len() == points3d.len()
                && reproj_residual_sum < best_reproj_residual_sum)
        {
            best_reproj_residual_sum = reproj_residual_sum;
            cam2_from_cam1 = *candidate;
            normal = *candidate_normal;
            std::mem::swap(&mut points3d, &mut tentative_points3d);
        }
    }
    Ok((cam2_from_cam1, normal, points3d))
}

/// Port of `colmap::HomographyMatrixFromPose`: `H = K2 (R - t n^T / d) K1^-1` for the plane
/// with (normalized) normal `n` at distance `d > 0` from camera 1.
pub fn homography_matrix_from_pose(
    k1: &Matrix3d,
    k2: &Matrix3d,
    r: &Matrix3d,
    t: Vector3d,
    n: Vector3d,
    d: f64,
) -> Result<Matrix3d> {
    check_gt!(d, 0.0);
    let nn = n.normalized();
    let tnt = Matrix3d::from_columns(t * nn.x, t * nn.y, t * nn.z);
    Ok(*k2 * (*r - tnt / d) * k1.inverse())
}

/// Port of `colmap::ComputeSquaredHomographyError`: the squared distance between `point2`
/// and the transfer of `point1` by `h`; `f64::MAX` when `point1` maps to infinity.
pub fn compute_squared_homography_error(point1: Vector2d, point2: Vector2d, h: &Matrix3d) -> f64 {
    let hp1 = *h * point1.homogeneous();
    if hp1.z == 0.0 {
        return f64::MAX;
    }
    (point2 - hp1.hnormalized()).squared_norm()
}
