//! Port of the free functions of COLMAP's `colmap/estimators/fundamental_matrix_degensac.h`
//! and `.cc`, DEGENSAC (Chum, Werner and Matas, "Two-View Geometry Estimation Unaffected by
//! a Dominant Plane", CVPR 2005): the epipole of F, the plane homography compatible with F
//! through three correspondences (H&Z Result 13.6), the H-degeneracy test of a sample and
//! the plane-and-parallax completion. The estimator that RANSAC runs, its options and
//! `EstimateFundamentalMatrixDegensac` are in [`super::fundamental_matrix_degensac_estimator`].
//! Built on [`crate::estimators::solvers::fundamental_matrix`] (8-point) and
//! [`crate::estimators::solvers::homography_matrix`]. Port of colmap-sharp's
//! `Estimators/FundamentalMatrixDegensac.cs`. Tests:
//! `tests/estimators/fundamental_matrix_degensac.rs` (fundamental_matrix_degensac_test.cc).
//!
//! Tier B (tolerance) for the closed-form pieces (epipole, compatible homography); Tier C
//! (outcome) for the degeneracy test on non-minimal samples and the completion, which draw
//! from the PRNG. The draws themselves are Tier A: they go through [`crate::math::random`]
//! in COLMAP's order with COLMAP's integer types (`i32` for the triplet indices, `usize` for
//! the Fisher-Yates and parallax-pair draws), so a seeded run consumes the same sequence as
//! COLMAP.
//!
//! Translation notes:
//! - `std::optional<Matrix3d>` becomes `Option<Matrix3d>`; functions with a `THROW_CHECK`
//!   return `Result<Option<_>>`.
//! - `std::llround` rounds halves away from zero, as [`f64::round`] does.

use crate::estimators::solvers::fundamental_matrix::FundamentalMatrixEightPointEstimator;
use crate::estimators::solvers::homography_matrix::HomographyMatrixEstimator;
use crate::geometry::essential_matrix::compute_squared_sampson_errors;
use crate::geometry::homography_matrix::compute_squared_homography_error;
use crate::geometry::rigid3::cross_product_matrix;
use crate::linalg::{Matrix3d, Vector2d, Vector3d};
use crate::math::random::random_uniform_integer;
use crate::optim::estimator::Estimator;
use crate::optim::support_measurement::{InlierSupport, InlierSupportMeasurer, SupportMeasurer};
use crate::{check_eq, check_ge, Result};

/// `FundamentalMatrixDegensacEstimator::kMinNumSamples`: the minimal (7-point) sample size.
pub(super) const DEGENSAC_MIN_NUM_SAMPLES: usize = 7;

// All C(7,3) = 35 triplets of the seven minimal-sample indices. Enumerated once so the
// sample degeneracy test does not recompute them per hypothesis.
const NUM_SAMPLE_TRIPLETS: usize = 35;
const SAMPLE_TRIPLETS: [[usize; 3]; NUM_SAMPLE_TRIPLETS] = [
    [0, 1, 2],
    [0, 1, 3],
    [0, 1, 4],
    [0, 1, 5],
    [0, 1, 6],
    [0, 2, 3],
    [0, 2, 4],
    [0, 2, 5],
    [0, 2, 6],
    [0, 3, 4],
    [0, 3, 5],
    [0, 3, 6],
    [0, 4, 5],
    [0, 4, 6],
    [0, 5, 6],
    [1, 2, 3],
    [1, 2, 4],
    [1, 2, 5],
    [1, 2, 6],
    [1, 3, 4],
    [1, 3, 5],
    [1, 3, 6],
    [1, 4, 5],
    [1, 4, 6],
    [1, 5, 6],
    [2, 3, 4],
    [2, 3, 5],
    [2, 3, 6],
    [2, 4, 5],
    [2, 4, 6],
    [2, 5, 6],
    [3, 4, 5],
    [3, 4, 6],
    [3, 5, 6],
    [4, 5, 6],
];

// Number of triplets randomly sampled from a non-minimal sample when searching for the
// dominant plane (a minimal sample enumerates all 35 exhaustively).
const NUM_SAMPLED_TRIPLETS: i32 = 50;

/// Port of `colmap::EpipoleFromFundamentalMatrix`: the epipole in the second image, i.e.
/// the left null vector `e2` of `F` with `F^T e2 = 0`, homogeneous and of unit length.
pub fn epipole_from_fundamental_matrix(f: &Matrix3d) -> Vector3d {
    // The epipole e2 is the left null vector (F^T e2 = 0), i.e. it is orthogonal to the
    // column space of the rank-2 matrix F and thus parallel to the cross product of two of
    // its columns. Using the column pair with the largest cross product is numerically
    // stable and avoids a full SVD.
    let e01 = f.col(0).cross(f.col(1));
    let e02 = f.col(0).cross(f.col(2));
    let e12 = f.col(1).cross(f.col(2));
    let n01 = e01.squared_norm();
    let n02 = e02.squared_norm();
    let n12 = e12.squared_norm();
    if n01 >= n02 && n01 >= n12 {
        return e01.normalized();
    }
    if n02 >= n12 {
        e02.normalized()
    } else {
        e12.normalized()
    }
}

// Plane-induced homography compatible with the epipolar geometry from three
// correspondences (H&Z Result 13.6), given the precomputed A = [e2]_x F and the epipole e2.
// Splitting this out lets the degeneracy test reuse A across all triplets of a sample
// instead of recomputing it each time.
fn homography_from_compatible(
    a: &Matrix3d,
    epipole2: Vector3d,
    points1: &[Vector2d; 3],
    points2: &[Vector2d; 3],
) -> Option<Matrix3d> {
    let mut m_rows = [Vector3d::default(); 3];
    let mut b = Vector3d::default();
    for i in 0..3 {
        let x1 = points1[i].homogeneous();
        let x2 = points2[i].homogeneous();
        let x2_cross_e2 = x2.cross(epipole2);
        let denom = x2_cross_e2.squared_norm();
        if denom < 1e-12 {
            // The point lies (near) the epipole, so the transfer is undefined.
            return None;
        }
        b[i] = x2.cross(*a * x1).dot(x2_cross_e2) / denom;
        m_rows[i] = x1;
    }
    let m = Matrix3d::from_rows(m_rows[0], m_rows[1], m_rows[2]);

    // Reject collinear first-image points (scale-invariant singularity check), then solve
    // with the closed-form 3x3 inverse.
    let det = m.determinant();
    let scale = m.row(0).norm() * m.row(1).norm() * m.row(2).norm();
    if scale < 1e-12 || det.abs() < 1e-9 * scale {
        return None;
    }
    let minv_b = m.inverse() * b;
    // A - e2 * (M^-1 b)^T.
    Some(
        *a - Matrix3d::from_columns(
            epipole2 * minv_b.x,
            epipole2 * minv_b.y,
            epipole2 * minv_b.z,
        ),
    )
}

/// Port of `colmap::HomographyFromFundamentalAndPoints`: the plane-induced homography
/// compatible with the epipolar geometry `F` from three point correspondences, following
/// Hartley & Zisserman, "Multiple View Geometry", Result 13.6:
///
/// `H = [e2]_x F - e2 (M^{-1} b)^T`
///
/// where the rows of M are the homogeneous first-image points x1_i, and
/// `b_i = (x2_i x ([e2]_x F x1_i)) . (x2_i x e2) / ||x2_i x e2||^2`.
///
/// `None` on degenerate input (collinear first-image points, or a point at the epipole).
pub fn homography_from_fundamental_and_points(
    f: &Matrix3d,
    epipole2: Vector3d,
    points1: &[Vector2d; 3],
    points2: &[Vector2d; 3],
) -> Option<Matrix3d> {
    // A = [e2]_x F is a particular homography compatible with F (H&Z Result 13.6).
    homography_from_compatible(
        &(cross_product_matrix(epipole2) * *f),
        epipole2,
        points1,
        points2,
    )
}

/// Port of `colmap::DetectSampleHDegeneracy`: tests a sample for H-degeneracy, i.e. whether
/// a fraction of at least `min_sample_h_inlier_ratio` of the correspondences lie on a
/// common scene plane. Plane homographies compatible with F are constructed from triplets
/// of the sample (all C(7,3) triplets for a minimal sample, otherwise a fixed number of
/// randomly sampled triplets) and the number of sample correspondences consistent with each
/// (squared forward transfer error <= `h_max_residual`) is counted. Returns the homography
/// of the triplet with the most consistent correspondences if that count reaches the
/// threshold, otherwise `None`.
pub fn detect_sample_h_degeneracy(
    f: &Matrix3d,
    sample_points1: &[Vector2d],
    sample_points2: &[Vector2d],
    h_max_residual: f64,
    min_sample_h_inlier_ratio: f64,
) -> Result<Option<Matrix3d>> {
    let num_samples = sample_points1.len();
    check_eq!(sample_points1.len(), sample_points2.len());
    check_ge!(num_samples, 3);

    let epipole2 = epipole_from_fundamental_matrix(f);
    // A = [e2]_x F is shared by all triplets; compute it once.
    let a = cross_product_matrix(epipole2) * *f;
    // `static_cast<int>(std::llround(x))`: `round` is llround's half-away-from-zero.
    let min_consistent = 3.max((min_sample_h_inlier_ratio * num_samples as f64).round() as i32);

    let mut best_num_consistent = min_consistent - 1;
    let mut best_h: Option<Matrix3d> = None;

    // Returns true if all sample points are consistent (nothing left to improve).
    let mut evaluate_triplet = |i: usize, j: usize, k: usize| -> bool {
        let tri_points1 = [sample_points1[i], sample_points1[j], sample_points1[k]];
        let tri_points2 = [sample_points2[i], sample_points2[j], sample_points2[k]];
        let Some(h) = homography_from_compatible(&a, epipole2, &tri_points1, &tri_points2) else {
            return false;
        };
        let num_consistent = sample_points1
            .iter()
            .zip(sample_points2)
            .filter(|(&p1, &p2)| compute_squared_homography_error(p1, p2, &h) <= h_max_residual)
            .count() as i32;
        if num_consistent > best_num_consistent {
            best_num_consistent = num_consistent;
            best_h = Some(h);
        }
        best_num_consistent >= num_samples as i32
    };

    if num_samples == DEGENSAC_MIN_NUM_SAMPLES {
        // Minimal sample: exhaustively enumerate all triplets.
        for triplet in &SAMPLE_TRIPLETS {
            if evaluate_triplet(triplet[0], triplet[1], triplet[2]) {
                break;
            }
        }
    } else {
        // Non-minimal sample (e.g. a local-optimization inlier set): sample triplets. On a
        // plane-dominated sample most triplets lie on the plane, so a modest number reliably
        // discovers it.
        let last = num_samples as i32 - 1;
        for _ in 0..NUM_SAMPLED_TRIPLETS {
            let i = random_uniform_integer::<i32>(0, last);
            let j = random_uniform_integer::<i32>(0, last);
            let k = random_uniform_integer::<i32>(0, last);
            if i == j || j == k || i == k {
                continue;
            }
            if evaluate_triplet(i as usize, j as usize, k as usize) {
                break;
            }
        }
    }

    Ok(best_h)
}

/// Port of `colmap::IsSampleHDegenerate`: convenience predicate wrapping
/// [`detect_sample_h_degeneracy`].
pub fn is_sample_h_degenerate(
    f: &Matrix3d,
    sample_points1: &[Vector2d],
    sample_points2: &[Vector2d],
    h_max_residual: f64,
    min_sample_h_inlier_ratio: f64,
) -> Result<bool> {
    Ok(detect_sample_h_degeneracy(
        f,
        sample_points1,
        sample_points2,
        h_max_residual,
        min_sample_h_inlier_ratio,
    )?
    .is_some())
}

// Squared forward transfer error of a correspondence under a homography.
fn transfer_error(point1: Vector2d, point2: Vector2d, h: &Matrix3d) -> f64 {
    compute_squared_homography_error(point1, point2, h)
}

// Draws `num` distinct entries of `scratch` into its front via a partial Fisher-Yates
// shuffle.
fn sample_distinct(num: usize, scratch: &mut [usize]) {
    let n = scratch.len();
    for i in 0..num {
        let j = i + random_uniform_integer::<usize>(0, n - 1 - i);
        scratch.swap(i, j);
    }
}

/// Port of `colmap::FundamentalFromPlaneAndParallax`: recovers the fundamental matrix from
/// a dominant plane homography and the off-plane parallax (plane-and-parallax model
/// completion). The seed homography is refit on its plane inliers (squared transfer error
/// <= `plane_max_residual`); then correspondences whose squared transfer error exceeds
/// `off_plane_min_residual` are treated as clean off-plane parallax sources, and the epipole
/// is recovered from a pair of them as `e2 = (x2_a x H x1_a) x (x2_b x H x1_b)`, giving
/// `F = [e2]_x H`. Pairs are sampled robustly and the best F is then refined by fitting the
/// 8-point algorithm to mixed samples of plane and off-plane inliers (so the epipole is
/// constrained by more than two off-plane points). The F with the largest squared-Sampson
/// support (threshold `sampson_max_residual`) over all correspondences is returned.
///
/// `None` if there are too few off-plane correspondences.
pub fn fundamental_from_plane_and_parallax(
    seed_h: &Matrix3d,
    points1: &[Vector2d],
    points2: &[Vector2d],
    sampson_max_residual: f64,
    plane_max_residual: f64,
    off_plane_min_residual: f64,
    max_trials: i32,
) -> Result<Option<Matrix3d>> {
    check_eq!(points1.len(), points2.len());

    // The seed homography is built from only three sample correspondences (via the
    // plane-corrupted sample fundamental matrix), so it is only approximate. Refit it on
    // all of its plane inliers to obtain an accurate dominant-plane homography; otherwise
    // the off-plane classification below is polluted with plane points and the epipole
    // recovery becomes unreliable.
    let mut h = *seed_h;
    let mut homography_estimator = HomographyMatrixEstimator;
    let mut plane_idxs: Vec<usize> = Vec::new();
    let mut plane_points1: Vec<Vector2d> = Vec::new();
    let mut plane_points2: Vec<Vector2d> = Vec::new();
    let mut homographies: Vec<Matrix3d> = Vec::new();
    const NUM_REFIT_ITERS: usize = 2;
    // A homography is well determined by a modest, spatially spread subset, so cap the
    // number of correspondences used for the DLT to avoid an O(N) SVD when the dominant
    // plane has many inliers.
    const MAX_HOMOGRAPHY_FIT_POINTS: usize = 64;
    for _ in 0..NUM_REFIT_ITERS {
        plane_idxs.clear();
        for (i, (&p1, &p2)) in points1.iter().zip(points2).enumerate() {
            if transfer_error(p1, p2, &h) <= plane_max_residual {
                plane_idxs.push(i);
            }
        }
        if plane_idxs.len() < HomographyMatrixEstimator::MIN_NUM_SAMPLES {
            break;
        }
        let num_fit = MAX_HOMOGRAPHY_FIT_POINTS.min(plane_idxs.len());
        if plane_idxs.len() > num_fit {
            sample_distinct(num_fit, &mut plane_idxs);
        }
        plane_points1.clear();
        plane_points2.clear();
        for &idx in &plane_idxs[..num_fit] {
            plane_points1.push(points1[idx]);
            plane_points2.push(points2[idx]);
        }
        homographies.clear();
        homography_estimator.estimate(&plane_points1, &plane_points2, &mut homographies)?;
        if homographies.is_empty() {
            break;
        }
        h = homographies[0];
    }

    // The dominant-plane homography is now fixed, so the transfer error of every
    // correspondence against it is reused below (off-plane classification here and the
    // plane/off-plane split for the mixed-sample refit) instead of recomputed.
    let plane_transfer_errors: Vec<f64> = points1
        .iter()
        .zip(points2)
        .map(|(&p1, &p2)| transfer_error(p1, p2, &h))
        .collect();

    // Only correspondences well off the plane carry reliable parallax: near-plane points
    // have tiny, noise-dominated parallax that destabilizes the epipole.
    let off_plane_idxs: Vec<usize> = (0..points1.len())
        .filter(|&i| plane_transfer_errors[i] > off_plane_min_residual)
        .collect();
    let num_off_plane = off_plane_idxs.len();
    if num_off_plane < 2 {
        return Ok(None);
    }

    // Precompute the transferred plane points H * x1 and the off-plane points, so the
    // epipole search below scores over only the off-plane subset.
    let lines: Vec<Vector3d> = off_plane_idxs
        .iter()
        .map(|&idx| {
            points2[idx]
                .homogeneous()
                .cross(h * points1[idx].homogeneous())
        })
        .collect();
    let off_points1: Vec<Vector2d> = off_plane_idxs.iter().map(|&idx| points1[idx]).collect();
    let off_points2: Vec<Vector2d> = off_plane_idxs.iter().map(|&idx| points2[idx]).collect();

    let mut support_measurer = InlierSupportMeasurer;

    // Plane correspondences are consistent with F = [e2]_x H for ANY epipole e2
    // (x2^T [e2]_x H x1 ~ x2^T [e2]_x x2 = 0 when H x1 ~ x2), so among the F = [e2]_x H
    // candidates only the off-plane correspondences discriminate. Scoring epipole
    // candidates over just the off-plane subset ranks them correctly at a fraction of the
    // cost of scoring all correspondences.
    let mut best_off_support = InlierSupport::default();
    let mut best_f: Option<Matrix3d> = None;
    let num_pairs = num_off_plane * (num_off_plane - 1) / 2;
    // `std::min<size_t>(static_cast<size_t>(std::max(max_trials, 1)), num_pairs)`.
    let num_trials = (max_trials.max(1) as usize).min(num_pairs);
    for _ in 0..num_trials {
        let a = random_uniform_integer::<usize>(0, num_off_plane - 1);
        let mut b = random_uniform_integer::<usize>(0, num_off_plane - 1);
        if a == b {
            b = if b + 1 == num_off_plane { 0 } else { b + 1 };
        }

        // The epipole is the intersection of the two parallax lines.
        let epipole2 = lines[a].cross(lines[b]);
        if epipole2.norm() < 1e-9 {
            continue;
        }
        let f = cross_product_matrix(epipole2.normalized()) * h;

        let residuals = compute_squared_sampson_errors(&off_points1, &off_points2, &f)?;
        let support = support_measurer.evaluate(&residuals, sampson_max_residual)?;
        if support_measurer.is_left_better(&support, &best_off_support) {
            best_off_support = support;
            best_f = Some(f);
        }
    }

    let Some(mut best_f) = best_f else {
        return Ok(None);
    };

    // Seed the full-data support of the best epipole candidate, against which the
    // mixed-sample refinement below is compared.
    let residuals = compute_squared_sampson_errors(points1, points2, &best_f)?;
    let mut best_support = support_measurer.evaluate(&residuals, sampson_max_residual)?;

    // Refine the recovered fundamental matrix by fitting the 8-point algorithm to mixed
    // samples of plane and off-plane inliers, so the epipole is constrained by several
    // off-plane points rather than just the two used above. Fitting on a balanced sample
    // avoids the plane bias that a full-inlier fit would incur.
    let mut plane_inliers: Vec<usize> = (0..points1.len())
        .filter(|&i| plane_transfer_errors[i] <= plane_max_residual)
        .collect();
    let mut off_plane_inliers: Vec<usize> = (0..points1.len())
        .filter(|&i| {
            residuals[i] <= sampson_max_residual
                && plane_transfer_errors[i] > off_plane_min_residual
        })
        .collect();

    const NUM_PLANE_SAMPLE: usize = 6;
    const MAX_NUM_OFF_PLANE_SAMPLE: usize = 4;
    if plane_inliers.len() >= NUM_PLANE_SAMPLE && off_plane_inliers.len() >= 2 {
        let num_off_sample = MAX_NUM_OFF_PLANE_SAMPLE.min(off_plane_inliers.len());
        let mut eight_point = FundamentalMatrixEightPointEstimator;
        let mut sample_points1: Vec<Vector2d> = Vec::new();
        let mut sample_points2: Vec<Vector2d> = Vec::new();
        let mut refined_models: Vec<Matrix3d> = Vec::new();
        const NUM_REFINE_TRIALS: usize = 15;
        for _ in 0..NUM_REFINE_TRIALS {
            sample_distinct(NUM_PLANE_SAMPLE, &mut plane_inliers);
            sample_distinct(num_off_sample, &mut off_plane_inliers);
            sample_points1.clear();
            sample_points2.clear();
            for &idx in &plane_inliers[..NUM_PLANE_SAMPLE] {
                sample_points1.push(points1[idx]);
                sample_points2.push(points2[idx]);
            }
            for &idx in &off_plane_inliers[..num_off_sample] {
                sample_points1.push(points1[idx]);
                sample_points2.push(points2[idx]);
            }

            refined_models.clear();
            eight_point.estimate(&sample_points1, &sample_points2, &mut refined_models)?;
            let Some(&refined) = refined_models.first() else {
                continue;
            };
            let residuals = compute_squared_sampson_errors(points1, points2, &refined)?;
            let support = support_measurer.evaluate(&residuals, sampson_max_residual)?;
            if support_measurer.is_left_better(&support, &best_support) {
                best_support = support;
                best_f = refined;
            }
        }
    }

    Ok(Some(best_f))
}
