//! Port of COLMAP's `colmap/estimators/solvers/relpose_shared_focal.h` / `.cc`, via
//! colmap-sharp's `Estimators/Solvers/RelativePoseSharedFocal.cs`: the relative pose
//! estimator for two views sharing one unknown focal length, on principal-point-centered
//! pixels. The minimal solve is PoseLib's [`relpose_6pt_shared_focal`]; the LO-RANSAC refiner
//! is [`crate::optim::TinySolver`] on [`TinyFocalSampsonErrorCostFunctor`] over the
//! pose-plus-log-focal manifold; residuals are the pixel Sampson error under
//! F = K^-1 E K^-1. [`RelativePoseSharedFocalEstimator::is_focal_identifiable`] screens out
//! the poses whose focal cannot be recovered. Neighbor: [`super::relpose_one_sided_focal`]
//! (one view calibrated), which shares `calibrated_rays` and `refine_pose_and_focal`.
//! Tests: `tests/estimators/relpose_shared_focal.rs` (`relpose_shared_focal_test.cc`).
//!
//! Tier B for `estimate` and `residuals`, Tier C for `refine`.

// `!(x > 0.0)` guards, as written in C++: a NaN focal, scale or denominator fails them too.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::poselib::relpose_6pt_shared_focal::relpose_6pt_shared_focal;
use super::poselib_utils::convert_poselib_pose_to_rigid3d;
use crate::estimators::cost_functions::{
    TinyEigenQuaternionManifold, TinyFocalSampsonErrorCostFunctor, TinyProductManifold,
    TinySphereManifold,
};
use crate::geometry::essential_matrix::{
    compute_squared_sampson_errors, essential_matrix_from_pose, pose_from_essential_matrix,
};
use crate::geometry::Rigid3d;
use crate::linalg::{Matrix3d, Quaterniond, Vector2d, Vector3d};
use crate::math::fns;
use crate::optim::{
    Estimator, LocalEstimator, TinyEuclideanManifold, TinySolver, TinySolverFunction,
    TinySolverOptions,
};
use crate::{check_eq, check_ge, Result};

/// The 6-DoF manifold of a relative pose plus one focal length: rotation on SO(3),
/// translation on the unit sphere, and the log-focal as a 1-D Euclidean parameter, in the
/// ambient layout [qx, qy, qz, qw, tx, ty, tz, log_f] (C++'s
/// `ProductManifold<EigenQuaternionManifold, SphereManifold<3>, EuclideanManifold<1>>`, used
/// by both focal estimators).
type RelativePoseFocalManifold = TinyProductManifold<
    TinyEigenQuaternionManifold,
    TinyProductManifold<TinySphereManifold, TinyEuclideanManifold>,
>;

/// Focal-calibrated, normalized camera rays (x / f, y / f, 1) from centered image points.
pub(super) fn calibrated_rays(points: &[Vector2d], focal: f64) -> Vec<Vector3d> {
    let inv_f = 1.0 / focal;
    points
        .iter()
        .map(|p| Vector3d::new(p.x * inv_f, p.y * inv_f, 1.0).normalized())
        .collect()
}

/// The shared tail of both focal estimators' `Refine`: TinySolver over `functor` (the pose
/// plus log-focal manifold, 25 iterations) from `cam2_from_cam1` and `focal`. Returns the
/// refined (E, focal) only if the solve stayed finite and left a non-degenerate baseline;
/// otherwise the caller keeps the seed model.
pub(super) fn refine_pose_and_focal<F: TinySolverFunction>(
    functor: &F,
    cam2_from_cam1: &Rigid3d,
    focal: f64,
) -> Result<Option<(Matrix3d, f64)>> {
    let manifold = RelativePoseFocalManifold::new(
        TinyEigenQuaternionManifold,
        TinyProductManifold::new(TinySphereManifold, TinyEuclideanManifold::new(1)?),
    );
    let mut solver = TinySolver::new(manifold)?;
    let options = TinySolverOptions {
        max_num_iterations: 25,
        ..TinySolverOptions::default()
    };

    let q = cam2_from_cam1.rotation.normalized();
    let t = cam2_from_cam1.translation.normalized();
    let mut x = [q.x, q.y, q.z, q.w, t.x, t.y, t.z, fns::ln(focal)];
    solver.solve(functor, &mut x, &options)?;

    // Keep the refined estimate only if the solve stayed finite and left a non-degenerate
    // baseline; otherwise fall back to the decomposed pose and seed focal.
    let translation = Vector3d::new(x[4], x[5], x[6]);
    if x.iter().all(|v| v.is_finite()) && translation.squared_norm() > 0.0 {
        let refined = Rigid3d::new(
            Quaterniond::new(x[3], x[0], x[1], x[2]).normalized(),
            translation,
        );
        return Ok(Some((essential_matrix_from_pose(&refined), fns::exp(x[7]))));
    }
    Ok(None)
}

/// The estimated model of [`RelativePoseSharedFocalEstimator`] (C++'s `M_t`): a calibrated
/// essential matrix plus the shared focal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelativePoseSharedFocalModel {
    /// The calibrated essential matrix.
    pub e: Matrix3d,
    /// The shared focal length, in pixels.
    pub focal: f64,
}

impl Default for RelativePoseSharedFocalModel {
    /// C++'s default: E = identity, focal = 0.
    fn default() -> Self {
        Self {
            e: Matrix3d::identity(),
            focal: 0.0,
        }
    }
}

/// Pixel-space fundamental matrix F = Kinv * E * Kinv with Kinv = diag(1/f, 1/f, 1), from a
/// calibrated essential matrix and the shared focal length.
/// Eigen's diagonal products scale rows, then columns, with no further arithmetic.
fn fundamental_from_essential_shared_focal(e: &Matrix3d, focal: f64) -> Matrix3d {
    let inv_f = 1.0 / focal;
    let k_inv = [inv_f, inv_f, 1.0];
    let mut f = *e;
    for row in 0..3 {
        for col in 0..3 {
            f[(row, col)] = k_inv[row] * e[(row, col)] * k_inv[col];
        }
    }
    f
}

/// Sine of the angle between the baseline and the plane spanned by the two optical axes:
/// |b . (a1 x a2)| for unit axes a1 = e_z, a2 = R^T e_z and unit baseline b. Zero iff the axes
/// are coplanar, approaching 1 for skew axes. Only directions enter, so the score is
/// invariant to the translation scale.
fn axes_skewness(cam2_from_cam1: &Rigid3d) -> f64 {
    let axis1 = Vector3d::new(0.0, 0.0, 1.0);
    let axis2 = cam2_from_cam1.rotation.inverse() * Vector3d::new(0.0, 0.0, 1.0);
    let baseline_dir = cam2_from_cam1.tgt_origin_in_src().normalized();
    if !(baseline_dir.x.is_finite() && baseline_dir.y.is_finite() && baseline_dir.z.is_finite()) {
        return 0.0; // Pure rotation: no baseline direction; fully degenerate.
    }
    baseline_dir.dot(axis1.cross(axis2)).abs()
}

/// Scale-invariant asymmetry |d1 - d2| / (|d1| + |d2|) in [0, 1] of the distances d1, d2 of
/// the two camera centers from the closest-approach point of their optical axes. For coplanar
/// axes that point is their intersection, and the score is zero exactly for the isosceles
/// configuration. Closest approach is used rather than a true intersection so the distances
/// stay well-defined for merely near-coplanar axes, where the intersection is
/// ill-conditioned.
///
/// As the axes approach parallel the two distances diverge while their difference stays
/// bounded, so the score decays to zero on its own; the parallel singularity therefore needs
/// no special case.
fn isosceles_deviation(cam2_from_cam1: &Rigid3d) -> f64 {
    let center2 = cam2_from_cam1.tgt_origin_in_src();
    let axis1 = Vector3d::new(0.0, 0.0, 1.0);
    let axis2 = cam2_from_cam1.rotation.inverse() * Vector3d::new(0.0, 0.0, 1.0);

    // Closest approach of the lines d1 * axis1 and center2 + d2 * axis2.
    let cos_axes = axis1.dot(axis2);
    let sin_sq_axes = 1.0 - cos_axes * cos_axes;
    if sin_sq_axes == 0.0 {
        return 0.0; // Exactly parallel axes: singular.
    }
    let proj1 = center2.dot(axis1);
    let proj2 = center2.dot(axis2);
    let dist1 = (proj1 - cos_axes * proj2) / sin_sq_axes;
    let dist2 = (cos_axes * proj1 - proj2) / sin_sq_axes;

    let dist_sum = dist1.abs() + dist2.abs();
    if dist_sum == 0.0 {
        return 0.0; // Both centers at the intersection point.
    }
    (dist1 - dist2).abs() / dist_sum
}

/// Isotropic normalization scale: the mean magnitude of the centered image points across both
/// views. Returns 0 if there are no points.
fn compute_scale_for_normalization(points1: &[Vector2d], points2: &[Vector2d]) -> f64 {
    if points1.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0;
    for (p1, p2) in points1.iter().zip(points2) {
        sum += p1.norm() + p2.norm();
    }
    sum / (2.0 * points1.len() as f64)
}

/// Port of `colmap::RelativePoseSharedFocalEstimator`: relative pose with one unknown, shared
/// focal length, wrapping PoseLib's minimal 6-point solver.
///
/// Both images are assumed to share a single unknown focal length, e.g. two images captured
/// by the same camera without a reliable focal-length prior.
///
/// Two-view focal recovery is singular for coplanar optical axes, i.e. axes that intersect or
/// are parallel, in the general case of two independent unknown focals (S. Bougnoux, From
/// projective to Euclidean space under any practical situation, a criticism of
/// self-calibration, ICCV, 1998). For a single shared focal, coplanar axes alone are not
/// singular. They are singular only if the axes are additionally parallel, or intersect with
/// the camera centers equidistant from the point of intersection, following p. 5 of
/// H. Stewenius, D. Nister, F. Kahl, F. Schaffalitzky, A minimal solution for relative pose
/// with unknown focal length, Image and Vision Computing, 26(7), 2008.
/// [`Self::is_focal_identifiable`] tests a recovered pose against the singular family, so
/// callers can reject an unreliable focal. Neither reference gives a measure of distance from
/// it, so the scores and thresholds behind the test are heuristic.
///
/// Inputs (X/Y) are principal-point-centered image points (u - cx, v - cy), not calibrated
/// rays. Residuals and refinement are measured as pixel-space squared Sampson error via
/// F = diag(1/f, 1/f, 1) * E * diag(1/f, 1/f, 1).
///
/// The estimator serves as both the global (minimal) and local (refinement) estimator inside
/// LO-RANSAC: as a [`LocalEstimator`] it refines a copy of the current best model
/// ([`Self::refine`], C++'s `Refine` hook). This nonlinear refinement is required because the
/// minimal solver consumes exactly six points and has no non-minimal least-squares
/// counterpart.
#[derive(Clone, Copy, Debug, Default)]
pub struct RelativePoseSharedFocalEstimator;

impl RelativePoseSharedFocalEstimator {
    /// The minimum number of samples needed to estimate a model.
    pub const MIN_NUM_SAMPLES: usize = 6;

    /// Estimates relative pose and shared focal from >= 6 centered point pairs by wrapping
    /// `poselib::relpose_6pt_shared_focal` (uses the first six points). Clears `models`.
    pub fn estimate(
        points1: &[Vector2d],
        points2: &[Vector2d],
        models: &mut Vec<RelativePoseSharedFocalModel>,
    ) -> Result<()> {
        check_eq!(points1.len(), points2.len());
        check_ge!(points1.len(), Self::MIN_NUM_SAMPLES);
        models.clear();

        // The minimal solver is formulated for unit bearing vectors, so raw pixel coordinates
        // make it numerically unstable: after appending the homogeneous 1 and normalizing,
        // that third component becomes negligible and the focal length is recovered from it.
        // Isotropically rescale the (already principal-point-centered) points to unit
        // magnitude first; the recovered focal is expressed in the rescaled units and undone
        // below. This mirrors the normalization PoseLib applies before running the solver.
        let scale = compute_scale_for_normalization(points1, points2);
        if !(scale > 0.0) {
            return Ok(());
        }
        let inv_scale = 1.0 / scale;

        let rays1: Vec<Vector3d> = points1
            .iter()
            .map(|p| (inv_scale * *p).homogeneous().normalized())
            .collect();
        let rays2: Vec<Vector3d> = points2
            .iter()
            .map(|p| (inv_scale * *p).homogeneous().normalized())
            .collect();

        let mut image_pairs = Vec::new();
        relpose_6pt_shared_focal(&rays1, &rays2, &mut image_pairs);

        models.reserve(image_pairs.len());
        for image_pair in &image_pairs {
            // Undo the isotropic rescaling on the recovered focal; the pose (and thus the
            // essential matrix) is scale-invariant. The solver may return degenerate
            // solutions, so guard the focal defensively.
            let focal = image_pair.camera1.focal() * scale;
            if !(focal > 0.0) {
                continue;
            }
            let cam2_from_cam1 = convert_poselib_pose_to_rigid3d(&image_pair.pose);
            models.push(RelativePoseSharedFocalModel {
                e: essential_matrix_from_pose(&cam2_from_cam1),
                focal,
            });
        }
        Ok(())
    }

    /// Nonlinear local optimization of the joint 6-DoF pose + shared focal, in place,
    /// starting from `model`. This is the entry point used by LO-RANSAC for local
    /// optimization. Returns true and overwrites `model` with the refined model on success.
    /// On a degenerate decomposition (or non-positive focal) it returns false and leaves
    /// `model` unchanged.
    pub fn refine(
        points1: &[Vector2d],
        points2: &[Vector2d],
        model: &mut RelativePoseSharedFocalModel,
    ) -> Result<bool> {
        check_eq!(points1.len(), points2.len());
        check_ge!(points1.len(), Self::MIN_NUM_SAMPLES);

        if !(model.focal > 0.0) {
            return Ok(false);
        }

        // Decompose the current essential matrix into a relative pose, resolving the
        // four-fold ambiguity by cheirality over the focal-calibrated sample rays.
        let cam_rays1 = calibrated_rays(points1, model.focal);
        let cam_rays2 = calibrated_rays(points2, model.focal);
        let (cam2_from_cam1, valid_indices) =
            pose_from_essential_matrix(&model.e, &cam_rays1, &cam_rays2)?;
        if valid_indices.is_empty() {
            // Degenerate configuration: leave the initial model unchanged.
            return Ok(false);
        }

        // Nonlinear pixel-space Sampson refinement of the joint 6-DoF (5-DoF pose plus shared
        // focal) via TinySolver, applying the shared-focal relative pose manifold. Plain
        // least squares: the points are assumed to be the inlier set, so robustness comes
        // from the RANSAC inlier selection.
        let functor = TinyFocalSampsonErrorCostFunctor::new(points1, points2)?;
        if let Some((e, focal)) = refine_pose_and_focal(&functor, &cam2_from_cam1, model.focal)? {
            model.e = e;
            model.focal = focal;
        }
        Ok(true)
    }

    /// Squared pixel-space Sampson error of each centered point pair under the fundamental
    /// matrix implied by the model (F = Kinv * E * Kinv), into `residuals` (resized);
    /// `f64::MAX` for every pair when the focal is not positive.
    pub fn residuals(
        points1: &[Vector2d],
        points2: &[Vector2d],
        model: &RelativePoseSharedFocalModel,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        check_eq!(points1.len(), points2.len());
        if !(model.focal > 0.0) {
            residuals.clear();
            residuals.resize(points1.len(), f64::MAX);
            return Ok(());
        }
        let f = fundamental_from_essential_shared_focal(&model.e, model.focal);
        *residuals = compute_squared_sampson_errors(points1, points2, &f)?;
        Ok(())
    }

    /// Whether the focal recovered for the given pose should be treated as reliable, i.e.
    /// whether the pose is clear of the singular family described on the type. Tested as two
    /// predicates: the axes are sufficiently skew, or, failing that, sufficiently far from
    /// isosceles.
    pub fn is_focal_identifiable(cam2_from_cam1: &Rigid3d) -> bool {
        // Minimum sine of the angle between the baseline and the plane of the two optical
        // axes for the axes to count as skew. Skew axes cannot be singular, so this admits
        // them outright. Turntable and object-scan capture, the common near-coplanar case,
        // falls below it and is deferred to the second predicate.
        const K_MIN_AXES_SKEW: f64 = 0.05;
        // Minimum relative difference between the distances of the two camera centers from
        // the intersection of their optical axes, for near-coplanar axes to count as clear of
        // the isosceles singularity. Also rejects near-parallel axes, whose distances diverge
        // together. Both thresholds sit at the knee of a synthetic focal-accuracy sweep.
        const K_MIN_ISOSCELES_DEVIATION: f64 = 0.05;

        // Skew optical axes neither intersect nor are parallel, so they can never be singular
        // and the second predicate need not be consulted.
        if axes_skewness(cam2_from_cam1) > K_MIN_AXES_SKEW {
            return true;
        }
        // The axes are (near-)coplanar: singular only if they additionally are
        // (near-)parallel, or intersect with the two centers (near-)equidistant from the
        // intersection point. A zero baseline (pure rotation) lands here with both distances
        // zero and is rejected.
        isosceles_deviation(cam2_from_cam1) > K_MIN_ISOSCELES_DEVIATION
    }
}

impl Estimator for RelativePoseSharedFocalEstimator {
    type X = Vector2d;
    type Y = Vector2d;
    type M = RelativePoseSharedFocalModel;
    const MIN_NUM_SAMPLES: usize = 6;

    fn estimate(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        models: &mut Vec<RelativePoseSharedFocalModel>,
    ) -> Result<()> {
        Self::estimate(points1, points2, models)
    }

    fn residuals(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        model: &RelativePoseSharedFocalModel,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        Self::residuals(points1, points2, model, residuals)
    }
}

impl LocalEstimator for RelativePoseSharedFocalEstimator {
    type X = Vector2d;
    type Y = Vector2d;
    type M = RelativePoseSharedFocalModel;
    const MIN_NUM_SAMPLES: usize = 6;

    /// Refines a copy of `initial_model` and pushes it when [`Self::refine`] succeeds.
    fn estimate_local(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        initial_model: &RelativePoseSharedFocalModel,
        models: &mut Vec<RelativePoseSharedFocalModel>,
    ) -> Result<()> {
        let mut model = *initial_model;
        if Self::refine(points1, points2, &mut model)? {
            models.push(model);
        }
        Ok(())
    }

    fn residuals(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        model: &RelativePoseSharedFocalModel,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        Self::residuals(points1, points2, model, residuals)
    }
}
