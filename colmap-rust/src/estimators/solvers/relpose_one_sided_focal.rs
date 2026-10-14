//! Port of COLMAP's `colmap/estimators/solvers/relpose_one_sided_focal.h` / `.cc`, via
//! colmap-sharp's `Estimators/Solvers/RelativePoseOneSidedFocal.cs`: the relative pose
//! estimator with one unknown focal length (the first view's), the second view calibrated.
//! The minimal solve is PoseLib's [`relpose_6pt_onesided_focal`]; the LO-RANSAC refiner is
//! [`crate::optim::TinySolver`] on [`TinyOneSidedFocalTangentSampsonErrorCostFunctor`] over
//! the pose-plus-log-focal manifold (shared with [`super::relpose_shared_focal`], with its
//! `calibrated_rays` and `refine_pose_and_focal`); residuals are the tangent Sampson error in
//! pixels under M = E * K1inv. Tests: `tests/estimators/relpose_one_sided_focal.rs`
//! (`relpose_one_sided_focal_test.cc`).
//!
//! Tier B for `estimate` and `residuals`, Tier C for `refine`.

// `!(x > 0.0)` guards, as written in C++: a NaN focal, scale or denominator fails them too.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use super::poselib::relpose_6pt_onesided_focal::relpose_6pt_onesided_focal;
use super::poselib_utils::convert_poselib_pose_to_rigid3d;
use super::relpose_shared_focal::{calibrated_rays, refine_pose_and_focal};
use crate::estimators::cost_functions::TinyOneSidedFocalTangentSampsonErrorCostFunctor;
use crate::geometry::essential_matrix::{
    essential_matrix_from_pose, pose_from_essential_matrix, squared_pixel_gradient_norm,
};
use crate::geometry::pose::CamRayWithJac;
use crate::linalg::{Matrix3d, Vector2d, Vector3d};
use crate::optim::{Estimator, LocalEstimator};
use crate::{check_eq, check_ge, Result};

/// The estimated model of [`RelativePoseOneSidedFocalEstimator`] (C++'s `M_t`): a calibrated
/// essential matrix plus the recovered focal length of the first (uncalibrated) view.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RelativePoseOneSidedFocalModel {
    /// The calibrated essential matrix.
    pub e: Matrix3d,
    /// The first view's focal length, in pixels.
    pub focal: f64,
}

impl Default for RelativePoseOneSidedFocalModel {
    /// C++'s default: E = identity, focal = 0.
    fn default() -> Self {
        Self {
            e: Matrix3d::identity(),
            focal: 0.0,
        }
    }
}

/// Mixed epipolar matrix M = E * K1inv with K1inv = diag(1/f1, 1/f1, 1), which relates image
/// points of the uncalibrated view to bearing rays of the calibrated one:
/// ray2^T M point1 = 0. Eigen's diagonal product scales the columns with no further
/// arithmetic.
fn mixed_epipolar_matrix(e: &Matrix3d, focal1: f64) -> Matrix3d {
    let k1_inv = [1.0 / focal1, 1.0 / focal1, 1.0];
    let mut m = *e;
    for row in 0..3 {
        for col in 0..3 {
            m[(row, col)] = e[(row, col)] * k1_inv[col];
        }
    }
    m
}

/// Port of `colmap::RelativePoseOneSidedFocalEstimator`: relative pose with one unknown focal
/// length, wrapping PoseLib's minimal 6-point one-sided focal solver.
///
/// The first view's focal is unknown and recovered, the second view's is known. Treating such
/// a pair as fully uncalibrated (a plain fundamental matrix) would discard the known
/// intrinsics of the calibrated side. Unlike the shared-focal solver it has no singularity at
/// coplanar optical axes, as the calibrated view fixes the scale, so it needs no
/// identifiability score.
///
/// Inputs are asymmetric: X are principal-point-centered points of the uncalibrated view
/// (u - cx), in pixels, since its focal is the unknown. Y are calibrated bearing rays of the
/// second view with their unprojection Jacobians, so any camera model is admissible there,
/// including non-pinhole ones and rays with z <= 0. The epipolar constraint is
/// ray2^T M point1 = 0 with M = E * diag(1/f1, 1/f1, 1).
///
/// Residuals are squared tangent Sampson errors, in pixels. Measuring only the first view's
/// distance to the epipolar line would minimize over the subspace where the second view's
/// point is held fixed, which upper-bounds the true reprojection error by an amount that
/// grows with the second camera's field of view. A *plain* Sampson error is indeed ill-posed
/// here, since it combines the two views' gradients in different units (pixels and rays), but
/// the second view's unprojection Jacobian is exactly the conversion that repairs this. See
/// `compute_squared_tangent_sampson_error`.
///
/// X stays in raw pixels rather than becoming a ray with a Jacobian, and that asymmetry is
/// load-bearing: it keeps the first view's measurement Jacobian at the constant [I2; 0] and
/// confines f1 to M. Unprojecting the first view would place f1 inside its Jacobian, and
/// refining over f1 would then require the mixed second derivative
/// d^2(ray1)/d(pixel1)d(f1), which no camera model exposes.
///
/// The estimator serves as both the global (minimal) and local (refinement) estimator inside
/// LO-RANSAC: as a [`LocalEstimator`] it refines a copy of the current best model
/// ([`Self::refine`], C++'s `Refine` hook). This is required because the minimal solver
/// consumes exactly six points and has no non-minimal least-squares counterpart.
#[derive(Clone, Copy, Debug, Default)]
pub struct RelativePoseOneSidedFocalEstimator;

impl RelativePoseOneSidedFocalEstimator {
    /// The minimum number of samples needed to estimate a model.
    pub const MIN_NUM_SAMPLES: usize = 6;

    /// Estimates relative pose and the first view's focal from >= 6 correspondences by
    /// wrapping `poselib::relpose_6pt_onesided_focal` (uses the first six). Clears `models`.
    pub fn estimate(
        img_points1: &[Vector2d],
        cam_rays2_with_jac: &[CamRayWithJac],
        models: &mut Vec<RelativePoseOneSidedFocalModel>,
    ) -> Result<()> {
        check_eq!(img_points1.len(), cam_rays2_with_jac.len());
        check_ge!(img_points1.len(), Self::MIN_NUM_SAMPLES);
        models.clear();

        // Unlike poselib::relpose_6pt_shared_focal, this solver conditions its own input: it
        // isotropically rescales the uncalibrated points internally and undoes that scaling
        // on the recovered focal, which therefore comes back in the pixel units of
        // `img_points1`. So pass the centered points through as plain homogeneous coordinates
        // and do not pre-normalize them here.
        let img_points1_homogeneous: Vec<Vector3d> =
            img_points1.iter().map(|p| p.homogeneous()).collect();
        // Already unit bearings, which keeps the nullspace computation well conditioned.
        // Renormalizing here would turn a zeroed entry, left behind by a failed unprojection,
        // into NaN and poison the whole sample.
        let cam_rays2: Vec<Vector3d> = cam_rays2_with_jac.iter().map(|r| r.ray).collect();

        // C++ passes use_elim = false: the full elimination template, which solves for the
        // focal directly as an unknown. The default compact template instead recovers it from
        // the fundamental matrix via a semi-calibrated Kruppa step, which is numerically
        // fragile and loses accuracy even on exact points, propagating into E. The larger
        // template costs only marginally more runtime, as both variants end in the same
        // eigendecomposition, which dominates. [`relpose_6pt_onesided_focal`] ports only that
        // template, so it takes no flag.
        let mut image_pairs = Vec::new();
        relpose_6pt_onesided_focal(&img_points1_homogeneous, &cam_rays2, &mut image_pairs);

        models.reserve(image_pairs.len());
        for image_pair in &image_pairs {
            // The solver may return degenerate solutions, so guard the focal defensively.
            let focal = image_pair.camera1.focal();
            if !(focal > 0.0) {
                continue;
            }
            let cam2_from_cam1 = convert_poselib_pose_to_rigid3d(&image_pair.pose);
            models.push(RelativePoseOneSidedFocalModel {
                e: essential_matrix_from_pose(&cam2_from_cam1),
                focal,
            });
        }
        Ok(())
    }

    /// Nonlinear local optimization of the joint 6-DoF pose plus the unknown focal, in place,
    /// starting from `model`. This is the entry point used by LO-RANSAC for local
    /// optimization. Returns true and overwrites `model` with the refined model on success.
    /// On a degenerate decomposition (or non-positive focal) it returns false and leaves
    /// `model` unchanged.
    pub fn refine(
        img_points1: &[Vector2d],
        cam_rays2_with_jac: &[CamRayWithJac],
        model: &mut RelativePoseOneSidedFocalModel,
    ) -> Result<bool> {
        check_eq!(img_points1.len(), cam_rays2_with_jac.len());
        check_ge!(img_points1.len(), Self::MIN_NUM_SAMPLES);

        if !(model.focal > 0.0) {
            return Ok(false);
        }

        // Decompose the current essential matrix into a relative pose, resolving the
        // four-fold ambiguity by cheirality. The first view's rays follow from the current
        // focal estimate; the second view's are already calibrated. Only the bearings are
        // materialized, since that is all pose_from_essential_matrix needs.
        let cam_rays1 = calibrated_rays(img_points1, model.focal);
        let cam_rays2: Vec<Vector3d> = cam_rays2_with_jac.iter().map(|r| r.ray).collect();
        let (cam2_from_cam1, valid_indices) =
            pose_from_essential_matrix(&model.e, &cam_rays1, &cam_rays2)?;
        if valid_indices.is_empty() {
            // Degenerate configuration: leave the initial model unchanged.
            return Ok(false);
        }

        // Nonlinear refinement of the joint 6-DoF (5-DoF pose plus the unknown focal) via
        // TinySolver (analytic Jacobians), applying the one-sided focal relative pose
        // manifold. The functor minimizes the same tangent Sampson error that `residuals`
        // scores, so local optimization improves exactly the quantity LO-RANSAC measures.
        // Plain least squares: the points are assumed to be the inlier set, so robustness
        // comes from the RANSAC inlier selection.
        let functor =
            TinyOneSidedFocalTangentSampsonErrorCostFunctor::new(img_points1, cam_rays2_with_jac)?;
        if let Some((e, focal)) = refine_pose_and_focal(&functor, &cam2_from_cam1, model.focal)? {
            model.e = e;
            model.focal = focal;
        }
        Ok(true)
    }

    /// Squared tangent Sampson error, in pixels, of each correspondence under M = E * K1inv,
    /// into `residuals` (resized); `f64::MAX` for every pair when the focal is not positive,
    /// and for a pair whose constraint gradients vanish (e.g. a zeroed ray left by a failed
    /// unprojection).
    pub fn residuals(
        img_points1: &[Vector2d],
        cam_rays2_with_jac: &[CamRayWithJac],
        model: &RelativePoseOneSidedFocalModel,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        check_eq!(img_points1.len(), cam_rays2_with_jac.len());
        if !(model.focal > 0.0) {
            residuals.clear();
            residuals.resize(img_points1.len(), f64::MAX);
            return Ok(());
        }
        let m = mixed_epipolar_matrix(&model.e, model.focal);
        let m_transpose = m.transpose();
        residuals.resize(img_points1.len(), 0.0);
        for (i, (point1, cam_ray2_with_jac)) in
            img_points1.iter().zip(cam_rays2_with_jac).enumerate()
        {
            let cam_ray2 = cam_ray2_with_jac.ray;
            let m_point1 = m * point1.homogeneous();
            let num = cam_ray2.dot(m_point1);
            // Constraint gradients in each view's own pixels. The first view needs no
            // Jacobian, as its d(x, y, 1)/d(x, y) merely selects the first two rows. This is
            // compute_squared_tangent_sampson_error specialized to that structure, which is
            // worth inlining here because RANSAC evaluates it per hypothesis per
            // correspondence.
            let mt_ray2 = m_transpose * cam_ray2;
            let denom = (mt_ray2.x * mt_ray2.x + mt_ray2.y * mt_ray2.y)
                + squared_pixel_gradient_norm(&cam_ray2_with_jac.jacobian, m_point1);
            if !(denom > 0.0) {
                // Degenerate, e.g. a zeroed entry left by a failed unprojection.
                residuals[i] = f64::MAX;
                continue;
            }
            residuals[i] = num * num / denom;
        }
        Ok(())
    }
}

impl Estimator for RelativePoseOneSidedFocalEstimator {
    type X = Vector2d;
    type Y = CamRayWithJac;
    type M = RelativePoseOneSidedFocalModel;
    const MIN_NUM_SAMPLES: usize = 6;

    fn estimate(
        &mut self,
        img_points1: &[Vector2d],
        cam_rays2_with_jac: &[CamRayWithJac],
        models: &mut Vec<RelativePoseOneSidedFocalModel>,
    ) -> Result<()> {
        Self::estimate(img_points1, cam_rays2_with_jac, models)
    }

    fn residuals(
        &mut self,
        img_points1: &[Vector2d],
        cam_rays2_with_jac: &[CamRayWithJac],
        model: &RelativePoseOneSidedFocalModel,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        Self::residuals(img_points1, cam_rays2_with_jac, model, residuals)
    }
}

impl LocalEstimator for RelativePoseOneSidedFocalEstimator {
    type X = Vector2d;
    type Y = CamRayWithJac;
    type M = RelativePoseOneSidedFocalModel;
    const MIN_NUM_SAMPLES: usize = 6;

    /// Refines a copy of `initial_model` and pushes it when [`Self::refine`] succeeds.
    fn estimate_local(
        &mut self,
        img_points1: &[Vector2d],
        cam_rays2_with_jac: &[CamRayWithJac],
        initial_model: &RelativePoseOneSidedFocalModel,
        models: &mut Vec<RelativePoseOneSidedFocalModel>,
    ) -> Result<()> {
        let mut model = *initial_model;
        if Self::refine(img_points1, cam_rays2_with_jac, &mut model)? {
            models.push(model);
        }
        Ok(())
    }

    fn residuals(
        &mut self,
        img_points1: &[Vector2d],
        cam_rays2_with_jac: &[CamRayWithJac],
        model: &RelativePoseOneSidedFocalModel,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        Self::residuals(img_points1, cam_rays2_with_jac, model, residuals)
    }
}
