//! Port of COLMAP's `colmap/estimators/solvers/generalized_relative_pose.h` and `.cc` (the
//! GR6P half): [`GRNPObservation`] and [`GR6PEstimator`], the minimal relative pose
//! `rig2_from_rig1` between two generalized (multi-camera) cameras from six 2D-2D
//! correspondences. It solves with PoseLib's [`gen_relpose_6pt`] on rays and camera origins
//! expressed in the rig frames, and scores each correspondence with the squared tangent
//! Sampson error (in pixels) of [`crate::geometry::essential_matrix`]. RANSAC runs it through
//! [`crate::optim::Estimator`]. Port of colmap-sharp's
//! `Estimators/Solvers/GeneralizedRelativePose.cs`. Tests:
//! `tests/estimators/generalized_relative_pose.rs` (generalized_relative_pose_test.cc, the
//! GR6P cases).
//!
//! Not here yet: `GR8PEstimator` (Kneip and Li's eight-point eigenvalue solver, the rest of
//! `generalized_relative_pose.cc`), which will share [`GRNPObservation`].
//!
//! Tier B for the solver (an eigenvalue problem, docs/CPP_DIVERGENCES.md entry 256); the
//! RANSAC runs of the tests are Tier C.

use super::poselib::gen_relpose_6pt::gen_relpose_6pt;
use super::poselib_utils::convert_poselib_pose_to_rigid3d;
use crate::geometry::essential_matrix::{
    compute_squared_tangent_sampson_error, essential_matrix_from_pose,
};
use crate::geometry::{CamRayWithJac, Rigid3d};
use crate::linalg::Vector3d;
use crate::optim::Estimator;
use crate::{check_eq, Result};

/// One observation of a generalized camera: the pose of the observing camera in the rig and
/// its bearing. Port of `colmap::GRNPObservation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GRNPObservation {
    /// The pose of the observing camera in the rig.
    pub cam_from_rig: Rigid3d,
    /// Bearing in the camera frame, bundled with its Jacobian d(ray) / d(pixel) so
    /// `Residuals` can score in pixel units with the tangent Sampson error. Only the
    /// residual uses the Jacobian; the solvers read the ray alone.
    pub ray_with_jac_in_cam: CamRayWithJac,
}

/// Minimal generalized relative pose estimator based on poselib. The model is the
/// `rig2_from_rig1` relative pose between the generalized cameras. Port of
/// `colmap::GR6PEstimator`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GR6PEstimator;

impl Estimator for GR6PEstimator {
    type X = GRNPObservation;
    type Y = GRNPObservation;
    type M = Rigid3d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 6;

    /// Estimate the possible solutions of the GR6P problem from exactly six 2D-2D point
    /// correspondences.
    fn estimate(
        &mut self,
        points1: &[GRNPObservation],
        points2: &[GRNPObservation],
        rigs2_from_rigs1: &mut Vec<Rigid3d>,
    ) -> Result<()> {
        check_eq!(points1.len(), 6);
        check_eq!(points2.len(), 6);

        rigs2_from_rigs1.clear();

        let mut origins_in_rig1 = [Vector3d::zeros(); 6];
        let mut origins_in_rig2 = [Vector3d::zeros(); 6];
        let mut rays_in_rig1 = [Vector3d::zeros(); 6];
        let mut rays_in_rig2 = [Vector3d::zeros(); 6];
        for i in 0..6 {
            origins_in_rig1[i] = points1[i].cam_from_rig.tgt_origin_in_src();
            origins_in_rig2[i] = points2[i].cam_from_rig.tgt_origin_in_src();
            rays_in_rig1[i] =
                points1[i].cam_from_rig.rotation.inverse() * points1[i].ray_with_jac_in_cam.ray;
            rays_in_rig2[i] =
                points2[i].cam_from_rig.rotation.inverse() * points2[i].ray_with_jac_in_cam.ray;
        }

        let mut poses = Vec::new();
        gen_relpose_6pt(
            &origins_in_rig1,
            &rays_in_rig1,
            &origins_in_rig2,
            &rays_in_rig2,
            &mut poses,
        );

        rigs2_from_rigs1.reserve(poses.len());
        for pose in &poses {
            rigs2_from_rigs1.push(convert_poselib_pose_to_rigid3d(pose));
        }
        Ok(())
    }

    /// The squared tangent Sampson error (in pixels) between corresponding points.
    fn residuals(
        &mut self,
        points1: &[GRNPObservation],
        points2: &[GRNPObservation],
        rig2_from_rig1: &Rigid3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        check_eq!(points1.len(), points2.len());
        residuals.resize(points1.len(), 0.0);
        // Score in pixel units with the tangent Sampson error. Each correspondence
        // has its own essential matrix, formed from the two cameras' rig poses, and
        // is evaluated in each camera's own frame, where that camera's ray Jacobian
        // lives - so no rig-frame rotation of the rays or Jacobians is needed.
        for i in 0..points1.len() {
            let cam2_from_cam1 =
                points2[i].cam_from_rig * *rig2_from_rig1 * points1[i].cam_from_rig.inverse();
            let e = essential_matrix_from_pose(&cam2_from_cam1);
            residuals[i] = compute_squared_tangent_sampson_error(
                &points1[i].ray_with_jac_in_cam,
                &points2[i].ray_with_jac_in_cam,
                &e,
            );
        }
        Ok(())
    }
}
