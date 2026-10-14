//! Port of COLMAP's `src/colmap/estimators/solvers/generalized_absolute_pose.h` and `.cc`:
//! [`GP3PEstimator`], the minimal absolute pose of a generalized (multi-camera) camera, i.e.
//! the `rig_from_world` pose, from three 2D-3D correspondences observed by cameras at known
//! poses in the rig. It solves with PoseLib's [`gp3p`], falling back to PoseLib's [`p3p`]
//! when all three rays share one origin (a panoramic rig), where gp3p's translation
//! elimination is degenerate. RANSAC runs it through [`crate::optim::Estimator`]. Port of
//! colmap-sharp's `Estimators/Solvers/GeneralizedAbsolutePose.cs`. Tests:
//! `tests/estimators/generalized_absolute_pose.rs` (`generalized_absolute_pose_test.cc` 1:1).
//!
//! Tier B for the solver; the RANSAC runs of the tests are Tier C.
//!
//! Translation notes:
//! - `GP3PEstimator::X_t` is the struct [`GP3PObservation`] and `ResidualType` is
//!   [`GP3PResidualType`].
//! - COLMAP's `LOG(FATAL_THROW)` for an invalid residual type has no counterpart: the Rust
//!   enum has no invalid value.
//! - `Residuals` resizes with `resize(n, 0)`, which keeps stale values in a reused vector,
//!   but every slot is written below, as in C++.

use super::poselib::gp3p::gp3p;
use super::poselib::p3p::p3p;
use super::poselib::CameraPose;
use super::poselib_utils::convert_poselib_pose_to_rigid3d;
use crate::geometry::Rigid3d;
use crate::linalg::{Matrix3x4d, Vector3d, MACHINE_EPSILON};
use crate::optim::Estimator;
use crate::{check_eq, Result};

/// The generalized image observation: the relative pose of a camera in the generalized
/// camera and a ray in the camera frame. Port of `colmap::GP3PEstimator::X_t`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GP3PObservation {
    /// The pose of the observing camera in the rig, stored as matrix for fast residuals.
    pub cam_from_rig: Matrix3x4d,
    /// The ray in the camera frame.
    pub ray_in_cam: Vector3d,
}

/// Whether to compute the cosine similarity or the reprojection error.
/// \[WARNING\] The reprojection error being in normalized coordinates, the unique error
/// threshold of RANSAC corresponds to different pixel values in the different cameras of the
/// rig if they have different intrinsics. Port of `colmap::GP3PEstimator::ResidualType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GP3PResidualType {
    /// `1 - cos` of the angle between the observed ray and the point, squared.
    #[default]
    CosineDistance,
    /// The squared reprojection error on the normalized image plane.
    ReprojectionError,
}

/// Minimal generalized absolute pose estimator. Port of `colmap::GP3PEstimator`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GP3PEstimator {
    residual_type: GP3PResidualType,
}

impl GP3PEstimator {
    /// An estimator scoring with `residual_type` (COLMAP's default is
    /// [`GP3PResidualType::CosineDistance`], also this type's `Default`).
    pub const fn new(residual_type: GP3PResidualType) -> Self {
        Self { residual_type }
    }

    /// The residual type the estimator scores with.
    pub const fn residual_type(&self) -> GP3PResidualType {
        self.residual_type
    }
}

impl Estimator for GP3PEstimator {
    /// The generalized image observations.
    type X = GP3PObservation;
    /// The observed 3D feature points in the world frame.
    type Y = Vector3d;
    /// The estimated `rig_from_world` pose of the generalized camera.
    type M = Rigid3d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 3;

    /// Estimate the most probable solution of the GP3P problem from a set of three 2D-3D
    /// point correspondences.
    fn estimate(
        &mut self,
        points2d: &[GP3PObservation],
        points3d: &[Vector3d],
        rigs_from_world: &mut Vec<Rigid3d>,
    ) -> Result<()> {
        check_eq!(points2d.len(), 3);
        check_eq!(points3d.len(), 3);

        rigs_from_world.clear();

        let mut rays_in_rig = [Vector3d::zeros(); 3];
        let mut origins_in_rig = [Vector3d::zeros(); 3];
        for i in 0..3 {
            let rig_from_cam_rotation = points2d[i].cam_from_rig.left_cols3().transpose();
            rays_in_rig[i] = (rig_from_cam_rotation * points2d[i].ray_in_cam).normalized();
            origins_in_rig[i] = rig_from_cam_rotation * -points2d[i].cam_from_rig.col(3);
        }

        let mut poses: Vec<CameraPose> = Vec::with_capacity(8);
        if origins_in_rig[0].is_approx_with(origins_in_rig[1], 1e-6)
            && origins_in_rig[0].is_approx_with(origins_in_rig[2], 1e-6)
        {
            // In case of a panoramic camera/rig, fall back to P3P.
            p3p(&rays_in_rig, points3d, &mut poses);
            for pose in &mut poses {
                pose.t += origins_in_rig[0];
            }
        } else {
            gp3p(&origins_in_rig, &rays_in_rig, points3d, &mut poses);
        }

        rigs_from_world.extend(poses.iter().map(convert_poselib_pose_to_rigid3d));
        Ok(())
    }

    /// Calculate the squared cosine distance error between the rays given a set of 2D-3D
    /// point correspondences and the rig pose of the generalized camera.
    fn residuals(
        &mut self,
        points2d: &[GP3PObservation],
        points3d: &[Vector3d],
        rig_from_world: &Rigid3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        check_eq!(points2d.len(), points3d.len());
        residuals.resize(points2d.len(), 0.0);

        // Precompute matrix to avoid repeated quaternion-to-matrix conversion.
        let rig_from_world_matrix = rig_from_world.to_matrix();

        for i in 0..points2d.len() {
            let point3d_in_cam = points2d[i].cam_from_rig
                * (rig_from_world_matrix * points3d[i].homogeneous()).homogeneous();

            // Check if 3D point is in front of camera.
            if point3d_in_cam.z > MACHINE_EPSILON {
                let ray = points2d[i].ray_in_cam;

                residuals[i] = match self.residual_type {
                    GP3PResidualType::CosineDistance => {
                        let cosine_dist = 1.0 - point3d_in_cam.normalized().dot(ray.normalized());
                        cosine_dist * cosine_dist
                    }
                    GP3PResidualType::ReprojectionError => {
                        let diff = ray.hnormalized() - point3d_in_cam.hnormalized();
                        diff.squared_norm()
                    }
                };
            } else {
                residuals[i] = f64::MAX;
            }
        }
        Ok(())
    }
}
