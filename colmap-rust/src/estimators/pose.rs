//! Port of the robust estimation half of COLMAP's `colmap/estimators/pose.h` and `pose.cc`:
//! [`AbsolutePoseEstimationOptions`], [`estimate_absolute_pose`] (P3P + EPnP LO-RANSAC, or
//! P4Pf RANSAC when the focal length is estimated) and [`estimate_relative_pose`] (five-point
//! essential matrix LO-RANSAC on the tangent Sampson error, then the cheirality-resolved
//! decomposition). Built on [`crate::estimators::solvers`] (`absolute_pose`, `epnp`,
//! `essential_matrix`) and [`crate::geometry::essential_matrix`]. Port of colmap-sharp's
//! `Estimators/PoseEstimation.cs`. Tests: `tests/estimators/pose.rs` (pose_test.cc, the
//! `EstimateAbsolutePose` and `EstimateRelativePose` cases).
//!
//! Not here yet: `AbsolutePoseRefinementOptions`, `RefineAbsolutePose`, `RefineRelativePose`
//! and `RefineEssentialMatrix`, which run Ceres problems (the bundle adjustment phase).
//!
//! Tier C (RANSAC) throughout.
//!
//! Translation notes:
//! - COLMAP returns `bool` and writes the pose, the inlier count and the mask through
//!   pointers; here the functions return `Ok(None)` for COLMAP's `false` and the outputs in
//!   [`PoseEstimate`] otherwise. COLMAP's `EstimateRelativePose` also writes the pose, count
//!   and mask on its last `false` (no correspondence passes cheirality); no caller reads
//!   them then, so they are dropped with the `None`.
//! - `ImgFromCamFunc` captures `camera` by pointer in COLMAP; the callback here owns a clone
//!   of the camera, which the non-focal branch never modifies.

use std::sync::Arc;

use crate::estimators::solvers::{
    EPNPEstimator, EssentialMatrixTangentSampsonEstimator, ImgFromCamFunc, P3PEstimator,
    P4PFEstimator, Point2DWithRay,
};
use crate::geometry::essential_matrix::pose_from_essential_matrix;
use crate::geometry::pose::CamRayWithJac;
use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{Matrix3x4d, Quaterniond, Vector2d, Vector3d};
use crate::optim::{EstimateAsLocal, LoRansac, Ransac, RansacOptions};
use crate::scene::camera::Camera;
use crate::{check_eq, Result};

/// Port of `colmap::AbsolutePoseEstimationOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbsolutePoseEstimationOptions {
    /// Whether to estimate the focal length.
    pub estimate_focal_length: bool,
    /// Options used for P3P RANSAC.
    pub ransac_options: RansacOptions,
}

impl Default for AbsolutePoseEstimationOptions {
    fn default() -> Self {
        Self {
            estimate_focal_length: false,
            ransac_options: RansacOptions {
                max_error: 12.0,
                // Use high confidence to avoid preemptive termination of P3P RANSAC - too
                // early termination may lead to bad registration.
                min_num_trials: 100,
                max_num_trials: 10000,
                confidence: 0.99999,
                ..RansacOptions::default()
            },
        }
    }
}

impl AbsolutePoseEstimationOptions {
    /// Port of `AbsolutePoseEstimationOptions::Check`.
    pub fn check(&self) -> Result<()> {
        self.ransac_options.check()
    }
}

/// The outputs of a successful [`estimate_absolute_pose`] or [`estimate_relative_pose`]:
/// COLMAP's pose, `num_inliers` and `inlier_mask` out-parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseEstimate {
    /// The estimated pose: `cam_from_world` for the absolute pose, `cam2_from_cam1` for the
    /// relative pose.
    pub pose: Rigid3d,
    /// Number of inliers in RANSAC.
    pub num_inliers: usize,
    /// Inlier mask of the correspondences.
    pub inlier_mask: Vec<bool>,
}

// `Rigid3d(Eigen::Quaterniond(model.leftCols<3>()), model.col(3))`.
fn rigid3d_from_matrix(model: &Matrix3x4d) -> Rigid3d {
    Rigid3d::new(
        Quaterniond::from_rotation_matrix(model.left_cols3()),
        model.col(3),
    )
}

/// Port of `colmap::EstimateAbsolutePose`: estimate the absolute pose (optionally the focal
/// length) from 2D-3D correspondences.
///
/// Focal length estimation runs P4Pf RANSAC on the points centered at the camera's
/// principal point; the estimated focal length(s) are written to `camera` in place.
/// Otherwise P3P LO-RANSAC with EPnP local optimization, projecting through `camera`.
/// `Ok(None)` when no pose is found.
pub fn estimate_absolute_pose(
    options: &AbsolutePoseEstimationOptions,
    points2d: &[Vector2d],
    points3d: &[Vector3d],
    camera: &mut Camera,
) -> Result<Option<PoseEstimate>> {
    check_eq!(points2d.len(), points3d.len());
    options.check()?;

    if options.estimate_focal_length {
        // TODO(jsch): Implement non-minimal solver for LORANSAC refinement. Experiments
        // showed marginal difference between RANSAC/LORANSAC for PNPF after refining the
        // estimates of this function using RefineAbsolutePose.
        let principal_point =
            Vector2d::new(camera.principal_point_x()?, camera.principal_point_y()?);
        let points2d_centered: Vec<Vector2d> = points2d
            .iter()
            .map(|&point2d| point2d - principal_point)
            .collect();
        let focal_length_idxs = camera.focal_length_idxs()?;
        let mut ransac: Ransac<P4PFEstimator> = Ransac::new(
            &options.ransac_options,
            P4PFEstimator::new(focal_length_idxs.len() == 1),
        )?;
        let report = ransac.estimate(&points2d_centered, points3d)?;
        if report.success {
            let focal_lengths = [report.model.focal_lengths.x, report.model.focal_lengths.y];
            for (k, &idx) in focal_length_idxs.iter().enumerate() {
                camera.params[idx] = focal_lengths[k];
            }
            return Ok(Some(PoseEstimate {
                pose: rigid3d_from_matrix(&report.model.cam_from_world),
                num_inliers: report.support.num_inliers,
                inlier_mask: report.inlier_mask,
            }));
        }
    } else {
        let points2d_with_rays: Vec<Point2DWithRay> = points2d
            .iter()
            .map(|&point2d| {
                Point2DWithRay::new(
                    point2d,
                    camera
                        .cam_ray_from_img(point2d)
                        .unwrap_or(Vector3d::new(0.0, 0.0, 0.0)),
                )
            })
            .collect();

        let projector = camera.clone();
        let img_from_cam_func: ImgFromCamFunc =
            Arc::new(move |cam_point| projector.img_from_cam(cam_point, true));
        let mut ransac: LoRansac<P3PEstimator, EstimateAsLocal<EPNPEstimator>> = LoRansac::new(
            &options.ransac_options,
            P3PEstimator::new(img_from_cam_func.clone()),
            EstimateAsLocal(EPNPEstimator::new(img_from_cam_func)),
        )?;
        let report = ransac.estimate(&points2d_with_rays, points3d)?;
        if report.success {
            return Ok(Some(PoseEstimate {
                pose: rigid3d_from_matrix(&report.model),
                num_inliers: report.support.num_inliers,
                inlier_mask: report.inlier_mask,
            }));
        }
    }

    Ok(None)
}

/// Port of `colmap::EstimateRelativePose`: estimate the relative pose `cam2_from_cam1` from
/// 2D-2D correspondences. The first camera is at the origin without rotation; the second
/// pose is the world-to-image transformation, i.e. `x2 = [R | t] * X2`.
///
/// Inliers are scored with the pixel-unit tangent Sampson error, so each ray comes with its
/// unprojection Jacobian d(ray)/d(pixel) (`Camera::cam_ray_from_img_with_jac`);
/// `ransac_options.max_error` is in pixels. `Ok(None)` when RANSAC fails, the decomposition
/// is not finite, or no inlier passes the cheirality test.
pub fn estimate_relative_pose(
    ransac_options: &RansacOptions,
    cam_rays1_with_jac: &[CamRayWithJac],
    cam_rays2_with_jac: &[CamRayWithJac],
) -> Result<Option<PoseEstimate>> {
    check_eq!(cam_rays1_with_jac.len(), cam_rays2_with_jac.len());

    let mut ransac: LoRansac<
        EssentialMatrixTangentSampsonEstimator,
        EssentialMatrixTangentSampsonEstimator,
    > = LoRansac::new(
        ransac_options,
        EssentialMatrixTangentSampsonEstimator,
        EssentialMatrixTangentSampsonEstimator,
    )?;
    let report = ransac.estimate(cam_rays1_with_jac, cam_rays2_with_jac)?;

    if !report.success {
        return Ok(None);
    }

    let mut inlier_cam_rays1 = Vec::with_capacity(report.support.num_inliers);
    let mut inlier_cam_rays2 = Vec::with_capacity(report.support.num_inliers);
    for ((ray1, ray2), &is_inlier) in cam_rays1_with_jac
        .iter()
        .zip(cam_rays2_with_jac)
        .zip(&report.inlier_mask)
    {
        if is_inlier {
            inlier_cam_rays1.push(ray1.ray);
            inlier_cam_rays2.push(ray2.ray);
        }
    }

    let (cam2_from_cam1, valid_indices) =
        pose_from_essential_matrix(&report.model, &inlier_cam_rays1, &inlier_cam_rays2)?;

    let rotation = cam2_from_cam1.rotation.coeffs();
    let translation = cam2_from_cam1.translation;
    if [rotation.x, rotation.y, rotation.z, rotation.w]
        .iter()
        .chain(&[translation.x, translation.y, translation.z])
        .any(|v| v.is_nan())
    {
        return Ok(None);
    }

    if valid_indices.is_empty() {
        return Ok(None);
    }
    Ok(Some(PoseEstimate {
        pose: cam2_from_cam1,
        num_inliers: report.support.num_inliers,
        inlier_mask: report.inlier_mask,
    }))
}
