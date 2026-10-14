//! Port of COLMAP's `src/colmap/estimators/solvers/absolute_pose.h` and `.cc`, except
//! `EPNPEstimator`, which is in the sibling [`super::epnp`]: the minimal absolute-pose
//! estimators [`P3PEstimator`] (PoseLib's P3P, [`super::poselib::p3p`]) and
//! [`P4PFEstimator`] (PoseLib's P4Pf, [`super::poselib::p4pf`]), the [`Point2DWithRay`]
//! input type, the [`ImgFromCamFunc`] projection callback and
//! [`compute_squared_reprojection_error`], which P3P and EPnP score with. RANSAC runs them
//! through [`crate::optim::Estimator`]. Port of colmap-sharp's
//! `Estimators/Solvers/AbsolutePose.cs`. Tests: `tests/estimators/absolute_pose.rs`
//! (`absolute_pose_test.cc` 1:1).
//!
//! Tier B (minimal solvers); the RANSAC runs of the tests are Tier C.
//!
//! Translation notes:
//! - `ImgFromCamFunc`, a `std::function` returning `std::optional<Eigen::Vector2d>`, is an
//!   `Arc<dyn Fn>` that is `Send + Sync`, so the estimators holding it stay `Clone` (RANSAC
//!   clones its estimator) and can later run on worker threads.
//! - `P4PFEstimator::M_t` is the struct [`P4PFModel`].
//! - COLMAP's `Residuals` methods are `const` or `static`; here they take `&mut self` as the
//!   [`Estimator`] trait requires and leave the estimator untouched.

use std::sync::Arc;

use super::poselib::p3p::p3p;
use super::poselib::p4pf::{p4pf, p4pf_fx_fy};
use super::poselib::CameraPose;
use crate::linalg::{Matrix3x4d, Vector2d, Vector3d, MACHINE_EPSILON};
use crate::optim::Estimator;
use crate::{check_eq, Result};

/// Function mapping 3D point in the camera frame to 2D point in the image.
/// Returns `None` if the point projection is invalid (e.g., behind the camera).
/// Port of `colmap::ImgFromCamFunc`.
pub type ImgFromCamFunc = Arc<dyn Fn(Vector3d) -> Option<Vector2d> + Send + Sync>;

/// A 2D image observation together with its camera ray. Port of `colmap::Point2DWithRay`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Point2DWithRay {
    /// The 2D image point in pixels.
    pub image_point: Vector2d,
    /// The normalized 3D ray direction in the camera frame.
    pub camera_ray: Vector3d,
}

impl Point2DWithRay {
    /// An observation from its image point and camera ray.
    pub const fn new(image_point: Vector2d, camera_ray: Vector3d) -> Self {
        Self {
            image_point,
            camera_ray,
        }
    }
}

/// P3P estimator: the pose `cam_from_world` (3x4) from three 2D-3D correspondences. Port of
/// `colmap::P3PEstimator`.
#[derive(Clone)]
pub struct P3PEstimator {
    img_from_cam_func: ImgFromCamFunc,
}

impl P3PEstimator {
    /// An estimator scoring in pixels through `img_from_cam_func`.
    pub fn new(img_from_cam_func: ImgFromCamFunc) -> Self {
        Self { img_from_cam_func }
    }
}

impl Estimator for P3PEstimator {
    /// The 2D image feature observations.
    type X = Point2DWithRay;
    /// The observed 3D features in the world frame.
    type Y = Vector3d;
    /// The transformation from the world to the camera frame.
    type M = Matrix3x4d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 3;

    /// Estimate the most probable solution of the P3P problem from a set of three 2D-3D
    /// point correspondences.
    fn estimate(
        &mut self,
        points2d: &[Point2DWithRay],
        points3d: &[Vector3d],
        cams_from_world: &mut Vec<Matrix3x4d>,
    ) -> Result<()> {
        check_eq!(points2d.len(), 3);
        check_eq!(points3d.len(), 3);

        cams_from_world.clear();

        let rays = [
            points2d[0].camera_ray,
            points2d[1].camera_ray,
            points2d[2].camera_ray,
        ];

        let mut poses: Vec<CameraPose> = Vec::with_capacity(4);
        let num_poses = p3p(&rays, points3d, &mut poses);

        cams_from_world.extend(poses[..num_poses].iter().map(CameraPose::rt));
        Ok(())
    }

    /// Calculate the squared reprojection error given a set of 2D-3D point correspondences
    /// and a projection matrix.
    fn residuals(
        &mut self,
        points2d: &[Point2DWithRay],
        points3d: &[Vector3d],
        cam_from_world: &Matrix3x4d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        compute_squared_reprojection_error(
            points2d,
            points3d,
            cam_from_world,
            &self.img_from_cam_func,
            residuals,
        )
    }
}

/// The P4Pf model: pose and focal lengths. Port of `colmap::P4PFEstimator::M_t`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct P4PFModel {
    /// The transformation from the world to the camera frame.
    pub cam_from_world: Matrix3x4d,
    /// The focal lengths (fx, fy) of the camera. Equal when the focal length is shared
    /// (e.g. single-focal camera models).
    pub focal_lengths: Vector2d,
}

/// Minimal solver for 6-DOF pose and focal length. The 2D points are expected to be
/// normalized by the principal point. Port of `colmap::P4PFEstimator`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct P4PFEstimator {
    share_focal_length: bool,
}

impl Default for P4PFEstimator {
    /// The default estimator, with a shared focal length (`share_focal_length = true`).
    fn default() -> Self {
        Self::new(true)
    }
}

impl P4PFEstimator {
    /// If `share_focal_length` is true, a single shared focal length is estimated (suitable
    /// for single-focal camera models, e.g. SIMPLE_PINHOLE). Otherwise, separate focal
    /// lengths for x and y are estimated (e.g. PINHOLE, OPENCV).
    pub const fn new(share_focal_length: bool) -> Self {
        Self { share_focal_length }
    }
}

impl Estimator for P4PFEstimator {
    /// The 2D image feature observations, normalized by the principal point.
    type X = Vector2d;
    /// The observed 3D features in the world frame.
    type Y = Vector3d;
    type M = P4PFModel;

    const MIN_NUM_SAMPLES: usize = 4;

    fn estimate(
        &mut self,
        points2d: &[Vector2d],
        points3d: &[Vector3d],
        models: &mut Vec<P4PFModel>,
    ) -> Result<()> {
        check_eq!(points2d.len(), 4);
        check_eq!(points3d.len(), 4);

        models.clear();

        let mut poses = Vec::with_capacity(8);
        let mut focals_x = Vec::with_capacity(8);
        let num_poses;
        let focals_y = if self.share_focal_length {
            // Estimate a single shared focal length. filter_solutions additionally
            // removes solutions whose aspect ratio (fx/fy) is far from 1.
            num_poses = p4pf(points2d, points3d, &mut poses, &mut focals_x, true);
            focals_x.clone()
        } else {
            // Estimate separate focal lengths for x and y. filter_solutions only
            // removes solutions with non-positive focal lengths.
            let mut focals_y = Vec::with_capacity(8);
            num_poses = p4pf_fx_fy(
                points2d,
                points3d,
                &mut poses,
                &mut focals_x,
                &mut focals_y,
                true,
            );
            focals_y
        };

        for i in 0..num_poses {
            models.push(P4PFModel {
                cam_from_world: poses[i].rt(),
                focal_lengths: Vector2d::new(focals_x[i], focals_y[i]),
            });
        }
        Ok(())
    }

    fn residuals(
        &mut self,
        points2d: &[Vector2d],
        points3d: &[Vector3d],
        model: &P4PFModel,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        let num_points2d = points2d.len();
        check_eq!(num_points2d, points3d.len());
        residuals.resize(num_points2d, 0.0);
        for i in 0..num_points2d {
            let point3d_in_cam = model.cam_from_world * points3d[i].homogeneous();
            // Check if 3D point is in front of camera.
            if point3d_in_cam.z > MACHINE_EPSILON {
                residuals[i] = (model
                    .focal_lengths
                    .cwise_product(point3d_in_cam.hnormalized())
                    - points2d[i])
                    .squared_norm();
            } else {
                residuals[i] = f64::MAX;
            }
        }
        Ok(())
    }
}

/// Compute squared reprojection error in pixels: `residuals[i]` is the squared distance
/// between `points2d[i].image_point` and the projection of `points3d[i]`, or `f64::MAX`
/// where `img_from_cam_func` has no projection. Port of
/// `colmap::ComputeSquaredReprojectionError`.
pub fn compute_squared_reprojection_error(
    points2d: &[Point2DWithRay],
    points3d: &[Vector3d],
    cam_from_world: &Matrix3x4d,
    img_from_cam_func: &ImgFromCamFunc,
    residuals: &mut Vec<f64>,
) -> Result<()> {
    let num_points = points2d.len();
    check_eq!(num_points, points3d.len());
    residuals.resize(num_points, 0.0);
    for i in 0..num_points {
        let point3d_in_cam = *cam_from_world * points3d[i].homogeneous();
        residuals[i] = match img_from_cam_func(point3d_in_cam) {
            Some(proj_image_point) => (proj_image_point - points2d[i].image_point).squared_norm(),
            None => f64::MAX,
        };
    }
    Ok(())
}
