//! Port of the `ReconstructionAlignmentEstimator` of COLMAP's `estimators/alignment.cc`:
//! the LO-RANSAC estimator of
//! [`align_reconstructions_via_reprojections`](super::align_reconstructions_via_reprojections),
//! which samples common images, estimates a similarity transform from their projection
//! centers and scores each image by how many of its common 3D points reproject in both
//! directions. Port of colmap-sharp's `Estimators/Alignment.Estimator.cs`.
//! Tests: `tests/estimators/alignment.rs`.
//!
//! Translation note: COLMAP's samples are `const Image*` pairs; here they are the image ids
//! of the source and target reconstruction, which the estimator holds by reference.

use crate::estimators::solvers::similarity_transform::estimate_sim3d;
use crate::geometry::Sim3d;
use crate::optim::estimator::Estimator;
use crate::scene::projection::calculate_squared_reprojection_error;
use crate::scene::reconstruction::Reconstruction;
use crate::util::types::ImageId;
use crate::Result;

use super::{image_cam_from_world, image_projection_center};

/// Port of `ReconstructionAlignmentEstimator` (alignment.cc's anonymous namespace).
#[derive(Clone, Debug)]
pub struct ReconstructionAlignmentEstimator<'a> {
    max_squared_reproj_error: f64,
    src_reconstruction: &'a Reconstruction,
    tgt_reconstruction: &'a Reconstruction,
}

impl<'a> ReconstructionAlignmentEstimator<'a> {
    /// The estimator for the given reprojection error threshold (in pixels).
    pub fn new(
        max_reproj_error: f64,
        src_reconstruction: &'a Reconstruction,
        tgt_reconstruction: &'a Reconstruction,
    ) -> Result<Self> {
        crate::check_ge!(max_reproj_error, 0.0);
        Ok(Self {
            max_squared_reproj_error: max_reproj_error * max_reproj_error,
            src_reconstruction,
            tgt_reconstruction,
        })
    }
}

impl Estimator for ReconstructionAlignmentEstimator<'_> {
    type X = ImageId;
    type Y = ImageId;
    type M = Sim3d;

    const MIN_NUM_SAMPLES: usize = 3;

    /// Estimate 3D similarity transform from corresponding projection centers.
    fn estimate(
        &mut self,
        src_images: &[ImageId],
        tgt_images: &[ImageId],
        models: &mut Vec<Sim3d>,
    ) -> Result<()> {
        crate::check_ge!(src_images.len(), 3);
        crate::check_ge!(tgt_images.len(), 3);
        crate::check_eq!(src_images.len(), tgt_images.len());

        models.clear();

        let mut proj_centers1 = Vec::with_capacity(src_images.len());
        let mut proj_centers2 = Vec::with_capacity(tgt_images.len());
        for (&src_image_id, &tgt_image_id) in src_images.iter().zip(tgt_images) {
            let src_image = self.src_reconstruction.image(src_image_id)?;
            let tgt_image = self.tgt_reconstruction.image(tgt_image_id)?;
            crate::check_eq!(src_image.image_id(), tgt_image.image_id());
            proj_centers1.push(image_projection_center(self.src_reconstruction, src_image)?);
            proj_centers2.push(image_projection_center(self.tgt_reconstruction, tgt_image)?);
        }

        let mut tgt_from_src = Sim3d::default();
        if !estimate_sim3d(&proj_centers1, &proj_centers2, &mut tgt_from_src)? {
            return Ok(());
        }

        models.push(tgt_from_src);
        Ok(())
    }

    /// For each image, determine the ratio of 3D points that correctly project from one
    /// image to the other image and vice versa for the given tgt_from_src. The residual is
    /// then defined as 1 minus this ratio, i.e., an error threshold of 0.3 means that 70%
    /// of the points for that image must reproject within the given maximum reprojection
    /// error threshold. (COLMAP squares that difference before storing it.)
    fn residuals(
        &mut self,
        src_images: &[ImageId],
        tgt_images: &[ImageId],
        tgt_from_src: &Sim3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        crate::check_eq!(src_images.len(), tgt_images.len());

        let src_from_tgt = tgt_from_src.inverse();

        residuals.clear();
        residuals.resize(src_images.len(), 0.0);

        for (i, (&src_image_id, &tgt_image_id)) in src_images.iter().zip(tgt_images).enumerate() {
            let src_image = self.src_reconstruction.image(src_image_id)?;
            let tgt_image = self.tgt_reconstruction.image(tgt_image_id)?;

            crate::check_eq!(src_image.image_id(), tgt_image.image_id());

            let src_camera = self.src_reconstruction.camera(src_image.camera_id())?;
            let tgt_camera = self.tgt_reconstruction.camera(tgt_image.camera_id())?;

            let src_cam_from_world =
                image_cam_from_world(self.src_reconstruction, src_image)?.to_matrix();
            let tgt_cam_from_world =
                image_cam_from_world(self.tgt_reconstruction, tgt_image)?.to_matrix();

            crate::check_eq!(src_image.num_points2d(), tgt_image.num_points2d());

            let mut num_inliers = 0usize;
            let mut num_common_points = 0usize;

            for point2d_idx in 0..src_image.num_points2d() {
                // Check if both images have a 3D point.

                let src_point2d = src_image.point2d(point2d_idx);
                if !src_point2d.has_point3d() {
                    continue;
                }

                let tgt_point2d = tgt_image.point2d(point2d_idx);
                if !tgt_point2d.has_point3d() {
                    continue;
                }

                num_common_points += 1;

                let src_point_in_tgt =
                    *tgt_from_src * self.src_reconstruction.point3d(src_point2d.point3d_id)?.xyz;
                if calculate_squared_reprojection_error(
                    tgt_point2d.xy,
                    src_point_in_tgt,
                    &tgt_cam_from_world,
                    tgt_camera,
                ) > self.max_squared_reproj_error
                {
                    continue;
                }

                let tgt_point_in_src =
                    src_from_tgt * self.tgt_reconstruction.point3d(tgt_point2d.point3d_id)?.xyz;
                if calculate_squared_reprojection_error(
                    src_point2d.xy,
                    tgt_point_in_src,
                    &src_cam_from_world,
                    src_camera,
                ) > self.max_squared_reproj_error
                {
                    continue;
                }

                num_inliers += 1;
            }

            residuals[i] = if num_common_points == 0 {
                1.0
            } else {
                let negative_inlier_ratio = 1.0 - num_inliers as f64 / num_common_points as f64;
                negative_inlier_ratio * negative_inlier_ratio
            };
        }
        Ok(())
    }
}
