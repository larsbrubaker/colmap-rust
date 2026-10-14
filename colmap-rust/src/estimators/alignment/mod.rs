//! Port of COLMAP's `estimators/alignment.h` / `.cc`, part 1: aligning a reconstruction to
//! image locations or pose priors, aligning two reconstructions to each other (via
//! reprojections, projection centers or shared 3D points), and measuring per-image alignment
//! errors ([`ImageAlignmentError`], [`AlignmentErrorSummary`]). Merging reconstructions and
//! restoring metric rig scales are in [`merge`]; the LO-RANSAC estimator of
//! [`align_reconstructions_via_reprojections`] is in [`reprojection_estimator`]. The
//! similarity transforms come from [`crate::estimators::solvers::similarity_transform`].
//! Port of colmap-sharp's `Estimators/Alignment.cs`.
//! Tests: `tests/estimators/alignment.rs` (`alignment_test.cc` 1:1).
//!
//! Tier C (outcome) for the robust alignments (LO-RANSAC); Tier B for
//! [`compute_image_alignment_error`] and [`AlignmentErrorSummary::compute`] (scalar, but
//! through the quaternion angular distance and selection-reordered sums).
//!
//! Translation notes:
//! - `Sim3d* tgt_from_src` becomes `&mut Sim3d`, left unchanged on failure as in C++.
//!   `AlignReconstructionToLocations` accepts a null pointer in C++; no caller passes one.
//! - COLMAP's `LOG(WARNING)` / `LOG(INFO)` / `VLOG` lines are not emitted (the core has no
//!   logger; docs/CPP_DIVERGENCES.md entry 277).
//! - [`align_reconstructions_via_points`] picks, per source point, the target point seen
//!   most often along its track; `std::max_element` over a hash map breaks ties by hash
//!   order, the port by first appearance along the track (entry 277).

pub mod merge;
pub mod reprojection_estimator;

use std::collections::{BTreeSet, HashMap};

use crate::estimators::solvers::similarity_transform::estimate_sim3d_robust;
use crate::geometry::pose::transform_camera_world;
use crate::geometry::{PosePrior, Rigid3d, Sim3d};
use crate::linalg::Vector3d;
use crate::math::utils::{mean, median, percentile, rad_to_deg, CHI_SQUARE_95_THREE_DOF};
use crate::optim::estimator::EstimateAsLocal;
use crate::optim::loransac::LoRansac;
use crate::optim::ransac::RansacOptions;
use crate::scene::image::Image;
use crate::scene::reconstruction::Reconstruction;
use crate::util::types::{ImageId, Point3DId, SensorType};
use crate::Result;

pub use merge::{align_reconstruction_to_orig_rig_scales, merge_reconstructions};
use reprojection_estimator::ReconstructionAlignmentEstimator;

/// `image.HasPose()` for an image of `reconstruction`: false without a frame.
pub(crate) fn image_has_pose(reconstruction: &Reconstruction, image: &Image) -> bool {
    image.has_pose(reconstruction.frames().get(&image.frame_id()))
}

/// `image.CamFromWorld()` for an image of `reconstruction`, through its frame and rig.
pub(crate) fn image_cam_from_world(
    reconstruction: &Reconstruction,
    image: &Image,
) -> Result<Rigid3d> {
    let frame = reconstruction.frame(image.frame_id())?;
    let rig = reconstruction.rig(frame.rig_id())?;
    image.cam_from_world(frame, rig)
}

/// `image.ProjectionCenter()` for an image of `reconstruction`, through its frame and rig.
pub(crate) fn image_projection_center(
    reconstruction: &Reconstruction,
    image: &Image,
) -> Result<Vector3d> {
    let frame = reconstruction.frame(image.frame_id())?;
    let rig = reconstruction.rig(frame.rig_id())?;
    image.projection_center(frame, rig)
}

/// Robustly align reconstruction to given image locations (projection centers). Port of
/// `colmap::AlignReconstructionToLocations`.
pub fn align_reconstruction_to_locations(
    src_reconstruction: &Reconstruction,
    tgt_image_names: &[String],
    tgt_image_locations: &[Vector3d],
    min_common_images: i32,
    ransac_options: &RansacOptions,
    tgt_from_src: &mut Sim3d,
) -> Result<bool> {
    crate::check_ge!(min_common_images, 3);
    crate::check_eq!(tgt_image_names.len(), tgt_image_locations.len());

    // Find out which images are contained in the reconstruction and get the
    // positions of their camera centers.
    let mut common_image_ids = BTreeSet::<ImageId>::new();
    let mut src = Vec::new();
    let mut dst = Vec::new();
    for (name, location) in tgt_image_names.iter().zip(tgt_image_locations) {
        let Some(src_image) = src_reconstruction.find_image_with_name(name) else {
            continue;
        };

        if !image_has_pose(src_reconstruction, src_image) {
            continue;
        }

        // Ignore duplicate images.
        if !common_image_ids.insert(src_image.image_id()) {
            continue;
        }

        src.push(image_projection_center(src_reconstruction, src_image)?);
        dst.push(*location);
    }

    // Only compute the alignment if there are enough correspondences.
    // `min_common_images >= 3` was checked, so the cast is exact.
    if common_image_ids.len() < min_common_images as usize {
        return Ok(false);
    }

    let mut estimated = Sim3d::default();
    let report = estimate_sim3d_robust(&src, &dst, ransac_options, &mut estimated)?;

    if report.support.num_inliers < min_common_images as usize {
        return Ok(false);
    }

    *tgt_from_src = estimated;
    Ok(true)
}

/// Robustly align reconstruction to given pose priors. If `max_error` is not set in the
/// RANSAC options, derive it from the median position covariance. Port of
/// `colmap::AlignReconstructionToPosePriors`.
pub fn align_reconstruction_to_pose_priors(
    src_reconstruction: &Reconstruction,
    tgt_pose_priors: &[PosePrior],
    ransac_options: &RansacOptions,
    prior_position_fallback_stddev: f64,
    tgt_from_src: &mut Sim3d,
) -> Result<bool> {
    crate::check_gt!(prior_position_fallback_stddev, 0.0);
    let mut ransac_options = *ransac_options;

    let mut src = Vec::with_capacity(tgt_pose_priors.len());
    let mut tgt = Vec::with_capacity(tgt_pose_priors.len());
    let mut rms_vars = Vec::with_capacity(tgt_pose_priors.len());

    // Only looked up, never iterated, so a std HashMap is fine.
    let mut tgt_image_to_pose_prior = HashMap::<u64, &PosePrior>::new();
    for pose_prior in tgt_pose_priors {
        if pose_prior.corr_data_id.sensor_id.sensor_type == SensorType::Camera
            && pose_prior.has_position()
        {
            let id = pose_prior.corr_data_id.id;
            crate::check!(
                tgt_image_to_pose_prior.insert(id, pose_prior).is_none(),
                "Duplicate pose prior for image {id}"
            );
        }
    }

    for image_id in src_reconstruction.reg_image_ids()? {
        if let Some(pose_prior) = tgt_image_to_pose_prior.get(&u64::from(image_id)) {
            let image = src_reconstruction.image(image_id)?;
            src.push(image_projection_center(src_reconstruction, image)?);
            tgt.push(pose_prior.position);
            let trace = pose_prior.position_covariance.trace();
            if trace > 0.0 {
                rms_vars.push(trace / 3.0);
            }
        }
    }

    if src.len() < 3 {
        // COLMAP: LOG(WARNING) << "Not enough valid pose priors for alignment".
        return Ok(false);
    }

    if ransac_options.max_error <= 0.0 {
        if rms_vars.is_empty() {
            // COLMAP: LOG(WARNING) << "No pose priors with valid covariance found.".
            rms_vars.push(prior_position_fallback_stddev * prior_position_fallback_stddev);
        }

        // Scale the median RMS variance by the 95% chi-square quantile for 3 DOF.
        ransac_options.max_error = (CHI_SQUARE_95_THREE_DOF * median(&mut rms_vars)?).sqrt();
    }

    Ok(estimate_sim3d_robust(&src, &tgt, &ransac_options, tgt_from_src)?.success)
}

/// Robustly compute alignment between reconstructions by finding images that are
/// registered in both reconstructions. The alignment is then estimated robustly inside
/// RANSAC from corresponding projection centers. An alignment is verified by reprojecting
/// common 3D point observations. The `min_inlier_observations` threshold determines how
/// many observations in a common image must reproject within the given threshold. Port of
/// `colmap::AlignReconstructionsViaReprojections`.
pub fn align_reconstructions_via_reprojections(
    src_reconstruction: &Reconstruction,
    tgt_reconstruction: &Reconstruction,
    min_inlier_observations: f64,
    max_reproj_error: f64,
    tgt_from_src: &mut Sim3d,
) -> Result<bool> {
    crate::check_ge!(min_inlier_observations, 0.0);
    crate::check_le!(min_inlier_observations, 1.0);

    let ransac_options = RansacOptions {
        max_error: 1.0 - min_inlier_observations,
        min_inlier_ratio: 0.2,
        ..RansacOptions::default()
    };

    let estimator = ReconstructionAlignmentEstimator::new(
        max_reproj_error,
        src_reconstruction,
        tgt_reconstruction,
    )?;
    let mut ransac = LoRansac::<_, EstimateAsLocal<_>>::new(
        &ransac_options,
        estimator.clone(),
        EstimateAsLocal(estimator),
    )?;

    let common_image_ids = src_reconstruction.find_common_reg_image_ids(tgt_reconstruction)?;

    if common_image_ids.len() < 3 {
        return Ok(false);
    }

    let src_images: Vec<ImageId> = common_image_ids.iter().map(|ids| ids.0).collect();
    let tgt_images: Vec<ImageId> = common_image_ids.iter().map(|ids| ids.1).collect();

    let report = ransac.estimate(&src_images, &tgt_images)?;

    if report.success {
        *tgt_from_src = report.model;
    }

    Ok(report.success)
}

/// Robustly compute alignment between reconstructions by finding images that are
/// registered in both reconstructions. The alignment is then estimated robustly inside
/// RANSAC from corresponding projection centers and by minimizing the Euclidean distance
/// between them in world space. Port of `colmap::AlignReconstructionsViaProjCenters`.
pub fn align_reconstructions_via_proj_centers(
    src_reconstruction: &Reconstruction,
    tgt_reconstruction: &Reconstruction,
    max_proj_center_error: f64,
    tgt_from_src: &mut Sim3d,
) -> Result<bool> {
    crate::check_gt!(max_proj_center_error, 0.0);

    let mut ref_image_names = Vec::new();
    let mut ref_proj_centers = Vec::new();
    for image in tgt_reconstruction.images().values() {
        if image_has_pose(tgt_reconstruction, image) {
            ref_image_names.push(image.name().to_string());
            ref_proj_centers.push(image_projection_center(tgt_reconstruction, image)?);
        }
    }

    let ransac_options = RansacOptions {
        max_error: max_proj_center_error,
        ..RansacOptions::default()
    };
    align_reconstruction_to_locations(
        src_reconstruction,
        &ref_image_names,
        &ref_proj_centers,
        /* min_common_images= */ 3,
        &ransac_options,
        tgt_from_src,
    )
}

/// Robustly compute the alignment between reconstructions that share the same 2D points.
/// It is estimated by minimizing the 3D distance between corresponding 3D points. Port of
/// `colmap::AlignReconstructionsViaPoints`.
pub fn align_reconstructions_via_points(
    src_reconstruction: &Reconstruction,
    tgt_reconstruction: &Reconstruction,
    min_common_observations: usize,
    max_error: f64,
    min_inlier_ratio: f64,
    tgt_from_src: &mut Sim3d,
) -> Result<bool> {
    crate::check_gt!(min_common_observations, 0);
    crate::check_gt!(max_error, 0.0);
    crate::check_ge!(min_inlier_ratio, 0.0);
    crate::check_le!(min_inlier_ratio, 1.0);

    let mut src_xyz = Vec::new();
    let mut tgt_xyz = Vec::new();
    // The counts in first-seen order (the tie-break of max_element below, entry 277).
    let mut counts: Vec<(Point3DId, usize)> = Vec::new();
    // Associate 3D points using point2D_idx
    for src_point3d in src_reconstruction.points3d().values() {
        counts.clear();
        // Count how often a 3D point in tgt is associated to this 3D point.
        for track_el in src_point3d.track.elements() {
            let tgt_image = tgt_reconstruction.image(track_el.image_id)?;
            if !image_has_pose(tgt_reconstruction, tgt_image) {
                continue;
            }
            let tgt_point2d = tgt_image.point2d(track_el.point2d_idx);
            if tgt_point2d.has_point3d() {
                // As in COLMAP, the first association counts as 0, so a count is the
                // number of associations minus one.
                match counts.iter_mut().find(|c| c.0 == tgt_point2d.point3d_id) {
                    Some(count) => count.1 += 1,
                    None => counts.push((tgt_point2d.point3d_id, 0)),
                }
            }
        }
        // The 3D point in tgt who is associated the most is selected (std::max_element
        // keeps the first maximum).
        let Some(mut best_point3d) = counts.first().copied() else {
            continue;
        };
        for &count in &counts[1..] {
            if best_point3d.1 < count.1 {
                best_point3d = count;
            }
        }
        if best_point3d.1 >= min_common_observations {
            src_xyz.push(src_point3d.xyz);
            tgt_xyz.push(tgt_reconstruction.point3d(best_point3d.0)?.xyz);
        }
    }
    crate::check_eq!(src_xyz.len(), tgt_xyz.len());
    // COLMAP: LOG(INFO) << "Found " << n << " / " << NumPoints3D() << " valid correspondences."

    let ransac_options = RansacOptions {
        max_error,
        min_inlier_ratio,
        ..RansacOptions::default()
    };
    Ok(estimate_sim3d_robust(&src_xyz, &tgt_xyz, &ransac_options, tgt_from_src)?.success)
}

/// The alignment error of one image in the target coordinate frame. Port of
/// `colmap::ImageAlignmentError`.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageAlignmentError {
    /// The image's name.
    pub image_name: String,
    /// The angle between the aligned source and the target camera orientation, in degrees.
    pub rotation_error_deg: f64,
    /// The distance between the aligned source and the target projection center.
    pub proj_center_error: f64,
}

impl Default for ImageAlignmentError {
    /// COLMAP's member initializers: an empty name and both errors -1.
    fn default() -> Self {
        Self {
            image_name: String::new(),
            rotation_error_deg: -1.0,
            proj_center_error: -1.0,
        }
    }
}

/// Compute image alignment errors in the target coordinate frame, for the images
/// registered in both reconstructions (in the source's registration order). Port of
/// `colmap::ComputeImageAlignmentError`.
pub fn compute_image_alignment_error(
    src_reconstruction: &Reconstruction,
    tgt_reconstruction: &Reconstruction,
    tgt_from_src: &Sim3d,
) -> Result<Vec<ImageAlignmentError>> {
    let common_image_ids = src_reconstruction.find_common_reg_image_ids(tgt_reconstruction)?;
    let mut errors = Vec::with_capacity(common_image_ids.len());
    for (src_image_id, tgt_image_id) in common_image_ids {
        let src_image = src_reconstruction.image(src_image_id)?;
        let tgt_world_from_src_cam = transform_camera_world(
            tgt_from_src,
            &image_cam_from_world(src_reconstruction, src_image)?,
        )
        .inverse();
        let tgt_image = tgt_reconstruction.image(tgt_image_id)?;
        let tgt_world_from_tgt_cam = image_cam_from_world(tgt_reconstruction, tgt_image)?.inverse();

        errors.push(ImageAlignmentError {
            image_name: src_image.name().to_string(),
            rotation_error_deg: rad_to_deg(
                tgt_world_from_src_cam
                    .rotation
                    .angular_distance(tgt_world_from_tgt_cam.rotation),
            ),
            proj_center_error: (tgt_world_from_src_cam.translation
                - tgt_world_from_tgt_cam.translation)
                .norm(),
        });
    }
    Ok(errors)
}

/// Order statistics of one kind of alignment error. Port of
/// `colmap::AlignmentErrorSummary::Statistics`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AlignmentErrorStatistics {
    /// The smallest error.
    pub min: f64,
    /// The largest error.
    pub max: f64,
    /// The mean error.
    pub mean: f64,
    /// The median error.
    pub median: f64,
    /// The 90th percentile.
    pub p90: f64,
    /// The 99th percentile.
    pub p99: f64,
}

impl AlignmentErrorStatistics {
    /// COLMAP's `ComputeStatistics` lambda, in its order of calls: `Percentile` and `Median`
    /// reorder the values in place (as in C++), so the mean is summed over the order the
    /// first two percentiles left. All zero for no values.
    fn compute(values: &mut [f64]) -> Result<Self> {
        if values.is_empty() {
            return Ok(Self::default());
        }
        let min = percentile(values, 0.0)?;
        let max = percentile(values, 100.0)?;
        let mean = mean(values)?;
        let median = median(values)?;
        let p90 = percentile(values, 90.0)?;
        let p99 = percentile(values, 99.0)?;
        Ok(Self {
            min,
            max,
            mean,
            median,
            p90,
            p99,
        })
    }
}

/// Summary of alignment errors for image poses. Port of `colmap::AlignmentErrorSummary`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct AlignmentErrorSummary {
    /// Statistics of the rotation errors, in degrees.
    pub rotation_errors_deg: AlignmentErrorStatistics,
    /// Statistics of the projection center errors.
    pub proj_center_errors: AlignmentErrorStatistics,
}

impl AlignmentErrorSummary {
    /// `AlignmentErrorSummary::Compute`: the statistics of the errors, all zero for none.
    pub fn compute(errors: &[ImageAlignmentError]) -> Result<Self> {
        let mut rotation_errors_deg: Vec<f64> = errors
            .iter()
            .map(|error| error.rotation_error_deg)
            .collect();
        let mut proj_center_errors: Vec<f64> =
            errors.iter().map(|error| error.proj_center_error).collect();
        Ok(Self {
            rotation_errors_deg: AlignmentErrorStatistics::compute(&mut rotation_errors_deg)?,
            proj_center_errors: AlignmentErrorStatistics::compute(&mut proj_center_errors)?,
        })
    }
}
