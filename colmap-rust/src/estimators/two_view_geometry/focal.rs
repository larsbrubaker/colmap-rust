//! The focal-recovering paths of COLMAP's `two_view_geometry.cc`:
//! `EstimateSharedFocalTwoViewGeometry` (one camera with an unknown focal on both sides, the
//! six-point shared-focal solver) and `EstimateOneSidedFocalTwoViewGeometry` (exactly one
//! side calibrated). Both compete with a homography for the planar/panoramic degeneracy and
//! surface the estimated intrinsics through `TwoViewGeometry::camera1` / `camera2`.
//! Dispatched from `mod.rs`; the shared tail is in `calibrated.rs`.

use super::calibrated::{
    estimate_homography_matrix, finish_inlier_ratio_and_rest, homography_ransac_options,
    MatrixReport,
};
use super::pose::extract_inlier_cam_rays;
use super::{
    degenerate_geometry, extract_inlier_matches, is_camera_calibrated, matched_img_points,
    min_num_inliers, ransac_options_with_min_inlier_ratio, TwoViewGeometryOptions,
};
use crate::estimators::solvers::{
    RelativePoseOneSidedFocalEstimator, RelativePoseSharedFocalEstimator,
};
use crate::feature::types::{FeatureMatch, FeatureMatches};
use crate::geometry::essential_matrix::pose_from_essential_matrix;
use crate::geometry::pose::CamRayWithJac;
use crate::linalg::{Matrix3d, Vector2d};
use crate::optim::{LoRansac, MEstimatorSupportMeasurer};
use crate::scene::camera::Camera;
use crate::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use crate::{check, check_ne, Result};

/// Port of `colmap::EstimateSharedFocalTwoViewGeometry`: two views of one camera whose
/// shared focal length is unknown. The estimated focal is published as `camera1` /
/// `camera2` (identical) together with E and `F = K^-T E K^-1`, under `UNCALIBRATED`.
pub fn estimate_shared_focal_two_view_geometry(
    camera: &Camera,
    points1: &[Vector2d],
    points2: &[Vector2d],
    matches: &[FeatureMatch],
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    check!(options.check());

    let min_num_inliers = min_num_inliers(options.min_num_inliers);
    if matches.len() < min_num_inliers {
        return Ok(degenerate_geometry());
    }

    // Extract corresponding points. The shared-focal solver operates on
    // principal-point-centered image points, while the homography and watermark checks
    // operate on raw image points.
    let principal_point = camera.principal_point()?;
    let (img_points1, img_points2) = matched_img_points(points1, points2, matches);
    let centered_points1: Vec<Vector2d> =
        img_points1.iter().map(|&p| p - principal_point).collect();
    let centered_points2: Vec<Vector2d> =
        img_points2.iter().map(|&p| p - principal_point).collect();

    let ransac_options = ransac_options_with_min_inlier_ratio(options);

    // Shared-focal relative pose. Residuals are pixel-space squared Sampson error, so the
    // pixel threshold in `ransac_options` is used unscaled.
    let mut sf_ransac: LoRansac<
        RelativePoseSharedFocalEstimator,
        RelativePoseSharedFocalEstimator,
        MEstimatorSupportMeasurer,
    > = LoRansac::new(
        &ransac_options,
        RelativePoseSharedFocalEstimator,
        RelativePoseSharedFocalEstimator,
    )?;
    let sf_report = sf_ransac.estimate(&centered_points1, &centered_points2)?;

    // Estimate a homography to detect planar/panoramic degeneracies, where two-view focal
    // recovery is ill-posed and the 6-point solver returns a meaningless focal length.
    let h_ransac_options = homography_ransac_options(
        &ransac_options,
        options.max_h_inlier_ratio,
        sf_report.support.num_inliers,
        matches.len(),
    );
    let h_report = estimate_homography_matrix(&h_ransac_options, &img_points1, &img_points2)?;
    let mut geometry = TwoViewGeometry {
        h: Some(h_report.model),
        ..TwoViewGeometry::default()
    };

    let (sf_num, h_num) = (sf_report.support.num_inliers, h_report.support.num_inliers);
    if (!sf_report.success && !h_report.success)
        || (sf_num < min_num_inliers && h_num < min_num_inliers)
    {
        geometry.config = ConfigurationType::Degenerate;
        return Ok(geometry);
    }

    let h_sf_inlier_ratio = h_num as f64 / sf_num as f64;

    let best_inlier_mask: &[bool];
    let num_inliers: usize;
    if sf_report.success
        && sf_num >= min_num_inliers
        && h_sf_inlier_ratio <= options.max_h_inlier_ratio
    {
        // Shared-focal configuration. Labeled UNCALIBRATED (the focal is estimated, not a
        // trusted prior); the estimated intrinsics are surfaced via camera1/camera2 so
        // consumers can distinguish it from a plain uncalibrated pair and seed the recovered
        // focal.
        num_inliers = sf_num;
        best_inlier_mask = &sf_report.inlier_mask;
        geometry.config = ConfigurationType::Uncalibrated;
        geometry.e = Some(sf_report.model.e);
        // Both views share the camera, so camera1 and camera2 are identical.
        let mut estimated_camera = camera.clone();
        estimated_camera.set_focal_length(sf_report.model.focal)?;
        geometry.camera1 = Some(estimated_camera.clone());
        geometry.camera2 = Some(estimated_camera);
        // Also expose F = K^-T E K^-1 (K = diag(f, f, 1) at the principal point), which view
        // graph calibration requires from every UNCALIBRATED pair to calibrate focal
        // lengths. It is also the only epipolar model persisted to the database, which has
        // no columns for the estimated intrinsics.
        let mut k = Matrix3d::identity();
        k[(0, 0)] = sf_report.model.focal;
        k[(1, 1)] = sf_report.model.focal;
        k[(0, 2)] = principal_point.x;
        k[(1, 2)] = principal_point.y;
        let k_inv = k.inverse();
        geometry.f = Some(k_inv.transpose() * sf_report.model.e * k_inv);
    } else if h_report.success && h_num >= min_num_inliers {
        num_inliers = h_num;
        best_inlier_mask = &h_report.inlier_mask;
        geometry.config = ConfigurationType::PlanarOrPanoramic;
    } else {
        geometry.config = ConfigurationType::Degenerate;
        return Ok(geometry);
    }

    geometry.inlier_matches = extract_inlier_matches(matches, num_inliers, best_inlier_mask);

    // If the focal is unidentifiable (parallel or isosceles-intersecting optical axes), drop
    // the estimated intrinsics so the pair degrades to a plain uncalibrated pair (F is
    // valid).
    if let (Some(estimated1), Some(estimated2), Some(e)) =
        (&geometry.camera1, &geometry.camera2, &geometry.e)
    {
        let (inlier_cam_rays1, inlier_cam_rays2) = extract_inlier_cam_rays(
            estimated1,
            points1,
            estimated2,
            points2,
            &geometry.inlier_matches,
        );
        let (cam2_from_cam1, valid_indices) =
            pose_from_essential_matrix(e, &inlier_cam_rays1, &inlier_cam_rays2)?;
        if valid_indices.is_empty()
            || !RelativePoseSharedFocalEstimator::is_focal_identifiable(&cam2_from_cam1)
        {
            geometry.e = None;
            geometry.camera1 = None;
            geometry.camera2 = None;
        }
    }

    finish_inlier_ratio_and_rest(
        geometry,
        camera,
        points1,
        camera,
        points2,
        matches.len(),
        (&img_points1, &img_points2),
        num_inliers,
        best_inlier_mask,
        options,
    )
}

/// Port of `colmap::EstimateOneSidedFocalTwoViewGeometry`: exactly one of the two cameras is
/// calibrated (a focal prior or spherical); the other side's focal is recovered and published
/// as that side's `camera1` / `camera2`, under `UNCALIBRATED`.
pub fn estimate_one_sided_focal_two_view_geometry(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    matches: &[FeatureMatch],
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    check!(options.check());
    // Exactly one side must be calibrated, otherwise this is the calibrated or the
    // shared-focal/uncalibrated problem.
    check_ne!(is_camera_calibrated(camera1), is_camera_calibrated(camera2));

    // The solver requires the uncalibrated view first. If it is the second one, run with the
    // roles swapped and map the result back via invert().
    if is_camera_calibrated(camera1) {
        let swapped_matches: FeatureMatches = matches
            .iter()
            .map(|m| FeatureMatch::new(m.point2d_idx2, m.point2d_idx1))
            .collect();
        let mut geometry = estimate_one_sided_focal_two_view_geometry(
            camera2,
            points2,
            camera1,
            points1,
            &swapped_matches,
            options,
        )?;
        geometry.invert();
        return Ok(geometry);
    }

    let min_num_inliers = min_num_inliers(options.min_num_inliers);
    if matches.len() < min_num_inliers {
        return Ok(degenerate_geometry());
    }

    // The solver takes the uncalibrated view as principal-point-centered image points and
    // the calibrated view as bearing rays with their unprojection Jacobians, which undoes its
    // distortion exactly and lets the residual be measured in that view's pixels. The
    // homography and watermark checks use raw image points.
    let principal_point1 = camera1.principal_point()?;
    let (img_points1, img_points2) = matched_img_points(points1, points2, matches);
    let centered_points1: Vec<Vector2d> =
        img_points1.iter().map(|&p| p - principal_point1).collect();
    let cam_rays2_with_jac: Vec<CamRayWithJac> = img_points2
        .iter()
        .map(|&p| {
            camera2
                .cam_ray_from_img_with_jac(p)
                .unwrap_or(CamRayWithJac::zero())
        })
        .collect();

    let ransac_options = ransac_options_with_min_inlier_ratio(options);

    // One-sided focal relative pose. Residuals are squared tangent Sampson errors in pixels,
    // so the pixel threshold in `ransac_options` applies unscaled.
    let mut focal_ransac: LoRansac<
        RelativePoseOneSidedFocalEstimator,
        RelativePoseOneSidedFocalEstimator,
        MEstimatorSupportMeasurer,
    > = LoRansac::new(
        &ransac_options,
        RelativePoseOneSidedFocalEstimator,
        RelativePoseOneSidedFocalEstimator,
    )?;
    let focal_report = focal_ransac.estimate(&centered_points1, &cam_rays2_with_jac)?;

    // Homography, to detect planar/panoramic degeneracies where the epipolar geometry is
    // ill-constrained and the recovered focal is meaningless. Only meaningful if the
    // calibrated view has a pinhole image plane; for a spherical one it is skipped, as in the
    // spherical path. A default report has no inliers, so the checks below then behave as a
    // failed estimate.
    let has_image_plane = !camera2.is_spherical();
    let h_ransac_options = homography_ransac_options(
        &ransac_options,
        options.max_h_inlier_ratio,
        focal_report.support.num_inliers,
        matches.len(),
    );
    let h_report = if has_image_plane {
        estimate_homography_matrix(&h_ransac_options, &img_points1, &img_points2)?
    } else {
        // COLMAP still constructs the H LO-RANSAC here (which validates its options).
        h_ransac_options.check()?;
        MatrixReport::default()
    };
    let mut geometry = TwoViewGeometry::default();
    if h_report.success {
        // Only set on success: a failed report's model is not a homography, and the swap
        // path above would invert it.
        geometry.h = Some(h_report.model);
    }

    let (focal_num, h_num) = (
        focal_report.support.num_inliers,
        h_report.support.num_inliers,
    );
    if (!focal_report.success && !h_report.success)
        || (focal_num < min_num_inliers && h_num < min_num_inliers)
    {
        geometry.config = ConfigurationType::Degenerate;
        return Ok(geometry);
    }

    let h_focal_inlier_ratio = h_num as f64 / focal_num as f64;

    let best_inlier_mask: &[bool];
    let num_inliers: usize;
    if focal_report.success
        && focal_num >= min_num_inliers
        && h_focal_inlier_ratio <= options.max_h_inlier_ratio
    {
        // The focal is estimated rather than a trusted prior, hence UNCALIBRATED, and is
        // surfaced via camera1 so consumers can tell this apart from a plain uncalibrated
        // pair. camera2 stays unset: its intrinsics were an input, not an estimate.
        num_inliers = focal_num;
        best_inlier_mask = &focal_report.inlier_mask;
        geometry.config = ConfigurationType::Uncalibrated;
        geometry.e = Some(focal_report.model.e);
        let mut estimated_camera1 = camera1.clone();
        estimated_camera1.set_focal_length(focal_report.model.focal)?;
        if has_image_plane {
            // Also expose F, so that epipolar consumers unaware of the estimated focal can
            // use this config directly. It assumes a pinhole image plane on both sides, so
            // it is only an approximation for a distorted second camera; the exact geometry
            // is carried by E plus the rays. A spherical second view has no calibration
            // matrix at all, so no F is published.
            let k1_inv = estimated_camera1.calibration_matrix()?.inverse();
            let k2_inv = camera2.calibration_matrix()?.inverse();
            geometry.f = Some(k2_inv.transpose() * focal_report.model.e * k1_inv);
        }
        geometry.camera1 = Some(estimated_camera1);
    } else if h_report.success && h_num >= min_num_inliers {
        num_inliers = h_num;
        best_inlier_mask = &h_report.inlier_mask;
        geometry.config = ConfigurationType::PlanarOrPanoramic;
    } else {
        geometry.config = ConfigurationType::Degenerate;
        return Ok(geometry);
    }

    geometry.inlier_matches = extract_inlier_matches(matches, num_inliers, best_inlier_mask);
    finish_inlier_ratio_and_rest(
        geometry,
        camera1,
        points1,
        camera2,
        points2,
        matches.len(),
        (&img_points1, &img_points2),
        num_inliers,
        best_inlier_mask,
        options,
    )
}
