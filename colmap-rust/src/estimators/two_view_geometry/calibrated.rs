//! The epipolar paths of COLMAP's `two_view_geometry.cc`: `EstimateCalibratedTwoViewGeometry`
//! (E, F and H compete), `EstimateUncalibratedTwoViewGeometry` (F and H),
//! `EstimateSphericalTwoViewGeometry` (bearing-based E and a ray-space H),
//! `EstimateCalibratedHomography` (`force_H_use`), and their robust estimation helpers
//! (`EstimateFundamentalMatrix`, `EstimateHomographyMatrixFromRays`). Dispatched from
//! `mod.rs` (`estimate_two_view_geometry`); the pose step is in `pose.rs`.

use super::{
    degenerate_geometry, detect_watermark_matches, estimate_two_view_geometry_pose,
    extract_inlier_matches, matched_img_points, min_num_inliers,
    ransac_options_with_min_inlier_ratio, TwoViewGeometryOptions,
};
use crate::estimators::fundamental_matrix_degensac_estimator::{
    estimate_fundamental_matrix_degensac, FundamentalMatrixDegensacOptions,
};
use crate::estimators::solvers::{
    CamRayWithImgPoint, EssentialMatrixTangentSampsonEstimator,
    FundamentalMatrixEightPointEstimator, FundamentalMatrixSampsonEstimator,
    FundamentalMatrixSevenPointEstimator, HomographyMatrixEstimator, HomographyMatrixRayEstimator,
};
use crate::feature::types::FeatureMatch;
use crate::geometry::pose::CamRayWithJac;
use crate::linalg::{Matrix3d, Vector2d, Vector3d};
use crate::optim::{
    EstimateAsLocal, LoRansac, MEstimatorSupport, MEstimatorSupportMeasurer, RansacOptions,
    RansacReport,
};
use crate::scene::camera::Camera;
use crate::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use crate::{check, check_eq, Result};

/// `LORANSAC<..., MEstimatorSupportMeasurer>::Report` of a 3x3 matrix model (F or H).
pub(super) type MatrixReport = RansacReport<Matrix3d, MEstimatorSupport>;

/// Port of `EstimateFundamentalMatrix`: robustly estimate the fundamental matrix, either
/// with plain LO-RANSAC (Sampson-refined or eight-point local optimization) or with DEGENSAC
/// (robust to a dominant scene plane), depending on the options.
fn estimate_fundamental_matrix(
    options: &TwoViewGeometryOptions,
    ransac_options: &RansacOptions,
    points1: &[Vector2d],
    points2: &[Vector2d],
) -> Result<MatrixReport> {
    if options.use_degensac {
        let degensac_options = FundamentalMatrixDegensacOptions {
            ransac: *ransac_options,
            use_sampson_refinement: options.use_sampson_refinement,
            ..FundamentalMatrixDegensacOptions::default()
        };
        return estimate_fundamental_matrix_degensac(points1, points2, &degensac_options);
    }
    if options.use_sampson_refinement {
        let mut ransac: LoRansac<
            FundamentalMatrixSevenPointEstimator,
            FundamentalMatrixSampsonEstimator,
            MEstimatorSupportMeasurer,
        > = LoRansac::new(
            ransac_options,
            FundamentalMatrixSevenPointEstimator,
            FundamentalMatrixSampsonEstimator,
        )?;
        return ransac.estimate(points1, points2);
    }
    let mut ransac: LoRansac<
        FundamentalMatrixSevenPointEstimator,
        EstimateAsLocal<FundamentalMatrixEightPointEstimator>,
        MEstimatorSupportMeasurer,
    > = LoRansac::new(
        ransac_options,
        FundamentalMatrixSevenPointEstimator,
        EstimateAsLocal(FundamentalMatrixEightPointEstimator),
    )?;
    ransac.estimate(points1, points2)
}

/// `LORANSAC<HomographyMatrixEstimator, HomographyMatrixEstimator,
/// MEstimatorSupportMeasurer>(options).Estimate(points1, points2)`.
pub(super) fn estimate_homography_matrix(
    ransac_options: &RansacOptions,
    points1: &[Vector2d],
    points2: &[Vector2d],
) -> Result<MatrixReport> {
    let mut ransac: LoRansac<
        HomographyMatrixEstimator,
        EstimateAsLocal<HomographyMatrixEstimator>,
        MEstimatorSupportMeasurer,
    > = LoRansac::new(
        ransac_options,
        HomographyMatrixEstimator,
        EstimateAsLocal(HomographyMatrixEstimator),
    )?;
    ransac.estimate(points1, points2)
}

/// The ray-space homography LO-RANSAC, scoring residuals in `camera2`'s pixels.
fn estimate_ray_homography(
    ransac_options: &RansacOptions,
    camera2: &Camera,
    cam_rays1: &[Vector3d],
    cam_rays2: &[CamRayWithImgPoint],
) -> Result<MatrixReport> {
    let estimator = HomographyMatrixRayEstimator::new(Some(camera2));
    let mut ransac: LoRansac<
        HomographyMatrixRayEstimator,
        EstimateAsLocal<HomographyMatrixRayEstimator>,
        MEstimatorSupportMeasurer,
    > = LoRansac::new(ransac_options, estimator, EstimateAsLocal(estimator))?;
    ransac.estimate(cam_rays1, cam_rays2)
}

/// Port of `EstimateHomographyMatrixFromRays`: robustly estimate the pixel-space homography
/// of a distorted camera pair. A world plane relates image points projectively only under a
/// pinhole projection, so the estimate is made on bearing rays, where it holds for any
/// central camera, and conjugated back by the calibration matrices.
fn estimate_homography_matrix_from_rays(
    ransac_options: &RansacOptions,
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
) -> Result<MatrixReport> {
    check_eq!(points1.len(), points2.len());

    let cam_rays1: Vec<Vector3d> = points1
        .iter()
        .map(|&p| camera1.cam_ray_from_img(p).unwrap_or(Vector3d::zeros()))
        .collect();
    let cam_rays2: Vec<CamRayWithImgPoint> = points2
        .iter()
        .map(|&p| CamRayWithImgPoint {
            ray: camera2.cam_ray_from_img(p).unwrap_or(Vector3d::zeros()),
            img_point: p,
        })
        .collect();

    let mut report = estimate_ray_homography(ransac_options, camera2, &cam_rays1, &cam_rays2)?;
    // The estimator maps rays to rays, so publish K2 H K1^-1 to keep the stored homography
    // in pixel space. K carries no distortion, so for a distorted camera that is the
    // homography in its virtual pinhole frame.
    report.model =
        camera2.calibration_matrix()? * report.model * camera1.calibration_matrix()?.inverse();
    Ok(report)
}

/// Bearing rays with their unprojection Jacobians for each match (zero where unprojection
/// fails, which the tangent Sampson residual rejects).
fn matched_cam_rays_with_jac(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    matches: &[FeatureMatch],
) -> (Vec<CamRayWithJac>, Vec<CamRayWithJac>) {
    matches
        .iter()
        .map(|m| {
            (
                camera1
                    .cam_ray_from_img_with_jac(points1[m.point2d_idx1 as usize])
                    .unwrap_or(CamRayWithJac::zero()),
                camera2
                    .cam_ray_from_img_with_jac(points2[m.point2d_idx2 as usize])
                    .unwrap_or(CamRayWithJac::zero()),
            )
        })
        .unzip()
}

/// The essential matrix LO-RANSAC on the pixel-unit tangent Sampson error.
fn estimate_essential_matrix(
    ransac_options: &RansacOptions,
    cam_rays1: &[CamRayWithJac],
    cam_rays2: &[CamRayWithJac],
) -> Result<MatrixReport> {
    let mut ransac: LoRansac<
        EssentialMatrixTangentSampsonEstimator,
        EssentialMatrixTangentSampsonEstimator,
        MEstimatorSupportMeasurer,
    > = LoRansac::new(
        ransac_options,
        EssentialMatrixTangentSampsonEstimator,
        EssentialMatrixTangentSampsonEstimator,
    )?;
    ransac.estimate(cam_rays1, cam_rays2)
}

/// The tail every non-degenerate path shares: the inlier matches of the chosen mask, the
/// `min_inlier_ratio` check (when `check_min_inlier_ratio`), watermark detection and the
/// optional pose.
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_geometry(
    mut geometry: TwoViewGeometry,
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    matches: &[FeatureMatch],
    matched_points: (&[Vector2d], &[Vector2d]),
    num_inliers: usize,
    inlier_mask: &[bool],
    check_min_inlier_ratio: bool,
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    geometry.inlier_matches = extract_inlier_matches(matches, num_inliers, inlier_mask);
    if check_min_inlier_ratio {
        finish_inlier_ratio_and_rest(
            geometry,
            camera1,
            points1,
            camera2,
            points2,
            matches.len(),
            matched_points,
            num_inliers,
            inlier_mask,
            options,
        )
    } else {
        finish_watermark_and_pose(
            geometry,
            camera1,
            points1,
            camera2,
            points2,
            matched_points,
            num_inliers,
            inlier_mask,
            options,
        )
    }
}

/// The `min_inlier_ratio` check, then [`finish_watermark_and_pose`].
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_inlier_ratio_and_rest(
    mut geometry: TwoViewGeometry,
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    num_matches: usize,
    matched_points: (&[Vector2d], &[Vector2d]),
    num_inliers: usize,
    inlier_mask: &[bool],
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    // Check inlier ratio threshold.
    if options.min_inlier_ratio > 0.0 {
        let inlier_ratio = num_inliers as f64 / num_matches as f64;
        if inlier_ratio < options.min_inlier_ratio {
            geometry.config = ConfigurationType::Degenerate;
            return Ok(geometry);
        }
    }
    finish_watermark_and_pose(
        geometry,
        camera1,
        points1,
        camera2,
        points2,
        matched_points,
        num_inliers,
        inlier_mask,
        options,
    )
}

/// Watermark detection and the optional relative pose.
#[allow(clippy::too_many_arguments)]
fn finish_watermark_and_pose(
    mut geometry: TwoViewGeometry,
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    matched_points: (&[Vector2d], &[Vector2d]),
    num_inliers: usize,
    inlier_mask: &[bool],
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    if options.detect_watermark
        && detect_watermark_matches(
            camera1,
            matched_points.0,
            camera2,
            matched_points.1,
            num_inliers,
            inlier_mask,
            options,
        )?
    {
        geometry.config = ConfigurationType::Watermark;
    }
    if options.compute_relative_pose {
        estimate_two_view_geometry_pose(camera1, points1, camera2, points2, &mut geometry)?;
    }
    Ok(geometry)
}

/// Port of `EstimateCalibratedHomography`: the `force_H_use` path, a homography alone.
pub(super) fn estimate_calibrated_homography(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    matches: &[FeatureMatch],
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    let min_num_inliers = min_num_inliers(options.min_num_inliers);
    if matches.len() < min_num_inliers {
        return Ok(degenerate_geometry());
    }

    let (img_points1, img_points2) = matched_img_points(points1, points2, matches);

    // Estimate planar or panoramic model. Estimated on image points rather than rays: the
    // caller only guarantees a pinhole projection here, not a focal length prior, so no rays
    // can be built.
    let ransac_options = ransac_options_with_min_inlier_ratio(options);
    let h_report = estimate_homography_matrix(&ransac_options, &img_points1, &img_points2)?;
    let mut geometry = TwoViewGeometry {
        h: Some(h_report.model),
        ..TwoViewGeometry::default()
    };

    if !h_report.success || h_report.support.num_inliers < min_num_inliers {
        geometry.config = ConfigurationType::Degenerate;
        return Ok(geometry);
    }
    geometry.config = ConfigurationType::PlanarOrPanoramic;

    finish_geometry(
        geometry,
        camera1,
        points1,
        camera2,
        points2,
        matches,
        (&img_points1, &img_points2),
        h_report.support.num_inliers,
        &h_report.inlier_mask,
        false,
        options,
    )
}

/// `options.max_H_inlier_ratio * competing_num_inliers / matches.size()`, floored by the
/// RANSAC options' own ratio: budget the homography search for the inlier ratio it must
/// reach to be selected, since a weaker one is discarded anyway.
pub(super) fn homography_ransac_options(
    ransac_options: &RansacOptions,
    max_h_inlier_ratio: f64,
    competing_num_inliers: usize,
    num_matches: usize,
) -> RansacOptions {
    let mut h_ransac_options = *ransac_options;
    h_ransac_options.min_inlier_ratio = ransac_options
        .min_inlier_ratio
        .max(max_h_inlier_ratio * competing_num_inliers as f64 / num_matches as f64);
    h_ransac_options
}

/// Port of `EstimateUncalibratedTwoViewGeometry`: two independent unknown focals; F and H
/// compete.
pub(super) fn estimate_uncalibrated_two_view_geometry(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    matches: &[FeatureMatch],
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    let min_num_inliers = min_num_inliers(options.min_num_inliers);
    if matches.len() < min_num_inliers {
        return Ok(degenerate_geometry());
    }

    let (img_points1, img_points2) = matched_img_points(points1, points2, matches);

    // Estimate epipolar model.
    let f_report =
        estimate_fundamental_matrix(options, &options.ransac_options, &img_points1, &img_points2)?;

    // Estimate planar or panoramic model. Estimated on image points rather than rays, as the
    // intrinsics are unknown here and no rays can be built without them. The fundamental
    // matrix above shares that frame, so the comparison below stays self-consistent.
    let h_ransac_options = homography_ransac_options(
        &options.ransac_options,
        options.max_h_inlier_ratio,
        f_report.support.num_inliers,
        matches.len(),
    );
    let h_report = estimate_homography_matrix(&h_ransac_options, &img_points1, &img_points2)?;

    let mut geometry = TwoViewGeometry {
        f: Some(f_report.model),
        h: Some(h_report.model),
        ..TwoViewGeometry::default()
    };

    if (!f_report.success && !h_report.success)
        || (f_report.support.num_inliers < min_num_inliers
            && h_report.support.num_inliers < min_num_inliers)
    {
        geometry.config = ConfigurationType::Degenerate;
        return Ok(geometry);
    }

    // Determine inlier ratios of different models.
    let h_f_inlier_ratio =
        h_report.support.num_inliers as f64 / f_report.support.num_inliers as f64;

    let mut best_inlier_mask = &f_report.inlier_mask;
    let mut num_inliers = f_report.support.num_inliers;
    if h_f_inlier_ratio > options.max_h_inlier_ratio {
        geometry.config = ConfigurationType::PlanarOrPanoramic;
        if h_report.support.num_inliers >= f_report.support.num_inliers {
            num_inliers = h_report.support.num_inliers;
            best_inlier_mask = &h_report.inlier_mask;
        }
    } else {
        geometry.config = ConfigurationType::Uncalibrated;
    }

    finish_geometry(
        geometry,
        camera1,
        points1,
        camera2,
        points2,
        matches,
        (&img_points1, &img_points2),
        num_inliers,
        best_inlier_mask,
        false,
        options,
    )
}

/// Port of `EstimateSphericalTwoViewGeometry`: two-view geometry for an image pair where at
/// least one camera is omnidirectional (no pinhole image plane, e.g. EQUIRECTANGULAR) and
/// both sides have known intrinsics. The fundamental matrix is not geometrically meaningful
/// for such cameras, so the pair is classified from the bearing-based essential matrix and a
/// ray-space homography. A spherical camera paired with one of unknown focal length is
/// routed to the one-sided focal path instead, which recovers that focal rather than relying
/// on the camera's placeholder.
pub(super) fn estimate_spherical_two_view_geometry(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    matches: &[FeatureMatch],
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    check!(options.check());

    let min_num_inliers = min_num_inliers(options.min_num_inliers);
    if matches.len() < min_num_inliers {
        return Ok(degenerate_geometry());
    }

    // For a mixed spherical/perspective pair the perspective camera's bearings degrade if its
    // focal length is off, but an essential matrix can still often be estimated as long as
    // the focal length is not completely wrong. Rather than gate on a focal prior, we attempt
    // the estimation and let the inlier count below decide whether the result is degenerate.
    // The raw image points are kept only for watermark detection.
    let (img_points1, img_points2) = matched_img_points(points1, points2, matches);
    let (cam_rays1_with_jac, cam_rays2_with_jac) =
        matched_cam_rays_with_jac(camera1, points1, camera2, points2, matches);

    // Only the bearing-based essential matrix is meaningful: the fundamental matrix and
    // homography assume a pinhole image plane that an omnidirectional camera does not have.
    // The tangent Sampson residual is in pixels, so the threshold is used unscaled.
    let ransac_options = ransac_options_with_min_inlier_ratio(options);
    let e_report =
        estimate_essential_matrix(&ransac_options, &cam_rays1_with_jac, &cam_rays2_with_jac)?;
    let mut geometry = TwoViewGeometry {
        e: Some(e_report.model),
        ..TwoViewGeometry::default()
    };

    // Detect the planar/panoramic degeneracy: under pure rotation E vanishes and its pose
    // decomposition is meaningless, which is the usual capture mode for a 360 degree camera.
    // Kept in ray space, as spherical cameras have no K.
    let cam_rays1: Vec<Vector3d> = cam_rays1_with_jac.iter().map(|r| r.ray).collect();
    let cam_rays2: Vec<CamRayWithImgPoint> = cam_rays2_with_jac
        .iter()
        .zip(&img_points2)
        .map(|(r, &img_point)| CamRayWithImgPoint {
            ray: r.ray,
            img_point,
        })
        .collect();
    // Budget the search for the ratio the homography must beat to be selected below.
    let h_ransac_options = homography_ransac_options(
        &ransac_options,
        options.max_h_inlier_ratio,
        e_report.support.num_inliers,
        matches.len(),
    );
    let h_report = estimate_ray_homography(&h_ransac_options, camera2, &cam_rays1, &cam_rays2)?;
    if h_report.success {
        geometry.h = Some(h_report.model);
    }

    if (!e_report.success || e_report.support.num_inliers < min_num_inliers)
        && (!h_report.success || h_report.support.num_inliers < min_num_inliers)
    {
        geometry.config = ConfigurationType::Degenerate;
        return Ok(geometry);
    }

    let mut best_inlier_mask = &e_report.inlier_mask;
    let mut num_inliers = e_report.support.num_inliers;
    let h_e_inlier_ratio =
        h_report.support.num_inliers as f64 / e_report.support.num_inliers as f64;
    if e_report.success
        && e_report.support.num_inliers >= min_num_inliers
        && h_e_inlier_ratio <= options.max_h_inlier_ratio
    {
        geometry.config = ConfigurationType::Calibrated;
    } else {
        geometry.config = ConfigurationType::PlanarOrPanoramic;
        if h_report.support.num_inliers > num_inliers {
            num_inliers = h_report.support.num_inliers;
            best_inlier_mask = &h_report.inlier_mask;
        }
    }

    finish_geometry(
        geometry,
        camera1,
        points1,
        camera2,
        points2,
        matches,
        (&img_points1, &img_points2),
        num_inliers,
        best_inlier_mask,
        true,
        options,
    )
}

/// Port of `colmap::EstimateCalibratedTwoViewGeometry`: estimate the two-view geometry of a
/// pair whose cameras both have known intrinsics; E, F and H compete. Spherical cameras go to
/// the spherical path.
pub fn estimate_calibrated_two_view_geometry(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    matches: &[FeatureMatch],
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    check!(options.check());

    if camera1.is_spherical() || camera2.is_spherical() {
        return estimate_spherical_two_view_geometry(
            camera1, points1, camera2, points2, matches, options,
        );
    }

    let min_num_inliers = min_num_inliers(options.min_num_inliers);
    if matches.len() < min_num_inliers {
        return Ok(degenerate_geometry());
    }

    let (img_points1, img_points2) = matched_img_points(points1, points2, matches);
    let (cam_rays1_with_jac, cam_rays2_with_jac) =
        matched_cam_rays_with_jac(camera1, points1, camera2, points2, matches);

    // Estimate epipolar models.
    let ransac_options = ransac_options_with_min_inlier_ratio(options);

    // The tangent Sampson residual is in pixels, matching the fundamental matrix and
    // homography paths below, so all three share the same unscaled pixel threshold.
    let e_report =
        estimate_essential_matrix(&ransac_options, &cam_rays1_with_jac, &cam_rays2_with_jac)?;
    let f_report =
        estimate_fundamental_matrix(options, &ransac_options, &img_points1, &img_points2)?;

    // Estimate planar or panoramic model. The competing model depends on the branch taken
    // below so the homography must reach the smallest count of the two.
    let h_ransac_options = homography_ransac_options(
        &ransac_options,
        options.max_h_inlier_ratio,
        e_report
            .support
            .num_inliers
            .min(f_report.support.num_inliers),
        matches.len(),
    );
    // Undistorted pinhole cameras keep the pixel estimator, where the two are algebraically
    // equivalent.
    let h_report = if camera1.is_undistorted()? && camera2.is_undistorted()? {
        estimate_homography_matrix(&h_ransac_options, &img_points1, &img_points2)?
    } else {
        estimate_homography_matrix_from_rays(
            &h_ransac_options,
            camera1,
            &img_points1,
            camera2,
            &img_points2,
        )?
    };

    let mut geometry = TwoViewGeometry {
        e: Some(e_report.model),
        f: Some(f_report.model),
        h: Some(h_report.model),
        ..TwoViewGeometry::default()
    };

    let (e_num, f_num, h_num) = (
        e_report.support.num_inliers,
        f_report.support.num_inliers,
        h_report.support.num_inliers,
    );
    if (!e_report.success && !f_report.success && !h_report.success)
        || (e_num < min_num_inliers && f_num < min_num_inliers && h_num < min_num_inliers)
    {
        geometry.config = ConfigurationType::Degenerate;
        return Ok(geometry);
    }

    // Determine inlier ratios of different models.
    let e_f_inlier_ratio = e_num as f64 / f_num as f64;
    let h_f_inlier_ratio = h_num as f64 / f_num as f64;
    let h_e_inlier_ratio = h_num as f64 / e_num as f64;

    let mut best_inlier_mask: &[bool];
    let mut num_inliers: usize;

    if e_report.success
        && e_f_inlier_ratio > options.min_e_f_inlier_ratio
        && e_num >= min_num_inliers
    {
        // Calibrated configuration.

        // Always use the model with maximum matches.
        if e_num >= f_num {
            num_inliers = e_num;
            best_inlier_mask = &e_report.inlier_mask;
        } else {
            num_inliers = f_num;
            best_inlier_mask = &f_report.inlier_mask;
        }

        if h_e_inlier_ratio > options.max_h_inlier_ratio {
            geometry.config = ConfigurationType::PlanarOrPanoramic;
            if h_num > num_inliers {
                num_inliers = h_num;
                best_inlier_mask = &h_report.inlier_mask;
            }
        } else {
            geometry.config = ConfigurationType::Calibrated;
        }
    } else if f_report.success && f_num >= min_num_inliers {
        // Uncalibrated configuration.

        num_inliers = f_num;
        best_inlier_mask = &f_report.inlier_mask;

        if h_f_inlier_ratio > options.max_h_inlier_ratio {
            geometry.config = ConfigurationType::PlanarOrPanoramic;
            if h_num > num_inliers {
                num_inliers = h_num;
                best_inlier_mask = &h_report.inlier_mask;
            }
        } else {
            geometry.config = ConfigurationType::Uncalibrated;
        }
    } else if h_report.success && h_num >= min_num_inliers {
        num_inliers = h_num;
        best_inlier_mask = &h_report.inlier_mask;
        geometry.config = ConfigurationType::PlanarOrPanoramic;
    } else {
        geometry.config = ConfigurationType::Degenerate;
        return Ok(geometry);
    }

    finish_geometry(
        geometry,
        camera1,
        points1,
        camera2,
        points2,
        matches,
        (&img_points1, &img_points2),
        num_inliers,
        best_inlier_mask,
        true,
        options,
    )
}
