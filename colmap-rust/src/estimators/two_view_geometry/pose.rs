//! The relative pose step of COLMAP's `two_view_geometry.cc`: `EstimateTwoViewGeometryPose`
//! (decompose the selected E, F or H of a verified pair into `cam2_from_cam1` and its median
//! triangulation angle) and `MaybeDecomposeRelativePoses` (the same for every pair of a
//! [`DatabaseCache`] that has no pose yet, refitting a matrix that an older database did not
//! store). Called by the estimation paths in `calibrated.rs` and `focal.rs`.

use super::ConfigurationType;
use crate::estimators::solvers::{
    EssentialMatrixEightPointEstimator, FundamentalMatrixEightPointEstimator,
    HomographyMatrixEstimator,
};
use crate::feature::types::FeatureMatch;
use crate::geometry::essential_matrix::{
    essential_from_fundamental_matrix, pose_from_essential_matrix,
};
use crate::geometry::homography_matrix::pose_from_homography_matrix;
use crate::geometry::rigid3::Rigid3d;
use crate::geometry::triangulation::{
    calculate_angle_between_vectors, calculate_triangulation_angles,
};
use crate::linalg::{Matrix3d, Vector2d, Vector3d};
use crate::math::utils::median;
use crate::optim::Estimator;
use crate::scene::camera::Camera;
use crate::scene::database_cache::DatabaseCache;
use crate::scene::two_view_geometry::TwoViewGeometry;
use crate::util::types::pair_id_to_image_pair;
use crate::{check_notnull, Result};

/// Port of `ExtractInlierCamRays`: the bearing rays of each inlier match (zero where
/// unprojection fails).
pub(super) fn extract_inlier_cam_rays(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    inlier_matches: &[FeatureMatch],
) -> (Vec<Vector3d>, Vec<Vector3d>) {
    inlier_matches
        .iter()
        .map(|m| {
            (
                camera1
                    .cam_ray_from_img(points1[m.point2d_idx1 as usize])
                    .unwrap_or(Vector3d::zeros()),
                camera2
                    .cam_ray_from_img(points2[m.point2d_idx2 as usize])
                    .unwrap_or(Vector3d::zeros()),
            )
        })
        .unzip()
}

/// Port of `EstimateTwoViewGeometryPoseFromCamRays`: decompose the model the solver selected
/// into `geometry.cam2_from_cam1` and set `geometry.tri_angle`; `PLANAR_OR_PANORAMIC` is
/// resolved into `PLANAR` or `PANORAMIC`. `Ok(false)` if no pose is recovered.
fn estimate_two_view_geometry_pose_from_cam_rays(
    camera1: &Camera,
    camera2: &Camera,
    inlier_cam_rays1: &[Vector3d],
    inlier_cam_rays2: &[Vector3d],
    geometry: &mut TwoViewGeometry,
) -> Result<bool> {
    let mut points3d: Vec<Vector3d> = Vec::new();

    // Omnidirectional cameras (no focal length, e.g. EQUIRECTANGULAR) have no calibration
    // matrix, so they never reach the fundamental-matrix branch below. The homography branch
    // handles them by decomposing through the identity, their homography already being in
    // ray space.
    let cam2_from_cam1: Rigid3d;
    let mut valid_indices: Vec<usize> = Vec::new();
    // Decompose the model the solver selected. A calibrated pair, or one whose intrinsics a
    // solver estimated (camera1/camera2 set), selected E, with its rays already calibrated
    // accordingly. A plain uncalibrated pair selected F and recovers E from it with the
    // current calibration below. Note E is stored for every pair, so its presence does not
    // imply it is the selected model.
    if geometry.config == ConfigurationType::Calibrated
        || geometry.camera1.is_some()
        || geometry.camera2.is_some()
    {
        let e = check_notnull!(geometry.e.as_ref());
        (cam2_from_cam1, valid_indices) =
            pose_from_essential_matrix(e, inlier_cam_rays1, inlier_cam_rays2)?;
        if valid_indices.is_empty() {
            return Ok(false);
        }
    } else if geometry.config == ConfigurationType::Uncalibrated {
        let f = check_notnull!(geometry.f.as_ref());
        let e = essential_from_fundamental_matrix(
            &camera2.calibration_matrix()?,
            f,
            &camera1.calibration_matrix()?,
        );
        (cam2_from_cam1, valid_indices) =
            pose_from_essential_matrix(&e, inlier_cam_rays1, inlier_cam_rays2)?;
        if valid_indices.is_empty() {
            return Ok(false);
        }
    } else if matches!(
        geometry.config,
        ConfigurationType::Planar
            | ConfigurationType::Panoramic
            | ConfigurationType::PlanarOrPanoramic
    ) {
        let h = *check_notnull!(geometry.h.as_ref());
        // The decomposition removes the calibration first. The spherical path stores its
        // homography in ray space, having no calibration matrix, so such a pair passes the
        // identity on both sides.
        let is_ray_space = camera1.is_spherical() || camera2.is_spherical();
        let (k1, k2) = if is_ray_space {
            (Matrix3d::identity(), Matrix3d::identity())
        } else {
            (camera1.calibration_matrix()?, camera2.calibration_matrix()?)
        };
        let (pose, _normal, points) =
            pose_from_homography_matrix(&h, &k1, &k2, inlier_cam_rays1, inlier_cam_rays2)?;
        cam2_from_cam1 = pose;
        points3d = points;
        if geometry.config == ConfigurationType::PlanarOrPanoramic {
            geometry.config = if cam2_from_cam1.translation.squared_norm() < 1e-12 {
                ConfigurationType::Panoramic
            } else {
                ConfigurationType::Planar
            };
        }

        if geometry.config == ConfigurationType::Panoramic {
            geometry.tri_angle = 0.0;
        }

        if geometry.config == ConfigurationType::Planar && points3d.is_empty() {
            return Ok(false);
        }
    } else {
        return Ok(false);
    }

    geometry.cam2_from_cam1 = Some(cam2_from_cam1);

    if !valid_indices.is_empty() {
        // Essential-matrix paths (CALIBRATED / UNCALIBRATED): the triangulation angle is the
        // parallax between the surviving corresponding rays, which needs no explicit
        // triangulation.
        let cam1_from_cam2_rotation = cam2_from_cam1.rotation.inverse();
        let mut tri_angles: Vec<f64> = valid_indices
            .iter()
            .map(|&idx| {
                let angle = calculate_angle_between_vectors(
                    inlier_cam_rays1[idx],
                    cam1_from_cam2_rotation * inlier_cam_rays2[idx],
                );
                // `EIGEN_PI` is `long double`, cast back to double before the subtraction.
                angle.min(std::f64::consts::PI - angle)
            })
            .collect();
        geometry.tri_angle = median(&mut tri_angles)?;
    } else if !points3d.is_empty() {
        // Homography path (PLANAR): the triangulation angle is computed from the 3D points
        // recovered by pose_from_homography_matrix.
        let proj_center1 = Vector3d::zeros();
        let proj_center2 = cam2_from_cam1.tgt_origin_in_src();
        let mut tri_angles = calculate_triangulation_angles(proj_center1, proj_center2, &points3d);
        geometry.tri_angle = median(&mut tri_angles)?;
    }

    Ok(true)
}

/// Port of `colmap::EstimateTwoViewGeometryPose`: estimate the relative pose of a verified
/// geometry (CALIBRATED, UNCALIBRATED, PLANAR, PANORAMIC or PLANAR_OR_PANORAMIC) from its
/// inlier matches. Sets `cam2_from_cam1` and `tri_angle`; `Ok(false)` when the geometry has
/// no usable model or no pose is recovered.
pub fn estimate_two_view_geometry_pose(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    geometry: &mut TwoViewGeometry,
) -> Result<bool> {
    // We need a valid epipolar geometry to estimate the relative pose.
    if !matches!(
        geometry.config,
        ConfigurationType::Calibrated
            | ConfigurationType::Uncalibrated
            | ConfigurationType::Planar
            | ConfigurationType::Panoramic
            | ConfigurationType::PlanarOrPanoramic
    ) {
        return Ok(false);
    }
    if geometry.inlier_matches.is_empty() {
        return Ok(false);
    }

    // Calibrate rays with the intrinsics the solver estimated where available (its focal,
    // not the camera's stale default), else the given cameras.
    let effective_camera1 = geometry.camera1.as_ref().unwrap_or(camera1);
    let effective_camera2 = geometry.camera2.as_ref().unwrap_or(camera2);
    let (inlier_cam_rays1, inlier_cam_rays2) = extract_inlier_cam_rays(
        effective_camera1,
        points1,
        effective_camera2,
        points2,
        &geometry.inlier_matches,
    );
    estimate_two_view_geometry_pose_from_cam_rays(
        camera1,
        camera2,
        &inlier_cam_rays1,
        &inlier_cam_rays2,
        geometry,
    )
}

/// Eigen's `isZero()`: every coefficient within its `dummy_precision` (1e-12) of zero.
fn is_zero(v: Vector3d) -> bool {
    const PRECISION: f64 = 1e-12;
    v.x.abs() <= PRECISION && v.y.abs() <= PRECISION && v.z.abs() <= PRECISION
}

/// Port of `ExtractInlierImagePoints`: the image points of each inlier match.
fn extract_inlier_image_points(
    points1: &[Vector2d],
    points2: &[Vector2d],
    inlier_matches: &[FeatureMatch],
) -> (Vec<Vector2d>, Vec<Vector2d>) {
    super::matched_img_points(points1, points2, inlier_matches)
}

/// Port of `MaybeFitMissingTwoViewGeometryMatrix`: fit the E/F/H matrix matching
/// `geometry.config` from the existing inlier matches when it was not persisted in the
/// database (e.g. databases from older COLMAP versions that stored only the configuration).
/// The 8-point estimators are used because the inlier matches have already been filtered by
/// RANSAC at storage time (inlier counts are typically well above 8); pairs with fewer
/// inliers would yield unreliable geometry and are returned as a failed fit. `Ok(true)` if
/// the relevant matrix is present or was fit; `Ok(false)` if the inlier set is too small, the
/// fit returned no model, or the configuration has no matrix.
fn maybe_fit_missing_two_view_geometry_matrix(
    points1: &[Vector2d],
    points2: &[Vector2d],
    inlier_cam_rays1: &[Vector3d],
    inlier_cam_rays2: &[Vector3d],
    geometry: &mut TwoViewGeometry,
) -> Result<bool> {
    let mut models: Vec<Matrix3d> = Vec::new();
    match geometry.config {
        ConfigurationType::Calibrated => {
            if geometry.e.is_some() {
                return Ok(true);
            }
            let (valid_cam_rays1, valid_cam_rays2): (Vec<Vector3d>, Vec<Vector3d>) =
                inlier_cam_rays1
                    .iter()
                    .zip(inlier_cam_rays2)
                    .filter(|(r1, r2)| !is_zero(**r1) && !is_zero(**r2))
                    .map(|(r1, r2)| (*r1, *r2))
                    .unzip();
            if valid_cam_rays1.len() < EssentialMatrixEightPointEstimator::MIN_NUM_SAMPLES {
                return Ok(false);
            }
            EssentialMatrixEightPointEstimator::estimate(
                &valid_cam_rays1,
                &valid_cam_rays2,
                &mut models,
            )?;
            match models.first() {
                Some(&model) => geometry.e = Some(model),
                None => return Ok(false),
            }
            Ok(true)
        }
        ConfigurationType::Uncalibrated => {
            if geometry.f.is_some() {
                return Ok(true);
            }
            let (inlier_points1, inlier_points2) =
                extract_inlier_image_points(points1, points2, &geometry.inlier_matches);
            if inlier_points1.len() < FundamentalMatrixEightPointEstimator::MIN_NUM_SAMPLES {
                return Ok(false);
            }
            FundamentalMatrixEightPointEstimator.estimate(
                &inlier_points1,
                &inlier_points2,
                &mut models,
            )?;
            match models.first() {
                Some(&model) => geometry.f = Some(model),
                None => return Ok(false),
            }
            Ok(true)
        }
        ConfigurationType::Planar
        | ConfigurationType::Panoramic
        | ConfigurationType::PlanarOrPanoramic => {
            if geometry.h.is_some() {
                return Ok(true);
            }
            let (inlier_points1, inlier_points2) =
                extract_inlier_image_points(points1, points2, &geometry.inlier_matches);
            if inlier_points1.len() < HomographyMatrixEstimator::MIN_NUM_SAMPLES {
                return Ok(false);
            }
            HomographyMatrixEstimator.estimate(&inlier_points1, &inlier_points2, &mut models)?;
            match models.first() {
                Some(&model) => geometry.h = Some(model),
                None => return Ok(false),
            }
            Ok(true)
        }
        _ => Ok(false),
    }
}

/// Port of `colmap::MaybeDecomposeRelativePoses`: decompose the relative pose of every valid
/// image pair of the cache's correspondence graph that has none yet (fitting a missing
/// E/F/H from the inlier matches first), normalize its translation to unit length and store
/// it back in the graph. Pairs that fail keep their geometry.
pub fn maybe_decompose_relative_poses(database_cache: &mut DatabaseCache) -> Result<()> {
    for pair_id in database_cache.correspondence_graph().image_pairs() {
        let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;

        let mut two_view_geometry = database_cache
            .correspondence_graph()
            .extract_two_view_geometry(image_id1, image_id2, true)?;

        if two_view_geometry.cam2_from_cam1.is_some() {
            continue;
        }

        let is_invalid = matches!(
            two_view_geometry.config,
            ConfigurationType::Undefined
                | ConfigurationType::Degenerate
                | ConfigurationType::Watermark
                | ConfigurationType::Multiple
        );
        if is_invalid || two_view_geometry.inlier_matches.is_empty() {
            // COLMAP counts an empty inlier set as a failed decomposition; the counts only
            // feed its log line.
            continue;
        }

        let image1 = database_cache.image(image_id1)?;
        let image2 = database_cache.image(image_id2)?;
        let camera1 = database_cache.camera(image1.camera_id())?;
        let camera2 = database_cache.camera(image2.camera_id())?;

        let points1: Vec<Vector2d> = image1.points2d().iter().map(|p| p.xy).collect();
        let points2: Vec<Vector2d> = image2.points2d().iter().map(|p| p.xy).collect();

        // Calibrate rays with the intrinsics the solver estimated where available (its
        // focal, not the camera's stale default), else the given cameras.
        let effective_camera1 = two_view_geometry.camera1.as_ref().unwrap_or(camera1);
        let effective_camera2 = two_view_geometry.camera2.as_ref().unwrap_or(camera2);
        let (inlier_cam_rays1, inlier_cam_rays2) = extract_inlier_cam_rays(
            effective_camera1,
            &points1,
            effective_camera2,
            &points2,
            &two_view_geometry.inlier_matches,
        );

        if !maybe_fit_missing_two_view_geometry_matrix(
            &points1,
            &points2,
            &inlier_cam_rays1,
            &inlier_cam_rays2,
            &mut two_view_geometry,
        )? {
            continue;
        }

        let success = estimate_two_view_geometry_pose_from_cam_rays(
            camera1,
            camera2,
            &inlier_cam_rays1,
            &inlier_cam_rays2,
            &mut two_view_geometry,
        )?;

        if success {
            if let Some(cam2_from_cam1) = two_view_geometry.cam2_from_cam1.as_mut() {
                let norm = cam2_from_cam1.translation.norm();
                if norm > 1e-12 {
                    cam2_from_cam1.translation /= norm;
                }
                database_cache
                    .correspondence_graph_mut()
                    .update_two_view_geometry(image_id1, image_id2, two_view_geometry)?;
            }
        }
    }
    Ok(())
}
