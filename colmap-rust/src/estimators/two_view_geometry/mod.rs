//! Port of COLMAP's `colmap/estimators/two_view_geometry.h` and `two_view_geometry.cc`:
//! robust verification of an image pair's matches into a [`TwoViewGeometry`] (E/F/H, its
//! configuration, the inlier matches and optionally the relative pose). Built on the minimal
//! solvers of [`crate::estimators::solvers`], DEGENSAC
//! ([`crate::estimators::fundamental_matrix_degensac_estimator`]) and LO-RANSAC
//! ([`crate::optim::LoRansac`]) with COLMAP's `MEstimatorSupportMeasurer`. Port of
//! colmap-sharp's `Estimators/TwoViewGeometryEstimation*.cs`. Split by responsibility:
//!
//! - this file: [`TwoViewGeometryOptions`], the [`estimate_two_view_geometry`] dispatch, the
//!   multiple-model estimation, watermark detection, stationary-match filtering,
//!   [`two_view_geometry_from_known_relative_pose`] and the shared helpers;
//! - `calibrated.rs`: the calibrated, uncalibrated, spherical and forced-homography paths;
//! - `focal.rs`: the shared-focal and one-sided-focal paths;
//! - `pose.rs`: the relative pose decomposition ([`estimate_two_view_geometry_pose`]) and
//!   [`maybe_decompose_relative_poses`];
//! - `rig.rs`: [`estimate_rig_two_view_geometries`].
//!
//! Tests: `tests/estimators/two_view_geometry.rs` (`two_view_geometry_test.cc`).
//!
//! Tier C (outcome) for the RANSAC-driven estimators; Tier B for the pose decomposition and
//! [`two_view_geometry_from_known_relative_pose`] (decompositions, no randomness).
//!
//! Translation notes:
//! - `THROW_CHECK*` failures are `Err`; every entry point returns `Result`.
//! - `static_cast<size_t>(options.min_num_inliers)` keeps C++'s conversion (a negative value
//!   wraps to a huge count), see `min_num_inliers`.
//! - COLMAP's `LOG(WARNING)` / `LOG(INFO)` lines are not emitted (the core has no logger), and
//!   the `Timer` of `MaybeDecomposeRelativePoses`, which only fed the log, is dropped. That is
//!   a deliberate divergence: `EstimateTwoViewGeometry` emits a one-time
//!   `LOG_FIRST_N(WARNING)` when it marks a pair degenerate because `force_H_use` was asked of
//!   a non-pinhole camera, and another when it marks an uncalibrated fisheye pair degenerate
//!   (asking for a focal length prior); `MaybeDecomposeRelativePoses` logs `LOG(INFO)` lines
//!   with its decomposed and failed counts and its run time. COLMAP's glog is not ported; the
//!   lines are diagnostic only, and every returned geometry, configuration and stored pose is
//!   unchanged. When a logger lands, this note goes away (as the same notes on
//!   `scene::correspondence_graph::CorrespondenceGraph::add_two_view_geometry` and
//!   `geometry::pose_prior::gravity_from_exif_orientation` do).
//!   `estimate_two_view_geometry_uncalibrated_fisheye_is_degenerate` and
//!   `estimate_two_view_geometry_force_h_use_with_fisheye_is_degenerate`
//!   (`tests/estimators/two_view_geometry.rs`), and `maybe_decompose_relative_poses_*`
//!   (`tests/estimators/two_view_geometry_misc.rs`; `two_view_geometry_test.cc` 1:1) pass.
//! - Inlier masks are `Vec<bool>` (COLMAP's `std::vector<char>`).

mod calibrated;
mod focal;
mod pose;
mod rig;

pub use calibrated::estimate_calibrated_two_view_geometry;
pub use focal::{
    estimate_one_sided_focal_two_view_geometry, estimate_shared_focal_two_view_geometry,
};
pub use pose::{estimate_two_view_geometry_pose, maybe_decompose_relative_poses};
pub use rig::{estimate_rig_two_view_geometries, ImagePairGeometry, ImagePairMatches};

use crate::estimators::solvers::TranslationTransformEstimator;
use crate::feature::types::{FeatureMatch, FeatureMatches};
use crate::geometry::essential_matrix::{
    compute_squared_tangent_sampson_errors, essential_matrix_from_pose,
};
use crate::geometry::pose::CamRayWithJac;
use crate::geometry::rigid3::Rigid3d;
use crate::linalg::Vector2d;
use crate::math::fns;
use crate::optim::{EstimateAsLocal, LoRansac, RansacOptions};
use crate::scene::camera::Camera;
use crate::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use crate::{check, check_ge, check_gt, Result};

/// Port of `colmap::TwoViewGeometryOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TwoViewGeometryOptions {
    /// Minimum number of inliers for non-degenerate two-view geometry.
    pub min_num_inliers: i32,
    /// Minimum ratio of inliers to total matches for non-degenerate geometry. Disabled by
    /// default, only effective when > 0.
    pub min_inlier_ratio: f64,
    /// In case both cameras are calibrated, the calibration is verified by estimating an
    /// essential and fundamental matrix and comparing their fractions of number of inliers.
    /// If the essential matrix produces a similar number of inliers
    /// (`min_e_f_inlier_ratio * F_num_inliers`), the calibration is assumed to be correct.
    pub min_e_f_inlier_ratio: f64,
    /// In case an epipolar geometry can be verified, it is checked whether the geometry
    /// describes a planar scene or panoramic view (pure rotation) described by a homography.
    /// This is a degenerate case, since epipolar geometry is only defined for a moving
    /// camera. If the inlier ratio of a homography comes close to the inlier ratio of the
    /// epipolar geometry, a planar or panoramic configuration is assumed.
    pub max_h_inlier_ratio: f64,
    /// In case of valid two-view geometry, it is checked whether the geometry describes a
    /// pure translation in the border region of the image. If more than a certain ratio of
    /// inlier points conform with a pure image translation, a watermark is assumed.
    pub watermark_min_inlier_ratio: f64,
    /// Watermark matches have to be in the border region of the image. The border region is
    /// defined as the area around the image borders and is defined as a fraction of the
    /// image diagonal.
    pub watermark_border_size: f64,
    /// Whether to enable watermark detection. A watermark causes a pure translation in the
    /// image space with inliers in the border region.
    pub detect_watermark: bool,
    /// Whether to ignore watermark models in multiple model estimation.
    pub multiple_ignore_watermark: bool,
    /// Maximum translational error of matched points to be considered inliers of a
    /// watermark.
    pub watermark_detection_max_error: f64,
    /// Whether to filter stationary matches. This is useful when a camera is rigidly mounted
    /// on a moving vehicle and the vehicle itself is visible.
    pub filter_stationary_matches: bool,
    /// Maximum displacement for points to be considered stationary matches.
    pub stationary_matches_max_error: f64,
    /// In case the user asks for it, only going to estimate a Homography between both
    /// cameras.
    pub force_h_use: bool,
    /// Use DEGENSAC (Chum et al., CVPR 2005) for the fundamental matrix instead of plain
    /// LO-RANSAC, making estimation robust to a dominant scene plane.
    pub use_degensac: bool,
    /// Locally optimize the fundamental matrix by nonlinearly minimizing the Sampson error
    /// over the inlier set, instead of refitting the linear eight-point algorithm. The
    /// refinement optimizes the same residual RANSAC scores and keeps the model rank 2
    /// throughout, so no singular value has to be truncated afterwards. On by default.
    pub use_sampson_refinement: bool,
    /// Whether to compute the relative pose between the two views.
    pub compute_relative_pose: bool,
    /// Recursively estimate multiple configurations by removing the previous set of inliers
    /// from the matches until not enough inliers are found. Inlier matches are concatenated
    /// and the configuration type is `MULTIPLE` if multiple models could be estimated. This
    /// is useful to estimate the two-view geometry for images with large distortion or
    /// multiple rigidly moving objects in the scene.
    ///
    /// Note that in case the model type is `MULTIPLE`, only the `inlier_matches` field will
    /// be initialized.
    pub multiple_models: bool,
    /// Options used to robustly estimate the geometry.
    pub ransac_options: RansacOptions,
}

impl Default for TwoViewGeometryOptions {
    fn default() -> Self {
        Self {
            min_num_inliers: 15,
            min_inlier_ratio: 0.0,
            min_e_f_inlier_ratio: 0.95,
            max_h_inlier_ratio: 0.8,
            watermark_min_inlier_ratio: 0.7,
            watermark_border_size: 0.1,
            detect_watermark: true,
            multiple_ignore_watermark: true,
            watermark_detection_max_error: 4.0,
            filter_stationary_matches: false,
            stationary_matches_max_error: 4.0,
            force_h_use: false,
            use_degensac: false,
            use_sampson_refinement: true,
            compute_relative_pose: false,
            multiple_models: false,
            ransac_options: RansacOptions {
                max_error: 4.0,
                confidence: 0.999,
                min_num_trials: 100,
                max_num_trials: 10000,
                min_inlier_ratio: 0.25,
                ..RansacOptions::default()
            },
        }
    }
}

impl TwoViewGeometryOptions {
    /// Port of `TwoViewGeometryOptions::Check` (`CHECK_OPTION_*`: false on a violation; NaN
    /// fails, as the C++ comparisons do).
    pub fn check(&self) -> bool {
        let in_unit = |v: f64| (0.0..=1.0).contains(&v);
        let r = &self.ransac_options;
        self.min_num_inliers >= 0
            && in_unit(self.min_e_f_inlier_ratio)
            && in_unit(self.max_h_inlier_ratio)
            && in_unit(self.watermark_min_inlier_ratio)
            && in_unit(self.watermark_border_size)
            && r.max_error > 0.0
            && in_unit(r.min_inlier_ratio)
            && in_unit(r.confidence)
            && r.min_num_trials <= r.max_num_trials
            && r.random_seed >= -1
    }
}

/// `static_cast<size_t>(options.min_num_inliers)`: C++'s conversion, so a negative option
/// wraps to a huge count rather than to zero.
pub(super) fn min_num_inliers(value: i32) -> usize {
    value as isize as usize
}

/// The RANSAC options of the calibrated, spherical and focal paths: `options.ransac_options`
/// with `min_inlier_ratio` replaced by the option's own when that is enabled (> 0).
pub(super) fn ransac_options_with_min_inlier_ratio(
    options: &TwoViewGeometryOptions,
) -> RansacOptions {
    let mut ransac_options = options.ransac_options;
    if options.min_inlier_ratio > 0.0 {
        ransac_options.min_inlier_ratio = options.min_inlier_ratio;
    }
    ransac_options
}

/// A geometry with only its configuration set to `DEGENERATE`.
pub(super) fn degenerate_geometry() -> TwoViewGeometry {
    TwoViewGeometry {
        config: ConfigurationType::Degenerate,
        ..TwoViewGeometry::default()
    }
}

/// The image points of each match: `(points1[idx1], points2[idx2])` lists.
pub(super) fn matched_img_points(
    points1: &[Vector2d],
    points2: &[Vector2d],
    matches: &[FeatureMatch],
) -> (Vec<Vector2d>, Vec<Vector2d>) {
    matches
        .iter()
        .map(|m| {
            (
                points1[m.point2d_idx1 as usize],
                points2[m.point2d_idx2 as usize],
            )
        })
        .unzip()
}

/// Port of `ExtractInlierMatches`: the matches whose mask entry is set, in order.
pub(super) fn extract_inlier_matches(
    matches: &[FeatureMatch],
    num_inliers: usize,
    inlier_mask: &[bool],
) -> FeatureMatches {
    let mut inlier_matches = FeatureMatches::with_capacity(num_inliers);
    for (feature_match, &is_inlier) in matches.iter().zip(inlier_mask) {
        if is_inlier {
            inlier_matches.push(*feature_match);
        }
    }
    inlier_matches
}

/// Port of `ExtractOutlierMatches`: the matches whose index pair is not among
/// `inlier_matches`, in order.
fn extract_outlier_matches(
    matches: &[FeatureMatch],
    inlier_matches: &[FeatureMatch],
) -> Result<FeatureMatches> {
    check_ge!(matches.len(), inlier_matches.len());
    // Only membership is queried, so the set's iteration order never reaches an output.
    let inlier_matches_set: std::collections::HashSet<(u32, u32)> = inlier_matches
        .iter()
        .map(|m| (m.point2d_idx1, m.point2d_idx2))
        .collect();
    Ok(matches
        .iter()
        .filter(|m| !inlier_matches_set.contains(&(m.point2d_idx1, m.point2d_idx2)))
        .copied()
        .collect())
}

/// Port of `IsCameraCalibrated`: whether a camera's intrinsics are known. It either carries
/// a focal prior, or is spherical and has no focal length to estimate in the first place.
pub(super) fn is_camera_calibrated(camera: &Camera) -> bool {
    camera.is_spherical() || camera.has_prior_focal_length
}

/// Port of `colmap::EstimateTwoViewGeometry`: estimate the two-view geometry from
/// calibrated or uncalibrated image pairs, depending on whether a prior focal length is
/// given or not.
pub fn estimate_two_view_geometry(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    mut matches: FeatureMatches,
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    if options.filter_stationary_matches {
        filter_stationary_matches(
            options.stationary_matches_max_error,
            points1,
            points2,
            &mut matches,
        );
    }
    if options.multiple_models {
        let mut multiple_model_options = *options;
        // Set to false to prevent recursive calls to this function.
        multiple_model_options.multiple_models = false;
        // Set to false to prevent redundant filtering of stationary matches.
        multiple_model_options.filter_stationary_matches = false;
        estimate_multiple_two_view_geometries(
            camera1,
            points1,
            camera2,
            points2,
            &matches,
            &multiple_model_options,
        )
    } else if options.force_h_use {
        // In image coordinates, a homography relates two views of a plane only under a
        // pinhole projection. Fisheye and spherical models map the plane non-linearly, so
        // the estimated homography would be meaningless. Such pairs are marked degenerate
        // (COLMAP also logs a one-time warning).
        if !camera1.is_perspective_pinhole() || !camera2.is_perspective_pinhole() {
            return Ok(degenerate_geometry());
        }
        calibrated::estimate_calibrated_homography(
            camera1, points1, camera2, points2, &matches, options,
        )
    } else if is_camera_calibrated(camera1) != is_camera_calibrated(camera2)
        && if is_camera_calibrated(camera1) {
            camera2
        } else {
            camera1
        }
        .is_perspective_pinhole()
    {
        // Exactly one side is known, either from a focal prior or because it is spherical
        // and has no focal at all. Recover the other side's focal rather than discard the
        // known intrinsics. Only the uncalibrated side must be a pinhole projection; the
        // calibrated side enters as rays, so any model is admissible there.
        estimate_one_sided_focal_two_view_geometry(
            camera1, points1, camera2, points2, &matches, options,
        )
    } else if camera1.is_spherical() || camera2.is_spherical() {
        // No pinhole image plane, so the fundamental matrix is not meaningful. Mixed
        // spherical/uncalibrated pairs are caught above.
        calibrated::estimate_spherical_two_view_geometry(
            camera1, points1, camera2, points2, &matches, options,
        )
    } else if camera1.camera_id == camera2.camera_id
        && !camera1.has_prior_focal_length
        && camera1.is_perspective_pinhole()
    {
        // A single shared unknown focal. Multi-focal models (e.g. PINHOLE) are seeded
        // isotropically (fx = fy = f) and refined later by bundle adjustment; distortion is
        // absorbed by the epipolar fit, as in the fundamental-matrix path.
        estimate_shared_focal_two_view_geometry(camera1, points1, points2, &matches, options)
    } else if camera1.has_prior_focal_length && camera2.has_prior_focal_length {
        // Both focals are known, so the pair reduces to the relative pose.
        estimate_calibrated_two_view_geometry(camera1, points1, camera2, points2, &matches, options)
    } else if !camera1.is_perspective_pinhole() || !camera2.is_perspective_pinhole() {
        // Without a focal-length prior, the only remaining option is the
        // fundamental-matrix path below, which assumes a pinhole projection that a fisheye
        // camera does not have. The calibrated path above does handle fisheye, as it works
        // on bearing vectors. COLMAP also logs a one-time warning asking for a focal prior.
        Ok(degenerate_geometry())
    } else {
        // Two independent unknown focals, recovered from the fundamental matrix.
        calibrated::estimate_uncalibrated_two_view_geometry(
            camera1, points1, camera2, points2, &matches, options,
        )
    }
}

/// Port of `EstimateMultipleTwoViewGeometries`: repeatedly estimate a geometry on the
/// matches the previous ones left as outliers, until a degenerate one.
fn estimate_multiple_two_view_geometries(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    matches: &[FeatureMatch],
    options: &TwoViewGeometryOptions,
) -> Result<TwoViewGeometry> {
    let mut remaining_matches: FeatureMatches = matches.to_vec();
    let mut geometries: Vec<TwoViewGeometry> = Vec::new();
    loop {
        let geometry = estimate_two_view_geometry(
            camera1,
            points1,
            camera2,
            points2,
            remaining_matches.clone(),
            options,
        )?;
        if geometry.config == ConfigurationType::Degenerate {
            break;
        }

        remaining_matches = extract_outlier_matches(&remaining_matches, &geometry.inlier_matches)?;

        if !options.multiple_ignore_watermark || geometry.config != ConfigurationType::Watermark {
            geometries.push(geometry);
        }
    }

    let num_geometries = geometries.len();
    if num_geometries == 0 {
        return Ok(degenerate_geometry());
    }
    if num_geometries == 1 {
        return Ok(geometries.swap_remove(0));
    }
    let mut multi_geometry = TwoViewGeometry {
        config: ConfigurationType::Multiple,
        ..TwoViewGeometry::default()
    };
    for geometry in &geometries {
        multi_geometry
            .inlier_matches
            .extend_from_slice(&geometry.inlier_matches);
    }
    Ok(multi_geometry)
}

/// Port of `colmap::DetectWatermarkMatches`: whether the inlier matches are a watermark, a
/// pure image translation with most inliers in the border region of both images. `points1`
/// and `points2` are the matched points (one per mask entry).
pub fn detect_watermark_matches(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    num_inliers: usize,
    inlier_mask: &[bool],
    options: &TwoViewGeometryOptions,
) -> Result<bool> {
    check!(options.check());

    // Check if inlier points in border region and extract inlier matches.

    // `camera.width * camera.width + ...` is `size_t` arithmetic, converted for the sqrt.
    let diagonal1 =
        fns::sqrt((camera1.width * camera1.width + camera1.height * camera1.height) as f64);
    let diagonal2 =
        fns::sqrt((camera2.width * camera2.width + camera2.height * camera2.height) as f64);
    let border_size1 = options.watermark_border_size * diagonal1;
    let border_size2 = options.watermark_border_size * diagonal2;
    // `Eigen::AlignedBox2d::contains`: min <= p <= max componentwise.
    let contains = |camera: &Camera, border_size: f64, p: Vector2d| {
        border_size <= p.x
            && border_size <= p.y
            && p.x <= camera.width as f64 - border_size
            && p.y <= camera.height as f64 - border_size
    };

    let mut inlier_points1: Vec<Vector2d> = Vec::with_capacity(num_inliers);
    let mut inlier_points2: Vec<Vector2d> = Vec::with_capacity(num_inliers);
    let mut num_matches_in_border = 0usize;
    for (i, &is_inlier) in inlier_mask.iter().enumerate() {
        if is_inlier {
            inlier_points1.push(points1[i]);
            inlier_points2.push(points2[i]);
            if !contains(camera1, border_size1, points1[i])
                && !contains(camera2, border_size2, points2[i])
            {
                num_matches_in_border += 1;
            }
        }
    }

    let matches_in_border_ratio = num_matches_in_border as f64 / num_inliers as f64;
    if matches_in_border_ratio < options.watermark_min_inlier_ratio {
        return Ok(false);
    }

    // Check if matches follow a translational model.

    let mut ransac_options = options.ransac_options;
    ransac_options.max_error = options.watermark_detection_max_error;
    ransac_options.min_inlier_ratio = options.watermark_min_inlier_ratio;

    let mut ransac: LoRansac<
        TranslationTransformEstimator,
        EstimateAsLocal<TranslationTransformEstimator>,
    > = LoRansac::new(
        &ransac_options,
        TranslationTransformEstimator,
        EstimateAsLocal(TranslationTransformEstimator),
    )?;
    let report = ransac.estimate(&inlier_points1, &inlier_points2)?;

    let inlier_ratio = report.support.num_inliers as f64 / num_inliers as f64;
    Ok(inlier_ratio >= options.watermark_min_inlier_ratio)
}

/// Port of `colmap::FilterStationaryMatches`: remove the matches whose two points are at most
/// `max_error` apart.
pub fn filter_stationary_matches(
    max_error: f64,
    points1: &[Vector2d],
    points2: &[Vector2d],
    matches: &mut FeatureMatches,
) {
    let max_error_squared = max_error * max_error;
    matches.retain(|m| {
        (points1[m.point2d_idx1 as usize] - points2[m.point2d_idx2 as usize]).squared_norm()
            > max_error_squared
    });
}

/// Port of `colmap::TwoViewGeometryFromKnownRelativePose`: the `CALIBRATED` geometry of a
/// known `cam2_from_cam1`, keeping the matches whose tangent Sampson error is within
/// `max_error` pixels (COLMAP's defaults: `min_num_inliers` 15, `max_error` 4).
#[allow(clippy::too_many_arguments)]
pub fn two_view_geometry_from_known_relative_pose(
    camera1: &Camera,
    points1: &[Vector2d],
    camera2: &Camera,
    points2: &[Vector2d],
    cam2_from_cam1: &Rigid3d,
    matches: &[FeatureMatch],
    min_num_inliers: i32,
    max_error: f64,
) -> Result<TwoViewGeometry> {
    check_ge!(min_num_inliers, 0);
    check_gt!(max_error, 0.0);

    let num_matches = matches.len();
    if num_matches < min_num_inliers as usize {
        return Ok(degenerate_geometry());
    }

    // Score in pixels with the tangent Sampson error, matching the design of
    // EstimateCalibratedTwoViewGeometry, so that a pair filtered here and a pair verified
    // there are held to the same threshold.
    let (rays1, rays2): (Vec<CamRayWithJac>, Vec<CamRayWithJac>) = matches
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
        .unzip();

    let e = essential_matrix_from_pose(cam2_from_cam1);
    let residuals = compute_squared_tangent_sampson_errors(&rays1, &rays2, &e)?;
    let squared_max_error = max_error * max_error;
    let inlier_matches: FeatureMatches = matches
        .iter()
        .zip(&residuals)
        .filter(|(_, &residual)| residual <= squared_max_error)
        .map(|(m, _)| *m)
        .collect();
    if inlier_matches.len() < min_num_inliers as usize {
        return Ok(degenerate_geometry());
    }
    Ok(TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        cam2_from_cam1: Some(*cam2_from_cam1),
        e: Some(e),
        inlier_matches,
        ..TwoViewGeometry::default()
    })
}
