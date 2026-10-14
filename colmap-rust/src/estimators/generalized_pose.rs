//! Port of the robust estimation half of COLMAP's `colmap/estimators/generalized_pose.h` and
//! `generalized_pose.cc`: pose estimation for multi-camera rigs (generalized cameras).
//! [`estimate_generalized_absolute_pose`] runs GP3P RANSAC with unique-inlier support,
//! [`estimate_generalized_relative_pose`] runs GR6P/GR8P LO-RANSAC (or, when both rigs are
//! panoramic, the central [`crate::estimators::pose::estimate_relative_pose`]) and
//! [`estimate_structure_less_absolute_pose`] registers a camera from 2D-2D correspondences
//! only (Zheng and Wu 2013). Built on [`crate::estimators::solvers`]'s `generalized_absolute_pose`
//! and `generalized_relative_pose`; the single-camera versions are in
//! [`crate::estimators::pose`]. Port of colmap-sharp's `Estimators/GeneralizedPoseEstimation.cs`.
//! Tests: `tests/estimators/generalized_pose.rs` (generalized_pose_test.cc).
//!
//! Not here yet: `RefineGeneralizedAbsolutePose`, which runs a Ceres problem (the nonlinear
//! least squares phase).
//!
//! Tier C (RANSAC) throughout.
//!
//! Translation notes:
//! - COLMAP returns `bool` and writes the outputs through pointers; here the functions return
//!   `Ok(None)` for COLMAP's `false` and the outputs otherwise. The relative pose's two
//!   `std::optional<Rigid3d>` outputs, of which COLMAP sets exactly one, are the
//!   [`GeneralizedRelativePose`] enum.
//! - Camera indices are `usize` (`size_t`), so `ThrowCheckCameras`' `>= 0` check always holds.
//! - `IsPanoramicRig` takes the smallest camera index as its reference, where COLMAP takes the
//!   first element of a hash set (docs/CPP_DIVERGENCES.md, entry 262).
//! - Empty correspondence lists return `Ok(None)` where COLMAP dereferences an empty range
//!   (docs/CPP_DIVERGENCES.md, entry 263).

use crate::estimators::pose::{estimate_relative_pose, PoseEstimate};
use crate::estimators::solvers::{
    GP3PEstimator, GP3PObservation, GP3PResidualType, GR6PEstimator, GR8PEstimator, GRNPObservation,
};
use crate::geometry::pose::CamRayWithJac;
use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{Matrix3x4d, Vector2d, Vector3d};
use crate::optim::{
    EstimateAsLocal, Estimator, LoRansac, RandomSampler, Ransac, RansacOptions, Sampler,
    UniqueInlierSupportMeasurer,
};
use crate::scene::camera::Camera;
use crate::{check, check_eq, check_gt, check_lt, Result};

/// Port of `colmap::StructureLessAbsolutePoseEstimationOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StructureLessAbsolutePoseEstimationOptions {
    /// Options used for RANSAC.
    pub ransac_options: RansacOptions,
}

impl Default for StructureLessAbsolutePoseEstimationOptions {
    fn default() -> Self {
        Self {
            ransac_options: RansacOptions {
                max_error: 6.0,
                // Use high confidence to avoid preemptive termination o RANSAC
                // - too early termination may lead to bad registration.
                min_num_trials: 100,
                max_num_trials: 10000,
                confidence: 0.99999,
                ..RansacOptions::default()
            },
        }
    }
}

impl StructureLessAbsolutePoseEstimationOptions {
    /// Port of `StructureLessAbsolutePoseEstimationOptions::Check`.
    pub fn check(&self) -> Result<()> {
        self.ransac_options.check()
    }
}

/// The pose [`estimate_generalized_relative_pose`] found: COLMAP's `rig2_from_rig1` and
/// `pano2_from_pano1` outputs, of which it sets exactly one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GeneralizedRelativePose {
    /// `rig2_from_rig1`, with metric translation: at least one rig is non-panoramic.
    Rig(Rigid3d),
    /// `pano2_from_pano1`: both rigs are panoramic (their cameras share one optical center),
    /// so the rigs are treated as central cameras and the translation is known only up to
    /// scale (unit length).
    Panoramic(Rigid3d),
}

impl GeneralizedRelativePose {
    /// COLMAP's `rig2_from_rig1` output: the pose when the rigs are not both panoramic.
    pub fn rig2_from_rig1(&self) -> Option<Rigid3d> {
        match *self {
            Self::Rig(pose) => Some(pose),
            Self::Panoramic(_) => None,
        }
    }

    /// COLMAP's `pano2_from_pano1` output: the pose when both rigs are panoramic.
    pub fn pano2_from_pano1(&self) -> Option<Rigid3d> {
        match *self {
            Self::Rig(_) => None,
            Self::Panoramic(pose) => Some(pose),
        }
    }

    /// Whichever of the two poses is set, as COLMAP's callers read it
    /// (`maybe_rig2_from_rig1.has_value() ? ... : maybe_pano2_from_pano1.value()`).
    pub fn pose(&self) -> Rigid3d {
        match *self {
            Self::Rig(pose) | Self::Panoramic(pose) => pose,
        }
    }
}

/// The outputs of a successful [`estimate_generalized_relative_pose`].
#[derive(Clone, Debug, PartialEq)]
pub struct GeneralizedRelativePoseEstimate {
    /// The relative pose of the second rig.
    pub pose: GeneralizedRelativePose,
    /// Number of inliers in RANSAC.
    pub num_inliers: usize,
    /// Inlier mask of the correspondences.
    pub inlier_mask: Vec<bool>,
}

/// Port of `colmap::EstimateGeneralizedAbsolutePose`: estimate the generalized absolute pose
/// `rig_from_world` from 2D-3D correspondences, where `camera_idxs[i]` is the rig camera
/// (index into `cams_from_rig` and `cameras`) that observes correspondence `i`.
///
/// `options.max_error` is in pixels; it is converted to each correspondence's camera and
/// averaged. The returned [`PoseEstimate::num_inliers`] counts unique 3D points: a point seen
/// by several rig cameras (field-of-view overlap) counts once. `Ok(None)` when no pose is
/// found.
pub fn estimate_generalized_absolute_pose(
    options: &RansacOptions,
    points2d: &[Vector2d],
    points3d: &[Vector3d],
    camera_idxs: &[usize],
    cams_from_rig: &[Rigid3d],
    cameras: &[Camera],
) -> Result<Option<PoseEstimate>> {
    check_eq!(points2d.len(), points3d.len());
    check_eq!(points2d.len(), camera_idxs.len());
    throw_check_cameras(camera_idxs, cams_from_rig, cameras)?;
    options.check()?;
    if points2d.is_empty() {
        return Ok(None);
    }

    // Precompute cam_from_rig matrices for fast residual computation
    let cams_from_rig_matrices: Vec<Matrix3x4d> =
        cams_from_rig.iter().map(Rigid3d::to_matrix).collect();

    let rig_points2d: Vec<GP3PObservation> = points2d
        .iter()
        .zip(camera_idxs)
        .map(|(&point2d, &camera_idx)| GP3PObservation {
            cam_from_rig: cams_from_rig_matrices[camera_idx],
            ray_in_cam: cameras[camera_idx]
                .cam_ray_from_img(point2d)
                .unwrap_or(Vector3d::new(0.0, 0.0, 0.0)),
        })
        .collect();

    // Associate unique ids to each 3D point.
    // Needed for UniqueInlierSupportMeasurer to avoid counting the same
    // 3D point multiple times due to FoV overlap in rig.
    // TODO(sarlinpe): Allow passing unique_point3D_ids as argument.
    let unique_point3d_ids = compute_unique_point_ids(points3d);

    // Average of the errors over the cameras, weighted by the number of
    // correspondences
    let mut options_copy = *options;
    options_copy.max_error = compute_max_error_in_camera(camera_idxs, cameras, options.max_error)?;

    let mut ransac: Ransac<GP3PEstimator, UniqueInlierSupportMeasurer> = Ransac::with_parts(
        &options_copy,
        GP3PEstimator::new(GP3PResidualType::ReprojectionError),
        UniqueInlierSupportMeasurer::new(unique_point3d_ids),
        RandomSampler::new(GP3PEstimator::MIN_NUM_SAMPLES),
    )?;
    let report = ransac.estimate(&rig_points2d, points3d)?;
    if !report.success {
        return Ok(None);
    }

    Ok(Some(PoseEstimate {
        pose: report.model,
        num_inliers: report.support.num_unique_inliers,
        inlier_mask: report.inlier_mask,
    }))
}

/// Port of `colmap::EstimateGeneralizedRelativePose`: estimate the relative pose of two rigs
/// from 2D-2D correspondences, where `camera_idxs1[i]` and `camera_idxs2[i]` are the rig
/// cameras observing correspondence `i` in the first and second rig.
///
/// When both rigs are panoramic the result is [`GeneralizedRelativePose::Panoramic`] (the
/// translation is known up to scale), otherwise [`GeneralizedRelativePose::Rig`].
/// `ransac_options.max_error` is in pixels: both branches score with the pixel-unit tangent
/// Sampson error. `Ok(None)` when no pose is found.
pub fn estimate_generalized_relative_pose(
    ransac_options: &RansacOptions,
    points2d1: &[Vector2d],
    points2d2: &[Vector2d],
    camera_idxs1: &[usize],
    camera_idxs2: &[usize],
    cams_from_rig: &[Rigid3d],
    cameras: &[Camera],
) -> Result<Option<GeneralizedRelativePoseEstimate>> {
    check_eq!(points2d1.len(), points2d2.len());
    throw_check_cameras(camera_idxs1, cams_from_rig, cameras)?;
    throw_check_cameras(camera_idxs2, cams_from_rig, cameras)?;
    ransac_options.check()?;

    let num_points = points2d1.len();
    if num_points == 0 {
        return Ok(None);
    }
    // COLMAP indexes `camera_idxs1[i]` / `camera_idxs2[i]` for every point without checking
    // their lengths; a check keeps a short list an error rather than an out-of-range read
    // (docs/CPP_DIVERGENCES.md, entry 263).
    check_eq!(camera_idxs1.len(), num_points);
    check_eq!(camera_idxs2.len(), num_points);

    // Both branches below score with the pixel-unit tangent Sampson error, so the
    // RANSAC threshold is the plain pixel ransac_options throughout. No
    // per-camera conversion to normalized/angular units is needed.
    if is_panoramic_rig(camera_idxs1, cams_from_rig)
        && is_panoramic_rig(camera_idxs2, cams_from_rig)
    {
        // EstimateRelativePose treats the panoramic rig as one central camera, so
        // each ray carries its unprojection Jacobian, rotated into the rig frame by
        // the same rotation as the ray. Unprojectable points are zeroed, which the
        // tangent Sampson residual reports as infinite (rejected).
        let rays_in_rig = |points2d: &[Vector2d], camera_idxs: &[usize]| -> Vec<CamRayWithJac> {
            points2d
                .iter()
                .zip(camera_idxs)
                .map(|(&point2d, &camera_idx)| {
                    let rig_from_cam = cams_from_rig[camera_idx]
                        .rotation
                        .inverse()
                        .to_rotation_matrix();
                    match cameras[camera_idx].cam_ray_from_img_with_jac(point2d) {
                        Some(rj) => CamRayWithJac {
                            ray: rig_from_cam * rj.ray,
                            jacobian: rig_from_cam * rj.jacobian,
                        },
                        None => CamRayWithJac::zero(),
                    }
                })
                .collect()
        };
        let cam_rays1_with_jac = rays_in_rig(points2d1, camera_idxs1);
        let cam_rays2_with_jac = rays_in_rig(points2d2, camera_idxs2);
        return Ok(estimate_relative_pose(
            ransac_options,
            &cam_rays1_with_jac,
            &cam_rays2_with_jac,
        )?
        .map(|estimate| GeneralizedRelativePoseEstimate {
            pose: GeneralizedRelativePose::Panoramic(estimate.pose),
            num_inliers: estimate.num_inliers,
            inlier_mask: estimate.inlier_mask,
        }));
    }

    let observations = |points2d: &[Vector2d], camera_idxs: &[usize]| -> Vec<GRNPObservation> {
        points2d
            .iter()
            .zip(camera_idxs)
            .map(|(&point2d, &camera_idx)| GRNPObservation {
                cam_from_rig: cams_from_rig[camera_idx],
                ray_with_jac_in_cam: cameras[camera_idx]
                    .cam_ray_from_img_with_jac(point2d)
                    .unwrap_or(CamRayWithJac::zero()),
            })
            .collect()
    };
    let points1 = observations(points2d1, camera_idxs1);
    let points2 = observations(points2d2, camera_idxs2);

    let mut ransac: LoRansac<GR6PEstimator, EstimateAsLocal<GR8PEstimator>> = LoRansac::new(
        ransac_options,
        GR6PEstimator,
        EstimateAsLocal(GR8PEstimator),
    )?;
    let report = ransac.estimate(&points1, &points2)?;
    if !report.success {
        return Ok(None);
    }

    Ok(Some(GeneralizedRelativePoseEstimate {
        pose: GeneralizedRelativePose::Rig(report.model),
        num_inliers: report.support.num_inliers,
        inlier_mask: report.inlier_mask,
    }))
}

/// Port of `colmap::EstimateStructureLessAbsolutePose`: estimate the absolute pose
/// `query_cam_from_world` of a query camera from 2D-2D correspondences with registered world
/// cameras, without known 3D points ("Structure from Motion Using Structure-less Resection",
/// Zheng and Wu, 2013). `world_camera_idxs[i]` is the world camera (index into
/// `world_cams_from_world` and `world_cameras`) observing `world_points2d[i]`, which
/// corresponds to `query_points2d[i]`.
///
/// The world cameras act as one generalized camera, so they must not share an optical center:
/// a panoramic set of world cameras returns `Ok(None)`, as does a failed estimate.
pub fn estimate_structure_less_absolute_pose(
    options: &StructureLessAbsolutePoseEstimationOptions,
    query_points2d: &[Vector2d],
    world_points2d: &[Vector2d],
    world_camera_idxs: &[usize],
    world_cams_from_world: &[Rigid3d],
    world_cameras: &[Camera],
    query_camera: &Camera,
) -> Result<Option<PoseEstimate>> {
    check_eq!(world_points2d.len(), query_points2d.len());
    check_eq!(world_points2d.len(), world_camera_idxs.len());
    check_eq!(world_cams_from_world.len(), world_cameras.len());
    throw_check_cameras(world_camera_idxs, world_cams_from_world, world_cameras)?;
    options.check()?;

    // Not in COLMAP, whose IsPanoramicRig dereferences the first element of an empty set
    // (docs/CPP_DIVERGENCES.md, entry 263).
    if world_points2d.is_empty() {
        return Ok(None);
    }

    if is_panoramic_rig(world_camera_idxs, world_cams_from_world) {
        return Ok(None);
    }

    let world_obs: Vec<GRNPObservation> = world_points2d
        .iter()
        .zip(world_camera_idxs)
        .map(|(&point2d, &world_camera_idx)| GRNPObservation {
            cam_from_rig: world_cams_from_world[world_camera_idx],
            ray_with_jac_in_cam: world_cameras[world_camera_idx]
                .cam_ray_from_img_with_jac(point2d)
                .unwrap_or(CamRayWithJac::zero()),
        })
        .collect();
    let query_obs: Vec<GRNPObservation> = query_points2d
        .iter()
        .map(|&point2d| GRNPObservation {
            cam_from_rig: Rigid3d::default(),
            ray_with_jac_in_cam: query_camera
                .cam_ray_from_img_with_jac(point2d)
                .unwrap_or(CamRayWithJac::zero()),
        })
        .collect();

    // GR6P/GR8P score with the pixel-unit tangent Sampson error, so the RANSAC
    // threshold is the plain pixel max_error. No per-camera conversion needed.
    let mut ransac: LoRansac<GR6PEstimator, EstimateAsLocal<GR8PEstimator>> = LoRansac::new(
        &options.ransac_options,
        GR6PEstimator,
        EstimateAsLocal(GR8PEstimator),
    )?;
    let report = ransac.estimate(&world_obs, &query_obs)?;
    if !report.success {
        return Ok(None);
    }

    Ok(Some(PoseEstimate {
        pose: report.model,
        num_inliers: report.support.num_inliers,
        inlier_mask: report.inlier_mask,
    }))
}

// Port of the anonymous `ThrowCheckCameras`. COLMAP's `std::minmax_element` on an empty index
// list dereferences `end()` (undefined); an empty list has nothing to check here
// (docs/CPP_DIVERGENCES.md, entry 263).
fn throw_check_cameras(
    camera_idxs: &[usize],
    cams_from_rig: &[Rigid3d],
    cameras: &[Camera],
) -> Result<()> {
    check!(!cameras.is_empty());
    check_eq!(cams_from_rig.len(), cameras.len());
    if let Some(&max_camera_idx) = camera_idxs.iter().max() {
        check_lt!(max_camera_idx, cameras.len());
    }
    Ok(())
}

// Port of the anonymous `IsPanoramicRig`: whether every camera in `camera_idxs` has the same
// origin in the rig (relative tolerance 1e-6). The reference camera is the smallest index,
// where COLMAP takes the first element of a `FlatHashSet` (docs/CPP_DIVERGENCES.md, entry
// 262). Callers pass a non-empty list.
fn is_panoramic_rig(camera_idxs: &[usize], cams_from_rig: &[Rigid3d]) -> bool {
    let mut camera_idx_set = camera_idxs.to_vec();
    camera_idx_set.sort_unstable();
    camera_idx_set.dedup();
    let first_origin_in_rig = cams_from_rig[camera_idx_set[0]].tgt_origin_in_src();
    camera_idx_set[1..].iter().all(|&camera_idx| {
        let other_origin_in_rig = cams_from_rig[camera_idx].tgt_origin_in_src();
        first_origin_in_rig.is_approx_with(other_origin_in_rig, 1e-6)
    })
}

// Port of the anonymous `ComputeMaxErrorInCamera`: the pixel threshold converted to each
// correspondence's camera, averaged.
fn compute_max_error_in_camera(
    camera_idxs: &[usize],
    cameras: &[Camera],
    max_error_px: f64,
) -> Result<f64> {
    check_gt!(max_error_px, 0.0);
    let mut max_error_cam = 0.0;
    for &camera_idx in camera_idxs {
        max_error_cam += cameras[camera_idx].cam_from_img_threshold(max_error_px);
    }
    Ok(max_error_cam / camera_idxs.len() as f64)
}

// Port of the anonymous `LowerVector3d`: lexicographic x, y, z order.
fn lower_vector3d(v1: &Vector3d, v2: &Vector3d) -> bool {
    if v1.x < v2.x {
        true
    } else if v1.x == v2.x {
        if v1.y < v2.y {
            true
        } else if v1.y == v2.y {
            v1.z < v2.z
        } else {
            false
        }
    } else {
        false
    }
}

// Port of the anonymous `ComputeUniquePointIds`: sort the points lexicographically and give
// each run of points within relative 1e-5 of the run's first point the run's first sorted
// position as its id. `std::sort` is unstable; only exactly equal points tie, and they fall in
// one run with the same id whatever their order, so the stable sort here gives COLMAP's ids.
fn compute_unique_point_ids(points3d: &[Vector3d]) -> Vec<usize> {
    let mut point3d_ids: Vec<usize> = (0..points3d.len()).collect();
    point3d_ids.sort_by(|&i, &j| {
        if lower_vector3d(&points3d[i], &points3d[j]) {
            std::cmp::Ordering::Less
        } else if lower_vector3d(&points3d[j], &points3d[i]) {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    });

    let mut unique_it = 0;
    let mut unique_point3d_ids = vec![0; points3d.len()];
    for (current_it, &point3d_id) in point3d_ids.iter().enumerate() {
        if !points3d[point3d_ids[unique_it]].is_approx_with(points3d[point3d_id], 1e-5) {
            unique_it = current_it;
        }
        unique_point3d_ids[point3d_id] = unique_it;
    }
    unique_point3d_ids
}
