//! Port of COLMAP's `colmap/estimators/triangulation.h` and `triangulation.cc`: robust
//! multi-view point triangulation. [`TriangulationEstimator`] triangulates one 3D point from
//! a minimal (two-view DLT) or non-minimal (multi-view DLT on bearings) sample and rejects it
//! on cheirality or an insufficient triangulation angle; [`estimate_triangulation`] runs it
//! inside LO-RANSAC over every pair of views ([`crate::optim::combination_sampler`]). Port of
//! colmap-sharp's `Estimators/TriangulationEstimation.cs`. Builds on
//! [`crate::geometry::triangulation`] (the DLT solvers) and [`crate::scene::projection`] (the
//! residuals and the cheirality test). Consumers: the incremental triangulator and mapper.
//!
//! COLMAP has no `estimators/triangulation_test.cc` (`EstimateTriangulation` is covered
//! upstream only through the mapper tests and pycolmap's binding smoke tests), so the tests
//! are Rust-only, ported from colmap-sharp's: `tests/estimators/rust_only_triangulation.rs`.
//! Tier C (RANSAC) for `estimate_triangulation`; the estimator itself is Tier B (SVD).
//!
//! Translation notes: `PoseData::camera` is COLMAP's `const Camera*`, here a borrow, so the
//! estimator carries the lifetime of the cameras it scores against. `THROW_CHECK_NOTNULL` on
//! the camera has no counterpart (a reference is never null). COLMAP returns `bool` and
//! writes the inlier mask and point through pointers; here [`estimate_triangulation`]
//! returns `Ok(None)` for COLMAP's `false`.

use crate::geometry::rigid3::Rigid3d;
use crate::geometry::triangulation::{
    calculate_triangulation_angle, triangulate_multi_view_point_from_rays, triangulate_point,
    triangulate_point_from_rays,
};
use crate::linalg::{Matrix3x4d, Vector2d, Vector3d};
use crate::math::deg_to_rad;
use crate::optim::combination_sampler::CombinationSampler;
use crate::optim::estimator::{EstimateAsLocal, Estimator};
use crate::optim::loransac::LoRansac;
use crate::optim::ransac::RansacOptions;
use crate::optim::support_measurement::InlierSupportMeasurer;
use crate::scene::camera::Camera;
use crate::scene::projection::{
    calculate_angular_reprojection_error_from_ray, calculate_squared_reprojection_error,
    has_point_positive_depth,
};
use crate::{check_eq, check_ge, Result};

/// Port of `TriangulationEstimator::ResidualType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualType {
    /// Squared angle between the observed ray and the ray to the point.
    AngularError,
    /// Squared reprojection error in pixels.
    ReprojectionError,
}

/// Port of `TriangulationEstimator::PointData`: one observation.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PointData {
    /// Image observation in pixels. Only needs to be set for `ReprojectionError`.
    pub img_point: Vector2d,
    /// Unit bearing vector in the camera frame (`Camera::CamRayFromImg`). The canonical
    /// observation representation for all camera models, including omnidirectional
    /// (EQUIRECTANGULAR) back-hemisphere rays that the 2D normalized representation cannot
    /// encode.
    pub cam_ray: Vector3d,
}

/// Port of `TriangulationEstimator::PoseData`: the pose and calibration of an observation.
#[derive(Clone, Copy, Debug)]
pub struct PoseData<'a> {
    /// The projection matrix for the image of the observation.
    pub cam_from_world: Matrix3x4d,
    /// The projection center for the image of the observation.
    pub proj_center: Vector3d,
    /// The camera for the image of the observation.
    pub camera: &'a Camera,
}

/// Port of `colmap::TriangulationEstimator`: estimates a 3D point from multiple
/// observations. The triangulation must satisfy the following constraints: sufficient
/// triangulation angle between observation pairs, and all observations must satisfy the
/// cheirality constraint.
#[derive(Clone, Copy, Debug)]
pub struct TriangulationEstimator<'a> {
    min_tri_angle: f64,
    residual_type: ResidualType,
    _cameras: std::marker::PhantomData<&'a Camera>,
}

impl TriangulationEstimator<'_> {
    /// Port of `TriangulationEstimator(min_tri_angle, residual_type)`; `min_tri_angle` in
    /// radians.
    pub fn new(min_tri_angle: f64, residual_type: ResidualType) -> Result<Self> {
        check_ge!(min_tri_angle, 0.0);
        Ok(Self {
            min_tri_angle,
            residual_type,
            _cameras: std::marker::PhantomData,
        })
    }
}

impl<'a> Estimator for TriangulationEstimator<'a> {
    type X = PointData;
    type Y = PoseData<'a>;
    type M = Vector3d;

    const MIN_NUM_SAMPLES: usize = 2;

    /// Port of `TriangulationEstimator::Estimate`: pushes the triangulated point when it
    /// passes the cheirality and triangulation angle checks.
    fn estimate(
        &mut self,
        point_data: &[PointData],
        pose_data: &[PoseData<'a>],
        models: &mut Vec<Vector3d>,
    ) -> Result<()> {
        check_ge!(point_data.len(), 2);
        check_eq!(point_data.len(), pose_data.len());

        models.clear();

        let xyz = if point_data.len() == 2 {
            // More efficient closed-form solution for the two-view case.
            let all_cams_perspective = pose_data.iter().all(|pose| pose.camera.is_perspective());
            let xyz = if all_cams_perspective {
                triangulate_point(
                    &pose_data[0].cam_from_world,
                    &pose_data[1].cam_from_world,
                    point_data[0].cam_ray.hnormalized(),
                    point_data[1].cam_ray.hnormalized(),
                )
            } else {
                triangulate_point_from_rays(
                    &pose_data[0].cam_from_world,
                    &pose_data[1].cam_from_world,
                    point_data[0].cam_ray,
                    point_data[1].cam_ray,
                )
            };
            match xyz {
                Some(xyz) => xyz,
                None => return Ok(()),
            }
        } else {
            let cams_from_world: Vec<Matrix3x4d> =
                pose_data.iter().map(|pose| pose.cam_from_world).collect();
            let cam_rays: Vec<Vector3d> = point_data.iter().map(|point| point.cam_ray).collect();
            match triangulate_multi_view_point_from_rays(&cams_from_world, &cam_rays)? {
                Some(xyz) => xyz,
                None => return Ok(()),
            }
        };

        // Cheirality. Perspective cameras require positive depth (the point in front of the
        // local +Z axis). Omnidirectional cameras (e.g. EQUIRECTANGULAR) have no single
        // front, but the point must still lie in the half-space the observed bearing points
        // toward.
        for (pose, point) in pose_data.iter().zip(point_data) {
            if pose.camera.is_perspective() {
                if !has_point_positive_depth(&pose.cam_from_world, xyz) {
                    return Ok(());
                }
            } else if (pose.cam_from_world * xyz.homogeneous()).dot(point.cam_ray) <= 0.0 {
                return Ok(());
            }
        }

        // Require a sufficient triangulation angle for at least one pair of views.
        for i in 0..pose_data.len() {
            for j in 0..i {
                if calculate_triangulation_angle(
                    pose_data[i].proj_center,
                    pose_data[j].proj_center,
                    xyz,
                ) >= self.min_tri_angle
                {
                    models.push(xyz);
                    return Ok(());
                }
            }
        }
        Ok(())
    }

    /// Port of `TriangulationEstimator::Residuals`: squared reprojection or angular error of
    /// every observation.
    fn residuals(
        &mut self,
        point_data: &[PointData],
        pose_data: &[PoseData<'a>],
        xyz: &Vector3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        check_eq!(point_data.len(), pose_data.len());

        residuals.resize(point_data.len(), 0.0);

        for (i, (point, pose)) in point_data.iter().zip(pose_data).enumerate() {
            residuals[i] = match self.residual_type {
                ResidualType::ReprojectionError => calculate_squared_reprojection_error(
                    point.img_point,
                    *xyz,
                    &pose.cam_from_world,
                    pose.camera,
                ),
                ResidualType::AngularError => {
                    let angular_error = calculate_angular_reprojection_error_from_ray(
                        point.cam_ray,
                        *xyz,
                        &pose.cam_from_world,
                    );
                    angular_error * angular_error
                }
            };
        }
        Ok(())
    }
}

/// Port of `colmap::EstimateTriangulationOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EstimateTriangulationOptions {
    /// Minimum triangulation angle in radians.
    pub min_tri_angle: f64,
    /// The employed residual type.
    pub residual_type: ResidualType,
    /// RANSAC options for `TriangulationEstimator`.
    pub ransac_options: RansacOptions,
}

impl Default for EstimateTriangulationOptions {
    fn default() -> Self {
        Self {
            min_tri_angle: 0.0,
            residual_type: ResidualType::AngularError,
            ransac_options: RansacOptions {
                max_error: deg_to_rad(2.0),
                confidence: 0.9999,
                min_inlier_ratio: 0.02,
                max_num_trials: 10000,
                ..RansacOptions::default()
            },
        }
    }
}

impl EstimateTriangulationOptions {
    /// Port of `EstimateTriangulationOptions::Check`.
    pub fn check(&self) -> Result<()> {
        check_ge!(self.min_tri_angle, 0.0);
        self.ransac_options.check()
    }
}

/// The result of a successful [`estimate_triangulation`]: COLMAP's `inlier_mask` and `xyz`
/// out-parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct TriangulationResult {
    /// Whether each observation is an inlier of `xyz`.
    pub inlier_mask: Vec<bool>,
    /// The triangulated point.
    pub xyz: Vector3d,
}

/// Port of `colmap::EstimateTriangulation`: robustly estimate a 3D point from observations
/// in multiple views using LO-RANSAC over all view pairs. `Ok(None)` is COLMAP's `false`
/// (no model found).
pub fn estimate_triangulation(
    options: &EstimateTriangulationOptions,
    points: &[Vector2d],
    cams_from_world: &[Rigid3d],
    cameras: &[&Camera],
) -> Result<Option<TriangulationResult>> {
    check_ge!(points.len(), 2);
    check_eq!(points.len(), cams_from_world.len());
    check_eq!(points.len(), cameras.len());
    options.check()?;

    let mut point_data = Vec::with_capacity(points.len());
    let mut pose_data = Vec::with_capacity(points.len());
    for ((&point, cam_from_world), &camera) in points.iter().zip(cams_from_world).zip(cameras) {
        point_data.push(PointData {
            img_point: point,
            // Unit bearing in the camera frame. CamRayFromImg yields a valid ray for any
            // camera model, including omnidirectional (EQUIRECTANGULAR) back-hemisphere
            // observations that CamFromImg cannot represent. Fall back to a defined forward
            // bearing (+Z) if unprojection fails, so downstream normalize() in the DLT never
            // sees a zero vector (which would produce NaNs).
            cam_ray: camera
                .cam_ray_from_img(point)
                .unwrap_or(Vector3d::new(0.0, 0.0, 1.0)),
        });
        pose_data.push(PoseData {
            cam_from_world: cam_from_world.to_matrix(),
            proj_center: cam_from_world.tgt_origin_in_src(),
            camera,
        });
    }

    // Robustly estimate track using LORANSAC.
    let estimator = TriangulationEstimator::new(options.min_tri_angle, options.residual_type)?;
    let mut ransac: LoRansac<_, _, InlierSupportMeasurer, CombinationSampler> = LoRansac::new(
        &options.ransac_options,
        estimator,
        EstimateAsLocal(estimator),
    )?;
    let report = ransac.estimate(&point_data, &pose_data)?;
    if !report.success {
        return Ok(None);
    }
    Ok(Some(TriangulationResult {
        inlier_mask: report.inlier_mask,
        xyz: report.model,
    }))
}
