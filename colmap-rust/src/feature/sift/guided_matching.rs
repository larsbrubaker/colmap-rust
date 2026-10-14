//! The geometric half of `SiftCPUFeatureMatcher::MatchGuided` in `colmap/feature/sift.cc`:
//! `CamRaysWithJac` / `ComputeCamRaysWithJac`, `UseEssentialMatrixForGuidedMatching` and the
//! three guided filters (essential matrix with the tangent Sampson error on bearings,
//! fundamental matrix and homography on pixels). Port of colmap-sharp's
//! `Feature/SiftGuidedFilters.cs`. Neighbors: [`super::cpu_matcher`] (the caller),
//! [`crate::geometry::essential_matrix`] (the tangent Sampson error),
//! [`crate::scene::camera`] (`CamRayFromImgWithJac`). Tests:
//! `tests/feature/sift_guided_matching.rs` (`sift_test.cc`'s `MatchGuidedSiftFeaturesCPU`).
//!
//! The F and H filters are `f32` expressions evaluated in Eigen's order, left to right, for
//! 3 x 3 times 3-vector products (`(a0*b0 + a1*b1) + a2*b2`), without FMA. The filters are pure
//! functions of `(i1, i2)`.

use crate::check;
use crate::feature::types::FeatureKeypoint;
use crate::geometry::essential_matrix::compute_squared_tangent_sampson_error;
use crate::geometry::pose::CamRayWithJac;
use crate::linalg::{Matrix3d, Vector2d};
use crate::scene::camera::Camera;
use crate::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use crate::util::check::Result;

/// A guided filter: `(i1, i2)` true rejects the pair on geometric grounds.
pub(crate) type GuidedFilter<'a> = Box<dyn Fn(usize, usize) -> bool + 'a>;

/// Port of `UseEssentialMatrixForGuidedMatching`. Selects the epipolar model used to guide
/// matching. The essential matrix is preferred whenever the intrinsics are known, which
/// properly handles non-pinhole camera models where the fundamental matrix relationship does
/// not hold. The intrinsics are known either from priors (calibrated configs) or because the
/// two-view solver recovered them: such pairs are labeled UNCALIBRATED but carry the estimated
/// intrinsics in `camera1` / `camera2`. Either side alone suffices, to support solvers that
/// estimate only one side against an already calibrated view.
pub fn use_essential_matrix_for_guided_matching(geometry: &TwoViewGeometry) -> bool {
    if geometry.e.is_none() {
        return false;
    }
    match geometry.config {
        ConfigurationType::Calibrated | ConfigurationType::CalibratedRig => true,
        ConfigurationType::Uncalibrated => geometry.camera1.is_some() || geometry.camera2.is_some(),
        _ => false,
    }
}

/// The guided filter of `MatchGuided`; `Ok(None)` when the two-view geometry carries no usable
/// model, in which case COLMAP returns without matches. `Err` when a homography is to be used
/// with a spherical camera (COLMAP's `THROW_CHECK`).
pub(crate) fn create_guided_filter<'a>(
    max_error: f64,
    camera1: &'a Camera,
    keypoints1: &'a [FeatureKeypoint],
    camera2: &'a Camera,
    keypoints2: &'a [FeatureKeypoint],
    two_view_geometry: &'a TwoViewGeometry,
) -> Result<Option<GuidedFilter<'a>>> {
    let use_essential_matrix = use_essential_matrix_for_guided_matching(two_view_geometry);
    let use_fundamental_matrix = !use_essential_matrix
        && two_view_geometry.config == ConfigurationType::Uncalibrated
        && two_view_geometry.f.is_some();
    let use_homography = matches!(
        two_view_geometry.config,
        ConfigurationType::Planar
            | ConfigurationType::Panoramic
            | ConfigurationType::PlanarOrPanoramic
    ) && two_view_geometry.h.is_some();

    // Normalize with the intrinsics the solver estimated where available (its focal, not the
    // camera's stale default), else the given cameras.
    let effective_camera1 = two_view_geometry.camera1.as_ref().unwrap_or(camera1);
    let effective_camera2 = two_view_geometry.camera2.as_ref().unwrap_or(camera2);

    // A spherical camera has no meaningful image plane, so a homography over pixel coordinates
    // is not a valid model for it. No estimator produces one today; fail loudly rather than
    // silently dividing by a near-zero z.
    if use_homography {
        check!(!effective_camera1.is_spherical());
        check!(!effective_camera2.is_spherical());
    }

    // Both thresholds as COLMAP computes them: the E filter compares in double, F and H in
    // float.
    let max_residual_double = max_error * max_error;
    let max_residual = max_residual_double as f32;

    if use_essential_matrix {
        if let Some(e) = two_view_geometry.e {
            // The essential matrix path scores in pixels with the tangent Sampson error,
            // matching the two-view verification that produced E. Bearings are used rather
            // than normalized image plane coordinates so that the filter is defined for every
            // central camera model, including omnidirectional ones whose back hemisphere has
            // no image plane representation at all.
            let cam_rays1 = compute_cam_rays_with_jac(effective_camera1, keypoints1);
            let cam_rays2 = compute_cam_rays_with_jac(effective_camera2, keypoints2);
            return Ok(Some(Box::new(move |i1, i2| {
                match (&cam_rays1[i1], &cam_rays2[i2]) {
                    (Some(ray1), Some(ray2)) => {
                        compute_squared_tangent_sampson_error(ray1, ray2, &e) > max_residual_double
                    }
                    _ => true,
                }
            })));
        }
    }

    if use_fundamental_matrix {
        if let Some(f) = two_view_geometry.f {
            let f = to_float(&f);
            return Ok(Some(Box::new(move |i1, i2| {
                let k1 = &keypoints1[i1];
                let k2 = &keypoints2[i2];
                // epipolar_line1 = F * p1, epipolar_line2 = F^T * p2 with p = (x, y, 1).
                let l10 = f[0] * k1.x + f[1] * k1.y + f[2];
                let l11 = f[3] * k1.x + f[4] * k1.y + f[5];
                let l12 = f[6] * k1.x + f[7] * k1.y + f[8];
                let l20 = f[0] * k2.x + f[3] * k2.y + f[6];
                let l21 = f[1] * k2.x + f[4] * k2.y + f[7];
                let nom = k2.x * l10 + k2.y * l11 + l12;
                let denom_sq = l10 * l10 + l11 * l11 + l20 * l20 + l21 * l21;
                nom * nom > max_residual * denom_sq
            })));
        }
    }

    if use_homography {
        if let Some(h) = two_view_geometry.h {
            let h = to_float(&h);
            return Ok(Some(Box::new(move |i1, i2| {
                let k1 = &keypoints1[i1];
                let k2 = &keypoints2[i2];
                // ((H * p1).hnormalized() - p2).squaredNorm().
                let q0 = h[0] * k1.x + h[1] * k1.y + h[2];
                let q1 = h[3] * k1.x + h[4] * k1.y + h[5];
                let q2 = h[6] * k1.x + h[7] * k1.y + h[8];
                let dx = q0 / q2 - k2.x;
                let dy = q1 / q2 - k2.y;
                dx * dx + dy * dy > max_residual
            })));
        }
    }

    Ok(None)
}

/// Row-major `f32` copy of a 3 x 3 matrix (`Matrix3d::cast<float>()`).
fn to_float(m: &Matrix3d) -> [f32; 9] {
    let mut result = [0.0f32; 9];
    for r in 0..3 {
        for c in 0..3 {
            result[3 * r + c] = m[(r, c)] as f32;
        }
    }
    result
}

/// Port of `ComputeCamRaysWithJac`: unit bearing vectors and their pixel Jacobians for a set
/// of keypoints; `None` for a keypoint that cannot be unprojected (COLMAP's `valid` mask).
///
/// Keypoints that cannot be unprojected - back-hemisphere pixels of an omnidirectional camera
/// have no normalized image plane representation, and iterative undistortion can fail - must
/// be excluded from matching by the caller. Note that encoding invalidity as an extreme
/// coordinate does *not* work: the Sampson error is a ratio whose numerator and denominator
/// scale together, so a point pushed to infinity along a direction d converges to the finite
/// distance between its partner and the epipolar line of d, which admits rather than rejects
/// partners lying near that one line.
fn compute_cam_rays_with_jac(
    camera: &Camera,
    keypoints: &[FeatureKeypoint],
) -> Vec<Option<CamRayWithJac>> {
    keypoints
        .iter()
        .map(|k| camera.cam_ray_from_img_with_jac(Vector2d::new(f64::from(k.x), f64::from(k.y))))
        .collect()
}
