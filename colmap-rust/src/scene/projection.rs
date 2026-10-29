//! Port of `colmap/scene/projection.h` / `projection.cc`: reprojection errors (squared pixel
//! and angular) of a 3D point against an observation, and the positive-depth (cheirality)
//! test of a projection matrix. Neighbor: [`super::camera::Camera`], whose projection and
//! unprojection they use. Tests: `tests/scene/projection.rs` (`projection_test.cc` 1:1).
//!
//! Tier A: COLMAP's operation order; `acos` goes through [`crate::math::fns`].
//! `EIGEN_PI` is [`std::f64::consts::PI`].
//!
//! Translation notes:
//! - COLMAP overloads each function on the pose type (`Rigid3d` or a 3x4 matrix
//!   `cam_from_world`). Both overloads exist here through the [`CamFromWorld`] trait, which
//!   keeps each overload's own arithmetic (`Rigid3d * point` vs. `matrix * point.homogeneous()`).
//! - The angular-error overload taking a normalized camera ray instead of an image point
//!   is [`calculate_angular_reprojection_error_from_ray`].
//! - These run per point, so an unknown camera model panics (the per-point side of the
//!   split in the `sensor/models/dispatch.rs` header).

use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{Matrix3x4d, Vector2d, Vector3d};
use crate::math::fns;
use crate::scene::camera::Camera;

/// A `cam_from_world` pose as COLMAP's projection functions accept it: a [`Rigid3d`] or a
/// 3x4 projection matrix.
pub trait CamFromWorld {
    /// `cam_from_world * point3D` for a `Rigid3d`, `cam_from_world * point3D.homogeneous()`
    /// for a matrix.
    fn transform_point(&self, point3d: Vector3d) -> Vector3d;
}

impl CamFromWorld for Rigid3d {
    fn transform_point(&self, point3d: Vector3d) -> Vector3d {
        *self * point3d
    }
}

impl CamFromWorld for Matrix3x4d {
    fn transform_point(&self, point3d: Vector3d) -> Vector3d {
        *self * point3d.homogeneous()
    }
}

/// Pixel reprojection error is ill-defined for equirectangular (e.g. EQUIRECTANGULAR)
/// cameras: the azimuth is discontinuous at the ±π seam and its pixel scale diverges towards
/// the poles (azimuth pixels grow as 1/cos(elevation)). Instead measure the angular error
/// between the observed bearing and the 3D point and convert it to an equivalent pixel error
/// at the equator, consistent with the model's pixel<->angle scale used by
/// CamFromImgThreshold. This makes the (squared) error continuous across the seam and
/// uniform over the sphere.
fn squared_spherical_reprojection_error(angular_error: f64, camera: &Camera) -> f64 {
    let pixels_per_radian = camera.width as f64 / (2.0 * std::f64::consts::PI);
    let pixel_error = angular_error * pixels_per_radian;
    pixel_error * pixel_error
}

/// Port of `CalculateSquaredReprojectionError`: the squared Euclidean distance between the
/// observation in the image and the projection of the 3D point into the image. If the 3D
/// point is behind the camera, this returns `f64::MAX` (`DBL_MAX`). Spherical cameras use
/// the angular error converted to equator pixels (see above).
pub fn calculate_squared_reprojection_error<P: CamFromWorld>(
    point2d: Vector2d,
    point3d: Vector3d,
    cam_from_world: &P,
    camera: &Camera,
) -> f64 {
    if camera.is_spherical() {
        return squared_spherical_reprojection_error(
            calculate_angular_reprojection_error(point2d, point3d, cam_from_world, camera),
            camera,
        );
    }
    match camera.img_from_cam(cam_from_world.transform_point(point3d), true) {
        Some(proj_point2d) => (proj_point2d - point2d).squared_norm(),
        None => f64::MAX,
    }
}

/// Port of `CalculateAngularReprojectionError(point2D, point3D, cam_from_world, camera)`:
/// the angle between the observed viewing ray and the actual viewing ray from the camera
/// center to the 3D point; π when the observation cannot be unprojected.
pub fn calculate_angular_reprojection_error<P: CamFromWorld>(
    point2d: Vector2d,
    point3d: Vector3d,
    cam_from_world: &P,
    camera: &Camera,
) -> f64 {
    // Use the 3D bearing (full sphere) rather than the 2D CamFromImg, which cannot represent
    // back-hemisphere rays of omnidirectional (e.g. EQUIRECTANGULAR) cameras. Identical to
    // the legacy path for perspective cameras.
    match camera.cam_ray_from_img(point2d) {
        Some(cam_ray) => {
            calculate_angular_reprojection_error_from_ray(cam_ray, point3d, cam_from_world)
        }
        None => std::f64::consts::PI,
    }
}

/// Port of `CalculateAngularReprojectionError(cam_ray, point3D, cam_from_world)`: the angle
/// between the normalized observed camera ray and the ray to the 3D point.
pub fn calculate_angular_reprojection_error_from_ray<P: CamFromWorld>(
    cam_ray: Vector3d,
    point3d: Vector3d,
    cam_from_world: &P,
) -> f64 {
    let point3d_in_cam = cam_from_world.transform_point(point3d);
    let cos_angle = cam_ray.dot(point3d_in_cam.normalized());
    // std::clamp and f64::clamp agree, NaN included (both return it).
    fns::acos(cos_angle.clamp(-1.0, 1.0))
}

/// Port of `HasPointPositiveDepth`: whether the 3D point passes the cheirality constraint,
/// i.e. lies in front of the camera and not in the image plane (depth at least
/// `f64::EPSILON`).
pub fn has_point_positive_depth(cam_from_world: &Matrix3x4d, point3d: Vector3d) -> bool {
    // `row(2).dot(point3D.homogeneous())`. The row of a column-major 3x4 matrix is strided,
    // so Eigen cannot use the packet pairing that `Vector4d::dot` copies; the sum is the
    // module's plain left-to-right order (the fixed-size divergence in `linalg/mod.rs`).
    let m = cam_from_world;
    m[(2, 0)] * point3d.x + m[(2, 1)] * point3d.y + m[(2, 2)] * point3d.z + m[(2, 3)]
        >= f64::EPSILON
}
