//! Port of COLMAP's `src/colmap/estimators/solvers/poselib_utils.h` and `.cc`: conversions
//! between COLMAP's [`Camera`] and [`Rigid3d`] and PoseLib's camera ([`PoseLibCamera`]) and
//! pose ([`CameraPose`]). Used by the solvers that wrap PoseLib. Port of colmap-sharp's
//! `Estimators/Solvers/PoseLibUtils.cs`. Tests: `tests/estimators/poselib_utils.rs`
//! (`poselib_utils_test.cc`).
//!
//! Tier A: plain copies; the quaternion only changes order.

use super::poselib::poselib_camera::PoseLibCamera;
use super::poselib::CameraPose;
use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{Quaterniond, Vector4d};
use crate::scene::camera::Camera;
use crate::sensor::models::camera_model_name_to_id;

/// Convert COLMAP Camera to PoseLib Camera. COLMAP's `size_t` width and height narrow to
/// PoseLib's `int`, as the C++ constructor call does; `as` wraps the same way for sizes past
/// `i32::MAX`, which no image has.
pub fn convert_camera_to_poselib_camera(camera: &Camera) -> PoseLibCamera {
    PoseLibCamera::from_name(
        camera.model_name(),
        &camera.params,
        camera.width as i32,
        camera.height as i32,
    )
}

/// Convert PoseLib Camera to COLMAP Camera. PoseLib's `int` size converts to COLMAP's
/// `size_t`; a negative size (never produced by PoseLib) would wrap as in C++.
pub fn convert_poselib_camera_to_camera(camera: &PoseLibCamera) -> Camera {
    Camera {
        model_id: camera_model_name_to_id(camera.model_name()),
        width: camera.width as usize,
        height: camera.height as usize,
        params: camera.params.clone(),
        ..Camera::default()
    }
}

/// Convert COLMAP Rigid3d to PoseLib CameraPose.
pub fn convert_rigid3d_to_poselib_pose(rigid: &Rigid3d) -> CameraPose {
    // PoseLib uses (w, x, y, z) ordering for quaternion
    let r = rigid.rotation;
    CameraPose::new(Vector4d::new(r.w, r.x, r.y, r.z), rigid.translation)
}

/// Convert PoseLib CameraPose to COLMAP Rigid3d.
pub fn convert_poselib_pose_to_rigid3d(pose: &CameraPose) -> Rigid3d {
    // PoseLib quaternion is (w, x, y, z), Eigen::Quaterniond constructor is also
    // (w, x, y, z)
    Rigid3d::new(
        Quaterniond::new(pose.q.x, pose.q.y, pose.q.z, pose.q.w),
        pose.t,
    )
}
