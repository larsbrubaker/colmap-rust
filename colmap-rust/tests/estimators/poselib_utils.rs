// Port of COLMAP's `src/colmap/estimators/solvers/poselib_utils_test.cc`: round trips of
// colmap_rust::estimators::solvers::poselib_utils between COLMAP's camera and pose types and
// PoseLib's. Tier A for the camera (exact copies); the pose within COLMAP's `Rigid3dNear`
// tolerances.

use colmap_rust::estimators::solvers::poselib_utils::{
    convert_camera_to_poselib_camera, convert_poselib_camera_to_camera,
    convert_poselib_pose_to_rigid3d, convert_rigid3d_to_poselib_pose,
};
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Quaterniond, Vector3d};
use colmap_rust::scene::camera::Camera;
use colmap_rust::sensor::models::CameraModelId;

use crate::transform_matchers::rigid3d_near;

#[test]
fn poselib_utils_camera_round_trip() {
    let camera = Camera {
        model_id: CameraModelId::SimpleRadial,
        width: 1920,
        height: 1080,
        params: vec![1000.0, 960.0, 540.0, 0.1],
        ..Camera::default()
    };

    let poselib_camera = convert_camera_to_poselib_camera(&camera);
    let camera_back = convert_poselib_camera_to_camera(&poselib_camera);

    assert_eq!(camera.model_id, camera_back.model_id);
    assert_eq!(camera.width, camera_back.width);
    assert_eq!(camera.height, camera_back.height);
    assert_eq!(camera.params, camera_back.params);
}

#[test]
fn poselib_utils_rigid3d_round_trip() {
    let rotation = Quaterniond::new(0.5, 0.5, 0.5, 0.5).normalized();
    let translation = Vector3d::new(1.0, 2.0, 3.0);
    let rigid = Rigid3d::new(rotation, translation);

    let poselib_pose = convert_rigid3d_to_poselib_pose(&rigid);
    let rigid_back = convert_poselib_pose_to_rigid3d(&poselib_pose);

    assert!(rigid3d_near(&rigid_back, &rigid, 1e-10, 1e-10));
}
