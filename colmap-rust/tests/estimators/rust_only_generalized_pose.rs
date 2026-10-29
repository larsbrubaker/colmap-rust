// Rust-only tests of colmap_rust::estimators::generalized_pose's guards where COLMAP's
// generalized_pose.cc has undefined behavior: empty correspondence lists, and a camera index
// list shorter than the points (the divergence in the generalized_pose.rs header). The COLMAP
// cases are in `generalized_pose.rs`.

use colmap_rust::estimators::generalized_pose::{
    estimate_generalized_absolute_pose, estimate_generalized_relative_pose,
    estimate_structure_less_absolute_pose, StructureLessAbsolutePoseEstimationOptions,
};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Quaterniond, Vector2d, Vector3d};
use colmap_rust::optim::RansacOptions;
use colmap_rust::scene::camera::Camera;
use colmap_rust::sensor::models::CameraModelId;

fn camera() -> Camera {
    Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 100.0, 100, 100).unwrap()
}

#[test]
fn rust_only_generalized_pose_empty_correspondences_find_no_pose() {
    let cameras = vec![camera(), camera()];
    let cams_from_rig = vec![
        Rigid3d::default(),
        Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0)),
    ];
    let ransac_options = RansacOptions {
        max_error: 1.0,
        ..RansacOptions::default()
    };

    let absolute = estimate_generalized_absolute_pose(
        &ransac_options,
        &[],
        &[],
        &[],
        &cams_from_rig,
        &cameras,
    );
    assert_eq!(absolute.unwrap(), None);

    let relative = estimate_generalized_relative_pose(
        &ransac_options,
        &[],
        &[],
        &[],
        &[],
        &cams_from_rig,
        &cameras,
    );
    assert_eq!(relative.unwrap(), None);

    let structure_less = estimate_structure_less_absolute_pose(
        &StructureLessAbsolutePoseEstimationOptions::default(),
        &[],
        &[],
        &[],
        &cams_from_rig,
        &cameras,
        &camera(),
    );
    assert_eq!(structure_less.unwrap(), None);
}

#[test]
fn rust_only_generalized_relative_pose_short_camera_idxs_is_an_error() {
    let cameras = vec![camera()];
    let cams_from_rig = vec![Rigid3d::default()];
    let points = vec![Vector2d::new(50.0, 50.0); 2];
    let ransac_options = RansacOptions {
        max_error: 1.0,
        ..RansacOptions::default()
    };
    let result = estimate_generalized_relative_pose(
        &ransac_options,
        &points,
        &points,
        &[0],
        &[0, 0],
        &cams_from_rig,
        &cameras,
    );
    let error = result.unwrap_err().to_string();
    assert!(
        error.contains("Check failed: camera_idxs1.len()"),
        "{error}"
    );
}
