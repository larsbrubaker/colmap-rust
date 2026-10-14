// Port of colmap/scene/image_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::scene::image::Image. EXPECT_ANY_THROW becomes `is_err()`.
//
// COLMAP's image holds `Camera*` and `Frame*`; the Rust image holds the ids and takes the
// camera, frame and rig as parameters (docs/CPP_DIVERGENCES.md entry 170). The pointer checks
// become checks of the id-based API: `SetCameraPtr` / `SetFramePtr` succeeding is
// `verify_camera` / `verify_frame` being `Ok`, and two images sharing a frame pointer are two
// images with the same frame id resolving to the same frame.

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Quaterniond, Vector2d, Vector3d};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::frame::Frame;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::point2d::Point2D;
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{DataId, SensorId, SensorType, INVALID_CAMERA_ID, INVALID_IMAGE_ID};

/// The rig with id 1 and camera 1 as its reference sensor that most tests build.
fn rig1() -> Rig {
    let mut rig = Rig::new();
    rig.set_rig_id(1);
    rig.add_ref_sensor(SensorId::new(SensorType::Camera, 1))
        .unwrap();
    rig
}

/// Frame 1 of rig 1, optionally posed at the identity.
fn frame1(rig: &Rig, posed: bool) -> Frame {
    let mut frame = Frame::new();
    frame.set_frame_id(1);
    frame.set_rig_id(1).unwrap();
    if posed {
        frame.set_rig_from_world(Rigid3d::default());
    }
    frame.verify_rig(rig).unwrap();
    frame
}

/// Image 1 of camera 1, added to `frame` and bound to it.
fn image1_in(frame: &mut Frame, name: Option<&str>) -> Image {
    let mut image = Image::new();
    image.set_image_id(1);
    image.set_camera_id(1).unwrap();
    if let Some(name) = name {
        image.set_name(name);
    }
    frame.add_data_id(image.data_id()).unwrap();
    image.set_frame_id(1);
    image.verify_frame(frame).unwrap();
    image
}

#[test]
fn image_default() {
    let image = Image::new();
    assert_eq!(image.image_id(), INVALID_IMAGE_ID);
    assert_eq!(image.name(), "");
    assert_eq!(image.camera_id(), INVALID_CAMERA_ID);
    assert_eq!(
        image.data_id(),
        DataId::new(
            SensorId::new(SensorType::Camera, INVALID_CAMERA_ID),
            INVALID_IMAGE_ID
        )
    );
    assert!(!image.has_camera_id());
    // HasCameraPtr() is false: no camera verifies against the image.
    assert!(image.verify_camera(&Camera::default()).is_err());
    assert!(!image.has_pose(None));
    assert_eq!(image.num_points2d(), 0);
    assert_eq!(image.num_points3d(), 0);
    assert_eq!(image.points2d().len(), 0);
}

#[test]
fn image_equals() {
    let mut image = Image::new();
    let mut other = image.clone();
    assert_eq!(image, other);
    image.set_name("test");
    assert_ne!(image, other);
    other.set_name("test");
    assert_eq!(image, other);
}

#[test]
fn image_print() {
    let mut image = Image::new();
    image.set_image_id(1);
    image.set_camera_id(2).unwrap();
    image.set_frame_id(3);
    image.set_name("test");
    assert_eq!(
        image.display(None).to_string(),
        "Image(image_id=1, camera_id=2, frame_id=3, name=\"test\", has_pose=0, \
         triangulated=0/0)"
    );
}

#[test]
fn image_image_id() {
    let mut image = Image::new();
    assert_eq!(image.image_id(), INVALID_IMAGE_ID);
    image.set_image_id(1);
    assert_eq!(image.image_id(), 1);
}

#[test]
fn image_name() {
    let mut image = Image::new();
    assert_eq!(image.name(), "");
    image.set_name("test1");
    assert_eq!(image.name(), "test1");
    *image.name_mut() = "test2".to_string();
    assert_eq!(image.name(), "test2");
}

#[test]
fn image_camera_id() {
    let mut image = Image::new();
    assert_eq!(image.camera_id(), INVALID_CAMERA_ID);
    image.set_camera_id(1).unwrap();
    assert_eq!(image.camera_id(), 1);
}

#[test]
fn image_data_id() {
    let mut image = Image::new();
    image.set_image_id(1);
    image.set_camera_id(2).unwrap();
    assert_eq!(
        image.data_id(),
        DataId::new(SensorId::new(SensorType::Camera, 2), 1)
    );
}

#[test]
fn image_camera_ptr() {
    let mut image = Image::new();
    // CameraPtr() throws when unset: an invalid camera never verifies.
    assert!(image.verify_camera(&Camera::default()).is_err());
    let camera = Camera {
        camera_id: 1,
        ..Camera::default()
    };
    assert!(image.verify_camera(&camera).is_err());
    image.set_camera_id(2).unwrap();
    assert!(image.verify_camera(&camera).is_err());
    image.set_camera_id(1).unwrap();
    assert!(image.verify_camera(&camera).is_ok());
    // CameraPtr() == &camera: the image resolves to the camera with its id.
    assert_eq!(image.camera_id(), camera.camera_id);
}

#[test]
fn image_frame_ptr() {
    let rig = rig1();
    let mut frame = frame1(&rig, false);
    let mut image = Image::new();
    image.set_image_id(1);
    image.set_camera_id(1).unwrap();
    image.set_frame_id(1);
    assert!(image.verify_frame(&frame).is_err());
    frame.add_data_id(image.data_id()).unwrap();
    image.verify_frame(&frame).unwrap();
}

#[test]
fn image_set_reset_pose() {
    let rig = rig1();
    let mut frame = frame1(&rig, false);
    let image = image1_in(&mut frame, None);
    assert!(!image.has_pose(Some(&frame)));
    assert!(image.cam_from_world(&frame, &rig).is_err());
    frame.set_rig_from_world(Rigid3d::default());
    assert!(image.has_pose(Some(&frame)));
    let cam_from_world = image.cam_from_world(&frame, &rig).unwrap();
    assert_eq!(
        cam_from_world.rotation.coeffs(),
        Quaterniond::identity().coeffs()
    );
    assert_eq!(cam_from_world.translation, Vector3d::zeros());
    frame.reset_pose();
    assert!(!image.has_pose(Some(&frame)));
    assert!(image.cam_from_world(&frame, &rig).is_err());
}

#[test]
fn image_construct_copy() {
    let rig = rig1();
    let mut frame = frame1(&rig, true);
    let image = image1_in(&mut frame, Some("test"));
    let image_copy = image.clone();
    assert!(image.eq_with_frames(Some(&frame), &image_copy, Some(&frame)));
    assert_eq!(
        Rigid3d::default(),
        image_copy.cam_from_world(&frame, &rig).unwrap()
    );
    // Both images resolve to the same frame, so resetting its pose unposes both.
    assert_eq!(image_copy.frame_id(), image.frame_id());
    frame.reset_pose();
    assert!(!image.has_pose(Some(&frame)));
    assert!(!image_copy.has_pose(Some(&frame)));
}

#[test]
fn image_assign_copy() {
    let rig = rig1();
    let mut frame = frame1(&rig, true);
    let image = image1_in(&mut frame, Some("test"));
    let mut image_copy = Image::new();
    image_copy.clone_from(&image);
    assert!(image.eq_with_frames(Some(&frame), &image_copy, Some(&frame)));
    assert_eq!(
        image_copy.cam_from_world(&frame, &rig).unwrap(),
        Rigid3d::default()
    );
    assert_eq!(image_copy.frame_id(), image.frame_id());
    frame.reset_pose();
    assert!(!image.has_pose(Some(&frame)));
    assert!(!image_copy.has_pose(Some(&frame)));
}

#[test]
fn image_num_points2d() {
    let mut image = Image::new();
    assert_eq!(image.num_points2d(), 0);
    image.set_points2d_from_xy(&[Vector2d::zeros(); 10]);
    assert_eq!(image.num_points2d(), 10);
}

#[test]
fn image_num_points3d() {
    let mut image = Image::new();
    image.set_points2d_from_xy(&[Vector2d::zeros(); 10]);
    assert_eq!(image.num_points3d(), 0);
    image.set_point3d_for_point2d(0, 0).unwrap();
    assert_eq!(image.num_points3d(), 1);
    image.set_point3d_for_point2d(0, 1).unwrap();
    image.set_point3d_for_point2d(1, 2).unwrap();
    assert_eq!(image.num_points3d(), 2);
}

#[test]
fn image_points2d() {
    let mut image = Image::new();
    assert_eq!(image.points2d().len(), 0);
    let mut points2d = vec![Vector2d::zeros(); 10];
    points2d[0] = Vector2d::new(1.0, 2.0);
    image.set_points2d_from_xy(&points2d);
    assert_eq!(image.points2d().len(), 10);
    assert_eq!(image.point2d(0).xy.x, 1.0);
    assert_eq!(image.point2d(0).xy.y, 2.0);
    assert_eq!(image.num_points3d(), 0);
}

#[test]
fn image_points2d_with3d() {
    let mut image = Image::new();
    assert_eq!(image.points2d().len(), 0);
    let mut points2d = vec![Point2D::default(); 10];
    points2d[0].xy = Vector2d::new(1.0, 2.0);
    points2d[0].point3d_id = 1;
    image.set_points2d(points2d).unwrap();
    assert_eq!(image.points2d().len(), 10);
    assert_eq!(image.point2d(0).xy.x, 1.0);
    assert_eq!(image.point2d(0).xy.y, 2.0);
    assert_eq!(image.num_points3d(), 1);
}

#[test]
fn image_points3d() {
    let mut image = Image::new();
    image.set_points2d_from_xy(&[Vector2d::zeros(); 2]);
    assert!(!image.point2d(0).has_point3d());
    assert!(!image.point2d(1).has_point3d());
    assert_eq!(image.num_points3d(), 0);
    image.set_point3d_for_point2d(0, 0).unwrap();
    assert!(image.point2d(0).has_point3d());
    assert!(!image.point2d(1).has_point3d());
    assert_eq!(image.num_points3d(), 1);
    assert!(image.has_point3d(0));
    image.set_point3d_for_point2d(0, 1).unwrap();
    assert!(image.point2d(0).has_point3d());
    assert!(!image.point2d(1).has_point3d());
    assert_eq!(image.num_points3d(), 1);
    assert!(!image.has_point3d(0));
    assert!(image.has_point3d(1));
    image.set_point3d_for_point2d(1, 0).unwrap();
    assert!(image.point2d(0).has_point3d());
    assert!(image.point2d(1).has_point3d());
    assert_eq!(image.num_points3d(), 2);
    assert!(image.has_point3d(0));
    assert!(image.has_point3d(1));
    image.reset_point3d_for_point2d(0);
    assert!(!image.point2d(0).has_point3d());
    assert!(image.point2d(1).has_point3d());
    assert_eq!(image.num_points3d(), 1);
    assert!(image.has_point3d(0));
    assert!(!image.has_point3d(1));
    image.reset_point3d_for_point2d(1);
    assert!(!image.point2d(0).has_point3d());
    assert!(!image.point2d(1).has_point3d());
    assert_eq!(image.num_points3d(), 0);
    assert!(!image.has_point3d(0));
    assert!(!image.has_point3d(1));
    image.reset_point3d_for_point2d(0);
    assert!(!image.point2d(0).has_point3d());
    assert!(!image.point2d(1).has_point3d());
    assert_eq!(image.num_points3d(), 0);
    assert!(!image.has_point3d(0));
    assert!(!image.has_point3d(1));
}

#[test]
fn image_projection_center() {
    let rig = rig1();
    let mut frame = frame1(&rig, true);
    let image = image1_in(&mut frame, None);
    assert_eq!(
        image.projection_center(&frame, &rig).unwrap(),
        Vector3d::zeros()
    );
}

#[test]
fn image_viewing_direction() {
    let rig = rig1();
    let mut frame = frame1(&rig, true);
    let image = image1_in(&mut frame, None);
    assert_eq!(
        image.viewing_direction(&frame, &rig).unwrap(),
        Vector3d::new(0.0, 0.0, 1.0)
    );
}

#[test]
fn image_project_point() {
    let rig = rig1();
    let mut frame = frame1(&rig, true);
    let mut image = image1_in(&mut frame, None);
    let camera = Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1.0, 1, 1).unwrap();
    image.set_camera_id(camera.camera_id).unwrap();
    image.verify_camera(&camera).unwrap();
    let result = image
        .project_point(&frame, &rig, &camera, Vector3d::new(2.0, 0.0, 1.0))
        .unwrap();
    assert!(result.is_some());
    assert_eq!(result.unwrap(), Vector2d::new(2.5, 0.5));
    assert!(image
        .project_point(&frame, &rig, &camera, Vector3d::new(2.0, 0.0, 0.0))
        .unwrap()
        .is_none());
    assert!(image
        .project_point(&frame, &rig, &camera, Vector3d::new(2.0, 0.0, -1.0))
        .unwrap()
        .is_none());
}
