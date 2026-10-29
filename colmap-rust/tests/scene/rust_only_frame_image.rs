// Rust-only checks for `Frame` and `Image`, beyond the 1:1 ports in `frame.rs` and `image.rs`:
// the id-based ownership of the `scene` module header (methods reject a rig, frame or
// camera that is not the one the ids name; equality and printing with and without the frame)
// and the out-of-range point access of the `scene/image.rs` header.

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Quaterniond, Vector2d, Vector3d};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::frame::Frame;
use colmap_rust::scene::image::Image;
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{DataId, SensorId, SensorType};

fn rig(rig_id: u32) -> Rig {
    let mut rig = Rig::new();
    rig.set_rig_id(rig_id);
    rig.add_ref_sensor(SensorId::new(SensorType::Camera, 1))
        .unwrap();
    rig
}

/// Frame `frame_id` of rig 1 with image `frame_id` of camera 1, posed at `rig_from_world`.
fn frame_with_image(frame_id: u32, rig_from_world: Option<Rigid3d>) -> (Frame, Image) {
    let mut frame = Frame::new();
    frame.set_frame_id(frame_id);
    frame.set_rig_id(1).unwrap();
    frame.set_maybe_rig_from_world(rig_from_world);
    let mut image = Image::new();
    image.set_image_id(frame_id);
    image.set_camera_id(1).unwrap();
    image.set_frame_id(frame_id);
    frame.add_data_id(image.data_id()).unwrap();
    image.verify_frame(&frame).unwrap();
    (frame, image)
}

#[test]
fn rust_only_frame_rejects_a_rig_that_is_not_its_own() {
    let rig1 = rig(1);
    let rig2 = rig(2);
    let sensor = SensorId::new(SensorType::Camera, 1);

    // A frame without a rig id has no rig to compose with (COLMAP: null rig_ptr_).
    let mut frame = Frame::new();
    frame.set_rig_from_world(Rigid3d::default());
    assert!(frame.sensor_from_world(&rig1, sensor).is_err());
    assert!(frame
        .set_cam_from_world(&rig1, 1, Rigid3d::default())
        .is_err());

    frame.set_rig_id(1).unwrap();
    assert!(frame.sensor_from_world(&rig1, sensor).is_ok());
    assert!(frame.sensor_from_world(&rig2, sensor).is_err());
    assert!(frame
        .set_cam_from_world(&rig2, 1, Rigid3d::default())
        .is_err());
    assert!(frame
        .add_data_id_in_rig(&rig2, DataId::new(sensor, 1))
        .is_err());
    assert!(frame.verify_rig(&rig2).is_err());

    // Rebinding is setting the rig id first, as COLMAP's SetRigPtr on a bound frame does.
    frame.set_rig_id(2).unwrap();
    assert!(frame.verify_rig(&rig2).is_ok());
    assert!(frame.sensor_from_world(&rig2, sensor).is_ok());
}

#[test]
fn rust_only_frame_verify_rig_rejects_invalid_sensor_data() {
    let rig = rig(1);
    let mut frame = Frame::new();
    frame.set_rig_id(1).unwrap();
    frame
        .add_data_id(DataId::new(SensorId::new(SensorType::Invalid, 0), 1))
        .unwrap();
    let error = frame.verify_rig(&rig).unwrap_err();
    assert!(
        error.message().ends_with("Invalid sensor type: INVALID"),
        "{}",
        error.message()
    );
}

#[test]
fn rust_only_frame_set_rig_id_rejects_invalid_id() {
    let mut frame = Frame::new();
    assert!(frame.set_rig_id(u32::MAX).is_err());
}

#[test]
fn rust_only_frame_rig_from_world_mut_edits_the_pose() {
    let mut frame = Frame::new();
    assert!(frame.rig_from_world_mut().is_err());
    frame.set_rig_from_world(Rigid3d::default());
    frame.rig_from_world_mut().unwrap().translation = Vector3d::new(1.0, 2.0, 3.0);
    assert_eq!(
        frame.rig_from_world().unwrap().translation,
        Vector3d::new(1.0, 2.0, 3.0)
    );
    *frame.maybe_rig_from_world_mut() = None;
    assert!(!frame.has_pose());
}

#[test]
fn rust_only_image_rejects_a_frame_rig_or_camera_that_is_not_its_own() {
    let rig1 = rig(1);
    let rig2 = rig(2);
    let (frame1, image1) = frame_with_image(1, Some(Rigid3d::default()));
    let (frame2, _) = frame_with_image(2, Some(Rigid3d::default()));

    assert!(image1.cam_from_world(&frame1, &rig1).is_ok());
    assert!(image1.cam_from_world(&frame2, &rig1).is_err());
    assert!(image1.cam_from_world(&frame1, &rig2).is_err());
    assert!(image1.is_ref_in_frame(&frame1, &rig1).unwrap());
    assert!(image1.is_ref_in_frame(&frame2, &rig1).is_err());
    assert!(image1.is_ref_in_frame(&frame1, &rig2).is_err());
    assert!(image1.verify_frame(&frame2).is_err());
    // Another frame has no pose for this image.
    assert!(image1.has_pose(Some(&frame1)));
    assert!(!image1.has_pose(Some(&frame2)));

    let camera = Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1.0, 1, 1).unwrap();
    let other_camera =
        Camera::create_from_model_id(2, CameraModelId::SimplePinhole, 1.0, 1, 1).unwrap();
    let point = Vector3d::new(2.0, 0.0, 1.0);
    assert_eq!(
        image1
            .project_point(&frame1, &rig1, &camera, point)
            .unwrap(),
        Some(Vector2d::new(2.5, 0.5))
    );
    assert!(image1
        .project_point(&frame1, &rig1, &other_camera, point)
        .is_err());

    // An image without a frame id has no frame to compose with.
    let mut unbound = Image::new();
    unbound.set_camera_id(1).unwrap();
    assert!(unbound.cam_from_world(&frame1, &rig1).is_err());
}

#[test]
fn rust_only_image_is_ref_in_frame_for_a_non_reference_camera() {
    let mut rig = rig(1);
    rig.add_sensor(
        SensorId::new(SensorType::Camera, 2),
        Some(Rigid3d::new(
            Quaterniond::identity(),
            Vector3d::new(1.0, 0.0, 0.0),
        )),
    )
    .unwrap();
    let mut frame = Frame::new();
    frame.set_frame_id(1);
    frame.set_rig_id(1).unwrap();
    frame.set_rig_from_world(Rigid3d::default());
    let mut image = Image::new();
    image.set_image_id(7);
    image.set_camera_id(2).unwrap();
    image.set_frame_id(1);
    frame.add_data_id_in_rig(&rig, image.data_id()).unwrap();
    image.verify_frame(&frame).unwrap();
    assert!(!image.is_ref_in_frame(&frame, &rig).unwrap());
    // cam_from_world = sensor_from_rig * rig_from_world.
    assert_eq!(
        image.cam_from_world(&frame, &rig).unwrap().translation,
        Vector3d::new(1.0, 0.0, 0.0)
    );
    assert_eq!(
        image.projection_center(&frame, &rig).unwrap(),
        Vector3d::new(-1.0, 0.0, 0.0)
    );
}

#[test]
fn rust_only_image_equality_and_print_with_frames() {
    let (frame_a, image) = frame_with_image(1, Some(Rigid3d::default()));
    let moved = Rigid3d::new(Quaterniond::identity(), Vector3d::new(0.0, 0.0, 1.0));
    let (frame_b, other) = frame_with_image(1, Some(moved));
    let (frame_unposed, unposed) = frame_with_image(1, None);

    // Without the frames, the images' own data is equal.
    assert_eq!(image, other);
    assert_eq!(image, unposed);
    // COLMAP's full operator== also compares the poses.
    assert!(image.eq_with_frames(Some(&frame_a), &other, Some(&frame_a)));
    assert!(!image.eq_with_frames(Some(&frame_a), &other, Some(&frame_b)));
    assert!(!image.eq_with_frames(Some(&frame_a), &unposed, Some(&frame_unposed)));
    assert!(unposed.eq_with_frames(Some(&frame_unposed), &unposed, None));

    assert_eq!(
        image.display(Some(&frame_a)).to_string(),
        "Image(image_id=1, camera_id=1, frame_id=1, name=\"\", has_pose=1, triangulated=0/0)"
    );
    assert_eq!(
        Image::new().display(None).to_string(),
        "Image(image_id=Invalid, camera_id=Invalid, frame_id=Invalid, name=\"\", \
         has_pose=0, triangulated=0/0)"
    );
}

#[test]
fn rust_only_image_set_points2d_requires_empty_image() {
    let mut image = Image::new();
    image.set_points2d_from_xy(&[Vector2d::zeros(); 2]);
    assert!(image.set_points2d(Vec::new()).is_err());
    assert!(image.set_point3d_for_point2d(0, u64::MAX).is_err());
}

#[test]
#[should_panic]
fn rust_only_image_point2d_out_of_range_panics() {
    let mut image = Image::new();
    image.set_points2d_from_xy(&[Vector2d::zeros(); 2]);
    let _ = image.point2d(2);
}

#[test]
#[should_panic]
fn rust_only_image_set_point3d_out_of_range_panics() {
    let mut image = Image::new();
    image.set_points2d_from_xy(&[Vector2d::zeros(); 2]);
    let _ = image.set_point3d_for_point2d(2, 1);
}
