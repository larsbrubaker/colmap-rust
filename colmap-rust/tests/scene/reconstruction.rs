// Port of colmap/scene/reconstruction_test.cc, part 1, 1:1 (same test names, values and
// checks) for the ported methods of colmap_rust::scene::reconstruction::Reconstruction: the
// add, delete and register tests, and the test's helpers. The query and statistics tests are
// in reconstruction_queries.rs, the bounds and transform tests in reconstruction_transform.rs,
// and the tests that synthesize their scene in reconstruction_synthetic.rs. Tests of methods
// not yet ported (they need the database, PLY, image colors or image directories) land with
// those methods. COLMAP's `ExpectValidPtrs` (every frame's rig
// pointer and every image's camera and frame pointers point into the reconstruction) is
// `expect_valid_ptrs`: the ids it guards resolve and pass the `verify_*` checks
// (docs/CPP_DIVERGENCES.md entry 170). `EXPECT_ANY_THROW` is `is_err()`; a `try` / `catch`
// with `HasSubstr` checks the `Err` message, the Rust expression text where COLMAP's is C++
// (util/check.rs).

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Vector2d, Vector3d, Vector3ub};
use colmap_rust::math::random_eigen::random_eigen_vector3d;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::frame::Frame;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::point3d::Point3D;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::track::{Track, TrackElement};
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{DataId, ImageId};

/// Port of the test's `ExpectValidPtrs` (see the header).
pub(crate) fn expect_valid_ptrs(reconstruction: &Reconstruction) {
    for frame in reconstruction.frames().values() {
        let rig = reconstruction.rig(frame.rig_id()).unwrap();
        frame.verify_rig(rig).unwrap();
    }
    for image in reconstruction.images().values() {
        let camera = reconstruction.camera(image.camera_id()).unwrap();
        image.verify_camera(camera).unwrap();
        let frame = reconstruction.frame(image.frame_id()).unwrap();
        image.verify_frame(frame).unwrap();
    }
}

/// Port of the test's `GenerateReconstruction`: one pinhole camera in a trivial rig, and
/// `num_images` posed, registered frames of one image with 10 points each.
pub(crate) fn generate_reconstruction(num_images: ImageId, reconstruction: &mut Reconstruction) {
    const NUM_POINTS2D: usize = 10;

    let camera = Camera::create_from_model_id(1, CameraModelId::Pinhole, 1.0, 1, 1).unwrap();
    reconstruction.add_camera(camera.clone()).unwrap();
    let mut rig = Rig::new();
    rig.set_rig_id(1);
    rig.add_ref_sensor(camera.sensor_id()).unwrap();
    reconstruction.add_rig(rig.clone()).unwrap();

    for image_id in 1..=num_images {
        let mut frame = Frame::new();
        frame.set_frame_id(image_id);
        frame.set_rig_id(rig.rig_id()).unwrap();
        frame
            .add_data_id(DataId::new(camera.sensor_id(), image_id))
            .unwrap();
        frame.set_rig_from_world(Rigid3d::default());
        reconstruction.add_frame(frame.clone()).unwrap();
        let mut image = Image::new();
        image.set_image_id(image_id);
        image.set_camera_id(camera.camera_id).unwrap();
        image.set_frame_id(frame.frame_id());
        image.set_name(&format!("image{image_id}"));
        image.set_points2d_from_xy(&[Vector2d::zeros(); NUM_POINTS2D]);
        reconstruction.add_image(image).unwrap();
    }
}

fn simple_pinhole_camera() -> Camera {
    Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1.0, 1, 1).unwrap()
}

fn expect_counts(reconstruction: &Reconstruction, counts: [usize; 7]) {
    assert_eq!(reconstruction.num_rigs(), counts[0]);
    assert_eq!(reconstruction.num_cameras(), counts[1]);
    assert_eq!(reconstruction.num_frames(), counts[2]);
    assert_eq!(reconstruction.num_reg_frames(), counts[3]);
    assert_eq!(reconstruction.num_images(), counts[4]);
    assert_eq!(reconstruction.num_reg_images(), counts[5]);
    assert_eq!(reconstruction.num_points3d(), counts[6]);
}

pub(crate) fn has_pose(reconstruction: &Reconstruction, image_id: ImageId) -> bool {
    let image = reconstruction.image(image_id).unwrap();
    image.has_pose(reconstruction.frames().get(&image.frame_id()))
}

#[test]
fn reconstruction_empty() {
    let reconstruction = Reconstruction::new();
    expect_counts(&reconstruction, [0, 0, 0, 0, 0, 0, 0]);
}

#[test]
fn reconstruction_add_rig() {
    let mut reconstruction = Reconstruction::new();
    let mut rig = Rig::new();
    rig.set_rig_id(1);
    let camera = simple_pinhole_camera();
    rig.add_ref_sensor(camera.sensor_id()).unwrap();
    assert!(reconstruction.add_rig(rig.clone()).is_err());
    reconstruction.add_camera(camera).unwrap();
    reconstruction.add_rig(rig.clone()).unwrap();
    assert!(reconstruction.exists_rig(rig.rig_id()));
    assert_eq!(
        reconstruction.rig(rig.rig_id()).unwrap().rig_id(),
        rig.rig_id()
    );
    assert!(reconstruction.rigs().contains_key(&rig.rig_id()));
    assert_eq!(reconstruction.rigs().len(), 1);
    expect_counts(&reconstruction, [1, 1, 0, 0, 0, 0, 0]);
}

#[test]
fn reconstruction_add_camera() {
    let mut reconstruction = Reconstruction::new();
    let camera = simple_pinhole_camera();
    reconstruction.add_camera(camera.clone()).unwrap();
    assert!(reconstruction.exists_camera(camera.camera_id));
    assert_eq!(
        reconstruction.camera(camera.camera_id).unwrap().camera_id,
        camera.camera_id
    );
    assert!(reconstruction.cameras().contains_key(&camera.camera_id));
    assert_eq!(reconstruction.cameras().len(), 1);
    expect_counts(&reconstruction, [0, 1, 0, 0, 0, 0, 0]);
}

#[test]
fn reconstruction_add_camera_with_trivial_rig() {
    let mut reconstruction = Reconstruction::new();
    let camera = simple_pinhole_camera();
    reconstruction
        .add_camera_with_trivial_rig(camera.clone())
        .unwrap();
    assert!(reconstruction.exists_camera(camera.camera_id));
    assert_eq!(
        reconstruction.camera(camera.camera_id).unwrap().camera_id,
        camera.camera_id
    );
    assert!(reconstruction.cameras().contains_key(&camera.camera_id));
    assert_eq!(reconstruction.cameras().len(), 1);
    assert!(reconstruction.exists_rig(camera.camera_id));
    assert_eq!(
        reconstruction.rig(camera.camera_id).unwrap().rig_id(),
        camera.camera_id
    );
    assert_eq!(
        reconstruction.rig(camera.camera_id).unwrap().num_sensors(),
        1
    );
    expect_counts(&reconstruction, [1, 1, 0, 0, 0, 0, 0]);
}

#[test]
fn reconstruction_add_frame() {
    let mut reconstruction = Reconstruction::new();
    let camera = simple_pinhole_camera();
    reconstruction.add_camera(camera.clone()).unwrap();
    let mut rig = Rig::new();
    rig.set_rig_id(1);
    let mut frame = Frame::new();
    frame.set_frame_id(1);
    frame.set_rig_id(rig.rig_id()).unwrap();
    frame
        .add_data_id(DataId::new(camera.sensor_id(), 1))
        .unwrap();
    if let Err(e) = reconstruction.add_frame(frame.clone()) {
        assert!(e.to_string().contains("Rig with ID 1 does not exist"));
    }
    reconstruction.add_rig(rig).unwrap();
    if let Err(e) = reconstruction.add_frame(frame.clone()) {
        assert!(e
            .to_string()
            .contains("Check failed: rig.has_sensor(data_id.sensor_id)"));
    }
    assert!(reconstruction.add_frame(frame.clone()).is_err());
    reconstruction
        .rig_mut(frame.rig_id())
        .unwrap()
        .add_ref_sensor(camera.sensor_id())
        .unwrap();
    reconstruction.add_frame(frame).unwrap();
    assert!(reconstruction.exists_frame(1));
    assert_eq!(reconstruction.frame(1).unwrap().frame_id(), 1);
    assert!(reconstruction.frames().contains_key(&1));
    assert_eq!(reconstruction.frames().len(), 1);
    assert_eq!(reconstruction.num_rigs(), 1);
    assert_eq!(reconstruction.num_cameras(), 1);
    assert_eq!(reconstruction.num_frames(), 1);
    assert_eq!(reconstruction.num_reg_frames(), 0);
    assert_eq!(reconstruction.num_images(), 0);
    assert_eq!(reconstruction.num_points3d(), 0);
    reconstruction
        .frame_mut(1)
        .unwrap()
        .set_rig_from_world(Rigid3d::default());
    reconstruction.register_frame(1).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), 1);
    expect_valid_ptrs(&reconstruction);
}

#[test]
fn reconstruction_add_image_wrong_frame_correspondence() {
    let mut reconstruction = Reconstruction::new();
    let camera = simple_pinhole_camera();
    reconstruction.add_camera(camera.clone()).unwrap();
    let mut rig = Rig::new();
    rig.set_rig_id(1);
    rig.add_ref_sensor(camera.sensor_id()).unwrap();
    reconstruction.add_rig(rig.clone()).unwrap();
    let mut frame = Frame::new();
    frame.set_frame_id(1);
    frame.set_rig_id(rig.rig_id()).unwrap();
    let mut image = Image::new();
    image.set_camera_id(camera.camera_id).unwrap();
    image.set_image_id(1);
    image.set_frame_id(frame.frame_id());
    frame.add_data_id(image.data_id()).unwrap();
    reconstruction.add_frame(frame).unwrap();
    reconstruction.add_image(image.clone()).unwrap();
    image.set_image_id(2);
    assert!(reconstruction.add_image(image).is_err());
}

#[test]
fn reconstruction_add_image() {
    let camera = simple_pinhole_camera();
    let mut rig = Rig::new();
    rig.set_rig_id(1);
    rig.add_ref_sensor(camera.sensor_id()).unwrap();
    let mut image = Image::new();
    image.set_camera_id(camera.camera_id).unwrap();
    image.set_image_id(1);
    image.set_frame_id(1);

    // Verify that adding an image fails if the frame does not exist.
    {
        let mut reconstruction = Reconstruction::new();
        reconstruction.add_camera(camera.clone()).unwrap();
        reconstruction.add_rig(rig.clone()).unwrap();
        if let Err(e) = reconstruction.add_image(image.clone()) {
            assert!(e.to_string().contains("Frame with ID 1 does not exist"));
        }
    }

    // Verify that adding an image fails if the frame has no matching data id.
    {
        let mut reconstruction = Reconstruction::new();
        reconstruction.add_camera(camera.clone()).unwrap();
        reconstruction.add_rig(rig.clone()).unwrap();
        let mut frame_without_data = Frame::new();
        frame_without_data.set_frame_id(1);
        frame_without_data.set_rig_id(rig.rig_id()).unwrap();
        reconstruction.add_frame(frame_without_data).unwrap();
        if let Err(e) = reconstruction.add_image(image.clone()) {
            assert!(e
                .to_string()
                .contains("Check failed: frame.has_data_id(image.data_id())"));
        }
    }

    // Successfully add the image when the frame has the matching data id.
    let mut reconstruction = Reconstruction::new();
    reconstruction.add_camera(camera).unwrap();
    reconstruction.add_rig(rig.clone()).unwrap();
    let mut frame = Frame::new();
    frame.set_frame_id(1);
    frame.set_rig_id(rig.rig_id()).unwrap();
    frame.add_data_id(image.data_id()).unwrap();
    reconstruction.add_frame(frame).unwrap();
    reconstruction.add_image(image).unwrap();
    assert!(reconstruction.exists_image(1));
    assert_eq!(reconstruction.image(1).unwrap().image_id(), 1);
    assert!(!has_pose(&reconstruction, 1));
    assert!(reconstruction.images().contains_key(&1));
    assert_eq!(reconstruction.images().len(), 1);
    expect_counts(&reconstruction, [1, 1, 1, 0, 1, 0, 0]);
    let frame_id = reconstruction.image(1).unwrap().frame_id();
    reconstruction
        .frame_mut(frame_id)
        .unwrap()
        .set_rig_from_world(Rigid3d::default());
    reconstruction.register_frame(1).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), 1);
    assert_eq!(reconstruction.num_reg_images(), 1);
    expect_valid_ptrs(&reconstruction);
}

#[test]
fn reconstruction_add_image_with_trivial_frame() {
    let mut reconstruction = Reconstruction::new();
    let camera = simple_pinhole_camera();
    reconstruction
        .add_camera_with_trivial_rig(camera.clone())
        .unwrap();
    let mut image = Image::new();
    image.set_camera_id(camera.camera_id).unwrap();
    image.set_image_id(1);
    reconstruction.add_image_with_trivial_frame(image).unwrap();

    assert!(reconstruction.exists_image(1));
    assert_eq!(reconstruction.image(1).unwrap().image_id(), 1);
    assert_eq!(reconstruction.image(1).unwrap().frame_id(), 1);
    assert!(!has_pose(&reconstruction, 1));
    assert!(reconstruction.images().contains_key(&1));
    assert_eq!(reconstruction.images().len(), 1);
    assert!(reconstruction.exists_frame(1));
    assert_eq!(reconstruction.frame(1).unwrap().num_data_ids(), 1);
    expect_counts(&reconstruction, [1, 1, 1, 0, 1, 0, 0]);
    let frame_id = reconstruction.image(1).unwrap().frame_id();
    reconstruction
        .frame_mut(frame_id)
        .unwrap()
        .set_rig_from_world(Rigid3d::default());
    reconstruction.register_frame(1).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), 1);
    assert_eq!(reconstruction.num_reg_images(), 1);
    expect_valid_ptrs(&reconstruction);
}

#[test]
fn reconstruction_add_image_with_trivial_frame_exists_non_trivial_rig() {
    let mut reconstruction = Reconstruction::new();
    let mut camera = simple_pinhole_camera();
    reconstruction
        .add_camera_with_trivial_rig(camera.clone())
        .unwrap();
    camera.camera_id = 2;
    reconstruction
        .rig_mut(1)
        .unwrap()
        .add_sensor(camera.sensor_id(), None)
        .unwrap();
    assert_eq!(reconstruction.rig(1).unwrap().num_sensors(), 2);

    let mut image = Image::new();
    image.set_image_id(1);
    // The rig has multiple cameras
    image.set_camera_id(1).unwrap();
    assert!(reconstruction
        .add_image_with_trivial_frame(image.clone())
        .is_err());
    // No rig with id 2 found in the reconstruction
    image.set_camera_id(2).unwrap();
    assert!(reconstruction.add_image_with_trivial_frame(image).is_err());
}

#[test]
fn reconstruction_add_image_with_trivial_frame_set_cam_from_world() {
    let mut reconstruction = Reconstruction::new();
    let camera = simple_pinhole_camera();
    reconstruction
        .add_camera_with_trivial_rig(camera.clone())
        .unwrap();
    let mut image = Image::new();
    image.set_camera_id(camera.camera_id).unwrap();
    image.set_image_id(1);
    reconstruction
        .add_image_with_trivial_frame_posed(image, Rigid3d::default())
        .unwrap();
    assert!(reconstruction.exists_image(1));
    assert_eq!(reconstruction.image(1).unwrap().image_id(), 1);
    assert_eq!(reconstruction.image(1).unwrap().frame_id(), 1);
    assert!(has_pose(&reconstruction, 1));
    assert!(reconstruction.images().contains_key(&1));
    assert_eq!(reconstruction.images().len(), 1);
    assert!(reconstruction.exists_frame(1));
    assert_eq!(reconstruction.frame(1).unwrap().num_data_ids(), 1);
    assert!(reconstruction.frame(1).unwrap().has_pose());
    expect_counts(&reconstruction, [1, 1, 1, 1, 1, 1, 0]);
    expect_valid_ptrs(&reconstruction);
}

#[test]
fn reconstruction_reg_image_ids() {
    let mut reconstruction = Reconstruction::new();

    let camera = simple_pinhole_camera();
    reconstruction.add_camera(camera.clone()).unwrap();

    let mut rig = Rig::new();
    rig.set_rig_id(1);
    rig.add_ref_sensor(camera.sensor_id()).unwrap();
    reconstruction.add_rig(rig.clone()).unwrap();

    let mut image1 = Image::new();
    image1.set_camera_id(camera.camera_id).unwrap();
    image1.set_image_id(1);
    image1.set_frame_id(1);
    let mut image2 = Image::new();
    image2.set_camera_id(camera.camera_id).unwrap();
    image2.set_image_id(2);
    image2.set_frame_id(1);

    let mut frame = Frame::new();
    frame.set_frame_id(1);
    frame.set_rig_id(rig.rig_id()).unwrap();
    frame.add_data_id(image1.data_id()).unwrap();
    frame.add_data_id(image2.data_id()).unwrap();
    let frame_id = frame.frame_id();
    reconstruction.add_frame(frame).unwrap();
    reconstruction
        .frame_mut(frame_id)
        .unwrap()
        .set_rig_from_world(Rigid3d::default());

    reconstruction.register_frame(frame_id).unwrap();

    // Throws because no image was added.
    assert!(reconstruction.reg_image_ids().is_err());
    assert_eq!(reconstruction.num_reg_frames(), 1);
    assert_eq!(reconstruction.num_reg_images(), 2);

    // Throws because second image was not added.
    reconstruction.add_image(image1).unwrap();
    assert!(reconstruction.reg_image_ids().is_err());
    assert_eq!(reconstruction.num_reg_frames(), 1);
    assert_eq!(reconstruction.num_reg_images(), 2);

    reconstruction.add_image(image2).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), 1);
    assert_eq!(reconstruction.num_reg_images(), 2);
    assert_eq!(reconstruction.reg_image_ids().unwrap(), vec![1, 2]);

    // Registering a frame twice is a no-op.
    reconstruction.register_frame(frame_id).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), 1);
    assert_eq!(reconstruction.num_reg_images(), 2);
    assert_eq!(reconstruction.reg_image_ids().unwrap(), vec![1, 2]);

    reconstruction.deregister_frame(frame_id).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), 0);
    assert_eq!(reconstruction.num_reg_images(), 0);
    assert!(reconstruction.reg_image_ids().unwrap().is_empty());

    // De-registering a frame twice is a no-op.
    reconstruction.deregister_frame(frame_id).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), 0);
    assert_eq!(reconstruction.num_reg_images(), 0);
    assert!(reconstruction.reg_image_ids().unwrap().is_empty());
}

#[test]
fn reconstruction_add_point3d() {
    let mut reconstruction = Reconstruction::new();
    let point3d_id = reconstruction
        .add_point3d(
            random_eigen_vector3d(),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    assert!(reconstruction.exists_point3d(point3d_id));
    assert_eq!(
        reconstruction.point3d(point3d_id).unwrap().track.length(),
        0
    );
    assert!(reconstruction.points3d().contains_key(&point3d_id));
    assert_eq!(reconstruction.points3d().len(), 1);
    assert_eq!(reconstruction.num_rigs(), 0);
    assert_eq!(reconstruction.num_cameras(), 0);
    assert_eq!(reconstruction.num_frames(), 0);
    assert_eq!(reconstruction.num_images(), 0);
    assert_eq!(reconstruction.num_reg_frames(), 0);
    assert_eq!(reconstruction.num_points3d(), 1);
    assert!(reconstruction.point3d_ids().contains(&point3d_id));

    let mut reconstruction2 = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction2);
    let mut point3d = Point3D {
        xyz: Vector3d::new(1.0, 2.0, 3.0),
        ..Point3D::default()
    };
    point3d.track.add_element_ids(1, 0);
    point3d.track.add_element_ids(2, 1);
    reconstruction2.add_point3d_with_id(5, point3d).unwrap();
    assert_eq!(reconstruction2.point3d(5).unwrap().track.length(), 2);
    assert!(reconstruction2.image(1).unwrap().point2d(0).has_point3d());
    assert!(reconstruction2.image(2).unwrap().point2d(1).has_point3d());
    assert_eq!(reconstruction2.num_rigs(), 1);
    assert_eq!(reconstruction2.num_cameras(), 1);
    assert_eq!(reconstruction2.num_frames(), 2);
    assert_eq!(reconstruction2.num_images(), 2);
    assert_eq!(reconstruction2.num_reg_frames(), 2);
    assert_eq!(reconstruction2.num_points3d(), 1);
    assert!(reconstruction2.point3d_ids().contains(&5));
}

#[test]
fn reconstruction_add_observation() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(3, &mut reconstruction);
    let mut track = Track::new();
    track.add_element_ids(1, 0);
    track.add_element_ids(2, 1);
    let point3d_id = reconstruction
        .add_point3d(random_eigen_vector3d(), track, Vector3ub::new(0, 0, 0))
        .unwrap();
    let image = |r: &Reconstruction, id| r.image(id).unwrap().clone();
    assert_eq!(image(&reconstruction, 1).num_points3d(), 1);
    assert!(image(&reconstruction, 1).point2d(0).has_point3d());
    assert!(!image(&reconstruction, 1).point2d(1).has_point3d());
    assert_eq!(image(&reconstruction, 2).num_points3d(), 1);
    assert!(!image(&reconstruction, 2).point2d(0).has_point3d());
    assert!(image(&reconstruction, 2).point2d(1).has_point3d());
    assert_eq!(
        reconstruction.point3d(point3d_id).unwrap().track.length(),
        2
    );
    reconstruction
        .add_observation(point3d_id, TrackElement::new(3, 2))
        .unwrap();
    assert_eq!(image(&reconstruction, 3).num_points3d(), 1);
    assert!(image(&reconstruction, 3).point2d(2).has_point3d());
    assert_eq!(
        reconstruction.point3d(point3d_id).unwrap().track.length(),
        3
    );
}

#[test]
fn reconstruction_merge_points3d() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    let black = Vector3ub::new(0, 0, 0);
    let point3d_id1 = reconstruction
        .add_point3d(Vector3d::new(0.0, 0.0, 0.0), Track::new(), black)
        .unwrap();
    reconstruction
        .add_observation(point3d_id1, TrackElement::new(1, 0))
        .unwrap();
    reconstruction
        .add_observation(point3d_id1, TrackElement::new(2, 0))
        .unwrap();
    reconstruction.point3d_mut(point3d_id1).unwrap().color = Vector3ub::new(0, 0, 0);
    let point3d_id2 = reconstruction
        .add_point3d(Vector3d::new(1.0, 1.0, 1.0), Track::new(), black)
        .unwrap();
    reconstruction
        .add_observation(point3d_id2, TrackElement::new(1, 1))
        .unwrap();
    reconstruction
        .add_observation(point3d_id2, TrackElement::new(2, 1))
        .unwrap();
    reconstruction.point3d_mut(point3d_id2).unwrap().color = Vector3ub::new(20, 20, 20);
    let merged_point3d_id = reconstruction
        .merge_points3d(point3d_id1, point3d_id2)
        .unwrap();
    assert!(!reconstruction.exists_point3d(point3d_id1));
    assert!(!reconstruction.exists_point3d(point3d_id2));
    assert!(reconstruction.exists_point3d(merged_point3d_id));
    for (image_id, point2d_idx) in [(1, 0), (1, 1), (2, 0), (2, 1)] {
        assert_eq!(
            reconstruction
                .image(image_id)
                .unwrap()
                .point2d(point2d_idx)
                .point3d_id,
            merged_point3d_id
        );
    }
    let merged = reconstruction.point3d(merged_point3d_id).unwrap();
    assert!(merged.xyz.is_approx(Vector3d::new(0.5, 0.5, 0.5)));
    assert_eq!(merged.color, Vector3ub::new(10, 10, 10));
}

#[test]
fn reconstruction_delete_point3d() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(1, &mut reconstruction);
    let point3d_id = reconstruction
        .add_point3d(
            random_eigen_vector3d(),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    reconstruction
        .add_observation(point3d_id, TrackElement::new(1, 0))
        .unwrap();
    reconstruction.delete_point3d(point3d_id).unwrap();
    assert!(!reconstruction.exists_point3d(point3d_id));
    assert_eq!(reconstruction.image(1).unwrap().num_points3d(), 0);
}

#[test]
fn reconstruction_delete_observation() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    let point3d_id = reconstruction
        .add_point3d(
            Vector3d::new(0.0, 0.0, 0.0),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    for point2d_idx in 0..3 {
        reconstruction
            .add_observation(point3d_id, TrackElement::new(1, point2d_idx))
            .unwrap();
    }
    reconstruction.delete_observation(1, 0).unwrap();
    assert_eq!(
        reconstruction.point3d(point3d_id).unwrap().track.length(),
        2
    );
    // COLMAP indexes `Image(point3D_id)`: the point id is 1, the image's id.
    let image_id = point3d_id as ImageId;
    assert!(!reconstruction
        .image(image_id)
        .unwrap()
        .point2d(0)
        .has_point3d());
    reconstruction.delete_observation(1, 1).unwrap();
    assert!(!reconstruction.exists_point3d(point3d_id));
    assert!(!reconstruction
        .image(image_id)
        .unwrap()
        .point2d(1)
        .has_point3d());
    assert!(!reconstruction
        .image(image_id)
        .unwrap()
        .point2d(2)
        .has_point3d());
}

#[test]
fn reconstruction_register_frame() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(1, &mut reconstruction);
    let expect_state = |r: &Reconstruction, num_reg_frames: usize, posed: bool| {
        assert_eq!(r.num_reg_frames(), num_reg_frames);
        assert_eq!(has_pose(r, 1), posed);
        assert_eq!(r.frame(1).unwrap().has_pose(), posed);
    };
    expect_state(&reconstruction, 1, true);
    reconstruction.register_frame(1).unwrap();
    expect_state(&reconstruction, 1, true);
    reconstruction.register_frame(1).unwrap();
    expect_state(&reconstruction, 1, true);
    reconstruction.deregister_frame(1).unwrap();
    expect_state(&reconstruction, 0, false);
    reconstruction.deregister_frame(1).unwrap();
    expect_state(&reconstruction, 0, false);
}
