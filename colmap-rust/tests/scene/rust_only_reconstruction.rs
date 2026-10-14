// Rust-only tests for colmap_rust::scene::reconstruction: the accessor errors, operator<<,
// TearDown on a hand-built model, and docs/CPP_DIVERGENCES.md entries 213-216 (ascending id
// order, id changes caught by is_valid, DeleteAllPoints2DAndPoints3D's triangulated count,
// failed operations leaving the reconstruction unchanged).

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Vector2d, Vector3d, Vector3ub};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::frame::Frame;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::point3d::Point3D;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::track::{Track, TrackElement};
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::util::types::ImageId;
use colmap_rust::ErrorKind;

/// Cameras 1..=n, each in its trivial rig, with one image of 4 points in a trivial frame;
/// the frames whose id is in `posed` get a pose and are registered.
fn model(num_images: ImageId, posed: &[ImageId]) -> Reconstruction {
    let mut reconstruction = Reconstruction::new();
    for image_id in 1..=num_images {
        let camera =
            Camera::create_from_model_id(image_id, CameraModelId::SimplePinhole, 1.0, 1, 1)
                .unwrap();
        reconstruction.add_camera_with_trivial_rig(camera).unwrap();
        let mut image = Image::new();
        image.set_image_id(image_id);
        image.set_camera_id(image_id).unwrap();
        image.set_name(&format!("image{image_id}"));
        image.set_points2d_from_xy(&[Vector2d::zeros(); 4]);
        if posed.contains(&image_id) {
            reconstruction
                .add_image_with_trivial_frame_posed(image, Rigid3d::default())
                .unwrap();
        } else {
            reconstruction.add_image_with_trivial_frame(image).unwrap();
        }
    }
    reconstruction
}

fn track(elements: &[(ImageId, u32)]) -> Track {
    let mut track = Track::new();
    for &(image_id, point2d_idx) in elements {
        track.add_element_ids(image_id, point2d_idx);
    }
    track
}

#[test]
fn rust_only_reconstruction_missing_objects_are_out_of_range() {
    let mut reconstruction = Reconstruction::new();
    let cases = [
        (
            reconstruction.rig(3).unwrap_err(),
            "Rig with ID 3 does not exist",
        ),
        (
            reconstruction.camera(4).unwrap_err(),
            "Camera with ID 4 does not exist",
        ),
        (
            reconstruction.frame(5).unwrap_err(),
            "Frame with ID 5 does not exist",
        ),
        (
            reconstruction.image(6).unwrap_err(),
            "Image with ID 6 does not exist",
        ),
        (
            reconstruction.point3d(7).unwrap_err(),
            "Point3D with ID 7 does not exist",
        ),
    ];
    for (err, message) in cases {
        assert_eq!(err.kind(), ErrorKind::OutOfRange);
        assert_eq!(err.message(), message);
    }
    assert!(reconstruction.image_mut(6).is_err());
    assert!(reconstruction.delete_point3d(7).is_err());
}

#[test]
fn rust_only_reconstruction_print() {
    let mut reconstruction = model(3, &[1, 2]);
    reconstruction
        .add_point3d(Vector3d::zeros(), Track::new(), Vector3ub::new(0, 0, 0))
        .unwrap();
    assert_eq!(
        reconstruction.to_string(),
        "Reconstruction(num_rigs=3, num_cameras=3, num_frames=3, num_reg_frames=2, \
         num_images=3, num_points3D=1)"
    );
}

#[test]
fn rust_only_reconstruction_iterates_in_ascending_id_order() {
    let mut reconstruction = model(4, &[]);
    for frame_id in [4, 2] {
        reconstruction
            .frame_mut(frame_id)
            .unwrap()
            .set_rig_from_world(Rigid3d::default());
        reconstruction.register_frame(frame_id).unwrap();
    }
    reconstruction.image_mut(3).unwrap().set_name("image1");
    // Two images named "image1": the smallest id wins (entry 213).
    assert_eq!(
        reconstruction
            .find_image_with_name("image1")
            .unwrap()
            .image_id(),
        1
    );
    let ids: Vec<ImageId> = reconstruction.images().keys().copied().collect();
    assert_eq!(ids, vec![1, 2, 3, 4]);
    // Registration order is kept.
    assert_eq!(reconstruction.reg_frame_ids(), &[4, 2]);
    assert_eq!(reconstruction.reg_image_ids().unwrap(), vec![4, 2]);
}

#[test]
fn rust_only_reconstruction_changed_ids_are_invalid() {
    let reconstruction = model(2, &[1, 2]);
    assert!(reconstruction.is_valid());
    // Moving an attached image to another camera or frame is not rejected at the setter
    // (entry 214); is_valid reports it.
    let mut moved_camera = reconstruction.clone();
    moved_camera.image_mut(1).unwrap().set_camera_id(9).unwrap();
    assert!(!moved_camera.is_valid());
    let mut moved_frame = reconstruction.clone();
    moved_frame.image_mut(1).unwrap().set_frame_id(2);
    assert!(!moved_frame.is_valid());
    let mut moved_rig = reconstruction.clone();
    moved_rig.frame_mut(1).unwrap().set_rig_id(2).unwrap();
    assert!(!moved_rig.is_valid());
}

#[test]
fn rust_only_reconstruction_delete_all_points2d_and_points3d_resets_counts() {
    let mut reconstruction = model(2, &[1, 2]);
    reconstruction
        .add_point3d(
            Vector3d::zeros(),
            track(&[(1, 0), (2, 0)]),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    reconstruction.delete_all_points2d_and_points3d().unwrap();
    assert_eq!(reconstruction.num_points3d(), 0);
    for image in reconstruction.images().values() {
        assert_eq!(image.num_points2d(), 0);
        // COLMAP leaves this at 1 (entry 215).
        assert_eq!(image.num_points3d(), 0);
    }
    assert!(reconstruction.is_valid());
}

#[test]
fn rust_only_reconstruction_failed_operations_leave_it_unchanged() {
    let mut reconstruction = model(2, &[1, 2]);
    let point3d_id = reconstruction
        .add_point3d(
            Vector3d::zeros(),
            track(&[(1, 0), (2, 0)]),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    let before = reconstruction.clone();

    // A track through a missing image (entry 216): nothing is marked, the id is not used.
    let bad = Point3D {
        track: track(&[(1, 1), (7, 0)]),
        ..Point3D::default()
    };
    assert!(reconstruction.add_point3d_with_id(10, bad).is_err());
    // A track element already observing another point.
    let taken = Point3D {
        track: track(&[(2, 1), (1, 0)]),
        ..Point3D::default()
    };
    assert!(reconstruction.add_point3d_with_id(11, taken).is_err());
    // An id that is taken.
    let duplicate = Point3D {
        track: track(&[(1, 2)]),
        ..Point3D::default()
    };
    assert!(reconstruction
        .add_point3d_with_id(point3d_id, duplicate)
        .is_err());
    // An observation of a missing point.
    assert!(reconstruction
        .add_observation(99, TrackElement::new(1, 3))
        .is_err());

    assert!(reconstruction.is_valid());
    for (image_id, image) in reconstruction.images() {
        assert_eq!(image, before.image(*image_id).unwrap());
    }
    assert_eq!(reconstruction.point3d_ids(), before.point3d_ids());
    // The next point still gets the id after the existing one.
    let next = reconstruction
        .add_point3d(Vector3d::zeros(), Track::new(), Vector3ub::new(0, 0, 0))
        .unwrap();
    assert_eq!(next, point3d_id + 1);
}

#[test]
fn rust_only_reconstruction_tear_down() {
    let mut reconstruction = model(3, &[1, 3]);
    let mut duplicate_track = track(&[(1, 0), (3, 0)]);
    duplicate_track.reserve(16);
    reconstruction
        .add_point3d(Vector3d::zeros(), duplicate_track, Vector3ub::new(0, 0, 0))
        .unwrap();
    reconstruction.tear_down();
    // Frame 2 had no pose: it, its image, its rig and its camera are gone.
    assert_eq!(
        reconstruction.frames().keys().copied().collect::<Vec<_>>(),
        vec![1, 3]
    );
    assert_eq!(
        reconstruction.images().keys().copied().collect::<Vec<_>>(),
        vec![1, 3]
    );
    assert_eq!(
        reconstruction.rigs().keys().copied().collect::<Vec<_>>(),
        vec![1, 3]
    );
    assert_eq!(
        reconstruction.cameras().keys().copied().collect::<Vec<_>>(),
        vec![1, 3]
    );
    assert_eq!(reconstruction.num_points3d(), 1);
    assert!(reconstruction.is_valid());
}

#[test]
fn rust_only_reconstruction_set_rigs_and_frames_moves_images() {
    let mut reconstruction = model(2, &[1, 2]);
    // One rig holding both cameras, one frame holding both images.
    let mut rig = reconstruction.rig(1).unwrap().clone();
    rig.add_sensor(
        reconstruction.camera(2).unwrap().sensor_id(),
        Some(Rigid3d::default()),
    )
    .unwrap();
    let mut frame = Frame::new();
    frame.set_frame_id(7);
    frame.set_rig_id(1).unwrap();
    for image_id in [1, 2] {
        frame
            .add_data_id(reconstruction.image(image_id).unwrap().data_id())
            .unwrap();
    }
    frame.set_rig_from_world(Rigid3d::default());
    reconstruction
        .set_rigs_and_frames(vec![rig], vec![frame])
        .unwrap();
    assert_eq!(reconstruction.num_rigs(), 1);
    assert_eq!(reconstruction.num_frames(), 1);
    assert_eq!(reconstruction.num_reg_frames(), 1);
    assert_eq!(reconstruction.num_reg_images(), 2);
    assert_eq!(reconstruction.image(1).unwrap().frame_id(), 7);
    assert_eq!(reconstruction.image(2).unwrap().frame_id(), 7);
    assert!(reconstruction.is_valid());
}
