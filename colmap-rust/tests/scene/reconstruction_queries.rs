// Port of colmap/scene/reconstruction_test.cc, part 2, 1:1 (same test names, values and
// checks): the tests of colmap_rust::scene::reconstruction_queries (FindImageWithName,
// FindCommonRegImageIds, the Compute* statistics, UpdatePoint3DErrors, IsValid,
// DeRegisterFrame and Point3DIds). The helpers and the conventions are in reconstruction.rs.

use colmap_rust::linalg::{Vector2d, Vector3d, Vector3ub};
use colmap_rust::math::random_eigen::random_eigen_vector3d;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::track::{Track, TrackElement};
use colmap_rust::util::types::{ImageId, INVALID_FRAME_ID};

use super::reconstruction::generate_reconstruction;

#[test]
fn reconstruction_find_image_with_name() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    assert!(std::ptr::eq(
        reconstruction.find_image_with_name("image1").unwrap(),
        reconstruction.image(1).unwrap()
    ));
    assert!(std::ptr::eq(
        reconstruction.find_image_with_name("image2").unwrap(),
        reconstruction.image(2).unwrap()
    ));
    assert!(reconstruction.find_image_with_name("image3").is_none());
}

#[test]
fn reconstruction_find_common_reg_image_ids() {
    let mut reconstruction1 = Reconstruction::new();
    generate_reconstruction(5, &mut reconstruction1);
    let mut reconstruction2 = Reconstruction::new();
    generate_reconstruction(5, &mut reconstruction2);
    reconstruction1.deregister_frame(1).unwrap();
    reconstruction1.image_mut(2).unwrap().set_name("foo");
    reconstruction2.deregister_frame(3).unwrap();
    reconstruction2.image_mut(4).unwrap().set_name("bar");
    let common_image_ids = reconstruction1
        .find_common_reg_image_ids(&reconstruction2)
        .unwrap();
    assert_eq!(common_image_ids.len(), 1);
    assert_eq!(common_image_ids[0].0, 5);
    assert_eq!(common_image_ids[0].1, 5);
    assert_eq!(
        common_image_ids,
        reconstruction2
            .find_common_reg_image_ids(&reconstruction1)
            .unwrap()
    );
}

/// The shared setup of the `Compute*` tests: two images and one point without a track.
fn reconstruction_with_trackless_point() -> (Reconstruction, u64) {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    let point3d_id = reconstruction
        .add_point3d(
            random_eigen_vector3d(),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    (reconstruction, point3d_id)
}

const OBSERVATIONS: [(ImageId, u32); 3] = [(1, 0), (1, 1), (2, 0)];

#[test]
fn reconstruction_compute_num_observations() {
    let (mut reconstruction, point3d_id1) = reconstruction_with_trackless_point();
    assert_eq!(reconstruction.compute_num_observations().unwrap(), 0);
    for (n, (image_id, point2d_idx)) in OBSERVATIONS.into_iter().enumerate() {
        reconstruction
            .add_observation(point3d_id1, TrackElement::new(image_id, point2d_idx))
            .unwrap();
        assert_eq!(reconstruction.compute_num_observations().unwrap(), n + 1);
    }
}

#[test]
fn reconstruction_compute_mean_track_length() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    assert_eq!(reconstruction.compute_mean_track_length().unwrap(), 0.0);
    let point3d_id1 = reconstruction
        .add_point3d(
            random_eigen_vector3d(),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    assert_eq!(reconstruction.compute_mean_track_length().unwrap(), 0.0);
    for (expected, (image_id, point2d_idx)) in [1.0, 2.0, 3.0].into_iter().zip(OBSERVATIONS) {
        reconstruction
            .add_observation(point3d_id1, TrackElement::new(image_id, point2d_idx))
            .unwrap();
        assert_eq!(
            reconstruction.compute_mean_track_length().unwrap(),
            expected
        );
    }
}

#[test]
fn reconstruction_compute_mean_observations_per_reg_image() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    assert_eq!(
        reconstruction
            .compute_mean_observations_per_reg_image()
            .unwrap(),
        0.0
    );
    let point3d_id1 = reconstruction
        .add_point3d(
            random_eigen_vector3d(),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    assert_eq!(
        reconstruction
            .compute_mean_observations_per_reg_image()
            .unwrap(),
        0.0
    );
    for (expected, (image_id, point2d_idx)) in [0.5, 1.0, 1.5].into_iter().zip(OBSERVATIONS) {
        reconstruction
            .add_observation(point3d_id1, TrackElement::new(image_id, point2d_idx))
            .unwrap();
        assert_eq!(
            reconstruction
                .compute_mean_observations_per_reg_image()
                .unwrap(),
            expected
        );
    }
}

#[test]
fn reconstruction_compute_mean_reprojection_error() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    assert_eq!(reconstruction.compute_mean_reprojection_error(), 0.0);
    let point3d_id1 = reconstruction
        .add_point3d(
            random_eigen_vector3d(),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    assert_eq!(reconstruction.compute_mean_reprojection_error(), 0.0);
    for (error, expected) in [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)] {
        reconstruction.point3d_mut(point3d_id1).unwrap().error = error;
        assert_eq!(reconstruction.compute_mean_reprojection_error(), expected);
    }
}

#[test]
fn reconstruction_update_point3d_errors() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    assert_eq!(reconstruction.compute_mean_reprojection_error(), 0.0);
    let mut track = Track::new();
    track.add_element_ids(1, 0);
    reconstruction.image_mut(1).unwrap().point2d_mut(0).xy = Vector2d::new(0.5, 0.5);
    let point3d_id = reconstruction
        .add_point3d(Vector3d::new(0.0, 0.0, 1.0), track, Vector3ub::new(0, 0, 0))
        .unwrap();
    assert_eq!(reconstruction.point3d(point3d_id).unwrap().error, -1.0);
    reconstruction.update_point3d_errors().unwrap();
    assert_eq!(reconstruction.point3d(point3d_id).unwrap().error, 0.0);
    reconstruction.point3d_mut(point3d_id).unwrap().xyz = Vector3d::new(0.0, 1.0, 1.0);
    reconstruction.update_point3d_errors().unwrap();
    assert_eq!(reconstruction.point3d(point3d_id).unwrap().error, 1.0);
}

#[test]
fn reconstruction_is_valid() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    let mut track = Track::new();
    track.add_element_ids(1, 0);
    track.add_element_ids(2, 1);
    reconstruction
        .add_point3d(random_eigen_vector3d(), track, Vector3ub::new(0, 0, 0))
        .unwrap();
    assert!(reconstruction.is_valid());

    // Test empty frame pointer for image: with ids, the image loses its frame id
    // (the `scene` module header).
    {
        let mut reconstruction_copy = reconstruction.clone();
        reconstruction_copy
            .image_mut(1)
            .unwrap()
            .set_frame_id(INVALID_FRAME_ID);
        assert!(!reconstruction_copy.is_valid());
    }

    // Test breaking track consistency by directly modifying point3D track.
    {
        let mut reconstruction_copy = reconstruction.clone();
        reconstruction_copy
            .point3d_mut(1)
            .unwrap()
            .track
            .set_element(0, TrackElement::new(1, 5));
        assert!(!reconstruction_copy.is_valid());
    }

    // Test registered frame without pose.
    {
        let mut reconstruction_copy = reconstruction.clone();
        reconstruction_copy.frame_mut(1).unwrap().reset_pose();
        assert!(!reconstruction_copy.is_valid());
    }
}

#[test]
fn reconstruction_deregister_frame() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(3, &mut reconstruction);
    let mut track = Track::new();
    track.add_element_ids(1, 0);
    track.add_element_ids(2, 0);
    let point3d_id = reconstruction
        .add_point3d(random_eigen_vector3d(), track, Vector3ub::new(0, 0, 0))
        .unwrap();

    assert_eq!(reconstruction.num_reg_frames(), 3);
    assert_eq!(reconstruction.num_reg_images(), 3);

    reconstruction.deregister_frame(1).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), 2);
    assert_eq!(reconstruction.num_reg_images(), 2);
    assert!(reconstruction.exists_frame(1));
    assert!(!reconstruction.frame(1).unwrap().has_pose());
    // The 3D point had observations in images 1 and 2; after de-registering frame 1, the
    // point should be deleted (track becomes too short).
    assert!(!reconstruction.exists_point3d(point3d_id));

    // De-registering an already de-registered frame is a no-op (with warning)
    reconstruction.deregister_frame(1).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), 2);
}

#[test]
fn reconstruction_point3d_ids() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    let p1 = reconstruction
        .add_point3d(
            Vector3d::new(1.0, 2.0, 3.0),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    let p2 = reconstruction
        .add_point3d(
            Vector3d::new(4.0, 5.0, 6.0),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();

    let ids = reconstruction.point3d_ids();
    assert_eq!(ids.len(), 2);
    assert_eq!(ids.iter().filter(|&&id| id == p1).count(), 1);
    assert_eq!(ids.iter().filter(|&&id| id == p2).count(), 1);
}
