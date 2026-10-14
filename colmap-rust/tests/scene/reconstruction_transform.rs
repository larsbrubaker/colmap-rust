// Port of colmap/scene/reconstruction_test.cc, part 3, 1:1 (same test names, values and
// checks): the tests of colmap_rust::scene::reconstruction_transform — Normalize,
// ComputeBoundsAndCentroidEmpty, ComputeBoundsAndCentroid, Crop and Transform. COLMAP's
// default arguments are spelled out (`DEFAULT_NORMALIZE_*`); `Image(i).CamFromWorld()` is
// `cam_from_world`, which resolves the image's frame and rig.

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::geometry::sim3::Sim3d;
use colmap_rust::linalg::{AlignedBox3d, Quaterniond, Vector3d, Vector3ub};
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::reconstruction_transform::{
    DEFAULT_NORMALIZE_EXTENT, DEFAULT_NORMALIZE_MAX_PERCENTILE, DEFAULT_NORMALIZE_MIN_PERCENTILE,
};
use colmap_rust::scene::track::{Track, TrackElement};
use colmap_rust::util::types::{FrameId, ImageId};

use crate::reconstruction::generate_reconstruction;

fn cam_from_world(reconstruction: &Reconstruction, image_id: ImageId) -> Rigid3d {
    let image = reconstruction.image(image_id).unwrap();
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    image.cam_from_world(frame, rig).unwrap()
}

fn expect_tz(reconstruction: &Reconstruction, image_id: ImageId, expected: f64) {
    let tz = cam_from_world(reconstruction, image_id).translation.z;
    assert!(
        (tz - expected).abs() <= 1e-6,
        "image {image_id}: {tz} vs {expected}"
    );
}

fn set_tz(reconstruction: &mut Reconstruction, frame_id: FrameId, tz: f64) {
    let frame = reconstruction.frame_mut(frame_id).unwrap();
    frame.set_rig_from_world(Rigid3d::default());
    frame.rig_from_world_mut().unwrap().translation.z = tz;
}

/// `Normalize(fixed_scale, extent, min_percentile, max_percentile)` with the default
/// `use_images = true`.
fn normalize(
    reconstruction: &mut Reconstruction,
    fixed_scale: bool,
    extent: f64,
    min_percentile: f64,
    max_percentile: f64,
) -> Sim3d {
    reconstruction
        .normalize(fixed_scale, extent, min_percentile, max_percentile, true)
        .unwrap()
}

fn normalize_default(reconstruction: &mut Reconstruction, fixed_scale: bool) -> Sim3d {
    normalize(
        reconstruction,
        fixed_scale,
        DEFAULT_NORMALIZE_EXTENT,
        DEFAULT_NORMALIZE_MIN_PERCENTILE,
        DEFAULT_NORMALIZE_MAX_PERCENTILE,
    )
}

#[test]
fn reconstruction_normalize() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(7, &mut reconstruction);
    let frame_ids: Vec<FrameId> = reconstruction.frames().keys().copied().collect();
    for frame_id in frame_ids {
        reconstruction.deregister_frame(frame_id).unwrap();
    }
    let mut tform = normalize_default(&mut reconstruction, false);
    assert_eq!(tform.scale, 1.0);
    assert_eq!(tform.rotation, Quaterniond::identity());
    assert_eq!(tform.translation, Vector3d::new(0.0, 0.0, 0.0));
    set_tz(&mut reconstruction, 1, -20.0);
    set_tz(&mut reconstruction, 2, -10.0);
    set_tz(&mut reconstruction, 3, 0.0);
    reconstruction.register_frame(1).unwrap();
    reconstruction.register_frame(2).unwrap();
    reconstruction.register_frame(3).unwrap();
    normalize_default(&mut reconstruction, true);
    expect_tz(&reconstruction, 1, -10.0);
    expect_tz(&reconstruction, 2, 0.0);
    expect_tz(&reconstruction, 3, 10.0);
    normalize_default(&mut reconstruction, false);
    expect_tz(&reconstruction, 1, -5.0);
    expect_tz(&reconstruction, 2, 0.0);
    expect_tz(&reconstruction, 3, 5.0);
    normalize(
        &mut reconstruction,
        false,
        5.0,
        DEFAULT_NORMALIZE_MIN_PERCENTILE,
        DEFAULT_NORMALIZE_MAX_PERCENTILE,
    );
    expect_tz(&reconstruction, 1, -2.5);
    expect_tz(&reconstruction, 2, 0.0);
    expect_tz(&reconstruction, 3, 2.5);
    normalize(&mut reconstruction, false, 10.0, 0.0, 1.0);
    expect_tz(&reconstruction, 1, -5.0);
    expect_tz(&reconstruction, 2, 0.0);
    expect_tz(&reconstruction, 3, 5.0);
    tform = normalize(
        &mut reconstruction,
        false,
        20.0,
        DEFAULT_NORMALIZE_MIN_PERCENTILE,
        DEFAULT_NORMALIZE_MAX_PERCENTILE,
    );
    expect_tz(&reconstruction, 1, -10.0);
    expect_tz(&reconstruction, 2, 0.0);
    expect_tz(&reconstruction, 3, 10.0);
    reconstruction.transform(&tform.inverse());
    expect_tz(&reconstruction, 1, -5.0);
    expect_tz(&reconstruction, 2, 0.0);
    expect_tz(&reconstruction, 3, 5.0);
    reconstruction.transform(&tform);
    expect_tz(&reconstruction, 1, -10.0);
    expect_tz(&reconstruction, 2, 0.0);
    expect_tz(&reconstruction, 3, 10.0);
    // `Image(i).FramePtr()->...`: images 4..7 are in frames 4..7.
    set_tz(&mut reconstruction, 4, -7.5);
    set_tz(&mut reconstruction, 5, -5.0);
    set_tz(&mut reconstruction, 6, 5.0);
    set_tz(&mut reconstruction, 7, 7.5);
    reconstruction.register_frame(4).unwrap();
    reconstruction.register_frame(5).unwrap();
    reconstruction.register_frame(6).unwrap();
    reconstruction.register_frame(7).unwrap();
    normalize(&mut reconstruction, false, 10.0, 0.0, 1.0);
    expect_tz(&reconstruction, 1, -5.0);
    expect_tz(&reconstruction, 2, 0.0);
    expect_tz(&reconstruction, 3, 5.0);
    expect_tz(&reconstruction, 4, -3.75);
    expect_tz(&reconstruction, 5, -2.5);
    expect_tz(&reconstruction, 6, 2.5);
    expect_tz(&reconstruction, 7, 3.75);
}

fn expect_vec_near(actual: Vector3d, expected: [f64; 3]) {
    for (a, e) in [actual.x, actual.y, actual.z].into_iter().zip(expected) {
        assert!((a - e).abs() <= 1e-6, "{actual:?} vs {expected:?}");
    }
}

#[test]
fn reconstruction_compute_bounds_and_centroid_empty() {
    let reconstruction = Reconstruction::new();
    let centroid = reconstruction.compute_centroid(0.0, 1.0, false).unwrap();
    let bbox = reconstruction
        .compute_bounding_box(0.0, 1.0, false)
        .unwrap();
    expect_vec_near(centroid, [0.0, 0.0, 0.0]);
    expect_vec_near(bbox.min, [0.0, 0.0, 0.0]);
    expect_vec_near(bbox.max, [0.0, 0.0, 0.0]);
}

#[test]
fn reconstruction_compute_bounds_and_centroid() {
    let mut reconstruction = Reconstruction::new();
    let black = Vector3ub::new(0, 0, 0);
    reconstruction
        .add_point3d(Vector3d::new(3.0, 0.0, 0.0), Track::new(), black)
        .unwrap();
    reconstruction
        .add_point3d(Vector3d::new(0.0, 3.0, 0.0), Track::new(), black)
        .unwrap();
    reconstruction
        .add_point3d(Vector3d::new(0.0, 0.0, 3.0), Track::new(), black)
        .unwrap();
    let centroid = reconstruction.compute_centroid(0.0, 1.0, false).unwrap();
    let bbox = reconstruction
        .compute_bounding_box(0.0, 1.0, false)
        .unwrap();
    expect_vec_near(centroid, [1.0, 1.0, 1.0]);
    expect_vec_near(bbox.min, [0.0, 0.0, 0.0]);
    expect_vec_near(bbox.max, [3.0, 3.0, 3.0]);
}

fn add_observed_point(
    reconstruction: &mut Reconstruction,
    xyz: [f64; 3],
    image_id: ImageId,
    point2d_idx: u32,
) {
    let point_id = reconstruction
        .add_point3d(
            Vector3d::new(xyz[0], xyz[1], xyz[2]),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    reconstruction
        .add_observation(point_id, TrackElement::new(image_id, point2d_idx))
        .unwrap();
}

#[test]
fn reconstruction_crop() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(3, &mut reconstruction);
    add_observed_point(&mut reconstruction, [0.0, 0.0, 0.0], 1, 1);
    add_observed_point(&mut reconstruction, [0.5, 0.5, 0.0], 1, 2);
    add_observed_point(&mut reconstruction, [1.0, 1.0, 0.0], 2, 3);
    add_observed_point(&mut reconstruction, [0.0, 0.0, 0.5], 2, 4);
    add_observed_point(&mut reconstruction, [0.5, 0.5, 1.0], 3, 5);

    // Check correct reconstruction setup
    assert_eq!(reconstruction.num_cameras(), 1);
    assert_eq!(reconstruction.num_images(), 3);
    assert_eq!(reconstruction.num_reg_frames(), 3);
    assert_eq!(reconstruction.num_points3d(), 5);

    // Test emtpy reconstruction after cropping.
    let cropped1 = reconstruction
        .crop(&AlignedBox3d::new(
            Vector3d::new(-1.0, -1.0, -1.0),
            Vector3d::new(-0.5, -0.5, -0.5),
        ))
        .unwrap();
    assert_eq!(cropped1.num_cameras(), 1);
    assert_eq!(cropped1.num_images(), 3);
    assert_eq!(cropped1.num_reg_frames(), 0);
    assert_eq!(cropped1.num_points3d(), 0);

    // Test reconstruction with contents after cropping
    let cropped2 = reconstruction
        .crop(&AlignedBox3d::new(
            Vector3d::new(0.0, 0.0, 0.0),
            Vector3d::new(0.75, 0.75, 0.75),
        ))
        .unwrap();
    assert_eq!(cropped2.num_cameras(), 1);
    assert_eq!(cropped2.num_images(), 3);
    assert_eq!(cropped2.num_reg_frames(), 2);
    assert_eq!(cropped2.num_points3d(), 3);
    let has_pose = |image_id: ImageId| {
        let image = cropped2.image(image_id).unwrap();
        image.has_pose(Some(cropped2.frame(image.frame_id()).unwrap()))
    };
    assert!(has_pose(1));
    assert!(has_pose(2));
    assert!(!has_pose(3));
}

#[test]
fn reconstruction_transform() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(3, &mut reconstruction);
    let point3d_id = reconstruction
        .add_point3d(
            Vector3d::new(1.0, 1.0, 1.0),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )
        .unwrap();
    reconstruction
        .add_observation(point3d_id, TrackElement::new(1, 1))
        .unwrap();
    reconstruction
        .add_observation(point3d_id, TrackElement::new(2, 1))
        .unwrap();
    reconstruction.transform(&Sim3d::new(
        2.0,
        Quaterniond::identity(),
        Vector3d::new(0.0, 1.0, 2.0),
    ));
    let image = reconstruction.image(1).unwrap();
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    assert_eq!(
        image.projection_center(frame, rig).unwrap(),
        Vector3d::new(0.0, 1.0, 2.0)
    );
    assert_eq!(
        reconstruction.point3d(point3d_id).unwrap().xyz,
        Vector3d::new(2.0, 3.0, 4.0)
    );
}
