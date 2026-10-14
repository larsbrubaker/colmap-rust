// Rust-only tests of colmap_rust::scene::database: the SQLite behavior the in-memory store
// reproduces (AUTOINCREMENT ids, UNIQUE indices, foreign keys and cascades, the deprecated
// pose prior API, the keypoint blob widths) and the all-or-nothing writes of
// docs/CPP_DIVERGENCES.md entry 266. database_test.cc covers none of these directly.

use colmap_rust::feature::types::{FeatureDescriptors, FeatureKeypoint};
use colmap_rust::geometry::pose_prior::PosePrior;
use colmap_rust::linalg::RowMajorMatrix;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::database_merge::feature_keypoints_from_blob;
use colmap_rust::scene::frame::Frame;
use colmap_rust::scene::image::Image;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::check::ErrorKind;
use colmap_rust::util::types::{DataId, SensorId, SensorType};

fn camera_sensor(id: u32) -> SensorId {
    SensorId::new(SensorType::Camera, id)
}

fn write_image(database: &Database, name: &str, camera_id: u32) -> u32 {
    let mut image = Image::new();
    image.set_name(name);
    image.set_camera_id(camera_id).unwrap();
    database.write_image(&image, false).unwrap()
}

#[test]
fn rust_only_auto_increment_ids_never_reuse_ids() {
    let database = Database::new();
    let mut camera = Camera::default();
    assert_eq!(database.write_camera(&camera, false).unwrap(), 1);
    camera.camera_id = 10;
    assert_eq!(database.write_camera(&camera, true).unwrap(), 10);
    assert_eq!(database.write_camera(&camera, false).unwrap(), 11);
    // Clearing a table does not reset its AUTOINCREMENT sequence.
    database.clear_cameras().unwrap();
    assert_eq!(database.write_camera(&camera, false).unwrap(), 12);
}

#[test]
fn rust_only_pose_prior_ids_follow_the_current_maximum() {
    let database = Database::new();
    let mut pose_prior = PosePrior {
        corr_data_id: DataId::new(camera_sensor(1), 1),
        ..PosePrior::default()
    };
    assert_eq!(database.write_pose_prior(&pose_prior, false).unwrap(), 1);
    database.clear_pose_priors().unwrap();
    // A plain INTEGER PRIMARY KEY restarts at max(rowid) + 1 = 1 once the table is empty.
    assert_eq!(database.write_pose_prior(&pose_prior, false).unwrap(), 1);
    pose_prior.corr_data_id = DataId::new(camera_sensor(1), 2);
    assert_eq!(database.write_pose_prior(&pose_prior, false).unwrap(), 2);
}

#[test]
fn rust_only_deprecated_pose_prior_api_fails() {
    let database = Database::new();
    let error = database.exists_pose_prior(1, true).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RuntimeError);
    assert!(error.message().starts_with("PosePrior API has changed"));
    assert!(database.read_pose_prior(1, true).is_err());
    assert!(!database.exists_pose_prior(1, false).unwrap());
}

#[test]
fn rust_only_image_names_are_unique_and_need_a_camera() {
    let database = Database::new();
    let camera_id = database.write_camera(&Camera::default(), false).unwrap();
    write_image(&database, "a", camera_id);
    let mut image = Image::new();
    image.set_name("a");
    image.set_camera_id(camera_id).unwrap();
    let error = database.write_image(&image, false).unwrap_err();
    assert_eq!(
        error.message(),
        "SQLite error: UNIQUE constraint failed: images.name"
    );
    image.set_name("b");
    image.set_camera_id(camera_id + 1).unwrap();
    let error = database.write_image(&image, false).unwrap_err();
    assert_eq!(
        error.message(),
        "SQLite error: FOREIGN KEY constraint failed"
    );
    // Neither failed write stored a row or consumed an id.
    assert_eq!(database.num_images().unwrap(), 1);
    image.set_camera_id(camera_id).unwrap();
    assert_eq!(database.write_image(&image, false).unwrap(), 2);
}

#[test]
fn rust_only_failed_rig_write_stores_nothing() {
    let database = Database::new();
    let mut rig = Rig::new();
    rig.add_ref_sensor(camera_sensor(1)).unwrap();
    rig.add_sensor(camera_sensor(2), None).unwrap();
    assert_eq!(database.write_rig(&rig, false).unwrap(), 1);
    // camera 2 already is a non-reference sensor of rig 1.
    let mut rig2 = Rig::new();
    rig2.add_ref_sensor(camera_sensor(3)).unwrap();
    rig2.add_sensor(camera_sensor(2), None).unwrap();
    assert!(database.write_rig(&rig2, false).is_err());
    assert_eq!(database.num_rigs().unwrap(), 1);
    let mut rig3 = Rig::new();
    rig3.add_ref_sensor(camera_sensor(3)).unwrap();
    assert_eq!(database.write_rig(&rig3, false).unwrap(), 2);
}

#[test]
fn rust_only_frame_data_belongs_to_one_frame() {
    let database = Database::new();
    let mut rig = Rig::new();
    rig.add_ref_sensor(camera_sensor(1)).unwrap();
    let rig_id = database.write_rig(&rig, false).unwrap();
    let mut frame = Frame::new();
    frame.set_rig_id(rig_id).unwrap();
    frame.add_data_id(DataId::new(camera_sensor(1), 7)).unwrap();
    database.write_frame(&frame, false).unwrap();
    assert!(database.write_frame(&frame, false).is_err());
    assert_eq!(database.num_frames().unwrap(), 1);
    // A frame needs an existing rig.
    let mut orphan = Frame::new();
    orphan.set_rig_id(rig_id + 1).unwrap();
    assert!(database.write_frame(&orphan, false).is_err());
    // Clearing the rigs cascades to their frames.
    database.clear_rigs().unwrap();
    assert_eq!(database.num_frames().unwrap(), 0);
}

#[test]
fn rust_only_clearing_images_cascades_and_cameras_need_no_images() {
    let database = Database::new();
    let camera_id = database.write_camera(&Camera::default(), false).unwrap();
    let image_id = write_image(&database, "a", camera_id);
    database
        .write_keypoints(image_id, &[FeatureKeypoint::default()])
        .unwrap();
    database
        .write_descriptors(image_id, &FeatureDescriptors::default())
        .unwrap();
    // images.camera_id references the cameras without a cascade.
    assert!(database.clear_cameras().is_err());
    assert_eq!(database.num_cameras().unwrap(), 1);
    database.clear_images().unwrap();
    assert!(!database.exists_keypoints(image_id).unwrap());
    assert!(!database.exists_descriptors(image_id).unwrap());
    database.clear_cameras().unwrap();
    // Keypoints need an existing image.
    assert!(database
        .write_keypoints(image_id, &[FeatureKeypoint::default()])
        .is_err());
}

#[test]
fn rust_only_keypoint_blob_widths() {
    let blob2 = RowMajorMatrix::from_vec(1, 2, vec![1.0f32, 2.0]);
    assert_eq!(
        feature_keypoints_from_blob(&blob2).unwrap(),
        vec![FeatureKeypoint::new(1.0, 2.0)]
    );
    let blob4 = RowMajorMatrix::from_vec(1, 4, vec![1.0f32, 2.0, 3.0, 0.0]);
    assert_eq!(
        feature_keypoints_from_blob(&blob4).unwrap(),
        vec![FeatureKeypoint::from_scale_orientation(1.0, 2.0, 3.0, 0.0).unwrap()]
    );
    let blob6 = RowMajorMatrix::from_vec(1, 6, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(
        feature_keypoints_from_blob(&blob6).unwrap(),
        vec![FeatureKeypoint::from_affine(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
    );
    let blob3 = RowMajorMatrix::from_vec(1, 3, vec![1.0f32, 2.0, 3.0]);
    assert!(feature_keypoints_from_blob(&blob3).is_err());
}

#[test]
fn rust_only_update_of_missing_rows_is_a_no_op() {
    let database = Database::new();
    let camera = Camera {
        camera_id: 5,
        ..Camera::default()
    };
    database.update_camera(&camera).unwrap();
    assert_eq!(database.num_cameras().unwrap(), 0);
    let mut image = Image::new();
    image.set_image_id(3);
    image.set_name("x");
    database.update_image(&image).unwrap();
    assert_eq!(database.num_images().unwrap(), 0);
}
