// Port of colmap/scene/database_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::scene::database (the in-memory Database, its writes and Merge /
// LoadRandomDatabaseDescriptors). The `ParameterizedDatabaseTests` cases run on the single
// in-memory implementation and are named `parameterized_database_tests_<case>`;
// `GetParam()(kInMemorySqliteDatabasePath)` is `Database::new()`.
//
// Skipped (SQLite database files are out of scope, PORTING_PLAN.md): OpenFile,
// OpenCloseFile, OpenFileWithNonASCIIPath.
//
// COLMAP's gtest main seeds the PRNG with 0 before every test, so tests drawing random poses
// or matrices call `set_prng_seed(0)` first. `FeatureDescriptorsData::Random` draws Eigen's
// `std::rand` bytes, whose values these checks do not depend on; `random_bytes` draws them
// from the seeded PRNG. EXPECT_ANY_THROW is `is_err()`.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use colmap_rust::feature::types::{
    FeatureDescriptors, FeatureDescriptorsData, FeatureExtractorType, FeatureKeypoint, FeatureMatch,
};
use colmap_rust::geometry::pose_prior::{CoordinateSystem, PosePrior};
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Matrix3d, RowMajorMatrix, Vector3d};
use colmap_rust::math::random::{random_uniform_integer, set_prng_seed};
use colmap_rust::math::random_eigen::{
    random_eigen_matrix3d, random_eigen_quaterniond, random_eigen_vector3d,
};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::database::{Database, DatabaseTransaction};
use colmap_rust::scene::frame::Frame;
use colmap_rust::scene::image::Image;
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{
    image_pair_to_pair_id, DataId, ImageId, SensorId, SensorType, INVALID_CAMERA_ID,
};

fn random_rigid3d() -> Rigid3d {
    Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d())
}

fn simple_pinhole() -> Camera {
    Camera::create_from_model_id(INVALID_CAMERA_ID, CameraModelId::SimplePinhole, 1.0, 1, 1)
        .unwrap()
}

/// `FeatureDescriptorsData::Random(rows, cols)`: arbitrary bytes, from the seeded PRNG.
fn random_bytes(rows: usize, cols: usize) -> FeatureDescriptorsData {
    let data = (0..rows * cols)
        .map(|_| random_uniform_integer::<u16>(0, 255) as u8)
        .collect();
    RowMajorMatrix::from_vec(rows, cols, data)
}

fn keypoints(n: usize) -> Vec<FeatureKeypoint> {
    vec![FeatureKeypoint::default(); n]
}

fn matches(n: usize) -> Vec<FeatureMatch> {
    vec![FeatureMatch::default(); n]
}

fn camera_sensor(id: u32) -> SensorId {
    SensorId::new(SensorType::Camera, id)
}

fn imu_sensor(id: u32) -> SensorId {
    SensorId::new(SensorType::Imu, id)
}

#[test]
fn parameterized_database_tests_open_in_memory() {
    let _database = Database::new();
}

#[test]
fn parameterized_database_tests_open_close_in_memory() {
    let database = Database::new();
    database.close();
    // Any database operation after closing the database should fail.
    assert!(database.exists_camera(42).is_err());
    database.close();
}

#[test]
fn parameterized_database_tests_transaction() {
    let database = Database::new();
    let _database_transaction = DatabaseTransaction::new(&database).unwrap();
}

#[test]
fn parameterized_database_tests_transaction_multi_threaded() {
    let database = Arc::new(Database::new());

    const NUM_THREADS: usize = 3;
    let mut threads = Vec::with_capacity(NUM_THREADS);
    for _ in 0..NUM_THREADS {
        let database = Arc::clone(&database);
        threads.push(thread::spawn(move || {
            let _database_transaction = DatabaseTransaction::new(&database).unwrap();
            thread::sleep(Duration::from_millis(10));
        }));
    }

    for thread in threads {
        thread.join().unwrap();
    }
}

#[test]
fn parameterized_database_tests_empty() {
    let database = Database::new();
    assert_eq!(database.num_cameras().unwrap(), 0);
    assert_eq!(database.num_frames().unwrap(), 0);
    assert_eq!(database.num_images().unwrap(), 0);
    assert_eq!(database.num_keypoints().unwrap(), 0);
    assert_eq!(database.max_num_keypoints().unwrap(), 0);
    assert_eq!(database.num_descriptors().unwrap(), 0);
    assert_eq!(database.max_num_descriptors().unwrap(), 0);
    assert_eq!(database.num_matches().unwrap(), 0);
    assert_eq!(database.num_matched_image_pairs().unwrap(), 0);
    assert_eq!(database.num_verified_image_pairs().unwrap(), 0);
}

#[test]
fn parameterized_database_tests_rig() {
    set_prng_seed(0);
    let database = Database::new();
    assert_eq!(database.num_rigs().unwrap(), 0);
    let mut rig = Rig::new();
    rig.add_ref_sensor(camera_sensor(1)).unwrap();
    rig.set_rig_id(database.write_rig(&rig, false).unwrap());
    assert_eq!(database.num_rigs().unwrap(), 1);
    assert!(database.exists_rig(rig.rig_id()).unwrap());
    assert_eq!(database.read_rig(rig.rig_id()).unwrap(), rig);

    database.clear_rigs().unwrap();
    assert_eq!(database.num_rigs().unwrap(), 0);

    rig.add_sensor(camera_sensor(2), Some(random_rigid3d()))
        .unwrap();
    rig.add_sensor(imu_sensor(3), None).unwrap();
    rig.add_sensor(imu_sensor(4), Some(random_rigid3d()))
        .unwrap();
    rig.set_rig_id(database.write_rig(&rig, false).unwrap());
    assert_eq!(database.num_rigs().unwrap(), 1);
    assert!(database.exists_rig(rig.rig_id()).unwrap());
    assert_eq!(database.read_rig(rig.rig_id()).unwrap(), rig);
    assert_eq!(
        database.read_rig_with_sensor(camera_sensor(1)).unwrap(),
        Some(rig.clone())
    );
    assert_eq!(
        database.read_rig_with_sensor(imu_sensor(4)).unwrap(),
        Some(rig.clone())
    );
    assert_eq!(database.read_rig_with_sensor(imu_sensor(42)).unwrap(), None);
    *rig.sensor_from_rig_mut(camera_sensor(2)).unwrap() = random_rigid3d();
    database.update_rig(&rig).unwrap();
    assert_eq!(database.read_rig(rig.rig_id()).unwrap(), rig);
    let mut rig2 = Rig::new();
    rig2.add_ref_sensor(imu_sensor(10)).unwrap();
    rig2.set_rig_id(rig.rig_id() + 1);
    database.write_rig(&rig2, /*use_rig_id=*/ true).unwrap();
    assert_eq!(database.num_rigs().unwrap(), 2);
    assert!(database.exists_rig(rig.rig_id()).unwrap());
    assert!(database.exists_rig(rig2.rig_id()).unwrap());
    assert_eq!(database.read_all_rigs().unwrap().len(), 2);
    assert_eq!(database.read_all_rigs().unwrap()[0].rig_id(), rig.rig_id());
    assert_eq!(database.read_all_rigs().unwrap()[1].rig_id(), rig2.rig_id());
    database.clear_rigs().unwrap();
    assert_eq!(database.num_rigs().unwrap(), 0);
}

#[test]
fn parameterized_database_tests_camera() {
    let database = Database::new();
    assert_eq!(database.num_cameras().unwrap(), 0);
    let mut camera = simple_pinhole();
    camera.camera_id = database.write_camera(&camera, false).unwrap();
    assert_eq!(database.num_cameras().unwrap(), 1);
    assert!(database.exists_camera(camera.camera_id).unwrap());
    assert_eq!(
        database.read_camera(camera.camera_id).unwrap().camera_id,
        camera.camera_id
    );
    assert_eq!(
        database.read_camera(camera.camera_id).unwrap().model_id,
        camera.model_id
    );
    assert_eq!(database.read_camera(camera.camera_id).unwrap(), camera);
    camera
        .set_focal_length(2.0 * camera.focal_length().unwrap())
        .unwrap();
    database.update_camera(&camera).unwrap();
    assert_eq!(database.read_camera(camera.camera_id).unwrap(), camera);
    let mut camera2 = camera.clone();
    camera2.camera_id = camera.camera_id + 1;
    database.write_camera(&camera2, true).unwrap();
    assert_eq!(database.num_cameras().unwrap(), 2);
    assert!(database.exists_camera(camera.camera_id).unwrap());
    assert!(database.exists_camera(camera2.camera_id).unwrap());
    assert_eq!(database.read_all_cameras().unwrap().len(), 2);
    assert_eq!(database.read_all_cameras().unwrap(), vec![camera, camera2]);
    database.clear_cameras().unwrap();
    assert_eq!(database.num_cameras().unwrap(), 0);
}

#[test]
fn parameterized_database_tests_frame() {
    let database = Database::new();
    let mut rig = Rig::new();
    rig.add_ref_sensor(camera_sensor(1)).unwrap();
    rig.set_rig_id(database.write_rig(&rig, false).unwrap());
    assert_eq!(database.num_frames().unwrap(), 0);

    let mut frame = Frame::new();
    frame.set_rig_id(rig.rig_id()).unwrap();
    frame.set_frame_id(database.write_frame(&frame, false).unwrap());
    assert_eq!(database.num_frames().unwrap(), 1);
    assert!(database.exists_frame(frame.frame_id()).unwrap());
    assert_eq!(database.read_frame(frame.frame_id()).unwrap(), frame);

    database.clear_frames().unwrap();
    assert_eq!(database.num_frames().unwrap(), 0);

    frame.add_data_id(DataId::new(imu_sensor(1), 2)).unwrap();
    frame.add_data_id(DataId::new(camera_sensor(1), 3)).unwrap();
    frame.set_frame_id(database.write_frame(&frame, false).unwrap());
    assert_eq!(database.num_frames().unwrap(), 1);
    assert!(database.exists_frame(frame.frame_id()).unwrap());
    assert_eq!(database.read_frame(frame.frame_id()).unwrap(), frame);

    frame.add_data_id(DataId::new(camera_sensor(2), 4)).unwrap();
    database.update_frame(&frame).unwrap();
    assert_eq!(database.read_frame(frame.frame_id()).unwrap(), frame);
    let mut frame2 = Frame::new();
    frame2.set_rig_id(rig.rig_id()).unwrap();
    frame2
        .add_data_id(DataId::new(camera_sensor(2), 5))
        .unwrap();
    frame2.set_frame_id(frame.frame_id() + 1);
    database
        .write_frame(&frame2, /*use_frame_id=*/ true)
        .unwrap();
    assert_eq!(database.num_frames().unwrap(), 2);
    assert!(database.exists_frame(frame.frame_id()).unwrap());
    assert!(database.exists_frame(frame2.frame_id()).unwrap());
    assert_eq!(database.read_all_frames().unwrap().len(), 2);
    assert_eq!(
        database.read_all_frames().unwrap()[0].frame_id(),
        frame.frame_id()
    );
    assert_eq!(
        database.read_all_frames().unwrap()[1].frame_id(),
        frame2.frame_id()
    );

    database.clear_frames().unwrap();
    assert_eq!(database.num_frames().unwrap(), 0);
}

#[test]
fn parameterized_database_tests_image() {
    let database = Database::new();
    let mut camera = simple_pinhole();
    camera.camera_id = database.write_camera(&camera, false).unwrap();
    let mut rig = Rig::new();
    rig.add_ref_sensor(camera_sensor(camera.camera_id)).unwrap();
    rig.set_rig_id(database.write_rig(&rig, false).unwrap());
    assert_eq!(database.num_images().unwrap(), 0);
    let mut image = Image::new();
    image.set_name("test");
    image.set_camera_id(camera.camera_id).unwrap();
    image.set_image_id(database.write_image(&image, false).unwrap());
    let mut frame = Frame::new();
    frame.set_rig_id(rig.rig_id()).unwrap();
    frame.add_data_id(image.data_id()).unwrap();
    frame.set_frame_id(database.write_frame(&frame, false).unwrap());
    image.set_frame_id(frame.frame_id());
    assert_eq!(database.num_images().unwrap(), 1);
    assert!(database.exists_image(image.image_id()).unwrap());
    assert_eq!(database.read_image(image.image_id()).unwrap(), image);
    assert_eq!(
        database.read_image_with_name(image.name()).unwrap(),
        Some(image.clone())
    );
    assert_eq!(database.read_image_with_name("foobar").unwrap(), None);
    image.set_name("test_changed");
    database.update_image(&image).unwrap();
    assert_eq!(database.read_image(image.image_id()).unwrap(), image);
    let mut image2 = image.clone();
    image2.set_name("test2");
    image2.set_image_id(image.image_id() + 1);
    frame.add_data_id(image2.data_id()).unwrap();
    database.update_frame(&frame).unwrap();
    assert_eq!(
        database
            .write_image(&image2, /*use_image_id=*/ true)
            .unwrap(),
        image2.image_id()
    );
    assert_eq!(database.num_images().unwrap(), 2);
    assert!(database.exists_image(image.image_id()).unwrap());
    assert!(database.exists_image(image2.image_id()).unwrap());
    assert_eq!(database.read_all_images().unwrap(), vec![image, image2]);
    database.clear_images().unwrap();
    assert_eq!(database.num_images().unwrap(), 0);
}

#[test]
fn parameterized_database_tests_pose_prior() {
    set_prng_seed(0);
    let database = Database::new();
    let mut camera = Camera::default();
    camera.camera_id = database.write_camera(&camera, false).unwrap();
    let mut image = Image::new();
    image.set_camera_id(camera.camera_id).unwrap();
    assert_eq!(database.num_pose_priors().unwrap(), 0);
    let mut pose_prior = PosePrior {
        corr_data_id: image.data_id(),
        position: Vector3d::new(0.1, 0.2, 0.3),
        position_covariance: random_eigen_matrix3d(),
        coordinate_system: CoordinateSystem::Cartesian,
        gravity: random_eigen_vector3d(),
        ..PosePrior::default()
    };
    pose_prior.pose_prior_id = database.write_pose_prior(&pose_prior, false).unwrap();
    assert!(database.write_pose_prior(&pose_prior, false).is_err());
    assert_eq!(database.num_pose_priors().unwrap(), 1);
    assert_eq!(
        database
            .read_pose_prior(
                pose_prior.pose_prior_id,
                /*is_deprecated_image_prior=*/ false
            )
            .unwrap(),
        pose_prior
    );
    pose_prior.position_covariance = Matrix3d::identity();
    database.update_pose_prior(&pose_prior).unwrap();
    assert_eq!(
        database
            .read_pose_prior(
                pose_prior.pose_prior_id,
                /*is_deprecated_image_prior=*/ false
            )
            .unwrap(),
        pose_prior
    );
    assert_eq!(database.read_all_pose_priors().unwrap(), vec![pose_prior]);
    database.clear_pose_priors().unwrap();
    assert_eq!(database.num_pose_priors().unwrap(), 0);
}

/// A database with one default camera and one image named `name`; returns the image.
fn database_with_image(database: &Database, name: &str) -> Image {
    let mut camera = Camera::default();
    camera.camera_id = database.write_camera(&camera, false).unwrap();
    let mut image = Image::new();
    image.set_name(name);
    image.set_camera_id(camera.camera_id).unwrap();
    image.set_image_id(database.write_image(&image, false).unwrap());
    image
}

#[test]
fn parameterized_database_tests_keypoints() {
    let database = Database::new();
    let mut image = database_with_image(&database, "test");
    assert_eq!(database.num_keypoints().unwrap(), 0);
    assert_eq!(
        database.num_keypoints_for_image(image.image_id()).unwrap(),
        0
    );
    let keypoints1 = keypoints(10);
    database
        .write_keypoints(image.image_id(), &keypoints1)
        .unwrap();
    assert_eq!(
        keypoints1,
        database.read_keypoints(image.image_id()).unwrap()
    );
    assert_eq!(database.num_keypoints().unwrap(), 10);
    assert_eq!(database.max_num_keypoints().unwrap(), 10);
    assert_eq!(
        database.num_keypoints_for_image(image.image_id()).unwrap(),
        10
    );
    let mut keypoints2 = keypoints(20);
    image.set_name("test2");
    image.set_image_id(database.write_image(&image, false).unwrap());
    database
        .write_keypoints(image.image_id(), &keypoints2)
        .unwrap();
    assert_eq!(
        keypoints2,
        database.read_keypoints(image.image_id()).unwrap()
    );
    assert_eq!(database.num_keypoints().unwrap(), 30);
    assert_eq!(database.max_num_keypoints().unwrap(), 20);
    assert_eq!(
        database.num_keypoints_for_image(image.image_id()).unwrap(),
        20
    );
    keypoints2[0].x += 1.0;
    database
        .update_keypoints(image.image_id(), &keypoints2)
        .unwrap();
    assert_eq!(
        keypoints2,
        database.read_keypoints(image.image_id()).unwrap()
    );
    database.clear_keypoints().unwrap();
    assert_eq!(database.num_keypoints().unwrap(), 0);
    assert_eq!(database.max_num_keypoints().unwrap(), 0);
    assert_eq!(
        database.num_keypoints_for_image(image.image_id()).unwrap(),
        0
    );
}

#[test]
fn parameterized_database_tests_read_keypoints_empty() {
    let database = Database::new();
    let image = database_with_image(&database, "test");
    // Reading keypoints for an image with no keypoints should return empty.
    let keypoints = database.read_keypoints(image.image_id()).unwrap();
    assert!(keypoints.is_empty());
}

#[test]
fn parameterized_database_tests_descriptors() {
    set_prng_seed(0);
    let database = Database::new();
    let mut image = database_with_image(&database, "test");
    assert_eq!(database.num_descriptors().unwrap(), 0);
    assert_eq!(
        database
            .num_descriptors_for_image(image.image_id())
            .unwrap(),
        0
    );
    let descriptors = FeatureDescriptors::new(FeatureExtractorType::Sift, random_bytes(10, 128));
    database
        .write_descriptors(image.image_id(), &descriptors)
        .unwrap();
    let descriptors_read = database.read_descriptors(image.image_id()).unwrap();
    assert_eq!(descriptors.data.rows(), descriptors_read.data.rows());
    assert_eq!(descriptors.data.cols(), descriptors_read.data.cols());
    assert_eq!(descriptors.extractor_type, descriptors_read.extractor_type);
    for r in 0..descriptors.data.rows() {
        for c in 0..descriptors.data.cols() {
            assert_eq!(descriptors.data[(r, c)], descriptors_read.data[(r, c)]);
        }
    }
    assert_eq!(database.num_descriptors().unwrap(), 10);
    assert_eq!(database.max_num_descriptors().unwrap(), 10);
    assert_eq!(
        database
            .num_descriptors_for_image(image.image_id())
            .unwrap(),
        10
    );
    let descriptors2 = FeatureDescriptors::new(
        FeatureExtractorType::Undefined,
        FeatureDescriptorsData::new(20, 128),
    );
    image.set_name("test2");
    image.set_image_id(database.write_image(&image, false).unwrap());
    database
        .write_descriptors(image.image_id(), &descriptors2)
        .unwrap();
    let descriptors2_read = database.read_descriptors(image.image_id()).unwrap();
    assert_eq!(
        descriptors2.extractor_type,
        descriptors2_read.extractor_type
    );
    assert_eq!(database.num_descriptors().unwrap(), 30);
    assert_eq!(database.max_num_descriptors().unwrap(), 20);
    assert_eq!(
        database
            .num_descriptors_for_image(image.image_id())
            .unwrap(),
        20
    );
    database.clear_descriptors().unwrap();
    assert_eq!(database.num_descriptors().unwrap(), 0);
    assert_eq!(database.max_num_descriptors().unwrap(), 0);
    assert_eq!(
        database
            .num_descriptors_for_image(image.image_id())
            .unwrap(),
        0
    );
}

#[test]
fn parameterized_database_tests_descriptor_feature_type_default() {
    // Test that descriptors written with UNDEFINED type are read back correctly,
    // and that the database default (for migration) is SIFT (0).
    let database = Database::new();
    let mut image = database_with_image(&database, "test");

    // Write descriptors with UNDEFINED type (default).
    let descriptors = FeatureDescriptors {
        data: FeatureDescriptorsData::new(5, 128),
        ..FeatureDescriptors::default()
    };
    assert_eq!(descriptors.extractor_type, FeatureExtractorType::Undefined);
    database
        .write_descriptors(image.image_id(), &descriptors)
        .unwrap();

    // Read back and verify the type is preserved.
    let descriptors_read = database.read_descriptors(image.image_id()).unwrap();
    assert_eq!(
        descriptors_read.extractor_type,
        FeatureExtractorType::Undefined
    );

    // Write another image with SIFT type.
    image.set_name("test2");
    image.set_image_id(database.write_image(&image, false).unwrap());
    let descriptors_sift = FeatureDescriptors::new(
        FeatureExtractorType::Sift,
        FeatureDescriptorsData::new(5, 128),
    );
    database
        .write_descriptors(image.image_id(), &descriptors_sift)
        .unwrap();

    let descriptors_sift_read = database.read_descriptors(image.image_id()).unwrap();
    assert_eq!(
        descriptors_sift_read.extractor_type,
        FeatureExtractorType::Sift
    );
}

#[test]
fn parameterized_database_tests_matches() {
    let database = Database::new();
    let image_id1: ImageId = 1;
    let image_id2: ImageId = 2;
    const NUM_MATCHES: usize = 1000;
    let mut matches12 = matches(NUM_MATCHES);
    let mut matches21 = matches(NUM_MATCHES);
    for i in 0..matches12.len() {
        matches12[i].point2d_idx1 = i as u32;
        matches12[i].point2d_idx2 = 10000 + i as u32;
        matches21[i].point2d_idx1 = 10000 + i as u32;
        matches21[i].point2d_idx2 = i as u32;
    }

    let expect_valid_matches = |database: &Database| {
        assert_eq!(database.num_matched_image_pairs().unwrap(), 1);
        let matches_read12 = database.read_matches(image_id1, image_id2).unwrap();
        assert_eq!(matches12.len(), matches_read12.len());
        for i in 0..matches12.len() {
            assert_eq!(matches12[i].point2d_idx1, matches_read12[i].point2d_idx1);
            assert_eq!(matches12[i].point2d_idx2, matches_read12[i].point2d_idx2);
        }
        let matches_read21 = database.read_matches(image_id2, image_id1).unwrap();
        assert_eq!(matches12.len(), matches_read21.len());
        for i in 0..matches12.len() {
            assert_eq!(matches12[i].point2d_idx1, matches_read21[i].point2d_idx2);
            assert_eq!(matches12[i].point2d_idx2, matches_read21[i].point2d_idx1);
        }
    };

    assert_eq!(database.num_matched_image_pairs().unwrap(), 0);
    database
        .write_matches(image_id1, image_id2, &matches12)
        .unwrap();
    expect_valid_matches(&database);
    database.delete_matches(image_id1, image_id2).unwrap();
    assert_eq!(database.num_matched_image_pairs().unwrap(), 0);
    database
        .write_matches(image_id2, image_id1, &matches21)
        .unwrap();
    expect_valid_matches(&database);

    let pair_id12 = image_pair_to_pair_id(image_id1, image_id2).unwrap();
    assert_eq!(database.read_all_matches_blob().unwrap().len(), 1);
    assert_eq!(database.read_all_matches_blob().unwrap()[0].0, pair_id12);
    let matches = database.read_all_matches().unwrap();
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].0, pair_id12);
    let pair_ids_and_num_matches = database.read_num_matches().unwrap();
    assert_eq!(pair_ids_and_num_matches.len(), 1);
    assert_eq!(pair_ids_and_num_matches[0].0, pair_id12);
    assert_eq!(pair_ids_and_num_matches[0].1 as usize, matches[0].1.len());
    assert_eq!(database.num_matches().unwrap(), NUM_MATCHES);
    database.delete_matches(image_id1, image_id2).unwrap();
    assert_eq!(database.num_matches().unwrap(), 0);
    database
        .write_matches(image_id1, image_id2, &matches12)
        .unwrap();
    assert_eq!(database.num_matches().unwrap(), NUM_MATCHES);
    database.clear_matches().unwrap();
    assert_eq!(database.num_matches().unwrap(), 0);
}

#[path = "database_pairs.rs"]
mod database_pairs;
