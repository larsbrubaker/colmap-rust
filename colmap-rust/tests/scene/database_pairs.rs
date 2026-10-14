// The second half of the database_test.cc port (tests/scene/database.rs, whose header holds
// the conventions): the two-view geometry cases, Merge and the
// LoadRandomDatabaseDescriptorsTest suite.

use colmap_rust::feature::types::{
    FeatureDescriptors, FeatureExtractorType, FeatureKeypoint, FeatureMatch,
};
use colmap_rust::geometry::pose_prior::PosePrior;
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{random_eigen_matrix3d, random_eigen_vector3d};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::database_merge::load_random_database_descriptors;
use colmap_rust::scene::frame::Frame;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{image_pair_to_pair_id, DataId, ImageId, SensorType};

use super::{keypoints, matches, random_bytes, random_rigid3d, simple_pinhole};

#[test]
fn parameterized_database_tests_two_view_geometry() {
    set_prng_seed(0);
    let database = Database::new();
    let image_id1: ImageId = 1;
    let image_id2: ImageId = 2;
    let mut two_view_geometry = TwoViewGeometry {
        inlier_matches: vec![FeatureMatch::default(); 1000],
        config: ConfigurationType::PlanarOrPanoramic,
        f: Some(random_eigen_matrix3d()),
        e: Some(random_eigen_matrix3d()),
        h: Some(random_eigen_matrix3d()),
        cam2_from_cam1: Some(random_rigid3d()),
        // Distinct cameras so the swap performed on inverse reads is observable.
        camera1: Some(
            Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 100.0, 640, 480).unwrap(),
        ),
        camera2: Some(
            Camera::create_from_model_id(2, CameraModelId::Pinhole, 200.0, 800, 600).unwrap(),
        ),
        ..TwoViewGeometry::default()
    };
    database
        .write_two_view_geometry(image_id1, image_id2, &two_view_geometry)
        .unwrap();
    let two_view_geometry_read = database
        .read_two_view_geometry(image_id1, image_id2)
        .unwrap();
    assert_eq!(
        two_view_geometry.inlier_matches.len(),
        two_view_geometry_read.inlier_matches.len()
    );
    for (expected, read) in two_view_geometry
        .inlier_matches
        .iter()
        .zip(&two_view_geometry_read.inlier_matches)
    {
        assert_eq!(expected.point2d_idx1, read.point2d_idx1);
        assert_eq!(expected.point2d_idx2, read.point2d_idx2);
    }

    assert_eq!(two_view_geometry.config, two_view_geometry_read.config);
    assert_eq!(two_view_geometry.f, two_view_geometry_read.f);
    assert_eq!(two_view_geometry.e, two_view_geometry_read.e);
    assert_eq!(two_view_geometry.h, two_view_geometry_read.h);
    assert!(two_view_geometry.cam2_from_cam1.is_some());
    assert!(two_view_geometry_read.cam2_from_cam1.is_some());
    let pose = two_view_geometry.cam2_from_cam1.unwrap();
    let pose_read = two_view_geometry_read.cam2_from_cam1.unwrap();
    assert_eq!(pose.rotation.coeffs(), pose_read.rotation.coeffs());
    assert_eq!(pose.translation, pose_read.translation);
    assert_eq!(two_view_geometry.camera1, two_view_geometry_read.camera1);
    assert_eq!(two_view_geometry.camera2, two_view_geometry_read.camera2);

    let two_view_geometry_read_inv = database
        .read_two_view_geometry(image_id2, image_id1)
        .unwrap();
    assert_eq!(
        two_view_geometry_read_inv.inlier_matches.len(),
        two_view_geometry_read.inlier_matches.len()
    );
    for (inv, read) in two_view_geometry_read_inv
        .inlier_matches
        .iter()
        .zip(&two_view_geometry_read.inlier_matches)
    {
        assert_eq!(inv.point2d_idx2, read.point2d_idx1);
        assert_eq!(inv.point2d_idx1, read.point2d_idx2);
    }

    assert_eq!(
        two_view_geometry_read_inv.config,
        two_view_geometry_read.config
    );
    assert_eq!(
        two_view_geometry_read_inv.f.unwrap().transpose(),
        two_view_geometry_read.f.unwrap()
    );
    assert_eq!(
        two_view_geometry_read_inv.e.unwrap().transpose(),
        two_view_geometry_read.e.unwrap()
    );
    assert!(two_view_geometry_read_inv
        .h
        .unwrap()
        .inverse()
        .is_approx(two_view_geometry_read.h.unwrap()));
    assert!(two_view_geometry_read_inv.cam2_from_cam1.is_some());
    let pose_inv = two_view_geometry_read_inv.cam2_from_cam1.unwrap();
    assert!(pose_inv.rotation.is_approx(pose_read.inverse().rotation));
    assert!(pose_inv
        .translation
        .is_approx(pose_read.inverse().translation));
    // The inverse read swaps the two cameras.
    assert_eq!(
        two_view_geometry_read_inv.camera1,
        two_view_geometry.camera2
    );
    assert_eq!(
        two_view_geometry_read_inv.camera2,
        two_view_geometry.camera1
    );

    let two_view_geometries = database.read_two_view_geometries().unwrap();
    assert_eq!(two_view_geometries.len(), 1);
    assert_eq!(
        two_view_geometries[0].0,
        image_pair_to_pair_id(image_id1, image_id2).unwrap()
    );
    let all0 = &two_view_geometries[0].1;
    assert_eq!(two_view_geometry.config, all0.config);
    assert_eq!(two_view_geometry.f, all0.f);
    assert_eq!(two_view_geometry.e, all0.e);
    assert_eq!(two_view_geometry.h, all0.h);
    assert!(all0.cam2_from_cam1.is_some());
    assert_eq!(
        pose.rotation.coeffs(),
        all0.cam2_from_cam1.unwrap().rotation.coeffs()
    );
    assert_eq!(pose.translation, all0.cam2_from_cam1.unwrap().translation);
    assert_eq!(two_view_geometry.camera1, all0.camera1);
    assert_eq!(two_view_geometry.camera2, all0.camera2);
    assert_eq!(
        two_view_geometry.inlier_matches.len(),
        all0.inlier_matches.len()
    );
    let pair_ids_and_num_inliers = database.read_two_view_geometry_num_inliers().unwrap();
    assert_eq!(pair_ids_and_num_inliers.len(), 1);
    assert_eq!(
        pair_ids_and_num_inliers[0].0,
        image_pair_to_pair_id(image_id1, image_id2).unwrap()
    );
    assert_eq!(
        pair_ids_and_num_inliers[0].1 as usize,
        two_view_geometry.inlier_matches.len()
    );
    assert_eq!(database.num_inlier_matches().unwrap(), 1000);
    database
        .delete_inlier_matches(image_id1, image_id2)
        .unwrap();
    assert!(database
        .exists_two_view_geometry(image_id1, image_id2)
        .unwrap());
    assert_eq!(database.num_inlier_matches().unwrap(), 0);
    database
        .delete_two_view_geometry(image_id1, image_id2)
        .unwrap();
    assert!(!database
        .exists_two_view_geometry(image_id1, image_id2)
        .unwrap());
    assert_eq!(database.num_inlier_matches().unwrap(), 0);
    database
        .write_two_view_geometry(image_id1, image_id2, &two_view_geometry)
        .unwrap();
    assert!(database
        .write_two_view_geometry(image_id1, image_id2, &two_view_geometry)
        .is_err());
    assert_eq!(database.num_inlier_matches().unwrap(), 1000);
    database.clear_two_view_geometries().unwrap();
    assert_eq!(database.num_inlier_matches().unwrap(), 0);
    two_view_geometry.inlier_matches.clear();
    database
        .write_two_view_geometry(image_id1, image_id2, &two_view_geometry)
        .unwrap();
    assert_eq!(
        two_view_geometry.cam2_from_cam1,
        database
            .read_two_view_geometry(image_id1, image_id2)
            .unwrap()
            .cam2_from_cam1
    );

    // Test with E and F set, but H missing.
    database.clear_two_view_geometries().unwrap();
    let two_view_geometry_no_h = TwoViewGeometry {
        inlier_matches: vec![FeatureMatch::default(); 10],
        config: ConfigurationType::Calibrated,
        e: Some(random_eigen_matrix3d()),
        f: Some(random_eigen_matrix3d()),
        ..TwoViewGeometry::default()
    };
    database
        .write_two_view_geometry(image_id1, image_id2, &two_view_geometry_no_h)
        .unwrap();
    let two_view_geometry_no_h_read = database
        .read_two_view_geometry(image_id1, image_id2)
        .unwrap();
    assert!(two_view_geometry_no_h_read.e.is_some());
    assert!(two_view_geometry_no_h_read.f.is_some());
    assert_eq!(two_view_geometry_no_h.e, two_view_geometry_no_h_read.e);
    assert_eq!(two_view_geometry_no_h.f, two_view_geometry_no_h_read.f);
    assert!(two_view_geometry_no_h_read.h.is_none());
}

#[test]
fn parameterized_database_tests_two_view_geometry_without_cameras() {
    set_prng_seed(0);
    let database = Database::new();
    let image_id1: ImageId = 1;
    let image_id2: ImageId = 2;

    // A geometry that leaves the estimated cameras unset (as for configurations
    // that consume fixed intrinsics) round-trips them as nullopt.
    let two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        e: Some(random_eigen_matrix3d()),
        ..TwoViewGeometry::default()
    };
    assert!(two_view_geometry.camera1.is_none());
    assert!(two_view_geometry.camera2.is_none());
    database
        .write_two_view_geometry(image_id1, image_id2, &two_view_geometry)
        .unwrap();

    let read = database
        .read_two_view_geometry(image_id1, image_id2)
        .unwrap();
    assert!(read.camera1.is_none());
    assert!(read.camera2.is_none());

    let all = database.read_two_view_geometries().unwrap();
    assert_eq!(all.len(), 1);
    assert!(all[0].1.camera1.is_none());
    assert!(all[0].1.camera2.is_none());
}

#[test]
fn parameterized_database_tests_merge() {
    set_prng_seed(0);
    let database1 = Database::new();
    let database2 = Database::new();

    // This test intentionally uses custom, large, partially overlapping IDs from
    // rigs/frames/images/cameras which then require remapping of the IDs. This is
    // to ensure that the database can handle this case.

    let mut camera1 = simple_pinhole();
    camera1.camera_id = 50;
    database1
        .write_camera(&camera1, /*use_camera_id=*/ true)
        .unwrap();
    let mut camera2 = simple_pinhole();
    camera2.camera_id = 60;
    database1
        .write_camera(&camera2, /*use_camera_id=*/ true)
        .unwrap();
    let mut camera3 = simple_pinhole();
    camera3.camera_id = 55;
    database2
        .write_camera(&camera3, /*use_camera_id=*/ true)
        .unwrap();
    let mut camera4 = simple_pinhole();
    camera4.camera_id = 60;
    database2
        .write_camera(&camera4, /*use_camera_id=*/ true)
        .unwrap();

    let mut rig1 = Rig::new();
    rig1.set_rig_id(100);
    rig1.add_ref_sensor(camera1.sensor_id()).unwrap();
    rig1.add_sensor(camera2.sensor_id(), Some(Rigid3d::default()))
        .unwrap();
    database1.write_rig(&rig1, /*use_rig_id=*/ true).unwrap();

    let mut rig2 = Rig::new();
    rig2.set_rig_id(200);
    rig2.add_ref_sensor(camera3.sensor_id()).unwrap();
    rig2.add_sensor(camera4.sensor_id(), Some(Rigid3d::default()))
        .unwrap();
    database2.write_rig(&rig2, /*use_rig_id=*/ true).unwrap();

    let image_id1: ImageId = 300;
    let image_id2: ImageId = 400;
    let image_id3: ImageId = 350;
    let image_id4: ImageId = 400;

    let mut image = Image::new();
    for (database, image_id, camera, name) in [
        (&database1, image_id1, &camera1, "test1"),
        (&database1, image_id2, &camera2, "test2"),
        (&database2, image_id3, &camera3, "test3"),
        (&database2, image_id4, &camera4, "test4"),
    ] {
        image.set_image_id(image_id);
        image.set_camera_id(camera.camera_id).unwrap();
        image.set_name(name);
        database
            .write_image(&image, /*use_image_id=*/ true)
            .unwrap();
    }

    let mut frame1 = Frame::new();
    frame1.set_rig_id(rig1.rig_id()).unwrap();
    frame1
        .add_data_id(DataId::new(camera1.sensor_id(), image_id1))
        .unwrap();
    frame1
        .add_data_id(DataId::new(camera2.sensor_id(), image_id2))
        .unwrap();
    frame1.set_frame_id(1000);
    database1
        .write_frame(&frame1, /*use_frame_id=*/ true)
        .unwrap();
    let mut frame2 = Frame::new();
    frame2.set_rig_id(rig2.rig_id()).unwrap();
    frame2
        .add_data_id(DataId::new(camera3.sensor_id(), image_id3))
        .unwrap();
    frame2
        .add_data_id(DataId::new(camera4.sensor_id(), image_id4))
        .unwrap();
    frame2.set_frame_id(2000);
    database2
        .write_frame(&frame2, /*use_frame_id=*/ true)
        .unwrap();

    let pose_prior1 = PosePrior {
        corr_data_id: DataId::new(camera1.sensor_id(), image_id1),
        position: random_eigen_vector3d(),
        ..PosePrior::default()
    };
    // COLMAP stores the returned id in `pose_prior1`, which is not read again.
    database1.write_pose_prior(&pose_prior1, false).unwrap();

    let pose_prior2 = PosePrior {
        corr_data_id: DataId::new(camera3.sensor_id(), image_id3),
        position: random_eigen_vector3d(),
        ..PosePrior::default()
    };
    // COLMAP stores the returned id in `pose_prior2`, which is not read again.
    database2.write_pose_prior(&pose_prior2, false).unwrap();

    let keypoints_with_x = |n: usize, x: f32| {
        let mut kps: Vec<FeatureKeypoint> = keypoints(n);
        kps[0].x = x;
        kps
    };
    let keypoints1 = keypoints_with_x(10, 100.0);
    let keypoints2 = keypoints_with_x(20, 200.0);
    let keypoints3 = keypoints_with_x(30, 300.0);
    let keypoints4 = keypoints_with_x(40, 400.0);

    let descriptors1 =
        FeatureDescriptors::new(FeatureExtractorType::Undefined, random_bytes(10, 128));
    let descriptors2 =
        FeatureDescriptors::new(FeatureExtractorType::Undefined, random_bytes(20, 128));
    let descriptors3 =
        FeatureDescriptors::new(FeatureExtractorType::Undefined, random_bytes(30, 128));
    let descriptors4 =
        FeatureDescriptors::new(FeatureExtractorType::Undefined, random_bytes(40, 128));

    database1.write_keypoints(image_id1, &keypoints1).unwrap();
    database1.write_keypoints(image_id2, &keypoints2).unwrap();
    database2.write_keypoints(image_id3, &keypoints3).unwrap();
    database2.write_keypoints(image_id4, &keypoints4).unwrap();
    database1
        .write_descriptors(image_id1, &descriptors1)
        .unwrap();
    database1
        .write_descriptors(image_id2, &descriptors2)
        .unwrap();
    database2
        .write_descriptors(image_id3, &descriptors3)
        .unwrap();
    database2
        .write_descriptors(image_id4, &descriptors4)
        .unwrap();
    database1
        .write_matches(image_id1, image_id2, &matches(10))
        .unwrap();
    database2
        .write_matches(image_id3, image_id4, &matches(10))
        .unwrap();
    database1
        .write_two_view_geometry(image_id1, image_id2, &TwoViewGeometry::default())
        .unwrap();
    database2
        .write_two_view_geometry(image_id3, image_id4, &TwoViewGeometry::default())
        .unwrap();

    let merged_database = Database::new();
    Database::merge(&database1, &database2, &merged_database).unwrap();
    assert_eq!(merged_database.num_rigs().unwrap(), 2);
    assert_eq!(merged_database.num_cameras().unwrap(), 4);
    assert_eq!(merged_database.num_frames().unwrap(), 2);
    assert_eq!(merged_database.num_images().unwrap(), 4);
    assert_eq!(merged_database.num_pose_priors().unwrap(), 2);
    assert_eq!(merged_database.num_keypoints().unwrap(), 100);
    assert_eq!(merged_database.num_descriptors().unwrap(), 100);
    assert_eq!(merged_database.num_matches().unwrap(), 20);
    assert_eq!(merged_database.num_inlier_matches().unwrap(), 0);
    let frames = merged_database.read_all_frames().unwrap();
    assert_eq!(frames[0].num_data_ids(), frame1.num_data_ids());
    assert_eq!(frames[1].num_data_ids(), frame2.num_data_ids());
    for frame in &frames {
        for data_id in frame.data_ids() {
            match data_id.sensor_id.sensor_type {
                SensorType::Camera => {
                    assert!(merged_database.exists_camera(data_id.sensor_id.id).unwrap());
                    assert!(merged_database.exists_image(data_id.id as ImageId).unwrap());
                }
                other => panic!("Unexpected sensor type: {other}"),
            }
        }
    }
    for pose_prior in merged_database.read_all_pose_priors().unwrap() {
        let data_id = pose_prior.corr_data_id;
        match data_id.sensor_id.sensor_type {
            SensorType::Camera => {
                assert!(merged_database.exists_camera(data_id.sensor_id.id).unwrap());
                assert!(merged_database.exists_image(data_id.id as ImageId).unwrap());
            }
            other => panic!("Unexpected sensor type: {other}"),
        }
    }

    let images = merged_database.read_all_images().unwrap();
    assert_eq!(images[0].camera_id(), 1);
    assert_eq!(images[1].camera_id(), 2);
    assert_eq!(images[2].camera_id(), 3);
    assert_eq!(images[3].camera_id(), 4);
    assert_eq!(merged_database.read_keypoints(1).unwrap().len(), 10);
    assert_eq!(merged_database.read_keypoints(2).unwrap().len(), 20);
    assert_eq!(merged_database.read_keypoints(3).unwrap().len(), 30);
    assert_eq!(merged_database.read_keypoints(4).unwrap().len(), 40);
    assert_eq!(merged_database.read_keypoints(1).unwrap()[0].x, 100.0);
    assert_eq!(merged_database.read_keypoints(2).unwrap()[0].x, 200.0);
    assert_eq!(merged_database.read_keypoints(3).unwrap()[0].x, 300.0);
    assert_eq!(merged_database.read_keypoints(4).unwrap()[0].x, 400.0);
    for (image_id, descriptors) in [
        (1, &descriptors1),
        (2, &descriptors2),
        (3, &descriptors3),
        (4, &descriptors4),
    ] {
        let read = merged_database.read_descriptors(image_id).unwrap();
        assert_eq!(read.extractor_type, descriptors.extractor_type);
        assert_eq!(read.data.len(), descriptors.data.len());
    }
    assert!(merged_database.exists_matches(1, 2).unwrap());
    assert!(!merged_database.exists_matches(2, 3).unwrap());
    assert!(!merged_database.exists_matches(2, 4).unwrap());
    assert!(merged_database.exists_matches(3, 4).unwrap());

    merged_database.clear_all_tables().unwrap();
    assert_eq!(merged_database.num_rigs().unwrap(), 0);
    assert_eq!(merged_database.num_cameras().unwrap(), 0);
    assert_eq!(merged_database.num_frames().unwrap(), 0);
    assert_eq!(merged_database.num_images().unwrap(), 0);
    assert_eq!(merged_database.num_pose_priors().unwrap(), 0);
    assert_eq!(merged_database.num_keypoints().unwrap(), 0);
    assert_eq!(merged_database.num_descriptors().unwrap(), 0);
    assert_eq!(merged_database.num_matches().unwrap(), 0);
}

/// Helper to create a database with images and descriptors.
fn create_database_with_random_descriptors(num_descriptors_per_image: &[usize]) -> Database {
    set_prng_seed(0);
    let database = Database::new();

    let mut camera = Camera::default();
    camera.camera_id = database.write_camera(&camera, false).unwrap();

    for (i, &num_descriptors) in num_descriptors_per_image.iter().enumerate() {
        let mut image = Image::new();
        image.set_name(&format!("image{i}"));
        image.set_camera_id(camera.camera_id).unwrap();
        image.set_image_id(database.write_image(&image, false).unwrap());
        database
            .write_descriptors(
                image.image_id(),
                &FeatureDescriptors::new(
                    FeatureExtractorType::Sift,
                    random_bytes(num_descriptors, 128),
                ),
            )
            .unwrap();
    }

    database
}

#[test]
fn load_random_database_descriptors_test_load_empty() {
    let database = Database::new();
    let result = load_random_database_descriptors(&database, -1).unwrap();
    assert_eq!(result.data.rows(), 0);
    assert_eq!(result.data.cols(), 0);
    assert_eq!(result.extractor_type, FeatureExtractorType::Undefined);
}

#[test]
fn load_random_database_descriptors_test_load_all() {
    let database = create_database_with_random_descriptors(&[10, 20, 30]);
    let result = load_random_database_descriptors(&database, -1).unwrap();
    assert_eq!(result.data.rows(), 60);
    assert_eq!(result.data.cols(), 128);
    assert_eq!(result.extractor_type, FeatureExtractorType::Sift);
}

#[test]
fn load_random_database_descriptors_test_load_all_with_large_max() {
    let database = create_database_with_random_descriptors(&[15, 15]);
    let result = load_random_database_descriptors(&database, 1000).unwrap();
    assert_eq!(result.data.rows(), 30);
    assert_eq!(result.data.cols(), 128);
}

#[test]
fn load_random_database_descriptors_test_load_subset() {
    let database = create_database_with_random_descriptors(&[10, 20, 30]);
    let result = load_random_database_descriptors(&database, 10).unwrap();
    assert_eq!(result.data.rows(), 10);
    assert_eq!(result.data.cols(), 128);
    assert_eq!(result.extractor_type, FeatureExtractorType::Sift);
}

#[test]
fn load_random_database_descriptors_test_load_subset_with_some_empty() {
    let database = create_database_with_random_descriptors(&[0, 10, 0, 15, 0, 20, 0]);
    let result = load_random_database_descriptors(&database, 15).unwrap();
    assert_eq!(result.data.rows(), 15);
    assert_eq!(result.data.cols(), 128);
    assert_eq!(result.extractor_type, FeatureExtractorType::Sift);
}

#[test]
fn load_random_database_descriptors_test_load_exact_total() {
    let database = create_database_with_random_descriptors(&[0, 10, 0, 10, 0]);
    let result = load_random_database_descriptors(&database, 20).unwrap();
    assert_eq!(result.data.rows(), 20);
    assert_eq!(result.data.cols(), 128);
}
