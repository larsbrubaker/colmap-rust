// Port of colmap/scene/database_cache_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::scene::database_cache::DatabaseCache over the in-memory Database.
// `DatabaseCache::Create(*database, {})` is `DatabaseCache::create(&database,
// &DatabaseCacheOptions::default())`; the const and non-const `CorrespondenceGraph()`
// accessors are `correspondence_graph` / `correspondence_graph_mut`. COLMAP's gtest main seeds
// the PRNG with 0 before every test, so the fixtures call `set_prng_seed(0)` first.

use std::collections::HashSet;

use colmap_rust::feature::types::{FeatureKeypoint, FeatureMatch};
use colmap_rust::geometry::pose_prior::PosePrior;
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{
    random_eigen_matrix3d, random_eigen_quaterniond, random_eigen_vector3d,
};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::database_cache::{DatabaseCache, DatabaseCacheOptions};
use colmap_rust::scene::frame::Frame;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{pair_id_to_image_pair, INVALID_CAMERA_ID};

use crate::transform_matchers::rigid3d_near;

fn simple_pinhole(focal_length: f64, size: usize) -> Camera {
    Camera::create_from_model_id(
        INVALID_CAMERA_ID,
        CameraModelId::SimplePinhole,
        focal_length,
        size,
        size,
    )
    .unwrap()
}

fn keypoints(n: usize) -> Vec<FeatureKeypoint> {
    vec![FeatureKeypoint::default(); n]
}

fn test_two_view_geometry() -> TwoViewGeometry {
    TwoViewGeometry {
        inlier_matches: vec![FeatureMatch::new(0, 1)],
        config: ConfigurationType::PlanarOrPanoramic,
        f: Some(random_eigen_matrix3d()),
        e: Some(random_eigen_matrix3d()),
        h: Some(random_eigen_matrix3d()),
        cam2_from_cam1: Some(Rigid3d::new(
            random_eigen_quaterniond(),
            random_eigen_vector3d(),
        )),
        ..TwoViewGeometry::default()
    }
}

fn create_test_database() -> Database {
    set_prng_seed(0);
    let database = Database::new();

    let mut camera1 = simple_pinhole(1.0, 1);
    camera1.camera_id = database.write_camera(&camera1, false).unwrap();
    let mut camera2 = simple_pinhole(2.0, 2);
    camera2.camera_id = database.write_camera(&camera2, false).unwrap();

    let mut rig = Rig::new();
    rig.add_ref_sensor(camera1.sensor_id()).unwrap();
    rig.add_sensor(camera2.sensor_id(), None).unwrap();
    let rig_id = database.write_rig(&rig, false).unwrap();

    let mut images = Vec::new();
    for (name, camera_id) in [
        ("image1", camera1.camera_id),
        ("image2", camera2.camera_id),
        ("image3", camera1.camera_id),
        ("image4", camera2.camera_id),
    ] {
        let mut image = Image::new();
        image.set_name(name);
        image.set_camera_id(camera_id).unwrap();
        image.set_image_id(database.write_image(&image, false).unwrap());
        images.push(image);
    }
    let [image1, image2, image3, image4] = <[Image; 4]>::try_from(images).unwrap();

    // COLMAP's fixture stores the returned id in `pose_prior1` and then assigns image2's data
    // id to `pose_prior1` (already written) instead of `pose_prior2`, so `pose_prior2` keeps
    // the default corresponding data id. Neither later assignment is read; both are left out.
    let pose_prior1 = PosePrior {
        corr_data_id: image1.data_id(),
        position: random_eigen_vector3d(),
        ..PosePrior::default()
    };
    database.write_pose_prior(&pose_prior1, false).unwrap();
    let pose_prior2 = PosePrior {
        position: random_eigen_vector3d(),
        ..PosePrior::default()
    };
    database.write_pose_prior(&pose_prior2, false).unwrap();

    let mut frame1 = Frame::new();
    frame1.set_rig_id(rig_id).unwrap();
    frame1.add_data_id(image1.data_id()).unwrap();
    frame1.add_data_id(image2.data_id()).unwrap();
    database.write_frame(&frame1, false).unwrap();
    let mut frame2 = Frame::new();
    frame2.set_rig_id(rig_id).unwrap();
    frame2.add_data_id(image3.data_id()).unwrap();
    frame2.add_data_id(image4.data_id()).unwrap();
    database.write_frame(&frame2, false).unwrap();

    database
        .write_keypoints(image1.image_id(), &keypoints(10))
        .unwrap();
    database
        .write_keypoints(image2.image_id(), &keypoints(5))
        .unwrap();
    database
        .write_keypoints(image3.image_id(), &keypoints(6))
        .unwrap();
    database
        .write_keypoints(image4.image_id(), &keypoints(3))
        .unwrap();

    let two_view_geometry = test_two_view_geometry();
    database
        .write_two_view_geometry(image1.image_id(), image2.image_id(), &two_view_geometry)
        .unwrap();
    database
        .write_two_view_geometry(image2.image_id(), image3.image_id(), &two_view_geometry)
        .unwrap();
    database
        .write_two_view_geometry(image3.image_id(), image4.image_id(), &two_view_geometry)
        .unwrap();

    database
}

/// Asserts the correspondence and observation counts of one image in the cache's graph.
fn expect_graph_counts(
    cache: &DatabaseCache,
    image_id: u32,
    correspondences: u32,
    observations: u32,
) {
    let graph = cache.correspondence_graph();
    assert!(graph.exists_image(image_id));
    assert_eq!(
        graph.num_correspondences_for_image(image_id).unwrap(),
        correspondences
    );
    assert_eq!(
        graph.num_observations_for_image(image_id).unwrap(),
        observations
    );
}

fn names(images: &[&Image]) -> HashSet<String> {
    images
        .iter()
        .map(|image| image.name().to_string())
        .collect()
}

#[test]
fn database_cache_empty() {
    let cache = DatabaseCache::new();
    assert_eq!(cache.num_rigs(), 0);
    assert_eq!(cache.num_cameras(), 0);
    assert_eq!(cache.num_frames(), 0);
    assert_eq!(cache.num_images(), 0);
    assert_eq!(cache.num_pose_priors(), 0);
}

#[test]
fn database_cache_construct_from_database() {
    let database = create_test_database();
    let cache = DatabaseCache::create(&database, &DatabaseCacheOptions::default()).unwrap();

    assert_eq!(cache.num_rigs(), 1);
    assert_eq!(cache.num_cameras(), 2);
    assert_eq!(cache.num_frames(), 2);
    assert_eq!(cache.num_images(), 4);
    assert_eq!(cache.num_pose_priors(), 2);

    for rig in database.read_all_rigs().unwrap() {
        assert!(cache.exists_rig(rig.rig_id()));
        assert_eq!(cache.rig(rig.rig_id()).unwrap(), &rig);
    }

    for camera in database.read_all_cameras().unwrap() {
        assert!(cache.exists_camera(camera.camera_id));
        assert_eq!(cache.camera(camera.camera_id).unwrap(), &camera);
    }

    for frame in database.read_all_frames().unwrap() {
        assert!(cache.exists_frame(frame.frame_id()));
        assert_eq!(cache.frame(frame.frame_id()).unwrap(), &frame);
    }

    let images = database.read_all_images().unwrap();
    for (image, num_points2d) in images.iter().zip([10, 5, 6, 3]) {
        assert!(cache.exists_image(image.image_id()));
        assert_eq!(
            cache.image(image.image_id()).unwrap().num_points2d(),
            num_points2d
        );
    }

    assert_eq!(
        cache.pose_priors(),
        database.read_all_pose_priors().unwrap()
    );

    expect_graph_counts(&cache, images[0].image_id(), 1, 1);
    expect_graph_counts(&cache, images[1].image_id(), 2, 2);
    let graph = cache.correspondence_graph();
    assert_eq!(
        graph
            .num_correspondences_for_image(images[2].image_id())
            .unwrap(),
        2
    );
    assert_eq!(
        graph
            .num_observations_for_image(images[2].image_id())
            .unwrap(),
        2
    );
    assert_eq!(
        graph
            .num_correspondences_for_image(images[3].image_id())
            .unwrap(),
        1
    );
    assert_eq!(
        graph
            .num_observations_for_image(images[3].image_id())
            .unwrap(),
        1
    );
}

#[test]
fn database_cache_construct_from_database_with_custom_images() {
    let database = create_test_database();

    // Note that the first two images are part of the same frame.
    let images = database.read_all_images().unwrap();
    let options = DatabaseCacheOptions {
        image_names: names(&[&images[0], &images[1]]),
        ..DatabaseCacheOptions::default()
    };
    let cache = DatabaseCache::create(&database, &options).unwrap();

    assert_eq!(cache.num_rigs(), 1);
    assert_eq!(cache.num_cameras(), 2);
    assert_eq!(cache.num_frames(), 1);
    assert_eq!(cache.num_images(), 2);
    assert_eq!(cache.num_pose_priors(), 2);

    expect_graph_counts(&cache, images[0].image_id(), 1, 1);
    expect_graph_counts(&cache, images[1].image_id(), 1, 1);
}

fn create_legacy_test_database() -> Database {
    set_prng_seed(0);
    let database = Database::new();

    let camera = simple_pinhole(1.0, 1);
    let camera_id = database.write_camera(&camera, false).unwrap();
    let mut image_ids = Vec::new();
    for name in ["image1", "image2", "image3"] {
        let mut image = Image::new();
        image.set_name(name);
        image.set_camera_id(camera_id).unwrap();
        image_ids.push(database.write_image(&image, false).unwrap());
    }
    let (image_id1, image_id2, image_id3) = (image_ids[0], image_ids[1], image_ids[2]);
    database.write_keypoints(image_id1, &keypoints(10)).unwrap();
    database.write_keypoints(image_id2, &keypoints(5)).unwrap();
    database.write_keypoints(image_id3, &keypoints(7)).unwrap();
    let two_view_geometry = test_two_view_geometry();
    database
        .write_two_view_geometry(image_id1, image_id2, &two_view_geometry)
        .unwrap();
    database
        .write_two_view_geometry(image_id1, image_id3, &two_view_geometry)
        .unwrap();

    database
}

#[test]
fn database_cache_construct_from_legacy_database_without_rigs_and_frames() {
    let database = create_legacy_test_database();
    let cache = DatabaseCache::create(&database, &DatabaseCacheOptions::default()).unwrap();
    assert_eq!(cache.num_cameras(), 1);
    assert_eq!(cache.num_images(), 3);
    assert_eq!(cache.num_pose_priors(), 0);
    assert!(cache.exists_camera(1));
    assert_eq!(
        cache.camera(1).unwrap().model_id,
        CameraModelId::SimplePinhole
    );
    assert!(cache.exists_image(1));
    assert!(cache.exists_image(2));
    assert!(cache.exists_image(3));
    assert_eq!(cache.image(1).unwrap().num_points2d(), 10);
    assert_eq!(cache.image(2).unwrap().num_points2d(), 5);
    assert_eq!(cache.image(3).unwrap().num_points2d(), 7);
    expect_graph_counts(&cache, 1, 2, 1);
    expect_graph_counts(&cache, 2, 1, 1);
    expect_graph_counts(&cache, 3, 1, 1);
}

#[test]
fn database_cache_construct_from_legacy_database_with_custom_images() {
    let database = create_legacy_test_database();
    let images = database.read_all_images().unwrap();
    let options = DatabaseCacheOptions {
        image_names: names(&[&images[0], &images[2]]),
        ..DatabaseCacheOptions::default()
    };
    let cache = DatabaseCache::create(&database, &options).unwrap();
    assert_eq!(cache.num_cameras(), 1);
    assert_eq!(cache.num_images(), 2);
    assert_eq!(cache.num_pose_priors(), 0);
    assert!(cache.exists_camera(1));
    assert_eq!(
        cache.camera(1).unwrap().model_id,
        CameraModelId::SimplePinhole
    );
    assert!(cache.exists_image(1));
    assert!(cache.exists_image(3));
    assert_eq!(cache.image(1).unwrap().num_points2d(), 10);
    assert_eq!(cache.image(3).unwrap().num_points2d(), 7);
    expect_graph_counts(&cache, 1, 1, 1);
    expect_graph_counts(&cache, 3, 1, 1);
}

#[test]
fn database_cache_construct_from_custom() {
    set_prng_seed(0);
    let mut cache = DatabaseCache::new();
    assert_eq!(cache.num_rigs(), 0);
    assert_eq!(cache.num_cameras(), 0);
    assert_eq!(cache.num_frames(), 0);
    assert_eq!(cache.num_images(), 0);
    assert_eq!(cache.num_pose_priors(), 0);

    const RIG_ID: u32 = 41;
    let mut rig = Rig::new();
    rig.set_rig_id(RIG_ID);
    cache.add_rig(rig).unwrap();

    const CAMERA_ID: u32 = 42;
    cache
        .add_camera(
            Camera::create_from_model_id(CAMERA_ID, CameraModelId::SimplePinhole, 1.0, 1, 1)
                .unwrap(),
        )
        .unwrap();

    const FRAME_ID: u32 = 43;
    let mut frame = Frame::new();
    frame.set_frame_id(FRAME_ID);
    cache.add_frame(frame).unwrap();

    const IMAGE_ID: u32 = 44;
    let mut image = Image::new();
    image.set_image_id(IMAGE_ID);
    image.set_name("image");
    image.set_camera_id(CAMERA_ID).unwrap();
    cache.add_image(image).unwrap();

    const POSE_PRIOR_ID: u32 = 45;
    let pose_prior = PosePrior {
        position: random_eigen_vector3d(),
        pose_prior_id: POSE_PRIOR_ID,
        ..PosePrior::default()
    };
    cache.add_pose_prior(pose_prior);

    assert_eq!(cache.num_cameras(), 1);
    assert_eq!(cache.num_images(), 1);
    assert_eq!(cache.num_pose_priors(), 1);
    assert!(cache.exists_rig(RIG_ID));
    assert!(cache.exists_camera(CAMERA_ID));
    assert!(cache.exists_frame(FRAME_ID));
    assert!(cache.exists_image(IMAGE_ID));
    assert_eq!(cache.pose_priors(), [pose_prior]);
}

#[test]
fn database_cache_non_const_correspondence_graph() {
    let database = create_test_database();
    let mut cache = DatabaseCache::create(&database, &DatabaseCacheOptions::default()).unwrap();

    // Non-const overload returns a mutable graph.
    let mutable_graph = cache.correspondence_graph_mut();

    // Verify it can be used to update a two-view geometry.
    let image_pairs = mutable_graph.image_pairs();
    assert!(!image_pairs.is_empty());
    let (image_id1, image_id2) = pair_id_to_image_pair(image_pairs[0]).unwrap();

    let mut geom = mutable_graph
        .extract_two_view_geometry(image_id1, image_id2, /*extract_inlier_matches=*/ false)
        .unwrap();
    let new_pose = Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d());
    geom.cam2_from_cam1 = Some(new_pose);
    mutable_graph
        .update_two_view_geometry(image_id1, image_id2, geom)
        .unwrap();

    // Read back through the const accessor and verify the update is visible.
    let const_cache: &DatabaseCache = &cache;
    let updated = const_cache
        .correspondence_graph()
        .extract_two_view_geometry(image_id1, image_id2, false)
        .unwrap();
    assert!(updated.cam2_from_cam1.is_some());
    assert!(rigid3d_near(
        &updated.cam2_from_cam1.unwrap(),
        &new_pose,
        1e-6,
        1e-6
    ));
}

/// Rust-only: `CreateFromCache` (no database_cache_test.cc case) keeps whole frames — with
/// `load_all_images`, the one image named in the filter brings its frame partner, camera and
/// rig along — and copies the correspondences between the kept images. Without
/// `load_all_images` the same filter keeps nothing, since no matched pair lies inside it.
#[test]
fn rust_only_create_from_cache_keeps_whole_frames() {
    let database = create_test_database();
    let source = DatabaseCache::create(&database, &DatabaseCacheOptions::default()).unwrap();
    let images = database.read_all_images().unwrap();
    let options = DatabaseCacheOptions {
        image_names: names(&[&images[0]]),
        load_all_images: true,
        ..DatabaseCacheOptions::default()
    };
    let cache = DatabaseCache::create_from_cache(&source, &options).unwrap();
    assert_eq!(cache.num_images(), 2);
    assert!(cache.exists_image(images[0].image_id()));
    assert!(cache.exists_image(images[1].image_id()));
    assert_eq!(cache.num_frames(), 1);
    assert_eq!(cache.num_cameras(), 2);
    assert_eq!(cache.num_rigs(), 1);
    assert_eq!(cache.num_pose_priors(), 2);
    assert_eq!(cache.correspondence_graph().num_image_pairs(), 1);
    expect_graph_counts(&cache, images[0].image_id(), 1, 1);
    expect_graph_counts(&cache, images[1].image_id(), 1, 1);

    let connected_only = DatabaseCacheOptions {
        load_all_images: false,
        ..options
    };
    let cache = DatabaseCache::create_from_cache(&source, &connected_only).unwrap();
    assert_eq!(cache.num_images(), 0);
}
