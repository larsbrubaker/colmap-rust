// Port of colmap/scene/reconstruction_test.cc, part 5, 1:1 (same test names, values and
// checks): TranscribeImageIdsToDatabase, for colmap_rust::scene::reconstruction_database over
// the in-memory Database. COLMAP has no direct test of `Reconstruction::Load` (its mappers
// exercise it), so the `rust_only_*` tests here pin `load` and the failure path of
// `transcribe_image_ids_to_database` (docs/CPP_DIVERGENCES.md entry 273).

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::Vector2d;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::random_eigen_vector3d;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::database_cache::{DatabaseCache, DatabaseCacheOptions};
use colmap_rust::scene::frame::Frame;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset_with_database, SyntheticDatasetOptions};
use colmap_rust::scene::track::Track;
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{DataId, ImageId};

use crate::reconstruction::expect_valid_ptrs;

const K_IMAGE_NAMES: [&str; 3] = ["test_image1.jpg", "test_image2.jpg", "test_image3.jpg"];
const RECON_IMAGE_IDS: [ImageId; 3] = [100, 200, 300];

/// The test's setup: three named images in the database and a reconstruction holding the same
/// names under ids 100, 200 and 300, each in its own posed frame, plus a 3D point seen by all.
fn transcribe_setup() -> (Database, [Image; 3], Reconstruction, u64) {
    let database = Database::new();

    let camera = Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1.0, 1, 1).unwrap();
    database.write_camera(&camera, true).unwrap();

    // Write images to database.
    let db_images = K_IMAGE_NAMES.map(|name| {
        let mut db_image = Image::new();
        db_image.set_name(name);
        db_image.set_camera_id(camera.camera_id).unwrap();
        db_image.set_image_id(database.write_image(&db_image, false).unwrap());
        db_image
    });

    // Create a reconstruction with different image IDs but same names.
    let mut reconstruction = Reconstruction::new();
    reconstruction.add_camera(camera.clone()).unwrap();
    let mut rig = Rig::new();
    rig.set_rig_id(1);
    rig.add_ref_sensor(camera.sensor_id()).unwrap();
    reconstruction.add_rig(rig.clone()).unwrap();

    for (i, &image_id) in RECON_IMAGE_IDS.iter().enumerate() {
        let mut frame = Frame::new();
        frame.set_frame_id(image_id);
        frame.set_rig_id(rig.rig_id()).unwrap();
        frame
            .add_data_id(DataId::new(camera.sensor_id(), image_id))
            .unwrap();
        frame.set_rig_from_world(Rigid3d::identity());
        reconstruction.add_frame(frame).unwrap();

        let mut image = Image::new();
        image.set_image_id(image_id);
        image.set_camera_id(camera.camera_id).unwrap();
        image.set_frame_id(image_id);
        image.set_name(K_IMAGE_NAMES[i]);
        image.set_points2d_from_xy(&[Vector2d::new(0.0, 0.0); 10]);
        reconstruction.add_image(image).unwrap();
    }

    // Add a 3D point with observations to test track updates.
    let mut track = Track::new();
    track.add_element_ids(RECON_IMAGE_IDS[0], 0);
    track.add_element_ids(RECON_IMAGE_IDS[1], 1);
    track.add_element_ids(RECON_IMAGE_IDS[2], 2);
    let point3d_id = reconstruction
        .add_point3d(random_eigen_vector3d(), track, Default::default())
        .unwrap();
    (database, db_images, reconstruction, point3d_id)
}

#[test]
fn reconstruction_transcribe_image_ids_to_database() {
    set_prng_seed(0);
    let (database, [db_image1, db_image2, db_image3], mut reconstruction, point3d_id) =
        transcribe_setup();

    reconstruction
        .transcribe_image_ids_to_database(&database)
        .unwrap();

    // Verify image IDs were updated to match database.
    assert!(reconstruction.exists_image(db_image1.image_id()));
    assert!(reconstruction.exists_image(db_image2.image_id()));
    assert!(reconstruction.exists_image(db_image3.image_id()));
    assert!(!reconstruction.exists_image(RECON_IMAGE_IDS[0]));
    assert!(!reconstruction.exists_image(RECON_IMAGE_IDS[1]));
    assert!(!reconstruction.exists_image(RECON_IMAGE_IDS[2]));

    // Verify image names are preserved.
    assert_eq!(
        reconstruction.image(db_image1.image_id()).unwrap().name(),
        K_IMAGE_NAMES[0]
    );
    assert_eq!(
        reconstruction.image(db_image2.image_id()).unwrap().name(),
        K_IMAGE_NAMES[1]
    );
    assert_eq!(
        reconstruction.image(db_image3.image_id()).unwrap().name(),
        K_IMAGE_NAMES[2]
    );

    // Verify frame data IDs were updated.
    assert!(reconstruction
        .frame(RECON_IMAGE_IDS[0])
        .unwrap()
        .has_data_id(db_image1.data_id()));
    assert!(reconstruction
        .frame(RECON_IMAGE_IDS[1])
        .unwrap()
        .has_data_id(db_image2.data_id()));
    assert!(reconstruction
        .frame(RECON_IMAGE_IDS[2])
        .unwrap()
        .has_data_id(db_image3.data_id()));

    // Verify track elements were updated.
    let updated_track = &reconstruction.point3d(point3d_id).unwrap().track;
    assert_eq!(updated_track.length(), 3);
    let mut track_image_ids: Vec<ImageId> = updated_track
        .elements()
        .iter()
        .map(|el| el.image_id)
        .collect();
    track_image_ids.sort_unstable();
    let mut expected = vec![
        db_image1.image_id(),
        db_image2.image_id(),
        db_image3.image_id(),
    ];
    expected.sort_unstable();
    assert_eq!(track_image_ids, expected);
}

#[test]
fn rust_only_transcribe_image_ids_to_database_unknown_name_changes_nothing() {
    set_prng_seed(0);
    let (_, _, mut reconstruction, _) = transcribe_setup();
    let orig = reconstruction.clone();
    let other_database = Database::new();
    let error = reconstruction
        .transcribe_image_ids_to_database(&other_database)
        .unwrap_err();
    assert!(error
        .to_string()
        .contains("Image with name test_image1.jpg does not exist in database"));
    assert_eq!(reconstruction.images(), orig.images());
    assert_eq!(reconstruction.frames(), orig.frames());
    assert_eq!(reconstruction.points3d(), orig.points3d());
}

#[test]
fn rust_only_reconstruction_load_from_database_cache() {
    set_prng_seed(0);
    let database = Database::new();
    let mut synthetic = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 2,
        num_frames_per_rig: 3,
        num_points3d: 20,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut synthetic, &database).unwrap();
    let cache = DatabaseCache::create(&database, &DatabaseCacheOptions::default()).unwrap();

    let mut reconstruction = Reconstruction::new();
    reconstruction.load(&cache).unwrap();
    assert_eq!(reconstruction.num_cameras(), cache.num_cameras());
    assert_eq!(reconstruction.num_rigs(), cache.num_rigs());
    assert_eq!(reconstruction.num_frames(), cache.num_frames());
    assert_eq!(reconstruction.num_images(), cache.num_images());
    assert_eq!(reconstruction.num_reg_frames(), 0);
    for (&image_id, image) in cache.images() {
        let loaded = reconstruction.image(image_id).unwrap();
        assert_eq!(loaded.name(), image.name());
        assert_eq!(loaded.points2d(), image.points2d());
    }
    expect_valid_ptrs(&reconstruction);

    // Loading again keeps everything (all objects agree with the cache).
    let before = reconstruction.clone();
    reconstruction.load(&cache).unwrap();
    assert_eq!(reconstruction.images(), before.images());
    assert_eq!(reconstruction.frames(), before.frames());

    // An existing image without 2D points takes the cache's.
    let image_id = *cache.images().keys().next().unwrap();
    // (The loaded images observe no 3D points, so clearing the list keeps the counts right.)
    reconstruction
        .image_mut(image_id)
        .unwrap()
        .points2d_mut()
        .clear();
    reconstruction.load(&cache).unwrap();
    assert_eq!(
        reconstruction.image(image_id).unwrap().points2d(),
        cache.image(image_id).unwrap().points2d()
    );

    // A camera that disagrees with the cache fails the check.
    let camera_id = *cache.cameras().keys().next().unwrap();
    reconstruction.camera_mut(camera_id).unwrap().width += 1;
    assert!(reconstruction.load(&cache).is_err());
}
