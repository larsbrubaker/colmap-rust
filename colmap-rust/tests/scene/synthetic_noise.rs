// Port of colmap/scene/synthetic_test.cc, part 3, 1:1 (same test names, values and checks):
// the SynthesizeNoise tests (colmap_rust::scene::synthetic_noise::
// synthesize_noise_with_database over the in-memory Database) and SynthesizeImages.Nominal,
// whose images reach a sink instead of `image_path` (docs/CPP_DIVERGENCES.md entry 268): the
// test checks every image's bitmap has its camera's size, as COLMAP's reads each file back
// and checks. Each test calls `set_prng_seed(0)` first, as COLMAP's gtest main does.

use std::collections::{BTreeMap, HashMap};

use colmap_rust::geometry::pose_prior::{CoordinateSystem, PosePrior};
use colmap_rust::linalg::Vector2d;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{
    synthesize_dataset, synthesize_dataset_with_database, SyntheticDatasetOptions,
};
use colmap_rust::scene::synthetic_noise::{
    synthesize_images, synthesize_noise_with_database, SyntheticImageOptions, SyntheticNoiseOptions,
};
use colmap_rust::util::types::PosePriorId;

use crate::eigen_matchers::eigen_matrix_near;

#[test]
fn synthesize_noise_point2d_noise() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions::default();
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
    assert!(reconstruction.compute_mean_reprojection_error() < 1e-3);

    let synthetic_noise_options = SyntheticNoiseOptions {
        point2d_stddev: 0.1,
        ..SyntheticNoiseOptions::default()
    };
    synthesize_noise_with_database(&synthetic_noise_options, &mut reconstruction, &database)
        .unwrap();
    assert!(reconstruction.compute_mean_reprojection_error() > 1e-3);

    for (&image_id, image) in reconstruction.images() {
        let keypoints = database.read_keypoints(image_id).unwrap();
        for point2d_idx in 0..image.num_points2d() {
            let keypoint = keypoints[point2d_idx as usize];
            let expected = image.point2d(point2d_idx).xy;
            let actual = Vector2d::new(f64::from(keypoint.x), f64::from(keypoint.y));
            assert!(eigen_matrix_near(
                &[actual.x, actual.y],
                &[expected.x, expected.y],
                1e-6
            ));
        }
    }
}

#[test]
fn synthesize_noise_point3d_noise() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions::default();

    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
    assert!(reconstruction.compute_mean_reprojection_error() < 1e-3);

    let synthetic_noise_options = SyntheticNoiseOptions {
        point3d_stddev: 0.1,
        ..SyntheticNoiseOptions::default()
    };
    synthesize_noise_with_database(&synthetic_noise_options, &mut reconstruction, &database)
        .unwrap();
    assert!(reconstruction.compute_mean_reprojection_error() > 1e-3);
}

#[test]
fn synthesize_noise_rig_from_world_noise() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions::default();

    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
    assert!(reconstruction.compute_mean_reprojection_error() < 1e-3);

    let synthetic_noise_options = SyntheticNoiseOptions {
        rig_from_world_translation_stddev: 0.1,
        rig_from_world_rotation_stddev: 0.1,
        ..SyntheticNoiseOptions::default()
    };
    synthesize_noise_with_database(&synthetic_noise_options, &mut reconstruction, &database)
        .unwrap();
    assert!(reconstruction.compute_mean_reprojection_error() > 1e-3);
}

fn read_pose_priors(database: &Database) -> HashMap<PosePriorId, PosePrior> {
    let mut pose_priors = HashMap::new();
    for pose_prior in database.read_all_pose_priors().unwrap() {
        pose_priors.insert(pose_prior.pose_prior_id, pose_prior);
    }
    pose_priors
}

#[test]
fn synthesize_noise_prior_position_noise() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        prior_position: true,
        prior_position_coordinate_system: CoordinateSystem::Wgs84,
        ..SyntheticDatasetOptions::default()
    };

    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
    let orig_pose_priors = read_pose_priors(&database);

    let synthetic_noise_options = SyntheticNoiseOptions {
        prior_position_stddev: 0.1,
        ..SyntheticNoiseOptions::default()
    };
    synthesize_noise_with_database(&synthetic_noise_options, &mut reconstruction, &database)
        .unwrap();
    for pose_prior in database.read_all_pose_priors().unwrap() {
        let orig = orig_pose_priors[&pose_prior.pose_prior_id].position;
        // Check that some noise was added.
        assert_ne!(pose_prior.position, orig);
        // Check that the noisy positions are somewhat close to the original ones.
        assert!(eigen_matrix_near(
            &pose_prior.position.to_array(),
            &orig.to_array(),
            10.0 * synthetic_noise_options.prior_position_stddev
        ));
        assert!(pose_prior.has_position_cov());
        assert!(
            (pose_prior.position_covariance.trace() / 3.0
                - synthetic_noise_options.prior_position_stddev
                    * synthetic_noise_options.prior_position_stddev)
                .abs()
                <= 1e-6
        );
    }
}

#[test]
fn synthesize_noise_prior_gravity_noise() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        prior_gravity: true,
        ..SyntheticDatasetOptions::default()
    };

    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
    let orig_pose_priors = read_pose_priors(&database);

    let synthetic_noise_options = SyntheticNoiseOptions {
        prior_gravity_stddev: 0.1,
        ..SyntheticNoiseOptions::default()
    };
    synthesize_noise_with_database(&synthetic_noise_options, &mut reconstruction, &database)
        .unwrap();
    for pose_prior in database.read_all_pose_priors().unwrap() {
        let angle = pose_prior
            .gravity
            .dot(orig_pose_priors[&pose_prior.pose_prior_id].gravity)
            .acos();
        // Check that some noise was added.
        assert!(angle > 0.0);
        // Check that the noisy directions are somewhat close to the original ones.
        assert!(angle < 10.0 * synthetic_noise_options.prior_gravity_stddev);
    }
}

#[test]
fn synthesize_images_nominal() {
    set_prng_seed(0);
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 2,
        num_points3d: 80,
        num_points2d_without_point3d: 20,
        camera_width: 320,
        camera_height: 240,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();

    let mut written = BTreeMap::new();
    synthesize_images(
        &SyntheticImageOptions::default(),
        &reconstruction,
        |name, bitmap| {
            written.insert(name.to_owned(), bitmap);
        },
    )
    .unwrap();

    for image in reconstruction.images().values() {
        let bitmap = written.get(image.name()).expect("image was synthesized");
        let camera = reconstruction.camera(image.camera_id()).unwrap();
        assert_eq!(bitmap.width() as usize, camera.width);
        assert_eq!(bitmap.height() as usize, camera.height);
    }
}
