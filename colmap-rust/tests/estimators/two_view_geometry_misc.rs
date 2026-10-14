// Port of COLMAP's src/colmap/estimators/two_view_geometry_test.cc (same values and
// tolerances), the remaining cases: watermark detection, stationary matches, the seeded
// determinism cases, DEGENSAC, TwoViewGeometryFromKnownRelativePose,
// EstimateRigTwoViewGeometries, EstimateMultipleTwoViewGeometries and
// MaybeDecomposeRelativePoses. Fixtures are in two_view_geometry.rs. Tier C (RANSAC), Tier B
// for the known-pose case.
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does.
// `Database::Open(kInMemorySqliteDatabasePath)` is `Database::new()`.

use colmap_rust::estimators::two_view_geometry::{
    estimate_rig_two_view_geometries, estimate_two_view_geometry, maybe_decompose_relative_poses,
    two_view_geometry_from_known_relative_pose, ImagePairMatches, TwoViewGeometryOptions,
};
use colmap_rust::feature::types::{FeatureMatch, FeatureMatches};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::Vector2d;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::database_cache::{DatabaseCache, DatabaseCacheOptions};
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{
    synthesize_dataset, synthesize_dataset_with_database, SyntheticDatasetOptions,
};
use colmap_rust::scene::synthetic_noise::SyntheticNoiseOptions;
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::pair_id_to_image_pair;

use crate::expect_near;
use crate::transform_matchers::rigid3d_near;
use crate::two_view_geometry::{
    create_two_view_geometry_pose_test_data, create_two_view_geometry_test_data,
    extract_points_and_matches, image_cam_from_world, with_unit_translation,
    TwoViewGeometryTestData,
};

fn two_frames(num_points3d: i32) -> SyntheticDatasetOptions {
    SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d,
        camera_has_prior_focal_length: true,
        ..SyntheticDatasetOptions::default()
    }
}

fn estimate(data: &TwoViewGeometryTestData, options: &TwoViewGeometryOptions) -> TwoViewGeometry {
    estimate_two_view_geometry(
        &data.camera1,
        &data.points1,
        &data.camera2,
        &data.points2,
        data.matches.clone(),
        options,
    )
    .unwrap()
}

fn seeded(seed: i32, options: &TwoViewGeometryOptions) -> TwoViewGeometryOptions {
    let mut options = *options;
    options.ransac_options.random_seed = seed;
    options
}

#[test]
fn estimate_two_view_geometry_detect_watermark() {
    set_prng_seed(0);
    let mut data =
        create_two_view_geometry_test_data(&two_frames(100), &SyntheticNoiseOptions::default());

    let mut options = TwoViewGeometryOptions {
        detect_watermark: true,
        ..TwoViewGeometryOptions::default()
    };
    assert_ne!(
        estimate(&data, &options).config,
        ConfigurationType::Watermark
    );

    // Place the points on the left and right side of the images.
    let num_matches = data.matches.len();
    for i in 0..num_matches {
        let y = i as f64 / num_matches as f64 * data.camera1.height as f64;
        let m = data.matches[i];
        data.points1[m.point2d_idx1 as usize] = Vector2d::new(0.0, y);
        data.points2[m.point2d_idx2 as usize] = Vector2d::new((data.camera2.width - 1) as f64, y);
    }
    assert_eq!(
        estimate(&data, &options).config,
        ConfigurationType::Watermark
    );

    // Place the points on the top and bottom side of the images.
    for i in 0..num_matches {
        let x = i as f64 / num_matches as f64 * data.camera1.width as f64;
        let m = data.matches[i];
        data.points1[m.point2d_idx1 as usize] = Vector2d::new(x, 0.0);
        data.points2[m.point2d_idx2 as usize] = Vector2d::new(x, (data.camera2.height - 1) as f64);
    }
    assert_eq!(
        estimate(&data, &options).config,
        ConfigurationType::Watermark
    );

    // With disabled detection, expect a normal config.
    options.detect_watermark = false;
    assert_ne!(
        estimate(&data, &options).config,
        ConfigurationType::Watermark
    );
}

#[test]
fn estimate_two_view_geometry_ignore_stationary_matches() {
    set_prng_seed(0);
    let synthetic_dataset_options = two_frames(500);
    let mut data = create_two_view_geometry_test_data(
        &synthetic_dataset_options,
        &SyntheticNoiseOptions::default(),
    );
    for m in data.matches.clone() {
        data.points1[m.point2d_idx1 as usize] = data.points2[m.point2d_idx2 as usize];
    }

    let mut options = TwoViewGeometryOptions::default();
    let geometry1 = estimate(&data, &options);
    assert_eq!(geometry1.config, ConfigurationType::PlanarOrPanoramic);
    assert_eq!(
        geometry1.inlier_matches.len(),
        synthetic_dataset_options.num_points3d as usize
    );

    options.filter_stationary_matches = true;
    let geometry2 = estimate(&data, &options);
    assert_eq!(geometry2.config, ConfigurationType::Degenerate);
    assert_eq!(geometry2.inlier_matches.len(), 0);
}

/// Two frames of 500 points with 60% inlier matches and the given pixel noise.
fn noisy_data(
    camera_has_prior_focal_length: bool,
    point2d_stddev: f64,
    num_rigs: i32,
) -> TwoViewGeometryTestData {
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs,
        num_cameras_per_rig: 3 - num_rigs,
        num_frames_per_rig: 1,
        num_points3d: 500,
        inlier_match_ratio: 0.6,
        camera_has_prior_focal_length,
        sensor_from_rig_translation_stddev: if num_rigs == 1 {
            0.0
        } else {
            SyntheticDatasetOptions::default().sensor_from_rig_translation_stddev
        },
        ..SyntheticDatasetOptions::default()
    };
    let synthetic_noise_options = SyntheticNoiseOptions {
        point2d_stddev,
        ..SyntheticNoiseOptions::default()
    };
    create_two_view_geometry_test_data(&synthetic_dataset_options, &synthetic_noise_options)
}

#[test]
fn estimate_two_view_geometry_calibrated_deterministic() {
    set_prng_seed(0);
    // Moderate noise: E and F support similar inlier counts (unambiguously CALIBRATED), yet
    // different RANSAC seeds select different inlier sets and thus yield different E.
    let data = noisy_data(true, 3.0, 2);

    // This test verifies RANSAC seed reproducibility, not the default calibration
    // verification boundary.
    let options = TwoViewGeometryOptions {
        min_e_f_inlier_ratio: 0.8,
        ..TwoViewGeometryOptions::default()
    };
    let geometry1 = estimate(&data, &seeded(42, &options));
    assert_eq!(geometry1.config, ConfigurationType::Calibrated);
    let geometry2 = estimate(&data, &seeded(42, &options));
    assert_eq!(geometry2.config, ConfigurationType::Calibrated);
    // Using the same random seed should produce identical results.
    assert_eq!(geometry1.e, geometry2.e);

    let geometry3 = estimate(&data, &seeded(123, &options));
    assert_eq!(geometry3.config, ConfigurationType::Calibrated);
    // Using a different random seed may produce different results.
    assert_ne!(geometry1.e, geometry3.e);
}

#[test]
fn estimate_two_view_geometry_uncalibrated_deterministic() {
    set_prng_seed(0);
    let data = noisy_data(false, 5.0, 2);
    let options = TwoViewGeometryOptions::default();

    let geometry1 = estimate(&data, &seeded(42, &options));
    assert_eq!(geometry1.config, ConfigurationType::Uncalibrated);
    let geometry2 = estimate(&data, &seeded(42, &options));
    assert_eq!(geometry2.config, ConfigurationType::Uncalibrated);
    // Using the same random seed should produce identical results.
    assert_eq!(geometry1.f, geometry2.f);

    let geometry3 = estimate(&data, &seeded(123, &options));
    assert_eq!(geometry3.config, ConfigurationType::Uncalibrated);
    // Using a different random seed may produce different results.
    assert_ne!(geometry1.f, geometry3.f);
}

#[test]
fn estimate_two_view_geometry_uncalibrated_degensac() {
    set_prng_seed(0);
    let data = noisy_data(false, 5.0, 2);

    // The DEGENSAC-based fundamental matrix estimation is a drop-in replacement that
    // recovers a valid uncalibrated geometry on a general (non-planar) scene.
    let options = TwoViewGeometryOptions {
        use_degensac: true,
        ..seeded(42, &TwoViewGeometryOptions::default())
    };
    let geometry = estimate(&data, &options);
    assert_eq!(geometry.config, ConfigurationType::Uncalibrated);
    // `static_cast<size_t>(inlier_match_ratio * num_points3D * 0.8)`.
    assert!(geometry.inlier_matches.len() >= (0.6 * 500.0 * 0.8) as usize);
}

#[test]
fn estimate_two_view_geometry_planar_or_panoramic_deterministic() {
    set_prng_seed(0);
    let data = noisy_data(true, 5.0, 1);
    let options = TwoViewGeometryOptions {
        force_h_use: true,
        ..TwoViewGeometryOptions::default()
    };

    let geometry1 = estimate(&data, &seeded(42, &options));
    assert_eq!(geometry1.config, ConfigurationType::PlanarOrPanoramic);
    let geometry2 = estimate(&data, &seeded(42, &options));
    assert_eq!(geometry2.config, ConfigurationType::PlanarOrPanoramic);
    // Using the same random seed should produce identical results.
    assert_eq!(geometry1.h, geometry2.h);

    let geometry3 = estimate(&data, &seeded(123, &options));
    // COLMAP checks geometry2's config again here.
    assert_eq!(geometry2.config, ConfigurationType::PlanarOrPanoramic);
    // Using a different random seed may produce different results.
    assert_ne!(geometry1.h, geometry3.h);
}

#[test]
fn two_view_geometry_from_known_relative_pose_nominal() {
    const NUM_TESTS: u32 = 100;
    for seed in 0..NUM_TESTS {
        set_prng_seed(seed);
        let data = create_two_view_geometry_pose_test_data(ConfigurationType::Calibrated);
        let geometry = two_view_geometry_from_known_relative_pose(
            &data.camera1,
            &data.points1,
            &data.camera2,
            &data.points2,
            &data.geometry.cam2_from_cam1.unwrap(),
            &data.geometry.inlier_matches,
            15,
            4.0,
        )
        .unwrap();
        assert_eq!(geometry.cam2_from_cam1, data.geometry.cam2_from_cam1);
        assert_eq!(geometry.e, data.geometry.e);
        assert_eq!(geometry.inlier_matches, data.geometry.inlier_matches);
    }
}

struct RigTwoViewGeometryTestData {
    rig1: Rig,
    rig2: Rig,
    matches: Vec<ImagePairMatches>,
    reconstruction: Reconstruction,
}

fn create_rig_two_view_geometry_test_data(
    synthetic_dataset_options: &SyntheticDatasetOptions,
) -> RigTwoViewGeometryTestData {
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    synthesize_dataset_with_database(synthetic_dataset_options, &mut reconstruction, &database)
        .unwrap();
    assert_eq!(reconstruction.num_rigs(), 2);

    let rig1 = reconstruction.rig(1).unwrap().clone();
    let rig2 = reconstruction.rig(2).unwrap().clone();
    let camera_of = |image_id| {
        let image = reconstruction.image(image_id).unwrap();
        reconstruction.camera(image.camera_id()).unwrap()
    };
    let mut matches: Vec<ImagePairMatches> = Vec::new();
    for (pair_id, pair_matches) in database.read_all_matches().unwrap() {
        let (image_id1, image_id2) = pair_id_to_image_pair(pair_id).unwrap();
        let sensor1 = camera_of(image_id1).sensor_id();
        let sensor2 = camera_of(image_id2).sensor_id();
        if rig1.has_sensor(sensor1) && rig2.has_sensor(sensor2) {
            matches.push(((image_id1, image_id2), pair_matches));
        } else if rig1.has_sensor(sensor2) && rig2.has_sensor(sensor1) {
            // COLMAP swaps the pair but not the match indices.
            matches.push(((image_id2, image_id1), pair_matches));
        }
        // else: Ignore matches between sensors in the same rig.
    }

    RigTwoViewGeometryTestData {
        rig1,
        rig2,
        matches,
        reconstruction,
    }
}

#[test]
fn estimate_rig_two_view_geometries_nominal() {
    set_prng_seed(0);
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 3,
        num_frames_per_rig: 1,
        num_points3d: 200,
        // Use only inlier matches so that pose recovery is exact and thus numerically
        // stable across platforms and build configurations.
        inlier_match_ratio: 1.0,
        camera_has_prior_focal_length: true,
        ..SyntheticDatasetOptions::default()
    };
    let data = create_rig_two_view_geometry_test_data(&synthetic_dataset_options);

    let options = seeded(42, &TwoViewGeometryOptions::default());
    let geometries = estimate_rig_two_view_geometries(
        &data.rig1,
        &data.rig2,
        data.reconstruction.images(),
        data.reconstruction.cameras(),
        &data.matches,
        &options,
    )
    .unwrap();
    assert_eq!(geometries.len(), data.matches.len());
    for ((image_id1, image_id2), geometry) in &geometries {
        assert_eq!(geometry.config, ConfigurationType::CalibratedRig);
        let pose = geometry.cam2_from_cam1.expect("pose");
        let image1 = data.reconstruction.image(*image_id1).unwrap();
        let image2 = data.reconstruction.image(*image_id2).unwrap();
        let expected = image_cam_from_world(&data.reconstruction, image2)
            * image_cam_from_world(&data.reconstruction, image1).inverse();
        assert!(rigid3d_near(&pose, &expected, 1e-2, 1e-3));
        assert!(!geometry.inlier_matches.is_empty());
    }
}

fn multiple_models_options() -> TwoViewGeometryOptions {
    TwoViewGeometryOptions {
        multiple_models: true,
        ..seeded(42, &TwoViewGeometryOptions::default())
    }
}

#[test]
fn estimate_multiple_two_view_geometries_single_geometry() {
    set_prng_seed(0);
    let data =
        create_two_view_geometry_test_data(&two_frames(100), &SyntheticNoiseOptions::default());
    let geometry = estimate(&data, &multiple_models_options());
    assert_eq!(geometry.config, ConfigurationType::Calibrated);
    assert!(!geometry.inlier_matches.is_empty());
}

#[test]
fn estimate_multiple_two_view_geometries_no_geometry() {
    set_prng_seed(0);
    // Too few points.
    let data =
        create_two_view_geometry_test_data(&two_frames(5), &SyntheticNoiseOptions::default());
    let options = TwoViewGeometryOptions {
        // Require too many inliers.
        min_num_inliers: 100,
        ..multiple_models_options()
    };
    let geometry = estimate(&data, &options);
    assert_eq!(geometry.config, ConfigurationType::Degenerate);
    assert_eq!(geometry.inlier_matches.len(), 0);
}

#[test]
fn estimate_multiple_two_view_geometries_multiple_geometries() {
    set_prng_seed(0);
    // Create two separate synthetic datasets with different poses.
    let mut reconstruction1 = Reconstruction::new();
    synthesize_dataset(&two_frames(100), &mut reconstruction1).unwrap();
    let mut reconstruction2 = Reconstruction::new();
    synthesize_dataset(&two_frames(100), &mut reconstruction2).unwrap();

    let image1 = reconstruction1.image(1).unwrap();
    let image2 = reconstruction1.image(2).unwrap();
    let image3 = reconstruction2.image(1).unwrap();
    let image4 = reconstruction2.image(2).unwrap();

    let camera1: Camera = reconstruction1.camera(image1.camera_id()).unwrap().clone();
    let camera2: Camera = reconstruction1.camera(image2.camera_id()).unwrap().clone();
    assert_eq!(
        &camera1,
        reconstruction2.camera(image3.camera_id()).unwrap()
    );
    assert_eq!(
        &camera2,
        reconstruction2.camera(image4.camera_id()).unwrap()
    );

    let (points1, points2, _, matches1) =
        extract_points_and_matches(&reconstruction1, image1, image2);
    let (points3, points4, _, matches2) =
        extract_points_and_matches(&reconstruction2, image3, image4);

    let matches_offset = points1.len() as u32;
    let all_points1: Vec<Vector2d> = points1.iter().chain(&points3).copied().collect();
    let all_points2: Vec<Vector2d> = points2.iter().chain(&points4).copied().collect();
    let mut all_matches: FeatureMatches = matches1.clone();
    for m in &matches2 {
        all_matches.push(FeatureMatch::new(
            m.point2d_idx1 + matches_offset,
            m.point2d_idx2 + matches_offset,
        ));
    }

    let geometry = estimate_two_view_geometry(
        &camera1,
        &all_points1,
        &camera2,
        &all_points2,
        all_matches,
        &multiple_models_options(),
    )
    .unwrap();
    assert_eq!(geometry.config, ConfigurationType::Multiple);
    assert_eq!(
        geometry.inlier_matches.len(),
        matches1.len() + matches2.len()
    );
}

fn extract_geometry(cache: &DatabaseCache) -> TwoViewGeometry {
    cache
        .correspondence_graph()
        .extract_two_view_geometry(1, 2, false)
        .unwrap()
}

#[test]
fn maybe_decompose_relative_poses_nominal() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    synthesize_dataset_with_database(&two_frames(50), &mut reconstruction, &database).unwrap();

    // Load the database into a cache.
    let mut cache = DatabaseCache::create(&database, &DatabaseCacheOptions::default()).unwrap();

    // Verify the two-view geometry exists but has no decomposed pose yet.
    assert!(extract_geometry(&cache).cam2_from_cam1.is_none());

    // Decompose poses - should update cache without throwing.
    maybe_decompose_relative_poses(&mut cache).unwrap();

    // Verify the geometry was updated with a decomposed pose.
    let geometry_after = extract_geometry(&cache);
    assert!(geometry_after.cam2_from_cam1.is_some());

    // Calling again should skip already decomposed geometries.
    maybe_decompose_relative_poses(&mut cache).unwrap();

    let geometry_second = extract_geometry(&cache);
    let after = geometry_after.cam2_from_cam1.unwrap();
    let second = geometry_second.cam2_from_cam1.unwrap();
    assert_eq!(after.rotation.coeffs(), second.rotation.coeffs());
    assert_eq!(after.translation, second.translation);
}

// A pair whose focal a two-view solver recovered is UNCALIBRATED but carries the estimated
// intrinsics in camera1/camera2. MaybeDecomposeRelativePoses must calibrate the rays with
// those, not the camera's stale default focal. The pose survives a wrong focal (it comes
// from E alone); tri_angle, measured between the rays, does not.
#[test]
fn maybe_decompose_relative_poses_uses_solver_estimated_intrinsics() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 2,
        num_points3d: 100,
        camera_has_prior_focal_length: false,
        // A focal well away from the no-prior default of 1.2 * max(width, height), so that
        // using the wrong one is actually observable.
        camera_params: vec![1000.0, 512.0, 384.0, 0.05],
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&synthetic_dataset_options, &mut reconstruction, &database)
        .unwrap();

    let image1 = reconstruction.image(1).unwrap();
    let image2 = reconstruction.image(2).unwrap();
    assert_eq!(image1.camera_id(), image2.camera_id());
    let gt_cam2_from_cam1: Rigid3d = image_cam_from_world(&reconstruction, image2)
        * image_cam_from_world(&reconstruction, image1).inverse();

    // Emulate an intrinsics-estimating solver: the true focal is surfaced via
    // camera1/camera2, while the database camera must not be used to calibrate the rays.
    let true_camera = reconstruction.camera(image1.camera_id()).unwrap().clone();

    let mut geometry = database.read_two_view_geometry(1, 2).unwrap();
    assert_eq!(geometry.config, ConfigurationType::Uncalibrated);
    assert!(geometry.e.is_some());
    geometry.camera1 = Some(true_camera.clone());
    geometry.camera2 = Some(true_camera.clone());
    database.update_two_view_geometry(1, 2, &geometry).unwrap();

    // Reference run: the database camera happens to carry the correct focal.
    let cache_options = DatabaseCacheOptions::default();
    let mut reference_cache = DatabaseCache::create(&database, &cache_options).unwrap();
    maybe_decompose_relative_poses(&mut reference_cache).unwrap();
    let reference_geometry = extract_geometry(&reference_cache);
    let reference_pose = reference_geometry.cam2_from_cam1.expect("reference pose");
    assert!(reference_geometry.tri_angle > 0.0);

    // Give the database camera the focal it would carry without a prior,
    // default_focal_length_factor times the larger image dimension. The result must be
    // unchanged.
    let mut default_camera = true_camera.clone();
    default_camera
        .set_focal_length(1.2 * true_camera.width.max(true_camera.height) as f64)
        .unwrap();
    default_camera.has_prior_focal_length = false;
    assert!(
        (default_camera.focal_length().unwrap() / true_camera.focal_length().unwrap() - 1.0).abs()
            > 0.1
    );
    database.update_camera(&default_camera).unwrap();

    let mut cache = DatabaseCache::create(&database, &cache_options).unwrap();
    let geometry_before = extract_geometry(&cache);
    assert!(geometry_before.cam2_from_cam1.is_none());
    assert!(geometry_before.camera1.is_some());

    maybe_decompose_relative_poses(&mut cache).unwrap();

    let geometry_after = extract_geometry(&cache);
    let pose_after = geometry_after.cam2_from_cam1.expect("pose");

    // The pose is recovered from E alone; the triangulation angle is measured between the
    // calibrated rays.
    assert!(rigid3d_near(&pose_after, &reference_pose, 1e-6, 1e-6));
    expect_near(geometry_after.tri_angle, reference_geometry.tri_angle, 1e-6);

    // Sanity check that the reference itself is meaningful.
    assert!(rigid3d_near(
        &with_unit_translation(&reference_pose),
        &with_unit_translation(&gt_cam2_from_cam1),
        1e-3,
        1e-2
    ));
}

// Regression test for https://github.com/colmap/colmap/issues/4387: older COLMAP databases
// stored the two-view geometry config without persisting the E/F/H matrices.
// MaybeDecomposeRelativePoses must refit the missing matrix from the inlier matches.
#[test]
fn maybe_decompose_relative_poses_missing_matrix_from_old_database() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    synthesize_dataset_with_database(&two_frames(50), &mut reconstruction, &database).unwrap();

    // Simulate an older database that has the configuration but no E matrix.
    let mut geometry_legacy = database.read_two_view_geometry(1, 2).unwrap();
    assert_eq!(geometry_legacy.config, ConfigurationType::Calibrated);
    assert!(geometry_legacy.e.is_some());
    geometry_legacy.e = None;
    database
        .update_two_view_geometry(1, 2, &geometry_legacy)
        .unwrap();

    let mut cache = DatabaseCache::create(&database, &DatabaseCacheOptions::default()).unwrap();
    assert!(extract_geometry(&cache).cam2_from_cam1.is_none());

    maybe_decompose_relative_poses(&mut cache).unwrap();

    assert!(extract_geometry(&cache).cam2_from_cam1.is_some());
}
