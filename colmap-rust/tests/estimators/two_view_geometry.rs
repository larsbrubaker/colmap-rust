// Port of COLMAP's src/colmap/estimators/two_view_geometry_test.cc (same values and
// tolerances), first part: the test fixtures, the EstimateTwoViewGeometryPose cases and the
// EstimateTwoViewGeometry cases for spherical, fisheye and distorted cameras. The focal cases
// are in two_view_geometry_focal.rs, the rest in two_view_geometry_misc.rs. Tier C (RANSAC);
// the pose decomposition cases are Tier B.
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. COLMAP's
// ExtractPointsAndMatches walks `reconstruction.Points3D()`, a hash map; the Rust
// Reconstruction iterates its points in ascending id order, so the match order (and with it
// the RANSAC samples) can differ from COLMAP's. The assertions are outcome checks (Tier C).

use colmap_rust::estimators::two_view_geometry::{
    estimate_calibrated_two_view_geometry, estimate_two_view_geometry,
    estimate_two_view_geometry_pose, TwoViewGeometryOptions,
};
use colmap_rust::feature::types::{FeatureMatch, FeatureMatches};
use colmap_rust::geometry::essential_matrix::{
    essential_matrix_from_pose, fundamental_from_essential_matrix,
};
use colmap_rust::geometry::homography_matrix::homography_matrix_from_pose;
use colmap_rust::geometry::triangulation::calculate_triangulation_angles;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Vector2d, Vector3d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::utils::median;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::scene::synthetic_noise::{synthesize_noise, SyntheticNoiseOptions};
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use colmap_rust::sensor::models::CameraModelId;

use crate::transform_matchers::rigid3d_near;

pub fn image_cam_from_world(reconstruction: &Reconstruction, image: &Image) -> Rigid3d {
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    image.cam_from_world(frame, rig).unwrap()
}

fn image_projection_center(reconstruction: &Reconstruction, image: &Image) -> Vector3d {
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    image.projection_center(frame, rig).unwrap()
}

fn image_viewing_direction(reconstruction: &Reconstruction, image: &Image) -> Vector3d {
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    image.viewing_direction(frame, rig).unwrap()
}

/// `ExtractPointsAndMatches`: the 2D points of both images and one match per (two-view)
/// track, with the tracks' 3D points.
pub fn extract_points_and_matches(
    reconstruction: &Reconstruction,
    image1: &Image,
    image2: &Image,
) -> (Vec<Vector2d>, Vec<Vector2d>, Vec<Vector3d>, FeatureMatches) {
    let points1: Vec<Vector2d> = image1.points2d().iter().map(|p| p.xy).collect();
    let points2: Vec<Vector2d> = image2.points2d().iter().map(|p| p.xy).collect();
    let mut points3d = Vec::new();
    let mut matches = FeatureMatches::new();
    for point3d in reconstruction.points3d().values() {
        let track = &point3d.track;
        assert_eq!(track.length(), 2);
        points3d.push(point3d.xyz);
        let elem1 = track.element(0);
        let elem2 = track.element(1);
        let (idx1, idx2) =
            if elem1.image_id == image1.image_id() && elem2.image_id == image2.image_id() {
                (elem1.point2d_idx, elem2.point2d_idx)
            } else if elem1.image_id == image2.image_id() && elem2.image_id == image1.image_id() {
                (elem2.point2d_idx, elem1.point2d_idx)
            } else {
                panic!("Invalid track element.");
            };
        matches.push(FeatureMatch::new(idx1, idx2));
    }
    (points1, points2, points3d, matches)
}

pub struct TwoViewGeometryPoseTestData {
    pub camera1: Camera,
    pub camera2: Camera,
    pub points1: Vec<Vector2d>,
    pub points2: Vec<Vector2d>,
    pub geometry: TwoViewGeometry,
}

pub fn create_two_view_geometry_pose_test_data(
    config: ConfigurationType,
) -> TwoViewGeometryPoseTestData {
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d: 50,
        camera_has_prior_focal_length: true,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();

    let image1 = reconstruction.image(1).unwrap();
    let image2 = reconstruction.image(2).unwrap();

    let camera1 = reconstruction.camera(image1.camera_id()).unwrap().clone();
    let camera2 = reconstruction.camera(image2.camera_id()).unwrap().clone();
    let mut geometry = TwoViewGeometry {
        config,
        ..TwoViewGeometry::default()
    };
    let cam1_from_world = image_cam_from_world(&reconstruction, image1);
    let mut cam2_from_cam1 =
        image_cam_from_world(&reconstruction, image2) * cam1_from_world.inverse();
    let k1 = camera1.calibration_matrix().unwrap();
    let k2 = camera2.calibration_matrix().unwrap();

    match config {
        ConfigurationType::Calibrated => {
            geometry.e = Some(essential_matrix_from_pose(&cam2_from_cam1));
        }
        ConfigurationType::Uncalibrated => {
            geometry.f = Some(fundamental_from_essential_matrix(
                &k2,
                &essential_matrix_from_pose(&cam2_from_cam1),
                &k1,
            ));
        }
        ConfigurationType::Planar => {
            let homography_plane_normal = cam1_from_world.rotation
                * -(image_viewing_direction(&reconstruction, image1)
                    + image_viewing_direction(&reconstruction, image2))
                .normalized();
            const HOMOGRAPHY_PLANE_DISTANCE: f64 = 1.0;
            geometry.h = Some(
                homography_matrix_from_pose(
                    &k1,
                    &k2,
                    &cam2_from_cam1.rotation.to_rotation_matrix(),
                    cam2_from_cam1.translation,
                    homography_plane_normal,
                    HOMOGRAPHY_PLANE_DISTANCE,
                )
                .unwrap(),
            );
        }
        ConfigurationType::Panoramic => {
            cam2_from_cam1.translation = Vector3d::zeros();
            geometry.h = Some(
                homography_matrix_from_pose(
                    &k1,
                    &k2,
                    &cam2_from_cam1.rotation.to_rotation_matrix(),
                    cam2_from_cam1.translation,
                    Vector3d::new(0.0, 0.0, 1.0),
                    1.0,
                )
                .unwrap(),
            );
        }
        _ => panic!("Invalid configuration."),
    }
    geometry.cam2_from_cam1 = Some(cam2_from_cam1);

    let (points1, points2, points3d, matches) =
        extract_points_and_matches(&reconstruction, image1, image2);
    geometry.inlier_matches = matches;

    geometry.tri_angle = if config == ConfigurationType::Panoramic {
        0.0
    } else {
        let mut angles = calculate_triangulation_angles(
            image_projection_center(&reconstruction, image1),
            image_projection_center(&reconstruction, image2),
            &points3d,
        );
        median(&mut angles).unwrap()
    };

    TwoViewGeometryPoseTestData {
        camera1,
        camera2,
        points1,
        points2,
        geometry,
    }
}

fn check_equal_two_view_geometry(
    geometry: &TwoViewGeometry,
    expected_geometry: &TwoViewGeometry,
    tri_angle_tol: f64,
    rotation_tol: f64,
    translation_tol: f64,
    normalized_translation: bool,
) -> bool {
    let pose = geometry.cam2_from_cam1.unwrap();
    let expected_pose = expected_geometry.cam2_from_cam1.unwrap();
    let tri_angle_error = (geometry.tri_angle - expected_geometry.tri_angle).abs();
    let rotation_error = pose.rotation.angular_distance(expected_pose.rotation);
    let expected_translation = if normalized_translation {
        expected_pose.translation.normalized()
    } else {
        expected_pose.translation
    };
    let translation_error = (pose.translation - expected_translation).norm();
    if tri_angle_error > tri_angle_tol
        || rotation_error > rotation_tol
        || translation_error > translation_tol
    {
        eprintln!(
            "Two view geometries do not match with errors: tri_angle={tri_angle_error}, \
             rotation={rotation_error}, translation={translation_error}"
        );
        return false;
    }
    true
}

/// The geometry the pose tests hand the estimator: the test data's config, the given model
/// and its inlier matches.
fn input_geometry(
    data: &TwoViewGeometryPoseTestData,
    config: ConfigurationType,
) -> TwoViewGeometry {
    TwoViewGeometry {
        config,
        e: data.geometry.e,
        f: data.geometry.f,
        h: data.geometry.h,
        inlier_matches: data.geometry.inlier_matches.clone(),
        ..TwoViewGeometry::default()
    }
}

fn run_pose_tests(
    config: ConfigurationType,
    tri_angle_tol: f64,
    translation_tol: f64,
    normalized_translation: bool,
) {
    const NUM_TESTS: u32 = 100;
    let mut num_failures = 0;
    for seed in 0..NUM_TESTS {
        set_prng_seed(seed);
        let data = create_two_view_geometry_pose_test_data(config);
        let mut geometry = input_geometry(&data, config);
        assert!(estimate_two_view_geometry_pose(
            &data.camera1,
            &data.points1,
            &data.camera2,
            &data.points2,
            &mut geometry,
        )
        .unwrap());
        if !check_equal_two_view_geometry(
            &geometry,
            &data.geometry,
            tri_angle_tol,
            1e-6,
            translation_tol,
            normalized_translation,
        ) {
            num_failures += 1;
        }
    }
    assert_eq!(num_failures, 0);
}

#[test]
fn estimate_two_view_geometry_pose_calibrated() {
    run_pose_tests(ConfigurationType::Calibrated, 1e-6, 1e-6, true);
}

#[test]
fn estimate_two_view_geometry_pose_failure_due_to_insufficient_matches() {
    set_prng_seed(0);
    for config in [
        ConfigurationType::Calibrated,
        ConfigurationType::Uncalibrated,
        ConfigurationType::Planar,
        ConfigurationType::Panoramic,
    ] {
        let mut data = create_two_view_geometry_pose_test_data(config);
        data.geometry.inlier_matches.clear();
        // COLMAP copies only E here, whatever the configuration.
        let mut geometry = TwoViewGeometry {
            config: data.geometry.config,
            e: data.geometry.e,
            inlier_matches: data.geometry.inlier_matches.clone(),
            ..TwoViewGeometry::default()
        };
        assert!(
            !estimate_two_view_geometry_pose(
                &data.camera1,
                &data.points1,
                &data.camera2,
                &data.points2,
                &mut geometry,
            )
            .unwrap(),
            "{config:?}"
        );
    }
}

#[test]
fn estimate_two_view_geometry_pose_uncalibrated() {
    run_pose_tests(ConfigurationType::Uncalibrated, 1e-6, 1e-6, true);
}

#[test]
fn estimate_two_view_geometry_pose_planar() {
    run_pose_tests(ConfigurationType::Planar, 1e-3, 1e-5, false);
}

#[test]
fn estimate_two_view_geometry_pose_planar_or_panoramic() {
    const NUM_TESTS: u32 = 100;
    let mut num_failures = 0;
    for seed in 0..NUM_TESTS {
        set_prng_seed(seed);
        for config in [ConfigurationType::Planar, ConfigurationType::Panoramic] {
            let data = create_two_view_geometry_pose_test_data(config);
            let mut geometry = input_geometry(&data, ConfigurationType::PlanarOrPanoramic);
            geometry.e = None;
            geometry.f = None;
            assert!(estimate_two_view_geometry_pose(
                &data.camera1,
                &data.points1,
                &data.camera2,
                &data.points2,
                &mut geometry,
            )
            .unwrap());
            assert_eq!(geometry.config, config);
            if !check_equal_two_view_geometry(&geometry, &data.geometry, 1e-3, 1e-6, 1e-6, false) {
                num_failures += 1;
            }
        }
    }
    assert_eq!(num_failures, 0);
}

pub struct TwoViewGeometryTestData {
    pub camera1: Camera,
    pub camera2: Camera,
    pub points1: Vec<Vector2d>,
    pub points2: Vec<Vector2d>,
    pub matches: FeatureMatches,
    /// Ground-truth relative pose (cam2_from_cam1) of the synthetic dataset.
    pub cam2_from_cam1: Rigid3d,
}

pub fn create_two_view_geometry_test_data(
    synthetic_dataset_options: &SyntheticDatasetOptions,
    synthetic_noise_options: &SyntheticNoiseOptions,
) -> TwoViewGeometryTestData {
    let mut reconstruction = Reconstruction::new();
    synthesize_dataset(synthetic_dataset_options, &mut reconstruction).unwrap();
    synthesize_noise(synthetic_noise_options, &mut reconstruction).unwrap();

    assert_eq!(reconstruction.num_images(), 2);
    let image1 = reconstruction.image(1).unwrap();
    let image2 = reconstruction.image(2).unwrap();
    let (points1, points2, _points3d, matches) =
        extract_points_and_matches(&reconstruction, image1, image2);
    TwoViewGeometryTestData {
        camera1: reconstruction.camera(image1.camera_id()).unwrap().clone(),
        camera2: reconstruction.camera(image2.camera_id()).unwrap().clone(),
        points1,
        points2,
        matches,
        cam2_from_cam1: image_cam_from_world(&reconstruction, image2)
            * image_cam_from_world(&reconstruction, image1).inverse(),
    }
}

/// `Rigid3d(pose.rotation(), pose.translation().normalized())`.
pub fn with_unit_translation(pose: &Rigid3d) -> Rigid3d {
    Rigid3d::new(pose.rotation, pose.translation.normalized())
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

fn two_frame_options() -> SyntheticDatasetOptions {
    SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d: 200,
        ..SyntheticDatasetOptions::default()
    }
}

#[test]
fn estimate_two_view_geometry_spherical() {
    set_prng_seed(0);
    let synthetic_dataset_options = SyntheticDatasetOptions {
        camera_model_id: CameraModelId::Equirectangular,
        camera_width: 1000,
        camera_height: 500,
        camera_params: vec![1000.0, 500.0],
        ..two_frame_options()
    };
    let data = create_two_view_geometry_test_data(
        &synthetic_dataset_options,
        &SyntheticNoiseOptions::default(),
    );
    assert!(!data.camera1.is_perspective());
    assert!(!data.camera2.is_perspective());

    let options = TwoViewGeometryOptions {
        compute_relative_pose: true,
        ..TwoViewGeometryOptions::default()
    };

    // Spherical cameras have no pinhole image plane, so the fundamental matrix is not
    // estimated; the pair is classified from the bearing-based essential matrix and a
    // ray-space homography.
    let geometry = estimate(&data, &options);
    assert_eq!(geometry.config, ConfigurationType::Calibrated);
    assert!(geometry.e.is_some());
    assert!(geometry.f.is_none());
    assert!(geometry.h.is_some());
    assert!(geometry.inlier_matches.len() >= data.matches.len() / 2);

    // Rotation exactly, translation up to scale.
    let pose = geometry.cam2_from_cam1.expect("pose");
    assert!(rigid3d_near(
        &with_unit_translation(&pose),
        &with_unit_translation(&data.cam2_from_cam1),
        1e-3,
        1e-2
    ));

    // EstimateCalibratedTwoViewGeometry delegates to the spherical path rather than
    // estimating a meaningless fundamental matrix.
    let calibrated_geometry = estimate_calibrated_two_view_geometry(
        &data.camera1,
        &data.points1,
        &data.camera2,
        &data.points2,
        &data.matches,
        &options,
    )
    .unwrap();
    assert_eq!(calibrated_geometry.config, ConfigurationType::Calibrated);
    assert!(calibrated_geometry.e.is_some());
    assert!(calibrated_geometry.f.is_none());
    assert!(calibrated_geometry.h.is_some());
}

fn fisheye_options(has_prior_focal_length: bool) -> SyntheticDatasetOptions {
    SyntheticDatasetOptions {
        camera_model_id: CameraModelId::OpenCVFisheye,
        camera_params: vec![1280.0, 1280.0, 512.0, 384.0, 0.0, 0.0, 0.0, 0.0],
        camera_has_prior_focal_length: has_prior_focal_length,
        ..two_frame_options()
    }
}

#[test]
fn estimate_two_view_geometry_uncalibrated_fisheye_is_degenerate() {
    set_prng_seed(0);
    let data = create_two_view_geometry_test_data(
        &fisheye_options(false),
        &SyntheticNoiseOptions::default(),
    );
    assert!(data.camera1.is_perspective_fisheye());
    assert!(!data.camera1.has_prior_focal_length);

    // Without a focal length prior the only remaining model is the fundamental matrix, which
    // assumes a pinhole projection that a fisheye camera does not have.
    let geometry = estimate(&data, &TwoViewGeometryOptions::default());
    assert_eq!(geometry.config, ConfigurationType::Degenerate);
    assert!(geometry.f.is_none());
    assert!(geometry.h.is_none());
}

#[test]
fn estimate_two_view_geometry_calibrated_fisheye_is_estimated() {
    set_prng_seed(0);
    let data = create_two_view_geometry_test_data(
        &fisheye_options(true),
        &SyntheticNoiseOptions::default(),
    );
    assert!(data.camera1.is_perspective_fisheye());
    assert!(data.camera1.has_prior_focal_length);

    // With a known focal length the calibrated path applies, which works on bearing vectors
    // and is therefore valid for fisheye cameras.
    let geometry = estimate(&data, &TwoViewGeometryOptions::default());
    assert_eq!(geometry.config, ConfigurationType::Calibrated);
    assert!(geometry.e.is_some());
    assert!(geometry.inlier_matches.len() >= data.matches.len() / 2);
}

#[test]
fn estimate_two_view_geometry_force_h_use_with_fisheye_is_degenerate() {
    set_prng_seed(0);
    let data = create_two_view_geometry_test_data(
        &fisheye_options(true),
        &SyntheticNoiseOptions::default(),
    );
    assert!(data.camera1.is_perspective_fisheye());

    // A homography only relates two images of a plane under a pinhole projection, so it is
    // not estimated for fisheye cameras even when the caller explicitly asks for it.
    let options = TwoViewGeometryOptions {
        force_h_use: true,
        ..TwoViewGeometryOptions::default()
    };
    let geometry = estimate(&data, &options);
    assert_eq!(geometry.config, ConfigurationType::Degenerate);
    assert!(geometry.h.is_none());
}

/// A pure-rotation pair (one rig, two cameras with no relative translation) of 500 points.
fn panoramic_options(
    camera_model_id: CameraModelId,
    camera_params: Vec<f64>,
) -> SyntheticDatasetOptions {
    SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 2,
        num_frames_per_rig: 1,
        num_points3d: 500,
        camera_model_id,
        camera_params,
        camera_has_prior_focal_length: true,
        sensor_from_rig_translation_stddev: 0.0,
        ..SyntheticDatasetOptions::default()
    }
}

fn panoramic_geometry(data: &TwoViewGeometryTestData) -> TwoViewGeometry {
    let mut options = TwoViewGeometryOptions {
        compute_relative_pose: true,
        ..TwoViewGeometryOptions::default()
    };
    options.ransac_options.random_seed = 42;
    estimate(data, &options)
}

// A rotating 360 degree camera is the usual capture mode, and the essential matrix
// degenerates there, so the spherical path must catch it.
#[test]
fn estimate_two_view_geometry_panoramic_with_spherical_camera_is_detected() {
    set_prng_seed(42);
    let synthetic_dataset_options = SyntheticDatasetOptions {
        camera_width: 1000,
        camera_height: 500,
        camera_has_prior_focal_length: SyntheticDatasetOptions::default()
            .camera_has_prior_focal_length,
        ..panoramic_options(CameraModelId::Equirectangular, vec![1000.0, 500.0])
    };
    let data = create_two_view_geometry_test_data(
        &synthetic_dataset_options,
        &SyntheticNoiseOptions::default(),
    );
    assert!(data.camera1.is_spherical());

    let geometry = panoramic_geometry(&data);
    assert_eq!(geometry.config, ConfigurationType::Panoramic);
    assert!(geometry.e.is_some());
    assert!(geometry.h.is_some());
    assert!(geometry.f.is_none());
    assert!(geometry.inlier_matches.len() >= data.matches.len() / 2);
}

// With a strongly distorted camera the pixel-space homography cannot fit a pure-rotation
// pair, so it is estimated on bearing rays.
#[test]
fn estimate_two_view_geometry_panoramic_with_distorted_camera_is_detected() {
    set_prng_seed(42);
    let data = create_two_view_geometry_test_data(
        &panoramic_options(
            CameraModelId::OpenCV,
            vec![1280.0, 1280.0, 512.0, 384.0, 0.15, -0.03, 0.001, 0.001],
        ),
        &SyntheticNoiseOptions::default(),
    );
    assert!(data.camera1.is_perspective_pinhole());
    assert!(!data.camera1.is_undistorted().unwrap());

    let geometry = panoramic_geometry(&data);
    assert_eq!(geometry.config, ConfigurationType::Panoramic);
    assert!(geometry.h.is_some());
    assert!(geometry.inlier_matches.len() >= data.matches.len() / 2);
}

// As above for a fisheye camera, which reaches the calibrated path because it
// short-circuits only spherical models.
#[test]
fn estimate_two_view_geometry_panoramic_with_fisheye_camera_is_detected() {
    set_prng_seed(42);
    let data = create_two_view_geometry_test_data(
        &panoramic_options(
            CameraModelId::OpenCVFisheye,
            vec![1280.0, 1280.0, 512.0, 384.0, 0.05, -0.01, 0.0, 0.0],
        ),
        &SyntheticNoiseOptions::default(),
    );
    assert!(data.camera1.is_perspective_fisheye());

    let geometry = panoramic_geometry(&data);
    assert_eq!(geometry.config, ConfigurationType::Panoramic);
    assert!(geometry.h.is_some());
    assert!(geometry.inlier_matches.len() >= data.matches.len() / 2);
}
