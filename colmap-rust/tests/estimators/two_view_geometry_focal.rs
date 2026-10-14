// Port of COLMAP's src/colmap/estimators/two_view_geometry_test.cc (same values and
// tolerances), the focal-recovering EstimateTwoViewGeometry cases: SharedFocal,
// OneSidedFocal, SphericalAndCalibratedPerspective and SphericalAndUncalibratedPerspective.
// Fixtures are in two_view_geometry.rs. Tier C (RANSAC).
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. Arguments COLMAP draws
// inside one expression are drawn here in clang's left-to-right order.

use colmap_rust::estimators::solvers::RelativePoseSharedFocalEstimator;
use colmap_rust::estimators::two_view_geometry::{
    estimate_two_view_geometry, TwoViewGeometryOptions,
};
use colmap_rust::feature::types::{FeatureMatch, FeatureMatches};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Quaterniond, Vector2d, Vector3d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::random_eigen_vector3d;
use colmap_rust::math::utils::deg_to_rad;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::scene::two_view_geometry::ConfigurationType;
use colmap_rust::sensor::models::CameraModelId;

use crate::expect_near;
use crate::transform_matchers::rigid3d_near;
use crate::two_view_geometry::{image_cam_from_world, with_unit_translation};

/// `Rigid3d(Quaterniond(AngleAxisd(DegToRad(U(20, 60)), axis)), RandomEigenVectord<3>()
/// .normalized())`, drawing the axis first.
fn random_bounded_pose() -> Rigid3d {
    let axis = random_eigen_vector3d().normalized();
    let angle = deg_to_rad(random_uniform_real(20.0, 60.0));
    let translation = random_eigen_vector3d().normalized();
    Rigid3d::new(
        Quaterniond::from_angle_axis(AngleAxisd::new(angle, axis)),
        translation,
    )
}

/// Wide-baseline, wide field-of-view correspondences: 200 points in front of the first view
/// (|x/z|, |y/z| <= ~0.5) that project into both cameras, matched index to index.
fn wide_fov_correspondences(
    camera1: &Camera,
    camera2: &Camera,
    cam2_from_cam1: &Rigid3d,
) -> (Vec<Vector2d>, Vec<Vector2d>, FeatureMatches) {
    let mut points1 = Vec::new();
    let mut points2 = Vec::new();
    let mut matches = FeatureMatches::new();
    while points1.len() < 200 {
        let mut dir = random_eigen_vector3d();
        dir.z = dir.z.abs() + 2.0;
        let point_in_cam1 = random_uniform_real(2.0, 5.0) * dir.normalized();
        let point_in_cam2 = *cam2_from_cam1 * point_in_cam1;
        if point_in_cam2.z < 0.5 {
            continue; // Require cheirality in cam2 (cam1 holds by construction).
        }
        let (Some(xy1), Some(xy2)) = (
            camera1.img_from_cam(point_in_cam1, true),
            camera2.img_from_cam(point_in_cam2, true),
        ) else {
            continue;
        };
        let idx = points1.len() as u32;
        points1.push(xy1);
        points2.push(xy2);
        matches.push(FeatureMatch::new(idx, idx));
    }
    (points1, points2, matches)
}

fn relative_pose_options() -> TwoViewGeometryOptions {
    TwoViewGeometryOptions {
        compute_relative_pose: true,
        ..TwoViewGeometryOptions::default()
    }
}

#[test]
fn estimate_two_view_geometry_shared_focal() {
    set_prng_seed(0);
    // A single shared, uncalibrated, pinhole-projection camera observed from two frames
    // routes to the shared-focal solver, which jointly recovers the relative pose and the
    // unknown focal length (SIMPLE_PINHOLE, and PINHOLE seeded fx = fy = f).
    const FOCAL: f64 = 1280.0;
    for model_id in [CameraModelId::SimplePinhole, CameraModelId::Pinhole] {
        let mut camera = Camera::create_from_model_id(1, model_id, FOCAL, 2048, 2048).unwrap();
        camera.has_prior_focal_length = false;
        assert!(camera.is_perspective());

        // The focal is unidentifiable for singular optical axis configurations, so resample
        // until the pose is clear of that degeneracy, using the same test as the estimator.
        let cam2_from_cam1 = loop {
            let pose = random_bounded_pose();
            if RelativePoseSharedFocalEstimator::is_focal_identifiable(&pose) {
                break pose;
            }
        };
        let (points1, points2, matches) =
            wide_fov_correspondences(&camera, &camera, &cam2_from_cam1);

        let geometry = estimate_two_view_geometry(
            &camera,
            &points1,
            &camera,
            &points2,
            matches.clone(),
            &relative_pose_options(),
        )
        .unwrap();

        // The shared-focal solver reports an UNCALIBRATED pair carrying an essential matrix
        // and the estimated intrinsics in camera1/camera2.
        assert_eq!(geometry.config, ConfigurationType::Uncalibrated);
        assert!(geometry.e.is_some());
        let estimated1 = geometry.camera1.as_ref().expect("camera1");
        let estimated2 = geometry.camera2.as_ref().expect("camera2");
        // Both views share a single camera in this configuration.
        assert_eq!(estimated1, estimated2);
        assert!(geometry.inlier_matches.len() >= matches.len() / 2);
        expect_near(estimated1.focal_length_x().unwrap(), FOCAL, 0.05 * FOCAL);

        let pose = geometry.cam2_from_cam1.expect("pose");
        assert!(rigid3d_near(
            &with_unit_translation(&pose),
            &with_unit_translation(&cam2_from_cam1),
            1e-2,
            1e-1
        ));
    }
}

/// Asserts that F satisfies x2^T F x1 = 0 on the exact correspondences, in raw image
/// coordinates and in the given order (distance from x2 to its epipolar line, in pixels).
fn expect_valid_f(f: &Matrix3d, points_first: &[Vector2d], points_second: &[Vector2d]) {
    for (p1, p2) in points_first.iter().zip(points_second) {
        let x1 = Vector3d::new(p1.x, p1.y, 1.0);
        let x2 = Vector3d::new(p2.x, p2.y, 1.0);
        let line = *f * x1;
        let line_norm = Vector2d::new(line.x, line.y).norm();
        assert!(line_norm > 0.0);
        expect_near(x2.dot(line).abs() / line_norm, 0.0, 1e-6);
    }
}

#[test]
fn estimate_two_view_geometry_one_sided_focal() {
    set_prng_seed(0);
    // Two distinct cameras where exactly one has a known focal length route to the one-sided
    // focal solver. The pair is run in both orders, covering the internal swap.
    const UNCALIB_FOCAL: f64 = 1280.0;
    const CALIB_FOCAL: f64 = 900.0;
    for model_id in [CameraModelId::SimplePinhole, CameraModelId::Pinhole] {
        let mut uncalib_camera =
            Camera::create_from_model_id(1, model_id, UNCALIB_FOCAL, 2048, 2048).unwrap();
        uncalib_camera.has_prior_focal_length = false;
        let mut calib_camera =
            Camera::create_from_model_id(2, model_id, CALIB_FOCAL, 2048, 2048).unwrap();
        calib_camera.has_prior_focal_length = true;

        // No rejection sampling is needed: the known focal removes the coplanar-axes
        // singularity.
        let calib_from_uncalib = random_bounded_pose();
        let (uncalib_points, calib_points, matches) =
            wide_fov_correspondences(&uncalib_camera, &calib_camera, &calib_from_uncalib);

        // Order 1: the uncalibrated view first, the solver's native orientation.
        {
            let geometry = estimate_two_view_geometry(
                &uncalib_camera,
                &uncalib_points,
                &calib_camera,
                &calib_points,
                matches.clone(),
                &relative_pose_options(),
            )
            .unwrap();
            assert_eq!(geometry.config, ConfigurationType::Uncalibrated);
            assert!(geometry.e.is_some());
            // Only the uncalibrated side carries estimated intrinsics.
            let estimated1 = geometry.camera1.as_ref().expect("camera1");
            assert!(geometry.camera2.is_none());
            assert!(geometry.inlier_matches.len() >= matches.len() / 2);
            expect_near(
                estimated1.focal_length_x().unwrap(),
                UNCALIB_FOCAL,
                0.05 * UNCALIB_FOCAL,
            );
            expect_valid_f(&geometry.f.expect("F"), &uncalib_points, &calib_points);
            let pose = geometry.cam2_from_cam1.expect("pose");
            assert!(rigid3d_near(
                &with_unit_translation(&pose),
                &with_unit_translation(&calib_from_uncalib),
                1e-2,
                1e-1
            ));
        }

        // Order 2: the calibrated view first, exercising the swap-and-invert path.
        {
            let geometry = estimate_two_view_geometry(
                &calib_camera,
                &calib_points,
                &uncalib_camera,
                &uncalib_points,
                matches.clone(),
                &relative_pose_options(),
            )
            .unwrap();
            assert_eq!(geometry.config, ConfigurationType::Uncalibrated);
            assert!(geometry.e.is_some());
            // The estimated intrinsics follow the uncalibrated view, now the second one.
            assert!(geometry.camera1.is_none());
            let estimated2 = geometry.camera2.as_ref().expect("camera2");
            assert!(geometry.inlier_matches.len() >= matches.len() / 2);
            expect_near(
                estimated2.focal_length_x().unwrap(),
                UNCALIB_FOCAL,
                0.05 * UNCALIB_FOCAL,
            );
            expect_valid_f(&geometry.f.expect("F"), &calib_points, &uncalib_points);
            let uncalib_from_calib = calib_from_uncalib.inverse();
            let pose = geometry.cam2_from_cam1.expect("pose");
            assert!(rigid3d_near(
                &with_unit_translation(&pose),
                &with_unit_translation(&uncalib_from_calib),
                1e-2,
                1e-1
            ));
        }
    }
}

/// A perspective two-view dataset whose first image is re-projected through an
/// EQUIRECTANGULAR camera: `(reconstruction, spherical points, perspective points, matches,
/// perspective camera, perspective_from_spherical)`.
struct MixedPair {
    spherical_camera: Camera,
    perspective_camera: Camera,
    spherical_points: Vec<Vector2d>,
    perspective_points: Vec<Vector2d>,
    matches: FeatureMatches,
    perspective_from_spherical: Rigid3d,
}

fn mixed_spherical_perspective_pair() -> MixedPair {
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d: 200,
        camera_has_prior_focal_length: true,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();

    let image1 = reconstruction.image(1).unwrap();
    let image2 = reconstruction.image(2).unwrap();
    let perspective_camera = reconstruction.camera(image2.camera_id()).unwrap().clone();
    let spherical_camera =
        Camera::create_from_model_id(100, CameraModelId::Equirectangular, 0.0, 1000, 500).unwrap();
    assert!(spherical_camera.is_spherical());
    assert!(perspective_camera.is_perspective());

    let cam1_from_world = image_cam_from_world(&reconstruction, image1);
    let cam2_from_world = image_cam_from_world(&reconstruction, image2);
    let mut spherical_points = Vec::new();
    let mut perspective_points = Vec::new();
    let mut matches = FeatureMatches::new();
    for point3d in reconstruction.points3d().values() {
        let (Some(xy1), Some(xy2)) = (
            spherical_camera.img_from_cam(cam1_from_world * point3d.xyz, true),
            perspective_camera.img_from_cam(cam2_from_world * point3d.xyz, true),
        ) else {
            continue;
        };
        matches.push(FeatureMatch::new(
            spherical_points.len() as u32,
            perspective_points.len() as u32,
        ));
        spherical_points.push(xy1);
        perspective_points.push(xy2);
    }
    assert!(matches.len() >= 50);

    MixedPair {
        spherical_camera,
        perspective_camera,
        spherical_points,
        perspective_points,
        matches,
        perspective_from_spherical: cam2_from_world * cam1_from_world.inverse(),
    }
}

#[test]
fn estimate_two_view_geometry_spherical_and_calibrated_perspective() {
    set_prng_seed(0);
    // Both sides are calibrated: the pair routes to the bearing-based essential matrix path,
    // committing to CALIBRATED without estimating a fundamental matrix.
    let pair = mixed_spherical_perspective_pair();
    let geometry = estimate_two_view_geometry(
        &pair.spherical_camera,
        &pair.spherical_points,
        &pair.perspective_camera,
        &pair.perspective_points,
        pair.matches.clone(),
        &TwoViewGeometryOptions::default(),
    )
    .unwrap();
    assert_eq!(geometry.config, ConfigurationType::Calibrated);
    assert!(geometry.e.is_some());
    assert!(geometry.f.is_none());
    assert!(geometry.h.is_some());
    assert!(geometry.inlier_matches.len() >= pair.matches.len() / 2);
}

#[test]
fn estimate_two_view_geometry_spherical_and_uncalibrated_perspective() {
    set_prng_seed(0);
    // The spherical side is calibrated by construction, so the pair routes to the one-sided
    // focal solver and the perspective camera's focal is recovered rather than guessed.
    let pair = mixed_spherical_perspective_pair();
    let true_focal = pair.perspective_camera.focal_length_x().unwrap();

    // The estimator is handed a camera carrying a deliberately wrong focal and no prior, so
    // the recovered focal cannot merely echo its input.
    let mut uncalib_camera = pair.perspective_camera.clone();
    uncalib_camera.has_prior_focal_length = false;
    uncalib_camera.set_focal_length(0.5 * true_focal).unwrap();

    // A spherical view has no image plane, so there is no F or H to publish. Both orders are
    // run; the spherical-first case exercises the internal swap.
    {
        let geometry = estimate_two_view_geometry(
            &pair.spherical_camera,
            &pair.spherical_points,
            &uncalib_camera,
            &pair.perspective_points,
            pair.matches.clone(),
            &relative_pose_options(),
        )
        .unwrap();
        assert_eq!(geometry.config, ConfigurationType::Uncalibrated);
        assert!(geometry.e.is_some());
        assert!(geometry.f.is_none());
        assert!(geometry.h.is_none());
        // The estimated intrinsics follow the uncalibrated view, here the second.
        assert!(geometry.camera1.is_none());
        let estimated2 = geometry.camera2.as_ref().expect("camera2");
        expect_near(
            estimated2.focal_length_x().unwrap(),
            true_focal,
            0.05 * true_focal,
        );
        assert!(geometry.inlier_matches.len() >= pair.matches.len() / 2);
        let pose = geometry.cam2_from_cam1.expect("pose");
        assert!(rigid3d_near(
            &with_unit_translation(&pose),
            &with_unit_translation(&pair.perspective_from_spherical),
            1e-2,
            1e-1
        ));
    }

    {
        let swapped_matches: FeatureMatches = pair
            .matches
            .iter()
            .map(|m| FeatureMatch::new(m.point2d_idx2, m.point2d_idx1))
            .collect();
        let geometry = estimate_two_view_geometry(
            &uncalib_camera,
            &pair.perspective_points,
            &pair.spherical_camera,
            &pair.spherical_points,
            swapped_matches,
            &relative_pose_options(),
        )
        .unwrap();
        assert_eq!(geometry.config, ConfigurationType::Uncalibrated);
        assert!(geometry.e.is_some());
        assert!(geometry.f.is_none());
        assert!(geometry.h.is_none());
        let estimated1 = geometry.camera1.as_ref().expect("camera1");
        assert!(geometry.camera2.is_none());
        expect_near(
            estimated1.focal_length_x().unwrap(),
            true_focal,
            0.05 * true_focal,
        );
        let spherical_from_perspective = pair.perspective_from_spherical.inverse();
        let pose = geometry.cam2_from_cam1.expect("pose");
        assert!(rigid3d_near(
            &with_unit_translation(&pose),
            &with_unit_translation(&spherical_from_perspective),
            1e-2,
            1e-1
        ));
    }
}
