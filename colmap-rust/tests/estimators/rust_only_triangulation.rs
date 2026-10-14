// Rust-only tests of colmap_rust::estimators::triangulation, ported from colmap-sharp's
// ColmapSharp.Tests/Estimators/TriangulationEstimationTests.cs. COLMAP has no
// estimators/triangulation_test.cc (EstimateTriangulation is covered upstream only through the
// incremental mapper tests and pycolmap's binding smoke tests), so these pin the behavior on
// noise-free synthetic scenes: exact recovery, outlier rejection, omnidirectional cameras and
// the triangulation-angle requirement. Tier C (RANSAC).

use colmap_rust::estimators::triangulation::{
    estimate_triangulation, EstimateTriangulationOptions, ResidualType,
};
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Quaterniond, Vector2d, Vector3d};
use colmap_rust::math::deg_to_rad;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::point3d::Point3D;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::sensor::models::CameraModelId;

// The observations of one synthetic 3D point: pixels, poses and cameras of its track.
fn track_observations<'a>(
    reconstruction: &'a Reconstruction,
    point3d: &Point3D,
) -> (Vec<Vector2d>, Vec<Rigid3d>, Vec<&'a Camera>) {
    let mut points = Vec::new();
    let mut cams_from_world = Vec::new();
    let mut cameras = Vec::new();
    for element in point3d.track.elements() {
        let image = reconstruction.image(element.image_id).unwrap();
        let frame = reconstruction.frame(image.frame_id()).unwrap();
        let rig = reconstruction.rig(frame.rig_id()).unwrap();
        points.push(image.point2d(element.point2d_idx).xy);
        cams_from_world.push(image.cam_from_world(frame, rig).unwrap());
        cameras.push(reconstruction.camera(image.camera_id()).unwrap());
    }
    (points, cams_from_world, cameras)
}

fn synthesize_scene(model_id: CameraModelId) -> Reconstruction {
    let mut reconstruction = Reconstruction::new();
    let mut options = SyntheticDatasetOptions {
        num_rigs: 4,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d: 20,
        camera_model_id: model_id,
        ..SyntheticDatasetOptions::default()
    };
    if model_id == CameraModelId::Equirectangular {
        options.camera_params = Camera::create_from_model_id(
            1,
            model_id,
            0.0,
            options.camera_width,
            options.camera_height,
        )
        .unwrap()
        .params;
    }
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    reconstruction
}

fn nominal(residual_type: ResidualType) {
    let reconstruction = synthesize_scene(CameraModelId::SimpleRadial);
    let mut options = EstimateTriangulationOptions {
        residual_type,
        ..EstimateTriangulationOptions::default()
    };
    if residual_type == ResidualType::ReprojectionError {
        options.ransac_options.max_error = 2.0;
    }

    let mut num_tested = 0;
    for (point3d_id, point3d) in reconstruction.points3d() {
        let (points, cams_from_world, cameras) = track_observations(&reconstruction, point3d);
        if points.len() < 2 {
            continue;
        }
        num_tested += 1;
        let result = estimate_triangulation(&options, &points, &cams_from_world, &cameras)
            .unwrap()
            .unwrap_or_else(|| panic!("point {point3d_id}: success"));
        assert_eq!(result.inlier_mask.len(), points.len(), "point {point3d_id}");
        assert!(
            result.inlier_mask.iter().all(|&x| x),
            "point {point3d_id}: all inliers"
        );
        assert!(
            (result.xyz - point3d.xyz).norm() < 1e-6,
            "point {point3d_id}: xyz"
        );
    }
    assert!(num_tested > 0);
}

#[test]
fn rust_only_estimate_triangulation_nominal_angular_error() {
    nominal(ResidualType::AngularError);
}

#[test]
fn rust_only_estimate_triangulation_nominal_reprojection_error() {
    nominal(ResidualType::ReprojectionError);
}

#[test]
fn rust_only_estimate_triangulation_rejects_outlier() {
    let reconstruction = synthesize_scene(CameraModelId::SimpleRadial);
    let options = EstimateTriangulationOptions::default();

    let mut num_tested = 0;
    for (point3d_id, point3d) in reconstruction.points3d() {
        let (mut points, cams_from_world, cameras) = track_observations(&reconstruction, point3d);
        if points.len() < 4 {
            continue;
        }
        // One grossly wrong observation among at least three correct ones.
        points[0] += Vector2d::new(150.0, -120.0);
        num_tested += 1;
        let result = estimate_triangulation(&options, &points, &cams_from_world, &cameras)
            .unwrap()
            .unwrap_or_else(|| panic!("point {point3d_id}: success"));
        assert_eq!(result.inlier_mask.len(), points.len(), "point {point3d_id}");
        assert!(!result.inlier_mask[0], "point {point3d_id}: mask");
        assert!(
            result.inlier_mask[1..].iter().all(|&x| x),
            "point {point3d_id}: mask"
        );
        assert!(
            (result.xyz - point3d.xyz).norm() < 1e-6,
            "point {point3d_id}: xyz"
        );
    }
    assert!(num_tested > 0);
}

#[test]
fn rust_only_estimate_triangulation_equirectangular() {
    let reconstruction = synthesize_scene(CameraModelId::Equirectangular);
    let options = EstimateTriangulationOptions::default();

    let mut num_tested = 0;
    for (point3d_id, point3d) in reconstruction.points3d() {
        let (points, cams_from_world, cameras) = track_observations(&reconstruction, point3d);
        if points.len() < 2 {
            continue;
        }
        num_tested += 1;
        let result = estimate_triangulation(&options, &points, &cams_from_world, &cameras)
            .unwrap()
            .unwrap_or_else(|| panic!("point {point3d_id}: success"));
        assert!(
            (result.xyz - point3d.xyz).norm() < 1e-6,
            "point {point3d_id}: xyz"
        );
    }
    assert!(num_tested > 0);
}

#[test]
fn rust_only_estimate_triangulation_min_tri_angle() {
    let camera =
        Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 100.0, 200, 200).unwrap();
    let point3d = Vector3d::new(0.0, 0.0, 10.0);
    // Baseline 0.1 at depth 10: a triangulation angle of about 0.57 degrees.
    let cams_from_world = [
        Rigid3d::identity(),
        Rigid3d::new(Quaterniond::identity(), Vector3d::new(-0.1, 0.0, 0.0)),
    ];
    let points: Vec<Vector2d> = cams_from_world
        .iter()
        .map(|t| camera.img_from_cam(*t * point3d, true).unwrap())
        .collect();
    let cameras = [&camera, &camera];

    let mut options = EstimateTriangulationOptions {
        min_tri_angle: deg_to_rad(0.5),
        ..EstimateTriangulationOptions::default()
    };
    let accepted = estimate_triangulation(&options, &points, &cams_from_world, &cameras).unwrap();
    options.min_tri_angle = deg_to_rad(1.0);
    let rejected = estimate_triangulation(&options, &points, &cams_from_world, &cameras).unwrap();

    let accepted = accepted.expect("accepted at 0.5 degrees");
    assert!((accepted.xyz - point3d).norm() < 1e-8);
    assert!(rejected.is_none());
}

#[test]
fn rust_only_triangulation_options_check() {
    let mut options = EstimateTriangulationOptions::default();
    assert!(options.check().is_ok());
    options.min_tri_angle = -1.0;
    assert!(options.check().is_err());
}
