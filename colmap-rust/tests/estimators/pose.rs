// Port of COLMAP's src/colmap/estimators/pose_test.cc (same values and tolerances), testing
// colmap_rust::estimators::pose: the EstimateAbsolutePose and EstimateRelativePose cases.
// The RefineAbsolutePose and RefineEssentialMatrix cases wait for their Ceres-based
// functions. Tier C (RANSAC).
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does.

use crate::expect_near;
use crate::transform_matchers::rigid3d_near;
use colmap_rust::estimators::pose::{
    estimate_absolute_pose, estimate_relative_pose, AbsolutePoseEstimationOptions,
};
use colmap_rust::geometry::pose::CamRayWithJac;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Quaterniond, Vector2d, Vector3d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::random_eigen_vector3d;
use colmap_rust::optim::RansacOptions;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::sensor::models::CameraModelId;

struct AbsolutePoseProblem {
    cam_from_world: Rigid3d,
    camera: Camera,
    points2d: Vec<Vector2d>,
    points3d: Vec<Vector3d>,
}

fn create_absolute_pose_test_data(
    camera_model_id: CameraModelId,
    camera_params: Vec<f64>,
) -> AbsolutePoseProblem {
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 2,
        num_points3d: 100,
        camera_model_id,
        camera_params,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut reconstruction).unwrap();

    let image = reconstruction.image(1).unwrap();
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    let camera = reconstruction.camera(image.camera_id()).unwrap().clone();
    assert_eq!(camera.model_id, camera_model_id);
    let mut points2d = Vec::new();
    let mut points3d = Vec::new();
    for point2d in image.points2d() {
        if point2d.has_point3d() {
            points2d.push(point2d.xy);
            points3d.push(reconstruction.point3d(point2d.point3d_id).unwrap().xyz);
        }
    }
    AbsolutePoseProblem {
        cam_from_world: image.cam_from_world(frame, rig).unwrap(),
        camera,
        points2d,
        points3d,
    }
}

fn default_absolute_pose_test_data() -> AbsolutePoseProblem {
    create_absolute_pose_test_data(
        CameraModelId::SimpleRadial,
        vec![1280.0, 512.0, 384.0, 0.05],
    )
}

#[test]
fn estimate_absolute_pose_nominal() {
    set_prng_seed(0);
    let problem = default_absolute_pose_test_data();

    let options = AbsolutePoseEstimationOptions::default();
    let mut camera = problem.camera.clone();
    let estimate =
        estimate_absolute_pose(&options, &problem.points2d, &problem.points3d, &mut camera)
            .unwrap()
            .expect("success");
    assert!(rigid3d_near(
        &estimate.pose,
        &problem.cam_from_world,
        1e-6,
        1e-6
    ));
    assert_eq!(camera, problem.camera);
    assert_eq!(estimate.num_inliers, problem.points2d.len());
    assert!(estimate.inlier_mask.iter().all(|&x| x));
}

#[test]
fn estimate_absolute_pose_estimate_focal_length() {
    set_prng_seed(0);
    let problem = default_absolute_pose_test_data();

    let options = AbsolutePoseEstimationOptions {
        estimate_focal_length: true,
        ..AbsolutePoseEstimationOptions::default()
    };
    let mut camera = problem.camera.clone();
    let estimate =
        estimate_absolute_pose(&options, &problem.points2d, &problem.points3d, &mut camera)
            .unwrap()
            .expect("success");
    assert!(rigid3d_near(
        &estimate.pose,
        &problem.cam_from_world,
        1e-3,
        1e-2
    ));
    expect_near(
        camera.focal_length().unwrap(),
        problem.camera.focal_length().unwrap(),
        5.0,
    );
    camera
        .set_focal_length(problem.camera.focal_length().unwrap())
        .unwrap();
    assert_eq!(camera, problem.camera);
    assert_eq!(estimate.num_inliers, problem.points2d.len());
    assert!(estimate.inlier_mask.iter().all(|&x| x));
}

#[test]
fn estimate_absolute_pose_estimate_separate_focal_lengths() {
    set_prng_seed(0);
    // PINHOLE camera with distinct fx and fy (fx, fy, cx, cy).
    let problem =
        create_absolute_pose_test_data(CameraModelId::Pinhole, vec![1280.0, 1000.0, 512.0, 384.0]);

    let options = AbsolutePoseEstimationOptions {
        estimate_focal_length: true,
        ..AbsolutePoseEstimationOptions::default()
    };
    let mut camera = problem.camera.clone();
    let estimate =
        estimate_absolute_pose(&options, &problem.points2d, &problem.points3d, &mut camera)
            .unwrap()
            .expect("success");
    assert!(rigid3d_near(
        &estimate.pose,
        &problem.cam_from_world,
        1e-3,
        1e-2
    ));
    expect_near(
        camera.focal_length_x().unwrap(),
        problem.camera.focal_length_x().unwrap(),
        5.0,
    );
    expect_near(
        camera.focal_length_y().unwrap(),
        problem.camera.focal_length_y().unwrap(),
        5.0,
    );
    assert_eq!(estimate.num_inliers, problem.points2d.len());
    assert!(estimate.inlier_mask.iter().all(|&x| x));
}

// Rays with their unprojection Jacobians of 100 random points in front of both cameras.
fn relative_pose_rays(
    camera: &Camera,
    cam1_from_world: &Rigid3d,
    cam2_from_world: &Rigid3d,
) -> (Vec<CamRayWithJac>, Vec<CamRayWithJac>) {
    let mut cam_rays1_with_jac = Vec::with_capacity(100);
    let mut cam_rays2_with_jac = Vec::with_capacity(100);
    for _ in 0..100 {
        let point3d = random_eigen_vector3d() + Vector3d::new(0.0, 0.0, 3.0);
        let point2d1 = camera
            .img_from_cam(*cam1_from_world * point3d, true)
            .unwrap();
        let point2d2 = camera
            .img_from_cam(*cam2_from_world * point3d, true)
            .unwrap();
        cam_rays1_with_jac.push(camera.cam_ray_from_img_with_jac(point2d1).unwrap());
        cam_rays2_with_jac.push(camera.cam_ray_from_img_with_jac(point2d2).unwrap());
    }
    (cam_rays1_with_jac, cam_rays2_with_jac)
}

fn relative_pose_setup() -> (Camera, Rigid3d, Rigid3d) {
    let camera =
        Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 512.0, 1024, 1024).unwrap();
    let cam1_from_world = Rigid3d::identity();
    let cam2_from_world = Rigid3d::new(
        Quaterniond::identity(),
        Vector3d::new(1.0, 0.1, 0.2).normalized(),
    );
    (camera, cam1_from_world, cam2_from_world)
}

#[test]
fn estimate_relative_pose_nominal() {
    set_prng_seed(0);
    let (camera, cam1_from_world, cam2_from_world) = relative_pose_setup();

    // EstimateRelativePose scores with the pixel-unit tangent Sampson error, so it takes
    // rays with their unprojection Jacobians (CamRayWithJac). Points are placed in front of
    // both cameras and projected to pixels.
    let (cam_rays1_with_jac, cam_rays2_with_jac) =
        relative_pose_rays(&camera, &cam1_from_world, &cam2_from_world);

    let options = RansacOptions {
        max_error: 1.0, // pixels
        ..RansacOptions::default()
    };
    let estimate = estimate_relative_pose(&options, &cam_rays1_with_jac, &cam_rays2_with_jac)
        .unwrap()
        .expect("success");

    assert!(rigid3d_near(
        &estimate.pose,
        &(cam2_from_world * cam1_from_world.inverse()),
        1e-3,
        1e-3
    ));
    assert_eq!(estimate.num_inliers, cam_rays1_with_jac.len());
    assert!(estimate.inlier_mask.iter().all(|&x| x));
}

#[test]
fn estimate_relative_pose_zero_sentinel_rays_excluded() {
    set_prng_seed(0);
    let (camera, cam1_from_world, cam2_from_world) = relative_pose_setup();
    let (mut cam_rays1_with_jac, mut cam_rays2_with_jac) =
        relative_pose_rays(&camera, &cam1_from_world, &cam2_from_world);

    // Emulate unprojectable correspondences: CamRayFromImgWithJac returns nullopt and
    // callers substitute the CamRayWithJac::Zero() sentinel. These must be rejected
    // (infinite tangent Sampson residual) and never counted as inliers, while the pose is
    // still recovered from the remaining correspondences.
    let mut is_zeroed = vec![false; cam_rays1_with_jac.len()];
    for i in [3, 17, 42, 88] {
        cam_rays1_with_jac[i] = CamRayWithJac::zero();
        cam_rays2_with_jac[i] = CamRayWithJac::zero();
        is_zeroed[i] = true;
    }

    let options = RansacOptions {
        max_error: 1.0, // pixels
        ..RansacOptions::default()
    };
    let estimate = estimate_relative_pose(&options, &cam_rays1_with_jac, &cam_rays2_with_jac)
        .unwrap()
        .expect("success");

    assert!(rigid3d_near(
        &estimate.pose,
        &(cam2_from_world * cam1_from_world.inverse()),
        1e-3,
        1e-3
    ));
    assert_eq!(estimate.num_inliers, cam_rays1_with_jac.len() - 4);
    assert_eq!(estimate.inlier_mask.len(), is_zeroed.len());
    for (i, (&is_inlier, &zeroed)) in estimate.inlier_mask.iter().zip(&is_zeroed).enumerate() {
        assert_eq!(is_inlier, !zeroed, "correspondence {i}");
    }
}
