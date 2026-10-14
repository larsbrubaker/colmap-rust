// Port of colmap/scene/projection_test.cc, 1:1 (same test names, values and tolerances),
// for colmap_rust::scene::projection. COLMAP's `Rigid3d` and `Matrix3x4d` overloads are the
// same generic functions here, called once with each pose type; `EIGEN_PI` is
// `std::f64::consts::PI`, and the test's own `std::sin`/`std::cos` go through
// `colmap_rust::math::fns`.

use std::f64::consts::PI;

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Matrix3x4d, Quaterniond, Vector2d, Vector3d};
use colmap_rust::math::fns;
use colmap_rust::math::random_eigen::random_eigen_vector3d;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::projection::{
    calculate_angular_reprojection_error, calculate_squared_reprojection_error,
    has_point_positive_depth,
};
use colmap_rust::sensor::models::CameraModelId;

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} vs. {expected} (tolerance {tolerance})"
    );
}

#[test]
fn calculate_squared_reprojection_error_nominal() {
    let cam_from_world = Rigid3d::new(Quaterniond::identity(), Vector3d::zeros());
    let cam_from_world_mat: Matrix3x4d = cam_from_world.to_matrix();

    let point3d = random_eigen_vector3d().cwise_abs();
    let point2d_h = cam_from_world_mat * point3d.homogeneous();
    let point2d = point2d_h.hnormalized();

    let camera = Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1.0, 0, 0).unwrap();

    assert_near(
        calculate_squared_reprojection_error(point2d, point3d, &cam_from_world, &camera),
        0.0,
        1e-6,
    );
    assert_near(
        calculate_squared_reprojection_error(point2d, point3d, &cam_from_world_mat, &camera),
        0.0,
        1e-6,
    );

    let shifted = point2d + Vector2d::new(1.0, 1.0);
    assert_near(
        calculate_squared_reprojection_error(shifted, point3d, &cam_from_world, &camera),
        2.0,
        1e-6,
    );
    assert_near(
        calculate_squared_reprojection_error(shifted, point3d, &cam_from_world_mat, &camera),
        2.0,
        1e-6,
    );
}

#[test]
fn calculate_squared_reprojection_error_spherical() {
    let camera = Camera::create_from_model_id(
        1,
        CameraModelId::Equirectangular,
        /*focal_length=*/ 0.0,
        1000,
        500,
    )
    .unwrap();
    let cam_from_world = Rigid3d::default();
    let px_per_rad = camera.width as f64 / (2.0 * PI);

    // Exact observations have ~0 error, including a back-hemisphere match that straddles
    // the azimuth seam (observed at x = 0, point projects to x = w), which a raw pixel error
    // would penalize by ~width^2.
    for cam_point in [
        Vector3d::new(0.0, 0.0, 1.0),
        Vector3d::new(0.0, 0.0, -1.0),
        Vector3d::new(1.0, 0.0, 0.0),
    ] {
        let img_point = camera.img_from_cam(cam_point, true).unwrap();
        assert_near(
            calculate_squared_reprojection_error(img_point, cam_point, &cam_from_world, &camera),
            0.0,
            1e-9,
        );
    }

    // Check that the error is continuous across the seam.
    assert_near(
        calculate_squared_reprojection_error(
            Vector2d::new(0.0, camera.height as f64 / 2.0),
            Vector3d::new(0.0, 0.0, -1.0),
            &cam_from_world,
            &camera,
        ),
        0.0,
        1e-9,
    );

    // Pole invariance: the same angular offset yields the same squared error at the equator
    // and near the pole. Raw equirectangular pixel error would diverge towards the pole,
    // dropping otherwise-valid observations.
    const OFFSET: f64 = 0.01; // radians
    let equator_point = Vector3d::new(0.0, 0.0, 1.0);
    let equator_obs = camera
        .img_from_cam(Vector3d::new(fns::sin(OFFSET), 0.0, fns::cos(OFFSET)), true)
        .unwrap();
    let pole_point = Vector3d::new(0.0, -1.0, 0.0);
    let pole_obs = camera
        .img_from_cam(
            Vector3d::new(0.0, -fns::cos(OFFSET), fns::sin(OFFSET)),
            true,
        )
        .unwrap();
    let expected = (OFFSET * px_per_rad) * (OFFSET * px_per_rad);
    assert_near(
        calculate_squared_reprojection_error(equator_obs, equator_point, &cam_from_world, &camera),
        expected,
        1e-6,
    );
    assert_near(
        calculate_squared_reprojection_error(pole_obs, pole_point, &cam_from_world, &camera),
        expected,
        1e-6,
    );
}

#[test]
fn calculate_angular_reprojection_error_nominal() {
    let cam_from_world = Rigid3d::new(Quaterniond::identity(), Vector3d::zeros());
    let cam_from_world_mat = cam_from_world.to_matrix();

    let camera = Camera {
        model_id: CameraModelId::SimplePinhole,
        params: vec![1.0, 0.0, 0.0],
        ..Camera::default()
    };

    let error = |x: f64, y: f64, point3d: Vector3d| {
        calculate_angular_reprojection_error(
            Vector2d::new(x, y),
            point3d,
            &cam_from_world_mat,
            &camera,
        )
    };

    let error1 = error(0.0, 0.0, Vector3d::new(0.0, 0.0, 1.0));
    assert_near(error1, 0.0, 1e-6);

    let error2 = error(0.0, 0.0, Vector3d::new(0.0, 1.0, 1.0));
    assert_near(error2, PI / 4.0, 1e-6);

    let error3 = error(0.0, 0.0, Vector3d::new(0.0, 5.0, 5.0));
    assert_near(error3, PI / 4.0, 1e-6);

    let error4 = error(1.0, 0.0, Vector3d::new(0.0, 0.0, 1.0));
    assert_near(error4, PI / 4.0, 1e-6);

    let error5 = error(2.0, 0.0, Vector3d::new(0.0, 0.0, 1.0));
    assert_near(error5, 1.10714872, 1e-6);

    let error6 = error(2.0, 0.0, Vector3d::new(1.0, 0.0, 1.0));
    assert_near(error6, 1.10714872 - PI / 4.0, 1e-6);

    let error7 = error(2.0, 0.0, Vector3d::new(5.0, 0.0, 5.0));
    assert_near(error7, 1.10714872 - PI / 4.0, 1e-6);

    let error8 = error(1.0, 0.0, Vector3d::new(-1.0, 0.0, 1.0));
    assert_near(error8, PI / 2.0, 1e-6);

    let error9 = error(1.0, 0.0, Vector3d::new(-1.0, 0.0, 0.0));
    assert_near(error9, PI * 3.0 / 4.0, 1e-6);

    let error10 = error(1.0, 0.0, Vector3d::new(-1.0, 0.0, -1.0));
    assert_near(error10, PI, 1e-6);

    let error11 = error(1.0, 0.0, Vector3d::new(0.0, 0.0, -1.0));
    assert_near(error11, PI * 3.0 / 4.0, 1e-6);
}

#[test]
fn has_point_positive_depth_nominal() {
    let cam_from_world = Rigid3d::new(Quaterniond::identity(), Vector3d::zeros());
    let cam_from_world_mat = cam_from_world.to_matrix();

    // In the image plane
    let check1 = has_point_positive_depth(&cam_from_world_mat, Vector3d::new(0.0, 0.0, 0.0));
    assert!(!check1);
    let check2 = has_point_positive_depth(&cam_from_world_mat, Vector3d::new(0.0, 2.0, 0.0));
    assert!(!check2);

    // Infront of camera
    let check3 = has_point_positive_depth(&cam_from_world_mat, Vector3d::new(0.0, 0.0, 1.0));
    assert!(check3);

    // Behind camera
    let check4 = has_point_positive_depth(&cam_from_world_mat, Vector3d::new(0.0, 0.0, -1.0));
    assert!(!check4);
}
