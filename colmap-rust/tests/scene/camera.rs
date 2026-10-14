// Port of colmap/scene/camera_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::scene::camera::Camera.
//
// `EXPECT_THROW(.., std::domain_error)` follows docs/CPP_DIVERGENCES.md entry 101: on the
// metadata methods it is `Err` with `ErrorKind::DomainError` and COLMAP's message; on the
// per-point methods (`CamFromImg`, `CamFromImgThreshold`, `ImgFromCam`) it is a panic with
// that message, caught here with `catch_unwind`. `operator<<` becomes `Display`; C++'s
// default `check_cheirality = true` is passed explicitly.

use std::panic::{catch_unwind, AssertUnwindSafe};

use colmap_rust::linalg::{Matrix2d, Matrix2x3d, Matrix3d, Vector2d, Vector3d};
use colmap_rust::scene::camera::Camera;
use colmap_rust::sensor::models::{CameraModel, CameraModelId, SimplePinholeCameraModel};
use colmap_rust::util::types::{SensorId, SensorType, INVALID_CAMERA_ID};
use colmap_rust::{ColmapError, ErrorKind};

const DOES_NOT_EXIST: &str = "Camera model does not exist";

fn assert_domain_error<T: std::fmt::Debug>(result: Result<T, ColmapError>) {
    let error = result.expect_err("expected std::domain_error");
    assert_eq!(error.kind(), ErrorKind::DomainError);
    assert_eq!(error.message(), DOES_NOT_EXIST);
}

fn assert_domain_panic<T>(f: impl FnOnce() -> T) {
    let payload = match catch_unwind(AssertUnwindSafe(f)) {
        Ok(_) => panic!("expected a panic with \"{DOES_NOT_EXIST}\""),
        Err(payload) => payload,
    };
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .unwrap_or("");
    assert!(message.contains(DOES_NOT_EXIST), "panic message: {message}");
}

fn create(model_id: CameraModelId, focal_length: f64, width: usize, height: usize) -> Camera {
    Camera::create_from_model_id(1, model_id, focal_length, width, height).unwrap()
}

#[test]
fn camera_empty() {
    let camera = Camera::default();
    assert_eq!(camera.camera_id, INVALID_CAMERA_ID);
    assert_eq!(
        camera.sensor_id(),
        SensorId::new(SensorType::Camera, INVALID_CAMERA_ID)
    );
    assert_eq!(camera.model_id, CameraModelId::Invalid);
    assert_eq!(camera.model_name(), "");
    assert_eq!(camera.width, 0);
    assert_eq!(camera.height, 0);
    assert!(!camera.has_prior_focal_length);
    assert_domain_error(camera.focal_length_idxs());
    assert_domain_error(camera.principal_point_idxs());
    assert_domain_error(camera.extra_params_idxs());
    assert_domain_error(camera.meta_data_params_idxs());
    assert_domain_error(camera.params_info());
    assert_eq!(camera.params_to_string(), "");
    assert_eq!(camera.params.len(), 0);
    assert_eq!(camera.params.len(), 0);
    assert_eq!(camera.params.as_ptr(), camera.params.as_ptr());
}

#[test]
fn camera_equals() {
    let mut camera = create(SimplePinholeCameraModel::MODEL_ID, 1.0, 1, 1);
    let mut other = camera.clone();
    assert_eq!(camera, other);
    camera.set_focal_length(2.).unwrap();
    assert_ne!(camera, other);
    other.set_focal_length(2.).unwrap();
    assert_eq!(camera, other);
}

#[test]
fn camera_print() {
    let camera = create(SimplePinholeCameraModel::MODEL_ID, 1.0, 1, 1);
    assert_eq!(
        camera.to_string(),
        "Camera(camera_id=1, model=SIMPLE_PINHOLE, width=1, height=1, \
         params=[1, 0.5, 0.5] (f, cx, cy))"
    );
}

#[test]
fn camera_camera_id() {
    let mut camera = Camera::default();
    assert_eq!(camera.camera_id, INVALID_CAMERA_ID);
    camera.camera_id = 1;
    assert_eq!(camera.camera_id, 1);
}

#[test]
fn camera_sensor_id() {
    let camera = Camera {
        camera_id: 1,
        ..Camera::default()
    };
    assert_eq!(camera.sensor_id(), SensorId::new(SensorType::Camera, 1));
}

#[test]
fn camera_focal_length() {
    let mut camera = create(SimplePinholeCameraModel::MODEL_ID, 1.0, 1, 1);
    assert_eq!(camera.focal_length().unwrap(), 1.0);
    assert_eq!(camera.focal_length_x().unwrap(), 1.0);
    assert_eq!(camera.focal_length_y().unwrap(), 1.0);
    camera.set_focal_length(2.0).unwrap();
    assert_eq!(camera.focal_length().unwrap(), 2.0);
    assert_eq!(camera.focal_length_x().unwrap(), 2.0);
    assert_eq!(camera.focal_length_y().unwrap(), 2.0);
    camera = create(CameraModelId::Pinhole, 1.0, 1, 1);
    assert_eq!(camera.focal_length_x().unwrap(), 1.0);
    assert_eq!(camera.focal_length_y().unwrap(), 1.0);
    camera.set_focal_length_x(2.0).unwrap();
    assert_eq!(camera.focal_length_x().unwrap(), 2.0);
    assert_eq!(camera.focal_length_y().unwrap(), 1.0);
    assert_eq!(camera.mean_focal_length().unwrap(), 1.5);
    camera.set_focal_length_y(2.0).unwrap();
    assert_eq!(camera.focal_length_x().unwrap(), 2.0);
    assert_eq!(camera.focal_length_y().unwrap(), 2.0);
    assert_eq!(camera.mean_focal_length().unwrap(), 2.0);
}

#[test]
fn camera_principal_point() {
    let mut camera = create(CameraModelId::Pinhole, 1.0, 1, 1);
    assert_eq!(camera.principal_point_x().unwrap(), 0.5);
    assert_eq!(camera.principal_point_y().unwrap(), 0.5);
    assert_eq!(camera.principal_point().unwrap(), Vector2d::new(0.5, 0.5));
    camera.set_principal_point_x(2.0).unwrap();
    assert_eq!(camera.principal_point_x().unwrap(), 2.0);
    assert_eq!(camera.principal_point_y().unwrap(), 0.5);
    assert_eq!(camera.principal_point().unwrap(), Vector2d::new(2.0, 0.5));
    camera.set_principal_point_y(2.0).unwrap();
    assert_eq!(camera.principal_point_x().unwrap(), 2.0);
    assert_eq!(camera.principal_point_y().unwrap(), 2.0);
    assert_eq!(camera.principal_point().unwrap(), Vector2d::new(2.0, 2.0));
}

#[test]
fn camera_param_idxs() {
    let mut camera = Camera::default();
    assert_domain_error(camera.focal_length_idxs());
    assert_domain_error(camera.principal_point_idxs());
    assert_domain_error(camera.extra_params_idxs());
    assert_domain_error(camera.meta_data_params_idxs());
    camera.model_id = CameraModelId::FullOpenCV;
    assert_eq!(camera.focal_length_idxs().unwrap().len(), 2);
    assert_eq!(camera.principal_point_idxs().unwrap().len(), 2);
    assert_eq!(camera.extra_params_idxs().unwrap().len(), 8);
    assert_eq!(camera.meta_data_params_idxs().unwrap().len(), 0);
}

#[test]
fn camera_calibration_matrix() {
    let camera = create(CameraModelId::Pinhole, 1.0, 1, 1);
    let k = camera.calibration_matrix().unwrap();
    let k_ref = Matrix3d::from_rows(
        Vector3d::new(1.0, 0.0, 0.5),
        Vector3d::new(0.0, 1.0, 0.5),
        Vector3d::new(0.0, 0.0, 1.0),
    );
    assert_eq!(k, k_ref);
}

#[test]
fn camera_params_info() {
    let mut camera = Camera::default();
    assert_domain_error(camera.params_info());
    camera.model_id = CameraModelId::SimpleRadial;
    assert_eq!(camera.params_info().unwrap(), "f, cx, cy, k");
}

#[test]
fn camera_params_to_string() {
    let camera = create(SimplePinholeCameraModel::MODEL_ID, 1.0, 1, 1);
    assert_eq!(camera.params_to_string(), "1, 0.5, 0.5");
}

#[test]
fn camera_params_from_string() {
    let mut camera = Camera {
        model_id: SimplePinholeCameraModel::MODEL_ID,
        ..Camera::default()
    };
    assert!(camera.set_params_from_string("1, 0.5, 0.5").unwrap());
    let params = vec![1.0, 0.5, 0.5];
    assert_eq!(camera.params, params);
    assert!(!camera.set_params_from_string("1, 0.5").unwrap());
    assert_eq!(camera.params, params);
}

#[test]
fn camera_verify_params() {
    let mut camera = Camera::default();
    assert_domain_error(camera.verify_params());
    camera = create(SimplePinholeCameraModel::MODEL_ID, 1.0, 1, 1);
    assert!(camera.verify_params().unwrap());
    camera.params.resize(2, 0.0);
    assert!(!camera.verify_params().unwrap());
}

#[test]
fn camera_is_undistorted() {
    let mut camera = create(SimplePinholeCameraModel::MODEL_ID, 1.0, 1, 1);
    assert!(camera.is_undistorted().unwrap());
    camera = create(CameraModelId::SimpleRadial, 1.0, 1, 1);
    assert!(camera.is_undistorted().unwrap());
    camera.params = vec![1.0, 0.5, 0.5, 0.005];
    assert!(!camera.is_undistorted().unwrap());
    camera = create(CameraModelId::Radial, 1.0, 1, 1);
    assert!(camera.is_undistorted().unwrap());
    camera.params = vec![1.0, 0.5, 0.5, 0.0, 0.005];
    assert!(!camera.is_undistorted().unwrap());
    camera = create(CameraModelId::OpenCV, 1.0, 1, 1);
    assert!(camera.is_undistorted().unwrap());
    camera.params = vec![1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.001];
    assert!(!camera.is_undistorted().unwrap());
    camera = create(CameraModelId::FullOpenCV, 1.0, 1, 1);
    assert!(camera.is_undistorted().unwrap());
    camera.params = vec![1.0, 1.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001];
    assert!(!camera.is_undistorted().unwrap());
}

#[test]
fn camera_has_bogus_params() {
    let mut camera = Camera::default();
    assert_domain_error(camera.has_bogus_params(0.0, 0.0, 0.0));
    camera = create(SimplePinholeCameraModel::MODEL_ID, 1.0, 1, 1);
    assert!(!camera.has_bogus_params(0.1, 1.1, 1.0).unwrap());
    assert!(!camera.has_bogus_params(0.1, 1.1, 0.0).unwrap());
    assert!(camera.has_bogus_params(0.1, 0.99, 1.0).unwrap());
    assert!(camera.has_bogus_params(1.01, 1.1, 1.0).unwrap());
    camera = create(CameraModelId::SimpleRadial, 1.0, 1, 1);
    assert!(!camera.has_bogus_params(0.1, 1.1, 1.0).unwrap());
    camera.params[3] = 1.01;
    assert!(camera.has_bogus_params(0.1, 1.1, 1.0).unwrap());
    camera.params[3] = -0.5;
    assert!(!camera.has_bogus_params(0.1, 1.1, 1.0).unwrap());
    camera.params[3] = -1.01;
    assert!(camera.has_bogus_params(0.1, 1.1, 1.0).unwrap());
}

/// The checks `CreateFromModelId` and `CreateFromModelName` share in camera_test.cc.
fn check_simple_pinhole(camera: &Camera) {
    assert_eq!(camera.camera_id, 1);
    assert_eq!(camera.model_id, SimplePinholeCameraModel::MODEL_ID);
    assert_eq!(camera.model_name(), "SIMPLE_PINHOLE");
    assert_eq!(camera.width, 1);
    assert_eq!(camera.height, 1);
    assert!(!camera.has_prior_focal_length);
    assert_eq!(camera.focal_length_idxs().unwrap().len(), 1);
    assert_eq!(camera.principal_point_idxs().unwrap().len(), 2);
    assert_eq!(camera.extra_params_idxs().unwrap().len(), 0);
    assert_eq!(camera.meta_data_params_idxs().unwrap().len(), 0);
    assert_eq!(camera.params_info().unwrap(), "f, cx, cy");
    assert_eq!(camera.params_to_string(), "1, 0.5, 0.5");
    assert_eq!(camera.focal_length().unwrap(), 1.0);
    assert_eq!(camera.principal_point_x().unwrap(), 0.5);
    assert_eq!(camera.principal_point_y().unwrap(), 0.5);
    assert!(camera.verify_params().unwrap());
    assert!(!camera.has_bogus_params(0.1, 2.0, 1.0).unwrap());
    assert!(camera.has_bogus_params(0.1, 0.5, 1.0).unwrap());
    assert_eq!(camera.params.len(), SimplePinholeCameraModel::NUM_PARAMS);
    assert_eq!(camera.params.len(), SimplePinholeCameraModel::NUM_PARAMS);
}

#[test]
fn camera_create_from_model_id() {
    let camera =
        Camera::create_from_model_id(1, SimplePinholeCameraModel::MODEL_ID, 1.0, 1, 1).unwrap();
    check_simple_pinhole(&camera);
}

#[test]
fn camera_create_from_model_name() {
    let camera = Camera::create_from_model_name(1, "SIMPLE_PINHOLE", 1.0, 1, 1).unwrap();
    check_simple_pinhole(&camera);
}

#[test]
fn camera_cam_from_img() {
    let mut camera = Camera::default();
    assert_domain_panic(|| camera.cam_from_img(Vector2d::zeros()));
    camera = create(CameraModelId::SimplePinhole, 1.0, 1, 1);
    assert_eq!(
        camera.cam_from_img(Vector2d::new(0.0, 0.0)).unwrap(),
        Vector2d::new(-0.5, -0.5)
    );
    assert_eq!(
        camera.cam_from_img(Vector2d::new(0.5, 0.5)).unwrap(),
        Vector2d::new(0.0, 0.0)
    );
}

#[test]
fn camera_cam_from_img_threshold() {
    let mut camera = Camera::default();
    assert_domain_panic(|| camera.cam_from_img_threshold(0.0));
    camera = create(CameraModelId::SimplePinhole, 1.0, 1, 1);
    assert_eq!(camera.cam_from_img_threshold(0.0), 0.0);
    assert_eq!(camera.cam_from_img_threshold(1.0), 1.0);
    camera.set_focal_length(2.0).unwrap();
    assert_eq!(camera.cam_from_img_threshold(1.0), 0.5);
    camera = create(CameraModelId::Pinhole, 1.0, 1, 1);
    camera.set_focal_length_y(3.0).unwrap();
    assert_eq!(camera.cam_from_img_threshold(1.0), 0.5);
}

#[test]
fn camera_img_from_cam() {
    let mut camera = Camera::default();
    assert_domain_panic(|| camera.img_from_cam(Vector3d::zeros(), true));
    camera = create(CameraModelId::SimplePinhole, 1.0, 1, 1);
    assert_eq!(
        camera
            .img_from_cam(Vector3d::new(0.0, 0.0, 1.0), true)
            .unwrap(),
        Vector2d::new(0.5, 0.5)
    );
    assert_eq!(
        camera
            .img_from_cam(Vector3d::new(-0.5, -0.5, 1.0), true)
            .unwrap(),
        Vector2d::new(0.0, 0.0)
    );
}

#[test]
fn camera_img_from_cam_without_cheirality_check() {
    let camera = create(CameraModelId::SimplePinhole, 1.0, 1, 1);
    let cam_point = Vector3d::new(-0.5, -0.5, -1.0);
    assert!(camera.img_from_cam(cam_point, true).is_none());
    assert_eq!(
        camera
            .img_from_cam(cam_point, /*check_cheirality=*/ false)
            .unwrap(),
        Vector2d::new(1.0, 1.0)
    );
}

#[test]
fn camera_cam_ray_from_img_with_jac() {
    // Covers a perspective model, a distorted one, and a spherical one. For the spherical
    // camera the sampled pixels include the back hemisphere, where CamFromImg fails
    // outright but CamRayFromImg (and hence this) must not.
    let cameras = [
        Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 650.0, 1024, 768).unwrap(),
        Camera::create_from_model_id(2, CameraModelId::SimpleRadial, 650.0, 1024, 768).unwrap(),
        Camera::create_from_model_id(3, CameraModelId::Equirectangular, 0.0, 1000, 500).unwrap(),
    ];

    for camera in &cameras {
        for x in [1.0, 250.0, 512.0, 800.0, 999.0] {
            for y in [1.0, 120.0, 250.0, 400.0, 499.0] {
                let image_point = Vector2d::new(x, y);
                let ray_and_jac = camera.cam_ray_from_img_with_jac(image_point);
                let ray_and_jac = ray_and_jac
                    .unwrap_or_else(|| panic!("model {} at {x} {y}", camera.model_name()));
                let (cam_ray, j_ray) = (ray_and_jac.ray, ray_and_jac.jacobian);

                // The bearing is a unit vector that reprojects to the source pixel.
                assert!((cam_ray.norm() - 1.0).abs() <= 1e-12);
                let reprojected = camera.img_from_cam(cam_ray, true);
                let reprojected = reprojected.expect("reprojected");
                assert!((reprojected - image_point).norm() <= 1e-8);

                // The Jacobian inverts the projection on the tangent plane.
                let mut j_uvw = Matrix2x3d::zeros();
                assert!(camera
                    .img_from_cam_with_jac(cam_ray, Some(&mut j_uvw), true)
                    .is_some());
                assert!((j_uvw * j_ray - Matrix2d::identity()).norm() <= 1e-10);
                assert!((j_ray.transpose() * cam_ray).norm() <= 1e-10 * j_ray.norm());
            }
        }
    }
}

#[test]
fn camera_cam_ray_from_img_with_jac_unprojectable() {
    // A pixel the camera cannot unproject (iterative undistortion diverges) must yield no
    // ray + Jacobian, rather than a garbage one.
    let mut camera = create(CameraModelId::OpenCV, 100.0, 100, 200);
    camera.params[4] = -0.5; // k1
    camera.params[5] = 0.5; // k2
    camera.params[6] = -0.5; // p1
    let unprojectable = Vector2d::new(50.0, 150.0);
    assert!(camera.cam_from_img(unprojectable).is_none());
    assert!(camera.cam_ray_from_img_with_jac(unprojectable).is_none());
}

#[test]
fn camera_rescale() {
    let mut camera = create(CameraModelId::SimplePinhole, 1.0, 1, 1);
    camera.rescale(2.0).unwrap();
    assert_eq!(camera.width, 2);
    assert_eq!(camera.height, 2);
    assert_eq!(camera.focal_length().unwrap(), 2.0);
    assert_eq!(camera.principal_point_x().unwrap(), 1.0);
    assert_eq!(camera.principal_point_y().unwrap(), 1.0);

    camera = create(CameraModelId::Pinhole, 1.0, 1, 1);
    camera.rescale(2.0).unwrap();
    assert_eq!(camera.width, 2);
    assert_eq!(camera.height, 2);
    assert_eq!(camera.focal_length_x().unwrap(), 2.0);
    assert_eq!(camera.focal_length_y().unwrap(), 2.0);
    assert_eq!(camera.principal_point_x().unwrap(), 1.0);
    assert_eq!(camera.principal_point_y().unwrap(), 1.0);

    camera = create(CameraModelId::Pinhole, 1.0, 2, 2);
    camera.rescale(0.5).unwrap();
    assert_eq!(camera.width, 1);
    assert_eq!(camera.height, 1);
    assert_eq!(camera.focal_length_x().unwrap(), 0.5);
    assert_eq!(camera.focal_length_y().unwrap(), 0.5);
    assert_eq!(camera.principal_point_x().unwrap(), 0.5);
    assert_eq!(camera.principal_point_y().unwrap(), 0.5);

    camera = create(CameraModelId::Pinhole, 1.0, 2, 2);
    camera.rescale_to(1, 1);
    assert_eq!(camera.width, 1);
    assert_eq!(camera.height, 1);
    assert_eq!(camera.focal_length_x().unwrap(), 0.5);
    assert_eq!(camera.focal_length_y().unwrap(), 0.5);
    assert_eq!(camera.principal_point_x().unwrap(), 0.5);
    assert_eq!(camera.principal_point_y().unwrap(), 0.5);

    camera = create(CameraModelId::Pinhole, 1.0, 2, 2);
    camera.rescale_to(4, 4);
    assert_eq!(camera.width, 4);
    assert_eq!(camera.height, 4);
    assert_eq!(camera.focal_length_x().unwrap(), 2.0);
    assert_eq!(camera.focal_length_y().unwrap(), 2.0);
    assert_eq!(camera.principal_point_x().unwrap(), 2.0);
    assert_eq!(camera.principal_point_y().unwrap(), 2.0);
}

#[test]
fn camera_spherical() {
    let mut camera = create(
        CameraModelId::Equirectangular,
        /*focal_length=*/ 0.0,
        1000,
        500,
    );
    assert_eq!(camera.params, vec![1000.0, 500.0]);
    assert!(!camera.is_perspective());
    assert!(camera.is_spherical());

    // No focal length / pinhole image plane. The (w, h) parameters form the metadata group;
    // there is no principal point or extra (distortion).
    assert!(camera.focal_length_idxs().unwrap().is_empty());
    assert!(camera.principal_point_idxs().unwrap().is_empty());
    assert!(camera.extra_params_idxs().unwrap().is_empty());
    assert_eq!(camera.meta_data_params_idxs().unwrap().len(), 2);
    assert_eq!(camera.mean_focal_length().unwrap(), 0.0);
    // CalibrationMatrix is undefined without a focal length.
    assert!(camera.calibration_matrix().is_err());
    // Spherical images have no lens distortion to undistort.
    assert!(camera.is_undistorted().unwrap());

    // Rescaling keeps the (w, h) parameters consistent with the dimensions.
    camera.rescale(0.5).unwrap();
    assert_eq!(camera.width, 500);
    assert_eq!(camera.height, 250);
    assert_eq!(camera.params, vec![500.0, 250.0]);
    assert!(camera.is_spherical());

    // A perspective camera is not spherical.
    let pinhole = Camera::create_from_model_id(2, CameraModelId::Pinhole, 1.0, 1, 1).unwrap();
    assert!(pinhole.is_perspective());
    assert!(!pinhole.is_spherical());
}

#[test]
fn camera_is_perspective_pinhole() {
    let pinhole = Camera::create_from_model_id(1, CameraModelId::Pinhole, 1.0, 1, 1).unwrap();
    assert!(pinhole.is_perspective_pinhole());

    let radial = Camera::create_from_model_id(2, CameraModelId::SimpleRadial, 1.0, 1, 1).unwrap();
    assert!(radial.is_perspective_pinhole());

    let fisheye = Camera::create_from_model_id(3, CameraModelId::OpenCVFisheye, 1.0, 1, 1).unwrap();
    assert!(fisheye.is_perspective());
    assert!(fisheye.is_perspective_fisheye());
    assert!(!fisheye.is_perspective_pinhole());

    let spherical =
        Camera::create_from_model_id(4, CameraModelId::Equirectangular, 0.0, 1000, 500).unwrap();
    assert!(!spherical.is_perspective_fisheye());
    assert!(!spherical.is_perspective_pinhole());
}
