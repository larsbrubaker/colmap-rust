// Port of colmap/sensor/models_test.cc, 1:1: one test per gtest TEST(Suite, Name), named
// suite_name in snake case, with the same parameters, grids, expected values and tolerances.
// The templated helpers (TestModel<CameraModel>, TestCamToCamFromImg, ...) stay generic over
// the model type, the Rust form of COLMAP's CRTP models (colmap_rust::sensor::models).
// Tier A: the typed and dispatched calls must agree exactly (EXPECT_EQ); round trips use
// COLMAP's 1e-6 / 1e-12 tolerances.

use colmap_rust::linalg::{Vector2d, Vector3d};
use colmap_rust::sensor::models::*;

fn fisheye_valid<M: PerspectiveFisheyeModel>(params: &[f64], xy: Vector2d) -> bool {
    let mut uu = 0.0;
    let mut vv = 0.0;
    M::fisheye_from_img(params, xy.x, xy.y, &mut uu, &mut vv);
    let theta = (uu * uu + vv * vv).sqrt();
    theta < std::f64::consts::PI / 2.0
}

fn fisheye_camera_model_is_valid_pixel(
    model_id: CameraModelId,
    params: &[f64],
    xy: Vector2d,
) -> bool {
    match model_id {
        CameraModelId::SimpleRadialFisheye => {
            fisheye_valid::<SimpleRadialFisheyeCameraModel>(params, xy)
        }
        CameraModelId::RadialFisheye => fisheye_valid::<RadialFisheyeCameraModel>(params, xy),
        CameraModelId::OpenCVFisheye => fisheye_valid::<OpenCVFisheyeCameraModel>(params, xy),
        CameraModelId::ThinPrismFisheye => fisheye_valid::<ThinPrismFisheyeCameraModel>(params, xy),
        CameraModelId::RadTanThinPrismFisheye => {
            fisheye_valid::<RadTanThinPrismFisheyeModel>(params, xy)
        }
        CameraModelId::SimpleFisheye => fisheye_valid::<SimpleFisheyeCameraModel>(params, xy),
        CameraModelId::Fisheye => fisheye_valid::<FisheyeCameraModel>(params, xy),
        _ => panic!("Camera model does not exist or is not a fisheye camera"),
    }
}

fn assert_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b} (tolerance {tol})");
}

fn test_cam_to_cam_from_img<M: CameraModel>(params: &[f64], u0: f64, v0: f64, w0: f64) {
    let mut u = 0.0;
    let mut v = 0.0;
    let mut x = 0.0;
    let mut y = 0.0;
    M::img_from_cam(params, u0, v0, w0, &mut x, &mut y, true);
    let xy = camera_model_img_from_cam(M::MODEL_ID, params, Vector3d::new(u0, v0, w0), true)
        .expect("ImgFromCam failed");
    assert_eq!(x, xy.x);
    assert_eq!(y, xy.y);
    M::cam_from_img(params, x, y, &mut u, &mut v);
    assert_near(u, u0 / w0, 1e-6);
    assert_near(v, v0 / w0, 1e-6);
}

fn test_cam_from_img_to_img<M: CameraModel>(params: &[f64], x0: f64, y0: f64) {
    let mut u = 0.0;
    let mut v = 0.0;
    let mut x = 0.0;
    let mut y = 0.0;
    M::cam_from_img(params, x0, y0, &mut u, &mut v);
    let uv = camera_model_cam_from_img(M::MODEL_ID, params, Vector2d::new(x0, y0))
        .expect("CamFromImg failed");
    assert_eq!(u, uv.x);
    assert_eq!(v, uv.y);
    for w in [0.5, 1.0, 2.0] {
        assert!(M::img_from_cam(
            params,
            w * u,
            w * v,
            w,
            &mut x,
            &mut y,
            true
        ));
        assert_near(x, x0, 1e-6);
        assert_near(y, y0, 1e-6);
    }
}

// Round-trip a pixel through the 3D bearing interface: CamRayFromImg yields a unit ray,
// ImgFromCam must project it back to the same pixel.
fn test_cam_ray_from_img_to_img<M: CameraModel>(params: &[f64], x0: f64, y0: f64) {
    let ray = camera_model_cam_ray_from_img(M::MODEL_ID, params, Vector2d::new(x0, y0))
        .expect("CamRayFromImg failed");
    assert_near(ray.norm(), 1.0, 1e-12);
    let xy = camera_model_img_from_cam(M::MODEL_ID, params, ray, true).expect("ImgFromCam failed");
    // The pixel round-trip is floored by the iterative Newton undistortion in CamFromImg
    // (~1e-7 worst case); matches the tolerance of the sibling CamFromImg/ImgFromCam
    // round-trip in TestCamFromImgToImg.
    assert_near(xy.x, x0, 1e-6);
    assert_near(xy.y, y0, 1e-6);
}

fn test_model<M: CameraModel>(params: &[f64]) {
    assert!(camera_model_verify_params(M::MODEL_ID, params));

    let default_params = camera_model_initialize_params(M::MODEL_ID, 100.0, 100, 100);
    assert!(camera_model_verify_params(M::MODEL_ID, &default_params));

    assert_eq!(camera_model_params_info(M::MODEL_ID), M::PARAMS_INFO);
    assert_eq!(
        camera_model_focal_length_idxs(M::MODEL_ID),
        M::FOCAL_LENGTH_IDXS
    );
    assert_eq!(
        camera_model_principal_point_idxs(M::MODEL_ID),
        M::PRINCIPAL_POINT_IDXS
    );
    assert_eq!(
        camera_model_extra_params_idxs(M::MODEL_ID),
        M::EXTRA_PARAMS_IDXS
    );
    assert!(camera_model_meta_data_params_idxs(M::MODEL_ID).is_empty());
    assert_eq!(camera_model_num_params(M::MODEL_ID), M::NUM_PARAMS);

    assert!(!camera_model_has_bogus_params(
        M::MODEL_ID,
        &default_params,
        100,
        100,
        0.1,
        2.0,
        1.0
    ));
    assert!(camera_model_has_bogus_params(
        M::MODEL_ID,
        &default_params,
        100,
        100,
        0.1,
        0.5,
        1.0
    ));
    assert!(camera_model_has_bogus_params(
        M::MODEL_ID,
        &default_params,
        100,
        100,
        1.5,
        2.0,
        1.0
    ));
    if !M::EXTRA_PARAMS_IDXS.is_empty() {
        assert!(camera_model_has_bogus_params(
            M::MODEL_ID,
            &default_params,
            100,
            100,
            0.1,
            2.0,
            -0.1
        ));
    }

    assert_eq!(
        camera_model_cam_from_img_threshold(M::MODEL_ID, params, 0.0),
        0.0
    );
    assert!(camera_model_cam_from_img_threshold(M::MODEL_ID, params, 1.0) > 0.0);
    assert_eq!(
        camera_model_cam_from_img_threshold(M::MODEL_ID, &default_params, 1.0),
        1.0 / 100.0
    );

    assert!(exists_camera_model_with_name(M::MODEL_NAME));
    assert!(!exists_camera_model_with_name(&format!(
        "{}FOO",
        M::MODEL_NAME
    )));

    assert!(exists_camera_model_with_id(M::MODEL_ID));
    // static_cast<CameraModelId>(123456789): a Rust enum cannot hold that value, so the
    // conversion from the raw value is where it is rejected.
    assert!(!CameraModelId::from_i32(123456789).is_some_and(exists_camera_model_with_id));

    assert_eq!(
        camera_model_name_to_id(camera_model_id_to_name(M::MODEL_ID)),
        M::MODEL_ID
    );
    assert_eq!(
        camera_model_id_to_name(camera_model_name_to_id(M::MODEL_NAME)),
        M::MODEL_NAME
    );

    // Same accumulating double loop counters as the C++ test.
    let mut u = -0.5;
    while u <= 0.5 {
        let mut v = -0.5;
        while v <= 0.5 {
            for w in [0.5, 1.0, 2.0] {
                test_cam_to_cam_from_img::<M>(params, u, v, w);
            }
            v += 0.1;
        }
        u += 0.1;
    }

    for x in (0..=800).step_by(50) {
        for y in (0..=800).step_by(50) {
            let xy = Vector2d::new(x as f64, y as f64);
            if camera_model_is_perspective_fisheye(M::MODEL_ID)
                && !fisheye_camera_model_is_valid_pixel(M::MODEL_ID, params, xy)
            {
                continue;
            }
            test_cam_from_img_to_img::<M>(params, x as f64, y as f64);
            test_cam_ray_from_img_to_img::<M>(params, x as f64, y as f64);
        }
    }

    let pp_idxs = M::PRINCIPAL_POINT_IDXS;
    test_cam_from_img_to_img::<M>(params, params[pp_idxs[0]], params[pp_idxs[1]]);
    test_cam_ray_from_img_to_img::<M>(params, params[pp_idxs[0]], params[pp_idxs[1]]);

    // Analytic ImgFromCamWithJac is validated separately in models_jacobian_test.cc.
}

#[test]
fn simple_pinhole_nominal() {
    test_model::<SimplePinholeCameraModel>(&[655.123, 386.123, 511.123]);
}

#[test]
fn pinhole_nominal() {
    test_model::<PinholeCameraModel>(&[651.123, 655.123, 386.123, 511.123]);
}

#[test]
fn spherical_nominal() {
    type M = EquirectangularCameraModel;
    let id = M::MODEL_ID;
    // params = (w, h) of the equirectangular image.
    let params = vec![800.0, 400.0];
    assert!(camera_model_verify_params(id, &params));

    assert_eq!(camera_model_params_info(id), "w,h");
    assert!(camera_model_focal_length_idxs(id).is_empty());
    assert!(camera_model_principal_point_idxs(id).is_empty());
    assert!(camera_model_extra_params_idxs(id).is_empty());
    assert_eq!(camera_model_meta_data_params_idxs(id), &[0usize, 1]);
    assert_eq!(camera_model_num_params(id), 2);

    // Perspective models have no metadata parameters.
    assert!(camera_model_meta_data_params_idxs(CameraModelId::Pinhole).is_empty());

    // EQUIRECTANGULAR is non-perspective, spherical, and never has bogus parameters.
    assert!(!camera_model_is_perspective(id));
    assert!(camera_model_is_perspective(CameraModelId::Pinhole));
    assert!(camera_model_is_spherical(id));
    assert!(!camera_model_is_spherical(CameraModelId::Pinhole));
    assert!(!camera_model_is_spherical(CameraModelId::OpenCVFisheye));
    assert!(!camera_model_has_bogus_params(
        id, &params, 800, 400, 0.1, 2.0, 1.0
    ));

    // InitializeParams ignores the focal length and returns (w, h).
    assert_eq!(
        camera_model_initialize_params(id, /*focal_length=*/ 123.0, 800, 400),
        params
    );

    // Full-sphere bearing round-trip CamRayFromImg -> ImgFromCam over the image interior
    // (avoiding the azimuth seam at x in {0, w} and the poles at y in {0, h}, where the
    // azimuth is undefined).
    for xi in (40..=760).step_by(40) {
        for yi in (40..=360).step_by(40) {
            test_cam_ray_from_img_to_img::<M>(&params, xi as f64, yi as f64);
        }
    }

    // Back-hemisphere pixels (azimuth near +/-pi) have no forward 2D representation, so the
    // 2D CamFromImg fails there while CamRayFromImg still yields a valid unit bearing.
    assert!(camera_model_cam_from_img(id, &params, Vector2d::new(0.0, 200.0)).is_none());
    assert!(camera_model_cam_ray_from_img(id, &params, Vector2d::new(0.0, 200.0)).is_some());

    // A forward-hemisphere pixel (azimuth ~0, image center column) also round-trips through
    // the 2D CamFromImg / ImgFromCam path.
    test_cam_from_img_to_img::<M>(&params, 400.0, 200.0);
}

#[test]
fn simple_radial_nominal() {
    test_model::<SimpleRadialCameraModel>(&[651.123, 386.123, 511.123, 0.0]);
    test_model::<SimpleRadialCameraModel>(&[651.123, 386.123, 511.123, 0.1]);
}

#[test]
fn radial_nominal() {
    test_model::<RadialCameraModel>(&[651.123, 386.123, 511.123, 0.0, 0.0]);
    test_model::<RadialCameraModel>(&[651.123, 386.123, 511.123, 0.1, 0.0]);
    test_model::<RadialCameraModel>(&[651.123, 386.123, 511.12, 0.0, 0.05]);
    test_model::<RadialCameraModel>(&[651.123, 386.123, 511.123, 0.05, 0.03]);
}

#[test]
fn open_cv_nominal() {
    test_model::<OpenCVCameraModel>(&[
        651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001,
    ]);
}

#[test]
fn open_cv_fisheye_nominal() {
    test_model::<OpenCVFisheyeCameraModel>(&[
        651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001,
    ]);
}

#[test]
fn full_open_cv_nominal() {
    test_model::<FullOpenCVCameraModel>(&[
        651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001, 0.001, 0.02, -0.02, 0.001,
    ]);
}

#[test]
fn fov_nominal() {
    test_model::<FOVCameraModel>(&[651.123, 655.123, 386.123, 511.123, 0.0]);
    test_model::<FOVCameraModel>(&[651.123, 655.123, 386.123, 511.123, 0.9]);
    test_model::<FOVCameraModel>(&[651.123, 655.123, 386.123, 511.123, 1e-6]);
    test_model::<FOVCameraModel>(&[651.123, 655.123, 386.123, 511.123, 1e-2]);
    assert_eq!(
        *camera_model_initialize_params(CameraModelId::FOV, 100.0, 100, 100)
            .last()
            .unwrap(),
        1e-2
    );
}

#[test]
fn simple_radial_fisheye_nominal() {
    test_model::<SimpleRadialFisheyeCameraModel>(&[651.123, 386.123, 511.123, 0.0]);
    test_model::<SimpleRadialFisheyeCameraModel>(&[651.123, 386.123, 511.123, 0.1]);
}

#[test]
fn radial_fisheye_nominal() {
    test_model::<RadialFisheyeCameraModel>(&[651.123, 386.123, 511.123, 0.0, 0.0]);
    test_model::<RadialFisheyeCameraModel>(&[651.123, 386.123, 511.123, 0.0, 0.1]);
    test_model::<RadialFisheyeCameraModel>(&[651.123, 386.123, 511.123, 0.0, 0.05]);
    test_model::<RadialFisheyeCameraModel>(&[651.123, 386.123, 511.123, 0.0, 0.03]);
}

#[test]
fn thin_prism_fisheye_nominal() {
    test_model::<ThinPrismFisheyeCameraModel>(&[
        651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001, 0.001, 0.02, -0.02, 0.001,
    ]);
}

#[test]
fn rad_tan_thin_prism_fisheye_nominal() {
    let params = [
        651.123, 655.123, 386.123, 511.123, -0.0232, 0.0924, -0.0591, 0.003, 0.0048, -0.0009,
        0.0002, 0.0005, -0.0009, -0.0001, 0.00007, -0.00017,
    ];
    test_model::<RadTanThinPrismFisheyeModel>(&params);
}

#[test]
fn simple_division_nominal() {
    test_model::<SimpleDivisionCameraModel>(&[651.123, 386.123, 511.123, 0.0]);
    test_model::<SimpleDivisionCameraModel>(&[651.123, 386.123, 511.123, 0.1]);
    test_model::<SimpleDivisionCameraModel>(&[651.123, 386.123, 511.123, -0.1]);
}

#[test]
fn division_nominal() {
    test_model::<DivisionCameraModel>(&[651.123, 655.123, 386.123, 511.123, 0.0]);
    test_model::<DivisionCameraModel>(&[651.123, 655.123, 386.123, 511.123, 0.1]);
    test_model::<DivisionCameraModel>(&[651.123, 655.123, 386.123, 511.123, -0.1]);
}

#[test]
fn simple_fisheye_camera_nominal() {
    test_model::<SimpleFisheyeCameraModel>(&[651.123, 386.123, 511.123]);
}

#[test]
fn fisheye_camera_nominal() {
    test_model::<FisheyeCameraModel>(&[651.123, 655.123, 386.123, 511.123]);
}

#[test]
fn eucm_camera_nominal() {
    test_model::<EUCMCameraModel>(&[651.123, 655.123, 386.123, 511.123, 0.56, 0.87]);
    test_model::<EUCMCameraModel>(&[400.0, 400.0, 400.0, 400.0, 0.88, 0.64]);
    test_model::<EUCMCameraModel>(&[651.123, 655.123, 386.123, 511.123, 0.0, 1.0]);
    test_model::<EUCMCameraModel>(&[651.123, 655.123, 386.123, 511.123, 0.5, 1.0]);
}

#[test]
fn eucm_camera_rejects_invalid_extra_params() {
    for params in [
        [651.123, 655.123, 386.123, 511.123, -0.01, 0.87],
        [651.123, 655.123, 386.123, 511.123, 1.01, 0.87],
        [651.123, 655.123, 386.123, 511.123, 0.56, 0.00],
        [651.123, 655.123, 386.123, 511.123, 0.56, -0.3],
    ] {
        assert!(camera_model_has_bogus_params(
            CameraModelId::EUCM,
            &params,
            1000,
            1000,
            0.1,
            2.0,
            1.0
        ));
    }
}

#[test]
fn camera_model_rescale_perspective() {
    // Distinct per-axis scale factors to verify each is applied to the right parameter; all
    // results are exactly representable.
    let scale_x = 2.0;
    let scale_y = 3.0;

    // Two focal lengths (fx, fy): each scales along its own axis, as does the principal
    // point (cx, cy).
    {
        let mut params = vec![100.0, 200.0, 50.0, 80.0]; // fx, fy, cx, cy
        camera_model_rescale(CameraModelId::Pinhole, scale_x, scale_y, &mut params);
        assert_eq!(params, vec![200.0, 600.0, 100.0, 240.0]);
    }

    // Single shared focal length scales by the mean of the two factors.
    {
        let mut params = vec![100.0, 50.0, 80.0]; // f, cx, cy
        camera_model_rescale(CameraModelId::SimplePinhole, scale_x, scale_y, &mut params);
        assert_eq!(params, vec![250.0, 100.0, 240.0]); // f *= 2.5
    }

    // Extra (distortion) parameters are resolution independent and untouched.
    {
        let mut params = vec![100.0, 50.0, 80.0, 0.3]; // f, cx, cy, k
        camera_model_rescale(CameraModelId::SimpleRadial, scale_x, scale_y, &mut params);
        assert_eq!(params, vec![250.0, 100.0, 240.0, 0.3]);
    }
}

#[test]
fn camera_model_rescale_spherical() {
    // The (w, h) image-size parameters track the rescaled image dimensions.
    let mut params = vec![800.0, 400.0]; // w, h
    camera_model_rescale(
        CameraModelId::Equirectangular,
        /*scale_x=*/ 2.0,
        /*scale_y=*/ 0.5,
        &mut params,
    );
    assert_eq!(params, vec![1600.0, 200.0]);
}
