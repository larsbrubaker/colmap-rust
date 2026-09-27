//! Rust-only tests of the analytic projection Jacobians (`jacobian*.rs`) and
//! `cam_ray_from_img_jacobian`, beside the 1:1 port of `models_jacobian_test.cc` in
//! `jacobian_tests.rs`: the value-only and failure paths COLMAP's grid never reaches, the
//! dispatch's handling of a failed projection, and a check that `cam_ray_from_img_jacobian`
//! really is the derivative of `camera_model_cam_ray_from_img`.

use crate::linalg::{Matrix2x3d, Vector2d, Vector3d};

use super::super::*;

/// One parameter set per model (COLMAP's test values), for the all-models sweeps.
fn all_models() -> Vec<(CameraModelId, Vec<f64>)> {
    vec![
        (
            CameraModelId::SimplePinhole,
            vec![655.123, 386.123, 511.123],
        ),
        (
            CameraModelId::Pinhole,
            vec![651.123, 655.123, 386.123, 511.123],
        ),
        (
            CameraModelId::SimpleRadial,
            vec![651.123, 386.123, 511.123, 0.1],
        ),
        (
            CameraModelId::Radial,
            vec![651.123, 386.123, 511.123, 0.05, 0.03],
        ),
        (
            CameraModelId::OpenCV,
            vec![
                651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001,
            ],
        ),
        (
            CameraModelId::FullOpenCV,
            vec![
                651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001, 0.001, 0.02,
                -0.02, 0.001,
            ],
        ),
        (
            CameraModelId::FOV,
            vec![651.123, 655.123, 386.123, 511.123, 0.9],
        ),
        (
            CameraModelId::SimpleRadialFisheye,
            vec![651.123, 386.123, 511.123, 0.1],
        ),
        (
            CameraModelId::RadialFisheye,
            vec![651.123, 386.123, 511.123, 0.1, 0.02],
        ),
        (
            CameraModelId::OpenCVFisheye,
            vec![
                651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.001, 0.001,
            ],
        ),
        (
            CameraModelId::ThinPrismFisheye,
            vec![
                651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.001, 0.001, 0.001, 0.002,
                0.001, -0.001,
            ],
        ),
        (
            CameraModelId::RadTanThinPrismFisheye,
            vec![
                651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.005, 0.001, 0.0005, 0.0002,
                -0.001, 0.001, 0.001, -0.001, 0.0005, -0.0005,
            ],
        ),
        (
            CameraModelId::SimpleDivision,
            vec![651.123, 386.123, 511.123, 0.1],
        ),
        (
            CameraModelId::Division,
            vec![651.123, 655.123, 386.123, 511.123, -0.1],
        ),
        (
            CameraModelId::SimpleFisheye,
            vec![651.123, 386.123, 511.123],
        ),
        (
            CameraModelId::Fisheye,
            vec![651.123, 655.123, 386.123, 511.123],
        ),
        (
            CameraModelId::EUCM,
            vec![651.123, 655.123, 386.123, 511.123, 0.6, 1.2],
        ),
        (CameraModelId::Equirectangular, vec![1000.0, 500.0]),
    ]
}

/// Every model's kernel must write the same pixel whether or not it computes Jacobians:
/// COLMAP's value-only path (`J_params = J_uvw = nullptr`) skips work, never changes the
/// projection. Bit-identical, through the dispatch and through the typed kernel.
#[test]
fn rust_only_img_from_cam_with_jac_value_path_matches_jacobian_path() {
    assert_eq!(all_models().len(), CameraModelId::ALL.len() - 1);
    for (model_id, params) in all_models() {
        for uvw in [
            Vector3d::new(0.1, -0.2, 1.0),
            Vector3d::new(-0.4, 0.3, 0.5),
            Vector3d::new(0.0, 0.0, 2.0),
            Vector3d::new(1e-20, -1e-20, 1.0),
        ] {
            let mut j_uvw = Matrix2x3d::zeros();
            let with =
                camera_model_img_from_cam_with_jac(model_id, &params, uvw, Some(&mut j_uvw), true)
                    .unwrap_or_else(|| panic!("{model_id} projects {uvw:?}"));
            let without = camera_model_img_from_cam_with_jac(model_id, &params, uvw, None, true)
                .unwrap_or_else(|| panic!("{model_id} projects {uvw:?} without J"));
            assert_eq!(
                with.x.to_bits(),
                without.x.to_bits(),
                "{model_id} x at {uvw:?}"
            );
            assert_eq!(
                with.y.to_bits(),
                without.y.to_bits(),
                "{model_id} y at {uvw:?}"
            );

            // J_params alone must not change the pixel either.
            let mut x = 0.0;
            let mut y = 0.0;
            let num_params = camera_model_num_params(model_id).expect("a real model");
            let mut j_params = vec![f64::NAN; 2 * num_params];
            let ok = typed_with_params_only(model_id, &params, uvw, &mut x, &mut y, &mut j_params);
            assert!(ok, "{model_id} projects {uvw:?} with J_params");
            assert_eq!(x.to_bits(), with.x.to_bits(), "{model_id} x with J_params");
            assert_eq!(y.to_bits(), with.y.to_bits(), "{model_id} y with J_params");
            assert!(
                j_params.iter().all(|j| j.is_finite()),
                "{model_id} writes every J_params entry at {uvw:?}: {j_params:?}"
            );
        }
    }
}

/// `M::img_from_cam_with_jac` with only `J_params` requested, for a runtime id.
fn typed_with_params_only(
    model_id: CameraModelId,
    params: &[f64],
    uvw: Vector3d,
    x: &mut f64,
    y: &mut f64,
    j_params: &mut [f64],
) -> bool {
    fn run<M: CameraModelWithJac>(
        params: &[f64],
        uvw: Vector3d,
        x: &mut f64,
        y: &mut f64,
        j_params: &mut [f64],
    ) -> bool {
        M::img_from_cam_with_jac(
            params,
            uvw.x,
            uvw.y,
            uvw.z,
            x,
            y,
            Some(j_params),
            None,
            true,
        )
    }
    match model_id {
        CameraModelId::SimplePinhole => {
            run::<SimplePinholeCameraModel>(params, uvw, x, y, j_params)
        }
        CameraModelId::Pinhole => run::<PinholeCameraModel>(params, uvw, x, y, j_params),
        CameraModelId::SimpleRadial => run::<SimpleRadialCameraModel>(params, uvw, x, y, j_params),
        CameraModelId::Radial => run::<RadialCameraModel>(params, uvw, x, y, j_params),
        CameraModelId::OpenCV => run::<OpenCVCameraModel>(params, uvw, x, y, j_params),
        CameraModelId::FullOpenCV => run::<FullOpenCVCameraModel>(params, uvw, x, y, j_params),
        CameraModelId::FOV => run::<FOVCameraModel>(params, uvw, x, y, j_params),
        CameraModelId::SimpleRadialFisheye => {
            run::<SimpleRadialFisheyeCameraModel>(params, uvw, x, y, j_params)
        }
        CameraModelId::RadialFisheye => {
            run::<RadialFisheyeCameraModel>(params, uvw, x, y, j_params)
        }
        CameraModelId::OpenCVFisheye => {
            run::<OpenCVFisheyeCameraModel>(params, uvw, x, y, j_params)
        }
        CameraModelId::ThinPrismFisheye => {
            run::<ThinPrismFisheyeCameraModel>(params, uvw, x, y, j_params)
        }
        CameraModelId::RadTanThinPrismFisheye => {
            run::<RadTanThinPrismFisheyeModel>(params, uvw, x, y, j_params)
        }
        CameraModelId::SimpleDivision => {
            run::<SimpleDivisionCameraModel>(params, uvw, x, y, j_params)
        }
        CameraModelId::Division => run::<DivisionCameraModel>(params, uvw, x, y, j_params),
        CameraModelId::SimpleFisheye => {
            run::<SimpleFisheyeCameraModel>(params, uvw, x, y, j_params)
        }
        CameraModelId::Fisheye => run::<FisheyeCameraModel>(params, uvw, x, y, j_params),
        CameraModelId::EUCM => run::<EUCMCameraModel>(params, uvw, x, y, j_params),
        CameraModelId::Equirectangular => {
            run::<EquirectangularCameraModel>(params, uvw, x, y, j_params)
        }
        CameraModelId::Invalid => unreachable!("no kernel for Invalid"),
    }
}

/// The kernels fail exactly where `ImgFromCam` does, and the dispatch then returns `None`
/// and leaves the caller's Jacobian untouched (COLMAP only writes `*J_uvw` on success).
#[test]
fn rust_only_img_from_cam_with_jac_failures_match_img_from_cam() {
    let cases: Vec<(CameraModelId, Vec<f64>, Vector3d, bool)> = vec![
        // Behind or on the camera plane, with and without the cheirality check.
        (
            CameraModelId::Pinhole,
            vec![1.0, 1.0, 0.0, 0.0],
            Vector3d::new(0.1, 0.2, -1.0),
            true,
        ),
        (
            CameraModelId::Pinhole,
            vec![1.0, 1.0, 0.0, 0.0],
            Vector3d::new(0.1, 0.2, -1.0),
            false,
        ),
        (
            CameraModelId::Pinhole,
            vec![1.0, 1.0, 0.0, 0.0],
            Vector3d::new(0.1, 0.2, 0.0),
            false,
        ),
        (
            CameraModelId::OpenCVFisheye,
            vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            Vector3d::new(0.1, 0.2, 1e-17),
            true,
        ),
        // The division model's negative discriminant (w^2 < 4 k rho2).
        (
            CameraModelId::SimpleDivision,
            vec![1.0, 0.0, 0.0, 1.0],
            Vector3d::new(1.0, 1.0, 1.0),
            true,
        ),
        // EUCM: beta * q + w^2 < 0, and a denominator at the camera plane.
        (
            CameraModelId::EUCM,
            vec![1.0, 1.0, 0.0, 0.0, 0.5, -10.0],
            Vector3d::new(1.0, 1.0, 1.0),
            true,
        ),
        (
            CameraModelId::EUCM,
            vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            Vector3d::new(0.0, 0.0, -1.0),
            false,
        ),
        // EQUIRECTANGULAR: only the zero vector has no direction.
        (
            CameraModelId::Equirectangular,
            vec![100.0, 50.0],
            Vector3d::zeros(),
            true,
        ),
        (
            CameraModelId::Equirectangular,
            vec![100.0, 50.0],
            Vector3d::new(0.0, 0.0, -1.0),
            true,
        ),
    ];
    for (model_id, params, uvw, check_cheirality) in cases {
        let expected = camera_model_img_from_cam(model_id, &params, uvw, check_cheirality);
        let sentinel = Matrix2x3d::new(7.0, 7.0, 7.0, 7.0, 7.0, 7.0);
        let mut j_uvw = sentinel;
        let got = camera_model_img_from_cam_with_jac(
            model_id,
            &params,
            uvw,
            Some(&mut j_uvw),
            check_cheirality,
        );
        assert_eq!(
            got.is_some(),
            expected.is_some(),
            "{model_id} at {uvw:?} (check_cheirality {check_cheirality})"
        );
        if let (Some(got), Some(expected)) = (got, expected) {
            assert!(
                got.is_approx_with(expected, 1e-12),
                "{model_id}: {got:?} vs {expected:?}"
            );
        } else {
            assert_eq!(j_uvw, sentinel, "{model_id}: J_uvw written on failure");
        }
    }
}

#[test]
#[should_panic(expected = "Camera model does not exist")]
fn rust_only_camera_model_img_from_cam_with_jac_invalid_id_panics() {
    let _ = camera_model_img_from_cam_with_jac(
        CameraModelId::Invalid,
        &[],
        Vector3d::new(0.0, 0.0, 1.0),
        None,
        true,
    );
}

/// `cam_ray_from_img_jacobian` claims to be `d(cam ray) / d(pixel)` of
/// `camera_model_cam_ray_from_img`. Check that claim against central differences of the
/// unprojection itself, for a pinhole, a distorted pinhole (iterative undistortion), a
/// fisheye and the spherical model. Step 1e-3 px: truncation error ~ h^2 |J'''| and
/// rounding error ~ eps / h both stay far below the 1e-9 bound on entries of size ~1e-3.
#[test]
fn rust_only_cam_ray_from_img_jacobian_is_the_derivative_of_cam_ray_from_img() {
    let cases: Vec<(CameraModelId, Vec<f64>, Vector2d)> = vec![
        (
            CameraModelId::Pinhole,
            vec![651.123, 655.123, 386.123, 511.123],
            Vector2d::new(500.0, 300.0),
        ),
        (
            CameraModelId::OpenCV,
            vec![
                651.123, 655.123, 386.123, 511.123, -0.1, 0.02, -0.001, 0.001,
            ],
            Vector2d::new(250.0, 600.0),
        ),
        (
            CameraModelId::OpenCVFisheye,
            vec![
                651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.001, 0.001,
            ],
            Vector2d::new(100.0, 150.0),
        ),
        (
            CameraModelId::Equirectangular,
            vec![1000.0, 500.0],
            Vector2d::new(730.0, 120.0),
        ),
    ];
    let h = 1e-3;
    for (model_id, params, xy) in cases {
        let ray = camera_model_cam_ray_from_img(model_id, &params, xy).expect("unprojects");
        let mut j_uvw = Matrix2x3d::zeros();
        let reprojected =
            camera_model_img_from_cam_with_jac(model_id, &params, ray, Some(&mut j_uvw), true)
                .expect("projects");
        assert!(
            reprojected.is_approx_with(xy, 1e-9),
            "{model_id}: round trip"
        );
        let j_ray = cam_ray_from_img_jacobian(ray, j_uvw).expect("full rank");
        for (col, step) in [Vector2d::new(h, 0.0), Vector2d::new(0.0, h)]
            .into_iter()
            .enumerate()
        {
            let plus = camera_model_cam_ray_from_img(model_id, &params, xy + step).unwrap();
            let minus = camera_model_cam_ray_from_img(model_id, &params, xy - step).unwrap();
            let numeric = (plus - minus) / (2.0 * h);
            let analytic = j_ray.col(col);
            assert!(
                (numeric - analytic).norm() <= 1e-9,
                "{model_id} column {col}: numeric {numeric:?} vs analytic {analytic:?}"
            );
        }
    }
}
