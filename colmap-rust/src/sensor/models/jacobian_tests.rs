//! Port of `colmap/sensor/models_jacobian_test.cc`, 1:1: one test per gtest
//! `TEST(Suite, Name)`, named `suite_name` in snake case, with the same parameters, grids and
//! tolerances, plus `rust_only_*` tests of the paths COLMAP's test does not reach. A unit-test
//! module rather than a file under `tests/` because COLMAP checks the analytic Jacobians
//! (`jacobian*.rs`) against `ImgFromCam` on `ceres::Jet<double, num_params + 3>`, and the Jet
//! here (`jet.rs`) is crate-private. Run: `cargo test -p colmap-rust --lib jacobian_tests`.
//!
//! Tier B: analytic and automatic derivatives are different formulas for the same value, so
//! they agree to COLMAP's 1e-10, not bitwise. Failed expectations are collected and reported
//! together, like gtest's non-fatal `EXPECT_*`; an `ASSERT_*` failure ends the helper call.

use crate::linalg::{Matrix2d, Matrix2x3d, Vector3d};

use super::jet::Jet;
use super::*;

/// The failed non-fatal expectations of one test.
#[derive(Default)]
struct Log {
    failures: Vec<String>,
}

impl Log {
    /// `EXPECT_TRUE`; returns `ok` so an `ASSERT_TRUE` can return early.
    fn check(&mut self, ok: bool, what: impl FnOnce() -> String) -> bool {
        if !ok {
            self.failures.push(what());
        }
        ok
    }

    /// `EXPECT_NEAR(a, b, tol)`.
    fn near(&mut self, a: f64, b: f64, tol: f64, what: impl FnOnce() -> String) {
        let ok = (a - b).abs() <= tol;
        self.check(ok, || format!("{}: {a} vs {b} (tol {tol})", what()));
    }

    fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} failed expectations:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

/// Validate ImgFromCamWithJac against ImgFromCam using Jets. `N` is COLMAP's
/// `kNumDerivs = num_params + 3`, spelled out because a Rust const generic cannot be computed
/// from the model's associated constant.
fn test_img_from_cam_with_jac<M: CameraModelWithJac, const N: usize>(
    log: &mut Log,
    params: &[f64],
    u: f64,
    v: f64,
    w: f64,
) {
    let num_params = M::NUM_PARAMS;
    const NUM_UVW: usize = 3;
    assert_eq!(N, num_params + NUM_UVW, "Jet width for {}", M::MODEL_NAME);
    let at = || format!("{} ({u}, {v}, {w})", M::MODEL_NAME);

    // Compute using ImgFromCamWithJac
    let mut x_jac = 0.0;
    let mut y_jac = 0.0;
    let mut j_params = vec![0.0; 2 * num_params];
    let mut j_uvw = [0.0; 2 * NUM_UVW];
    if !log.check(
        M::img_from_cam_with_jac(
            params,
            u,
            v,
            w,
            &mut x_jac,
            &mut y_jac,
            Some(&mut j_params),
            Some(&mut j_uvw),
            true,
        ),
        || format!("ImgFromCamWithJac {}", at()),
    ) {
        return;
    }

    // Compute using ImgFromCam with Jets for auto-differentiation
    // Jets track derivatives: first kNumParams for params, next 3 for u, v, w.
    let params_jet: Vec<Jet<N>> = (0..num_params)
        .map(|i| Jet::variable(params[i], i))
        .collect();
    let u_jet = Jet::<N>::variable(u, num_params);
    let v_jet = Jet::<N>::variable(v, num_params + 1);
    let w_jet = Jet::<N>::variable(w, num_params + 2);

    let mut x_jet = Jet::<N>::constant(0.0);
    let mut y_jet = Jet::<N>::constant(0.0);
    if !log.check(
        M::img_from_cam(
            &params_jet,
            u_jet,
            v_jet,
            w_jet,
            &mut x_jet,
            &mut y_jet,
            true,
        ),
        || format!("ImgFromCam<Jet> {}", at()),
    ) {
        return;
    }

    // Compare function values
    log.near(x_jac, x_jet.a, 1e-10, || format!("x {}", at()));
    log.near(y_jac, y_jet.a, 1e-10, || format!("y {}", at()));

    // Compare Jacobian w.r.t. params (2 x num_params, row-major)
    for i in 0..num_params {
        log.near(j_params[i], x_jet.v[i], 1e-10, || {
            format!("J_params mismatch at dx/dparam[{i}] {}", at())
        });
        log.near(j_params[num_params + i], y_jet.v[i], 1e-10, || {
            format!("J_params mismatch at dy/dparam[{i}] {}", at())
        });
    }

    // Compare Jacobian w.r.t. uvw (2 x 3, row-major)
    for i in 0..NUM_UVW {
        log.near(j_uvw[i], x_jet.v[num_params + i], 1e-10, || {
            format!("J_uvw mismatch at dx/d(uvw)[{i}] {}", at())
        });
        log.near(j_uvw[NUM_UVW + i], y_jet.v[num_params + i], 1e-10, || {
            format!("J_uvw mismatch at dy/d(uvw)[{i}] {}", at())
        });
    }
}

/// Validate the runtime dispatch and the unprojection Jacobian derived from it.
fn test_cam_ray_jacobian<M: CameraModelWithJac>(
    log: &mut Log,
    params: &[f64],
    u: f64,
    v: f64,
    w: f64,
) {
    let uvw = Vector3d::new(u, v, w);
    let at = || format!("{} ({u}, {v}, {w})", M::MODEL_NAME);

    // Reference: the templated per-model kernel, written 2x3 row-major.
    let mut x_ref = 0.0;
    let mut y_ref = 0.0;
    let mut j_ref_data = [0.0; 6];
    if !log.check(
        M::img_from_cam_with_jac(
            params,
            u,
            v,
            w,
            &mut x_ref,
            &mut y_ref,
            /* J_params= */ None,
            Some(&mut j_ref_data),
            true,
        ),
        || format!("ImgFromCamWithJac {}", at()),
    ) {
        return;
    }
    let j_ref = Matrix2x3d::from_row_major(j_ref_data);

    // 1. The runtime dispatch must agree with the templated kernel. The compiler
    // may round the separately optimized call paths slightly differently.
    let mut j_uvw = Matrix2x3d::zeros();
    let xy = camera_model_img_from_cam_with_jac(M::MODEL_ID, params, uvw, Some(&mut j_uvw), true);
    let Some(xy) = xy else {
        log.check(false, || format!("dispatch has value {}", at()));
        return;
    };
    log.near(xy.x, x_ref, 1e-10, || format!("dispatch x {}", at()));
    log.near(xy.y, y_ref, 1e-10, || format!("dispatch y {}", at()));
    log.check(j_uvw.is_approx_with(j_ref, 1e-12), || {
        format!("dispatch J_uvw isApprox {}", at())
    });

    // Passing nullptr must skip the Jacobian but still project.
    let xy_no_jac = camera_model_img_from_cam_with_jac(M::MODEL_ID, params, uvw, None, true);
    let Some(xy_no_jac) = xy_no_jac else {
        log.check(false, || {
            format!("dispatch without Jacobian has value {}", at())
        });
        return;
    };
    log.check(xy_no_jac.is_approx_with(xy, 1e-12), || {
        format!("dispatch without Jacobian isApprox {}", at())
    });

    // 2. Central projection depends only on the ray direction, so the projection
    // is homogeneous of degree zero and Euler's identity gives J_uvw * uvw == 0.
    // This is the assumption that makes the pseudo-inverse below equal the
    // tangent-plane unprojection Jacobian; if it fails, that derivation is wrong.
    log.check(
        (j_uvw * uvw).norm() <= 1e-10 * j_uvw.norm() * uvw.norm(),
        || format!("J_uvw * uvw == 0 {}", at()),
    );

    // The closed-form pseudo-inverse is only valid at a unit bearing.
    let cam_ray = uvw.normalized();
    let Some(j_ray) = cam_ray_from_img_jacobian(cam_ray, j_uvw) else {
        log.check(false, || {
            format!("CamRayFromImgJacobian has value {}", at())
        });
        return;
    };

    // 3. Pseudo-inverse round trip: J_uvw is surjective onto image space.
    log.check(
        (j_uvw * j_ray - Matrix2d::identity()).norm() <= 1e-10,
        || format!("J_uvw * J_ray == I {}", at()),
    );

    // 4. The recovered Jacobian maps into the tangent plane at the ray.
    log.check(
        (j_ray.transpose() * uvw).norm() <= 1e-10 * j_ray.norm() * uvw.norm(),
        || format!("J_ray^T * uvw == 0 {}", at()),
    );
    log.check(
        (j_ray.transpose() * cam_ray).norm() <= 1e-10 * j_ray.norm(),
        || format!("J_ray^T * cam_ray == 0 {}", at()),
    );
}

/// Validate the analytic ImgFromCamWithJac over a grid of camera-space points. The float
/// loop counters are COLMAP's: the grid drifts from multiples of 0.1 exactly as the C++ one
/// does.
fn test_model_img_from_cam_with_jac<M: CameraModelWithJac, const N: usize>(params: &[f64]) {
    let mut log = Log::default();
    let mut u = -0.5;
    while u <= 0.5 {
        let mut v = -0.5;
        while v <= 0.5 {
            for w in [0.5, 1.0, 2.0] {
                test_img_from_cam_with_jac::<M, N>(&mut log, params, u, v, w);
                test_cam_ray_jacobian::<M>(&mut log, params, u, v, w);
            }
            v += 0.1;
        }
        u += 0.1;
    }
    log.finish();
}

#[test]
fn simple_pinhole_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<SimplePinholeCameraModel, 6>(&[655.123, 386.123, 511.123]);
}

#[test]
fn pinhole_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<PinholeCameraModel, 7>(&[
        651.123, 655.123, 386.123, 511.123,
    ]);
}

#[test]
fn simple_radial_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<SimpleRadialCameraModel, 7>(&[
        651.123, 386.123, 511.123, 0.0,
    ]);
    test_model_img_from_cam_with_jac::<SimpleRadialCameraModel, 7>(&[
        651.123, 386.123, 511.123, 0.1,
    ]);
}

#[test]
fn radial_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<RadialCameraModel, 8>(&[
        651.123, 386.123, 511.123, 0.0, 0.0,
    ]);
    test_model_img_from_cam_with_jac::<RadialCameraModel, 8>(&[
        651.123, 386.123, 511.123, 0.1, 0.0,
    ]);
    test_model_img_from_cam_with_jac::<RadialCameraModel, 8>(&[
        651.123, 386.123, 511.12, 0.0, 0.05,
    ]);
    test_model_img_from_cam_with_jac::<RadialCameraModel, 8>(&[
        651.123, 386.123, 511.123, 0.05, 0.03,
    ]);
}

#[test]
fn open_cv_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<OpenCVCameraModel, 11>(&[
        651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001,
    ]);
}

#[test]
fn full_open_cv_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<FullOpenCVCameraModel, 15>(&[
        651.123, 655.123, 386.123, 511.123, -0.471, 0.223, -0.001, 0.001, 0.001, 0.02, -0.02, 0.001,
    ]);
}

#[test]
fn fov_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<FOVCameraModel, 8>(&[
        651.123, 655.123, 386.123, 511.123, 0.9,
    ]);
    test_model_img_from_cam_with_jac::<FOVCameraModel, 8>(&[
        651.123, 655.123, 386.123, 511.123, 0.5,
    ]);
}

#[test]
fn simple_radial_fisheye_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<SimpleRadialFisheyeCameraModel, 7>(&[
        651.123, 386.123, 511.123, 0.0,
    ]);
    test_model_img_from_cam_with_jac::<SimpleRadialFisheyeCameraModel, 7>(&[
        651.123, 386.123, 511.123, 0.1,
    ]);
}

#[test]
fn radial_fisheye_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<RadialFisheyeCameraModel, 8>(&[
        651.123, 386.123, 511.123, 0.0, 0.0,
    ]);
    test_model_img_from_cam_with_jac::<RadialFisheyeCameraModel, 8>(&[
        651.123, 386.123, 511.123, 0.1, 0.02,
    ]);
}

#[test]
fn open_cv_fisheye_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<OpenCVFisheyeCameraModel, 11>(&[
        651.123, 655.123, 386.123, 511.123, 0.0, 0.0, 0.0, 0.0,
    ]);
    test_model_img_from_cam_with_jac::<OpenCVFisheyeCameraModel, 11>(&[
        651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.001, 0.001,
    ]);
}

#[test]
fn thin_prism_fisheye_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<ThinPrismFisheyeCameraModel, 15>(&[
        651.123, 655.123, 386.123, 511.123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]);
    test_model_img_from_cam_with_jac::<ThinPrismFisheyeCameraModel, 15>(&[
        651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.001, 0.001, 0.001, 0.002, 0.001, -0.001,
    ]);
}

#[test]
fn rad_tan_thin_prism_fisheye_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<RadTanThinPrismFisheyeModel, 19>(&[
        651.123, 655.123, 386.123, 511.123, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0,
    ]);
    test_model_img_from_cam_with_jac::<RadTanThinPrismFisheyeModel, 19>(&[
        651.123, 655.123, 386.123, 511.123, -0.05, 0.02, -0.005, 0.001, 0.0005, 0.0002, -0.001,
        0.001, 0.001, -0.001, 0.0005, -0.0005,
    ]);
}

#[test]
fn simple_fisheye_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<SimpleFisheyeCameraModel, 6>(&[651.123, 386.123, 511.123]);
}

#[test]
fn fisheye_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<FisheyeCameraModel, 7>(&[
        651.123, 655.123, 386.123, 511.123,
    ]);
}

#[test]
fn simple_division_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<SimpleDivisionCameraModel, 7>(&[
        651.123, 386.123, 511.123, 0.0,
    ]);
    test_model_img_from_cam_with_jac::<SimpleDivisionCameraModel, 7>(&[
        651.123, 386.123, 511.123, 0.1,
    ]);
    test_model_img_from_cam_with_jac::<SimpleDivisionCameraModel, 7>(&[
        651.123, 386.123, 511.123, -0.1,
    ]);
}

#[test]
fn division_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<DivisionCameraModel, 8>(&[
        651.123, 655.123, 386.123, 511.123, 0.0,
    ]);
    test_model_img_from_cam_with_jac::<DivisionCameraModel, 8>(&[
        651.123, 655.123, 386.123, 511.123, 0.1,
    ]);
    test_model_img_from_cam_with_jac::<DivisionCameraModel, 8>(&[
        651.123, 655.123, 386.123, 511.123, -0.1,
    ]);
}

#[test]
fn eucm_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<EUCMCameraModel, 9>(&[
        651.123, 655.123, 386.123, 511.123, 0.0, 1.0,
    ]);
    test_model_img_from_cam_with_jac::<EUCMCameraModel, 9>(&[
        651.123, 655.123, 386.123, 511.123, 0.6, 1.2,
    ]);
}

#[test]
fn equirectangular_img_from_cam_with_jac() {
    test_model_img_from_cam_with_jac::<EquirectangularCameraModel, 5>(&[1000.0, 500.0]);
}

#[test]
fn cam_ray_from_img_jacobian_rank_deficient_returns_nullopt() {
    // Rank 1: both image directions respond identically, so the projection is
    // not locally invertible and there is no unprojection Jacobian.
    let cam_ray = Vector3d::new(0.0, 0.0, 1.0);
    let rank1 = Matrix2x3d::new(1.0, 2.0, 3.0, 2.0, 4.0, 6.0);
    assert!(cam_ray_from_img_jacobian(cam_ray, rank1).is_none());

    assert!(cam_ray_from_img_jacobian(cam_ray, Matrix2x3d::zeros()).is_none());

    // A well-conditioned Jacobian is accepted and inverts cleanly.
    let full_rank = Matrix2x3d::new(100.0, 0.0, 0.0, 0.0, 100.0, 0.0);
    let j_ray = cam_ray_from_img_jacobian(cam_ray, full_rank).expect("full rank has a J_ray");
    assert!((full_rank * j_ray - Matrix2d::identity()).norm() <= 1e-12);
}

#[path = "jacobian_tests_rust_only.rs"]
mod rust_only;
