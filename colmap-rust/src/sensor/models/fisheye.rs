//! The perspective fisheye camera models of `colmap/sensor/models.h` (the
//! `PERSPECTIVE_FISHEYE_CAMERA_MODEL_CASES`): OPENCV_FISHEYE, SIMPLE_RADIAL_FISHEYE,
//! RADIAL_FISHEYE, THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE, SIMPLE_FISHEYE and
//! FISHEYE. Each maps the normalized plane to the equidistant fisheye plane
//! ([`PerspectiveFisheyeModel::fisheye_from_normal`], `theta = atan(r)`), distorts it, and
//! scales it to pixels; unprojection inverts the distortion with [`iterative_undistortion`]
//! and maps back with `normal_from_fisheye`. The traits are in `mod.rs`.

use super::{
    has_projectable_depth, iterative_undistortion, CameraModel, CameraModelId, CameraModelKind,
    DistortedCameraModel, PerspectiveFisheyeModel, Scalar,
};

/// `ImgFromFisheye` of every fisheye model: `x = f1 * uu + c1` (single-focal: `f1 == f2`).
fn img_from_fisheye_by_idxs<M: CameraModel, T: Scalar>(
    params: &[T],
    uu: T,
    vv: T,
    x: &mut T,
    y: &mut T,
) {
    let f1 = params[M::FOCAL_LENGTH_IDXS[0]];
    let f2 = params[M::FOCAL_LENGTH_IDXS[M::FOCAL_LENGTH_IDXS.len() - 1]];
    let c1 = params[M::PRINCIPAL_POINT_IDXS[0]];
    let c2 = params[M::PRINCIPAL_POINT_IDXS[1]];
    *x = f1 * uu + c1;
    *y = f2 * vv + c2;
}

/// `FisheyeFromImg` of every fisheye model: `uu = (x - c1) / f1`.
fn fisheye_from_img_by_idxs<M: CameraModel, T: Scalar>(
    params: &[T],
    x: T,
    y: T,
    uu: &mut T,
    vv: &mut T,
) {
    let f1 = params[M::FOCAL_LENGTH_IDXS[0]];
    let f2 = params[M::FOCAL_LENGTH_IDXS[M::FOCAL_LENGTH_IDXS.len() - 1]];
    let c1 = params[M::PRINCIPAL_POINT_IDXS[0]];
    let c2 = params[M::PRINCIPAL_POINT_IDXS[1]];
    *uu = (x - c1) / f1;
    *vv = (y - c2) / f2;
}

/// `ImgFromCam` of the distorted fisheye models.
fn distorted_img_from_cam<M: DistortedCameraModel + PerspectiveFisheyeModel, T: Scalar>(
    params: &[T],
    u: T,
    v: T,
    w: T,
    x: &mut T,
    y: &mut T,
    check_cheirality: bool,
) -> bool {
    if !has_projectable_depth(w, check_cheirality) {
        return false;
    }

    let mut uu = T::from_f64(0.0);
    let mut vv = T::from_f64(0.0);
    M::fisheye_from_normal(u / w, v / w, &mut uu, &mut vv);

    // Distortion
    let mut duu = T::from_f64(0.0);
    let mut dvv = T::from_f64(0.0);
    M::distortion(
        &params[M::EXTRA_PARAMS_IDXS[0]..],
        uu,
        vv,
        &mut duu,
        &mut dvv,
    );

    // Transform to image coordinates
    M::img_from_fisheye(params, uu + duu, vv + dvv, x, y);
    true
}

/// `CamFromImg` of the distorted fisheye models. On a failed undistortion the outputs are
/// left untouched, as in COLMAP.
fn distorted_cam_from_img<M: DistortedCameraModel + PerspectiveFisheyeModel>(
    params: &[f64],
    x: f64,
    y: f64,
    u: &mut f64,
    v: &mut f64,
) -> bool {
    let mut uu = 0.0;
    let mut vv = 0.0;
    M::fisheye_from_img(params, x, y, &mut uu, &mut vv);
    if !iterative_undistortion::<M>(&params[M::EXTRA_PARAMS_IDXS[0]..], &mut uu, &mut vv) {
        return false;
    }
    M::normal_from_fisheye(uu, vv, u, v);
    true
}

/// `ImgFromCam` of the undistorted fisheye models (SIMPLE_FISHEYE, FISHEYE).
fn plain_img_from_cam<M: PerspectiveFisheyeModel, T: Scalar>(
    params: &[T],
    u: T,
    v: T,
    w: T,
    x: &mut T,
    y: &mut T,
    check_cheirality: bool,
) -> bool {
    if !has_projectable_depth(w, check_cheirality) {
        return false;
    }

    let mut uu = T::from_f64(0.0);
    let mut vv = T::from_f64(0.0);
    M::fisheye_from_normal(u / w, v / w, &mut uu, &mut vv);

    // No distortion

    // Transform to image coordinates
    M::img_from_fisheye(params, uu, vv, x, y);
    true
}

/// `CamFromImg` of the undistorted fisheye models.
fn plain_cam_from_img<M: PerspectiveFisheyeModel>(
    params: &[f64],
    x: f64,
    y: f64,
    u: &mut f64,
    v: &mut f64,
) -> bool {
    let mut uu = 0.0;
    let mut vv = 0.0;
    M::fisheye_from_img(params, x, y, &mut uu, &mut vv);
    // No undistortion needed
    M::normal_from_fisheye(uu, vv, u, v);
    true
}

/// Implements [`CameraModel`] and [`PerspectiveFisheyeModel`] for a fisheye model; the
/// projection bodies are the shared functions above, so only the constants differ.
macro_rules! fisheye_model {
    (
        $model:ident, $id:ident, $name:literal, $info:literal, $num:literal,
        focal: $focal:expr, pp: $pp:expr, extra: $extra:expr,
        init: $init:expr, distorted: $distorted:tt
    ) => {
        impl CameraModel for $model {
            const MODEL_ID: CameraModelId = CameraModelId::$id;
            const MODEL_NAME: &'static str = $name;
            const PARAMS_INFO: &'static str = $info;
            const NUM_PARAMS: usize = $num;
            const KIND: CameraModelKind = CameraModelKind::PerspectiveFisheye;
            const FOCAL_LENGTH_IDXS: &'static [usize] = $focal;
            const PRINCIPAL_POINT_IDXS: &'static [usize] = $pp;
            const EXTRA_PARAMS_IDXS: &'static [usize] = $extra;
            const METADATA_IDXS: &'static [usize] = &[];

            fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
                let init: fn(f64, f64, f64) -> Vec<f64> = $init;
                init(focal_length, width as f64 / 2.0, height as f64 / 2.0)
            }

            fn img_from_cam<T: Scalar>(
                params: &[T],
                u: T,
                v: T,
                w: T,
                x: &mut T,
                y: &mut T,
                check_cheirality: bool,
            ) -> bool {
                fisheye_model!(@img $distorted, params, u, v, w, x, y, check_cheirality)
            }

            fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
                fisheye_model!(@cam $distorted, params, x, y, u, v)
            }
        }

        impl PerspectiveFisheyeModel for $model {
            fn img_from_fisheye<T: Scalar>(params: &[T], uu: T, vv: T, x: &mut T, y: &mut T) {
                img_from_fisheye_by_idxs::<Self, T>(params, uu, vv, x, y)
            }

            fn fisheye_from_img<T: Scalar>(params: &[T], x: T, y: T, uu: &mut T, vv: &mut T) {
                fisheye_from_img_by_idxs::<Self, T>(params, x, y, uu, vv)
            }
        }
    };
    (@img true, $($a:expr),*) => { distorted_img_from_cam::<Self, T>($($a),*) };
    (@img false, $($a:expr),*) => { plain_img_from_cam::<Self, T>($($a),*) };
    (@cam true, $($a:expr),*) => { distorted_cam_from_img::<Self>($($a),*) };
    (@cam false, $($a:expr),*) => { plain_cam_from_img::<Self>($($a),*) };
}

/// OPENCV_FISHEYE: OpenCV's fisheye model, radial distortion of the equidistant plane up to
/// fourth degree. Parameters `fx, fy, cx, cy, k1, k2, k3, k4`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCVFisheyeCameraModel;

fisheye_model!(
    OpenCVFisheyeCameraModel, OpenCVFisheye, "OPENCV_FISHEYE", "fx, fy, cx, cy, k1, k2, k3, k4", 8,
    focal: &[0, 1], pp: &[2, 3], extra: &[4, 5, 6, 7],
    init: |f, cx, cy| vec![f, f, cx, cy, 0.0, 0.0, 0.0, 0.0], distorted: true
);

impl DistortedCameraModel for OpenCVFisheyeCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let k1 = extra_params[0];
        let k2 = extra_params[1];
        let k3 = extra_params[2];
        let k4 = extra_params[3];

        let theta2 = u * u + v * v;
        let theta4 = theta2 * theta2;
        let theta6 = theta4 * theta2;
        let theta8 = theta4 * theta4;
        let radial = k1 * theta2 + k2 * theta4 + k3 * theta6 + k4 * theta8;
        *du = u * radial;
        *dv = v * radial;
    }
}

/// SIMPLE_RADIAL_FISHEYE: OPENCV_FISHEYE with one focal length and one radial coefficient.
/// Parameters `f, cx, cy, k`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleRadialFisheyeCameraModel;

fisheye_model!(
    SimpleRadialFisheyeCameraModel, SimpleRadialFisheye, "SIMPLE_RADIAL_FISHEYE", "f, cx, cy, k", 4,
    focal: &[0], pp: &[1, 2], extra: &[3],
    init: |f, cx, cy| vec![f, cx, cy, 0.0], distorted: true
);

impl DistortedCameraModel for SimpleRadialFisheyeCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let k = extra_params[0];

        let theta2 = u * u + v * v;
        let radial = k * theta2;
        *du = u * radial;
        *dv = v * radial;
    }
}

/// RADIAL_FISHEYE: OPENCV_FISHEYE with one focal length and two radial coefficients.
/// Parameters `f, cx, cy, k1, k2`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RadialFisheyeCameraModel;

fisheye_model!(
    RadialFisheyeCameraModel, RadialFisheye, "RADIAL_FISHEYE", "f, cx, cy, k1, k2", 5,
    focal: &[0], pp: &[1, 2], extra: &[3, 4],
    init: |f, cx, cy| vec![f, cx, cy, 0.0, 0.0], distorted: true
);

impl DistortedCameraModel for RadialFisheyeCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let k1 = extra_params[0];
        let k2 = extra_params[1];

        let theta2 = u * u + v * v;
        let theta4 = theta2 * theta2;
        let radial = k1 * theta2 + k2 * theta4;
        *du = u * radial;
        *dv = v * radial;
    }
}

/// THIN_PRISM_FISHEYE: radial, tangential and thin-prism distortion (Weng et al., "Camera
/// Calibration with Distortion Models and Accuracy Evaluation", TPAMI 1992). Parameters
/// `fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThinPrismFisheyeCameraModel;

fisheye_model!(
    ThinPrismFisheyeCameraModel, ThinPrismFisheye, "THIN_PRISM_FISHEYE",
    "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1", 12,
    focal: &[0, 1], pp: &[2, 3], extra: &[4, 5, 6, 7, 8, 9, 10, 11],
    init: |f, cx, cy| vec![f, f, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    distorted: true
);

impl DistortedCameraModel for ThinPrismFisheyeCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let k1 = extra_params[0];
        let k2 = extra_params[1];
        let p1 = extra_params[2];
        let p2 = extra_params[3];
        let k3 = extra_params[4];
        let k4 = extra_params[5];
        let sx1 = extra_params[6];
        let sy1 = extra_params[7];
        let two = T::from_f64(2.0);

        let u2 = u * u;
        let uv = u * v;
        let v2 = v * v;
        let r2 = u2 + v2;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let r8 = r6 * r2;
        let radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        *du = u * radial + two * p1 * uv + p2 * (r2 + two * u2) + sx1 * r2;
        *dv = v * radial + two * p2 * uv + p1 * (r2 + two * v2) + sy1 * r2;
    }
}

/// RAD_TAN_THIN_PRISM_FISHEYE: Project Aria's Fisheye624 model, six radial, two tangential
/// and four thin-prism coefficients. Parameters
/// `fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RadTanThinPrismFisheyeModel;

fisheye_model!(
    RadTanThinPrismFisheyeModel, RadTanThinPrismFisheye, "RAD_TAN_THIN_PRISM_FISHEYE",
    "fx, fy, cx, cy, k0, k1, k2, k3, k4, k5, p0, p1, s0, s1, s2, s3", 16,
    focal: &[0, 1], pp: &[2, 3], extra: &[4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    init: |f, cx, cy| {
        let mut params = vec![0.0; 16];
        params[0] = f;
        params[1] = f;
        params[2] = cx;
        params[3] = cy;
        params
    },
    distorted: true
);

impl DistortedCameraModel for RadTanThinPrismFisheyeModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        const NUM_RADIAL_PARAMS: usize = 6;
        let radial_coeffs = &extra_params[..NUM_RADIAL_PARAMS];

        let p0 = extra_params[6];
        let p1 = extra_params[7];
        let s0 = extra_params[8];
        let s1 = extra_params[9];
        let s2 = extra_params[10];
        let s3 = extra_params[11];
        let two = T::from_f64(2.0);

        let theta2 = u * u + v * v;
        let mut th_radial = T::from_f64(1.0);
        let mut theta_power = T::from_f64(1.0);
        for &coeff in radial_coeffs {
            theta_power = theta_power * theta2;
            th_radial = th_radial + coeff * theta_power;
        }

        let x = th_radial * u;
        let y = th_radial * v;

        let x2 = x * x;
        let y2 = y * y;
        let xy = x * y;
        let r2 = x2 + y2;
        let r4 = r2 * r2;

        let dx_tang = two * p1 * xy + p0 * (r2 + two * x2);
        let dy_tang = two * p0 * xy + p1 * (r2 + two * y2);

        let dx_tp = s0 * r2 + s1 * r4;
        let dy_tp = s2 * r2 + s3 * r4;

        let x_distorted = x + dx_tang + dx_tp;
        let y_distorted = y + dy_tang + dy_tp;

        *du = x_distorted - u;
        *dv = y_distorted - v;
    }
}

/// SIMPLE_FISHEYE: the undistorted equidistant projection (`theta = r`) with one focal
/// length. Parameters `f, cx, cy`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleFisheyeCameraModel;

fisheye_model!(
    SimpleFisheyeCameraModel, SimpleFisheye, "SIMPLE_FISHEYE", "f, cx, cy", 3,
    focal: &[0], pp: &[1, 2], extra: &[],
    init: |f, cx, cy| vec![f, cx, cy], distorted: false
);

/// FISHEYE: the undistorted equidistant projection with two focal lengths. Parameters
/// `fx, fy, cx, cy`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FisheyeCameraModel;

fisheye_model!(
    FisheyeCameraModel, Fisheye, "FISHEYE", "fx, fy, cx, cy", 4,
    focal: &[0, 1], pp: &[2, 3], extra: &[],
    init: |f, cx, cy| vec![f, f, cx, cy], distorted: false
);
