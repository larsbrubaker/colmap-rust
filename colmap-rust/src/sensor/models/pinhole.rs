//! The perspective pinhole camera models of `colmap/sensor/models.h` that project as
//! `x = X / Z` and then apply a polynomial distortion of the normalized plane:
//! SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV and FULL_OPENCV. Their
//! unprojection inverts the distortion with [`iterative_undistortion`]. The remaining
//! pinhole-kind models (FOV, the division models, EUCM) are in `other.rs`; the trait they
//! implement is in `mod.rs`.

use super::{
    has_projectable_depth, iterative_undistortion, CameraModel, CameraModelId, CameraModelKind,
    DistortedCameraModel, Scalar,
};

/// `ImgFromCam` of the distorted pinhole models: `uu = u / w`, add the distortion, then
/// `x = f1 * x + c1` (a single-focal model has `f1 == f2 == f`).
fn distorted_img_from_cam<M: DistortedCameraModel, T: Scalar>(
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

    let f1 = params[M::FOCAL_LENGTH_IDXS[0]];
    let f2 = params[M::FOCAL_LENGTH_IDXS[M::FOCAL_LENGTH_IDXS.len() - 1]];
    let c1 = params[M::PRINCIPAL_POINT_IDXS[0]];
    let c2 = params[M::PRINCIPAL_POINT_IDXS[1]];

    let uu = u / w;
    let vv = v / w;

    // Distortion
    let mut du = T::from_f64(0.0);
    let mut dv = T::from_f64(0.0);
    M::distortion(&params[M::EXTRA_PARAMS_IDXS[0]..], uu, vv, &mut du, &mut dv);
    *x = uu + du;
    *y = vv + dv;

    // Transform to image coordinates
    *x = f1 * *x + c1;
    *y = f2 * *y + c2;

    true
}

/// `CamFromImg` of the distorted pinhole models: lift to the normalized plane, then invert
/// the distortion iteratively.
fn distorted_cam_from_img<M: DistortedCameraModel>(
    params: &[f64],
    x: f64,
    y: f64,
    u: &mut f64,
    v: &mut f64,
) -> bool {
    let f1 = params[M::FOCAL_LENGTH_IDXS[0]];
    let f2 = params[M::FOCAL_LENGTH_IDXS[M::FOCAL_LENGTH_IDXS.len() - 1]];
    let c1 = params[M::PRINCIPAL_POINT_IDXS[0]];
    let c2 = params[M::PRINCIPAL_POINT_IDXS[1]];

    // Lift points to normalized plane
    *u = (x - c1) / f1;
    *v = (y - c2) / f2;

    iterative_undistortion::<M>(&params[M::EXTRA_PARAMS_IDXS[0]..], u, v)
}

/// SIMPLE_PINHOLE: no distortion, one focal length. Parameters `f, cx, cy`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimplePinholeCameraModel;

impl CameraModel for SimplePinholeCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::SimplePinhole;
    const MODEL_NAME: &'static str = "SIMPLE_PINHOLE";
    const PARAMS_INFO: &'static str = "f, cx, cy";
    const NUM_PARAMS: usize = 3;
    const KIND: CameraModelKind = CameraModelKind::PerspectivePinhole;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[0];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[1, 2];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[];
    const METADATA_IDXS: &'static [usize] = &[];

    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        vec![focal_length, width as f64 / 2.0, height as f64 / 2.0]
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
        if !has_projectable_depth(w, check_cheirality) {
            return false;
        }
        let f = params[0];
        let c1 = params[1];
        let c2 = params[2];

        // No Distortion

        // Transform to image coordinates
        *x = f * u / w + c1;
        *y = f * v / w + c2;
        true
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        let f = params[0];
        let c1 = params[1];
        let c2 = params[2];
        *u = (x - c1) / f;
        *v = (y - c2) / f;
        true
    }
}

/// PINHOLE: no distortion, two focal lengths. Parameters `fx, fy, cx, cy`.
#[derive(Debug, Clone, Copy, Default)]
pub struct PinholeCameraModel;

impl CameraModel for PinholeCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::Pinhole;
    const MODEL_NAME: &'static str = "PINHOLE";
    const PARAMS_INFO: &'static str = "fx, fy, cx, cy";
    const NUM_PARAMS: usize = 4;
    const KIND: CameraModelKind = CameraModelKind::PerspectivePinhole;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[0, 1];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[2, 3];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[];
    const METADATA_IDXS: &'static [usize] = &[];

    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        vec![
            focal_length,
            focal_length,
            width as f64 / 2.0,
            height as f64 / 2.0,
        ]
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
        if !has_projectable_depth(w, check_cheirality) {
            return false;
        }
        let f1 = params[0];
        let f2 = params[1];
        let c1 = params[2];
        let c2 = params[3];

        // No Distortion

        // Transform to image coordinates
        *x = f1 * u / w + c1;
        *y = f2 * v / w + c2;
        true
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        let f1 = params[0];
        let f2 = params[1];
        let c1 = params[2];
        let c2 = params[3];
        *u = (x - c1) / f1;
        *v = (y - c2) / f2;
        true
    }
}

/// SIMPLE_RADIAL: one focal length and one radial distortion parameter (like VisualSfM's
/// model, but the distortion is applied to the projections, not the measurements).
/// Parameters `f, cx, cy, k`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleRadialCameraModel;

impl CameraModel for SimpleRadialCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::SimpleRadial;
    const MODEL_NAME: &'static str = "SIMPLE_RADIAL";
    const PARAMS_INFO: &'static str = "f, cx, cy, k";
    const NUM_PARAMS: usize = 4;
    const KIND: CameraModelKind = CameraModelKind::PerspectivePinhole;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[0];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[1, 2];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[3];
    const METADATA_IDXS: &'static [usize] = &[];

    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        vec![focal_length, width as f64 / 2.0, height as f64 / 2.0, 0.0]
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
        distorted_img_from_cam::<Self, T>(params, u, v, w, x, y, check_cheirality)
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        distorted_cam_from_img::<Self>(params, x, y, u, v)
    }
}

impl DistortedCameraModel for SimpleRadialCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let k = extra_params[0];

        let u2 = u * u;
        let v2 = v * v;
        let r2 = u2 + v2;
        let radial = k * r2;
        *du = u * radial;
        *dv = v * radial;
    }
}

/// RADIAL: one focal length and two radial distortion parameters (Bundler's model, except
/// for an inverted z axis). Parameters `f, cx, cy, k1, k2`.
#[derive(Debug, Clone, Copy, Default)]
pub struct RadialCameraModel;

impl CameraModel for RadialCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::Radial;
    const MODEL_NAME: &'static str = "RADIAL";
    const PARAMS_INFO: &'static str = "f, cx, cy, k1, k2";
    const NUM_PARAMS: usize = 5;
    const KIND: CameraModelKind = CameraModelKind::PerspectivePinhole;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[0];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[1, 2];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[3, 4];
    const METADATA_IDXS: &'static [usize] = &[];

    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        vec![
            focal_length,
            width as f64 / 2.0,
            height as f64 / 2.0,
            0.0,
            0.0,
        ]
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
        distorted_img_from_cam::<Self, T>(params, u, v, w, x, y, check_cheirality)
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        distorted_cam_from_img::<Self>(params, x, y, u, v)
    }
}

impl DistortedCameraModel for RadialCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let k1 = extra_params[0];
        let k2 = extra_params[1];

        let u2 = u * u;
        let v2 = v * v;
        let r2 = u2 + v2;
        let radial = k1 * r2 + k2 * r2 * r2;
        *du = u * radial;
        *dv = v * radial;
    }
}

/// OPENCV: radial and tangential distortion up to second degree; not suited to the large
/// radial distortion of fisheye lenses. Parameters `fx, fy, cx, cy, k1, k2, p1, p2`.
#[derive(Debug, Clone, Copy, Default)]
pub struct OpenCVCameraModel;

impl CameraModel for OpenCVCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::OpenCV;
    const MODEL_NAME: &'static str = "OPENCV";
    const PARAMS_INFO: &'static str = "fx, fy, cx, cy, k1, k2, p1, p2";
    const NUM_PARAMS: usize = 8;
    const KIND: CameraModelKind = CameraModelKind::PerspectivePinhole;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[0, 1];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[2, 3];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[4, 5, 6, 7];
    const METADATA_IDXS: &'static [usize] = &[];

    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        vec![
            focal_length,
            focal_length,
            width as f64 / 2.0,
            height as f64 / 2.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ]
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
        distorted_img_from_cam::<Self, T>(params, u, v, w, x, y, check_cheirality)
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        distorted_cam_from_img::<Self>(params, x, y, u, v)
    }
}

impl DistortedCameraModel for OpenCVCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let k1 = extra_params[0];
        let k2 = extra_params[1];
        let p1 = extra_params[2];
        let p2 = extra_params[3];
        let two = T::from_f64(2.0);

        let u2 = u * u;
        let uv = u * v;
        let v2 = v * v;
        let r2 = u2 + v2;
        let radial = k1 * r2 + k2 * r2 * r2;
        *du = u * radial + two * p1 * uv + p2 * (r2 + two * u2);
        *dv = v * radial + two * p2 * uv + p1 * (r2 + two * v2);
    }
}

/// FULL_OPENCV: OpenCV's rational radial model plus tangential distortion. Parameters
/// `fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FullOpenCVCameraModel;

impl CameraModel for FullOpenCVCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::FullOpenCV;
    const MODEL_NAME: &'static str = "FULL_OPENCV";
    const PARAMS_INFO: &'static str = "fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6";
    const NUM_PARAMS: usize = 12;
    const KIND: CameraModelKind = CameraModelKind::PerspectivePinhole;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[0, 1];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[2, 3];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[4, 5, 6, 7, 8, 9, 10, 11];
    const METADATA_IDXS: &'static [usize] = &[];

    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        let mut params = vec![0.0; Self::NUM_PARAMS];
        params[0] = focal_length;
        params[1] = focal_length;
        params[2] = width as f64 / 2.0;
        params[3] = height as f64 / 2.0;
        params
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
        distorted_img_from_cam::<Self, T>(params, u, v, w, x, y, check_cheirality)
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        distorted_cam_from_img::<Self>(params, x, y, u, v)
    }
}

impl DistortedCameraModel for FullOpenCVCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let k1 = extra_params[0];
        let k2 = extra_params[1];
        let p1 = extra_params[2];
        let p2 = extra_params[3];
        let k3 = extra_params[4];
        let k4 = extra_params[5];
        let k5 = extra_params[6];
        let k6 = extra_params[7];
        let one = T::from_f64(1.0);
        let two = T::from_f64(2.0);

        let u2 = u * u;
        let uv = u * v;
        let v2 = v * v;
        let r2 = u2 + v2;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = (one + k1 * r2 + k2 * r4 + k3 * r6) / (one + k4 * r2 + k5 * r4 + k6 * r6);
        *du = u * radial + two * p1 * uv + p2 * (r2 + two * u2) - u;
        *dv = v * radial + two * p2 * uv + p1 * (r2 + two * v2) - v;
    }
}
