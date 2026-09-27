//! The camera models of `colmap/sensor/models.h` with closed-form or non-polynomial
//! projections: FOV, SIMPLE_DIVISION, DIVISION and EUCM (perspective pinhole models) and
//! EQUIRECTANGULAR (the one spherical model). The polynomial pinhole models are in
//! `pinhole.rs`, the fisheye models in `fisheye.rs`, the trait in `mod.rs`.

use std::f64::consts::PI;

use crate::math::fns;

use super::{
    has_projectable_depth, CameraModel, CameraModelId, CameraModelKind, DistortedCameraModel,
    Scalar,
};

/// FOV: radial distortion by the field-of-view parameter omega (Devernay and Faugeras,
/// "Straight lines have to be straight", 2001; Project Tango's equidistant calibration).
/// Parameters `fx, fy, cx, cy, omega`.
#[derive(Debug, Clone, Copy, Default)]
pub struct FOVCameraModel;

impl FOVCameraModel {
    /// `FOVCameraModel::Undistortion`: the closed-form inverse of `Distortion`.
    pub fn undistortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let omega = extra_params[0];

        // Chosen arbitrarily.
        let epsilon = T::from_f64(1e-4);

        let radius2 = u * u + v * v;
        let omega2 = omega * omega;

        let factor = if omega2 < epsilon {
            // Derivation of this case with Matlab:
            // syms radius omega;
            // factor(radius) = tan(radius * omega) / ...
            //                  (radius * 2*tan(omega/2));
            // simplify(taylor(factor, omega, 'order', 3))
            (omega2 * radius2) / T::from_f64(3.0) - omega2 / T::from_f64(12.0) + T::from_f64(1.0)
        } else if radius2 < epsilon {
            // Derivation of this case with Matlab:
            // syms radius omega;
            // factor(radius) = tan(radius * omega) / ...
            //                  (radius * 2*tan(omega/2));
            // simplify(taylor(factor, radius, 'order', 3))
            (omega * (omega * omega * radius2 + T::from_f64(3.0)))
                / (T::from_f64(6.0) * (omega / T::from_f64(2.0)).tan())
        } else {
            let radius = radius2.sqrt();
            let numerator = (radius * omega).tan();
            numerator / (radius * T::from_f64(2.0) * (omega / T::from_f64(2.0)).tan())
        };

        *du = u * factor;
        *dv = v * factor;
    }
}

impl CameraModel for FOVCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::FOV;
    const MODEL_NAME: &'static str = "FOV";
    const PARAMS_INFO: &'static str = "fx, fy, cx, cy, omega";
    const NUM_PARAMS: usize = 5;
    const KIND: CameraModelKind = CameraModelKind::PerspectivePinhole;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[0, 1];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[2, 3];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[4];
    const METADATA_IDXS: &'static [usize] = &[];

    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        vec![
            focal_length,
            focal_length,
            width as f64 / 2.0,
            height as f64 / 2.0,
            1e-2,
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

        // Distortion
        Self::distortion(&params[4..], u / w, v / w, x, y);

        // Transform to image coordinates
        *x = f1 * *x + c1;
        *y = f2 * *y + c2;
        true
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        let f1 = params[0];
        let f2 = params[1];
        let c1 = params[2];
        let c2 = params[3];

        // Lift points to normalized plane
        let uu = (x - c1) / f1;
        let vv = (y - c2) / f2;

        // Undistortion
        Self::undistortion(&params[4..], uu, vv, u, v);
        true
    }
}

impl DistortedCameraModel for FOVCameraModel {
    /// Unlike the other models' `Distortion`, FOV's returns the distorted point itself,
    /// `x * factor`, which `ImgFromCam` uses directly.
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        let omega = extra_params[0];

        // Chosen arbitrarily.
        let epsilon = T::from_f64(1e-4);

        let radius2 = u * u + v * v;
        let omega2 = omega * omega;

        let factor = if omega2 < epsilon {
            // Derivation of this case with Matlab:
            // syms radius omega;
            // factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
            //                  (radius * omega);
            // simplify(taylor(factor, omega, 'order', 3))
            (omega2 * radius2) / T::from_f64(3.0) - omega2 / T::from_f64(12.0) + T::from_f64(1.0)
        } else if radius2 < epsilon {
            // Derivation of this case with Matlab:
            // syms radius omega;
            // factor(radius) = atan(radius * 2 * tan(omega / 2)) / ...
            //                  (radius * omega);
            // simplify(taylor(factor, radius, 'order', 3))
            let tan_half_omega = (omega / T::from_f64(2.0)).tan();
            (T::from_f64(-2.0)
                * tan_half_omega
                * (T::from_f64(4.0) * radius2 * tan_half_omega * tan_half_omega - T::from_f64(3.0)))
                / (T::from_f64(3.0) * omega)
        } else {
            let radius = radius2.sqrt();
            let numerator = (radius * T::from_f64(2.0) * (omega / T::from_f64(2.0)).tan()).atan();
            numerator / (radius * omega)
        };

        *du = u * factor;
        *dv = v * factor;
    }
}

/// `ImgFromCam` of the division models: solves `rho k r^2 - w r + rho = 0` for the radius.
/// `check_cheirality` is ignored, as in COLMAP.
fn division_img_from_cam<M: CameraModel, T: Scalar>(
    params: &[T],
    u: T,
    v: T,
    w: T,
    x: &mut T,
    y: &mut T,
) -> bool {
    // Division model projection:
    // (xp, 1+k*|xp|^2) ~= (x(1:2), x3)
    // Solving the quadratic: rho*k*r2 - x3 * r + rho = 0
    let f1 = params[M::FOCAL_LENGTH_IDXS[0]];
    let f2 = params[M::FOCAL_LENGTH_IDXS[M::FOCAL_LENGTH_IDXS.len() - 1]];
    let c1 = params[M::PRINCIPAL_POINT_IDXS[0]];
    let c2 = params[M::PRINCIPAL_POINT_IDXS[1]];
    let k = params[M::EXTRA_PARAMS_IDXS[0]];

    let rho = (u * u + v * v).sqrt();
    let disc_sq = w * w - T::from_f64(4.0) * rho * rho * k;

    if disc_sq < T::from_f64(0.0) {
        return false;
    }

    let disc = disc_sq.sqrt();
    let r = T::from_f64(2.0) / (w + disc);

    *x = f1 * r * u + c1;
    *y = f2 * r * v + c2;
    true
}

/// `CamFromImg` of the division models (closed form).
fn division_cam_from_img<M: CameraModel>(
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
    let k = params[M::EXTRA_PARAMS_IDXS[0]];

    // Lift to normalized coordinates
    let x0 = (x - c1) / f1;
    let y0 = (y - c2) / f2;
    let r2 = x0 * x0 + y0 * y0;

    // Closed-form unprojection for division model
    let denom = 1.0 + k * r2;
    *u = x0 / denom;
    *v = y0 / denom;
    true
}

/// `Distortion` of the division models. The division model doesn't use standard additive
/// distortion, but COLMAP defines this for compatibility with the iterative undistortion.
fn division_distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
    let k = extra_params[0];
    let r2 = u * u + v * v;
    let factor = k * r2 / (T::from_f64(1.0) + k * r2);
    *du = -u * factor;
    *dv = -v * factor;
}

/// SIMPLE_DIVISION: Fitzgibbon's one-parameter division model ("Simultaneous linear
/// estimation of multiple view geometry and lens distortion", 2001) with one focal length.
/// Parameters `f, cx, cy, k`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleDivisionCameraModel;

impl CameraModel for SimpleDivisionCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::SimpleDivision;
    const MODEL_NAME: &'static str = "SIMPLE_DIVISION";
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
        _check_cheirality: bool,
    ) -> bool {
        division_img_from_cam::<Self, T>(params, u, v, w, x, y)
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        division_cam_from_img::<Self>(params, x, y, u, v)
    }
}

impl DistortedCameraModel for SimpleDivisionCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        division_distortion(extra_params, u, v, du, dv);
    }
}

/// DIVISION: Fitzgibbon's division model with separate `fx`/`fy`. Parameters
/// `fx, fy, cx, cy, k`.
#[derive(Debug, Clone, Copy, Default)]
pub struct DivisionCameraModel;

impl CameraModel for DivisionCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::Division;
    const MODEL_NAME: &'static str = "DIVISION";
    const PARAMS_INFO: &'static str = "fx, fy, cx, cy, k";
    const NUM_PARAMS: usize = 5;
    const KIND: CameraModelKind = CameraModelKind::PerspectivePinhole;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[0, 1];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[2, 3];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[4];
    const METADATA_IDXS: &'static [usize] = &[];

    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        vec![
            focal_length,
            focal_length,
            width as f64 / 2.0,
            height as f64 / 2.0,
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
        _check_cheirality: bool,
    ) -> bool {
        division_img_from_cam::<Self, T>(params, u, v, w, x, y)
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        division_cam_from_img::<Self>(params, x, y, u, v)
    }
}

impl DistortedCameraModel for DivisionCameraModel {
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T) {
        division_distortion(extra_params, u, v, du, dv);
    }
}

/// EUCM: the Enhanced Unified Camera Model (Khomutenko, Garcia and Martinet, 2018).
/// Parameters `fx, fy, cx, cy, alpha, beta`.
#[derive(Debug, Clone, Copy, Default)]
pub struct EUCMCameraModel;

impl CameraModel for EUCMCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::EUCM;
    const MODEL_NAME: &'static str = "EUCM";
    const PARAMS_INFO: &'static str = "fx, fy, cx, cy, alpha, beta";
    const NUM_PARAMS: usize = 6;
    const KIND: CameraModelKind = CameraModelKind::PerspectivePinhole;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[0, 1];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[2, 3];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[4, 5];
    const METADATA_IDXS: &'static [usize] = &[];

    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        vec![
            focal_length,
            focal_length,
            width as f64 / 2.0,
            height as f64 / 2.0,
            0.0,
            1.0,
        ]
    }

    /// EUCM's override: the perspective base check, plus `alpha` in `[0, 1]` and
    /// `beta > 0`.
    fn has_bogus_extra_params(params: &[f64], max_extra_param: f64) -> bool {
        if Self::EXTRA_PARAMS_IDXS
            .iter()
            .any(|&idx| params[idx].abs() > max_extra_param)
        {
            return true;
        }

        let alpha = params[4];
        let beta = params[5];
        // Kept as COLMAP writes it: `!(0.0..=1.0).contains(&alpha)` would reject a NaN alpha.
        #[allow(clippy::manual_range_contains)]
        let bogus = alpha < 0.0 || alpha > 1.0 || beta <= 0.0;
        bogus
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

        let alpha = params[4];
        let beta = params[5];

        let rho2 = beta * (u * u + v * v) + w * w;
        if rho2 < T::from_f64(0.0) {
            return false;
        }
        let rho = rho2.sqrt();
        // C++ `(1.0 - alpha)`: Ceres' mixed double/Jet subtraction.
        let den = alpha * rho + alpha.sub_from_f64(1.0) * w;
        if !has_projectable_depth(den, check_cheirality) {
            return false;
        }
        *x = u / den;
        *y = v / den;

        // Transform to image coordinates
        *x = f1 * *x + c1;
        *y = f2 * *y + c2;
        true
    }

    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        let f1 = params[0];
        let f2 = params[1];
        let c1 = params[2];
        let c2 = params[3];

        let alpha = params[4];
        let beta = params[5];

        // Lift points to normalized plane
        *u = (x - c1) / f1;
        *v = (y - c2) / f2;

        let r2 = *u * *u + *v * *v;
        let gamma = 1.0 - alpha;
        let radicand = 1.0 - (alpha - gamma) * beta * r2;
        if radicand < 0.0 {
            return false;
        }
        let helper_den = alpha * fns::sqrt(radicand) + gamma;
        if helper_den < f64::EPSILON {
            return false;
        }
        let helper = (1.0 - alpha * alpha * beta * r2) / helper_den;
        if helper < f64::EPSILON {
            return false;
        }

        *u /= helper;
        *v /= helper;
        true
    }
}

/// EQUIRECTANGULAR: the full 360x180 degree sphere on an equirectangular panorama, azimuth
/// across the width and elevation down the height. The model is fully specified by the
/// image size: parameters `w, h` (metadata, not optimizable); no focal length, principal
/// point or distortion.
///
/// COLMAP writes `EIGEN_PI`, a `long double` literal; this uses `f64` pi, which equals it on
/// macOS (where `long double` is `double`) and may differ on x86-64 Linux
/// (docs/CPP_DIVERGENCES.md, entry 100).
#[derive(Debug, Clone, Copy, Default)]
pub struct EquirectangularCameraModel;

impl CameraModel for EquirectangularCameraModel {
    const MODEL_ID: CameraModelId = CameraModelId::Equirectangular;
    const MODEL_NAME: &'static str = "EQUIRECTANGULAR";
    const PARAMS_INFO: &'static str = "w,h";
    const NUM_PARAMS: usize = 2;
    const KIND: CameraModelKind = CameraModelKind::Spherical;
    const FOCAL_LENGTH_IDXS: &'static [usize] = &[];
    const PRINCIPAL_POINT_IDXS: &'static [usize] = &[];
    const EXTRA_PARAMS_IDXS: &'static [usize] = &[];
    const METADATA_IDXS: &'static [usize] = &[0, 1];

    /// Ignores the focal length and returns `(width, height)`.
    fn initialize_params(_focal_length: f64, width: usize, height: usize) -> Vec<f64> {
        vec![width as f64, height as f64]
    }

    fn has_bogus_params(
        _params: &[f64],
        _width: usize,
        _height: usize,
        _min_focal_length_ratio: f64,
        _max_focal_length_ratio: f64,
        _max_extra_param: f64,
    ) -> bool {
        false
    }

    /// No focal length, so pixel thresholds convert with the angular resolution at the
    /// equator (2 pi rad per W pixels in azimuth).
    fn cam_from_img_threshold(params: &[f64], threshold: f64) -> f64 {
        threshold * (2.0 * PI) / params[0]
    }

    /// Projects any non-zero direction, including the back hemisphere (all 4 pi of the
    /// sphere are representable); `check_cheirality` is ignored.
    fn img_from_cam<T: Scalar>(
        params: &[T],
        u: T,
        v: T,
        w: T,
        x: &mut T,
        y: &mut T,
        _check_cheirality: bool,
    ) -> bool {
        let width = params[0];
        let height = params[1];

        let horizontal = (u * u + w * w).sqrt();
        // Degenerate: zero direction vector.
        if horizontal + v.abs() < T::from_f64(f64::EPSILON) {
            return false;
        }

        // Azimuth theta in (-pi, pi], measured from the +Z axis (forward). +X is theta = +pi/2.
        let theta = u.atan2(w);
        // Elevation phi in [-pi/2, pi/2], measured from the equator. -Y (up) is +pi/2.
        let phi = (-v).atan2(horizontal);

        *x = (theta / T::from_f64(2.0 * PI) + T::from_f64(0.5)) * width;
        *y = (T::from_f64(0.5) - phi / T::from_f64(PI)) * height;
        true
    }

    /// The normalized coordinates `(X / Z, Y / Z)` of the pixel's ray, valid only in the
    /// forward hemisphere (`Z > 0`); use `cam_ray_from_img` for the full sphere.
    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool {
        let width = params[0];
        let height = params[1];

        let theta = 2.0 * PI * (x / width - 0.5);
        let phi = PI * (0.5 - y / height);

        let cos_phi = fns::cos(phi);
        let rx = cos_phi * fns::sin(theta);
        let ry = -fns::sin(phi);
        let rz = cos_phi * fns::cos(theta);

        if rz <= f64::EPSILON {
            return false;
        }

        *u = rx / rz;
        *v = ry / rz;
        true
    }

    /// The bearing straight from the azimuth/elevation parametrization, valid for every
    /// pixel (the perspective default goes through `cam_from_img`, which fails for the back
    /// hemisphere).
    fn cam_ray_from_img(
        params: &[f64],
        x: f64,
        y: f64,
        rx: &mut f64,
        ry: &mut f64,
        rz: &mut f64,
    ) -> bool {
        let width = params[0];
        let height = params[1];
        let theta = 2.0 * PI * (x / width - 0.5);
        let phi = PI * (0.5 - y / height);
        let cos_phi = fns::cos(phi);
        *rx = cos_phi * fns::sin(theta);
        *ry = -fns::sin(phi);
        *rz = cos_phi * fns::cos(theta);
        true
    }
}
