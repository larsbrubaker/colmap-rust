//! `ImgFromCamWithJac` of the closed-form models (`other.rs`): FOV, SIMPLE_DIVISION, DIVISION,
//! EUCM and EQUIRECTANGULAR. Port of their kernels in `colmap/sensor/models_jacobian.h`; the
//! trait and shared helpers are in `jacobian.rs`.

use std::f64::consts::PI;

use crate::math::fns;

use super::jacobian::{division_scale_with_jac, uvw_jac_from_ab_jac, CameraModelWithJac};
use super::{
    has_projectable_depth, DivisionCameraModel, EUCMCameraModel, EquirectangularCameraModel,
    FOVCameraModel, SimpleDivisionCameraModel,
};

impl CameraModelWithJac for FOVCameraModel {
    fn img_from_cam_with_jac(
        params: &[f64],
        u: f64,
        v: f64,
        w: f64,
        x: &mut f64,
        y: &mut f64,
        j_params: Option<&mut [f64]>,
        j_uvw: Option<&mut [f64; 6]>,
        check_cheirality: bool,
    ) -> bool {
        if !has_projectable_depth(w, check_cheirality) {
            return false;
        }

        let f1 = params[0];
        let f2 = params[1];
        let c1 = params[2];
        let c2 = params[3];
        let omega = params[4];

        let inv_w = 1.0 / w;
        let a = u * inv_w;
        let b = v * inv_w;

        let radius2 = a * a + b * b;
        let omega2 = omega * omega;

        // Chosen to match FOVCameraModel::Distortion.
        const EPSILON: f64 = 1e-4;

        // The distortion scales (a, b) by a radially symmetric factor. We compute the
        // factor and its partials factor_r = d(factor)/d(radius2) and factor_omega =
        // d(factor)/d(omega), matching whichever branch FOVCameraModel::Distortion
        // selects so that the analytic Jacobian agrees with autodiff everywhere.
        let factor;
        let factor_r;
        let factor_omega;
        if omega2 < EPSILON {
            factor = (omega2 * radius2) / 3.0 - omega2 / 12.0 + 1.0;
            factor_r = omega2 / 3.0;
            factor_omega = 2.0 * omega * radius2 / 3.0 - omega / 6.0;
        } else if radius2 < EPSILON {
            let t = fns::tan(omega / 2.0);
            let t2 = t * t;
            // Q = t * (4 * t^2 * radius2 - 3), factor = -2 * Q / (3 * omega).
            let q = t * (4.0 * t2 * radius2 - 3.0);
            factor = -2.0 * q / (3.0 * omega);
            factor_r = -8.0 * t * t2 / (3.0 * omega);
            let dt_domega = 0.5 * (1.0 + t2);
            let q_omega = dt_domega * (12.0 * t2 * radius2 - 3.0);
            factor_omega = -2.0 / (3.0 * omega2) * (q_omega * omega - q);
        } else {
            let radius = fns::sqrt(radius2);
            let t = fns::tan(omega / 2.0);
            let arg = 2.0 * radius * t;
            let atan_arg = fns::atan(arg);
            let denom_arg = 1.0 + arg * arg;
            // denom_arg divides both derivative numerators; hoist its reciprocal.
            let inv_denom_arg = 1.0 / denom_arg;
            factor = atan_arg / (radius * omega);
            factor_r = (2.0 * t * radius * inv_denom_arg - atan_arg)
                / (2.0 * radius2 * radius * omega);
            factor_omega =
                (radius * omega * (1.0 + t * t) * inv_denom_arg - atan_arg) / (radius * omega2);
        }

        let du = a * factor;
        let dv = b * factor;

        *x = f1 * du + c1;
        *y = f2 * dv + c2;

        if let Some(j_uvw) = j_uvw {
            // d(du, dv) / d(a, b) with du = a * factor, dv = b * factor and factor a
            // function of radius2 = a^2 + b^2.
            let cross = 2.0 * a * b * factor_r;
            let da00 = factor + 2.0 * a * a * factor_r;
            let da11 = factor + 2.0 * b * b * factor_r;
            let j_ab = [f1 * da00, f1 * cross, f2 * cross, f2 * da11];
            uvw_jac_from_ab_jac(&j_ab, a, b, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x5 matrix (row-major): d(x, y) / d(fx, fy, cx, cy, omega).
            j_params[..10].copy_from_slice(&[
                du,
                0.0,
                1.0,
                0.0,
                f1 * a * factor_omega,
                0.0,
                dv,
                0.0,
                1.0,
                f2 * b * factor_omega,
            ]);
        }

        true
    }
}

impl CameraModelWithJac for SimpleDivisionCameraModel {
    /// Ignores `check_cheirality`, as COLMAP does: the division model's own surface test
    /// (the discriminant) decides.
    fn img_from_cam_with_jac(
        params: &[f64],
        u: f64,
        v: f64,
        w: f64,
        x: &mut f64,
        y: &mut f64,
        j_params: Option<&mut [f64]>,
        j_uvw: Option<&mut [f64; 6]>,
        _check_cheirality: bool,
    ) -> bool {
        let f = params[0];
        let c1 = params[1];
        let c2 = params[2];
        let k = params[3];

        let with_jac = j_uvw.is_some() || j_params.is_some();
        let Some(scale) = division_scale_with_jac(u, v, w, k, with_jac) else {
            return false;
        };
        let r = scale.r;
        let [dr_du, dr_dv, dr_dw, dr_dk] = scale.dr;

        *x = f * r * u + c1;
        *y = f * r * v + c2;

        if let Some(j_uvw) = j_uvw {
            *j_uvw = [
                f * (r + u * dr_du),
                f * u * dr_dv,
                f * u * dr_dw,
                f * v * dr_du,
                f * (r + v * dr_dv),
                f * v * dr_dw,
            ];
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x4 matrix (row-major): d(x, y) / d(f, cx, cy, k).
            j_params[..8].copy_from_slice(&[
                r * u,
                1.0,
                0.0,
                f * u * dr_dk,
                r * v,
                0.0,
                1.0,
                f * v * dr_dk,
            ]);
        }

        true
    }
}

impl CameraModelWithJac for DivisionCameraModel {
    /// Ignores `check_cheirality`, as COLMAP does: the division model's own surface test
    /// (the discriminant) decides.
    fn img_from_cam_with_jac(
        params: &[f64],
        u: f64,
        v: f64,
        w: f64,
        x: &mut f64,
        y: &mut f64,
        j_params: Option<&mut [f64]>,
        j_uvw: Option<&mut [f64; 6]>,
        _check_cheirality: bool,
    ) -> bool {
        let f1 = params[0];
        let f2 = params[1];
        let c1 = params[2];
        let c2 = params[3];
        let k = params[4];

        let with_jac = j_uvw.is_some() || j_params.is_some();
        let Some(scale) = division_scale_with_jac(u, v, w, k, with_jac) else {
            return false;
        };
        let r = scale.r;
        let [dr_du, dr_dv, dr_dw, dr_dk] = scale.dr;

        *x = f1 * r * u + c1;
        *y = f2 * r * v + c2;

        if let Some(j_uvw) = j_uvw {
            *j_uvw = [
                f1 * (r + u * dr_du),
                f1 * u * dr_dv,
                f1 * u * dr_dw,
                f2 * v * dr_du,
                f2 * (r + v * dr_dv),
                f2 * v * dr_dw,
            ];
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x5 matrix (row-major): d(x, y) / d(fx, fy, cx, cy, k).
            j_params[..10].copy_from_slice(&[
                r * u,
                0.0,
                1.0,
                0.0,
                f1 * u * dr_dk,
                0.0,
                r * v,
                0.0,
                1.0,
                f2 * v * dr_dk,
            ]);
        }

        true
    }
}

impl CameraModelWithJac for EUCMCameraModel {
    fn img_from_cam_with_jac(
        params: &[f64],
        u: f64,
        v: f64,
        w: f64,
        x: &mut f64,
        y: &mut f64,
        j_params: Option<&mut [f64]>,
        j_uvw: Option<&mut [f64; 6]>,
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

        let q = u * u + v * v;
        let rho2 = beta * q + w * w;
        if rho2 < 0.0 {
            return false;
        }
        let rho = fns::sqrt(rho2);
        let den = alpha * rho + (1.0 - alpha) * w;
        if !has_projectable_depth(den, check_cheirality) {
            return false;
        }

        let xn = u / den;
        let yn = v / den;

        *x = f1 * xn + c1;
        *y = f2 * yn + c2;

        if j_uvw.is_some() || j_params.is_some() {
            let inv_rho = 1.0 / rho;
            let inv_den = 1.0 / den;
            let inv_den2 = inv_den * inv_den;
            // Derivatives of the denominator den = alpha*rho + (1-alpha)*w.
            let dden_du = alpha * beta * u * inv_rho;
            let dden_dv = alpha * beta * v * inv_rho;
            let dden_dw = alpha * w * inv_rho + (1.0 - alpha);
            let dden_dalpha = rho - w;
            let dden_dbeta = alpha * q * 0.5 * inv_rho;

            if let Some(j_uvw) = j_uvw {
                let dxn_du = inv_den - u * dden_du * inv_den2;
                let dxn_dv = -u * dden_dv * inv_den2;
                let dxn_dw = -u * dden_dw * inv_den2;
                let dyn_du = -v * dden_du * inv_den2;
                let dyn_dv = inv_den - v * dden_dv * inv_den2;
                let dyn_dw = -v * dden_dw * inv_den2;
                *j_uvw = [
                    f1 * dxn_du,
                    f1 * dxn_dv,
                    f1 * dxn_dw,
                    f2 * dyn_du,
                    f2 * dyn_dv,
                    f2 * dyn_dw,
                ];
            }

            if let Some(j_params) = j_params {
                // J_params is a 2x6 matrix (row-major):
                //   d(x, y) / d(fx, fy, cx, cy, alpha, beta)
                let dxn_dalpha = -u * dden_dalpha * inv_den2;
                let dxn_dbeta = -u * dden_dbeta * inv_den2;
                let dyn_dalpha = -v * dden_dalpha * inv_den2;
                let dyn_dbeta = -v * dden_dbeta * inv_den2;
                j_params[..12].copy_from_slice(&[
                    xn,
                    0.0,
                    1.0,
                    0.0,
                    f1 * dxn_dalpha,
                    f1 * dxn_dbeta,
                    0.0,
                    yn,
                    0.0,
                    1.0,
                    f2 * dyn_dalpha,
                    f2 * dyn_dbeta,
                ]);
            }
        }

        true
    }
}

impl CameraModelWithJac for EquirectangularCameraModel {
    /// Ignores `check_cheirality`, as COLMAP does: every direction but the zero vector
    /// projects.
    fn img_from_cam_with_jac(
        params: &[f64],
        u: f64,
        v: f64,
        w: f64,
        x: &mut f64,
        y: &mut f64,
        j_params: Option<&mut [f64]>,
        j_uvw: Option<&mut [f64; 6]>,
        _check_cheirality: bool,
    ) -> bool {
        let width = params[0];
        let height = params[1];

        let horizontal = fns::sqrt(u * u + w * w);
        if horizontal + v.abs() < f64::EPSILON {
            return false;
        }

        let theta = fns::atan2(u, w);
        let phi = fns::atan2(-v, horizontal);

        // COLMAP divides by `EIGEN_PI`, a `long double` literal; `f64` pi is the same value
        // where long double is double (docs/CPP_DIVERGENCES.md, entry 100).
        const INV_2_PI: f64 = 1.0 / (2.0 * PI);
        const INV_PI: f64 = 1.0 / PI;

        *x = (theta * INV_2_PI + 0.5) * width;
        *y = (0.5 - phi * INV_PI) * height;

        if let Some(j_uvw) = j_uvw {
            let r2 = horizontal * horizontal; // horizontal^2
            let n2 = r2 + v * v; // full squared norm
            // Hoist the shared reciprocals: R2 and N2*horizontal each divide more than
            // one derivative, and without -ffast-math the compiler cannot factor the
            // repeated runtime division out on its own.
            let inv_r2 = 1.0 / r2;
            let inv_n2 = 1.0 / n2;
            let inv_n2_horizontal = inv_n2 / horizontal;
            // theta = atan2(u, w).
            let dtheta_du = w * inv_r2;
            let dtheta_dw = -u * inv_r2;
            // phi = atan2(-v, horizontal), horizontal = sqrt(u^2 + w^2).
            let dphi_du = u * v * inv_n2_horizontal;
            let dphi_dv = -horizontal * inv_n2;
            let dphi_dw = v * w * inv_n2_horizontal;

            *j_uvw = [
                width * INV_2_PI * dtheta_du,
                0.0,
                width * INV_2_PI * dtheta_dw,
                -height * INV_PI * dphi_du,
                -height * INV_PI * dphi_dv,
                -height * INV_PI * dphi_dw,
            ];
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x2 matrix (row-major): d(x, y) / d(width, height).
            j_params[..4].copy_from_slice(&[theta * INV_2_PI + 0.5, 0.0, 0.0, 0.5 - phi * INV_PI]);
        }

        true
    }
}
