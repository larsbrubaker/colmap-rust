//! `ImgFromCamWithJac` of the polynomial pinhole models (`pinhole.rs`): SIMPLE_PINHOLE,
//! PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV and FULL_OPENCV. Port of their kernels in
//! `colmap/sensor/models_jacobian.h`; the trait and shared helpers are in `jacobian.rs`.

use super::jacobian::{uvw_jac_from_ab_jac, CameraModelWithJac};
use super::{
    has_projectable_depth, FullOpenCVCameraModel, OpenCVCameraModel, PinholeCameraModel,
    RadialCameraModel, SimplePinholeCameraModel, SimpleRadialCameraModel,
};

impl CameraModelWithJac for SimplePinholeCameraModel {
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

        let f = params[0];
        let c1 = params[1];
        let c2 = params[2];

        let inv_w = 1.0 / w;
        let uu = u * inv_w;
        let vv = v * inv_w;

        *x = f * uu + c1;
        *y = f * vv + c2;

        if let Some(j_uvw) = j_uvw {
            // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w)
            // x = f * u / w + c1, y = f * v / w + c2
            let f_inv_w = f * inv_w;
            *j_uvw = [f_inv_w, 0.0, -f_inv_w * uu, 0.0, f_inv_w, -f_inv_w * vv];
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x3 matrix (row-major): d(x, y) / d(f, cx, cy)
            j_params[..6].copy_from_slice(&[uu, 1.0, 0.0, vv, 0.0, 1.0]);
        }

        true
    }
}

impl CameraModelWithJac for PinholeCameraModel {
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

        let inv_w = 1.0 / w;
        let uu = u * inv_w;
        let vv = v * inv_w;

        *x = f1 * uu + c1;
        *y = f2 * vv + c2;

        if let Some(j_uvw) = j_uvw {
            // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w)
            // x = fx * u / w + cx, y = fy * v / w + cy
            *j_uvw = [
                f1 * inv_w,
                0.0,
                -f1 * inv_w * uu,
                0.0,
                f2 * inv_w,
                -f2 * inv_w * vv,
            ];
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x4 matrix (row-major): d(x, y) / d(fx, fy, cx, cy)
            j_params[..8].copy_from_slice(&[uu, 0.0, 1.0, 0.0, 0.0, vv, 0.0, 1.0]);
        }

        true
    }
}

impl CameraModelWithJac for SimpleRadialCameraModel {
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

        let f = params[0];
        let c1 = params[1];
        let c2 = params[2];
        let k = params[3];

        let inv_w = 1.0 / w;
        let uu = u * inv_w;
        let vv = v * inv_w;

        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let r2 = uu2 + vv2;
        let k_r2 = k * r2;
        let alpha = 1.0 + k_r2;
        let xd = alpha * uu;
        let yd = alpha * vv;

        *x = f * xd + c1;
        *y = f * yd + c2;

        if let Some(j_uvw) = j_uvw {
            // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w)
            //
            // x = f * alpha * uu + c1, y = f * alpha * vv + c2
            // where alpha = 1 + k * r2, r2 = uu^2 + vv^2, uu = u/w, vv = v/w
            //
            // Using chain rule:
            // dx/du = f/w * (alpha + 2*k*uu^2)
            // dx/dv = f/w * 2*k*uu*vv
            // dx/dw = -f*uu/w * (1 + 3*k*r2)
            // dy/du = f/w * 2*k*uu*vv
            // dy/dv = f/w * (alpha + 2*k*vv^2)
            // dy/dw = -f*vv/w * (1 + 3*k*r2)
            let two_k = 2.0 * k;
            let f_inv_w = f * inv_w;
            let beta = 1.0 + 3.0 * k_r2;
            let two_k_uu_vv = two_k * uu * vv;

            *j_uvw = [
                f_inv_w * (alpha + two_k * uu2),
                f_inv_w * two_k_uu_vv,
                -f_inv_w * uu * beta,
                f_inv_w * two_k_uu_vv,
                f_inv_w * (alpha + two_k * vv2),
                -f_inv_w * vv * beta,
            ];
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x4 matrix (row-major): d(x, y) / d(f, cx, cy, k)
            //
            // x = f * alpha * uu + cx, y = f * alpha * vv + cy
            //
            // dx/df = alpha * uu, dx/dcx = 1, dx/dcy = 0, dx/dk = f * uu * r2
            // dy/df = alpha * vv, dy/dcx = 0, dy/dcy = 1, dy/dk = f * vv * r2
            j_params[..8].copy_from_slice(&[xd, 1.0, 0.0, f * uu * r2, yd, 0.0, 1.0, f * vv * r2]);
        }

        true
    }
}

impl CameraModelWithJac for RadialCameraModel {
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

        let f = params[0];
        let c1 = params[1];
        let c2 = params[2];
        let k1 = params[3];
        let k2 = params[4];

        let inv_w = 1.0 / w;
        let uu = u * inv_w;
        let vv = v * inv_w;

        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let r2 = uu2 + vv2;
        let r4 = r2 * r2;
        let radial = k1 * r2 + k2 * r4;
        let xd = uu * (1.0 + radial);
        let yd = vv * (1.0 + radial);

        *x = f * xd + c1;
        *y = f * yd + c2;

        if let Some(j_uvw) = j_uvw {
            // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w).
            // With xd = uu * (1 + radial), yd = vv * (1 + radial),
            // radial = k1 * r2 + k2 * r2^2, r2 = uu^2 + vv^2, the distortion Jacobian
            // in normalized coordinates (uu, vv) is:
            //   d(xd)/d(uu) = 1 + radial + 2 * uu^2 * d_radial_d_r2
            //   d(xd)/d(vv) = 2 * uu * vv * d_radial_d_r2
            //   d(yd)/d(uu) = 2 * uu * vv * d_radial_d_r2
            //   d(yd)/d(vv) = 1 + radial + 2 * vv^2 * d_radial_d_r2
            // where d_radial_d_r2 = k1 + 2 * k2 * r2. The chain rule through
            // (uu, vv) = (u/w, v/w) yields the columns below.
            let d_radial_d_r2 = k1 + 2.0 * k2 * r2;
            let cross = 2.0 * uu * vv * d_radial_d_r2;
            let a00 = f * (1.0 + radial + 2.0 * uu2 * d_radial_d_r2);
            let a01 = f * cross;
            let a10 = f * cross;
            let a11 = f * (1.0 + radial + 2.0 * vv2 * d_radial_d_r2);
            uvw_jac_from_ab_jac(&[a00, a01, a10, a11], uu, vv, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x5 matrix (row-major): d(x, y) / d(f, cx, cy, k1, k2)
            j_params[..10].copy_from_slice(&[
                xd,
                1.0,
                0.0,
                f * uu * r2,
                f * uu * r4,
                yd,
                0.0,
                1.0,
                f * vv * r2,
                f * vv * r4,
            ]);
        }

        true
    }
}

impl CameraModelWithJac for OpenCVCameraModel {
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
        let k1 = params[4];
        let k2 = params[5];
        let p1 = params[6];
        let p2 = params[7];

        let inv_w = 1.0 / w;
        let uu = u * inv_w;
        let vv = v * inv_w;

        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let uv = uu * vv;
        let r2 = uu2 + vv2;
        let r4 = r2 * r2;
        let radial = k1 * r2 + k2 * r4;

        let du = uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu2);
        let dv = vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv2);
        let xd = uu + du;
        let yd = vv + dv;

        *x = f1 * xd + c1;
        *y = f2 * yd + c2;

        if let Some(j_uvw) = j_uvw {
            // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w).
            // Partial derivatives of the OpenCV distortion (radial + tangential) in
            // normalized coordinates (uu, vv), with d_radial_d_r2 = k1 + 2 * k2 * r2:
            //   d(du)/d(uu) = radial + 2*uu^2*d_radial_d_r2 + 2*p1*vv + 6*p2*uu
            //   d(du)/d(vv) = 2*uu*vv*d_radial_d_r2 + 2*p1*uu + 2*p2*vv
            //   d(dv)/d(uu) = 2*uu*vv*d_radial_d_r2 + 2*p2*vv + 2*p1*uu
            //   d(dv)/d(vv) = radial + 2*vv^2*d_radial_d_r2 + 2*p2*uu + 6*p1*vv
            // The chain rule through (uu, vv) = (u/w, v/w) yields the columns below.
            let d_radial_d_r2 = k1 + 2.0 * k2 * r2;
            let cross = 2.0 * uv * d_radial_d_r2;
            let du_duu = radial + 2.0 * uu2 * d_radial_d_r2 + 2.0 * p1 * vv + 6.0 * p2 * uu;
            let du_dvv = cross + 2.0 * p1 * uu + 2.0 * p2 * vv;
            let dv_duu = cross + 2.0 * p2 * vv + 2.0 * p1 * uu;
            let dv_dvv = radial + 2.0 * vv2 * d_radial_d_r2 + 2.0 * p2 * uu + 6.0 * p1 * vv;

            let a00 = f1 * (1.0 + du_duu);
            let a01 = f1 * du_dvv;
            let a10 = f2 * dv_duu;
            let a11 = f2 * (1.0 + dv_dvv);
            uvw_jac_from_ab_jac(&[a00, a01, a10, a11], uu, vv, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x8 matrix (row-major):
            //   d(x, y) / d(fx, fy, cx, cy, k1, k2, p1, p2)
            j_params[..16].copy_from_slice(&[
                xd,
                0.0,
                1.0,
                0.0,
                f1 * uu * r2,
                f1 * uu * r4,
                f1 * 2.0 * uv,
                f1 * (r2 + 2.0 * uu2),
                0.0,
                yd,
                0.0,
                1.0,
                f2 * vv * r2,
                f2 * vv * r4,
                f2 * (r2 + 2.0 * vv2),
                f2 * 2.0 * uv,
            ]);
        }

        true
    }
}

impl CameraModelWithJac for FullOpenCVCameraModel {
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
        let k1 = params[4];
        let k2 = params[5];
        let p1 = params[6];
        let p2 = params[7];
        let k3 = params[8];
        let k4 = params[9];
        let k5 = params[10];
        let k6 = params[11];

        let inv_w = 1.0 / w;
        let uu = u * inv_w;
        let vv = v * inv_w;

        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let uv = uu * vv;
        let r2 = uu2 + vv2;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        // Rational radial term: radial = num / den.
        let num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;
        let den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6;
        let inv_den = 1.0 / den;
        let radial = num * inv_den;

        let xd = uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu2);
        let yd = vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv2);

        *x = f1 * xd + c1;
        *y = f2 * yd + c2;

        if let Some(j_uvw) = j_uvw {
            // J_uvw is a 2x3 matrix (row-major): d(x, y) / d(u, v, w).
            // With xd = uu * radial + tangential_x, yd = vv * radial + tangential_y,
            // and radial = num / den, the derivative of the rational radial term is
            //   d(radial)/d(r2) = (num' * den - num * den') / den^2
            // with num' = k1 + 2*k2*r2 + 3*k3*r4, den' = k4 + 2*k5*r2 + 3*k6*r4.
            // The distortion Jacobian in normalized coordinates (uu, vv) is:
            //   d(xd)/d(uu) = radial + 2*uu^2*d_radial_d_r2 + 2*p1*vv + 6*p2*uu
            //   d(xd)/d(vv) = 2*uu*vv*d_radial_d_r2 + 2*p1*uu + 2*p2*vv
            //   d(yd)/d(uu) = 2*uu*vv*d_radial_d_r2 + 2*p2*vv + 2*p1*uu
            //   d(yd)/d(vv) = radial + 2*vv^2*d_radial_d_r2 + 2*p2*uu + 6*p1*vv
            // The chain rule through (uu, vv) = (u/w, v/w) yields the columns below.
            let num_prime = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4;
            let den_prime = k4 + 2.0 * k5 * r2 + 3.0 * k6 * r4;
            let d_radial_d_r2 = (num_prime * den - num * den_prime) * inv_den * inv_den;
            let cross = 2.0 * uv * d_radial_d_r2;
            let xd_duu = radial + 2.0 * uu2 * d_radial_d_r2 + 2.0 * p1 * vv + 6.0 * p2 * uu;
            let xd_dvv = cross + 2.0 * p1 * uu + 2.0 * p2 * vv;
            let yd_duu = cross + 2.0 * p2 * vv + 2.0 * p1 * uu;
            let yd_dvv = radial + 2.0 * vv2 * d_radial_d_r2 + 2.0 * p2 * uu + 6.0 * p1 * vv;

            let a00 = f1 * xd_duu;
            let a01 = f1 * xd_dvv;
            let a10 = f2 * yd_duu;
            let a11 = f2 * yd_dvv;
            uvw_jac_from_ab_jac(&[a00, a01, a10, a11], uu, vv, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x12 matrix (row-major):
            //   d(x, y) / d(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6)
            // The numerator coefficients enter as d(radial)/d(k_i) = r^(2i) / den; the
            // denominator coefficients as d(radial)/d(k_j) = -num * r^(2j) / den^2.
            let num_k1 = r2 * inv_den;
            let num_k2 = r4 * inv_den;
            let num_k3 = r6 * inv_den;
            let neg_num_inv_den2 = -num * inv_den * inv_den;
            let den_k4 = neg_num_inv_den2 * r2;
            let den_k5 = neg_num_inv_den2 * r4;
            let den_k6 = neg_num_inv_den2 * r6;

            j_params[..24].copy_from_slice(&[
                xd,
                0.0,
                1.0,
                0.0,
                f1 * uu * num_k1,
                f1 * uu * num_k2,
                f1 * 2.0 * uv,
                f1 * (r2 + 2.0 * uu2),
                f1 * uu * num_k3,
                f1 * uu * den_k4,
                f1 * uu * den_k5,
                f1 * uu * den_k6,
                0.0,
                yd,
                0.0,
                1.0,
                f2 * vv * num_k1,
                f2 * vv * num_k2,
                f2 * (r2 + 2.0 * vv2),
                f2 * 2.0 * uv,
                f2 * vv * num_k3,
                f2 * vv * den_k4,
                f2 * vv * den_k5,
                f2 * vv * den_k6,
            ]);
        }

        true
    }
}
