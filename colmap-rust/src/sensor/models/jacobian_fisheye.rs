//! `ImgFromCamWithJac` of the perspective fisheye models (`fisheye.rs`): SIMPLE_RADIAL_FISHEYE,
//! RADIAL_FISHEYE, OPENCV_FISHEYE, THIN_PRISM_FISHEYE, RAD_TAN_THIN_PRISM_FISHEYE,
//! SIMPLE_FISHEYE and FISHEYE. Port of their kernels in `colmap/sensor/models_jacobian.h`.
//! Each chains `d(pixel) / d(fisheye plane)` with [`fisheye_projection_with_jac`]'s
//! `d(fisheye plane) / d(a, b)` and [`uvw_jac_from_ab_jac`]'s `d(a, b) / d(u, v, w)`; the trait
//! and those helpers are in `jacobian.rs`.

use super::jacobian::{
    fisheye_projection_with_jac, mat_mul_2x2, uvw_jac_from_ab_jac, CameraModelWithJac,
};
use super::{
    has_projectable_depth, FisheyeCameraModel, OpenCVFisheyeCameraModel,
    RadTanThinPrismFisheyeModel, RadialFisheyeCameraModel, SimpleFisheyeCameraModel,
    SimpleRadialFisheyeCameraModel, ThinPrismFisheyeCameraModel,
};

/// The shared prologue of every fisheye kernel: `(a, b) = (u/w, v/w)` and its fisheye
/// projection `(uu, vv)`, with `J_fisheye` computed only when `J_uvw` is requested (COLMAP
/// passes `J_uvw ? J_fisheye : nullptr`). Returns `(inv_w, a, b, uu, vv, j_fisheye)`.
fn fisheye_prologue(u: f64, v: f64, w: f64, with_jac: bool) -> (f64, f64, f64, f64, f64, [f64; 4]) {
    let inv_w = 1.0 / w;
    let a = u * inv_w;
    let b = v * inv_w;

    let mut uu = 0.0;
    let mut vv = 0.0;
    let mut j_fisheye = [0.0; 4];
    fisheye_projection_with_jac(a, b, &mut uu, &mut vv, with_jac.then_some(&mut j_fisheye));
    (inv_w, a, b, uu, vv, j_fisheye)
}

impl CameraModelWithJac for SimpleRadialFisheyeCameraModel {
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

        let (inv_w, a, b, uu, vv, j_fisheye) = fisheye_prologue(u, v, w, j_uvw.is_some());

        // Single-parameter radial distortion in fisheye coordinates.
        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let t2 = uu2 + vv2;
        let radial = k * t2;
        let uu_d = uu + uu * radial;
        let vv_d = vv + vv * radial;

        *x = f * uu_d + c1;
        *y = f * vv_d + c2;

        if let Some(j_uvw) = j_uvw {
            let two_k = 2.0 * k;
            // I + d(distortion) / d(uu, vv).
            let ipjd = [
                1.0 + radial + two_k * uu2,
                two_k * uu * vv,
                two_k * uu * vv,
                1.0 + radial + two_k * vv2,
            ];
            let m = mat_mul_2x2(&ipjd, &j_fisheye);
            let j_ab = [f * m[0], f * m[1], f * m[2], f * m[3]];
            uvw_jac_from_ab_jac(&j_ab, a, b, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x4 matrix (row-major): d(x, y) / d(f, cx, cy, k).
            j_params[..8].copy_from_slice(&[
                uu_d,
                1.0,
                0.0,
                f * uu * t2,
                vv_d,
                0.0,
                1.0,
                f * vv * t2,
            ]);
        }

        true
    }
}

impl CameraModelWithJac for RadialFisheyeCameraModel {
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

        let (inv_w, a, b, uu, vv, j_fisheye) = fisheye_prologue(u, v, w, j_uvw.is_some());

        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let t2 = uu2 + vv2;
        let t4 = t2 * t2;
        let radial = k1 * t2 + k2 * t4;
        let uu_d = uu + uu * radial;
        let vv_d = vv + vv * radial;

        *x = f * uu_d + c1;
        *y = f * vv_d + c2;

        if let Some(j_uvw) = j_uvw {
            let d_radial = k1 + 2.0 * k2 * t2;
            let cross = 2.0 * uu * vv * d_radial;
            let ipjd = [
                1.0 + radial + 2.0 * uu2 * d_radial,
                cross,
                cross,
                1.0 + radial + 2.0 * vv2 * d_radial,
            ];
            let m = mat_mul_2x2(&ipjd, &j_fisheye);
            let j_ab = [f * m[0], f * m[1], f * m[2], f * m[3]];
            uvw_jac_from_ab_jac(&j_ab, a, b, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x5 matrix (row-major): d(x, y) / d(f, cx, cy, k1, k2).
            j_params[..10].copy_from_slice(&[
                uu_d,
                1.0,
                0.0,
                f * uu * t2,
                f * uu * t4,
                vv_d,
                0.0,
                1.0,
                f * vv * t2,
                f * vv * t4,
            ]);
        }

        true
    }
}

impl CameraModelWithJac for OpenCVFisheyeCameraModel {
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
        let k3 = params[6];
        let k4 = params[7];

        let (inv_w, a, b, uu, vv, j_fisheye) = fisheye_prologue(u, v, w, j_uvw.is_some());

        // Radial distortion in the theta-scaled fisheye coordinates.
        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let t2 = uu2 + vv2;
        let t4 = t2 * t2;
        let t6 = t4 * t2;
        let t8 = t4 * t4;
        let radial = k1 * t2 + k2 * t4 + k3 * t6 + k4 * t8;
        let uu_d = uu + uu * radial;
        let vv_d = vv + vv * radial;

        *x = f1 * uu_d + c1;
        *y = f2 * vv_d + c2;

        if let Some(j_uvw) = j_uvw {
            let d_radial = k1 + 2.0 * k2 * t2 + 3.0 * k3 * t4 + 4.0 * k4 * t6;
            let cross = 2.0 * uu * vv * d_radial;
            let ipjd = [
                1.0 + radial + 2.0 * uu2 * d_radial,
                cross,
                cross,
                1.0 + radial + 2.0 * vv2 * d_radial,
            ];
            let m = mat_mul_2x2(&ipjd, &j_fisheye);
            let j_ab = [f1 * m[0], f1 * m[1], f2 * m[2], f2 * m[3]];
            uvw_jac_from_ab_jac(&j_ab, a, b, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x8 matrix (row-major):
            //   d(x, y) / d(fx, fy, cx, cy, k1, k2, k3, k4)
            j_params[..16].copy_from_slice(&[
                uu_d,
                0.0,
                1.0,
                0.0,
                f1 * uu * t2,
                f1 * uu * t4,
                f1 * uu * t6,
                f1 * uu * t8,
                0.0,
                vv_d,
                0.0,
                1.0,
                f2 * vv * t2,
                f2 * vv * t4,
                f2 * vv * t6,
                f2 * vv * t8,
            ]);
        }

        true
    }
}

impl CameraModelWithJac for ThinPrismFisheyeCameraModel {
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
        let sx1 = params[10];
        let sy1 = params[11];

        let (inv_w, a, b, uu, vv, j_fisheye) = fisheye_prologue(u, v, w, j_uvw.is_some());

        // Radial + tangential + thin-prism distortion in fisheye coordinates.
        let uu2 = uu * uu;
        let vv2 = vv * vv;
        let uv = uu * vv;
        let r2 = uu2 + vv2;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let r8 = r4 * r4;
        let radial = k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8;
        let du = uu * radial + 2.0 * p1 * uv + p2 * (r2 + 2.0 * uu2) + sx1 * r2;
        let dv = vv * radial + 2.0 * p2 * uv + p1 * (r2 + 2.0 * vv2) + sy1 * r2;
        let uu_d = uu + du;
        let vv_d = vv + dv;

        *x = f1 * uu_d + c1;
        *y = f2 * vv_d + c2;

        if let Some(j_uvw) = j_uvw {
            let d_radial = k1 + 2.0 * k2 * r2 + 3.0 * k3 * r4 + 4.0 * k4 * r6;
            let cross = 2.0 * uv * d_radial;
            let du_duu =
                radial + 2.0 * uu2 * d_radial + 2.0 * p1 * vv + 6.0 * p2 * uu + 2.0 * sx1 * uu;
            let du_dvv = cross + 2.0 * p1 * uu + 2.0 * p2 * vv + 2.0 * sx1 * vv;
            let dv_duu = cross + 2.0 * p2 * vv + 2.0 * p1 * uu + 2.0 * sy1 * uu;
            let dv_dvv =
                radial + 2.0 * vv2 * d_radial + 2.0 * p2 * uu + 6.0 * p1 * vv + 2.0 * sy1 * vv;
            let ipjd = [1.0 + du_duu, du_dvv, dv_duu, 1.0 + dv_dvv];
            let m = mat_mul_2x2(&ipjd, &j_fisheye);
            let j_ab = [f1 * m[0], f1 * m[1], f2 * m[2], f2 * m[3]];
            uvw_jac_from_ab_jac(&j_ab, a, b, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x12 matrix (row-major):
            //   d(x, y) / d(fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, sx1, sy1)
            j_params[..24].copy_from_slice(&[
                uu_d,
                0.0,
                1.0,
                0.0,
                f1 * uu * r2,
                f1 * uu * r4,
                f1 * 2.0 * uv,
                f1 * (r2 + 2.0 * uu2),
                f1 * uu * r6,
                f1 * uu * r8,
                f1 * r2,
                0.0,
                0.0,
                vv_d,
                0.0,
                1.0,
                f2 * vv * r2,
                f2 * vv * r4,
                f2 * (r2 + 2.0 * vv2),
                f2 * 2.0 * uv,
                f2 * vv * r6,
                f2 * vv * r8,
                0.0,
                f2 * r2,
            ]);
        }

        true
    }
}

impl CameraModelWithJac for RadTanThinPrismFisheyeModel {
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
        let k = &params[4..10]; // k0..k5 (radial)
        let p0 = params[10];
        let p1 = params[11];
        let s0 = params[12];
        let s1 = params[13];
        let s2 = params[14];
        let s3 = params[15];

        let (inv_w, a, b, uu, vv, j_fisheye) = fisheye_prologue(u, v, w, j_uvw.is_some());

        // Radial distortion: (xr, yr) = th_radial * (uu, vv). Also accumulate its
        // derivative th_radial' w.r.t. theta2 and the powers theta2^(i+1) used by the
        // per-coefficient parameter Jacobians.
        let theta2 = uu * uu + vv * vv;
        let mut th_radial = 1.0;
        let mut d_th_radial = 0.0; // d(th_radial) / d(theta2)
        let mut theta_pow = [0.0; 6]; // theta2^(i+1)
        let mut power = 1.0;
        for i in 0..6 {
            let prev_power = power; // theta2^i
            power *= theta2; // theta2^(i+1)
            theta_pow[i] = power;
            th_radial += k[i] * power;
            d_th_radial += (i + 1) as f64 * k[i] * prev_power;
        }

        let xr = th_radial * uu;
        let yr = th_radial * vv;

        // Tangential + thin-prism distortion applied to (xr, yr).
        let xr2 = xr * xr;
        let yr2 = yr * yr;
        let xyr = xr * yr;
        let r2 = xr2 + yr2;
        let r4 = r2 * r2;

        let dx_tang = 2.0 * p1 * xyr + p0 * (r2 + 2.0 * xr2);
        let dy_tang = 2.0 * p0 * xyr + p1 * (r2 + 2.0 * yr2);
        let dx_tp = s0 * r2 + s1 * r4;
        let dy_tp = s2 * r2 + s3 * r4;

        let big_x = xr + dx_tang + dx_tp;
        let big_y = yr + dy_tang + dy_tp;

        *x = f1 * big_x + c1;
        *y = f2 * big_y + c2;

        if j_uvw.is_some() || j_params.is_some() {
            // B = d(X, Y) / d(xr, yr) (tangential + thin-prism stage), used by both the
            // point/pose and the radial-coefficient Jacobians.
            let b00 = 1.0 + 2.0 * p1 * yr + 6.0 * p0 * xr + 2.0 * s0 * xr + 4.0 * s1 * xr * r2;
            let b01 = 2.0 * p1 * xr + 2.0 * p0 * yr + 2.0 * s0 * yr + 4.0 * s1 * yr * r2;
            let b10 = 2.0 * p0 * yr + 2.0 * p1 * xr + 2.0 * s2 * xr + 4.0 * s3 * xr * r2;
            let b11 = 1.0 + 2.0 * p0 * xr + 6.0 * p1 * yr + 2.0 * s2 * yr + 4.0 * s3 * yr * r2;

            if let Some(j_uvw) = j_uvw {
                // A = d(xr, yr) / d(uu, vv) (radial stage).
                let cross = 2.0 * uu * vv * d_th_radial;
                let jac_a = [
                    th_radial + 2.0 * uu * uu * d_th_radial,
                    cross,
                    cross,
                    th_radial + 2.0 * vv * vv * d_th_radial,
                ];
                let jac_b = [b00, b01, b10, b11];
                let m2 = mat_mul_2x2(&jac_b, &jac_a); // d(X, Y) / d(uu, vv)
                let m = mat_mul_2x2(&m2, &j_fisheye); // d(X, Y) / d(a, b)
                let j_ab = [f1 * m[0], f1 * m[1], f2 * m[2], f2 * m[3]];
                uvw_jac_from_ab_jac(&j_ab, a, b, inv_w, j_uvw);
            }

            if let Some(j_params) = j_params {
                // J_params is a 2x16 matrix (row-major):
                //   d(x, y) / d(fx, fy, cx, cy, k0..k5, p0, p1, s0, s1, s2, s3)
                j_params[..32].fill(0.0);
                // Focal length and principal point.
                j_params[0] = big_x; // dx/dfx
                j_params[2] = 1.0; // dx/dcx
                j_params[16 + 1] = big_y; // dy/dfy
                j_params[16 + 3] = 1.0; // dy/dcy

                // Radial coefficients k0..k5 (params 4..9): dxr/dk_i = uu * theta2^(i+1),
                // dyr/dk_i = vv * theta2^(i+1), propagated through the tangential/prism
                // stage B.
                for i in 0..6 {
                    let dxr = uu * theta_pow[i];
                    let dyr = vv * theta_pow[i];
                    let d_x = b00 * dxr + b01 * dyr;
                    let d_y = b10 * dxr + b11 * dyr;
                    j_params[4 + i] = f1 * d_x;
                    j_params[16 + 4 + i] = f2 * d_y;
                }

                // Tangential coefficients p0, p1 (params 10, 11).
                j_params[10] = f1 * (r2 + 2.0 * xr2); // dX/dp0
                j_params[11] = f1 * 2.0 * xyr; // dX/dp1
                j_params[16 + 10] = f2 * 2.0 * xyr; // dY/dp0
                j_params[16 + 11] = f2 * (r2 + 2.0 * yr2); // dY/dp1

                // Thin-prism coefficients s0..s3 (params 12..15).
                j_params[12] = f1 * r2; // dX/ds0
                j_params[13] = f1 * r4; // dX/ds1
                j_params[16 + 14] = f2 * r2; // dY/ds2
                j_params[16 + 15] = f2 * r4; // dY/ds3
            }
        }

        true
    }
}

impl CameraModelWithJac for SimpleFisheyeCameraModel {
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

        let (inv_w, a, b, uu, vv, j_fisheye) = fisheye_prologue(u, v, w, j_uvw.is_some());

        *x = f * uu + c1;
        *y = f * vv + c2;

        if let Some(j_uvw) = j_uvw {
            let j_ab = [
                f * j_fisheye[0],
                f * j_fisheye[1],
                f * j_fisheye[2],
                f * j_fisheye[3],
            ];
            uvw_jac_from_ab_jac(&j_ab, a, b, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x3 matrix (row-major): d(x, y) / d(f, cx, cy).
            j_params[..6].copy_from_slice(&[uu, 1.0, 0.0, vv, 0.0, 1.0]);
        }

        true
    }
}

impl CameraModelWithJac for FisheyeCameraModel {
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

        let (inv_w, a, b, uu, vv, j_fisheye) = fisheye_prologue(u, v, w, j_uvw.is_some());

        *x = f1 * uu + c1;
        *y = f2 * vv + c2;

        if let Some(j_uvw) = j_uvw {
            let j_ab = [
                f1 * j_fisheye[0],
                f1 * j_fisheye[1],
                f2 * j_fisheye[2],
                f2 * j_fisheye[3],
            ];
            uvw_jac_from_ab_jac(&j_ab, a, b, inv_w, j_uvw);
        }

        if let Some(j_params) = j_params {
            // J_params is a 2x4 matrix (row-major): d(x, y) / d(fx, fy, cx, cy).
            j_params[..8].copy_from_slice(&[uu, 0.0, 1.0, 0.0, 0.0, vv, 0.0, 1.0]);
        }

        true
    }
}
