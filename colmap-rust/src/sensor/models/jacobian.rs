//! The analytic image-projection Jacobians of `colmap/sensor/models_jacobian.h` (4.2.0): the
//! [`CameraModelWithJac`] trait (COLMAP's per-model `ImgFromCamWithJac`) and the shared
//! helpers of COLMAP's `namespace internal`. The per-model kernels are in
//! `jacobian_pinhole.rs`, `jacobian_fisheye.rs` and `jacobian_other.rs`, split like
//! `pinhole.rs` / `fisheye.rs` / `other.rs`; the runtime dispatch
//! ([`super::camera_model_img_from_cam_with_jac`]) is in `dispatch.rs`.
//!
//! Jacobians are row-major 2xN arrays, as COLMAP's `double*` outputs are; C++'s `nullptr`
//! ("skip this Jacobian") is `None`. Every expression keeps COLMAP's operation order, so a
//! kernel rounds like the C++ up to libm and FMA contraction (docs/CPP_DIVERGENCES.md, entry
//! 100). Against `ImgFromCam` on a Jet (automatic differentiation) the analytic Jacobians
//! agree to COLMAP's 1e-10 (Tier B): they are different formulas for the same derivative.
//! Tests: `src/sensor/models/jacobian_tests.rs` (`models_jacobian_test.cc` 1:1).

use crate::math::fns;

use super::CameraModel;

/// A camera model with COLMAP's analytic projection Jacobian (`has_img_from_cam_with_jac`).
/// Every model COLMAP 4.2.0 dispatches over has one, which the dispatch in `dispatch.rs`
/// requires at compile time, as COLMAP's `static_assert` does.
pub trait CameraModelWithJac: CameraModel {
    /// Port of `ImgFromCamWithJac`: [`CameraModel::img_from_cam`] on `f64`, also writing the
    /// row-major 2 x `NUM_PARAMS` Jacobian `d(x, y) / d(params)` into `j_params` and the
    /// row-major 2x3 Jacobian `d(x, y) / d(u, v, w)` into `j_uvw` when they are given.
    /// Returns false (outputs unspecified) where `img_from_cam` fails. COLMAP's default for
    /// `check_cheirality` is true.
    #[allow(clippy::too_many_arguments)]
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
    ) -> bool;
}

/// Port of `internal::FisheyeProjectionWithJac`. Fisheye (equidistant) projection: maps
/// normalized coordinates `(a, b) = (u/w, v/w)` to the projected fisheye coordinates
/// `(uu, vv)`. When `j_fisheye` is given it also receives the 2x2 Jacobian
/// `d(uu, vv) / d(a, b)` in row-major order; pass `None` on the value-only path to skip that
/// work. Mirrors `BasePerspectiveFisheyeCameraModel::FisheyeFromNormal`.
pub(super) fn fisheye_projection_with_jac(
    a: f64,
    b: f64,
    uu: &mut f64,
    vv: &mut f64,
    j_fisheye: Option<&mut [f64; 4]>,
) {
    let r2 = a * a + b * b;
    let r = fns::sqrt(r2);
    if r < f64::EPSILON {
        // Identity in the limit r -> 0 (theta / r -> 1).
        *uu = a;
        *vv = b;
        if let Some(j) = j_fisheye {
            *j = [1.0, 0.0, 0.0, 1.0];
        }
        return;
    }
    let theta = fns::atan(r);
    let s = theta / r;
    *uu = s * a;
    *vv = s * b;
    if let Some(j) = j_fisheye {
        // With s = atan(r) / r and r = sqrt(a^2 + b^2), the Jacobian of (uu, vv) =
        // (s * a, s * b) w.r.t. (a, b) is s * I + g * outer((a, b), (a, b)), where
        // g = (ds/dr) / r = (r / (1 + r^2) - atan(r)) / r^3.
        let g = (r / (1.0 + r2) - theta) / (r2 * r);
        j[0] = s + a * a * g;
        j[1] = a * b * g;
        j[2] = a * b * g;
        j[3] = s + b * b * g;
    }
}

/// The division model's projection scale and its derivatives: `r` and, when requested,
/// `[dr/du, dr/dv, dr/dw, dr/dk]` (all zero otherwise).
pub(super) struct DivisionScale {
    /// The scale `r`.
    pub(super) r: f64,
    /// `d(r) / d(u, v, w, k)`.
    pub(super) dr: [f64; 4],
}

/// Port of `internal::DivisionScaleWithJac`. Solves the one-parameter division model's
/// projection scale `r` from the camera point `(u, v, w)`: the depth is scaled by
/// `r = 2 / (w + sqrt(w^2 - 4*k*rho2))`, with `rho2 = u^2 + v^2`. Returns `None` when the
/// point is behind the model's projection surface (negative discriminant). With `with_jac`
/// it also returns the derivatives of `r` w.r.t. `(u, v, w, k)`.
pub(super) fn division_scale_with_jac(
    u: f64,
    v: f64,
    w: f64,
    k: f64,
    with_jac: bool,
) -> Option<DivisionScale> {
    let rho2 = u * u + v * v;
    let disc_sq = w * w - 4.0 * rho2 * k;
    if disc_sq < 0.0 {
        return None;
    }
    let disc = fns::sqrt(disc_sq);
    let r = 2.0 / (w + disc);
    let mut dr = [0.0; 4];
    if with_jac {
        let inv_disc = 1.0 / disc;
        let r_sq = r * r;
        dr[0] = 2.0 * r_sq * k * u * inv_disc;
        dr[1] = 2.0 * r_sq * k * v * inv_disc;
        dr[2] = -0.5 * r_sq * (1.0 + w * inv_disc);
        dr[3] = r_sq * rho2 * inv_disc;
    }
    Some(DivisionScale { r, dr })
}

/// Port of `internal::MatMul2x2`: multiplies two row-major 2x2 matrices, `lhs * rhs`.
pub(super) fn mat_mul_2x2(lhs: &[f64; 4], rhs: &[f64; 4]) -> [f64; 4] {
    [
        lhs[0] * rhs[0] + lhs[1] * rhs[2],
        lhs[0] * rhs[1] + lhs[1] * rhs[3],
        lhs[2] * rhs[0] + lhs[3] * rhs[2],
        lhs[2] * rhs[1] + lhs[3] * rhs[3],
    ]
}

/// Port of `internal::UvwJacFromAbJac`. Given `j_ab` (row-major 2x2) `= d(x, y) / d(a, b)`
/// with `a = u/w` and `b = v/w`, computes the 2x3 Jacobian `j_uvw = d(x, y) / d(u, v, w)`
/// via the chain rule through `(a, b) = (u/w, v/w)`. The pinhole models' kernels spell the
/// same six expressions out inline with `(a, b) = (uu, vv)`, so they call this too.
pub(super) fn uvw_jac_from_ab_jac(j_ab: &[f64; 4], a: f64, b: f64, inv_w: f64, j_uvw: &mut [f64; 6]) {
    j_uvw[0] = j_ab[0] * inv_w;
    j_uvw[1] = j_ab[1] * inv_w;
    j_uvw[2] = -(j_ab[0] * a + j_ab[1] * b) * inv_w;
    j_uvw[3] = j_ab[2] * inv_w;
    j_uvw[4] = j_ab[3] * inv_w;
    j_uvw[5] = -(j_ab[2] * a + j_ab[3] * b) * inv_w;
}
