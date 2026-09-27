// Integration-test binary for `colmap_rust::geometry`: one file per COLMAP `*_test.cc`
// (rigid3, sim3, bbox, gps, normalization, pose, pose_prior, essential_matrix,
// homography_matrix, triangulation), plus the Rust-only pycolmap oracle comparisons
// `rust_only_transforms_oracle.rs` (fixture `tests/data/oracle/geometry_transforms.json`
// from `oracle/geometry_transforms.py`) and `rust_only_two_view_oracle.rs`
// (`geometry_two_view.json` from `oracle/geometry_two_view.py`), and the hand-built
// `rust_only_two_view.rs` cases for two-view functions COLMAP's tests do not reach. This file also holds
// `eigen_matrix_near` and `rigid3d_near`, the ports of COLMAP's `EigenMatrixNear`
// (util/eigen_matchers.h) and `Rigid3dNear` (geometry/rigid3_matchers.h) matchers the ported
// tests share. Layout convention: see `tests/math.rs`.
//
// Run: `cargo test -p colmap-rust --test geometry`.

use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Matrix3d, Matrix3x4d, Matrix6d, Vector2d, Vector3d, Vector4d};

#[path = "support/oracle_json.rs"]
mod oracle_json;

#[path = "geometry/bbox.rs"]
mod bbox;
#[path = "geometry/essential_matrix.rs"]
mod essential_matrix;
#[path = "geometry/gps.rs"]
mod gps;
#[path = "geometry/homography_matrix.rs"]
mod homography_matrix;
#[path = "geometry/normalization.rs"]
mod normalization;
#[path = "geometry/pose.rs"]
mod pose;
#[path = "geometry/pose_prior.rs"]
mod pose_prior;
#[path = "geometry/rigid3.rs"]
mod rigid3;
#[path = "geometry/rust_only_transforms_oracle.rs"]
mod rust_only_transforms_oracle;
#[path = "geometry/rust_only_two_view.rs"]
mod rust_only_two_view;
#[path = "geometry/rust_only_two_view_oracle.rs"]
mod rust_only_two_view_oracle;
#[path = "geometry/sim3.rs"]
mod sim3;
#[path = "geometry/triangulation.rs"]
mod triangulation;

/// The coefficients of a fixed-size Eigen-like value, in storage order.
pub trait Coeffs {
    fn coeffs(&self) -> Vec<f64>;
}

impl Coeffs for Vector2d {
    fn coeffs(&self) -> Vec<f64> {
        self.to_array().to_vec()
    }
}
impl Coeffs for Vector3d {
    fn coeffs(&self) -> Vec<f64> {
        self.to_array().to_vec()
    }
}
impl Coeffs for Vector4d {
    fn coeffs(&self) -> Vec<f64> {
        self.to_array().to_vec()
    }
}
impl Coeffs for Matrix3d {
    fn coeffs(&self) -> Vec<f64> {
        self.as_slice().to_vec()
    }
}
impl Coeffs for Matrix3x4d {
    fn coeffs(&self) -> Vec<f64> {
        self.as_slice().to_vec()
    }
}
impl Coeffs for Matrix6d {
    fn coeffs(&self) -> Vec<f64> {
        self.as_slice().to_vec()
    }
}

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Port of COLMAP's `EigenMatrixNear(rhs, tol)` matcher: when `rhs.isZero()` (Eigen's
/// default precision: every `|coefficient| <= 1e-12`), `lhs.norm() <= tol`; otherwise
/// Eigen's `lhs.isApprox(rhs, tol)`, `||lhs - rhs|| <= tol * min(||lhs||, ||rhs||)`.
pub fn eigen_matrix_near<T: Coeffs>(lhs: &T, rhs: &T, tol: f64) -> bool {
    let a = lhs.coeffs();
    let b = rhs.coeffs();
    if b.iter().all(|&v| v.abs() <= 1e-12) {
        return l2(&a) <= tol;
    }
    let diff: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    l2(&diff) <= tol * l2(&a).min(l2(&b))
}

/// `EigenMatrixNear(rhs)` with Eigen's default `dummy_precision()` (1e-12).
pub fn eigen_matrix_near_default<T: Coeffs>(lhs: &T, rhs: &T) -> bool {
    eigen_matrix_near(lhs, rhs, 1e-12)
}

/// Port of COLMAP's `Rigid3dNear(rhs, rtol, ttol)` matcher (geometry/rigid3_matchers.h):
/// the rotations within `rtol` angular distance, and the translations within `ttol`
/// (`norm() <= ttol` when `rhs`'s is zero by Eigen's `isZero()`, every `|coefficient| <=
/// 1e-12`; Eigen's `isApprox` otherwise).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // !(a <= b) is deliberate: NaN must fail.
pub fn rigid3d_near(lhs: &Rigid3d, rhs: &Rigid3d, rtol: f64, ttol: f64) -> bool {
    // Note the !(a <= b) form, as in COLMAP, to reject NaNs.
    if !(lhs.rotation.angular_distance(rhs.rotation) <= rtol) {
        return false;
    }
    if rhs.translation.to_array().iter().all(|c| c.abs() <= 1e-12) {
        lhs.translation.norm() <= ttol
    } else {
        lhs.translation.is_approx_with(rhs.translation, ttol)
    }
}

/// gtest's `EXPECT_NEAR(a, b, tol)`.
pub fn expect_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
}
