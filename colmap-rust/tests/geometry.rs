// Integration-test binary for `colmap_rust::geometry`: one file per COLMAP `*_test.cc`
// (rigid3, sim3, bbox, gps, normalization, pose, pose_prior), plus the Rust-only pycolmap
// oracle comparison `rust_only_transforms_oracle.rs` (fixture
// `tests/data/oracle/geometry_transforms.json` from `oracle/geometry_transforms.py`).
// This file also holds `eigen_matrix_near`, the port of COLMAP's `EigenMatrixNear` matcher
// (util/eigen_matchers.h) the ported tests share. Layout convention: see `tests/math.rs`.
//
// Run: `cargo test -p colmap-rust --test geometry`.

use colmap_rust::linalg::{Matrix3d, Matrix3x4d, Matrix6d, Vector2d, Vector3d, Vector4d};

#[path = "support/oracle_json.rs"]
mod oracle_json;

#[path = "geometry/bbox.rs"]
mod bbox;
#[path = "geometry/gps.rs"]
mod gps;
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
#[path = "geometry/sim3.rs"]
mod sim3;

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

/// Port of COLMAP's `EigenMatrixNear(rhs, tol)` matcher: when `rhs` is exactly zero,
/// `lhs.norm() <= tol`; otherwise Eigen's `lhs.isApprox(rhs, tol)`,
/// `||lhs - rhs|| <= tol * min(||lhs||, ||rhs||)`.
pub fn eigen_matrix_near<T: Coeffs>(lhs: &T, rhs: &T, tol: f64) -> bool {
    let a = lhs.coeffs();
    let b = rhs.coeffs();
    if b.iter().all(|&v| v == 0.0) {
        return l2(&a) <= tol;
    }
    let diff: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    l2(&diff) <= tol * l2(&a).min(l2(&b))
}

/// `EigenMatrixNear(rhs)` with Eigen's default `dummy_precision()` (1e-12).
pub fn eigen_matrix_near_default<T: Coeffs>(lhs: &T, rhs: &T) -> bool {
    eigen_matrix_near(lhs, rhs, 1e-12)
}

/// gtest's `EXPECT_NEAR(a, b, tol)`.
pub fn expect_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
}
