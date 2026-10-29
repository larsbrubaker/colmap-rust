//! `ceres::SphereManifold`: the vectors of R^n with the norm of x, stepped in an
//! (n - 1)-dimensional tangent space. Port of the `SphereManifold` class in colmap-sharp's
//! `ColmapSharp/Solver/SphereProductManifolds.cs`, which ports Ceres Solver 2.2.0's
//! `include/ceres/sphere_manifold.h`, `include/ceres/internal/sphere_manifold_functions.h`
//! and `include/ceres/internal/householder_vector.h` (BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md). The [`Manifold`] trait and the simple manifolds are in
//! `manifold.rs`; `product_manifold.rs` glues manifolds together.
//!
//! The step goes through the Householder reflection that maps x to the last axis, so the
//! tangent space is the first n - 1 coordinates there (Hartley and Zisserman's "homogeneous
//! vector" update). COLMAP uses it with n = 3 for a unit-length relative translation. Norms and
//! dot products are left-to-right sums where Ceres goes through Eigen (the divergence in the
//! `manifold.rs` header).
//!
//! Tests: `tests/solver/rust_only_manifold.rs`.

use super::manifold::{with_scratch, Manifold};
use crate::linalg::MACHINE_EPSILON;
use crate::math::fns;
use crate::{check, Result};

/// Port of `ceres::SphereManifold` (its dynamic-size form; COLMAP's `SphereManifold<3>`
/// computes the same values).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SphereManifold {
    size: usize,
}

impl SphereManifold {
    /// The sphere in R^`size`; `size` must be greater than 1 (Ceres CHECK-fails, the port
    /// returns the check's error).
    pub fn new(size: usize) -> Result<Self> {
        check!(
            size > 1,
            "The size of the manifold needs to be greater than 1."
        );
        Ok(Self { size })
    }
}

impl Manifold for SphereManifold {
    fn ambient_size(&self) -> usize {
        self.size
    }

    fn tangent_size(&self) -> usize {
        self.size - 1
    }

    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        let size = self.size;
        let norm_delta = norm(&delta[..size - 1]);
        if norm_delta == 0.0 {
            x_plus_delta[..size].copy_from_slice(&x[..size]);
            return true;
        }

        with_scratch(2 * size, |scratch| {
            let (v, y) = scratch.split_at_mut(size);
            let beta = compute_householder_vector(&x[..size], v);

            // y = [sin(|delta|) / |delta| * delta, cos(|delta|)], reflected back and scaled
            // to |x|.
            let sin_delta_by_delta = fns::sin(norm_delta) / norm_delta;
            for i in 0..size - 1 {
                y[i] = sin_delta_by_delta * delta[i];
            }
            y[size - 1] = fns::cos(norm_delta);
            apply_householder_vector(y, v, beta);
            let x_norm = norm(&x[..size]);
            for i in 0..size {
                x_plus_delta[i] = x_norm * y[i];
            }
        });
        true
    }

    fn plus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        // Row-major size x (size - 1): column i is -beta v_i v + e_i, all scaled by |x|.
        let size = self.size;
        let tangent_size = size - 1;
        with_scratch(size, |v| {
            let beta = compute_householder_vector(&x[..size], v);
            for i in 0..tangent_size {
                let scale = -beta * v[i];
                for r in 0..size {
                    jacobian[r * tangent_size + i] = scale * v[r];
                }
                jacobian[i * tangent_size + i] += 1.0;
            }
        });

        let x_norm = norm(&x[..size]);
        for value in &mut jacobian[..size * tangent_size] {
            *value *= x_norm;
        }
        true
    }

    fn minus(&self, y: &[f64], x: &[f64], y_minus_x: &mut [f64]) -> bool {
        let size = self.size;
        let tangent_size = size - 1;
        with_scratch(2 * size, |scratch| {
            let (v, hy) = scratch.split_at_mut(size);
            let beta = compute_householder_vector(&x[..size], v);

            hy.copy_from_slice(&y[..size]);
            apply_householder_vector(hy, v, beta);
            let x_norm = norm(&x[..size]);
            for value in hy.iter_mut() {
                *value /= x_norm;
            }

            let y_last = hy[tangent_size];
            let hy_norm = norm(&hy[..tangent_size]);
            if hy_norm == 0.0 {
                y_minus_x[..tangent_size].fill(0.0);
                y_minus_x[tangent_size - 1] = if y_last >= 0.0 {
                    0.0
                } else {
                    std::f64::consts::PI
                };
            } else {
                let scale = fns::atan2(hy_norm, y_last) / hy_norm;
                for i in 0..tangent_size {
                    y_minus_x[i] = scale * hy[i];
                }
            }
        });
        true
    }

    fn minus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        // Row-major (size - 1) x size: row i is -beta v_i v^T + e_i^T, all divided by |x|.
        let size = self.size;
        let tangent_size = size - 1;
        with_scratch(size, |v| {
            let beta = compute_householder_vector(&x[..size], v);
            for i in 0..tangent_size {
                let scale = -beta * v[i];
                for c in 0..size {
                    jacobian[i * size + c] = scale * v[c];
                }
                jacobian[i * size + i] += 1.0;
            }
        });

        let x_norm = norm(&x[..size]);
        for value in &mut jacobian[..size * tangent_size] {
            *value /= x_norm;
        }
        true
    }
}

/// The Euclidean norm, squares summed left to right.
fn norm(values: &[f64]) -> f64 {
    let mut sum = 0.0;
    for value in values {
        sum += value * value;
    }
    fns::sqrt(sum)
}

/// Port of `ceres::internal::ComputeHouseholderVector` (Golub and Van Loan, Algorithm 5.1.1,
/// with the pivot at the last coordinate): writes v and returns beta with
/// (I - beta v v^T) x = |x| e_n and v_n = 1.
fn compute_householder_vector(x: &[f64], v: &mut [f64]) -> f64 {
    let n = x.len();
    let mut sigma = 0.0;
    for value in &x[..n - 1] {
        sigma += value * value;
    }

    v.copy_from_slice(x);
    v[n - 1] = 1.0;

    let x_pivot = x[n - 1];
    if sigma <= MACHINE_EPSILON {
        return if x_pivot < 0.0 { 2.0 } else { 0.0 };
    }

    let mu = fns::sqrt(x_pivot * x_pivot + sigma);
    let v_pivot = if x_pivot <= 0.0 {
        x_pivot - mu
    } else {
        -sigma / (x_pivot + mu)
    };
    let beta = 2.0 * v_pivot * v_pivot / (sigma + v_pivot * v_pivot);
    for value in &mut v[..n - 1] {
        *value /= v_pivot;
    }
    beta
}

/// `ceres::internal::ApplyHouseholderVector` in place: y <- y - v (beta (v^T y)).
fn apply_householder_vector(y: &mut [f64], v: &[f64], beta: f64) {
    let mut dot = 0.0;
    for (vi, yi) in v.iter().zip(y.iter()) {
        dot += vi * yi;
    }

    let scale = beta * dot;
    for (yi, vi) in y.iter_mut().zip(v) {
        *yi -= vi * scale;
    }
}
