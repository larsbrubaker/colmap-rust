//! `ceres::ProductManifold`: the Cartesian product of manifolds, laid out block after block in
//! both the ambient and the tangent space, with block-diagonal Jacobians. Port of the
//! `ProductManifold` class in colmap-sharp's `ColmapSharp/Solver/SphereProductManifolds.cs`,
//! which ports Ceres Solver 2.2.0's `include/ceres/product_manifold.h` (BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md). Its components are any [`Manifold`]s (`manifold.rs`,
//! `sphere_manifold.rs`); COLMAP glues a quaternion to a translation or a sphere.
//!
//! Ceres' variadic constructor takes the components by value or by pointer; here they are
//! boxed trait objects, owned by the product.
//!
//! Tests: `tests/solver/rust_only_manifold.rs`.

use super::manifold::{with_scratch, Manifold};
use crate::{check_ge, Result};

/// Port of `ceres::ProductManifold`.
pub struct ProductManifold {
    manifolds: Vec<Box<dyn Manifold>>,
    ambient_offsets: Vec<usize>,
    tangent_offsets: Vec<usize>,
    ambient_size: usize,
    tangent_size: usize,
    /// The largest component Jacobian, Ceres' `buffer_size_`.
    buffer_size: usize,
}

impl ProductManifold {
    /// The product of `manifolds`, in order; Ceres' constructor takes at least two.
    pub fn new(manifolds: Vec<Box<dyn Manifold>>) -> Result<Self> {
        check_ge!(manifolds.len(), 2);
        let mut ambient_offsets = Vec::with_capacity(manifolds.len());
        let mut tangent_offsets = Vec::with_capacity(manifolds.len());
        let mut ambient_size = 0;
        let mut tangent_size = 0;
        let mut buffer_size = 0;
        for manifold in &manifolds {
            ambient_offsets.push(ambient_size);
            tangent_offsets.push(tangent_size);
            ambient_size += manifold.ambient_size();
            tangent_size += manifold.tangent_size();
            buffer_size = buffer_size.max(manifold.ambient_size() * manifold.tangent_size());
        }

        Ok(Self {
            manifolds,
            ambient_offsets,
            tangent_offsets,
            ambient_size,
            tangent_size,
            buffer_size,
        })
    }
}

/// Copies the row-major `rows x cols` `block` into `matrix` (row-major, `matrix_cols` wide)
/// at (`row0`, `col0`).
fn copy_block(
    block: &[f64],
    rows: usize,
    cols: usize,
    matrix: &mut [f64],
    matrix_cols: usize,
    row0: usize,
    col0: usize,
) {
    for r in 0..rows {
        let start = (row0 + r) * matrix_cols + col0;
        matrix[start..start + cols].copy_from_slice(&block[r * cols..(r + 1) * cols]);
    }
}

impl Manifold for ProductManifold {
    fn ambient_size(&self) -> usize {
        self.ambient_size
    }

    fn tangent_size(&self) -> usize {
        self.tangent_size
    }

    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        for (i, manifold) in self.manifolds.iter().enumerate() {
            let ambient = self.ambient_offsets[i];
            let tangent = self.tangent_offsets[i];
            if !manifold.plus(
                &x[ambient..],
                &delta[tangent..],
                &mut x_plus_delta[ambient..],
            ) {
                return false;
            }
        }
        true
    }

    fn plus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        jacobian[..self.ambient_size * self.tangent_size].fill(0.0);
        with_scratch(self.buffer_size, |buffer| {
            for (i, manifold) in self.manifolds.iter().enumerate() {
                let rows = manifold.ambient_size();
                let cols = manifold.tangent_size();
                let ambient = self.ambient_offsets[i];
                if !manifold.plus_jacobian(&x[ambient..], buffer) {
                    return false;
                }

                let tangent = self.tangent_offsets[i];
                copy_block(
                    buffer,
                    rows,
                    cols,
                    jacobian,
                    self.tangent_size,
                    ambient,
                    tangent,
                );
            }
            true
        })
    }

    fn minus(&self, y: &[f64], x: &[f64], y_minus_x: &mut [f64]) -> bool {
        for (i, manifold) in self.manifolds.iter().enumerate() {
            let ambient = self.ambient_offsets[i];
            let tangent = self.tangent_offsets[i];
            if !manifold.minus(&y[ambient..], &x[ambient..], &mut y_minus_x[tangent..]) {
                return false;
            }
        }
        true
    }

    fn minus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
        jacobian[..self.tangent_size * self.ambient_size].fill(0.0);
        with_scratch(self.buffer_size, |buffer| {
            for (i, manifold) in self.manifolds.iter().enumerate() {
                let rows = manifold.tangent_size();
                let cols = manifold.ambient_size();
                let ambient = self.ambient_offsets[i];
                if !manifold.minus_jacobian(&x[ambient..], buffer) {
                    return false;
                }

                let tangent = self.tangent_offsets[i];
                copy_block(
                    buffer,
                    rows,
                    cols,
                    jacobian,
                    self.ambient_size,
                    tangent,
                    ambient,
                );
            }
            true
        })
    }
}
