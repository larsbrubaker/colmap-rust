//! Ceres' manifolds: they let the solver optimize a parameter block that lives on a curved
//! space (a unit quaternion, a unit direction) or has frozen coordinates, through a
//! tangent-space step x' = Plus(x, delta). Port of colmap-sharp's
//! `ColmapSharp/Solver/Manifolds.cs`, which ports Ceres Solver 2.2.0's
//! `include/ceres/manifold.h` and `internal/ceres/manifold.cc` (BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md). This file holds the [`Manifold`] trait and the simple manifolds;
//! `sphere_manifold.rs` and `product_manifold.rs` hold the composite ones (colmap-sharp's
//! `SphereProductManifolds.cs`).
//!
//! COLMAP uses exactly these, through `colmap/estimators/cost_functions/manifold.h`:
//! `EuclideanManifold<3>` for positions, [`EigenQuaternionManifold`] for rotations (stored
//! x, y, z, w), [`SubsetManifold`] for constant camera parameters, `SphereManifold<3>` for a
//! unit baseline, and `ProductManifold` to glue a quaternion to a translation or a sphere.
//! [`QuaternionManifold`] (Ceres' w, x, y, z order) shares the code and is kept for
//! `solver/rotation.rs`' order.
//!
//! Relation to `estimators/cost_functions/tiny_manifold.rs`: those are COLMAP's own manifolds
//! for `TinySolver`, with different math (a right-multiplied, renormalized quaternion step
//! with a 0.5 L(q) Jacobian; a renormalized sphere step through a Hughes-Moller basis), no
//! `Minus`, and static dispatch. Building either family on the other would change one
//! family's numbers, so they share nothing beyond the idea.
//!
//! Translation notes:
//! - Jacobians are row-major, as Ceres': `plus_jacobian` is ambient x tangent,
//!   `minus_jacobian` tangent x ambient.
//! - The trait is object safe and `Send + Sync`, so a problem can hold `Box<dyn Manifold>`.
//! - Ceres' compile-time `EuclideanManifold<N>` / `SphereManifold<N>` sizes are runtime sizes
//!   (Ceres' `DYNAMIC` forms, which compute the same values).
//! - Vector norms sum squares left to right, a deliberate divergence. Ceres evaluates the
//!   norms in its manifolds through Eigen (`delta.norm()`, `x.norm()`, and in
//!   `sphere_manifold.rs` the Householder vector's `squaredNorm()` and dot product), which
//!   Eigen may vectorize and reduce in packet lanes; here the squares are summed left to right
//!   in coefficient order. The quaternion manifolds' norms are `std::hypot`'s three-argument
//!   form, which is followed exactly (`math::fns::hypot3`). Results can differ from COLMAP's
//!   Ceres in the last bits. Eigen is MPL-2.0 and not ported (contract rule 2), and its packet
//!   order depends on the target's SIMD width and the compiler, so there is no single order to
//!   match; a sequential order gives the same bits on every target, native and wasm. Tier C,
//!   like the solver that consumes them. Evidence: `tests/solver/rust_only_manifold.rs`
//!   (colmap-sharp's `ManifoldTests`, plus the sphere's poles) checks Plus/Minus round trips to
//!   1e-12 and the Jacobians against finite differences to 1e-8. Same as colmap-sharp entry 115
//!   (its `Solver/Manifolds.cs` part).
//!
//! Tests: `tests/solver/rust_only_manifold.rs` (colmap-sharp's C#-only `ManifoldTests`).

use crate::math::fns;
use crate::{check, Result};

/// Port of `ceres::Manifold`: a parameter block's space and its tangent-space step.
pub trait Manifold: Send + Sync {
    /// Dimension of the ambient space the parameter block is stored in.
    fn ambient_size(&self) -> usize;

    /// Dimension of the tangent space the solver steps in.
    fn tangent_size(&self) -> usize;

    /// `x_plus_delta = Plus(x, delta)`; false if the step is invalid.
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool;

    /// The row-major `ambient x tangent` Jacobian of `Plus(x, delta)` at delta = 0.
    fn plus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool;

    /// `y_minus_x = Minus(y, x)`, the inverse of Plus: `Plus(x, Minus(y, x)) = y`.
    fn minus(&self, y: &[f64], x: &[f64], y_minus_x: &mut [f64]) -> bool;

    /// The row-major `tangent x ambient` Jacobian of `Minus(y, x)` at y = x.
    fn minus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool;

    /// `tangent_matrix = ambient_matrix * PlusJacobian(x)`, both row-major with `num_rows`
    /// rows: how the solver turns a cost function's ambient Jacobian into a tangent one.
    fn right_multiply_by_plus_jacobian(
        &self,
        x: &[f64],
        num_rows: usize,
        ambient_matrix: &[f64],
        tangent_matrix: &mut [f64],
    ) -> bool {
        let tangent_size = self.tangent_size();
        if tangent_size == 0 {
            return true;
        }

        let ambient_size = self.ambient_size();
        with_scratch(ambient_size * tangent_size, |plus_jacobian| {
            if !self.plus_jacobian(x, plus_jacobian) {
                return false;
            }

            for r in 0..num_rows {
                for c in 0..tangent_size {
                    let mut sum = 0.0;
                    for k in 0..ambient_size {
                        sum += ambient_matrix[r * ambient_size + k]
                            * plus_jacobian[k * tangent_size + c];
                    }

                    tangent_matrix[r * tangent_size + c] = sum;
                }
            }

            true
        })
    }
}

/// Doubles of stack scratch; every manifold COLMAP builds fits (the largest Jacobian,
/// quaternion x sphere, is 7 x 5), so the solver's per-iteration calls don't allocate.
const STACK_SCRATCH: usize = 256;

/// Runs `f` on a zeroed scratch buffer of `len` doubles: on the stack up to
/// [`STACK_SCRATCH`], on the heap beyond.
pub(crate) fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
    if len <= STACK_SCRATCH {
        let mut stack = [0.0; STACK_SCRATCH];
        f(&mut stack[..len])
    } else {
        f(&mut vec![0.0; len])
    }
}

/// Writes the `n x n` identity to `matrix[..n * n]`.
fn set_identity(matrix: &mut [f64], n: usize) {
    let matrix = &mut matrix[..n * n];
    matrix.fill(0.0);
    for i in 0..n {
        matrix[i * n + i] = 1.0;
    }
}

/// Port of `ceres::EuclideanManifold`: R^n with Plus = +, Minus = -.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EuclideanManifold {
    size: usize,
}

impl EuclideanManifold {
    /// R^`size`. (Ceres' `CHECK_GE(size, 0)` is the type's.)
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Manifold for EuclideanManifold {
    fn ambient_size(&self) -> usize {
        self.size
    }

    fn tangent_size(&self) -> usize {
        self.size
    }

    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        for i in 0..self.size {
            x_plus_delta[i] = x[i] + delta[i];
        }
        true
    }

    fn plus_jacobian(&self, _x: &[f64], jacobian: &mut [f64]) -> bool {
        set_identity(jacobian, self.size);
        true
    }

    fn minus(&self, y: &[f64], x: &[f64], y_minus_x: &mut [f64]) -> bool {
        for i in 0..self.size {
            y_minus_x[i] = y[i] - x[i];
        }
        true
    }

    fn minus_jacobian(&self, _x: &[f64], jacobian: &mut [f64]) -> bool {
        set_identity(jacobian, self.size);
        true
    }

    fn right_multiply_by_plus_jacobian(
        &self,
        _x: &[f64],
        num_rows: usize,
        ambient_matrix: &[f64],
        tangent_matrix: &mut [f64],
    ) -> bool {
        let n = num_rows * self.size;
        tangent_matrix[..n].copy_from_slice(&ambient_matrix[..n]);
        true
    }
}

/// Port of `ceres::SubsetManifold`: R^n with some coordinates held constant; the tangent
/// space is the remaining coordinates, in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SubsetManifold {
    tangent_size: usize,
    constancy_mask: Vec<bool>,
}

impl SubsetManifold {
    /// The manifold of size `size` whose `constant_parameters` (distinct indices below
    /// `size`) never move. Ceres CHECK-fails on an out-of-range or repeated index; the port
    /// returns the check's error. (Its "greater than equal to zero" check is the type's.)
    pub fn new(size: usize, constant_parameters: &[usize]) -> Result<Self> {
        let mut constancy_mask = vec![false; size];
        if constant_parameters.is_empty() {
            return Ok(Self {
                tangent_size: size,
                constancy_mask,
            });
        }

        let mut constant = constant_parameters.to_vec();
        constant.sort_unstable();
        check!(
            constant[constant.len() - 1] < size,
            "Indices indicating constant parameter must be less than the size of the parameter block."
        );
        check!(
            constant.windows(2).all(|pair| pair[0] != pair[1]),
            "The set of constant parameters cannot contain duplicates"
        );
        for &index in constant_parameters {
            constancy_mask[index] = true;
        }

        Ok(Self {
            tangent_size: size - constant_parameters.len(),
            constancy_mask,
        })
    }
}

impl Manifold for SubsetManifold {
    fn ambient_size(&self) -> usize {
        self.constancy_mask.len()
    }

    fn tangent_size(&self) -> usize {
        self.tangent_size
    }

    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        let mut j = 0;
        for (i, &constant) in self.constancy_mask.iter().enumerate() {
            x_plus_delta[i] = if constant {
                x[i]
            } else {
                j += 1;
                x[i] + delta[j - 1]
            };
        }
        true
    }

    fn plus_jacobian(&self, _x: &[f64], jacobian: &mut [f64]) -> bool {
        let tangent_size = self.tangent_size;
        if tangent_size == 0 {
            return true;
        }

        jacobian[..self.constancy_mask.len() * tangent_size].fill(0.0);
        let mut c = 0;
        for (r, &constant) in self.constancy_mask.iter().enumerate() {
            if !constant {
                jacobian[r * tangent_size + c] = 1.0;
                c += 1;
            }
        }
        true
    }

    fn right_multiply_by_plus_jacobian(
        &self,
        _x: &[f64],
        num_rows: usize,
        ambient_matrix: &[f64],
        tangent_matrix: &mut [f64],
    ) -> bool {
        let tangent_size = self.tangent_size;
        if tangent_size == 0 {
            return true;
        }

        let ambient_size = self.constancy_mask.len();
        for r in 0..num_rows {
            let mut c = 0;
            for (idx, &constant) in self.constancy_mask.iter().enumerate() {
                if !constant {
                    tangent_matrix[r * tangent_size + c] = ambient_matrix[r * ambient_size + idx];
                    c += 1;
                }
            }
        }
        true
    }

    fn minus(&self, y: &[f64], x: &[f64], y_minus_x: &mut [f64]) -> bool {
        if self.tangent_size == 0 {
            return true;
        }

        let mut j = 0;
        for (i, &constant) in self.constancy_mask.iter().enumerate() {
            if !constant {
                y_minus_x[j] = y[i] - x[i];
                j += 1;
            }
        }
        true
    }

    fn minus_jacobian(&self, _x: &[f64], jacobian: &mut [f64]) -> bool {
        let ambient_size = self.constancy_mask.len();
        jacobian[..self.tangent_size * ambient_size].fill(0.0);
        let mut r = 0;
        for (c, &constant) in self.constancy_mask.iter().enumerate() {
            if !constant {
                jacobian[r * ambient_size + c] = 1.0;
                r += 1;
            }
        }
        true
    }
}

/// Where a quaternion manifold stores w, x, y and z.
#[derive(Clone, Copy)]
struct QuaternionLayout {
    w: usize,
    x: usize,
    y: usize,
    z: usize,
}

/// Ceres' order, [w, x, y, z].
const CERES_LAYOUT: QuaternionLayout = QuaternionLayout {
    w: 0,
    x: 1,
    y: 2,
    z: 3,
};

/// Eigen's storage order, [x, y, z, w].
const EIGEN_LAYOUT: QuaternionLayout = QuaternionLayout {
    w: 3,
    x: 0,
    y: 1,
    z: 2,
};

/// Ceres' `QuaternionPlusImpl`: x_plus_delta = QuaternionProduct(q_delta, x), where q_delta is
/// the quaternion constructed from delta, [cos|delta|, sin|delta| delta / |delta|].
fn quaternion_plus(o: QuaternionLayout, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) {
    let norm_delta = fns::hypot3(delta[0], delta[1], delta[2]);
    if norm_delta == 0.0 {
        // No change in rotation: return the quaternion as is.
        x_plus_delta[..4].copy_from_slice(&x[..4]);
        return;
    }

    let sin_delta_by_delta = fns::sin(norm_delta) / norm_delta;
    let q_w = fns::cos(norm_delta);
    let q_x = sin_delta_by_delta * delta[0];
    let q_y = sin_delta_by_delta * delta[1];
    let q_z = sin_delta_by_delta * delta[2];
    let (x_w, x_x, x_y, x_z) = (x[o.w], x[o.x], x[o.y], x[o.z]);

    x_plus_delta[o.w] = q_w * x_w - q_x * x_x - q_y * x_y - q_z * x_z;
    x_plus_delta[o.x] = q_w * x_x + q_x * x_w + q_y * x_z - q_z * x_y;
    x_plus_delta[o.y] = q_w * x_y - q_x * x_z + q_y * x_w + q_z * x_x;
    x_plus_delta[o.z] = q_w * x_z + q_x * x_y - q_y * x_x + q_z * x_w;
}

/// Ceres' `QuaternionPlusJacobianImpl`, row-major 4 x 3.
fn quaternion_plus_jacobian(o: QuaternionLayout, x: &[f64], jacobian: &mut [f64]) {
    jacobian[o.w * 3] = -x[o.x];
    jacobian[o.w * 3 + 1] = -x[o.y];
    jacobian[o.w * 3 + 2] = -x[o.z];
    jacobian[o.x * 3] = x[o.w];
    jacobian[o.x * 3 + 1] = x[o.z];
    jacobian[o.x * 3 + 2] = -x[o.y];
    jacobian[o.y * 3] = -x[o.z];
    jacobian[o.y * 3 + 1] = x[o.w];
    jacobian[o.y * 3 + 2] = x[o.x];
    jacobian[o.z * 3] = x[o.y];
    jacobian[o.z * 3 + 1] = -x[o.x];
    jacobian[o.z * 3 + 2] = x[o.w];
}

/// Ceres' `QuaternionMinusImpl`: ambient_y_minus_x = QuaternionProduct(y, -x), where -x is
/// the conjugate of x, mapped to the tangent space through its angle-axis.
fn quaternion_minus(o: QuaternionLayout, y: &[f64], x: &[f64], y_minus_x: &mut [f64]) {
    let (y_w, y_x, y_y, y_z) = (y[o.w], y[o.x], y[o.y], y[o.z]);
    let (x_w, x_x, x_y, x_z) = (x[o.w], x[o.x], x[o.y], x[o.z]);
    let a_w = y_w * x_w + y_x * x_x + y_y * x_y + y_z * x_z;
    let a_x = -y_w * x_x + y_x * x_w - y_y * x_z + y_z * x_y;
    let a_y = -y_w * x_y + y_x * x_z + y_y * x_w - y_z * x_x;
    let a_z = -y_w * x_z - y_x * x_y + y_y * x_x + y_z * x_w;

    let u_norm = fns::hypot3(a_x, a_y, a_z);
    if u_norm != 0.0 {
        let theta = fns::atan2(u_norm, a_w);
        y_minus_x[0] = theta * a_x / u_norm;
        y_minus_x[1] = theta * a_y / u_norm;
        y_minus_x[2] = theta * a_z / u_norm;
    } else {
        y_minus_x[..3].fill(0.0);
    }
}

/// Ceres' `QuaternionMinusJacobianImpl`, row-major 3 x 4.
fn quaternion_minus_jacobian(o: QuaternionLayout, x: &[f64], jacobian: &mut [f64]) {
    jacobian[o.w] = -x[o.x];
    jacobian[o.x] = x[o.w];
    jacobian[o.y] = -x[o.z];
    jacobian[o.z] = x[o.y];
    jacobian[4 + o.w] = -x[o.y];
    jacobian[4 + o.x] = x[o.z];
    jacobian[4 + o.y] = x[o.w];
    jacobian[4 + o.z] = -x[o.x];
    jacobian[8 + o.w] = -x[o.z];
    jacobian[8 + o.x] = -x[o.y];
    jacobian[8 + o.y] = x[o.x];
    jacobian[8 + o.z] = x[o.w];
}

/// Implements [`Manifold`] for a quaternion manifold with the given storage layout.
macro_rules! quaternion_manifold {
    ($name:ident, $layout:expr) => {
        impl Manifold for $name {
            fn ambient_size(&self) -> usize {
                4
            }

            fn tangent_size(&self) -> usize {
                3
            }

            fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
                quaternion_plus($layout, x, delta, x_plus_delta);
                true
            }

            fn plus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
                quaternion_plus_jacobian($layout, x, jacobian);
                true
            }

            fn minus(&self, y: &[f64], x: &[f64], y_minus_x: &mut [f64]) -> bool {
                quaternion_minus($layout, y, x, y_minus_x);
                true
            }

            fn minus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) -> bool {
                quaternion_minus_jacobian($layout, x, jacobian);
                true
            }
        }
    };
}

/// Port of `ceres::QuaternionManifold`: unit quaternions stored [w, x, y, z] (as
/// `ceres/rotation.h` expects), with the left-multiplicative step
/// Plus(x, delta) = [cos|delta|, sin|delta| delta / |delta|] * x.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct QuaternionManifold;

quaternion_manifold!(QuaternionManifold, CERES_LAYOUT);

/// Port of `ceres::EigenQuaternionManifold`: [`QuaternionManifold`]'s step on quaternions
/// stored [x, y, z, w], as Eigen and COLMAP's poses store them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EigenQuaternionManifold;

quaternion_manifold!(EigenQuaternionManifold, EIGEN_LAYOUT);
