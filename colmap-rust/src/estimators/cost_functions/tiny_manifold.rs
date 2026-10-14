//! Port of COLMAP's `colmap/estimators/cost_functions/tiny_manifold.h`, via colmap-sharp's
//! `Estimators/CostFunctions/TinyManifold.cs`: the manifold policies for
//! [`crate::optim::TinySolver`], mirroring the Ceres manifolds used elsewhere in COLMAP
//! (`cost_functions/manifold.h`, ported with the solver phase). Each implements
//! [`TinyManifold`] (declared in `optim/tiny_solver.rs` next to [`TinyEuclideanManifold`], as
//! `tiny_solver.h` declares `EuclideanManifold`). The solver evaluates them on doubles only;
//! it never differentiates through a manifold. Tests: `tests/estimators/tiny_manifold.rs`
//! (`tiny_manifold_test.cc`).
//!
//! Translation notes:
//! - Every C++ name gains a `Tiny` prefix (`colmap::SphereManifold` -> [`TinySphereManifold`])
//!   so they don't collide with the Ceres manifolds of the solver phase, which have the same
//!   names in C++ under namespace `ceres`; `TinyEuclideanManifold` set the pattern.
//! - C++'s variadic `ProductManifold<Manifolds...>` becomes the binary
//!   [`TinyProductManifold<H, R>`], nested for longer products
//!   (`TinyProductManifold<A, TinyProductManifold<B, C>>` is C++'s `ProductManifold<A, B, C>`,
//!   exactly as its recursion unrolls). C++'s single-manifold product is a thin wrapper with
//!   the manifold's own behavior, so in Rust it is just the manifold.
//! - [`TinySphereManifold`] is C++'s `SphereManifold<3>`, the only size COLMAP implements.
//!   Its tangent basis uses Hughes & Moller's unit-orthogonal vector, not Eigen's
//!   `unitOrthogonal()` (docs/CPP_DIVERGENCES.md, entry 251).
//! - C++'s compile-time `kAmbientSize`/`kTangentSize` are the trait's sizes; the product's
//!   block scratch lives on the stack (a fixed 64-double buffer per block, ample for every
//!   COLMAP product), so the solver stays allocation-free.

use super::quaternion_utils::{eigen_quaternion_from_angle_axis, quaternion_left_mult_matrix};
use crate::linalg::{Quaterniond, Vector3d};
use crate::optim::tiny_solver::TinyManifold;

#[cfg(doc)]
use crate::optim::tiny_solver::TinyEuclideanManifold;

/// Port of `colmap::EigenQuaternionManifold`: SO(3) for an Eigen (x, y, z, w) quaternion.
/// Retraction q_plus = normalize(q * ExpMap(delta)) (right multiplication), matching how the
/// previous tangent parameterization composed the increment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TinyEigenQuaternionManifold;

impl TinyManifold for TinyEigenQuaternionManifold {
    fn ambient_size(&self) -> usize {
        4
    }
    fn tangent_size(&self) -> usize {
        3
    }
    fn is_euclidean(&self) -> bool {
        false
    }
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) {
        let mut delta_q = [0.0; 4];
        eigen_quaternion_from_angle_axis(delta, &mut delta_q);
        // Eigen::Map<const Quaterniond>: coefficients stored x, y, z, w.
        let q = Quaterniond::new(x[3], x[0], x[1], x[2]);
        let dq = Quaterniond::new(delta_q[3], delta_q[0], delta_q[1], delta_q[2]);
        let out = (q * dq).normalized();
        x_plus_delta[0] = out.x;
        x_plus_delta[1] = out.y;
        x_plus_delta[2] = out.z;
        x_plus_delta[3] = out.w;
    }
    /// For the right-multiplication retraction q * ExpMap(delta), the Jacobian at delta = 0
    /// is 0.5 * L(q) restricted to the imaginary columns, where L(q) is the quaternion
    /// left-multiplication matrix. Row-major 4x3.
    fn plus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) {
        let left = quaternion_left_mult_matrix(Quaterniond::new(x[3], x[0], x[1], x[2]));
        for r in 0..4 {
            for c in 0..3 {
                jacobian[r * 3 + c] = 0.5 * left[(r, c)];
            }
        }
    }
}

/// Port of `colmap::SphereManifold<3>`: the unit 2-sphere in R^3. Retraction
/// x_plus = normalize(x + B * delta), where B is an orthonormal basis of the tangent plane at
/// x / ||x||. Only the 2-sphere is implemented, which is all COLMAP needs; the closed-form
/// basis keeps the solver allocation-free.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TinySphereManifold;

impl TinySphereManifold {
    /// Orthonormal basis (the two columns of a 3 x 2 matrix) of the tangent plane at
    /// x / ||x||: b1 a unit vector orthogonal to x_hat, b2 = x_hat x b1.
    fn tangent_basis(x: Vector3d) -> (Vector3d, Vector3d) {
        let x_hat = x.normalized();
        let b1 = unit_orthogonal(x_hat);
        (b1, x_hat.cross(b1))
    }
}

/// A unit vector orthogonal to `v`, by Hughes & Moller, "Building an Orthonormal Basis from a
/// Unit Vector", Journal of Graphics Tools 4(4), 1999: zero the component of smallest
/// magnitude (the first one on a tie), swap the other two and negate one. The other two
/// cannot both be zero for a nonzero v, so the result is well defined. Stands in for Eigen's
/// `unitOrthogonal()`, whose choice for 3-vectors Eigen does not document (divergence 251).
fn unit_orthogonal(v: Vector3d) -> Vector3d {
    let ax = v.x.abs();
    let ay = v.y.abs();
    let az = v.z.abs();
    let perpendicular = if ax <= ay && ax <= az {
        Vector3d::new(0.0, -v.z, v.y)
    } else if ay <= az {
        Vector3d::new(-v.z, 0.0, v.x)
    } else {
        Vector3d::new(-v.y, v.x, 0.0)
    };
    perpendicular.normalized()
}

impl TinyManifold for TinySphereManifold {
    fn ambient_size(&self) -> usize {
        3
    }
    fn tangent_size(&self) -> usize {
        2
    }
    fn is_euclidean(&self) -> bool {
        false
    }
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) {
        let xv = Vector3d::new(x[0], x[1], x[2]);
        let (b1, b2) = Self::tangent_basis(xv);
        // xv + B * d, with B * d summed left to right per row as Eigen's 3x2 product does.
        let out = Vector3d::new(
            xv.x + (b1.x * delta[0] + b2.x * delta[1]),
            xv.y + (b1.y * delta[0] + b2.y * delta[1]),
            xv.z + (b1.z * delta[0] + b2.z * delta[1]),
        )
        .normalized();
        x_plus_delta[0] = out.x;
        x_plus_delta[1] = out.y;
        x_plus_delta[2] = out.z;
    }
    /// The tangent basis B as a row-major 3x2 matrix. Assumes x is (approximately) unit
    /// norm, as maintained by the solver's sphere retraction; the tangent basis equals the
    /// Plus Jacobian only at unit x.
    fn plus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) {
        let (b1, b2) = Self::tangent_basis(Vector3d::new(x[0], x[1], x[2]));
        let (b1, b2) = ([b1.x, b1.y, b1.z], [b2.x, b2.y, b2.z]);
        for r in 0..3 {
            jacobian[r * 2] = b1[r];
            jacobian[r * 2 + 1] = b2[r];
        }
    }
}

/// Stack scratch per product block; larger blocks (none in COLMAP) fall back to the heap.
const BLOCK_SCRATCH: usize = 64;

/// Runs `f` on a zeroed scratch buffer of `len` doubles.
fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f64]) -> R) -> R {
    if len <= BLOCK_SCRATCH {
        let mut stack = [0.0; BLOCK_SCRATCH];
        f(&mut stack[..len])
    } else {
        f(&mut vec![0.0; len])
    }
}

/// Port of `colmap::ProductManifold<Head, Tail...>` (the recursive case): `head` and `rest`
/// act on adjacent parameter blocks, laid out in argument order, with a block-diagonal Plus
/// Jacobian. Nest it for longer products, e.g.
/// `TinyProductManifold<TinyEigenQuaternionManifold, TinyProductManifold<TinySphereManifold,
/// TinyEuclideanManifold>>` for a relative pose plus a shared focal length.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TinyProductManifold<H: TinyManifold, R: TinyManifold> {
    /// The first block's manifold.
    pub head: H,
    /// The manifold of the remaining blocks.
    pub rest: R,
}

impl<H: TinyManifold, R: TinyManifold> TinyProductManifold<H, R> {
    /// The product of `head` and `rest`.
    pub const fn new(head: H, rest: R) -> Self {
        Self { head, rest }
    }
}

impl<H: TinyManifold, R: TinyManifold> TinyManifold for TinyProductManifold<H, R> {
    fn ambient_size(&self) -> usize {
        self.head.ambient_size() + self.rest.ambient_size()
    }
    fn tangent_size(&self) -> usize {
        self.head.tangent_size() + self.rest.tangent_size()
    }
    fn is_euclidean(&self) -> bool {
        self.head.is_euclidean() && self.rest.is_euclidean()
    }
    fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) {
        let head_ambient = self.head.ambient_size();
        let head_tangent = self.head.tangent_size();
        self.head.plus(x, delta, x_plus_delta);
        self.rest.plus(
            &x[head_ambient..],
            &delta[head_tangent..],
            &mut x_plus_delta[head_ambient..],
        );
    }
    /// Row-major (ambient x tangent), block-diagonal: the head block sits in the top-left,
    /// the rest block in the bottom-right. (C++ stores single-tangent blocks column-major,
    /// which for one column is the same layout.)
    fn plus_jacobian(&self, x: &[f64], jacobian: &mut [f64]) {
        let tangent = self.tangent_size();
        let head_ambient = self.head.ambient_size();
        let head_tangent = self.head.tangent_size();
        let rest_ambient = self.rest.ambient_size();
        let rest_tangent = self.rest.tangent_size();
        jacobian[..self.ambient_size() * tangent].fill(0.0);

        with_scratch(head_ambient * head_tangent, |head_jacobian| {
            self.head.plus_jacobian(x, head_jacobian);
            for r in 0..head_ambient {
                for c in 0..head_tangent {
                    jacobian[r * tangent + c] = head_jacobian[r * head_tangent + c];
                }
            }
        });

        with_scratch(rest_ambient * rest_tangent, |rest_jacobian| {
            self.rest.plus_jacobian(&x[head_ambient..], rest_jacobian);
            for r in 0..rest_ambient {
                for c in 0..rest_tangent {
                    jacobian[(head_ambient + r) * tangent + head_tangent + c] =
                        rest_jacobian[r * rest_tangent + c];
                }
            }
        });
    }
}
