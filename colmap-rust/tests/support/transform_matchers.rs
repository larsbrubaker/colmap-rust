// COLMAP's `Rigid3dNear` (geometry/rigid3_matchers.h) and `Sim3dNear`
// (geometry/sim3_matchers.h) gmock matchers as predicates (test support, not a test), shared by
// the `geometry` and `estimators` test binaries. Port of colmap-sharp's
// `ColmapSharp.Tests/Rigid3dMatchers.cs` and `Sim3dMatchers.cs`.
//
// Faithful to an upstream slip, as colmap-sharp is: COLMAP's Sim3dNearMatcher initializes its
// scale tolerance from rtol (`stol_(rtol)`), so `stol` is ignored and the scale is compared
// against `rtol`.
//
// Include it from a test binary with `#[path = "support/transform_matchers.rs"] mod
// transform_matchers;`.

#![allow(dead_code)]

use colmap_rust::geometry::{Rigid3d, Sim3d};
use colmap_rust::linalg::Vector3d;

/// The translation half of both matchers: `norm() <= ttol` when `rhs` is zero by Eigen's
/// `isZero()` (every `|coefficient| <= 1e-12`; "isApprox() is not well-defined for zero
/// matrices"), Eigen's `isApprox(rhs, ttol)` otherwise.
fn translation_near(lhs: Vector3d, rhs: Vector3d, ttol: f64) -> bool {
    if rhs.to_array().iter().all(|c| c.abs() <= 1e-12) {
        lhs.norm() <= ttol
    } else {
        lhs.is_approx_with(rhs, ttol)
    }
}

/// Port of COLMAP's `Rigid3dNear(rhs, rtol, ttol)` matcher: the rotations within `rtol`
/// angular distance and the translations within `ttol` (see [`translation_near`]).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // !(a <= b) is deliberate: NaN must fail.
pub fn rigid3d_near(lhs: &Rigid3d, rhs: &Rigid3d, rtol: f64, ttol: f64) -> bool {
    // Note the !(a <= b) form, as in COLMAP, to reject NaNs.
    if !(lhs.rotation.angular_distance(rhs.rotation) <= rtol) {
        return false;
    }
    translation_near(lhs.translation, rhs.translation, ttol)
}

/// Port of COLMAP's `Sim3dNear(rhs, stol, rtol, ttol)` matcher: the scales within the scale
/// tolerance (which is `rtol`, see the file header), the rotations within `rtol` angular
/// distance and the translations within `ttol` (see [`translation_near`]).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // !(a <= b) is deliberate: NaN must fail.
pub fn sim3d_near(lhs: &Sim3d, rhs: &Sim3d, stol: f64, rtol: f64, ttol: f64) -> bool {
    // COLMAP's matcher stores stol_(rtol); see the file header.
    let _ = stol;
    let scale_tolerance = rtol;
    // Note the !(a <= b) form, as in COLMAP, to reject NaNs.
    if !((lhs.scale - rhs.scale).abs() <= scale_tolerance) {
        return false;
    }
    if !(lhs.rotation.angular_distance(rhs.rotation) <= rtol) {
        return false;
    }
    translation_near(lhs.translation, rhs.translation, ttol)
}
