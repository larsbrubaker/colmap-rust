//! Port of COLMAP's `src/colmap/estimators/cost_functions/`: the cost functions and
//! manifolds the estimators' refinements minimize.
//!
//! - [`quaternion_utils`]: quaternion multiplication matrices, rotation with Jacobian, and
//!   the Eigen-order angle-axis conversions (`quaternion_utils.h`).
//! - [`tiny_manifold`]: the manifold policies for `optim::TinySolver` (`tiny_manifold.h`).
//! - [`tiny_sampson_error`]: closed-form Sampson-error functors for `optim::TinySolver`
//!   (`tiny_sampson_error.h`).
//!
//! The Ceres cost functors (`sampson_error.h`, `reprojection_error.h`, `manifold.h`, ...)
//! arrive with the solver and bundle adjustment phases.

pub mod quaternion_utils;
pub mod tiny_manifold;
pub mod tiny_sampson_error;

pub use tiny_manifold::{TinyEigenQuaternionManifold, TinyProductManifold, TinySphereManifold};
pub use tiny_sampson_error::{
    TinyFundamentalSampsonErrorCostFunctor, TinyOneSidedFocalTangentSampsonErrorCostFunctor,
    TinyTangentSampsonErrorCostFunctor,
};
