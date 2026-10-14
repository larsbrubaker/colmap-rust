//! The Ceres replacement: forward-mode automatic differentiation (Jets), Ceres' rotation
//! conversions, and the cost-function contract COLMAP's refinements are written against. Ceres
//! itself is not ported; this is a port of colmap-sharp's own `ColmapSharp/Solver/` (our MIT
//! code, validated against pycolmap), which follows Ceres Solver 2.2.0 (BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md) where COLMAP's numbers depend on Ceres' rules.
//!
//! Layout:
//! - `scalar.rs`: [`Scalar`], the trait COLMAP's `template <typename T>` numeric code is
//!   generic over (`f64` and [`Jet`]), with `std::fmin` / `std::fmax` ([`fmin`], [`fmax`]);
//! - `jet.rs`: [`Jet<N>`], `ceres::Jet<double, N>` with a const-generic width;
//! - `rotation.rs`: the `ceres/rotation.h` conversions COLMAP calls, over [`Scalar`];
//! - `autodiff_cost_function.rs`: [`CostFunction`] (`ceres::CostFunction`, object safe),
//!   [`AutoDiffFunctor`] (a functor's templated `operator()`), and
//!   [`AutoDiffCostFunction<F, R, N>`];
//! - `tiny_solver_autodiff_function.rs`: [`TinySolverAutoDiffFunction<F, N>`], an autodiff
//!   functor as a [`crate::optim::TinySolverFunction`].
//!
//! Users: the camera models (`sensor::models`, generic over [`Scalar`]; the iterative
//! undistortion differentiates on `Jet<2>`), the quaternion utilities and Sampson cost
//! functors of `estimators::cost_functions`.
//!
//! Tests: `colmap-rust/tests/solver.rs` (Jet rules and the autodiff cost function, ported
//! from colmap-sharp's C#-only `JetTests` and `AutoDiffCostFunctionTests`).

pub mod autodiff_cost_function;
pub mod jet;
pub mod rotation;
pub mod scalar;
pub mod tiny_solver_autodiff_function;

pub use autodiff_cost_function::{AutoDiffCostFunction, AutoDiffFunctor, CostFunction};
pub use jet::Jet;
pub use scalar::{fmax, fmin, Scalar};
pub use tiny_solver_autodiff_function::TinySolverAutoDiffFunction;
