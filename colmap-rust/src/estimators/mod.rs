//! Port of COLMAP's `src/colmap/estimators/`: the model estimators RANSAC runs and the
//! robust estimation entry points built on them.
//!
//! - [`cost_functions`]: the cost functions and manifolds of `estimators/cost_functions/`.
//! - [`solvers`]: the minimal and least-squares solvers of `estimators/solvers/`.
//! - [`triangulation`]: robust multi-view point triangulation (`triangulation.h`).
//!
//! Tests: `colmap-rust/tests/estimators.rs`.

pub mod cost_functions;
pub mod solvers;
pub mod triangulation;
