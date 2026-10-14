//! Port of COLMAP's `src/colmap/estimators/`: the model estimators RANSAC runs and the
//! robust estimation entry points built on them.
//!
//! - [`cost_functions`]: the cost functions and manifolds of `estimators/cost_functions/`.
//! - [`solvers`]: the minimal and least-squares solvers of `estimators/solvers/`.
//! - [`fundamental_matrix_degensac`], [`fundamental_matrix_degensac_estimator`]: DEGENSAC
//!   fundamental matrix estimation, robust to a dominant plane (`fundamental_matrix_degensac.h`).
//! - [`triangulation`]: robust multi-view point triangulation (`triangulation.h`).
//!
//! Tests: `colmap-rust/tests/estimators.rs`.

pub mod cost_functions;
pub mod fundamental_matrix_degensac;
pub mod fundamental_matrix_degensac_estimator;
pub mod solvers;
pub mod triangulation;
