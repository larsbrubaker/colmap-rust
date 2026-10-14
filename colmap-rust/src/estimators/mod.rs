//! Port of COLMAP's `src/colmap/estimators/`: the model estimators RANSAC runs and the
//! robust estimation entry points built on them.
//!
//! - [`alignment`]: aligning and merging reconstructions and measuring alignment errors
//!   (`alignment.h`).
//! - [`cost_functions`]: the cost functions and manifolds of `estimators/cost_functions/`.
//! - [`solvers`]: the minimal and least-squares solvers of `estimators/solvers/`.
//! - [`fundamental_matrix_degensac`], [`fundamental_matrix_degensac_estimator`]: DEGENSAC
//!   fundamental matrix estimation, robust to a dominant plane (`fundamental_matrix_degensac.h`).
//! - [`generalized_pose`]: robust pose estimation for multi-camera rigs
//!   (`generalized_pose.h`, without the Ceres refiner).
//! - [`pose`]: robust absolute and relative pose estimation (`pose.h`, without the Ceres
//!   refiners).
//! - [`rotation_averaging`]: global rotation averaging over a pose graph
//!   (`rotation_averaging.h`, `rotation_averaging_impl.h`).
//! - [`triangulation`]: robust multi-view point triangulation (`triangulation.h`).
//! - [`two_view_geometry`]: robust two-view geometry verification of an image pair or of two
//!   rigs, and the relative pose decomposition (`two_view_geometry.h`).
//!
//! Tests: `colmap-rust/tests/estimators.rs`.

pub mod alignment;
pub mod cost_functions;
pub mod fundamental_matrix_degensac;
pub mod fundamental_matrix_degensac_estimator;
pub mod generalized_pose;
pub mod pose;
pub mod rotation_averaging;
pub mod solvers;
pub mod triangulation;
pub mod two_view_geometry;
