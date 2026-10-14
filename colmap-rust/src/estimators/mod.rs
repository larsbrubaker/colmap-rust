//! Port of COLMAP's `src/colmap/estimators/`: the model estimators RANSAC runs and the
//! robust estimation entry points built on them.
//!
//! - [`solvers`]: the minimal and least-squares solvers of `estimators/solvers/`.
//!
//! Tests: `colmap-rust/tests/estimators.rs`.

pub mod solvers;
