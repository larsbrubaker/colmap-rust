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
//!   functor as a [`crate::optim::TinySolverFunction`];
//! - `small_blas.rs`, `small_blas_fixed.rs`: Ceres' small dense row-major kernels
//!   ([`small_blas::Operation`]) and PSD block inversion, with const-generic unrolled kernels
//!   for the bundle adjustment shapes that round exactly as the loops do;
//! - `sparse_matrix.rs`: [`LinearOperator`], [`SparseMatrix`] (the minimizer's view of a
//!   Jacobian) and [`DenseSparseMatrix`];
//! - `block_structure.rs`, `block_sparse_matrix.rs`: the block-sparse Jacobian ([`Block`],
//!   [`Cell`], [`CompressedRow`], [`CompressedRowBlockStructure`], [`BlockSparseMatrix`]);
//! - `block_random_access_matrix.rs`, `block_random_access_sparse_matrix.rs`: the reduced
//!   camera matrix the Schur solvers accumulate into ([`BlockRandomAccessMatrix`], dense,
//!   sparse and diagonal).
//!
//! Users: the camera models (`sensor::models`, generic over [`Scalar`]; the iterative
//! undistortion differentiates on `Jet<2>`), the quaternion utilities and Sampson cost
//! functors of `estimators::cost_functions`.
//!
//! Tests: `colmap-rust/tests/solver.rs` (Jet rules and the autodiff cost function, ported
//! from colmap-sharp's C#-only `JetTests` and `AutoDiffCostFunctionTests`).
//! The matrix tests there port Ceres' `small_blas_test.cc` and
//! `block_random_access_{dense,sparse,diagonal}_matrix_test.cc` via colmap-sharp.

pub mod autodiff_cost_function;
pub mod block_random_access_matrix;
pub mod block_random_access_sparse_matrix;
pub mod block_sparse_matrix;
pub mod block_structure;
pub mod jet;
pub mod rotation;
pub mod scalar;
pub mod small_blas;
pub mod small_blas_fixed;
pub mod sparse_matrix;
pub mod tiny_solver_autodiff_function;

pub use autodiff_cost_function::{AutoDiffCostFunction, AutoDiffFunctor, CostFunction};
pub use block_random_access_matrix::{
    BlockRandomAccessDenseMatrix, BlockRandomAccessDiagonalMatrix, BlockRandomAccessMatrix,
    CellLocation,
};
pub use block_random_access_sparse_matrix::BlockRandomAccessSparseMatrix;
pub use block_sparse_matrix::BlockSparseMatrix;
pub use block_structure::{Block, Cell, CompressedRow, CompressedRowBlockStructure};
pub use jet::Jet;
pub use scalar::{fmax, fmin, Scalar};
pub use sparse_matrix::{DenseSparseMatrix, LinearOperator, SparseMatrix};
pub use tiny_solver_autodiff_function::TinySolverAutoDiffFunction;
