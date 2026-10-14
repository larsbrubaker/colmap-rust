// Integration-test binary for the Ceres replacement, `colmap_rust::solver` (layout
// convention: see the header of `tests/math.rs`). COLMAP has no tests of Ceres itself, so
// the files here are Ceres' own tests (ported 1:1 via colmap-sharp) and rust_only ports of
// colmap-sharp's C#-only `ColmapSharp.Tests/Solver/` tests; the COLMAP tests that run
// through the solver (tiny_solver_test.cc, tiny_sampson_error_test.cc) live in their own
// modules' binaries.
//
// Run: `cargo test -p colmap-rust --test solver`.

#[path = "support/net_random.rs"]
mod net_random;

#[path = "solver/ceres_polynomial.rs"]
mod ceres_polynomial;
#[path = "solver/corrector.rs"]
mod corrector;
#[path = "solver/loss_function.rs"]
mod loss_function;
#[path = "solver/ordered_groups.rs"]
mod ordered_groups;
#[path = "solver/parameter_block.rs"]
mod parameter_block;
#[path = "solver/problem.rs"]
mod problem;
#[path = "solver/problem_evaluate.rs"]
mod problem_evaluate;
#[path = "solver/program.rs"]
mod program;
#[path = "solver/residual_block.rs"]
mod residual_block;
#[path = "solver/rust_only_autodiff_cost_function.rs"]
mod rust_only_autodiff_cost_function;
#[path = "solver/rust_only_block_random_access_matrix.rs"]
mod rust_only_block_random_access_matrix;
#[path = "solver/rust_only_jet.rs"]
mod rust_only_jet;
#[path = "solver/rust_only_manifold.rs"]
mod rust_only_manifold;
#[path = "solver/rust_only_program_evaluator.rs"]
mod rust_only_program_evaluator;
#[path = "solver/rust_only_small_blas.rs"]
mod rust_only_small_blas;
#[path = "solver/rust_only_small_blas_fixed.rs"]
mod rust_only_small_blas_fixed;
#[path = "solver/rust_only_sparse_matrix.rs"]
mod rust_only_sparse_matrix;
