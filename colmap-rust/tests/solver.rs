// Integration-test binary for the Ceres replacement, `colmap_rust::solver` (layout
// convention: see the header of `tests/math.rs`). COLMAP has no tests of Ceres itself, so
// the files here are rust_only ports of colmap-sharp's C#-only `ColmapSharp.Tests/Solver/`
// tests; the COLMAP tests that run through the solver (tiny_solver_test.cc,
// tiny_sampson_error_test.cc) live in their own modules' binaries.
//
// Run: `cargo test -p colmap-rust --test solver`.

#[path = "solver/rust_only_autodiff_cost_function.rs"]
mod rust_only_autodiff_cost_function;
#[path = "solver/rust_only_jet.rs"]
mod rust_only_jet;
