// Integration-test binary for COLMAP's `math/` module.
//
// Test layout convention (applies to every module): Cargo builds one test binary per file
// directly in `tests/`, so each COLMAP module directory gets one top-level file,
// `tests/<module>.rs`, whose `#[path = "<module>/<file>.rs"] mod <file>;` lines pull in
// `tests/<module>/<file>.rs` (a test crate root resolves plain `mod` next to itself, hence
// the explicit path; Cargo does not build files in subdirectories as test binaries) — one
// file per ported `src/colmap/<module>/<file>_test.cc`, with the same test names in
// snake_case (`TEST(Rigid3d, Inverse)` -> `fn rigid3d_inverse()`). Rust-only files and
// tests are named `rust_only_*` / carry a `rust_only_` prefix on each test. Shared test data
// lives in `tests/data/` (oracle fixtures in `tests/data/oracle/`).
//
// Run one module: `cargo test -p colmap-rust --test math`.

#[path = "math/fns_probe.rs"]
mod fns_probe;

#[path = "math/connected_components.rs"]
mod connected_components;
#[path = "math/math.rs"]
mod math;
#[path = "math/random.rs"]
mod random;
#[path = "math/rust_only_random_oracle.rs"]
mod rust_only_random_oracle;
#[path = "math/spanning_tree.rs"]
mod spanning_tree;
#[path = "math/union_find.rs"]
mod union_find;
