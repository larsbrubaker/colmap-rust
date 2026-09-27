// Integration-test binary for `colmap_rust::linalg`, the Eigen replacement's fixed-size types.
// COLMAP has no test for Eigen itself, so every file here is Rust-only (`rust_only_*`), a port
// of colmap-sharp's C#-only tests (`ColmapSharp.Tests/LinearAlgebra/`), plus the pycolmap
// oracle comparison (`rust_only_rotation_oracle.rs`, fixture
// `tests/data/oracle/linear_algebra_rotations.json` from `oracle/linear_algebra_rotations.py`).
// Layout convention: see `tests/math.rs`.
//
// Run: `cargo test -p colmap-rust --test linalg`.

#[path = "support/oracle_json.rs"]
mod oracle_json;

#[path = "linalg/rust_only_aligned_box.rs"]
mod rust_only_aligned_box;
#[path = "linalg/rust_only_matrix.rs"]
mod rust_only_matrix;
#[path = "linalg/rust_only_quaternion.rs"]
mod rust_only_quaternion;
#[path = "linalg/rust_only_rotation_oracle.rs"]
mod rust_only_rotation_oracle;
#[path = "linalg/rust_only_vector.rs"]
mod rust_only_vector;
