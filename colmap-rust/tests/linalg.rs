// Integration-test binary for `colmap_rust::linalg`, the Eigen replacement: fixed- and
// dynamic-size types and the dense decompositions.
// COLMAP has no test for Eigen itself, so every file here is Rust-only (`rust_only_*`), a port
// of colmap-sharp's C#-only tests (`ColmapSharp.Tests/LinearAlgebra/`), plus the pycolmap
// oracle comparisons (`rust_only_rotation_oracle.rs`, `rust_only_decomposition_oracle.rs`,
// `rust_only_spectral_oracle.rs`; fixtures `tests/data/oracle/linear_algebra_{rotations,dense,
// spectral}.json` from the matching `oracle/linear_algebra_*.py`).
// Layout convention: see `tests/math.rs`.
//
// Run: `cargo test -p colmap-rust --test linalg`.

#[path = "support/oracle_json.rs"]
mod oracle_json;

#[path = "linalg/rust_only_aligned_box.rs"]
mod rust_only_aligned_box;
#[path = "linalg/rust_only_decomposition.rs"]
mod rust_only_decomposition;
#[path = "linalg/rust_only_decomposition_oracle.rs"]
mod rust_only_decomposition_oracle;
#[path = "linalg/rust_only_dynamic_matrix.rs"]
mod rust_only_dynamic_matrix;
#[path = "linalg/rust_only_matrix.rs"]
mod rust_only_matrix;
#[path = "linalg/rust_only_quaternion.rs"]
mod rust_only_quaternion;
#[path = "linalg/rust_only_rotation_oracle.rs"]
mod rust_only_rotation_oracle;
#[path = "linalg/rust_only_spectral_eigen.rs"]
mod rust_only_spectral_eigen;
#[path = "linalg/rust_only_spectral_oracle.rs"]
mod rust_only_spectral_oracle;
#[path = "linalg/rust_only_spectral_svd.rs"]
mod rust_only_spectral_svd;
#[path = "linalg/rust_only_vector.rs"]
mod rust_only_vector;
