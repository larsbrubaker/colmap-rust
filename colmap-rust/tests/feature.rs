// Integration-test binary for COLMAP's `feature/` module (layout convention: see the header of
// `tests/math.rs`): one file per colmap/feature/*_test.cc, ported 1:1, plus
// `feature/rust_only_feature_types.rs` (error paths and the documented divergences).
//
// Run: `cargo test -p colmap-rust --test feature`.

#[path = "feature/rust_only_feature_types.rs"]
mod rust_only_feature_types;
#[path = "feature/rust_only_feature_utils.rs"]
mod rust_only_feature_utils;
#[path = "feature/types.rs"]
mod types;
#[path = "feature/utils.rs"]
mod utils;
