// Integration-test binary for COLMAP's `feature/` module (layout convention: see the header of
// `tests/math.rs`): one file per colmap/feature/*_test.cc, ported 1:1, plus the `rust_only_*`
// files (error paths, the documented divergences, and VLFeat's SIFT filter against its oracle
// fixture, which has no COLMAP test of its own).
//
// Run: `cargo test -p colmap-rust --test feature`.

#[path = "support/oracle_json.rs"]
mod oracle_json;

#[path = "feature/extractor.rs"]
mod extractor;
#[path = "feature/index.rs"]
mod index;
#[path = "feature/matcher.rs"]
mod matcher;
#[path = "feature/rust_only_feature_types.rs"]
mod rust_only_feature_types;
#[path = "feature/rust_only_feature_utils.rs"]
mod rust_only_feature_utils;
#[path = "feature/rust_only_sift.rs"]
mod rust_only_sift;
#[path = "feature/rust_only_sift_text.rs"]
mod rust_only_sift_text;
#[path = "feature/rust_only_vl_sift_filter.rs"]
mod rust_only_vl_sift_filter;
#[path = "feature/sift.rs"]
mod sift;
#[path = "feature/sift_guided_matching.rs"]
mod sift_guided_matching;
#[path = "feature/sift_matching.rs"]
mod sift_matching;
#[path = "feature/types.rs"]
mod types;
#[path = "feature/utils.rs"]
mod utils;
