// Integration-test binary for COLMAP's `util/` module (layout convention: see the header of
// `tests/math.rs`).
//
// Run: `cargo test -p colmap-rust --test util`.

#[path = "util/logging.rs"]
mod logging;
#[path = "util/stream_format.rs"]
mod stream_format;
