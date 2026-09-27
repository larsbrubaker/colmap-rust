// Integration-test binary for COLMAP's `optim/` module (layout convention: see the header of
// `tests/math.rs`). `optim/test_estimators.rs` holds the test-only estimators shared by the
// RANSAC files.
//
// Run: `cargo test -p colmap-rust --test optim`.

#[path = "optim/test_estimators.rs"]
mod test_estimators;

#[path = "optim/combination_sampler.rs"]
mod combination_sampler;
#[path = "optim/loransac.rs"]
mod loransac;
#[path = "optim/progressive_sampler.rs"]
mod progressive_sampler;
#[path = "optim/random_sampler.rs"]
mod random_sampler;
#[path = "optim/ransac.rs"]
mod ransac;
#[path = "optim/rust_only_ransac.rs"]
mod rust_only_ransac;
#[path = "optim/rust_only_sampler_sequence.rs"]
mod rust_only_sampler_sequence;
#[path = "optim/sprt.rs"]
mod sprt;
#[path = "optim/support_measurement.rs"]
mod support_measurement;
