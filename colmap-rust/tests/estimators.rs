// Integration-test binary for COLMAP's `estimators/` module, including its `solvers/`
// subdirectory (layout convention: see the header of `tests/math.rs`).
//
// Run: `cargo test -p colmap-rust --test estimators`.

#[path = "support/transform_matchers.rs"]
mod transform_matchers;

#[path = "estimators/rust_only_sturm.rs"]
mod rust_only_sturm;
#[path = "estimators/similarity_transform.rs"]
mod similarity_transform;

/// gtest's `EXPECT_NEAR(a, b, tol)`.
pub fn expect_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
}
