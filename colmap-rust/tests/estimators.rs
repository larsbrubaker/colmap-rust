// Integration-test binary for COLMAP's `estimators/` module, including its `solvers/`
// subdirectory (layout convention: see the header of `tests/math.rs`).
//
// Run: `cargo test -p colmap-rust --test estimators`.

#[path = "support/transform_matchers.rs"]
mod transform_matchers;

#[path = "estimators/absolute_pose.rs"]
mod absolute_pose;
#[path = "estimators/generalized_absolute_pose.rs"]
mod generalized_absolute_pose;
#[path = "estimators/poselib_utils.rs"]
mod poselib_utils;
#[path = "estimators/rust_only_camera_pose.rs"]
mod rust_only_camera_pose;
#[path = "estimators/rust_only_gp3p.rs"]
mod rust_only_gp3p;
#[path = "estimators/rust_only_p3p.rs"]
mod rust_only_p3p;
#[path = "estimators/rust_only_p4pf.rs"]
mod rust_only_p4pf;
#[path = "estimators/rust_only_poselib_essential.rs"]
mod rust_only_poselib_essential;
#[path = "estimators/rust_only_re3q3.rs"]
mod rust_only_re3q3;
#[path = "estimators/rust_only_relpose_5pt.rs"]
mod rust_only_relpose_5pt;
#[path = "estimators/rust_only_relpose_6pt_focal.rs"]
mod rust_only_relpose_6pt_focal;
#[path = "estimators/rust_only_sturm.rs"]
mod rust_only_sturm;
#[path = "estimators/similarity_transform.rs"]
mod similarity_transform;

/// gtest's `EXPECT_NEAR(a, b, tol)`.
pub fn expect_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
}
