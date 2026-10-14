// Integration-test binary for COLMAP's `estimators/` module, including its `solvers/`
// subdirectory (layout convention: see the header of `tests/math.rs`).
//
// Run: `cargo test -p colmap-rust --test estimators`.

#[path = "support/eigen_matchers.rs"]
mod eigen_matchers;
#[path = "support/oracle_json.rs"]
mod oracle_json;
#[path = "support/transform_matchers.rs"]
mod transform_matchers;

#[path = "estimators/absolute_pose.rs"]
mod absolute_pose;
#[path = "estimators/affine_transform.rs"]
mod affine_transform;
#[path = "estimators/essential_matrix.rs"]
mod essential_matrix;
#[path = "estimators/fundamental_matrix.rs"]
mod fundamental_matrix;
#[path = "estimators/fundamental_matrix_degensac.rs"]
mod fundamental_matrix_degensac;
#[path = "estimators/generalized_absolute_pose.rs"]
mod generalized_absolute_pose;
#[path = "estimators/generalized_relative_pose.rs"]
mod generalized_relative_pose;
#[path = "estimators/homography_matrix.rs"]
mod homography_matrix;
#[path = "estimators/poselib_utils.rs"]
mod poselib_utils;
#[path = "estimators/quaternion_utils.rs"]
mod quaternion_utils;
#[path = "estimators/relpose_one_sided_focal.rs"]
mod relpose_one_sided_focal;
#[path = "estimators/relpose_shared_focal.rs"]
mod relpose_shared_focal;
#[path = "estimators/rust_only_camera_pose.rs"]
mod rust_only_camera_pose;
#[path = "estimators/rust_only_fundamental_matrix_degensac_oracle.rs"]
mod rust_only_fundamental_matrix_degensac_oracle;
#[path = "estimators/rust_only_gen_relpose_6pt.rs"]
mod rust_only_gen_relpose_6pt;
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
#[path = "estimators/rust_only_triangulation.rs"]
mod rust_only_triangulation;
#[path = "estimators/similarity_transform.rs"]
mod similarity_transform;
#[path = "estimators/tiny_manifold.rs"]
mod tiny_manifold;
#[path = "estimators/tiny_sampson_error.rs"]
mod tiny_sampson_error;
#[path = "estimators/translation_transform.rs"]
mod translation_transform;

/// gtest's `EXPECT_NEAR(a, b, tol)`.
pub fn expect_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
}
