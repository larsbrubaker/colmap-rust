// Integration-test binary for COLMAP's `sensor/` module (layout convention: see the header of
// `tests/math.rs`):
// - `sensor/models.rs`: colmap/sensor/models_test.cc ported 1:1;
// - `sensor/rust_only_models.rs`: invalid-id errors and `cam_ray_from_img_jacobian`;
// - `sensor/rust_only_camera_model_oracle.rs`: every camera model against the pycolmap
//   oracle (fixture `tests/data/oracle/camera_models.json` from `oracle/camera_models.py`).
// colmap/sensor/models_jacobian_test.cc is ported as unit tests in the library,
// `src/sensor/models/jacobian_tests.rs`, because it differentiates with the crate-private Jet
// (run: `cargo test -p colmap-rust --lib jacobian_tests`).
//
// Run: `cargo test -p colmap-rust --test sensor`.

#[path = "support/oracle_json.rs"]
mod oracle_json;

#[path = "sensor/models.rs"]
mod models;
#[path = "sensor/rust_only_camera_model_oracle.rs"]
mod rust_only_camera_model_oracle;
#[path = "sensor/rust_only_models.rs"]
mod rust_only_models;
