// Integration-test binary for COLMAP's `sensor/` module (layout convention: see the header of
// `tests/math.rs`):
// - `sensor/models.rs`: colmap/sensor/models_test.cc ported 1:1;
// - `sensor/rust_only_models.rs`: invalid-id errors and `cam_ray_from_img_jacobian`;
// - `sensor/rust_only_camera_model_oracle.rs`: every camera model against the pycolmap
//   oracle (fixture `tests/data/oracle/camera_models.json` from `oracle/camera_models.py`);
// - `sensor/rig.rs`: colmap/sensor/rig_test.cc ported 1:1;
// - `sensor/database.rs`: colmap/sensor/database_test.cc ported 1:1 (plus the specs table);
// - `sensor/bitmap.rs` and `sensor/bitmap_exif.rs`: colmap/sensor/bitmap_test.cc ported 1:1,
//   minus the file I/O cases (listed in `bitmap.rs`);
// - `sensor/rust_only_bitmap.rs`: interpolation edge cases, metadata, casts, JetColormap;
// - `sensor/rust_only_bitmap_rescale_oracle.rs`: Rescale against the pycolmap oracle
//   (fixture `tests/data/oracle/bitmap_rescale.json` from `oracle/fixture_bitmap_rescale.py`);
// - `sensor/rust_only_exif_reader.rs`: the EXIF parser on hand-built JPEG/TIFF blocks.
//
// Run: `cargo test -p colmap-rust --test sensor`.

#[path = "support/oracle_json.rs"]
mod oracle_json;

#[path = "sensor/bitmap.rs"]
mod bitmap;
#[path = "sensor/bitmap_exif.rs"]
mod bitmap_exif;
#[path = "sensor/database.rs"]
mod database;
#[path = "sensor/models.rs"]
mod models;
#[path = "sensor/rig.rs"]
mod rig;
#[path = "sensor/rust_only_bitmap.rs"]
mod rust_only_bitmap;
#[path = "sensor/rust_only_bitmap_rescale_oracle.rs"]
mod rust_only_bitmap_rescale_oracle;
#[path = "sensor/rust_only_camera_model_oracle.rs"]
mod rust_only_camera_model_oracle;
#[path = "sensor/rust_only_exif_reader.rs"]
mod rust_only_exif_reader;
#[path = "sensor/rust_only_models.rs"]
mod rust_only_models;
