// Port of colmap/sensor/database_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::sensor::database::CameraDatabase and the specs table behind it. COLMAP has no
// specs_test.cc; the `rust_only_*` cases pin the generated table's shape.

use colmap_rust::sensor::database::CameraDatabase;
use colmap_rust::sensor::specs::initialize_camera_specs;

#[test]
fn camera_database_initialization() {
    let database = CameraDatabase::new();
    let specs = initialize_camera_specs();
    assert_eq!(database.num_entries(), specs.len());
}

#[test]
fn camera_database_exact_match() {
    let database = CameraDatabase::new();
    let mut sensor_width = 0.0;
    assert!(database.query_sensor_width("canon", "digitalixus100is", &mut sensor_width));
    assert_eq!(sensor_width, f64::from(6.1600f32));
}

#[test]
fn camera_database_ambiguous_match() {
    let database = CameraDatabase::new();
    let mut sensor_width = 0.0;
    assert!(!database.query_sensor_width("canon", "digitalixus", &mut sensor_width));
    assert_eq!(sensor_width, f64::from(6.1600f32));
}

// Rust-only: the generated table holds every make and model of specs.cc (56 makes, 4415
// models at COLMAP 4.2.0), each make once (a repeated make would be merged by COLMAP's
// `specs[make]`), all lower case without separators, with positive widths.
#[test]
fn rust_only_specs_table_shape() {
    let specs = initialize_camera_specs();
    assert_eq!(specs.len(), 56);
    assert_eq!(specs.iter().map(|m| m.models.len()).sum::<usize>(), 4415);
    let mut makes: Vec<&str> = specs.iter().map(|m| m.make).collect();
    makes.sort_unstable();
    makes.dedup();
    assert_eq!(makes.len(), specs.len());
    assert_eq!(specs[0].make, "acer");
    assert_eq!(specs[0].models[0], ("ce5330", 5.75));
    for make in &specs {
        for &(model, width) in make.models {
            assert!(width > 0.0, "{} {model}", make.make);
            assert!(!model.contains(' ') && !model.contains('-'), "{model}");
        }
    }
}

// Rust-only: the cleaning of QuerySensorWidth (separators removed, ASCII lower case, the make
// stripped from the model) lets an EXIF-style make/model pair match exactly.
#[test]
fn rust_only_camera_database_cleans_exif_strings() {
    let database = CameraDatabase::new();
    let mut sensor_width = 0.0;
    assert!(database.query_sensor_width("Canon", "Canon Digital IXUS 100 IS", &mut sensor_width));
    assert_eq!(sensor_width, f64::from(6.1600f32));
    let mut unknown = -1.0;
    assert!(!database.query_sensor_width("nosuchmake", "nosuchmodel", &mut unknown));
    assert_eq!(unknown, -1.0);
}
