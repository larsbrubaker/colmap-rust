// Rust-only tests of colmap_rust::scene::rig (read_rig_config) and scene::rig_apply
// (apply_rig_config): cases rig_test.cc does not have. The translator cases (which texts read
// as a bool or a double) are colmap-sharp's RigConfigTests.CSharpOnly cases, whose expected
// results were taken from a libc++ harness (Apple clang, istringstream with ptree's extract
// logic); the rest pin the semantic checks with valid JSON (rig_test.cc's invalid configs
// also have trailing commas, which fail earlier, at the parse), the divergences in the docs of
// `read_doubles` (scene/rig.rs) and `update_rigs_and_frames_from_database`
// (scene/rig_apply.rs), and the sensor poses ApplyRigConfig derives from a reconstruction.

use colmap_rust::math::random::set_prng_seed;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::rig::{read_rig_config, RigCamera, RigConfig};
use colmap_rust::scene::rig_apply::apply_rig_config;
use colmap_rust::scene::synthetic::{synthesize_dataset_with_database, SyntheticDatasetOptions};
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::util::types::{SensorId, SensorType};

/// The error message of a failed read.
fn read_error(json: &str) -> String {
    read_rig_config(json).unwrap_err().message().to_string()
}

// A database with one camera and one image per name, image ids in the order given.
fn database_with_images(names: &[&str]) -> Database {
    let database = Database::new();
    let camera =
        Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1.0, 10, 10).unwrap();
    let camera_id = database.write_camera(&camera, false).unwrap();
    for name in names {
        let mut image = Image::new();
        image.set_name(name);
        image.set_camera_id(camera_id).unwrap();
        database.write_image(&image, false).unwrap();
    }
    database
}

fn single_camera_config(image_prefix: &str) -> Vec<RigConfig> {
    vec![RigConfig {
        cameras: vec![RigCamera {
            image_prefix: image_prefix.to_string(),
            ref_sensor: true,
            ..RigCamera::default()
        }],
    }]
}

#[test]
fn rust_only_apply_rig_config_empty_image_prefix_matches_nothing() {
    // COLMAP's StringStartsWith is false for an empty prefix, so the camera finds no image.
    let database = database_with_images(&["a", "b"]);
    let error = apply_rig_config(&single_camera_config(""), &database, None).unwrap_err();
    assert!(error
        .message()
        .contains("At least one image must exist for each camera in the rig"));
}

#[test]
fn rust_only_apply_rig_config_frame_names_in_utf8_byte_order() {
    // UTF-8 puts U+FF21 (EF BC A1) before U+1F600 (F0 9F 98 80); std::map<std::string> uses
    // the byte order, so the full-width name's frame is written first.
    let database = database_with_images(&["cam_\u{1F600}", "cam_\u{FF21}"]);
    apply_rig_config(&single_camera_config("cam_"), &database, None).unwrap();
    let frames = database.read_all_frames().unwrap();
    assert_eq!(frames.len(), 2);
    let ids = |i: usize| frames[i].image_ids().map(|d| d.id).collect::<Vec<_>>();
    assert_eq!(ids(0), [2]);
    assert_eq!(ids(1), [1]);
}

// A two-camera rig whose second camera has the given raw JSON "ref_sensor" value.
fn config_with_second_ref_sensor(ref_sensor_json: &str) -> String {
    format!(
        r#"[{{"cameras": [
            {{"image_prefix": "a/"}},
            {{"image_prefix": "b/", "ref_sensor": {ref_sensor_json}}}
        ]}}]"#
    )
}

#[test]
fn rust_only_read_rig_config_ref_sensor_reads_true_like_ptree() {
    for json in [
        r#""01""#,
        r#""+1""#,
        r#"" true ""#,
        r#""2true""#,
        "1",
        "true",
    ] {
        let configs = read_rig_config(&config_with_second_ref_sensor(json)).unwrap();
        assert!(configs[0].cameras[1].ref_sensor, "{json}");
    }
}

#[test]
fn rust_only_read_rig_config_ref_sensor_reads_false_like_ptree() {
    // A false ref_sensor leaves the rig without a reference sensor.
    for json in [r#""00""#, r#""-0""#, "false", "0"] {
        let message = read_error(&config_with_second_ref_sensor(json));
        assert!(
            message.contains("Rig must have one reference sensor"),
            "{json}: {message}"
        );
    }
}

#[test]
fn rust_only_read_rig_config_ref_sensor_rejects_like_ptree() {
    for json in [
        r#""2""#,
        r#""-1""#,
        r#""TRUE""#,
        r#""tru""#,
        r#""1x""#,
        r#""0true""#,
        r#""""#,
        r#""99999999999999999999""#,
        "[]",
    ] {
        let message = read_error(&config_with_second_ref_sensor(json));
        assert!(
            message.contains("conversion of data to type \"bool\" failed"),
            "{json}: {message}"
        );
    }
}

fn config_with_param(param_json: &str) -> String {
    format!(
        r#"[{{"cameras": [
            {{"image_prefix": "a/", "ref_sensor": true, "camera_model_name": "SIMPLE_PINHOLE",
              "camera_params": [{param_json}]}}
        ]}}]"#
    )
}

#[test]
fn rust_only_read_rig_config_param_reads_like_iostream() {
    let cases: [(&str, f64); 17] = [
        (r#""1.5""#, 1.5),
        (r#"" 2 ""#, 2.0),
        (r#""+.5""#, 0.5),
        (r#""5.""#, 5.0),
        (r#""1E5""#, 1e5),
        (r#""0x1p3""#, 8.0),
        (r#""0x10""#, 16.0),
        (r#""0x.8""#, 0.5),
        (r#""0e-400""#, 0.0),
        (r#""2.2250738585072014e-308""#, 2.2250738585072014e-308),
        ("1.7976931348623158e308", 1.7976931348623157e308),
        ("-0.25", -0.25),
        (r#""-0x10""#, -16.0),
        (r#""0x1.00000000000018p0""#, 1.0000000000000004),
        (r#""0x1.ffffffffffffffp-1023""#, 2.2250738585072014e-308),
        (r#""0x1p-1030""#, f64::from_bits(1 << 44)), // 2^-1030 = 8.6916947597937554e-311
        (r#""0x1p-1074""#, f64::from_bits(1)),       // 2^-1074 = 4.9406564584124654e-324
    ];
    for (json, expected) in cases {
        let configs = read_rig_config(&config_with_param(json)).unwrap();
        let params = &configs[0].cameras[0].camera.as_ref().unwrap().params;
        assert_eq!(params.len(), 1, "{json}");
        assert_eq!(params[0].to_bits(), expected.to_bits(), "{json}");
    }
}

#[test]
fn rust_only_read_rig_config_param_rejects_like_iostream() {
    for json in [
        r#""inf""#,
        r#""-inf""#,
        r#""nan""#,
        r#""NaN""#,
        r#""Infinity""#,
        r#""∞""#,
        "1e400",
        "-1e400",
        "1.7976931348623159e308",
        "1e-310",
        "1e-400",
        "2.2250738585072011e-308",
        r#""1,5""#,
        r#""1d""#,
        r#""1e5.""#,
        r#""1.2.3""#,
        r#""1e""#,
        r#"".""#,
        r#""""#,
        r#""0x""#,
        r#""0xg""#,
        r#""0x1p""#,
        r#""0x1p1024""#,
        r#""0x1.fffffffffffff8p1023""#,
        r#""0x1.8p-1074""#,
        r#""0x1p-1080""#,
    ] {
        let message = read_error(&config_with_param(json));
        assert!(
            message.contains("conversion of data to type \"double\" failed"),
            "{json}: {message}"
        );
    }
}

#[test]
fn rust_only_read_rig_config_exact_decimal_subnormal_reads() {
    // 2^-1074 written out exactly (751 significant digits) is exact, so strtod does not
    // report ERANGE; its first 17 digits alone are inexact and fail (as "1e-310" does).
    let exact = format!("\"{}\"", exact_decimal_of_min_subnormal());
    let configs = read_rig_config(&config_with_param(&exact)).unwrap();
    let params = &configs[0].cameras[0].camera.as_ref().unwrap().params;
    assert_eq!(params[0].to_bits(), 1);
    assert!(read_error(&config_with_param("4.9406564584124654e-324"))
        .contains("conversion of data to type \"double\" failed"));
}

/// The exact decimal expansion of 2^-1074 = 5^1074 / 10^1074, as "0.000...".
fn exact_decimal_of_min_subnormal() -> String {
    // 5^1074 in decimal, by schoolbook multiplication.
    let mut digits = vec![1u32]; // least significant first
    for _ in 0..1074 {
        let mut carry = 0;
        for digit in &mut digits {
            let product = *digit * 5 + carry;
            *digit = product % 10;
            carry = product / 10;
        }
        while carry > 0 {
            digits.push(carry % 10);
            carry /= 10;
        }
    }
    let significant: String = digits
        .iter()
        .rev()
        .map(|d| char::from(b'0' + *d as u8))
        .collect();
    format!("0.{}{}", "0".repeat(1074 - significant.len()), significant)
}

#[test]
fn rust_only_read_rig_config_semantic_checks() {
    let two_refs = r#"[{"cameras": [
        {"image_prefix": "a/", "ref_sensor": true},
        {"image_prefix": "b/", "ref_sensor": true}]}]"#;
    assert!(read_error(two_refs).contains("Rig must only have one reference sensor"));

    let no_ref = r#"[{"cameras": [{"image_prefix": "a/"}, {"image_prefix": "b/"}]}]"#;
    assert!(read_error(no_ref).contains("Rig must have one reference sensor"));

    let ref_with_pose = r#"[{"cameras": [
        {"image_prefix": "a/", "ref_sensor": true, "cam_from_rig_rotation": [1, 0, 0, 0]}]}]"#;
    assert!(read_error(ref_with_pose).contains("Reference sensor must not have cam_from_rig"));

    let missing_prefix = r#"[{"cameras": [{"ref_sensor": true}]}]"#;
    assert!(read_error(missing_prefix).contains("No such node (image_prefix)"));

    let missing_cameras = r#"[{"rig": []}]"#;
    assert!(read_error(missing_cameras).contains("No such node (cameras)"));

    // Boost rejects trailing commas, so rig_test.cc's invalid configs fail at the parse.
    assert!(read_error(r#"[{"cameras": [],}]"#).contains("JSON parse error"));
}

#[test]
fn rust_only_read_rig_config_pose_needs_four_and_three_entries() {
    // The divergence in `read_doubles`' doc: COLMAP writes past (or short of) its fixed-size vectors.
    let pose = |rotation: &str, translation: &str| {
        format!(
            r#"[{{"cameras": [{{"image_prefix": "a/", "ref_sensor": true}},
               {{"image_prefix": "b/", "cam_from_rig_rotation": [{rotation}],
                 "cam_from_rig_translation": [{translation}]}}]}}]"#
        )
    };
    assert!(read_error(&pose("1, 0, 0", "1, 2, 3")).contains("must have 4 entries"));
    assert!(read_error(&pose("1, 0, 0, 0, 0", "1, 2, 3")).contains("must have 4 entries"));
    assert!(read_error(&pose("1, 0, 0, 0", "1, 2")).contains("must have 3 entries"));
    let configs = read_rig_config(&pose("1, 0, 0, 0", "1, 2, 3")).unwrap();
    assert!(configs[0].cameras[1].cam_from_rig.is_some());
    // Only a rotation (or only a translation) leaves the pose unknown, as in COLMAP.
    let rotation_only = r#"[{"cameras": [{"image_prefix": "a/", "ref_sensor": true},
        {"image_prefix": "b/", "cam_from_rig_rotation": [1, 0, 0, 0]}]}]"#;
    assert!(read_rig_config(rotation_only).unwrap()[0].cameras[1]
        .cam_from_rig
        .is_none());
}

#[test]
fn rust_only_read_rig_config_reads_like_a_property_tree() {
    // Quoted numbers read as numbers, a duplicate key reads its first value, an object's
    // member values are its children (the root's are rigs, camera_params' are parameters),
    // and a scalar root has no rigs.
    let json = r#"{"first": {"cameras": [
        {"image_prefix": "a/", "image_prefix": "ignored/", "ref_sensor": "true",
         "camera_model_name": "SIMPLE_PINHOLE", "camera_params": {"f": "1.5", "c": 2, "k": 3}}]}}"#;
    let configs = read_rig_config(json).unwrap();
    assert_eq!(configs.len(), 1);
    let camera = &configs[0].cameras[0];
    assert_eq!(camera.image_prefix, "a/");
    assert!(camera.ref_sensor);
    assert_eq!(camera.camera.as_ref().unwrap().params, [1.5, 2.0, 3.0]);
    assert!(read_rig_config("5").unwrap().is_empty());
    // A container's text is "", which is not a double.
    let nested_param = config_with_param("[1]");
    assert!(read_error(&nested_param).contains("conversion of data to type \"double\" failed"));
    // A container's text is "": an array image prefix is the empty prefix.
    let array_prefix = r#"[{"cameras": [{"image_prefix": [], "ref_sensor": true}]}]"#;
    assert_eq!(
        read_rig_config(array_prefix).unwrap()[0].cameras[0].image_prefix,
        ""
    );
}

#[test]
fn rust_only_apply_rig_config_derives_sensor_from_rig_from_reconstruction() {
    // A two-camera rig with a known pose: ApplyRigConfig averages the relative poses of the
    // registered images and recovers it.
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 2,
        num_frames_per_rig: 4,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
    let sensor2 = SensorId::new(SensorType::Camera, 2);
    let expected = reconstruction
        .rig(1)
        .unwrap()
        .sensor_from_rig(sensor2)
        .unwrap();

    let configs = vec![RigConfig {
        cameras: vec![
            RigCamera {
                image_prefix: "camera000001_".to_string(),
                ref_sensor: true,
                ..RigCamera::default()
            },
            RigCamera {
                image_prefix: "camera000002_".to_string(),
                ..RigCamera::default()
            },
        ],
    }];
    apply_rig_config(&configs, &database, Some(&mut reconstruction)).unwrap();

    let rigs = database.read_all_rigs().unwrap();
    assert_eq!(rigs.len(), 1);
    let derived = rigs[0].sensor_from_rig(sensor2).unwrap();
    let rotation_error = (derived.rotation.coeffs() - expected.rotation.coeffs()).norm();
    let rotation_error_flipped = (derived.rotation.coeffs() + expected.rotation.coeffs()).norm();
    assert!(rotation_error.min(rotation_error_flipped) < 1e-9);
    assert!((derived.translation - expected.translation).norm() < 1e-9);

    // The reconstruction's rig carries the derived pose, and its frames register in
    // ascending id order (the divergence in `update_rigs_and_frames_from_database`' doc).
    let reconstruction_rig = reconstruction.rig(rigs[0].rig_id()).unwrap();
    assert_eq!(
        reconstruction_rig.sensor_from_rig(sensor2).unwrap(),
        derived
    );
    assert_eq!(reconstruction.num_reg_frames(), 4);
    let reg_frame_ids = reconstruction.reg_frame_ids().to_vec();
    let mut sorted = reg_frame_ids.clone();
    sorted.sort_unstable();
    assert_eq!(reg_frame_ids, sorted);
}
