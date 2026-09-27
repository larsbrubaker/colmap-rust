// Rust-only: every camera model against COLMAP through the pycolmap 4.2.0 Camera binding
// (port of colmap-sharp's CameraModelOracleTests.cs; complements the 1:1 port in models.rs).
// The fixture is written by oracle/camera_models.py into
// tests/data/oracle/camera_models.json: two random parameter sets per model, projections of a
// grid of camera points (plus points behind the camera with the cheirality check off), and
// unprojections (CamFromImg, which runs the iterative undistortion, and CamRayFromImg) of a
// grid of pixels. FOV gets a third set with omega = 5e-4 for its small-omega Taylor branch,
// and every case adds points on and next to the optical axis and pixels at and next to the
// principal point for the small-radius branches.
//
// Tier A, with documented exceptions:
// - Which calls fail must match exactly, for every model and field.
// - Bit-identical: CamFromImg of the pinhole models that unproject through the iterative
//   undistortion (SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV) and the plain pinholes, and
//   the pixel threshold. This pins the Jet, the 2x2 LU solve and the Newton iteration bit for
//   bit.
// - Everything else within 2e-14 * max(1, |expected|): the macOS arm64 pycolmap wheel fuses
//   single-statement multiply-adds (f * x + c1, u*u + v*v) into FMAs, which colmap-rust never
//   does (docs/CPP_DIVERGENCES.md, entry 100), and the models that call sin/cos/tan/atan go
//   through the libm crate rather than Apple libm (entry 1). colmap-sharp also holds
//   EQUIRECTANGULAR bit-exact because .NET calls the platform libm; here its trigonometry is
//   libm's, so it is in the tolerance set.

use colmap_rust::linalg::{Vector2d, Vector3d};
use colmap_rust::sensor::models::*;

use crate::oracle_json::Json;

// Scaled tolerance for the fields the wheel's FMA contraction and Apple libm reach.
const CONTRACTION_TOLERANCE: f64 = 2e-14;

const EXACT_CAM_FROM_IMG_MODELS: [&str; 6] = [
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "SIMPLE_RADIAL",
    "RADIAL",
    "OPENCV",
    "FULL_OPENCV",
];

fn load() -> Json {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/oracle/camera_models.json"
    );
    Json::parse(&std::fs::read_to_string(path).expect("read camera_models.json"))
}

fn rows(array: &Json) -> Vec<Vec<f64>> {
    array.as_array().iter().map(Json::as_f64s).collect()
}

/// A fixture value: `null` for a failed call, else an array of numbers or of the strings
/// "NaN" / "Infinity" / "-Infinity".
fn optional(value: &Json) -> Option<Vec<f64>> {
    match value {
        Json::Null => None,
        Json::Array(items) => Some(
            items
                .iter()
                .map(|item| match item {
                    Json::String(s) => s.parse::<f64>().expect("non-finite number"),
                    other => other.as_f64(),
                })
                .collect(),
        ),
        other => panic!("unexpected fixture value {other:?}"),
    }
}

fn compare(
    failures: &mut Vec<String>,
    exact: bool,
    what: &str,
    actual: Option<Vec<f64>>,
    expected: Option<Vec<f64>>,
) {
    let (actual, expected) = match (actual, expected) {
        (Some(a), Some(e)) => (a, e),
        (None, None) => return,
        (a, e) => {
            failures.push(format!(
                "{what}: expected {}, got {}",
                if e.is_none() { "failure" } else { "a value" },
                if a.is_none() { "failure" } else { "a value" }
            ));
            return;
        }
    };
    for (k, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
        if (a.is_nan() && e.is_nan()) || a.to_bits() == e.to_bits() {
            continue;
        }
        let scaled = (a - e).abs() / e.abs().max(1.0);
        let within_tolerance = scaled <= CONTRACTION_TOLERANCE; // false for NaN
        if exact || !within_tolerance {
            failures.push(format!(
                "{what}[{k}]: expected {e:?}, got {a:?} (scaled difference {scaled:.2e}{})",
                if exact { ", must be exact" } else { "" }
            ));
        }
    }
}

fn v2(v: Option<Vector2d>) -> Option<Vec<f64>> {
    v.map(|v| vec![v.x, v.y])
}

fn v3(v: Option<Vector3d>) -> Option<Vec<f64>> {
    v.map(|v| vec![v.x, v.y, v.z])
}

#[test]
fn rust_only_camera_models_match_pycolmap() {
    let root = load();
    let cam_points = rows(root.get("cam_points"));
    let back_points = rows(root.get("back_points"));
    let pixels = rows(root.get("pixels"));
    let near_axis_points = rows(root.get("near_axis_points"));
    let mut failures = Vec::new();
    let mut models_seen = std::collections::BTreeSet::new();

    for case in root.get("cases").as_array() {
        let model_name = case.get("model").as_str();
        models_seen.insert(model_name.to_string());
        let id = camera_model_name_to_id(model_name);
        assert_ne!(id, CameraModelId::Invalid, "unknown model {model_name}");
        let exact_cam_from_img = EXACT_CAM_FROM_IMG_MODELS.contains(&model_name);
        let params = case.get("params").as_f64s();
        let tag = |what: &str, i: usize| format!("{model_name} {params:?} {what}[{i}]");

        compare(
            &mut failures,
            true,
            &format!("{model_name} threshold"),
            Some(vec![camera_model_cam_from_img_threshold(id, &params, 1.5)]),
            Some(vec![case.get("threshold").as_f64()]),
        );

        let point_sets = [
            ("img_from_cam", &cam_points, true),
            ("img_from_cam_back", &back_points, false),
            ("near_axis_img_from_cam", &near_axis_points, true),
        ];
        for (field, points, check_cheirality) in point_sets {
            let expected = case.get(field).as_array();
            for (i, p) in points.iter().enumerate() {
                let xy = camera_model_img_from_cam(
                    id,
                    &params,
                    Vector3d::new(p[0], p[1], p[2]),
                    check_cheirality,
                );
                compare(
                    &mut failures,
                    false,
                    &tag(field, i),
                    v2(xy),
                    optional(&expected[i]),
                );
            }
        }

        let pp_pixels = rows(case.get("pp_pixels"));
        let pixel_sets = [
            ("pp_cam_from_img", "pp_cam_ray_from_img", &pp_pixels),
            ("cam_from_img", "cam_ray_from_img", &pixels),
        ];
        for (uv_field, ray_field, pixel_rows) in pixel_sets {
            let expected_uv = case.get(uv_field).as_array();
            let expected_ray = case.get(ray_field).as_array();
            for (i, p) in pixel_rows.iter().enumerate() {
                let pixel = Vector2d::new(p[0], p[1]);
                compare(
                    &mut failures,
                    exact_cam_from_img,
                    &tag(uv_field, i),
                    v2(camera_model_cam_from_img(id, &params, pixel)),
                    optional(&expected_uv[i]),
                );
                compare(
                    &mut failures,
                    false,
                    &tag(ray_field, i),
                    v3(camera_model_cam_ray_from_img(id, &params, pixel)),
                    optional(&expected_ray[i]),
                );
            }
        }
    }

    // Every model is covered by the fixture.
    for id in CameraModelId::ALL {
        if id != CameraModelId::Invalid {
            assert!(
                models_seen.contains(camera_model_id_to_name(id)),
                "fixture lacks {id}"
            );
        }
    }
    assert!(
        failures.is_empty(),
        "{} mismatches:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
