// Rust-only oracle test (bitmap_test.cc pins only Rescale's output dimensions; port of
// colmap-sharp's BitmapRescaleOracleTests.cs). Compares Bitmap::rescale
// (src/sensor/bitmap/resize.rs) with pycolmap 4.2.0's OpenImageIO resize on the seeded images
// of oracle/fixture_bitmap_rescale.py (fixture tests/data/oracle/bitmap_rescale.json).
//
// Tier B: every pixel within one gray level of pycolmap's (docs/CPP_DIVERGENCES.md, entry
// 121).

use colmap_rust::sensor::bitmap::{Bitmap, RescaleFilter};

use crate::oracle_json::Json;

fn load() -> Json {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/oracle/bitmap_rescale.json"
    );
    Json::parse(&std::fs::read_to_string(path).expect("read bitmap_rescale.json"))
}

#[test]
fn rust_only_rescale_matches_pycolmap_within_one_gray_level() {
    let root = load();
    let cases = root.get("cases").as_array();
    assert!(!cases.is_empty());
    for case in cases {
        let int = |key: &str| case.get(key).as_f64() as i32;
        let (width, height, channels) = (int("width"), int("height"), int("channels"));
        let (new_width, new_height) = (int("new_width"), int("new_height"));
        let filter = if case.get("filter").as_str() == "box" {
            RescaleFilter::Box
        } else {
            RescaleFilter::Bilinear
        };
        let input = case.get("input").as_f64s();
        let expected = case.get("output").as_f64s();

        let mut bitmap = Bitmap::new(width, height, channels == 3);
        for (pixel, &value) in bitmap.row_major_data_mut().iter_mut().zip(&input) {
            *pixel = value as u8;
        }
        bitmap.rescale(new_width, new_height, filter);

        let label = format!("{width}x{height}x{channels} -> {new_width}x{new_height} {filter:?}");
        assert_eq!(bitmap.num_bytes(), expected.len(), "{label}");
        let max_difference = bitmap
            .row_major_data()
            .iter()
            .zip(&expected)
            .map(|(&actual, &expected)| (f64::from(actual) - expected).abs())
            .fold(0.0, f64::max);
        assert!(
            max_difference <= 1.0,
            "{label}: max difference {max_difference}"
        );
    }
}
