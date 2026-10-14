// Rust-only tests for colmap_rust::feature::sift's CPU extractor: the VLFeat-to-UBC
// reordering, filter reuse, cancellation, the constructor checks, and the extractor against
// pycolmap's CPU SIFT on `sift.json`, written by oracle/fixture_sift.py (a white square and a
// seeded texture, under option sets that reach every scale-space path, normalization, upright mode, DoG-level
// truncation and the orientation limit). Port of colmap-sharp's
// SiftOracleTests.CSharpOnly_MatchesPycolmapWithinFmaTolerance, one test per fixture case.
//
// Not bit-exact, on purpose (docs/CPP_DIVERGENCES.md entry 230): the macOS arm64 wheel's VLFeat
// is compiled with Apple clang's default -ffp-contract=on, which fuses a*b + c into FMAs that
// round once; the port never fuses. The bit-exact Tier A check of the same code is
// rust_only_vl_sift_filter.rs, against VLFeat compiled unfused. Here the same keypoints must
// come out (same count and order, so the same detections and DoG-level truncation), with
// values within the drift fusing causes. The bounds are colmap-sharp's (observed there: at most
// 1.03e-3 in a keypoint value, 1 grey level in a descriptor byte, at most 0.05% of the bytes).

use colmap_rust::feature::extractor::{FeatureExtractionOptions, FeatureExtractor};
use colmap_rust::feature::sift::cpu_extractor::{
    transform_vlfeat_to_ubc_feature_descriptors, SiftCpuFeatureExtractor,
};
use colmap_rust::feature::sift::SiftNormalization;
use colmap_rust::feature::types::{FeatureDescriptors, FeatureExtractorType, FeatureKeypoints};
use colmap_rust::sensor::bitmap::Bitmap;
use colmap_rust::util::cancellation::CancelToken;

use crate::oracle_json::Json;
use crate::rust_only_vl_sift_filter::decode_base64;

const SIFT_FIXTURE: &str = include_str!("../data/oracle/sift.json");

/// A grey bitmap from a fixture image (base64 row-major pixels).
pub(crate) fn load_image(image: &Json) -> Bitmap {
    let width = image.get("width").as_f64() as i32;
    let height = image.get("height").as_f64() as i32;
    let pixels = decode_base64(image.get("pixels").as_str());
    let mut bitmap = Bitmap::new(width, height, false);
    bitmap.row_major_data_mut().copy_from_slice(&pixels);
    bitmap
}

/// The case's SIFT option overrides on top of the defaults.
fn parse_options(overrides: &Json) -> FeatureExtractionOptions {
    let mut options = FeatureExtractionOptions::default();
    let Json::Object(map) = overrides else {
        panic!("options must be an object");
    };
    for (name, value) in map {
        let sift = &mut options.sift;
        match name.as_str() {
            "normalization" => {
                sift.normalization = match value.as_f64() as i32 {
                    0 => SiftNormalization::L1Root,
                    1 => SiftNormalization::L2,
                    v => panic!("unknown normalization {v}"),
                }
            }
            "upright" => sift.upright = matches!(value, Json::Bool(true)),
            "max_num_features" => sift.max_num_features = value.as_f64() as i32,
            "first_octave" => sift.first_octave = value.as_f64() as i32,
            "num_octaves" => sift.num_octaves = value.as_f64() as i32,
            "max_num_orientations" => sift.max_num_orientations = value.as_f64() as i32,
            "peak_threshold" => sift.peak_threshold = value.as_f64(),
            other => panic!("Unknown SIFT option in fixture: {other}"),
        }
    }
    options
}

fn run_case(name: &str) {
    let root = Json::parse(SIFT_FIXTURE);
    let case = root
        .get("cases")
        .as_array()
        .iter()
        .find(|c| c.get("name").as_str() == name)
        .unwrap_or_else(|| panic!("no case {name}"));
    let bitmap = load_image(root.get("images").get(case.get("image").as_str()));
    let options = parse_options(case.get("options"));

    let mut extractor = SiftCpuFeatureExtractor::new(&options).unwrap();
    let mut keypoints = FeatureKeypoints::new();
    let mut descriptors = FeatureDescriptors::default();
    assert!(extractor
        .extract(
            &bitmap,
            &mut keypoints,
            Some(&mut descriptors),
            &CancelToken::new()
        )
        .unwrap());

    let expected_keypoints: Vec<Vec<f64>> = case
        .get("keypoints")
        .as_array()
        .iter()
        .map(Json::as_f64s)
        .collect();
    let expected_descriptors = decode_base64(case.get("descriptors").as_str());

    assert_eq!(keypoints.len(), expected_keypoints.len());
    assert_eq!(descriptors.extractor_type, FeatureExtractorType::Sift);
    assert_eq!(descriptors.data.rows(), expected_keypoints.len());
    assert_eq!(descriptors.data.cols(), 128);

    let mut max_keypoint_diff = 0.0f64;
    for (k, expected) in keypoints.iter().zip(&expected_keypoints) {
        let got = [k.x, k.y, k.a11, k.a12, k.a21, k.a22];
        for (g, e) in got.iter().zip(expected) {
            max_keypoint_diff = max_keypoint_diff.max((f64::from(*g) - e).abs());
        }
    }

    let mut max_byte_diff = 0;
    let mut mismatched_bytes = 0;
    assert_eq!(
        descriptors.data.as_slice().len(),
        expected_descriptors.len()
    );
    for (g, e) in descriptors
        .data
        .as_slice()
        .iter()
        .zip(&expected_descriptors)
    {
        let diff = (i32::from(*g) - i32::from(*e)).abs();
        max_byte_diff = max_byte_diff.max(diff);
        mismatched_bytes += usize::from(diff != 0);
    }

    assert!(
        max_keypoint_diff <= 2e-3,
        "{name}: keypoint diff {max_keypoint_diff}"
    );
    assert!(max_byte_diff <= 1, "{name}: byte diff {max_byte_diff}");
    assert!(
        mismatched_bytes <= expected_descriptors.len() / 1000,
        "{name}: {mismatched_bytes} mismatched bytes"
    );
}

#[test]
fn rust_only_sift_matches_pycolmap_square256() {
    run_case("Square256");
}

#[test]
fn rust_only_sift_matches_pycolmap_texture() {
    run_case("Texture");
}

#[test]
fn rust_only_sift_matches_pycolmap_texture_l2() {
    run_case("TextureL2");
}

#[test]
fn rust_only_sift_matches_pycolmap_texture_upright() {
    run_case("TextureUpright");
}

#[test]
fn rust_only_sift_matches_pycolmap_texture_max_features() {
    run_case("TextureMaxFeatures");
}

#[test]
fn rust_only_sift_matches_pycolmap_texture_first_octave0() {
    run_case("TextureFirstOctave0");
}

#[test]
fn rust_only_sift_matches_pycolmap_texture_first_octave1() {
    run_case("TextureFirstOctave1");
}

#[test]
fn rust_only_sift_matches_pycolmap_texture_first_octave_minus2() {
    run_case("TextureFirstOctaveMinus2");
}

#[test]
fn rust_only_sift_matches_pycolmap_texture_orientations4() {
    run_case("TextureOrientations4");
}

/// The extractor reuses its VLFeat filter across same-sized images and rebuilds it for a new
/// size; either way each image comes out exactly as a fresh extractor would give it.
#[test]
fn rust_only_sift_extractor_reuse_matches_fresh_extractor() {
    let root = Json::parse(SIFT_FIXTURE);
    let square = load_image(root.get("images").get("square"));
    let texture = load_image(root.get("images").get("texture"));
    let options = FeatureExtractionOptions::default();
    let cancel = CancelToken::new();

    let extract = |extractor: &mut SiftCpuFeatureExtractor, bitmap: &Bitmap| {
        let mut keypoints = FeatureKeypoints::new();
        let mut descriptors = FeatureDescriptors::default();
        extractor
            .extract(bitmap, &mut keypoints, Some(&mut descriptors), &cancel)
            .unwrap();
        (keypoints, descriptors)
    };

    let mut reused = SiftCpuFeatureExtractor::new(&options).unwrap();
    let sequence = [&texture, &square, &square, &texture];
    for bitmap in sequence {
        let got = extract(&mut reused, bitmap);
        let fresh = extract(&mut SiftCpuFeatureExtractor::new(&options).unwrap(), bitmap);
        assert_eq!(got, fresh);
    }
}

/// A cancelled extraction returns `Cancelled`, leaves the outputs alone, and the next
/// extraction is exactly a fresh extractor's.
#[test]
fn rust_only_sift_extractor_cancel_leaves_outputs_and_recovers() {
    let root = Json::parse(SIFT_FIXTURE);
    let texture = load_image(root.get("images").get("texture"));
    let options = FeatureExtractionOptions::default();
    let mut extractor = SiftCpuFeatureExtractor::new(&options).unwrap();

    let cancelled = CancelToken::new();
    cancelled.cancel();
    let mut keypoints = FeatureKeypoints::new();
    let mut descriptors = FeatureDescriptors::default();
    let error = extractor
        .extract(&texture, &mut keypoints, Some(&mut descriptors), &cancelled)
        .unwrap_err();
    assert_eq!(error.kind(), colmap_rust::ErrorKind::Cancelled);
    assert!(keypoints.is_empty());
    assert_eq!(descriptors, FeatureDescriptors::default());

    let cancel = CancelToken::new();
    extractor
        .extract(&texture, &mut keypoints, Some(&mut descriptors), &cancel)
        .unwrap();
    let mut fresh_keypoints = FeatureKeypoints::new();
    let mut fresh_descriptors = FeatureDescriptors::default();
    SiftCpuFeatureExtractor::new(&options)
        .unwrap()
        .extract(
            &texture,
            &mut fresh_keypoints,
            Some(&mut fresh_descriptors),
            &cancel,
        )
        .unwrap();
    assert_eq!(keypoints, fresh_keypoints);
    assert_eq!(descriptors, fresh_descriptors);
}

/// Without descriptors the keypoints are the same as with them.
#[test]
fn rust_only_sift_extractor_keypoints_without_descriptors() {
    let root = Json::parse(SIFT_FIXTURE);
    let texture = load_image(root.get("images").get("texture"));
    let options = FeatureExtractionOptions::default();
    let cancel = CancelToken::new();
    let mut with = FeatureKeypoints::new();
    let mut descriptors = FeatureDescriptors::default();
    SiftCpuFeatureExtractor::new(&options)
        .unwrap()
        .extract(&texture, &mut with, Some(&mut descriptors), &cancel)
        .unwrap();
    let mut without = FeatureKeypoints::new();
    SiftCpuFeatureExtractor::new(&options)
        .unwrap()
        .extract(&texture, &mut without, None, &cancel)
        .unwrap();
    assert_eq!(with, without);
    assert_eq!(with.len(), 154);
}

/// The constructor's checks: options must pass Check and not ask for the covariant
/// extractor; Extract needs a grey bitmap.
#[test]
fn rust_only_sift_extractor_checks() {
    let mut options = FeatureExtractionOptions::default();
    options.sift.force_covariant_extractor = true;
    assert!(SiftCpuFeatureExtractor::new(&options).is_err());
    let mut options = FeatureExtractionOptions::default();
    options.sift.max_num_features = 0;
    assert!(SiftCpuFeatureExtractor::new(&options).is_err());

    let rgb = Bitmap::new(8, 8, true);
    let mut extractor = SiftCpuFeatureExtractor::new(&FeatureExtractionOptions::default()).unwrap();
    let mut keypoints = FeatureKeypoints::new();
    assert!(extractor
        .extract(&rgb, &mut keypoints, None, &CancelToken::new())
        .is_err());
}

/// `TransformVLFeatToUBCFeatureDescriptors` reverses the rotational order of the 8
/// orientation bins of each of the 16 cells, keeping bin 0.
#[test]
fn rust_only_transform_vlfeat_to_ubc_feature_descriptors() {
    let values: Vec<u8> = (0..256).map(|v| v as u8).collect();
    let vlfeat = colmap_rust::linalg::RowMajorMatrix::from_vec(2, 128, values);
    let ubc = transform_vlfeat_to_ubc_feature_descriptors(&vlfeat);
    assert_eq!((ubc.rows(), ubc.cols()), (2, 128));
    for n in 0..2 {
        for cell in 0..16 {
            let base = 8 * cell;
            assert_eq!(ubc[(n, base)], vlfeat[(n, base)]);
            for k in 1..8 {
                assert_eq!(ubc[(n, base + 8 - k)], vlfeat[(n, base + k)]);
            }
        }
    }
}
