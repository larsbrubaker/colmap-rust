// Port of colmap/feature/extractor_test.cc (same test names and values) for
// colmap_rust::feature::extractor.
//
// The ONNX extractors (ALIKED, LoMa) are excluded (docs/LICENSE_AUDIT.md), so
// FeatureExtractionOptions has no aliked / loma sub-options, and the GPU extractor is excluded,
// so it has no use_gpu:
// - Copy and CopyAssignment are ported without their aliked / loma lines. The shared_ptr
//   pointer-inequality checks become "the copy is independent": mutating it leaves the
//   original alone (the sub-options are owned values, so a clone is always deep).
// - Move and MoveAssignment are skipped (PORTING_PLAN.md): they only check that moving keeps
//   the loma sub-options' shared_ptr (ONNX options, and shared_ptr use counts).
// - EffMaxImageSize and RequiresRGB run over all five types (they need no extractor).
// - CheckAndRequiresOpenGLWithNoGpu checks RequiresOpenGL over all five types and Check for
//   SIFT; Check of the four ALIKED / LoMa types is skipped: it needs their ONNX sub-options,
//   and Check rejects those types here.

use colmap_rust::feature::extractor::FeatureExtractionOptions;
use colmap_rust::feature::types::FeatureExtractorType;
use colmap_rust::ErrorKind;

const ALL_TYPES: [FeatureExtractorType; 5] = [
    FeatureExtractorType::Sift,
    FeatureExtractorType::AlikedN16Rot,
    FeatureExtractorType::AlikedN32,
    FeatureExtractorType::LomaB,
    FeatureExtractorType::LomaB128,
];

#[test]
fn feature_extraction_options_copy() {
    let mut options = FeatureExtractionOptions::new(FeatureExtractorType::Sift);
    options.max_image_size += 100;
    options.sift.max_num_features += 100;

    let mut copy = options.clone();

    // Verify fields are copied
    assert_eq!(copy.max_image_size, options.max_image_size);
    assert_eq!(copy.sift.max_num_features, options.sift.max_num_features);

    // Verify deep copy (the copy's sub-options are its own).
    copy.sift.max_num_features += 1;
    assert_ne!(copy.sift.max_num_features, options.sift.max_num_features);
}

#[test]
fn feature_extraction_options_eff_max_image_size() {
    let mut options = FeatureExtractionOptions::new(FeatureExtractorType::Sift);

    // When max_image_size is explicitly set, use that value.
    options.max_image_size = 2000;
    for extractor_type in ALL_TYPES {
        options.extractor_type = extractor_type;
        assert_eq!(options.eff_max_image_size().unwrap(), 2000);
    }

    // When max_image_size is non-positive, use type-specific defaults.
    for max_image_size in [-1, 0] {
        options.max_image_size = max_image_size;
        options.extractor_type = FeatureExtractorType::Sift;
        assert_eq!(options.eff_max_image_size().unwrap(), 3200);
        options.extractor_type = FeatureExtractorType::AlikedN16Rot;
        assert_eq!(options.eff_max_image_size().unwrap(), 1600);
        options.extractor_type = FeatureExtractorType::AlikedN32;
        assert_eq!(options.eff_max_image_size().unwrap(), 1600);
        options.extractor_type = FeatureExtractorType::LomaB;
        assert_eq!(options.eff_max_image_size().unwrap(), 1600);
        options.extractor_type = FeatureExtractorType::LomaB128;
        assert_eq!(options.eff_max_image_size().unwrap(), 1600);
    }
}

#[test]
#[allow(clippy::redundant_clone, clippy::self_assignment)]
fn feature_extraction_options_copy_assignment() {
    let mut options = FeatureExtractionOptions::new(FeatureExtractorType::Sift);
    options.max_image_size = 999;
    options.sift.max_num_features += 200;

    // Test copy assignment into a default-constructed instance.
    let mut assigned = FeatureExtractionOptions::new(FeatureExtractorType::Sift);
    assert_ne!(assigned, options);
    assigned.clone_from(&options);

    assert_eq!(assigned.max_image_size, 999);
    assert_eq!(
        assigned.sift.max_num_features,
        options.sift.max_num_features
    );

    // Mutating the copy must not affect the original.
    assigned.sift.max_num_features += 1;
    assert_ne!(
        assigned.sift.max_num_features,
        options.sift.max_num_features
    );

    // Test self-assignment: the value is unchanged.
    let before = options.clone();
    options = options.clone();
    assert_eq!(options, before);
    assert_eq!(options.max_image_size, 999);
}

#[test]
fn feature_extraction_options_requires_rgb() {
    let mut options = FeatureExtractionOptions::new(FeatureExtractorType::Sift);

    let test_cases = [
        (FeatureExtractorType::Sift, false),
        (FeatureExtractorType::AlikedN16Rot, true),
        (FeatureExtractorType::AlikedN32, true),
        (FeatureExtractorType::LomaB, true),
        (FeatureExtractorType::LomaB128, true),
    ];

    for (extractor_type, expected) in test_cases {
        options.extractor_type = extractor_type;
        assert_eq!(options.requires_rgb().unwrap(), expected);
    }
}

#[test]
fn feature_extraction_options_check_and_requires_opengl_with_no_gpu() {
    let mut options = FeatureExtractionOptions::new(FeatureExtractorType::Sift);

    for extractor_type in ALL_TYPES {
        options.extractor_type = extractor_type;
        if extractor_type == FeatureExtractorType::Sift {
            assert!(options.check().unwrap());
        }
        assert!(!options.requires_opengl().unwrap());
    }
}

#[test]
fn rust_only_unknown_extractor_type_is_a_runtime_error() {
    let mut options = FeatureExtractionOptions::new(FeatureExtractorType::Undefined);
    for result in [
        options.requires_rgb(),
        options.requires_opengl(),
        options.eff_max_image_size().map(|_| true),
        options.check(),
    ] {
        let error = result.unwrap_err();
        assert_eq!(error.kind(), ErrorKind::RuntimeError);
        assert_eq!(error.message(), "Unknown feature extractor type: UNDEFINED");
    }
    // A positive max_image_size skips the type's default, so Check reaches the type switch,
    // which fails an unknown type.
    options.max_image_size = 100;
    assert!(!options.check().unwrap());
}

#[test]
fn rust_only_onnx_extractor_types_fail_check() {
    for extractor_type in &ALL_TYPES[1..] {
        let options = FeatureExtractionOptions::new(*extractor_type);
        assert!(!options.check().unwrap(), "{extractor_type}");
    }
}
