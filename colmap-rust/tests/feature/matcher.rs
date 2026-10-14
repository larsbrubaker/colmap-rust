// Port of colmap/feature/matcher_test.cc (same test names and values) for
// colmap_rust::feature::matcher.
//
// The ONNX matchers (LightGlue, ALIKED, LoMa) are excluded (docs/LICENSE_AUDIT.md), so
// FeatureMatchingOptions has no aliked / loma sub-options, and the GPU matcher is excluded, so
// it has no use_gpu:
// - Copy and CopyAssignment are ported without their aliked lines. The shared_ptr
//   pointer-inequality checks become "the copy is independent": mutating it leaves the
//   original alone (the sub-options are owned values, so a clone is always deep).
// - RequiresOpenGL runs over all ten types (it needs no matcher).
// - Check runs for SIFT_BRUTEFORCE; Check of the nine ONNX types is skipped: it needs their
//   ONNX sub-options, and Check rejects those types here (rust_only test below).

use colmap_rust::feature::matcher::{create_feature_matcher, FeatureMatchingOptions};
use colmap_rust::feature::types::FeatureMatcherType;
use colmap_rust::ErrorKind;

const TYPES: [FeatureMatcherType; 10] = [
    FeatureMatcherType::SiftBruteforce,
    FeatureMatcherType::SiftLightglue,
    FeatureMatcherType::AlikedBruteforce,
    FeatureMatcherType::AlikedLightglue,
    FeatureMatcherType::LomaBruteforce,
    FeatureMatcherType::LomaB,
    FeatureMatcherType::LomaB128,
    FeatureMatcherType::LomaR,
    FeatureMatcherType::LomaL,
    FeatureMatcherType::LomaG,
];

#[test]
fn feature_matching_options_copy() {
    let mut options = FeatureMatchingOptions::default();
    options.max_num_matches += 100;
    options.sift.max_ratio *= 0.1;

    let mut copy = options.clone();

    // Verify fields are copied
    assert_eq!(copy.max_num_matches, options.max_num_matches);
    assert_eq!(copy.sift.max_ratio, options.sift.max_ratio);

    // Verify deep copy (the copy's sub-options are its own).
    copy.sift.max_ratio *= 2.0;
    assert_ne!(copy.sift.max_ratio, options.sift.max_ratio);
}

#[test]
#[allow(clippy::redundant_clone, clippy::self_assignment)]
fn feature_matching_options_copy_assignment() {
    let mut options = FeatureMatchingOptions::default();
    options.max_num_matches += 100;
    options.sift.max_ratio *= 0.1;

    let mut assigned = FeatureMatchingOptions::default();
    assigned.clone_from(&options);

    // Verify fields are copied.
    assert_eq!(assigned.max_num_matches, options.max_num_matches);
    assert_eq!(assigned.sift.max_ratio, options.sift.max_ratio);

    // Verify deep copy (mutating the copy leaves the original alone).
    assigned.sift.max_ratio *= 2.0;
    assert_ne!(assigned.sift.max_ratio, options.sift.max_ratio);

    // Test self-assignment: the values are unchanged.
    let prev_max_num_matches = options.max_num_matches;
    let prev_max_ratio = options.sift.max_ratio;
    options = options.clone();
    assert_eq!(options.max_num_matches, prev_max_num_matches);
    assert_eq!(options.sift.max_ratio, prev_max_ratio);
}

#[test]
fn feature_matching_options_requires_opengl() {
    let mut options = FeatureMatchingOptions::default();
    for matcher_type in TYPES {
        options.matcher_type = matcher_type;
        assert!(!options.requires_opengl().unwrap());
    }
}

#[test]
fn feature_matching_options_check() {
    let options = FeatureMatchingOptions::new(FeatureMatcherType::SiftBruteforce);
    assert!(options.check());
}

#[test]
fn rust_only_onnx_matcher_types_fail_check_and_create() {
    let mut options = FeatureMatchingOptions::default();
    for matcher_type in &TYPES[1..] {
        options.matcher_type = *matcher_type;
        assert!(!options.check());
        let error = create_feature_matcher(&options).err().unwrap();
        assert_eq!(error.kind(), ErrorKind::RuntimeError);
        assert!(error.message().contains("ONNX"), "{}", error.message());
    }
}

#[test]
fn rust_only_unknown_matcher_type_is_a_runtime_error() {
    let options = FeatureMatchingOptions::new(FeatureMatcherType::Undefined);
    assert!(!options.check());
    let error = options.requires_opengl().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RuntimeError);
    assert_eq!(error.message(), "Unknown feature matcher type: UNDEFINED");
    let error = create_feature_matcher(&options).err().unwrap();
    assert_eq!(error.message(), "Unknown feature matcher type: UNDEFINED");
}

#[test]
fn rust_only_check_rejects_bad_options() {
    let options = FeatureMatchingOptions {
        max_num_matches: -1,
        ..FeatureMatchingOptions::default()
    };
    assert!(!options.check());
    let mut options = FeatureMatchingOptions::default();
    options.sift.max_ratio = 0.0;
    assert!(!options.check());
    let mut options = FeatureMatchingOptions::default();
    options.sift.max_distance = f64::NAN;
    assert!(!options.check());
    // A failed check is the matcher constructor's THROW_CHECK.
    let error = create_feature_matcher(&options).err().unwrap();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
}
