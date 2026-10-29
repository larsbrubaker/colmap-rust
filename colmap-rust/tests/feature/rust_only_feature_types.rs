// Rust-only checks for colmap_rust::feature::types, beyond the 1:1 port in `types.rs`:
// - the enums' COLMAP spellings round-trip and an unknown spelling is MAKE_ENUM_CLASS's error;
// - the `THROW_CHECK`s of the keypoint constructors and `Rescale` return `Err`;
// - descriptor conversions reject UNDEFINED and a byte row that is not whole floats;
// - SIFT float -> u8 saturates out of range and maps NaN to 0 (`FeatureDescriptors::from_float`);
// - the matrix conversions check the column count (the `feature/types.rs` header).

use colmap_rust::feature::types::{
    keypoints_from_matrix, matches_from_matrix, FeatureDescriptors, FeatureDescriptorsFloat,
    FeatureExtractorType, FeatureKeypoint, FeatureMatcherType,
};
use colmap_rust::linalg::RowMajorMatrix;
use colmap_rust::ErrorKind;

#[test]
fn rust_only_feature_enum_strings_round_trip() {
    for t in FeatureExtractorType::ALL {
        assert_eq!(
            FeatureExtractorType::from_colmap_str(t.as_str()).unwrap(),
            t
        );
        assert_eq!(t.to_string(), t.as_str());
    }
    for t in FeatureMatcherType::ALL {
        assert_eq!(FeatureMatcherType::from_colmap_str(t.as_str()).unwrap(), t);
    }
    assert_eq!(FeatureExtractorType::AlikedN16Rot.as_str(), "ALIKED_N16ROT");
    assert_eq!(FeatureMatcherType::SiftBruteforce as i32, 0);
    assert_eq!(FeatureMatcherType::LomaG as i32, 9);
    let error = FeatureExtractorType::from_colmap_str("ORB").unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RuntimeError);
    assert_eq!(
        error.message(),
        "Unknown string value: ORB for enum: FeatureExtractorType"
    );
    let error = FeatureMatcherType::from_colmap_str("sift_bruteforce").unwrap_err();
    assert_eq!(
        error.message(),
        "Unknown string value: sift_bruteforce for enum: FeatureMatcherType"
    );
}

#[test]
fn rust_only_feature_keypoint_checks_fail() {
    let error = FeatureKeypoint::from_scale_orientation(1.0, 2.0, -1.0, 0.0).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert!(
        error
            .message()
            .ends_with("Check failed: scale >= 0.0 (-1 vs. 0) "),
        "{}",
        error.message()
    );
    assert!(FeatureKeypoint::from_scale_orientation(1.0, 2.0, f32::NAN, 0.0).is_err());
    assert!(FeatureKeypoint::from_shape_parameters(1.0, 2.0, 1.0, -1.0, 0.0, 0.0).is_err());
    let mut keypoint = FeatureKeypoint::new(1.0, 2.0);
    assert!(keypoint.rescale(0.0).is_err());
    assert!(keypoint.rescale_xy(1.0, -2.0).is_err());
    // A failed check leaves the keypoint unchanged.
    assert_eq!(keypoint, FeatureKeypoint::new(1.0, 2.0));
    keypoint.rescale(3.0).unwrap();
    assert_eq!(
        keypoint,
        FeatureKeypoint::from_affine(3.0, 6.0, 3.0, 0.0, 0.0, 3.0)
    );
}

#[test]
fn rust_only_feature_descriptors_reject_unsupported_type() {
    let bytes = FeatureDescriptors::new(
        FeatureExtractorType::Undefined,
        RowMajorMatrix::from_vec(1, 4, vec![1, 2, 3, 4]),
    );
    let error = bytes.to_float().unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert!(error
        .message()
        .ends_with("Unsupported feature type: UNDEFINED"));
    let floats = FeatureDescriptorsFloat::new(
        FeatureExtractorType::Undefined,
        RowMajorMatrix::from_vec(1, 1, vec![1.0]),
    );
    assert!(floats.to_bytes().is_err());
}

#[test]
fn rust_only_feature_descriptors_bytes_must_be_whole_floats() {
    let bytes = FeatureDescriptors::new(
        FeatureExtractorType::LomaB,
        RowMajorMatrix::from_vec(1, 6, vec![0; 6]),
    );
    let error = bytes.to_float().unwrap_err();
    assert!(
        error.message().contains("Check failed"),
        "{}",
        error.message()
    );
}

#[test]
fn rust_only_feature_descriptors_float_bytes_are_native_order() {
    let floats = FeatureDescriptorsFloat::new(
        FeatureExtractorType::AlikedN32,
        RowMajorMatrix::from_vec(2, 1, vec![1.5, -2.0]),
    );
    let bytes = floats.to_bytes().unwrap();
    assert_eq!(bytes.data.rows(), 2);
    assert_eq!(bytes.data.cols(), 4);
    assert_eq!(bytes.data.row(0), 1.5_f32.to_ne_bytes());
    assert_eq!(bytes.data.row(1), (-2.0_f32).to_ne_bytes());
    assert_eq!(bytes.to_float().unwrap(), floats);
}

#[test]
fn rust_only_feature_descriptors_sift_cast_saturates() {
    let floats = FeatureDescriptorsFloat::new(
        FeatureExtractorType::Sift,
        RowMajorMatrix::from_vec(1, 6, vec![12.9, 255.0, 300.0, -4.0, f32::NAN, 0.5]),
    );
    let bytes = floats.to_bytes().unwrap();
    assert_eq!(bytes.data.as_slice(), &[12, 255, 255, 0, 0, 0]);
}

#[test]
fn rust_only_feature_matrix_conversions_check_columns() {
    let keypoints = RowMajorMatrix::from_vec(1, 3, vec![1.0_f32, 2.0, 3.0]);
    assert!(keypoints_from_matrix(&keypoints).is_err());
    let negative_scale = RowMajorMatrix::from_vec(1, 4, vec![1.0_f32, 2.0, -3.0, 0.0]);
    assert!(keypoints_from_matrix(&negative_scale).is_err());
    let matches = RowMajorMatrix::from_vec(1, 3, vec![1_u32, 2, 3]);
    assert!(matches_from_matrix(&matches).is_err());
}
