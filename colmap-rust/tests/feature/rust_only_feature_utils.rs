// Rust-only tests for colmap_rust::feature::utils: the Eigen packet order of the row norms
// (ported from colmap-sharp's FeatureUtilsTests CSharpOnly_* tests), NaN for a zero row, the
// quantization's rounding and saturation, the top-scale tie order
// (`extract_top_scale_features`'s doc) and the checks that return `Err`.

use colmap_rust::feature::types::{FeatureDescriptors, FeatureKeypoint, FeatureKeypoints};
use colmap_rust::feature::utils::{
    extract_top_scale_features, feature_descriptors_to_unsigned_byte,
    l1_root_normalize_feature_descriptors, l2_normalize_feature_descriptors,
};
use colmap_rust::linalg::RowMajorMatrix;

/// A 1x128 row with one large value (index 0) among ones, chosen so the three plausible float
/// summation orders give three different sums:
/// - left to right: every "+ 1" onto 2^24 rounds away, so the sum stays 16777216;
/// - Eigen's packet reduction: lane 0 of the first accumulator also absorbs its 15 ones
///   (16777216), lanes 1-3 of it and all lanes of the second collect 16 ones each; adding
///   the accumulators gives lanes (16777232, 32, 32, 32), and (l0 + l2) + (l1 + l3) =
///   16777264 + 64 = 16777328, exact in float;
/// - the true sum, 16777343, is not representable at all.
fn large_value_among_ones(large: f32) -> RowMajorMatrix<f32> {
    let mut descriptors = RowMajorMatrix::from_vec(1, 128, vec![1.0f32; 128]);
    descriptors[(0, 0)] = large;
    descriptors
}

#[test]
fn rust_only_l1_root_normalize_sums_in_eigen_packet_order() {
    let mut descriptors = large_value_among_ones(16777216.0);
    l1_root_normalize_feature_descriptors(&mut descriptors);
    let scale = 1.0f32 / 16777328.0f32;
    assert_eq!(descriptors[(0, 0)], (16777216.0f32 * scale).sqrt());
    assert_eq!(descriptors[(0, 1)], scale.sqrt());
    assert_eq!(descriptors[(0, 127)], scale.sqrt());
}

#[test]
fn rust_only_l2_normalize_sums_in_eigen_packet_order() {
    // The squares are exact (4096^2 = 2^24, 1^2 = 1), so only the order is pinned.
    let mut descriptors = large_value_among_ones(4096.0);
    l2_normalize_feature_descriptors(&mut descriptors);
    let norm = 16777328.0f32.sqrt();
    assert_eq!(descriptors[(0, 0)], 4096.0f32 / norm);
    assert_eq!(descriptors[(0, 1)], 1.0f32 / norm);
    assert_eq!(descriptors[(0, 127)], 1.0f32 / norm);
}

#[test]
fn rust_only_short_rows_sum_left_to_right_then_tail() {
    // Rows shorter than one packet, one packet, and one packet plus a tail.
    for len in [1usize, 3, 4, 6, 9, 13] {
        let values: Vec<f32> = (0..len).map(|i| (i + 1) as f32).collect();
        let mut descriptors = RowMajorMatrix::from_vec(1, len, values.clone());
        l1_root_normalize_feature_descriptors(&mut descriptors);
        let total: f32 = values.iter().sum();
        for (c, &v) in values.iter().enumerate() {
            // Small integers sum exactly in any order.
            assert_eq!(
                descriptors[(0, c)],
                (v * (1.0f32 / total)).sqrt(),
                "len {len}"
            );
        }
    }
}

#[test]
fn rust_only_l2_normalize_zero_row_is_nan() {
    let mut descriptors = RowMajorMatrix::from_vec(1, 8, vec![0.0f32; 8]);
    l2_normalize_feature_descriptors(&mut descriptors);
    assert!(descriptors.as_slice().iter().all(|v| v.is_nan()));
}

#[test]
fn rust_only_unsigned_byte_rounds_half_away_and_saturates() {
    let values = vec![
        0.5f32 / 512.0, // 0.5 rounds away from zero to 1
        1.5 / 512.0,    // 1.5 -> 2
        2.5 / 512.0,    // 2.5 -> 3 (not banker's 2)
        0.6,            // 307.2 -> 255
        -0.1,           // -51.2 -> 0
        f32::NAN,       // TruncateCast maps NaN to the minimum
    ];
    let descriptors = RowMajorMatrix::from_vec(1, values.len(), values);
    let bytes = feature_descriptors_to_unsigned_byte(&descriptors);
    assert_eq!(bytes.as_slice(), &[1, 2, 3, 255, 0, 0]);
}

#[test]
fn rust_only_extract_top_scale_features_ties_keep_input_order() {
    let scales = [2.0f32, 3.0, 2.0, 3.0, 1.0, 2.0];
    let mut keypoints: FeatureKeypoints = scales
        .iter()
        .map(|&s| {
            let mut k = FeatureKeypoint::default();
            k.rescale(s).unwrap();
            k
        })
        .collect();
    let data = RowMajorMatrix::from_vec(6, 1, (0u8..6).collect());
    let mut descriptors = FeatureDescriptors {
        data,
        ..FeatureDescriptors::default()
    };
    extract_top_scale_features(&mut keypoints, &mut descriptors, 4).unwrap();
    // Scale 3 at inputs 1 and 3, then scale 2 at inputs 0 and 2 (input 5 is cut).
    assert_eq!(descriptors.data.as_slice(), &[1, 3, 0, 2]);
    assert_eq!(keypoints.len(), 4);
}

#[test]
fn rust_only_extract_top_scale_features_checks() {
    let mut keypoints: FeatureKeypoints = vec![FeatureKeypoint::default(); 2];
    let mut descriptors = FeatureDescriptors {
        data: RowMajorMatrix::new(3, 4),
        ..FeatureDescriptors::default()
    };
    assert!(extract_top_scale_features(&mut keypoints, &mut descriptors, 1).is_err());

    let mut descriptors = FeatureDescriptors {
        data: RowMajorMatrix::new(2, 4),
        ..FeatureDescriptors::default()
    };
    assert!(extract_top_scale_features(&mut keypoints, &mut descriptors, 0).is_err());
}
