// Port of colmap/feature/utils_test.cc, 1:1 (same test names, values and tolerances), for
// colmap_rust::feature::utils. `FeatureDescriptorsData::Random(r, c)` draws Eigen's
// `std::rand` bytes; `ExtractTopScaleFeatures` only moves rows, so `random_bytes` draws them
// from COLMAP's seeded PRNG instead (as in `types.rs`). The row norms that `EXPECT_NEAR`
// checks are computed in double, a tighter reference than Eigen's float `norm()`.

use colmap_rust::feature::types::{
    FeatureDescriptors, FeatureDescriptorsData, FeatureDescriptorsFloatData, FeatureKeypoint,
    FeatureKeypoints,
};
use colmap_rust::feature::utils::{
    extract_top_scale_features, feature_descriptors_to_unsigned_byte,
    feature_keypoints_to_points_vector, l1_root_normalize_feature_descriptors,
    l2_normalize_feature_descriptors,
};
use colmap_rust::linalg::{RowMajorMatrix, Vector2d};
use colmap_rust::math::random::{random_uniform_integer, set_prng_seed};
use colmap_rust::math::random_eigen::random_eigen_matrix_xf;

/// gtest's `EXPECT_NEAR(a, b, tol)`, evaluated in double like gtest.
fn expect_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
}

/// `descriptors.row(r).norm()`, in double.
fn row_norm(descriptors: &FeatureDescriptorsFloatData, r: usize) -> f64 {
    descriptors
        .row(r)
        .iter()
        .map(|&v| f64::from(v) * f64::from(v))
        .sum::<f64>()
        .sqrt()
}

/// `RandomEigenMatrixXf(rows, cols)` with `descriptors.array() += 1.0f`.
fn random_descriptors_plus_one(rows: usize, cols: usize) -> FeatureDescriptorsFloatData {
    let mut descriptors = random_eigen_matrix_xf(rows, cols);
    for v in descriptors.as_mut_slice() {
        *v += 1.0f32;
    }
    descriptors
}

/// Stand-in for `FeatureDescriptorsData::Random(rows, cols)` (see the header).
fn random_bytes(rows: usize, cols: usize) -> FeatureDescriptorsData {
    set_prng_seed(0);
    let data = (0..rows * cols)
        .map(|_| random_uniform_integer::<u16>(0, 255) as u8)
        .collect();
    RowMajorMatrix::from_vec(rows, cols, data)
}

#[test]
fn feature_keypoints_to_points_vector_nominal() {
    let mut keypoints: FeatureKeypoints = vec![FeatureKeypoint::default(); 2];
    keypoints[1].x = 0.1;
    keypoints[1].y = 0.2;
    let points = feature_keypoints_to_points_vector(&keypoints);
    assert_eq!(points[0], Vector2d::new(0.0, 0.0));
    // points[1].cast<float>() == Eigen::Vector2f(0.1, 0.2).
    assert_eq!(points[1].x as f32, 0.1f64 as f32);
    assert_eq!(points[1].y as f32, 0.2f64 as f32);
}

#[test]
fn l2_normalize_feature_descriptors_nominal() {
    let mut descriptors = random_descriptors_plus_one(100, 128);
    l2_normalize_feature_descriptors(&mut descriptors);
    for r in 0..descriptors.rows() {
        expect_near(row_norm(&descriptors, r), 1.0, 1e-6);
    }
}

#[test]
fn l1_root_normalize_feature_descriptors_nominal() {
    let mut descriptors = random_descriptors_plus_one(100, 128);
    l1_root_normalize_feature_descriptors(&mut descriptors);
    for r in 0..descriptors.rows() {
        expect_near(row_norm(&descriptors, r), 1.0, 1e-6);
    }
}

#[test]
fn feature_descriptors_to_unsigned_byte_nominal() {
    let descriptors = random_descriptors_plus_one(100, 128);
    let descriptors_uint8 = feature_descriptors_to_unsigned_byte(&descriptors);
    for r in 0..descriptors.rows() {
        for c in 0..descriptors.cols() {
            // static_cast<uint8_t>(std::min(255.0f, std::round(512.0f * descriptors(r, c)))).
            let expected = 255.0f32.min((512.0f32 * descriptors[(r, c)]).round()) as u8;
            assert_eq!(expected, descriptors_uint8[(r, c)]);
        }
    }
}

#[test]
fn extract_top_scale_features_nominal() {
    let mut keypoints: FeatureKeypoints = vec![FeatureKeypoint::default(); 5];
    keypoints[0].rescale(3.0).unwrap();
    keypoints[1].rescale(4.0).unwrap();
    keypoints[2].rescale(1.0).unwrap();
    keypoints[3].rescale(5.0).unwrap();
    keypoints[4].rescale(2.0).unwrap();
    let descriptors = FeatureDescriptors {
        data: random_bytes(5, 128),
        ..FeatureDescriptors::default()
    };

    let mut top_keypoints2 = keypoints.clone();
    let mut top_descriptors2 = descriptors.clone();
    extract_top_scale_features(&mut top_keypoints2, &mut top_descriptors2, 2).unwrap();
    assert_eq!(top_keypoints2.len(), 2);
    assert_eq!(
        top_keypoints2[0].compute_scale(),
        keypoints[3].compute_scale()
    );
    assert_eq!(
        top_keypoints2[1].compute_scale(),
        keypoints[1].compute_scale()
    );
    assert_eq!(top_descriptors2.data.rows(), 2);
    assert_eq!(top_descriptors2.data.row(0), descriptors.data.row(3));
    assert_eq!(top_descriptors2.data.row(1), descriptors.data.row(1));

    let mut top_keypoints5 = keypoints.clone();
    let mut top_descriptors5 = descriptors.clone();
    extract_top_scale_features(&mut top_keypoints5, &mut top_descriptors5, 5).unwrap();
    assert_eq!(top_keypoints5.len(), 5);
    assert_eq!(top_descriptors5.data.rows(), 5);
    assert_eq!(top_descriptors5.data, descriptors.data);

    let mut top_keypoints6 = keypoints.clone();
    let mut top_descriptors6 = descriptors.clone();
    extract_top_scale_features(&mut top_keypoints6, &mut top_descriptors6, 6).unwrap();
    // COLMAP checks top_keypoints5 here again (as written in utils_test.cc).
    assert_eq!(top_keypoints5.len(), 5);
    assert_eq!(top_descriptors6.data.rows(), 5);
    assert_eq!(top_descriptors6.data, descriptors.data);
}
