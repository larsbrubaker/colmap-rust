// Port of colmap/feature/index_test.cc (same test names, values and tolerances) for
// colmap_rust::feature::index. TEST_P(ParameterizedFeatureDescriptorIndexTests, Nominal) is
// one test per INSTANTIATE_TEST_SUITE_P row, named after the row:
// `parameterized_feature_descriptor_index_tests_nominal_<extractor>_<count>`. COLMAP's only
// index type is FAISS; here it is the exact search (docs/CPP_DIVERGENCES.md entry 231), so
// the 1000-descriptor rows, where COLMAP's faiss uses its IVF index, run the same search.

use colmap_rust::feature::index::{create_feature_descriptor_index, FeatureDescriptorIndexType};
use colmap_rust::feature::types::{FeatureDescriptorsFloat, FeatureExtractorType};
use colmap_rust::feature::utils::{
    feature_descriptors_to_unsigned_byte, l2_normalize_feature_descriptors,
};
use colmap_rust::linalg::RowMajorMatrix;
use colmap_rust::math::random::random_uniform_real;
use colmap_rust::util::cancellation::CancelToken;
use colmap_rust::ErrorKind;

/// `CreateRandomFeatureDescriptors(type, num_features)` of index_test.cc.
fn create_random_feature_descriptors(
    extractor_type: FeatureExtractorType,
    num_features: usize,
) -> FeatureDescriptorsFloat {
    let mut data = RowMajorMatrix::<f32>::new(num_features, 128);
    for i in 0..num_features {
        for j in 0..128 {
            data[(i, j)] = random_uniform_real(0.0f32, 1.0f32).powi(2);
        }
    }
    l2_normalize_feature_descriptors(&mut data);
    if extractor_type == FeatureExtractorType::Sift {
        // Mimic the real SIFT pipeline: convert to uint8 [0,255] then back to float. This
        // ensures values are integer-valued floats in [0, 255], which is required for
        // QT_8bit_direct quantization.
        let bytes = feature_descriptors_to_unsigned_byte(&data);
        data = RowMajorMatrix::from_vec(
            bytes.rows(),
            bytes.cols(),
            bytes.as_slice().iter().map(|&v| f32::from(v)).collect(),
        );
    }
    FeatureDescriptorsFloat::new(extractor_type, data)
}

fn squared_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// The body of `TEST_P(ParameterizedFeatureDescriptorIndexTests, Nominal)`.
fn run_nominal(extractor_type: FeatureExtractorType, num_descriptors: usize) {
    let mut index = create_feature_descriptor_index(FeatureDescriptorIndexType::FAISS, 1);
    let cancel = CancelToken::new();

    let index_descriptors = create_random_feature_descriptors(extractor_type, num_descriptors);
    let query_descriptors = &index_descriptors;
    index.build(&index_descriptors).unwrap();

    let mut indices = RowMajorMatrix::<i32>::new(0, 0);
    let mut distances = RowMajorMatrix::<f32>::new(0, 0);
    index
        .search(1, query_descriptors, &mut indices, &mut distances, &cancel)
        .unwrap();

    assert_eq!(indices.rows(), query_descriptors.data.rows());
    assert_eq!(indices.cols(), 1);
    assert_eq!(distances.rows(), query_descriptors.data.rows());
    assert_eq!(distances.cols(), 1);

    for i in 0..query_descriptors.data.rows() {
        assert_eq!(indices[(i, 0)], i as i32);
        assert!((distances[(i, 0)] - 0.0).abs() <= 1e-6);
    }

    index
        .search(2, query_descriptors, &mut indices, &mut distances, &cancel)
        .unwrap();
    assert_eq!(indices.rows(), query_descriptors.data.rows());
    assert_eq!(indices.cols(), 2);
    assert_eq!(distances.rows(), query_descriptors.data.rows());
    assert_eq!(distances.cols(), 2);

    for i in 0..query_descriptors.data.rows() {
        assert_eq!(indices[(i, 0)], i as i32);
        assert!((distances[(i, 0)] - 0.0).abs() <= 1e-6);
        assert_ne!(indices[(i, 1)], i as i32);
        let expected = squared_distance(
            query_descriptors.data.row(i),
            index_descriptors.data.row(indices[(i, 1)] as usize),
        );
        assert!((distances[(i, 1)] - expected).abs() <= 1e-6);
    }

    index
        .search(
            index_descriptors.data.rows() as i32 + 1,
            query_descriptors,
            &mut indices,
            &mut distances,
            &cancel,
        )
        .unwrap();
    assert_eq!(indices.rows(), query_descriptors.data.rows());
    assert_eq!(indices.cols(), index_descriptors.data.rows());
    assert_eq!(distances.rows(), query_descriptors.data.rows());
    assert_eq!(distances.cols(), index_descriptors.data.rows());
}

#[test]
fn parameterized_feature_descriptor_index_tests_nominal_sift_100() {
    run_nominal(FeatureExtractorType::Sift, 100);
}

#[test]
fn parameterized_feature_descriptor_index_tests_nominal_sift_1000() {
    run_nominal(FeatureExtractorType::Sift, 1000);
}

#[test]
fn parameterized_feature_descriptor_index_tests_nominal_aliked_n16rot_100() {
    run_nominal(FeatureExtractorType::AlikedN16Rot, 100);
}

#[test]
fn parameterized_feature_descriptor_index_tests_nominal_aliked_n16rot_1000() {
    run_nominal(FeatureExtractorType::AlikedN16Rot, 1000);
}

#[test]
fn feature_descriptor_index_tests_type_mismatch() {
    const NUM_DESCRIPTORS: usize = 100;

    let mut index = create_feature_descriptor_index(FeatureDescriptorIndexType::FAISS, 1);

    // Prepare SIFT descriptors for index build.
    let sift_desc = create_random_feature_descriptors(FeatureExtractorType::Sift, NUM_DESCRIPTORS);

    // Prepare descriptors with a different type.
    let aliked_desc =
        create_random_feature_descriptors(FeatureExtractorType::AlikedN16Rot, NUM_DESCRIPTORS);

    // Build should throw when descriptor types are inconsistent.
    index.build(&aliked_desc).unwrap();

    // Build correctly with SIFT so we can test Query mismatch.
    index.build(&sift_desc).unwrap();

    // Query should throw when descriptor types are inconsistent.
    let mut indices = RowMajorMatrix::<i32>::new(0, 0);
    let mut distances = RowMajorMatrix::<f32>::new(0, 0);
    let error = index
        .search(
            1,
            &aliked_desc,
            &mut indices,
            &mut distances,
            &CancelToken::new(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
}

#[test]
fn rust_only_index_search_edge_cases_and_cancellation() {
    let mut index = create_feature_descriptor_index(FeatureDescriptorIndexType::Default, 1);
    let cancel = CancelToken::new();
    let descriptors = create_random_feature_descriptors(FeatureExtractorType::Sift, 20);
    let mut indices = RowMajorMatrix::<i32>::new(3, 3);
    let mut distances = RowMajorMatrix::<f32>::new(3, 3);

    // An empty index (or no neighbours asked) gives 0 x 0 outputs.
    index
        .build(&FeatureDescriptorsFloat::new(
            FeatureExtractorType::Sift,
            RowMajorMatrix::new(0, 128),
        ))
        .unwrap();
    index
        .search(2, &descriptors, &mut indices, &mut distances, &cancel)
        .unwrap();
    assert_eq!((indices.rows(), indices.cols()), (0, 0));
    assert_eq!((distances.rows(), distances.cols()), (0, 0));

    // No queries leave the outputs untouched, as COLMAP does.
    index.build(&descriptors).unwrap();
    let mut indices = RowMajorMatrix::<i32>::new(3, 3);
    let no_queries =
        FeatureDescriptorsFloat::new(FeatureExtractorType::Sift, RowMajorMatrix::new(0, 128));
    index
        .search(2, &no_queries, &mut indices, &mut distances, &cancel)
        .unwrap();
    assert_eq!((indices.rows(), indices.cols()), (3, 3));

    // A width mismatch is a failed check.
    let narrow =
        FeatureDescriptorsFloat::new(FeatureExtractorType::Sift, RowMajorMatrix::new(1, 64));
    let error = index
        .search(2, &narrow, &mut indices, &mut distances, &cancel)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);

    // A cancelled search fails and leaves the outputs as they were.
    cancel.cancel();
    let error = index
        .search(2, &descriptors, &mut indices, &mut distances, &cancel)
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
    assert_eq!((indices.rows(), indices.cols()), (3, 3));
}

#[test]
fn rust_only_index_float_queries_on_byte_index_and_large_k_order() {
    // SIFT bytes indexed, queried with non-integral SIFT floats: the float path runs on the
    // converted index rows and agrees with the definition, ordered by (distance, index).
    let mut index = create_feature_descriptor_index(FeatureDescriptorIndexType::Default, 1);
    let cancel = CancelToken::new();
    let descriptors = create_random_feature_descriptors(FeatureExtractorType::Sift, 30);
    index.build(&descriptors).unwrap();
    let mut queries = descriptors.clone();
    for v in queries.data.as_mut_slice() {
        *v += 0.25;
    }
    let mut indices = RowMajorMatrix::<i32>::new(0, 0);
    let mut distances = RowMajorMatrix::<f32>::new(0, 0);
    // k = 12 > 8 takes the full-sort selection.
    index
        .search(12, &queries, &mut indices, &mut distances, &cancel)
        .unwrap();
    for q in 0..queries.data.rows() {
        let mut expected: Vec<(f32, i32)> = (0..descriptors.data.rows())
            .map(|i| {
                let d = squared_distance(queries.data.row(q), descriptors.data.row(i));
                (d, i as i32)
            })
            .collect();
        expected.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        for j in 0..12 {
            assert_eq!(indices[(q, j)], expected[j].1);
            assert_eq!(distances[(q, j)], expected[j].0);
        }
    }
}
