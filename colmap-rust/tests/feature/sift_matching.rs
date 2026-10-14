// Port of the CPU matcher tests of colmap/feature/sift_test.cc (same test names, values and
// checks) for colmap_rust::feature::sift::cpu_matcher, with the helpers they share
// (CreateRandomFeatureDescriptors, the descriptor-index cache helper, ...). The guided-matching
// geometry cases are in sift_guided_matching.rs (the same .cc file, split for the 800-line
// limit).
//
// Skipped (PORTING_PLAN.md): every GPU case - CreateSiftGPUMatcherOpenGL / CUDA,
// MatchSiftFeaturesGPU.*, MatchGuidedSiftFeaturesGPU.*, MatchSiftFeaturesCPUvsGPU and
// MatchGuidedSiftFeaturesCPUvsGPUGuided (SiftGPU, excluded by licence).
//

use std::collections::BTreeMap;
use std::sync::Arc;

use colmap_rust::feature::index::{
    create_feature_descriptor_index, FeatureDescriptorIndex, FeatureDescriptorIndexType,
};
use colmap_rust::feature::matcher::{FeatureMatcher, FeatureMatcherImage, FeatureMatchingOptions};
use colmap_rust::feature::sift::cpu_matcher::create_sift_feature_matcher;
use colmap_rust::feature::types::{
    FeatureDescriptors, FeatureExtractorType, FeatureKeypoint, FeatureMatcherType, FeatureMatches,
};
use colmap_rust::feature::utils::{
    feature_descriptors_to_unsigned_byte, l2_normalize_feature_descriptors,
};
use colmap_rust::linalg::{Matrix3d, RowMajorMatrix};
use colmap_rust::math::random::{libcxx, set_prng_seed, with_prng};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::util::cache::ThreadSafeLRUCache;
use colmap_rust::util::cancellation::CancelToken;
use colmap_rust::util::types::ImageId;
use colmap_rust::ErrorKind;

/// `CreateEmptyDescriptors()`: zero SIFT descriptors.
pub(crate) fn create_empty_descriptors() -> FeatureDescriptors {
    FeatureDescriptors::new(FeatureExtractorType::Sift, RowMajorMatrix::new(0, 128))
}

/// `CreateReversedDescriptors(src)`: `src.data.colwise().reverse()`, i.e. the rows in reverse
/// order.
pub(crate) fn create_reversed_descriptors(src: &FeatureDescriptors) -> FeatureDescriptors {
    FeatureDescriptors::new(src.extractor_type, reversed_rows(&src.data))
}

fn reversed_rows(data: &RowMajorMatrix<u8>) -> RowMajorMatrix<u8> {
    let mut values = Vec::with_capacity(data.len());
    for r in (0..data.rows()).rev() {
        values.extend_from_slice(data.row(r));
    }
    RowMajorMatrix::from_vec(data.rows(), data.cols(), values)
}

/// `CreateRandomFeatureDescriptors(num_features)` of sift_test.cc: ten random dimensions per
/// row set to 1, L2-normalized and quantized, from the PRNG seeded with 0.
pub(crate) fn create_random_feature_descriptors(num_features: usize) -> FeatureDescriptors {
    set_prng_seed(0);
    let mut descriptors_float = RowMajorMatrix::<f32>::new(num_features, 128);
    let mut dims: Vec<usize> = (0..128).collect();
    for i in 0..num_features {
        with_prng(|g| libcxx::shuffle(&mut dims, g));
        for &dim in &dims[..10] {
            descriptors_float[(i, dim)] = 1.0;
        }
    }
    l2_normalize_feature_descriptors(&mut descriptors_float);
    FeatureDescriptors::new(
        FeatureExtractorType::Sift,
        feature_descriptors_to_unsigned_byte(&descriptors_float),
    )
}

/// `CheckEqualMatches`.
fn check_equal_matches(matches1: &FeatureMatches, matches2: &FeatureMatches) {
    assert_eq!(matches1.len(), matches2.len());
    for (m1, m2) in matches1.iter().zip(matches2) {
        assert_eq!(m1.point2d_idx1, m2.point2d_idx1);
        assert_eq!(m1.point2d_idx2, m2.point2d_idx2);
    }
}

/// `ExpectReversedMatches`.
pub(crate) fn expect_reversed_matches(matches: &FeatureMatches) {
    assert_eq!(matches.len(), 2);
    assert_eq!(matches[0].point2d_idx1, 0);
    assert_eq!(matches[0].point2d_idx2, 1);
    assert_eq!(matches[1].point2d_idx1, 1);
    assert_eq!(matches[1].point2d_idx2, 0);
}

/// `ExpectReversedInlierMatches`.
pub(crate) fn expect_reversed_inlier_matches(two_view_geometry: &TwoViewGeometry) {
    expect_reversed_matches(&two_view_geometry.inlier_matches);
}

/// `CreatePlanarTwoViewGeometry`.
fn create_planar_two_view_geometry() -> TwoViewGeometry {
    TwoViewGeometry {
        config: ConfigurationType::PlanarOrPanoramic,
        h: Some(Matrix3d::identity()),
        ..TwoViewGeometry::default()
    }
}

/// `FeatureDescriptorIndexCacheHelper`: a `ThreadSafeLRUCache` of 100 indexes, each built from
/// its image's descriptors on first use. Returns the cache itself (COLMAP's helper owns it as
/// its `index_cache` member).
pub(crate) fn feature_descriptor_index_cache_helper(
    images: &[FeatureMatcherImage<'_>],
) -> Arc<ThreadSafeLRUCache<ImageId, dyn FeatureDescriptorIndex>> {
    let image_descriptors: BTreeMap<ImageId, FeatureDescriptors> = images
        .iter()
        .map(|image| (image.image_id, image.descriptors.unwrap().clone()))
        .collect();
    let cache = ThreadSafeLRUCache::new(100, move |image_id: &ImageId| {
        let mut index = create_feature_descriptor_index(FeatureDescriptorIndexType::Default, 1);
        index.build(&image_descriptors[image_id].to_float()?)?;
        let index: Arc<dyn FeatureDescriptorIndex> = Arc::from(index);
        Ok(index)
    });
    Arc::new(cache.unwrap())
}

/// The matcher factory of the `MatchGuidedSiftFeaturesCPU` tests: a CPU matcher over an index
/// cache of `images`.
pub(crate) fn create_cpu_matcher_with_index_cache(
    images: &[FeatureMatcherImage<'_>],
) -> Box<dyn FeatureMatcher> {
    let mut options = FeatureMatchingOptions::new(FeatureMatcherType::SiftBruteforce);
    options.sift.cpu_descriptor_index_cache = Some(feature_descriptor_index_cache_helper(images));
    create_sift_feature_matcher(&options).unwrap()
}

pub(crate) fn simple_pinhole_camera() -> Camera {
    Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 100.0, 100, 200).unwrap()
}

fn image<'a>(
    image_id: ImageId,
    camera: &'a Camera,
    keypoints: Option<&'a [FeatureKeypoint]>,
    descriptors: &'a FeatureDescriptors,
) -> FeatureMatcherImage<'a> {
    FeatureMatcherImage {
        image_id,
        camera: Some(camera),
        keypoints,
        descriptors: Some(descriptors),
        pose_prior: None,
    }
}

/// `FeatureKeypoint(x, y)`.
pub(crate) fn kp(x: f32, y: f32) -> FeatureKeypoint {
    FeatureKeypoint::new(x, y)
}

#[test]
fn sift_cpu_feature_matcher_nominal() {
    let camera = simple_pinhole_camera();
    let descriptors0 = create_empty_descriptors();
    let descriptors1 = create_random_feature_descriptors(2);
    let descriptors2 = create_reversed_descriptors(&descriptors1);
    let image0 = image(0, &camera, None, &descriptors0);
    let image1 = image(1, &camera, None, &descriptors1);
    let image2 = image(2, &camera, None, &descriptors2);

    let index_cache_helper = feature_descriptor_index_cache_helper(&[image0, image1, image2]);

    let mut options = FeatureMatchingOptions::new(FeatureMatcherType::SiftBruteforce);
    options.sift.cpu_brute_force_matcher = false;
    options.sift.cpu_descriptor_index_cache = Some(index_cache_helper);
    let mut matcher = create_sift_feature_matcher(&options).unwrap();
    let cancel = CancelToken::new();

    let mut matches = FeatureMatches::new();
    matcher
        .match_features(&image1, &image2, &mut matches, &cancel)
        .unwrap();
    expect_reversed_matches(&matches);

    matcher
        .match_features(&image1, &image2, &mut matches, &cancel)
        .unwrap();
    expect_reversed_matches(&matches);

    matcher
        .match_features(&image0, &image2, &mut matches, &cancel)
        .unwrap();
    assert_eq!(matches.len(), 0);
    matcher
        .match_features(&image1, &image0, &mut matches, &cancel)
        .unwrap();
    assert_eq!(matches.len(), 0);
    matcher
        .match_features(&image0, &image0, &mut matches, &cancel)
        .unwrap();
    assert_eq!(matches.len(), 0);
}

#[test]
fn sift_cpu_feature_matcher_type_mismatch() {
    let camera = simple_pinhole_camera();

    let sift_desc = create_random_feature_descriptors(2);
    assert_eq!(sift_desc.extractor_type, FeatureExtractorType::Sift);

    let mut undefined_desc = create_random_feature_descriptors(2);
    undefined_desc.extractor_type = FeatureExtractorType::Undefined;

    let image_sift = image(1, &camera, None, &sift_desc);
    let image_undefined = image(2, &camera, None, &undefined_desc);

    let mut options = FeatureMatchingOptions::new(FeatureMatcherType::SiftBruteforce);
    options.sift.cpu_brute_force_matcher = true;
    let mut matcher = create_sift_feature_matcher(&options).unwrap();

    let mut matches = FeatureMatches::new();
    let error = matcher
        .match_features(
            &image_sift,
            &image_undefined,
            &mut matches,
            &CancelToken::new(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
}

#[test]
fn match_guided_sift_features_cpu_type_mismatch() {
    let camera = simple_pinhole_camera();

    let sift_desc = create_random_feature_descriptors(2);
    assert_eq!(sift_desc.extractor_type, FeatureExtractorType::Sift);

    let mut undefined_desc = create_random_feature_descriptors(2);
    undefined_desc.extractor_type = FeatureExtractorType::Undefined;

    let keypoints_sift = [kp(1.0, 0.0), kp(2.0, 0.0)];
    let keypoints_undefined = [kp(2.0, 0.0), kp(1.0, 0.0)];
    let image_sift = image(1, &camera, Some(&keypoints_sift), &sift_desc);
    let image_undefined = image(2, &camera, Some(&keypoints_undefined), &undefined_desc);

    let mut options = FeatureMatchingOptions::new(FeatureMatcherType::SiftBruteforce);
    options.sift.cpu_brute_force_matcher = true;
    let mut matcher = create_sift_feature_matcher(&options).unwrap();

    let mut two_view_geometry = create_planar_two_view_geometry();

    let error = matcher
        .match_guided(
            1.0,
            &image_sift,
            &image_undefined,
            &mut two_view_geometry,
            &CancelToken::new(),
        )
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
}

/// The `TestFaissVsBruteForce` lambda: the brute-force and index matchers agree on every
/// pair; returns the number of matches between `descriptors1` and `descriptors2`.
fn test_faiss_vs_brute_force(
    options: &FeatureMatchingOptions,
    descriptors1: &FeatureDescriptors,
    descriptors2: &FeatureDescriptors,
) -> usize {
    let camera = simple_pinhole_camera();
    let descriptors0 = create_empty_descriptors();
    let image0 = image(0, &camera, None, &descriptors0);
    let image1 = image(1, &camera, None, descriptors1);
    let image2 = image(2, &camera, None, descriptors2);

    let index_cache_helper = feature_descriptor_index_cache_helper(&[image0, image1, image2]);

    let mut matches_bf = FeatureMatches::new();
    let mut matches_faiss = FeatureMatches::new();

    let mut custom_options = options.clone();
    custom_options.sift.cpu_brute_force_matcher = true;
    let mut bf_matcher = create_sift_feature_matcher(&custom_options).unwrap();
    custom_options.sift.cpu_brute_force_matcher = false;
    custom_options.sift.cpu_descriptor_index_cache = Some(index_cache_helper);
    let mut faiss_matcher = create_sift_feature_matcher(&custom_options).unwrap();
    let cancel = CancelToken::new();

    let mut run = |a: &FeatureMatcherImage<'_>, b: &FeatureMatcherImage<'_>| {
        bf_matcher
            .match_features(a, b, &mut matches_bf, &cancel)
            .unwrap();
        faiss_matcher
            .match_features(a, b, &mut matches_faiss, &cancel)
            .unwrap();
        check_equal_matches(&matches_bf, &matches_faiss);
        matches_bf.len()
    };

    let num_matches = run(&image1, &image2);
    run(&image0, &image2);
    run(&image1, &image0);
    run(&image0, &image0);
    num_matches
}

/// `data.row(r) = FeatureDescriptorsToUnsignedByte(data.row(r).cast<float>().normalized())`.
fn renormalize_row(data: &mut RowMajorMatrix<u8>, r: usize) {
    let row: Vec<f32> = data.row(r).iter().map(|&v| f32::from(v)).collect();
    let mut row = RowMajorMatrix::from_vec(1, row.len(), row);
    l2_normalize_feature_descriptors(&mut row);
    let bytes = feature_descriptors_to_unsigned_byte(&row);
    data.row_mut(r).copy_from_slice(bytes.row(0));
}

fn top_rows(descriptors: &FeatureDescriptors, n: usize) -> FeatureDescriptors {
    let cols = descriptors.data.cols();
    FeatureDescriptors::new(
        descriptors.extractor_type,
        RowMajorMatrix::from_vec(n, cols, descriptors.data.as_slice()[..n * cols].to_vec()),
    )
}

#[test]
fn sift_cpu_feature_matcher_faiss_vs_brute_force_nominal() {
    {
        let descriptors1 = create_random_feature_descriptors(50);
        let descriptors2 = create_random_feature_descriptors(50);
        let match_options = FeatureMatchingOptions::default();
        test_faiss_vs_brute_force(&match_options, &descriptors1, &descriptors2);
    }

    {
        let descriptors1 = create_random_feature_descriptors(50);
        let descriptors2 = create_reversed_descriptors(&descriptors1);
        let match_options = FeatureMatchingOptions::default();
        let num_matches = test_faiss_vs_brute_force(&match_options, &descriptors1, &descriptors2);
        assert_eq!(num_matches, 50);
    }

    // Check the ratio test.
    {
        let descriptors1 = create_random_feature_descriptors(50);
        let mut descriptors2 = descriptors1.clone();

        let mut match_options = FeatureMatchingOptions::default();
        let num_matches1 = test_faiss_vs_brute_force(&match_options, &descriptors1, &descriptors2);
        assert_eq!(num_matches1, 50);

        let row0 = descriptors2.data.row(0).to_vec();
        descriptors2.data.row_mut(49).copy_from_slice(&row0);
        // uint8 `+=` wraps, as in C++ (no value here reaches 256).
        descriptors2.data[(0, 0)] = descriptors2.data[(0, 0)].wrapping_add(50);
        renormalize_row(&mut descriptors2.data, 0);
        descriptors2.data[(49, 0)] = descriptors2.data[(49, 0)].wrapping_add(100);
        renormalize_row(&mut descriptors2.data, 49);

        match_options.sift.max_ratio = 0.4;
        let descriptors1_top49 = top_rows(&descriptors1, 49);
        let num_matches2 =
            test_faiss_vs_brute_force(&match_options, &descriptors1_top49, &descriptors2);
        assert_eq!(num_matches2, 48);

        match_options.sift.max_ratio = 0.6;
        let num_matches3 = test_faiss_vs_brute_force(&match_options, &descriptors1, &descriptors2);
        assert_eq!(num_matches3, 49);
    }

    // Check the cross check.
    {
        let mut descriptors1 = create_random_feature_descriptors(50);
        let descriptors2 = descriptors1.clone();
        let row1 = descriptors1.data.row(1).to_vec();
        descriptors1.data.row_mut(0).copy_from_slice(&row1);

        let mut match_options = FeatureMatchingOptions::default();

        match_options.sift.cross_check = false;
        let num_matches1 = test_faiss_vs_brute_force(&match_options, &descriptors1, &descriptors2);
        assert_eq!(num_matches1, 50);

        match_options.sift.cross_check = true;
        let num_matches2 = test_faiss_vs_brute_force(&match_options, &descriptors1, &descriptors2);
        assert_eq!(num_matches2, 48);
    }
}

#[test]
fn match_guided_sift_features_cpu_nominal() {
    let camera = simple_pinhole_camera();
    let descriptors0 = create_empty_descriptors();
    let descriptors1 = create_random_feature_descriptors(2);
    let descriptors2 = create_reversed_descriptors(&descriptors1);
    let descriptors3 = create_random_feature_descriptors(2);
    let keypoints0: [FeatureKeypoint; 0] = [];
    let keypoints1 = [kp(1.0, 0.0), kp(2.0, 0.0)];
    let keypoints2 = [kp(2.0, 0.0), kp(1.0, 0.0)];
    let keypoints3 = [kp(100.0, 0.0), kp(2.0, 0.0)];
    let image0 = image(0, &camera, Some(&keypoints0), &descriptors0);
    let image1 = image(1, &camera, Some(&keypoints1), &descriptors1);
    let image2 = image(2, &camera, Some(&keypoints2), &descriptors2);
    let image3 = image(3, &camera, Some(&keypoints3), &descriptors3);

    let mut matcher = create_cpu_matcher_with_index_cache(&[image0, image1, image2, image3]);
    let mut two_view_geometry = create_planar_two_view_geometry();
    let cancel = CancelToken::new();

    const MAX_ERROR: f64 = 1.0;

    let mut run = |a: &FeatureMatcherImage<'_>, b: &FeatureMatcherImage<'_>| {
        matcher
            .match_guided(MAX_ERROR, a, b, &mut two_view_geometry, &cancel)
            .unwrap();
        two_view_geometry.inlier_matches.clone()
    };

    expect_reversed_matches(&run(&image1, &image2));

    let matches = run(&image3, &image2);
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].point2d_idx1, 1);
    assert_eq!(matches[0].point2d_idx2, 0);

    assert_eq!(run(&image0, &image2).len(), 0);
    assert_eq!(run(&image1, &image0).len(), 0);
    assert_eq!(run(&image0, &image0).len(), 0);
}

#[test]
fn rust_only_sift_cpu_matcher_cancellation() {
    let camera = simple_pinhole_camera();
    let descriptors1 = create_random_feature_descriptors(20);
    let descriptors2 = create_reversed_descriptors(&descriptors1);
    let image1 = image(1, &camera, None, &descriptors1);
    let image2 = image(2, &camera, None, &descriptors2);
    let cancel = CancelToken::new();
    cancel.cancel();
    for brute_force in [false, true] {
        let mut options = FeatureMatchingOptions::new(FeatureMatcherType::SiftBruteforce);
        options.sift.cpu_brute_force_matcher = brute_force;
        options.sift.cpu_descriptor_index_cache =
            Some(feature_descriptor_index_cache_helper(&[image1, image2]));
        let mut matcher = create_sift_feature_matcher(&options).unwrap();
        let mut matches = vec![Default::default(); 3];
        let error = matcher
            .match_features(&image1, &image2, &mut matches, &cancel)
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert!(matches.is_empty());
    }
}

#[test]
fn rust_only_sift_cpu_matcher_index_path_needs_a_cache() {
    let camera = simple_pinhole_camera();
    let descriptors = create_random_feature_descriptors(2);
    let image1 = image(1, &camera, None, &descriptors);
    let options = FeatureMatchingOptions::new(FeatureMatcherType::SiftBruteforce);
    let mut matcher = create_sift_feature_matcher(&options).unwrap();
    let mut matches = FeatureMatches::new();
    let error = matcher
        .match_features(&image1, &image1, &mut matches, &CancelToken::new())
        .unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
}
