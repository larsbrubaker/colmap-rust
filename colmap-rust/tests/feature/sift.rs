// Port of the extraction tests of colmap/feature/sift_test.cc (same test names, values and
// checks) for colmap_rust::feature::sift. TEST_P(SiftCpuExtractionTest, Nominal) is one test
// per parameter row, named after the row: `sift_cpu_extraction_test_nominal_<row>`.
//
// The "Sift" row runs SiftCpuFeatureExtractor. The five Covariant* rows need the covariant
// extractor (VLFeat's covdet) and CreateSiftFeatureExtractor, which are not ported yet
// (PORTING_PLAN.md Phase 9). Skipped (PORTING_PLAN.md): ExtractSiftFeaturesGPU.Nominal
// (SiftGPU, excluded by licence).

use colmap_rust::feature::extractor::{FeatureExtractionOptions, FeatureExtractor};
use colmap_rust::feature::sift::cpu_extractor::SiftCpuFeatureExtractor;
use colmap_rust::feature::types::{FeatureDescriptors, FeatureExtractorType, FeatureKeypoints};
use colmap_rust::sensor::bitmap::{Bitmap, BitmapColor};
use colmap_rust::util::cancellation::CancelToken;

/// `CreateImageWithSquare(size)`: a black grey image with a white centered square.
pub(crate) fn create_image_with_square(size: i32) -> Bitmap {
    let mut bitmap = Bitmap::new(size, size, false);
    bitmap.fill(BitmapColor::new(0, 0, 0)).unwrap();
    for r in size / 2 - size / 8..size / 2 + size / 8 {
        for c in size / 2 - size / 8..size / 2 + size / 8 {
            bitmap.set_pixel(r, c, BitmapColor::gray(255));
        }
    }
    bitmap
}

/// `ValidateKeypoints`.
fn validate_keypoints(keypoints: &FeatureKeypoints, bitmap: &Bitmap) {
    for k in keypoints {
        assert!(k.x >= 0.0);
        assert!(k.y >= 0.0);
        assert!(k.x <= bitmap.width() as f32);
        assert!(k.y <= bitmap.height() as f32);
        assert!(k.compute_scale() > 0.0);
        assert!(f64::from(k.compute_orientation()) > -std::f64::consts::PI);
        assert!(f64::from(k.compute_orientation()) < std::f64::consts::PI);
    }
}

/// `ValidateDescriptorNorms`: `row.cast<float>().norm()` within 1 of 512.
fn validate_descriptor_norms(descriptors: &FeatureDescriptors) {
    assert_eq!(descriptors.extractor_type, FeatureExtractorType::Sift);
    for i in 0..descriptors.data.rows() {
        let sum: f32 = descriptors
            .data
            .row(i)
            .iter()
            .map(|&v| f32::from(v) * f32::from(v))
            .sum();
        assert!((sum.sqrt() - 512.0).abs() < 1.0);
    }
}

/// The body of `TEST_P(SiftCpuExtractionTest, Nominal)` for the non-covariant rows.
fn run_sift_cpu_extraction(upright: bool, expected_keypoints: usize) {
    let bitmap = create_image_with_square(256);

    let mut options = FeatureExtractionOptions::new(FeatureExtractorType::Sift);
    options.sift.upright = upright;
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

    assert_eq!(keypoints.len(), expected_keypoints);
    validate_keypoints(&keypoints, &bitmap);
    assert_eq!(descriptors.data.rows(), expected_keypoints);
    validate_descriptor_norms(&descriptors);
}

#[test]
fn sift_cpu_extraction_test_nominal_sift() {
    run_sift_cpu_extraction(false, 22);
}
