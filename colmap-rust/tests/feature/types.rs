// Port of colmap/feature/types_test.cc, 1:1 (same test names, values and tolerances), for
// colmap_rust::feature::types. COLMAP's overloaded keypoint constructors are the named
// `FeatureKeypoint::new` / `from_scale_orientation` / `from_affine`; `Rescale(x, y)` is
// `rescale_xy`. `FeatureDescriptorsData::Random(r, c)` draws Eigen's `std::rand` bytes, whose
// values these round trips do not depend on, so `random_bytes` draws them from COLMAP's
// seeded PRNG instead. gtest's `EXPECT_FLOAT_EQ` (within 4 ulp) is `expect_float_eq`.

use std::f64::consts::PI as M_PI;

use colmap_rust::feature::types::{
    keypoints_from_matrix, keypoints_to_matrix, matches_from_matrix, matches_to_matrix,
    FeatureDescriptors, FeatureDescriptorsData, FeatureExtractorType, FeatureKeypoint,
    FeatureKeypoints, FeatureMatch, FeatureMatches,
};
use colmap_rust::linalg::RowMajorMatrix;
use colmap_rust::math::random::{random_uniform_integer, set_prng_seed};
use colmap_rust::util::types::INVALID_POINT2D_IDX;

/// gtest's `EXPECT_NEAR(a, b, tol)`, evaluated in double like gtest.
fn expect_near(a: f32, b: f64, tol: f64) {
    let a = f64::from(a);
    assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
}

/// gtest's `EXPECT_FLOAT_EQ`: the two floats are within 4 ulp of each other.
fn expect_float_eq(a: f32, b: f32) {
    // Map the sign-magnitude bit patterns onto a monotonic integer line, as gtest does.
    fn biased(f: f32) -> i64 {
        let bits = f.to_bits();
        if bits & 0x8000_0000 != 0 {
            -i64::from(bits & 0x7fff_ffff)
        } else {
            i64::from(bits)
        }
    }
    assert!(
        (biased(a) - biased(b)).abs() <= 4,
        "{a} and {b} differ by more than 4 ulp"
    );
}

/// `FeatureDescriptorsData::Random(rows, cols)`: arbitrary bytes, from the seeded PRNG.
fn random_bytes(rows: usize, cols: usize) -> FeatureDescriptorsData {
    set_prng_seed(0);
    let data = (0..rows * cols)
        .map(|_| random_uniform_integer::<u16>(0, 255) as u8)
        .collect();
    RowMajorMatrix::from_vec(rows, cols, data)
}

fn kp_scale(x: f32, y: f32, scale: f32, orientation: f64) -> FeatureKeypoint {
    FeatureKeypoint::from_scale_orientation(x, y, scale, orientation as f32).unwrap()
}

fn kp_shape(sx: f32, sy: f32, orientation: f64, shear: f64) -> FeatureKeypoint {
    FeatureKeypoint::from_shape_parameters(1.0, 2.0, sx, sy, orientation as f32, shear as f32)
        .unwrap()
}

fn orientation_is_pi(keypoint: &FeatureKeypoint) -> bool {
    let orientation = f64::from(keypoint.compute_orientation());
    (orientation - M_PI).abs() < 1e-6 || (orientation + M_PI).abs() < 1e-6
}

/// The affine shape and the derived scale / orientation / shear checks that most blocks of
/// `FeatureKeypoints.Nominal` repeat.
#[allow(clippy::too_many_arguments)]
fn expect_shape(
    keypoint: &FeatureKeypoint,
    [a11, a12, a21, a22]: [f64; 4],
    scale: f64,
    scale_x: f64,
    scale_y: f64,
    orientation: Option<f64>,
    shear: f64,
) {
    expect_near(keypoint.a11, a11, 1e-6);
    expect_near(keypoint.a12, a12, 1e-6);
    expect_near(keypoint.a21, a21, 1e-6);
    expect_near(keypoint.a22, a22, 1e-6);
    expect_near(keypoint.compute_scale(), scale, 1e-6);
    expect_near(keypoint.compute_scale_x(), scale_x, 1e-6);
    expect_near(keypoint.compute_scale_y(), scale_y, 1e-6);
    match orientation {
        Some(orientation) => expect_near(keypoint.compute_orientation(), orientation, 1e-6),
        None => assert!(orientation_is_pi(keypoint)),
    }
    expect_near(keypoint.compute_shear(), shear, 1e-6);
}

fn expect_exact(keypoint: &FeatureKeypoint, x: f32, y: f32, affine: [f32; 4]) {
    assert_eq!(keypoint.x, x);
    assert_eq!(keypoint.y, y);
    assert_eq!(keypoint.a11, affine[0]);
    assert_eq!(keypoint.a12, affine[1]);
    assert_eq!(keypoint.a21, affine[2]);
    assert_eq!(keypoint.a22, affine[3]);
}

#[test]
fn feature_keypoints_nominal() {
    let mut keypoint = FeatureKeypoint::default();
    expect_exact(&keypoint, 0.0, 0.0, [1.0, 0.0, 0.0, 1.0]);

    let keypoints: FeatureKeypoints = vec![FeatureKeypoint::default(); 1];
    assert_eq!(keypoints.len(), 1);
    expect_exact(&keypoints[0], 0.0, 0.0, [1.0, 0.0, 0.0, 1.0]);

    keypoint = FeatureKeypoint::new(1.0, 2.0);
    expect_exact(&keypoint, 1.0, 2.0, [1.0, 0.0, 0.0, 1.0]);
    expect_shape(
        &keypoint,
        [1.0, 0.0, 0.0, 1.0],
        1.0,
        1.0,
        1.0,
        Some(0.0),
        0.0,
    );

    keypoint = kp_scale(1.0, 2.0, 0.0, 0.0);
    expect_exact(&keypoint, 1.0, 2.0, [0.0, 0.0, 0.0, 0.0]);

    keypoint = kp_scale(1.0, 2.0, 1.0, 0.0);
    expect_exact(&keypoint, 1.0, 2.0, [1.0, 0.0, 0.0, 1.0]);
    expect_shape(
        &keypoint,
        [1.0, 0.0, 0.0, 1.0],
        1.0,
        1.0,
        1.0,
        Some(0.0),
        0.0,
    );

    keypoint = kp_scale(1.0, 2.0, 1.0, M_PI / 2.0);
    assert_eq!(keypoint.x, 1.0);
    assert_eq!(keypoint.y, 2.0);
    expect_shape(
        &keypoint,
        [0.0, -1.0, 1.0, 0.0],
        1.0,
        1.0,
        1.0,
        Some(M_PI / 2.0),
        0.0,
    );

    keypoint = kp_scale(1.0, 2.0, 2.0, M_PI / 2.0);
    assert_eq!(keypoint.x, 1.0);
    assert_eq!(keypoint.y, 2.0);
    expect_shape(
        &keypoint,
        [0.0, -2.0, 2.0, 0.0],
        2.0,
        2.0,
        2.0,
        Some(M_PI / 2.0),
        0.0,
    );

    keypoint = kp_scale(1.0, 2.0, 2.0, M_PI);
    assert_eq!(keypoint.x, 1.0);
    assert_eq!(keypoint.y, 2.0);
    expect_shape(&keypoint, [-2.0, 0.0, 0.0, -2.0], 2.0, 2.0, 2.0, None, 0.0);

    keypoint = kp_shape(2.0, 2.0, M_PI, 0.0);
    assert_eq!(keypoint.x, 1.0);
    assert_eq!(keypoint.y, 2.0);
    expect_shape(&keypoint, [-2.0, 0.0, 0.0, -2.0], 2.0, 2.0, 2.0, None, 0.0);

    keypoint = kp_shape(2.0, 3.0, M_PI, 0.0);
    assert_eq!(keypoint.x, 1.0);
    assert_eq!(keypoint.y, 2.0);
    expect_shape(&keypoint, [-2.0, 0.0, 0.0, -3.0], 2.5, 2.0, 3.0, None, 0.0);

    // COLMAP's literal `2.12132025f`, kept as written (it rounds to the nearest float).
    #[allow(clippy::excessive_precision)]
    let a = f64::from(2.121_320_25_f32);
    keypoint = kp_shape(2.0, 3.0, -M_PI / 2.0, M_PI / 4.0);
    assert_eq!(keypoint.x, 1.0);
    assert_eq!(keypoint.y, 2.0);
    expect_shape(
        &keypoint,
        [0.0, a, -2.0, a],
        2.5,
        2.0,
        3.0,
        Some(-M_PI / 2.0),
        M_PI / 4.0,
    );

    keypoint = kp_shape(2.0, 3.0, M_PI / 2.0, M_PI / 4.0);
    assert_eq!(keypoint.x, 1.0);
    assert_eq!(keypoint.y, 2.0);
    expect_shape(
        &keypoint,
        [0.0, -a, 2.0, -a],
        2.5,
        2.0,
        3.0,
        Some(M_PI / 2.0),
        M_PI / 4.0,
    );

    keypoint.rescale_xy(2.0, 2.0).unwrap();
    assert_eq!(keypoint.x, 2.0);
    assert_eq!(keypoint.y, 4.0);
    expect_shape(
        &keypoint,
        [2.0 * 0.0, 2.0 * -a, 2.0 * 2.0, 2.0 * -a],
        2.0 * 2.5,
        2.0 * 2.0,
        2.0 * 3.0,
        Some(M_PI / 2.0),
        M_PI / 4.0,
    );

    keypoint.rescale_xy(1.0, 0.5).unwrap();
    assert_eq!(keypoint.x, 2.0);
    assert_eq!(keypoint.y, 2.0);
    expect_near(keypoint.a11, 0.0, 1e-6);
    expect_near(keypoint.a12, -a, 1e-6);
    expect_near(keypoint.a21, 4.0, 1e-6);
    expect_near(keypoint.a22, -a, 1e-6);
    expect_near(keypoint.compute_scale(), 3.5, 1e-6);
    expect_near(keypoint.compute_scale_x() - 2.0, 2.0, 1e-6);
    expect_near(keypoint.compute_scale_y(), 3.0, 1e-6);
    expect_near(keypoint.compute_orientation(), M_PI / 2.0, 1e-6);
    expect_near(keypoint.compute_shear(), M_PI / 4.0, 1e-6);

    assert_eq!(keypoint, keypoint);
    assert_ne!(keypoint, kp_scale(1.0, 2.0, 1.0, 0.0));
}

#[test]
fn feature_keypoint_rot90() {
    let kp = FeatureKeypoint::from_affine(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    let (w, h) = (10, 20);

    let mut kp1 = kp;
    kp1.rot90(1, w, h); // 90 CCW
    expect_float_eq(kp1.x, 2.0);
    expect_float_eq(kp1.y, 10.0 - 1.0);
    expect_float_eq(kp1.a11, 5.0);
    expect_float_eq(kp1.a12, 6.0);
    expect_float_eq(kp1.a21, -3.0);
    expect_float_eq(kp1.a22, -4.0);

    let mut kp2 = kp;
    kp2.rot90(2, w, h); // 180 CCW
    expect_float_eq(kp2.x, 10.0 - 1.0);
    expect_float_eq(kp2.y, 20.0 - 2.0);
    expect_float_eq(kp2.a11, -3.0);
    expect_float_eq(kp2.a12, -4.0);
    expect_float_eq(kp2.a21, -5.0);
    expect_float_eq(kp2.a22, -6.0);

    let mut kp3 = kp;
    kp3.rot90(3, w, h); // 270 CCW
    expect_float_eq(kp3.x, 20.0 - 2.0);
    expect_float_eq(kp3.y, 1.0);
    expect_float_eq(kp3.a11, -5.0);
    expect_float_eq(kp3.a12, -6.0);
    expect_float_eq(kp3.a21, 3.0);
    expect_float_eq(kp3.a22, 4.0);

    let mut kp_identity = kp;
    kp_identity.rot90(0, w, h);
    assert_eq!(kp_identity, kp);

    let mut kp_identity4 = kp;
    kp_identity4.rot90(4, w, h);
    assert_eq!(kp_identity4, kp);

    let mut kp_neg1 = kp;
    kp_neg1.rot90(-1, w, h); // same as 3
    assert_eq!(kp_neg1, kp3);
}

#[test]
fn feature_descriptors_nominal() {
    let descriptors = FeatureDescriptors::new(FeatureExtractorType::Sift, random_bytes(2, 3));
    assert_eq!(descriptors.extractor_type, FeatureExtractorType::Sift);
    assert_eq!(descriptors.data.rows(), 2);
    assert_eq!(descriptors.data.cols(), 3);
    let data = descriptors.data.as_slice();
    assert_eq!(descriptors.data[(0, 0)], data[0]);
    assert_eq!(descriptors.data[(0, 1)], data[1]);
    assert_eq!(descriptors.data[(0, 2)], data[2]);
    assert_eq!(descriptors.data[(1, 0)], data[3]);
    assert_eq!(descriptors.data[(1, 1)], data[4]);
    assert_eq!(descriptors.data[(1, 2)], data[5]);
}

#[test]
fn feature_descriptors_sift_conversion() {
    // SIFT uses value cast (uint8 <-> float)
    let original = FeatureDescriptors::new(FeatureExtractorType::Sift, random_bytes(10, 128));
    let as_float = original.to_float().unwrap();
    assert_eq!(as_float.extractor_type, FeatureExtractorType::Sift);
    assert_eq!(as_float.data.cols(), original.data.cols());
    let cast: Vec<f32> = original
        .data
        .as_slice()
        .iter()
        .map(|&v| f32::from(v))
        .collect();
    assert_eq!(
        as_float.data,
        RowMajorMatrix::from_vec(original.data.rows(), original.data.cols(), cast)
    );

    let recovered = as_float.to_bytes().unwrap();
    assert_eq!(recovered.extractor_type, FeatureExtractorType::Sift);
    assert_eq!(recovered.data, original.data);
}

#[test]
fn feature_descriptors_aliked_conversion() {
    // ALIKED uses reinterpret cast (float32 bytes <-> float)
    let original =
        FeatureDescriptors::new(FeatureExtractorType::AlikedN16Rot, random_bytes(10, 512));
    let as_float = original.to_float().unwrap();
    assert_eq!(as_float.extractor_type, FeatureExtractorType::AlikedN16Rot);
    assert_eq!(
        as_float.data.cols() * std::mem::size_of::<f32>(),
        original.data.cols()
    );

    let recovered = as_float.to_bytes().unwrap();
    assert_eq!(recovered.extractor_type, FeatureExtractorType::AlikedN16Rot);
    assert_eq!(recovered.data, original.data);
}

#[test]
fn feature_matches_nominal() {
    let feature_match = FeatureMatch::default();
    assert_eq!(feature_match.point2d_idx1, INVALID_POINT2D_IDX);
    assert_eq!(feature_match.point2d_idx2, INVALID_POINT2D_IDX);
    let matches: FeatureMatches = vec![FeatureMatch::default(); 1];
    assert_eq!(matches.len(), 1);
    assert_eq!(matches[0].point2d_idx1, INVALID_POINT2D_IDX);
    assert_eq!(matches[0].point2d_idx2, INVALID_POINT2D_IDX);

    assert_eq!(feature_match, feature_match);
    assert_ne!(feature_match, FeatureMatch::new(0, 1));
}

#[test]
fn keypoints_matrix_conversion_roundtrip() {
    let keypoints: FeatureKeypoints = vec![
        FeatureKeypoint::from_scale_orientation(1.0, 2.0, 3.0, 0.5).unwrap(),
        FeatureKeypoint::from_scale_orientation(4.0, 5.0, 1.0, -0.3).unwrap(),
        FeatureKeypoint::from_scale_orientation(10.0, 20.0, 0.5, 0.0).unwrap(),
    ];
    let recovered = keypoints_from_matrix(&keypoints_to_matrix(&keypoints)).unwrap();
    assert_eq!(recovered.len(), keypoints.len());
    for (recovered, keypoint) in recovered.iter().zip(&keypoints) {
        assert_eq!(recovered.x, keypoint.x);
        assert_eq!(recovered.y, keypoint.y);
        expect_near(
            recovered.compute_scale(),
            f64::from(keypoint.compute_scale()),
            1e-5,
        );
        expect_near(
            recovered.compute_orientation(),
            f64::from(keypoint.compute_orientation()),
            1e-5,
        );
    }
}

#[test]
fn matches_matrix_conversion_roundtrip() {
    let matches: FeatureMatches = vec![
        FeatureMatch::new(0, 5),
        FeatureMatch::new(3, 7),
        FeatureMatch::new(100, 200),
    ];
    assert_eq!(
        matches_from_matrix(&matches_to_matrix(&matches)).unwrap(),
        matches
    );
}
