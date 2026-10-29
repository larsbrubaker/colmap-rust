//! Port of `colmap/feature/utils.h` / `utils.cc`: keypoints to points, the L2 and L1-root
//! descriptor normalizations, float to uint8 descriptor quantization and top-scale feature
//! selection. Built on [`super::types`]; the SIFT extractor normalizes and quantizes every
//! descriptor through here. Port of colmap-sharp's `Feature/FeatureUtils.cs`.
//! Tests: `tests/feature/utils.rs` (`utils_test.cc` 1:1) and
//! `tests/feature/rust_only_feature_utils.rs`.
//!
//! Tier A (exact), because SIFT's quantized descriptors depend on it. Eigen's row norms are
//! vectorized reductions: COLMAP's build sums a contiguous float row with two 4-lane packet
//! accumulators (lanes `i mod 8 < 4` and `>= 4`), adds them, folds the packet as
//! `(l0 + l2) + (l1 + l3)` (NEON's and SSE's `predux<Packet4f>`), then adds the tail left over
//! after whole packets. `eigen_row_sum` repeats that order in scalar code, so the norms, and
//! the quantized descriptors, match bit for bit.
//!
//! Translation notes:
//! - `rowwise().normalize()` divides each row by its norm unconditionally (unlike
//!   `Matrix::normalize()`), so a zero row becomes NaN, as in COLMAP.
//! - `ExtractTopScaleFeatures` uses `std::partial_sort`, which leaves equal scales in an
//!   unspecified order; here ties keep their input order ([`extract_top_scale_features`]).
//! - Its `THROW_CHECK`s return `Err`.

use super::types::{FeatureDescriptors, FeatureDescriptorsData, FeatureDescriptorsFloatData};
use super::types::{FeatureKeypoint, FeatureKeypoints};
use crate::linalg::{RowMajorMatrix, Vector2d};
use crate::math::fns;
use crate::math::utils::truncate_cast;
use crate::util::check::Result;
use crate::{check_eq, check_gt};

/// Port of `colmap::FeatureKeypointsToPointsVector`: the keypoint locations as doubles.
pub fn feature_keypoints_to_points_vector(keypoints: &[FeatureKeypoint]) -> Vec<Vector2d> {
    keypoints
        .iter()
        .map(|k| Vector2d::new(f64::from(k.x), f64::from(k.y)))
        .collect()
}

/// Port of `colmap::L2NormalizeFeatureDescriptors`: each row divided by its L2 norm.
pub fn l2_normalize_feature_descriptors(descriptors: &mut FeatureDescriptorsFloatData) {
    for r in 0..descriptors.rows() {
        l2_normalize_row(descriptors.row_mut(r));
    }
}

/// Port of `colmap::L1RootNormalizeFeatureDescriptors`: each row scaled by the reciprocal of
/// its L1 norm, then square-rooted element-wise. See "Three things everyone should know to
/// improve object retrieval", Relja Arandjelovic and Andrew Zisserman, CVPR 2012.
pub fn l1_root_normalize_feature_descriptors(descriptors: &mut FeatureDescriptorsFloatData) {
    for r in 0..descriptors.rows() {
        l1_root_normalize_row(descriptors.row_mut(r));
    }
}

/// Port of `colmap::FeatureDescriptorsToUnsignedByte`: linear scaling from [0, 0.5] to
/// [0, 255], rounded, then truncated to 0..255. The factor 512 assumes L2-normalized
/// descriptors, whose largest element is in practice below 0.5 (utils.h).
pub fn feature_descriptors_to_unsigned_byte(
    descriptors: &FeatureDescriptorsFloatData,
) -> FeatureDescriptorsData {
    let values = descriptors
        .as_slice()
        .iter()
        .map(|&v| quantize_descriptor_value(v))
        .collect();
    RowMajorMatrix::from_vec(descriptors.rows(), descriptors.cols(), values)
}

/// Port of `colmap::ExtractTopScaleFeatures`: keeps the `num_features` keypoints of largest
/// scale (and their descriptor rows), largest first. A no-op when there are no more than
/// that many. Equal scales keep their input order.
///
/// Deliberate divergence, the tie order: COLMAP selects the largest scales with
/// `std::partial_sort`, which leaves keypoints of equal scale in an unspecified,
/// implementation-defined order (libc++'s heap-based selection). This sorts by scale, largest
/// first, with the input index as the tie-break (`f32::total_cmp`, so a NaN scale also has a
/// fixed place, where COLMAP's `>` comparator is not a strict weak order). Equal scales are
/// common: the SIFT extractor emits one keypoint per orientation, and all of them share the
/// scale. Reproducing libc++'s heap order would tie the result to one standard library; the
/// index tie-break is deterministic and stable. Only the order among equal scales differs, and
/// *which* keypoints are kept differs only when a tie straddles the cut, where COLMAP's choice
/// is arbitrary too. `extract_top_scale_features_nominal` (`tests/feature/utils.rs`, 1:1,
/// distinct scales) passes unchanged; `rust_only_extract_top_scale_features_ties_keep_input_order`
/// (`tests/feature/rust_only_feature_utils.rs`) pins the ties. Same as colmap-sharp entry 40.
pub fn extract_top_scale_features(
    keypoints: &mut FeatureKeypoints,
    descriptors: &mut FeatureDescriptors,
    num_features: usize,
) -> Result<()> {
    check_eq!(keypoints.len(), descriptors.data.rows());
    check_gt!(num_features, 0);

    if descriptors.data.rows() <= num_features {
        return Ok(());
    }

    let mut scales: Vec<(usize, f32)> = keypoints
        .iter()
        .enumerate()
        .map(|(i, k)| (i, k.compute_scale()))
        .collect();
    // Largest scale first, ties by input index (the doc above). `total_cmp` orders NaN scales
    // deterministically; COLMAP's `>` comparator is not a strict weak order with NaN.
    scales.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));

    let cols = descriptors.data.cols();
    let mut top_keypoints = Vec::with_capacity(num_features);
    let mut top_data = RowMajorMatrix::new(num_features, cols);
    for (i, &(index, _)) in scales.iter().take(num_features).enumerate() {
        top_keypoints.push(keypoints[index]);
        top_data
            .row_mut(i)
            .copy_from_slice(descriptors.data.row(index));
    }

    *keypoints = top_keypoints;
    descriptors.data = top_data;
    Ok(())
}

/// `row.normalize()` inside `rowwise().normalize()`: divide by the norm unconditionally.
pub(crate) fn l2_normalize_row(row: &mut [f32]) {
    let norm = fns::sqrtf(eigen_row_sum(row, true));
    for v in row.iter_mut() {
        *v /= norm;
    }
}

/// `row *= 1 / row.lpNorm<1>(); row = row.array().sqrt();`.
pub(crate) fn l1_root_normalize_row(row: &mut [f32]) {
    let scale = 1.0f32 / eigen_row_sum(row, false);
    for v in row.iter_mut() {
        *v = fns::sqrtf(*v * scale);
    }
}

/// One element of [`feature_descriptors_to_unsigned_byte`]: `TruncateCast<float,
/// uint8_t>(std::round(512.0f * value))`. `f32::round` rounds halfway cases away from zero,
/// like `std::round`.
pub(crate) fn quantize_descriptor_value(value: f32) -> u8 {
    let scaled_value = (512.0f32 * value).round();
    truncate_cast::<f32, u8>(scaled_value)
}

/// One term of [`eigen_row_sum`]: `x^2` for the squared norm, `|x|` for the L1 norm.
fn term(row: &[f32], squared: bool, i: usize) -> f32 {
    if squared {
        row[i] * row[i]
    } else {
        row[i].abs()
    }
}

/// Sum of `x^2` (or `|x|`) over a contiguous float row in the order of Eigen's vectorized
/// reduction with 4-lane packets (see the module header).
fn eigen_row_sum(row: &[f32], squared: bool) -> f32 {
    const PACKET_SIZE: usize = 4;
    let size = row.len();
    let aligned_size2 = size / (2 * PACKET_SIZE) * (2 * PACKET_SIZE);
    let aligned_size = size / PACKET_SIZE * PACKET_SIZE;

    if aligned_size == 0 {
        // No whole packet: a plain left-to-right reduction (Eigen asserts on an empty row;
        // an empty sum is 0 here).
        if size == 0 {
            return 0.0;
        }
        let mut plain = term(row, squared, 0);
        for i in 1..size {
            plain += term(row, squared, i);
        }
        return plain;
    }

    let mut p0 = [0.0f32; PACKET_SIZE];
    for (l, p) in p0.iter_mut().enumerate() {
        *p = term(row, squared, l);
    }

    if aligned_size > PACKET_SIZE {
        let mut p1 = [0.0f32; PACKET_SIZE];
        for (l, p) in p1.iter_mut().enumerate() {
            *p = term(row, squared, PACKET_SIZE + l);
        }

        let mut index = 2 * PACKET_SIZE;
        while index < aligned_size2 {
            for (l, (a, b)) in p0.iter_mut().zip(p1.iter_mut()).enumerate() {
                *a += term(row, squared, index + l);
                *b += term(row, squared, index + PACKET_SIZE + l);
            }
            index += 2 * PACKET_SIZE;
        }

        for (a, b) in p0.iter_mut().zip(p1.iter()) {
            *a += *b;
        }

        if aligned_size > aligned_size2 {
            for (l, p) in p0.iter_mut().enumerate() {
                *p += term(row, squared, aligned_size2 + l);
            }
        }
    }

    // predux<Packet4f>: (l0 + l2) + (l1 + l3).
    let mut res = (p0[0] + p0[2]) + (p0[1] + p0[3]);
    for i in aligned_size..size {
        res += term(row, squared, i);
    }
    res
}
