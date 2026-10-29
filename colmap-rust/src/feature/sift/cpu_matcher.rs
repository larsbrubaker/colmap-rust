//! The CPU matching half of `colmap/feature/sift.h` / `sift.cc`: `SiftCPUFeatureMatcher`,
//! `CreateSiftFeatureMatcher`, `ThrowCheckFeatureTypesMatch` and
//! `FindBestMatches{OneWay}{BruteForce,Index}`. Port of colmap-sharp's
//! `Feature/SiftMatcher.cs`. Neighbors: `super::match_kernels` (the fused integer distance
//! scan), [`super::guided_matching`] (the guided filters), [`crate::feature::index`] (the k-NN
//! index of the default, non brute-force path), [`crate::feature::matcher`] (the trait),
//! [`super::SiftMatchingOptions`]. Tests: `tests/feature/sift_matching.rs` and
//! `tests/feature/sift_guided_matching.rs` (`sift_test.cc`'s CPU matcher cases).
//!
//! Tier A (exact) given the same nearest neighbours: every distance is an integer (see
//! `super::match_kernels`) and the ratio / distance tests are COLMAP's `f32` expressions
//! (`acosf` / `sqrtf` on the same floats, through [`crate::math::fns`]). The guided
//! filters evaluate COLMAP's float (F, H) and double (E) expressions in Eigen's left-to-right
//! order without FMA; Apple clang may contract them, which can only flip a pair lying exactly
//! on the `max_error` boundary.
//!
//! Two differences in how, not what:
//! - COLMAP's brute-force and guided paths fill an N x M float matrix, then scan rows and the
//!   transposed matrix. Here one fused integer scan yields both directions (identical results,
//!   `super::match_kernels` header).
//! - The default path searches the [`crate::feature::index`] index, which is exact here (faiss
//!   is approximate for 512+ descriptors; the `feature/index.rs` header).
//!
//! The GPU matcher (SiftGPU) and SIFT LightGlue (ONNX) are excluded.

use std::sync::Arc;

use super::guided_matching::create_guided_filter;
use super::match_kernels::{
    scan_distances, DistanceType, Top2, L2_SENTINEL, SIFT_DIM, SQ_SIFT_DESCRIPTOR_NORM,
};
use crate::feature::index::FeatureDescriptorIndex;
use crate::feature::matcher::{FeatureMatcher, FeatureMatcherImage, FeatureMatchingOptions};
use crate::feature::types::{
    FeatureExtractorType, FeatureMatch, FeatureMatcherType, FeatureMatches,
};
use crate::linalg::RowMajorMatrix;
use crate::math::fns;
use crate::scene::two_view_geometry::TwoViewGeometry;
use crate::util::cancellation::CancelToken;
use crate::util::check::{ColmapError, ErrorKind, Result};
use crate::util::types::{ImageId, INVALID_IMAGE_ID};
use crate::{check, check_eq, check_notnull, fatal_throw};

/// Port of `CreateSiftFeatureMatcher`: the SIFT CPU matcher for `SIFT_BRUTEFORCE`. SIFT
/// LightGlue (ONNX) and the GPU matcher are excluded from this port, so `SIFT_LIGHTGLUE` is an
/// error, as is any non-SIFT type (COLMAP's `LOG(FATAL_THROW)`).
pub fn create_sift_feature_matcher(
    options: &FeatureMatchingOptions,
) -> Result<Box<dyn FeatureMatcher>> {
    match options.matcher_type {
        FeatureMatcherType::SiftBruteforce => Ok(Box::new(SiftCpuFeatureMatcher::new(options)?)),
        FeatureMatcherType::SiftLightglue => Err(ColmapError::new(
            ErrorKind::RuntimeError,
            "SIFT LightGlue matching needs ONNX models, which colmap-rust does not include.",
        )),
        other => fatal_throw!("Unknown SIFT feature matcher type: {}", other),
    }
}

/// Port of colmap's `SiftCPUFeatureMatcher`.
pub struct SiftCpuFeatureMatcher {
    options: FeatureMatchingOptions,
    prev_image_id1: ImageId,
    prev_image_id2: ImageId,
    index1: Option<Arc<dyn FeatureDescriptorIndex>>,
    index2: Option<Arc<dyn FeatureDescriptorIndex>>,
}

impl SiftCpuFeatureMatcher {
    /// A matcher with a copy of `options`, which must pass `Check` (COLMAP's `THROW_CHECK`).
    pub fn new(options: &FeatureMatchingOptions) -> Result<Self> {
        check!(options.check());
        Ok(Self {
            options: options.clone(),
            prev_image_id1: INVALID_IMAGE_ID,
            prev_image_id2: INVALID_IMAGE_ID,
            index1: None,
            index2: None,
        })
    }

    /// The descriptor indexes of the two images, fetched from the cache when an image changed
    /// (only on the index path).
    fn fetch_indexes(
        &mut self,
        image1: &FeatureMatcherImage<'_>,
        image2: &FeatureMatcherImage<'_>,
    ) -> Result<()> {
        let sift = &self.options.sift;
        if !sift.cpu_brute_force_matcher
            && (self.prev_image_id1 == INVALID_IMAGE_ID || self.prev_image_id1 != image1.image_id)
        {
            let cache = check_notnull!(sift.cpu_descriptor_index_cache.as_ref());
            self.index1 = Some(cache.get(image1.image_id)?);
            self.prev_image_id1 = image1.image_id;
        }
        if !sift.cpu_brute_force_matcher
            && (self.prev_image_id2 == INVALID_IMAGE_ID || self.prev_image_id2 != image2.image_id)
        {
            let cache = check_notnull!(sift.cpu_descriptor_index_cache.as_ref());
            self.index2 = Some(cache.get(image2.image_id)?);
            self.prev_image_id2 = image2.image_id;
        }
        Ok(())
    }
}

impl FeatureMatcher for SiftCpuFeatureMatcher {
    fn match_features(
        &mut self,
        image1: &FeatureMatcherImage<'_>,
        image2: &FeatureMatcherImage<'_>,
        matches: &mut FeatureMatches,
        cancel: &CancelToken,
    ) -> Result<()> {
        throw_check_feature_types_match(image1, image2, false)?;
        matches.clear();
        cancel.check_cancelled()?;
        self.fetch_indexes(image1, image2)?;

        let descriptors1 = check_notnull!(image1.descriptors);
        let descriptors2 = check_notnull!(image2.descriptors);
        if descriptors1.data.rows() == 0 || descriptors2.data.rows() == 0 {
            return Ok(());
        }

        let sift = &self.options.sift;
        let max_ratio = sift.max_ratio as f32;
        let max_distance = sift.max_distance as f32;
        if sift.cpu_brute_force_matcher {
            let mut rows = vec![Top2::init(0); descriptors1.data.rows()];
            let mut cols = vec![Top2::init(0); descriptors2.data.rows()];
            scan_distances(
                &descriptors1.data,
                &descriptors2.data,
                DistanceType::DotProduct,
                None,
                &mut rows,
                sift.cross_check.then_some(cols.as_mut_slice()),
                cancel,
            )?;
            let matches_2to1 = sift
                .cross_check
                .then(|| one_way_brute_force(&cols, max_ratio, max_distance));
            collect_matches(
                &one_way_brute_force(&rows, max_ratio, max_distance),
                matches_2to1.as_deref(),
                matches,
            );
            return Ok(());
        }

        let mut indices_1to2 = RowMajorMatrix::<i32>::new(0, 0);
        let mut l2_dists_1to2 = RowMajorMatrix::<f32>::new(0, 0);
        let mut indices_2to1 = RowMajorMatrix::<i32>::new(0, 0);
        let mut l2_dists_2to1 = RowMajorMatrix::<f32>::new(0, 0);

        check_notnull!(self.index2.as_ref()).search(
            2,
            &descriptors1.to_float()?,
            &mut indices_1to2,
            &mut l2_dists_1to2,
            cancel,
        )?;
        if sift.cross_check {
            check_notnull!(self.index1.as_ref()).search(
                2,
                &descriptors2.to_float()?,
                &mut indices_2to1,
                &mut l2_dists_2to1,
                cancel,
            )?;
        }

        find_best_matches_index(
            &indices_1to2,
            &l2_dists_1to2,
            &indices_2to1,
            &l2_dists_2to1,
            max_ratio,
            max_distance,
            sift.cross_check,
            matches,
        );
        Ok(())
    }

    fn match_guided(
        &mut self,
        max_error: f64,
        image1: &FeatureMatcherImage<'_>,
        image2: &FeatureMatcherImage<'_>,
        two_view_geometry: &mut TwoViewGeometry,
        cancel: &CancelToken,
    ) -> Result<()> {
        throw_check_feature_types_match(image1, image2, true)?;
        two_view_geometry.inlier_matches.clear();
        cancel.check_cancelled()?;
        self.fetch_indexes(image1, image2)?;

        let camera1 = check_notnull!(image1.camera);
        let camera2 = check_notnull!(image2.camera);
        let keypoints1 = check_notnull!(image1.keypoints);
        let keypoints2 = check_notnull!(image2.keypoints);
        let descriptors1 = check_notnull!(image1.descriptors);
        let descriptors2 = check_notnull!(image2.descriptors);

        let mut rows = vec![Top2::init(L2_SENTINEL); descriptors1.data.rows()];
        let mut cols = vec![Top2::init(L2_SENTINEL); descriptors2.data.rows()];
        {
            let guided_filter = match create_guided_filter(
                max_error,
                camera1,
                keypoints1,
                camera2,
                keypoints2,
                two_view_geometry,
            )? {
                Some(filter) => filter,
                None => return Ok(()),
            };

            // The guided filter indexes per-feature geometry (bearings with Jacobians, or the
            // keypoints themselves) by descriptor row, so the two must align.
            check_eq!(keypoints1.len(), descriptors1.data.rows());
            check_eq!(keypoints2.len(), descriptors2.data.rows());

            // COLMAP scans the full guided L2 matrix with FindBestMatchesIndex over every
            // column (indices LinSpaced), in both directions.
            scan_distances(
                &descriptors1.data,
                &descriptors2.data,
                DistanceType::L2,
                Some(&*guided_filter),
                &mut rows,
                Some(&mut cols),
                cancel,
            )?;
        }

        let sift = &self.options.sift;
        let max_ratio = sift.max_ratio as f32;
        let max_distance = sift.max_distance as f32;
        // FindBestMatchesIndex cross-checks only when the reverse neighbour lists have rows.
        let cross_check = sift.cross_check && descriptors2.data.rows() != 0;
        let matches_2to1 = cross_check.then(|| one_way_index(&cols, max_ratio, max_distance));
        collect_matches(
            &one_way_index(&rows, max_ratio, max_distance),
            matches_2to1.as_deref(),
            &mut two_view_geometry.inlier_matches,
        );
        Ok(())
    }
}

/// Port of `ThrowCheckFeatureTypesMatch`: both images carry 128-wide SIFT descriptors, and,
/// with `check_keypoints`, a camera and one keypoint per descriptor row.
fn throw_check_feature_types_match(
    image1: &FeatureMatcherImage<'_>,
    image2: &FeatureMatcherImage<'_>,
    check_keypoints: bool,
) -> Result<()> {
    let descriptors1 = check_notnull!(image1.descriptors);
    let descriptors2 = check_notnull!(image2.descriptors);
    check_eq!(descriptors1.extractor_type, FeatureExtractorType::Sift);
    check_eq!(descriptors2.extractor_type, FeatureExtractorType::Sift);
    check_eq!(descriptors1.data.cols(), SIFT_DIM);
    check_eq!(descriptors2.data.cols(), SIFT_DIM);
    if check_keypoints {
        check_notnull!(image1.camera);
        check_notnull!(image2.camera);
        let keypoints1 = check_notnull!(image1.keypoints);
        let keypoints2 = check_notnull!(image2.keypoints);
        check_eq!(descriptors1.data.rows(), keypoints1.len());
        check_eq!(descriptors2.data.rows(), keypoints2.len());
    }
    Ok(())
}

/// Port of `FindBestMatchesIndex`: the ratio / distance tests on each query's neighbours
/// (`indices_2to1` is only used for cross-checking when it has rows).
#[allow(clippy::too_many_arguments)] // COLMAP's signature.
fn find_best_matches_index(
    indices_1to2: &RowMajorMatrix<i32>,
    l2_dists_1to2: &RowMajorMatrix<f32>,
    indices_2to1: &RowMajorMatrix<i32>,
    l2_dists_2to1: &RowMajorMatrix<f32>,
    max_ratio: f32,
    max_distance: f32,
    cross_check: bool,
    matches: &mut FeatureMatches,
) {
    let matches_1to2 =
        find_best_matches_one_way_index(indices_1to2, l2_dists_1to2, max_ratio, max_distance);
    let matches_2to1 = (cross_check && indices_2to1.rows() != 0).then(|| {
        find_best_matches_one_way_index(indices_2to1, l2_dists_2to1, max_ratio, max_distance)
    });
    collect_matches(&matches_1to2, matches_2to1.as_deref(), matches);
}

/// Port of `FindBestMatchesOneWayIndex` over explicit neighbour lists: the best match of each
/// row, or -1.
fn find_best_matches_one_way_index(
    indices: &RowMajorMatrix<i32>,
    l2_dists: &RowMajorMatrix<f32>,
    max_ratio: f32,
    max_distance: f32,
) -> Vec<i32> {
    (0..indices.rows())
        .map(|d1_idx| {
            let mut best_d2_idx = -1;
            let mut best_l2_dist = f32::MAX;
            let mut second_best_l2_dist = f32::MAX;
            for n_idx in 0..indices.cols() {
                let d2_idx = indices[(d1_idx, n_idx)];
                let l2_dist = l2_dists[(d1_idx, n_idx)];
                if l2_dist < best_l2_dist {
                    best_d2_idx = d2_idx;
                    second_best_l2_dist = best_l2_dist;
                    best_l2_dist = l2_dist;
                } else if l2_dist < second_best_l2_dist {
                    second_best_l2_dist = l2_dist;
                }
            }
            if passes_index_tests(
                best_d2_idx,
                best_l2_dist,
                second_best_l2_dist,
                max_ratio,
                max_distance,
            ) {
                best_d2_idx
            } else {
                -1
            }
        })
        .collect()
}

/// `FindBestMatchesOneWayIndex` on the fused scan's states (keys are negated L2 distances).
fn one_way_index(states: &[Top2], max_ratio: f32, max_distance: f32) -> Vec<i32> {
    states
        .iter()
        .map(|s| {
            // The keys are integers below 2^24 in magnitude, so the casts are exact.
            let best = if s.best_idx == -1 {
                f32::MAX
            } else {
                -s.best as f32
            };
            let second = if s.second == L2_SENTINEL {
                f32::MAX
            } else {
                -s.second as f32
            };
            if passes_index_tests(s.best_idx, best, second, max_ratio, max_distance) {
                s.best_idx
            } else {
                -1
            }
        })
        .collect()
}

/// The distance and ratio tests of `FindBestMatchesOneWayIndex`.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // COLMAP's `>=` rejection, negated as written.
fn passes_index_tests(
    best_d2_idx: i32,
    best_l2_dist: f32,
    second_best_l2_dist: f32,
    max_ratio: f32,
    max_distance: f32,
) -> bool {
    // Check if any match found.
    if best_d2_idx == -1 {
        return false;
    }

    // Check if match distance passes threshold (kSqSiftDescriptorNorm * max_distance *
    // max_distance, evaluated in float).
    let max_l2_dist = SQ_SIFT_DESCRIPTOR_NORM as f32 * max_distance * max_distance;
    if best_l2_dist > max_l2_dist {
        return false;
    }

    // Check if match passes ratio test. Keep this comparison >= in order to ensure that the
    // case of best == second_best is detected.
    !(fns::sqrtf(best_l2_dist) >= max_ratio * fns::sqrtf(second_best_l2_dist))
}

/// Port of `FindBestMatchesOneWayBruteForce`'s tests on the fused scan's dot-product states:
/// the best match of each row, or -1.
fn one_way_brute_force(states: &[Top2], max_ratio: f32, max_distance: f32) -> Vec<i32> {
    let inv_sq_descriptor_norm = (1.0 / f64::from(SQ_SIFT_DESCRIPTOR_NORM)) as f32;
    states
        .iter()
        .map(|s| {
            // Check if any match found.
            if s.best_idx == -1 {
                return -1;
            }

            // Convert to L2 distance in which the thresholds are defined. The dot products are
            // integers below 2^24, so the casts are exact.
            let best_dist_normed = fns::acosf((inv_sq_descriptor_norm * s.best as f32).min(1.0));

            // Check if match distance passes threshold.
            if best_dist_normed > max_distance {
                return -1;
            }

            let second_best_dist_normed =
                fns::acosf((inv_sq_descriptor_norm * s.second as f32).min(1.0));

            // Check if match passes ratio test. Keep this comparison >= in order to ensure
            // that the case of best == second_best is detected.
            if best_dist_normed >= max_ratio * second_best_dist_normed {
                return -1;
            }

            s.best_idx
        })
        .collect()
}

/// The cross-check (or one-way) assembly shared by `FindBestMatchesBruteForce` and
/// `FindBestMatchesIndex`; matches are in ascending `point2D_idx1` order.
fn collect_matches(
    matches_1to2: &[i32],
    matches_2to1: Option<&[i32]>,
    matches: &mut FeatureMatches,
) {
    matches.clear();
    for (i1, &i2) in matches_1to2.iter().enumerate() {
        if i2 == -1 {
            continue;
        }
        // Both are non-negative descriptor indices here.
        if matches_2to1.is_none_or(|m| m[i2 as usize] == i1 as i32) {
            matches.push(FeatureMatch::new(i1 as u32, i2 as u32));
        }
    }
}
