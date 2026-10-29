//! Port of `SiftCPUFeatureExtractor` and `TransformVLFeatToUBCFeatureDescriptors` from
//! `colmap/feature/sift.cc`: VLFeat's SIFT ([`crate::feature::vlfeat`]) on a grey bitmap,
//! driven octave by octave, its descriptors normalized and quantized through
//! [`crate::feature::utils`]. Port of colmap-sharp's `SiftCpuFeatureExtractor`
//! (`Feature/Sift.cs`). Options live in [`super`]; the trait is
//! [`crate::feature::extractor::FeatureExtractor`].
//!
//! Tier A (exact) against unfused VLFeat: the VLFeat layer is bit-exact
//! (`tests/feature/rust_only_vl_sift_filter.rs`), and the normalization sums in Eigen's
//! reduction order. Against the macOS arm64 pycolmap wheel, which fuses multiply-adds, the
//! same keypoints come out with last-bit drift (`tests/feature/rust_only_sift.rs`).
//!
//! That drift is a deliberate divergence. COLMAP's `SiftCPUFeatureExtractor` runs VLFeat, which
//! the macOS arm64 pycolmap wheel compiles with Apple clang's default `-ffp-contract=on`:
//! multiply-adds in the Gaussian convolution, the DoG refinement and the descriptor binning are
//! fused into FMAs that round once. This port never fuses (CLAUDE.md "No FMA"), so against that
//! wheel it gives the same keypoints, in the same order and count, with last-bit drift: keypoint
//! values differ by up to about 1e-3 and descriptor bytes by at most one grey level, on at most
//! 0.1% of the bytes. Fusing depends on the compiler and target, so it cannot be reproduced on
//! every platform (wasm has no guaranteed FMA); unfused VLFeat is the portable reference, and
//! against it the port is bit-exact (`tests/feature/rust_only_vl_sift_filter.rs`, fixture
//! `sift_vlfeat.json`, VLFeat compiled with `-ffp-contract=off`).
//! `rust_only_sift_matches_pycolmap_*` (`tests/feature/rust_only_sift.rs`) checks the nine
//! `sift.json` cases against the wheel within 2e-3 per keypoint value, 1 per descriptor byte and
//! 0.1% mismatched bytes. Same as colmap-sharp entry 41.
//!
//! Translation notes:
//! - Cancellation (not in COLMAP): the filter checks the token between scale levels, and
//!   `extract` checks it before building the filter and between DoG levels of descriptor
//!   computation. A cancelled run drops the filter (it is left mid-octave), so the next image
//!   is extracted exactly as a fresh extractor would. The checks never touch the numbers.
//! - COLMAP's per-level containers are sized `max_num_orientations * num_keypoints` and
//!   shrunk afterwards; here they are pushed to, which gives the same rows.
//! - Each descriptor is normalized and quantized as a 1x128 row through the same row
//!   functions `L2NormalizeFeatureDescriptors` / `L1RootNormalizeFeatureDescriptors` and
//!   `FeatureDescriptorsToUnsignedByte` use.
//! - COLMAP reuses one 1x128 float buffer per octave and normalizes it in place. When VLFeat
//!   rejects a keypoint (out of bounds) it returns without writing, so that keypoint gets the
//!   previous descriptor, already normalized, normalized again. The buffer here is reused the
//!   same way, but across octaves too: COLMAP's buffer is a fresh, uninitialized Eigen matrix
//!   at each octave, so a rejected first keypoint of an octave reads indeterminate values
//!   there and the previous octave's last descriptor here.
//! - `vl_sift_new` cannot fail here (no allocation failure path), so `extract` never returns
//!   `Ok(false)`.
//! - COLMAP logs a warning for `darkness_adaptivity`; the core has no logger, and the option
//!   has no effect either way.

use super::{SiftExtractionOptions, SiftNormalization, SIFT_DESCRIPTOR_DIM};
use crate::check;
use crate::feature::extractor::{FeatureExtractionOptions, FeatureExtractor};
use crate::feature::types::{
    FeatureDescriptors, FeatureDescriptorsData, FeatureExtractorType, FeatureKeypoint,
    FeatureKeypoints,
};
use crate::feature::utils::{l1_root_normalize_row, l2_normalize_row, quantize_descriptor_value};
use crate::feature::vlfeat::sift_descriptor::VL_SIFT_DESCRIPTOR_LENGTH;
use crate::feature::vlfeat::sift_filter::VlSiftFilter;
use crate::linalg::RowMajorMatrix;
use crate::sensor::bitmap::Bitmap;
use crate::util::cancellation::CancelToken;
use crate::util::check::Result;

/// Port of COLMAP's `SiftCPUFeatureExtractor`: VLFeat SIFT on a grey bitmap. One instance
/// extracts from many images on one thread; it keeps its VLFeat filter while the image size
/// stays the same.
#[derive(Debug)]
pub struct SiftCpuFeatureExtractor {
    options: SiftExtractionOptions,
    sift: Option<VlSiftFilter>,
}

impl SiftCpuFeatureExtractor {
    /// The extractor for `options`, which must pass `Check` and not ask for the covariant
    /// extractor (`estimate_affine_shape`, `domain_size_pooling`,
    /// `force_covariant_extractor`).
    pub fn new(options: &FeatureExtractionOptions) -> Result<Self> {
        check!(options.check()?);
        let sift = &options.sift;
        check!(!sift.estimate_affine_shape);
        check!(!sift.domain_size_pooling);
        check!(!sift.force_covariant_extractor);
        Ok(Self {
            options: sift.clone(),
            sift: None,
        })
    }

    fn extract_with_filter(
        &mut self,
        bitmap: &Bitmap,
        keypoints: &mut FeatureKeypoints,
        descriptors: Option<&mut FeatureDescriptors>,
        cancel: &CancelToken,
    ) -> Result<bool> {
        let options = &self.options;
        let Some(sift) = self.sift.as_mut() else {
            return Ok(false);
        };
        sift.peak_thresh = options.peak_threshold;
        sift.edge_thresh = options.edge_threshold;
        let with_descriptors = descriptors.is_some();

        // Iterate through octaves.
        let mut level_num_features: Vec<usize> = Vec::new();
        let mut level_keypoints: Vec<FeatureKeypoints> = Vec::new();
        let mut level_descriptors: Vec<Vec<u8>> = Vec::new();
        let mut desc = [0.0f32; VL_SIFT_DESCRIPTOR_LENGTH];
        let mut first_octave = true;
        loop {
            if first_octave {
                let data_float: Vec<f32> = bitmap
                    .row_major_data()
                    .iter()
                    .map(|&v| f32::from(v) / 255.0f32)
                    .collect();
                if !sift.process_first_octave(&data_float, cancel)? {
                    break;
                }
                first_octave = false;
            } else if !sift.process_next_octave(cancel)? {
                break;
            }

            // Detect keypoints.
            sift.detect(cancel)?;

            // Extract detected keypoints.
            let vl_keypoints = sift.keypoints().to_vec();
            if vl_keypoints.is_empty() {
                continue;
            }

            // Extract features with different orientations per DOG level.
            let mut prev_level = -1;
            for vl_keypoint in &vl_keypoints {
                if vl_keypoint.is != prev_level {
                    // A DoG level's orientations and descriptors are one unit of work.
                    cancel.check_cancelled()?;

                    // Add containers for new DOG level.
                    level_num_features.push(0);
                    level_keypoints.push(Vec::new());
                    level_descriptors.push(Vec::new());
                }

                if let Some(n) = level_num_features.last_mut() {
                    *n += 1;
                }
                prev_level = vl_keypoint.is;

                // Extract feature orientations.
                let mut angles = [0.0f64; 4];
                let num_orientations = if options.upright {
                    1
                } else {
                    sift.calc_keypoint_orientations(&mut angles, vl_keypoint)
                };

                // Note that this is different from SiftGPU, which selects the top global
                // maxima as orientations while this selects the first two local maxima. It
                // is not clear which procedure is better.
                // max_num_orientations > 0 (checked), so the cast is exact.
                let num_used_orientations =
                    num_orientations.min(options.max_num_orientations as usize);

                for &angle in angles.iter().take(num_used_orientations) {
                    // FeatureKeypoint(float x, float y, float scale, float orientation): the
                    // double angle narrows to float at the call.
                    let keypoint = FeatureKeypoint::from_scale_orientation(
                        vl_keypoint.x + 0.5f32,
                        vl_keypoint.y + 0.5f32,
                        vl_keypoint.sigma,
                        angle as f32,
                    )?;
                    if let Some(level) = level_keypoints.last_mut() {
                        level.push(keypoint);
                    }
                    if with_descriptors {
                        sift.calc_keypoint_descriptor(&mut desc, vl_keypoint, angle);
                        match options.normalization {
                            SiftNormalization::L2 => l2_normalize_row(&mut desc),
                            SiftNormalization::L1Root => l1_root_normalize_row(&mut desc),
                        }
                        if let Some(level) = level_descriptors.last_mut() {
                            level.extend(desc.iter().map(|&v| quantize_descriptor_value(v)));
                        }
                    }
                }
            }
        }

        // Determine how many DOG levels to keep to satisfy max_num_features option.
        let mut first_level_to_keep = 0;
        let mut num_features = 0usize;
        let mut num_features_with_orientations = 0usize;
        // max_num_features > 0 (checked), so the cast is exact.
        let max_num_features = options.max_num_features as usize;
        for i in (0..level_keypoints.len()).rev() {
            num_features += level_num_features[i];
            num_features_with_orientations += level_keypoints[i].len();
            if num_features > max_num_features {
                first_level_to_keep = i;
                break;
            }
        }

        // Extract the features to be kept.
        keypoints.clear();
        keypoints.reserve(num_features_with_orientations);
        for level in &level_keypoints[first_level_to_keep..] {
            keypoints.extend_from_slice(level);
        }

        // Compute the descriptors for the detected keypoints.
        if let Some(descriptors) = descriptors {
            let mut data = Vec::with_capacity(num_features_with_orientations * SIFT_DESCRIPTOR_DIM);
            for level in &level_descriptors[first_level_to_keep..] {
                data.extend_from_slice(level);
            }
            let data =
                RowMajorMatrix::from_vec(num_features_with_orientations, SIFT_DESCRIPTOR_DIM, data);
            descriptors.data = transform_vlfeat_to_ubc_feature_descriptors(&data);
            descriptors.extractor_type = FeatureExtractorType::Sift;
        }

        Ok(true)
    }
}

impl FeatureExtractor for SiftCpuFeatureExtractor {
    /// Port of `SiftCPUFeatureExtractor::Extract`: keypoints (and, when `descriptors` is
    /// given, their descriptors) of a grey bitmap. Keypoints come out octave by octave, DoG
    /// level by level; when there are more than `max_num_features`, whole lowest (finest) DoG
    /// levels are dropped.
    fn extract(
        &mut self,
        bitmap: &Bitmap,
        keypoints: &mut FeatureKeypoints,
        descriptors: Option<&mut FeatureDescriptors>,
        cancel: &CancelToken,
    ) -> Result<bool> {
        check!(bitmap.is_grey());

        // Checked before the filter is built: its scale space can take gigabytes.
        cancel.check_cancelled()?;

        let reuse = self
            .sift
            .as_ref()
            .is_some_and(|s| s.width() == bitmap.width() && s.height() == bitmap.height());
        if !reuse {
            self.sift = Some(VlSiftFilter::new(
                bitmap.width(),
                bitmap.height(),
                self.options.num_octaves,
                self.options.octave_resolution,
                self.options.first_octave,
            ));
        }

        let result = self.extract_with_filter(bitmap, keypoints, descriptors, cancel);
        if result.is_err() {
            // A failed or cancelled run leaves the filter mid-octave; dropping it means the
            // next image is extracted exactly as a fresh extractor would.
            self.sift = None;
        }
        result
    }
}

/// Port of `TransformVLFeatToUBCFeatureDescriptors`: VLFeat stores the 8 orientation bins of
/// each spatial cell in the opposite rotational order from the original SIFT (UBC) format
/// that SiftGPU also uses; this reorders them.
pub fn transform_vlfeat_to_ubc_feature_descriptors(
    vlfeat_descriptors: &FeatureDescriptorsData,
) -> FeatureDescriptorsData {
    let rows = vlfeat_descriptors.rows();
    let cols = vlfeat_descriptors.cols();
    let mut ubc_descriptors = FeatureDescriptorsData::new(rows, cols);
    const Q: [usize; 8] = [0, 7, 6, 5, 4, 3, 2, 1];
    for n in 0..rows {
        for i in 0..4 {
            for j in 0..4 {
                for (k, &q) in Q.iter().enumerate() {
                    ubc_descriptors[(n, 8 * (j + 4 * i) + q)] =
                        vlfeat_descriptors[(n, 8 * (j + 4 * i) + k)];
                }
            }
        }
    }
    ubc_descriptors
}
