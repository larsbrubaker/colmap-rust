//! Port of `colmap/feature/sift.h` / `sift.cc`, COLMAP's SIFT on the CPU. Port of
//! colmap-sharp's `Feature/Sift.cs` and `Feature/SiftMatcher.cs`.
//! - This file: [`SiftExtractionOptions`] (with [`SiftNormalization`]),
//!   [`SiftMatchingOptions`] and `kSiftDescriptorDim`.
//! - [`cpu_extractor`]: `SiftCPUFeatureExtractor` (VLFeat's SIFT, [`super::vlfeat`], driven
//!   octave by octave) and `TransformVLFeatToUBCFeatureDescriptors`.
//! - [`cpu_matcher`]: `SiftCPUFeatureMatcher` and `CreateSiftFeatureMatcher`, over
//!   `match_kernels` (the fused integer distance scan) and [`guided_matching`] (the guided
//!   filters of `MatchGuided`).
//! - [`features_text`] ports `LoadSiftFeaturesFromTextFile`.
//!
//! SiftGPU is excluded by licence (docs/LICENSE_AUDIT.md): there is no GPU extractor or
//! matcher and no `use_gpu` branch. SIFT LightGlue (ONNX) is excluded too, so
//! [`SiftMatchingOptions`] has no `lightglue` options.
//! The covariant extractor (`CovariantSiftCPUFeatureExtractor`, VLFeat's covdet) and
//! `CreateSiftFeatureExtractor` are not ported yet (PORTING_PLAN.md Phase 9).
//! Tests: `tests/feature/sift.rs` (`sift_test.cc`'s extraction cases),
//! `tests/feature/sift_matching.rs` and `tests/feature/sift_guided_matching.rs` (its CPU
//! matcher cases), and `tests/feature/rust_only_sift.rs` (pycolmap fixtures, reuse,
//! cancellation).

use std::fmt;
use std::sync::Arc;

use super::index::FeatureDescriptorIndexCache;

pub mod cpu_extractor;
pub mod cpu_matcher;
pub mod features_text;
pub mod guided_matching;
pub(crate) mod match_kernels;

#[cfg(test)]
mod tests;

/// `kSiftDescriptorDim`: the length of a SIFT descriptor.
pub const SIFT_DESCRIPTOR_DIM: usize = 128;

/// Port of `SiftExtractionOptions::Normalization` (`MAKE_ENUM_CLASS(Normalization, 0,
/// L1_ROOT, L2)`), with COLMAP's numeric values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum SiftNormalization {
    /// `L1_ROOT`: L1-normalizes each descriptor followed by element-wise square rooting. This
    /// normalization is usually better than standard L2-normalization. See "Three things
    /// everyone should know to improve object retrieval", Relja Arandjelovic and Andrew
    /// Zisserman, CVPR 2012.
    #[default]
    L1Root = 0,
    /// `L2`: each vector is L2-normalized.
    L2 = 1,
}

/// Port of `colmap::SiftExtractionOptions`.
#[derive(Debug, Clone, PartialEq)]
pub struct SiftExtractionOptions {
    /// Maximum number of features to detect, keeping larger-scale features.
    pub max_num_features: i32,
    /// First octave in the pyramid, i.e. -1 upsamples the image by one level.
    pub first_octave: i32,
    /// Number of octaves.
    pub num_octaves: i32,
    /// Number of levels per octave.
    pub octave_resolution: i32,
    /// Peak threshold for detection.
    pub peak_threshold: f64,
    /// Edge threshold for detection.
    pub edge_threshold: f64,
    /// Estimate affine shape of SIFT features in the form of oriented ellipses as opposed to
    /// original SIFT which estimates oriented disks (covariant extractor only).
    pub estimate_affine_shape: bool,
    /// Maximum number of orientations per keypoint if not `estimate_affine_shape`.
    pub max_num_orientations: i32,
    /// Fix the orientation to 0 for upright features.
    pub upright: bool,
    /// Whether to adapt the feature detection depending on the image darkness. Only available
    /// in the OpenGL SiftGPU version, which is excluded, so it has no effect.
    pub darkness_adaptivity: bool,
    /// Domain-size pooling computes an average SIFT descriptor across multiple scales around
    /// the detected scale. This was proposed in "Domain-Size Pooling in Local Descriptors and
    /// Network Architectures", J. Dong and S. Soatto, CVPR 2015. This has been shown to
    /// outperform other SIFT variants and learned descriptors in "Comparative Evaluation of
    /// Hand-Crafted and Learned Local Features", Schönberger, Hardmeier, Sattler, Pollefeys,
    /// CVPR 2016. Covariant extractor only.
    pub domain_size_pooling: bool,
    /// Smallest domain-size pooling scale.
    pub dsp_min_scale: f64,
    /// Largest domain-size pooling scale.
    pub dsp_max_scale: f64,
    /// Number of domain-size pooling scales.
    pub dsp_num_scales: i32,
    /// Whether to force usage of the covariant VLFeat implementation. Otherwise, the
    /// covariant implementation is only used when `estimate_affine_shape` or
    /// `domain_size_pooling` are enabled, since the normal SIFT implementation is faster.
    pub force_covariant_extractor: bool,
    /// Descriptor normalization.
    pub normalization: SiftNormalization,
}

impl Default for SiftExtractionOptions {
    fn default() -> Self {
        Self {
            max_num_features: 8192,
            first_octave: -1,
            num_octaves: 4,
            octave_resolution: 3,
            // `0.02 / octave_resolution` in COLMAP's member initializer (an int 3).
            peak_threshold: 0.02 / 3.0,
            edge_threshold: 10.0,
            estimate_affine_shape: false,
            max_num_orientations: 2,
            upright: false,
            darkness_adaptivity: false,
            domain_size_pooling: false,
            dsp_min_scale: 1.0 / 6.0,
            dsp_max_scale: 3.0,
            dsp_num_scales: 10,
            force_covariant_extractor: false,
            normalization: SiftNormalization::L1Root,
        }
    }
}

impl SiftExtractionOptions {
    /// Port of `SiftExtractionOptions::Check` (`CHECK_OPTION_*`: false on a violation; NaN
    /// thresholds fail, as the C++ comparisons do).
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(a > b)`: NaN fails, as in C++.
    pub fn check(&self) -> bool {
        if self.max_num_features <= 0
            || self.octave_resolution <= 0
            || !(self.peak_threshold > 0.0)
            || !(self.edge_threshold > 0.0)
            || self.max_num_orientations <= 0
        {
            return false;
        }
        if self.domain_size_pooling
            && (!(self.dsp_min_scale > 0.0)
                || !(self.dsp_max_scale >= self.dsp_min_scale)
                || self.dsp_num_scales <= 0)
        {
            return false;
        }
        true
    }
}

/// Port of `colmap::SiftMatchingOptions` (minus the ONNX LightGlue options).
#[derive(Clone)]
pub struct SiftMatchingOptions {
    /// Maximum distance ratio between first and second best match.
    pub max_ratio: f64,
    /// Maximum distance to best match.
    pub max_distance: f64,
    /// Whether to enable cross checking in matching.
    pub cross_check: bool,
    /// Whether to use brute-force instead of index based CPU matching.
    pub cpu_brute_force_matcher: bool,
    /// Cache for reusing descriptor index for feature matching. Required unless
    /// `cpu_brute_force_matcher`. Shared, not copied, by `Clone` (COLMAP's raw pointer).
    pub cpu_descriptor_index_cache: Option<Arc<dyn FeatureDescriptorIndexCache>>,
}

impl Default for SiftMatchingOptions {
    fn default() -> Self {
        Self {
            max_ratio: 0.8,
            max_distance: 0.7,
            cross_check: true,
            cpu_brute_force_matcher: false,
            cpu_descriptor_index_cache: None,
        }
    }
}

impl fmt::Debug for SiftMatchingOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SiftMatchingOptions")
            .field("max_ratio", &self.max_ratio)
            .field("max_distance", &self.max_distance)
            .field("cross_check", &self.cross_check)
            .field("cpu_brute_force_matcher", &self.cpu_brute_force_matcher)
            .field(
                "cpu_descriptor_index_cache",
                &self.cpu_descriptor_index_cache.as_ref().map(|_| "<cache>"),
            )
            .finish()
    }
}

impl SiftMatchingOptions {
    /// Port of `SiftMatchingOptions::Check` (`CHECK_OPTION_*`: false on a violation; NaN
    /// fails, as the C++ comparisons do).
    #[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(a > b)`: NaN fails, as in C++.
    pub fn check(&self) -> bool {
        self.max_ratio > 0.0 && self.max_distance > 0.0
    }
}
