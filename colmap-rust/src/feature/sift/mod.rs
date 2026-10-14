//! Port of `colmap/feature/sift.h` / `sift.cc`, COLMAP's SIFT on the CPU. Port of
//! colmap-sharp's `Feature/Sift.cs` (extraction half).
//! - This file: [`SiftExtractionOptions`] (with [`SiftNormalization`]) and
//!   `kSiftDescriptorDim`.
//! - [`cpu_extractor`]: `SiftCPUFeatureExtractor` (VLFeat's SIFT, [`super::vlfeat`], driven
//!   octave by octave) and `TransformVLFeatToUBCFeatureDescriptors`.
//!
//! SiftGPU is excluded by licence (docs/LICENSE_AUDIT.md): there is no GPU extractor or
//! matcher and no `use_gpu` branch.
//! The covariant extractor (`CovariantSiftCPUFeatureExtractor`, VLFeat's covdet),
//! `CreateSiftFeatureExtractor` and the SIFT matcher are not ported yet (PORTING_PLAN.md
//! Phase 9).
//! Tests: `tests/feature/sift.rs` (`sift_test.cc`) and `tests/feature/rust_only_sift.rs`
//! (pycolmap fixtures, reuse, cancellation).

pub mod cpu_extractor;

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
