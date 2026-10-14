//! Port of `colmap/feature/extractor.h` / `extractor.cc`: [`FeatureExtractionOptions`] and the
//! [`FeatureExtractor`] trait that the extraction controller drives. Port of colmap-sharp's
//! `Feature/FeatureExtractor.cs`. Neighbors: [`super::sift`] (`SiftExtractionOptions` and the
//! SIFT CPU extractor), [`super::types`] (keypoints and descriptors).
//! Tests: `tests/feature/extractor.rs` (`extractor_test.cc`).
//!
//! The trait shape, which later phases (the extraction controller, the app) build on:
//! - `Extract(bitmap, keypoints*, descriptors*)` becomes [`FeatureExtractor::extract`] with
//!   `&mut` outputs (`descriptors` optional, as COLMAP's may be null) and a [`CancelToken`],
//!   which COLMAP's interface does not have: the app must be able to stop a multi-second
//!   extraction in the middle of an image. A cancelled extraction returns `Err` with
//!   `ErrorKind::Cancelled` and leaves the outputs as they were; cancellation never changes
//!   the result of an extraction that runs to completion.
//! - It returns `Result<bool>`: `Ok(false)` where COLMAP returns false (extraction failed),
//!   `Err` where COLMAP throws (a failed `THROW_CHECK`) or the token was cancelled.
//! - `&mut self`: an extractor keeps per-image state (the SIFT filter is reused while the image
//!   size stays the same), so one instance serves one thread, as in COLMAP.
//!
//! Out of scope (docs/LICENSE_AUDIT.md): the GPU extractor (SiftGPU, excluded by licence) and
//! the ONNX extractors (ALIKED, LoMa). So the options have no `use_gpu` / `gpu_index` and no
//! `aliked` / `loma` sub-options, and `Check` fails for the ONNX extractor types. The per-type
//! answers that need no sub-options (`RequiresRGB`, `RequiresOpenGL`, `EffMaxImageSize`) are
//! kept for every type.
//!
//! Translation notes:
//! - `FeatureExtractionTypeOptions`' `shared_ptr` sub-options with their deep-copying copy
//!   constructor and assignment become an owned [`SiftExtractionOptions`] field: `Clone` is
//!   the deep copy, and a Rust move keeps the value.
//! - `ThrowUnknownFeatureExtractorType` (a `std::runtime_error`) is `Err` with
//!   `ErrorKind::RuntimeError` and COLMAP's message, for `UNDEFINED`.
//! - `Check` is COLMAP's `CHECK_OPTION_*`, which logs and returns false, so it returns
//!   `Ok(false)` on a violation; it is `Err` only where COLMAP throws (an unknown type in
//!   `EffMaxImageSize`).

use super::sift::SiftExtractionOptions;
use super::types::{FeatureDescriptors, FeatureExtractorType, FeatureKeypoints};
use crate::sensor::bitmap::Bitmap;
use crate::util::cancellation::CancelToken;
use crate::util::check::{ColmapError, ErrorKind, Result};

/// Port of `colmap::FeatureExtractionOptions` (the fields of the in-scope CPU extractors).
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureExtractionOptions {
    /// The extractor type.
    pub extractor_type: FeatureExtractorType,
    /// SIFT extraction options (`FeatureExtractionTypeOptions::sift`).
    pub sift: SiftExtractionOptions,
    /// Maximum image size, otherwise image will be down-scaled. If non-positive, the
    /// appropriate size is selected automatically based on the extractor type.
    pub max_image_size: i32,
    /// Number of threads for feature extraction (-1: all cores).
    pub num_threads: i32,
}

impl Default for FeatureExtractionOptions {
    fn default() -> Self {
        Self::new(FeatureExtractorType::Sift)
    }
}

impl FeatureExtractionOptions {
    /// `FeatureExtractionOptions(type)`: default options for the given extractor type.
    pub fn new(extractor_type: FeatureExtractorType) -> Self {
        Self {
            extractor_type,
            sift: SiftExtractionOptions::default(),
            max_image_size: -1,
            num_threads: -1,
        }
    }

    /// Port of `FeatureExtractionOptions::RequiresRGB`: whether the selected extractor needs
    /// RGB (not grey) images.
    pub fn requires_rgb(&self) -> Result<bool> {
        match self.extractor_type {
            FeatureExtractorType::Sift => Ok(false),
            FeatureExtractorType::AlikedN16Rot
            | FeatureExtractorType::AlikedN32
            | FeatureExtractorType::LomaB
            | FeatureExtractorType::LomaB128 => Ok(true),
            FeatureExtractorType::Undefined => Err(unknown_type(self.extractor_type)),
        }
    }

    /// Port of `FeatureExtractionOptions::RequiresOpenGL`. Only the excluded SiftGPU
    /// extractor (`use_gpu` without the covariant options) needs OpenGL, so every known type
    /// answers false.
    pub fn requires_opengl(&self) -> Result<bool> {
        match self.extractor_type {
            FeatureExtractorType::Sift
            | FeatureExtractorType::AlikedN16Rot
            | FeatureExtractorType::AlikedN32
            | FeatureExtractorType::LomaB
            | FeatureExtractorType::LomaB128 => Ok(false),
            FeatureExtractorType::Undefined => Err(unknown_type(self.extractor_type)),
        }
    }

    /// Port of `FeatureExtractionOptions::EffMaxImageSize`: `max_image_size` when positive,
    /// otherwise the extractor type's default (3200 for SIFT, 1600 for the learned
    /// extractors).
    pub fn eff_max_image_size(&self) -> Result<i32> {
        if self.max_image_size > 0 {
            return Ok(self.max_image_size);
        }
        match self.extractor_type {
            FeatureExtractorType::Sift => Ok(3200),
            FeatureExtractorType::AlikedN16Rot
            | FeatureExtractorType::AlikedN32
            | FeatureExtractorType::LomaB
            | FeatureExtractorType::LomaB128 => Ok(1600),
            FeatureExtractorType::Undefined => Err(unknown_type(self.extractor_type)),
        }
    }

    /// Port of `FeatureExtractionOptions::Check`: `Ok(false)` on a violated option. The ONNX
    /// extractor types are not available in this port, so they fail the check.
    pub fn check(&self) -> Result<bool> {
        if self.eff_max_image_size()? <= 0 {
            return Ok(false);
        }
        Ok(match self.extractor_type {
            FeatureExtractorType::Sift => self.sift.check(),
            _ => false,
        })
    }
}

/// Port of `colmap::FeatureExtractor`: detects keypoints and computes their descriptors.
pub trait FeatureExtractor {
    /// Port of `FeatureExtractor::Extract`: the features of `bitmap` into `keypoints`
    /// (replaced) and, when given, `descriptors` (replaced). `Ok(false)` when extraction
    /// failed; `Err` on a failed check or when `cancel` was cancelled mid-extraction, with
    /// the outputs then left as they were.
    fn extract(
        &mut self,
        bitmap: &Bitmap,
        keypoints: &mut FeatureKeypoints,
        descriptors: Option<&mut FeatureDescriptors>,
        cancel: &CancelToken,
    ) -> Result<bool>;
}

/// `ThrowUnknownFeatureExtractorType`: COLMAP's `std::runtime_error`.
pub(crate) fn unknown_type(extractor_type: FeatureExtractorType) -> ColmapError {
    ColmapError::new(
        ErrorKind::RuntimeError,
        format!("Unknown feature extractor type: {extractor_type}"),
    )
}
