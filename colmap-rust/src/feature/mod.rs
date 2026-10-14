//! COLMAP's `feature/` module. Tests: `colmap-rust/tests/feature.rs`.
//! - [`extractor`]: `extractor.h` / `extractor.cc`, the extraction options and the
//!   [`extractor::FeatureExtractor`] trait.
//! - [`index`]: `index.h` / `index.cc`, the exact k-nearest-neighbour descriptor index (the
//!   faiss replacement) and the per-image index cache seam.
//! - [`matcher`]: `matcher.h` / `matcher.cc`, the matching options and the
//!   [`matcher::FeatureMatcher`] trait.
//! - [`sift`]: `sift.h` / `sift.cc`, the SIFT options, the CPU SIFT extractor and the CPU SIFT
//!   matcher.
//! - [`types`]: `types.h` / `types.cc`, keypoints, descriptors, matches, the extractor and
//!   matcher type enums, and the keypoint / match matrix conversions.
//! - [`utils`]: `utils.h` / `utils.cc`, keypoints to points, descriptor normalization and
//!   quantization, and top-scale feature selection.
//! - [`vlfeat`]: VLFeat's SIFT filter (`thirdparty/VLFeat`), which COLMAP's CPU SIFT
//!   extractor drives.

pub mod extractor;
pub mod index;
pub mod matcher;
pub mod sift;
pub mod types;
pub mod utils;
pub mod vlfeat;
