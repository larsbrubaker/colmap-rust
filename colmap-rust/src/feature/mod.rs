//! COLMAP's `feature/` module. Tests: `colmap-rust/tests/feature.rs`.
//! - [`types`]: `types.h` / `types.cc`, keypoints, descriptors, matches, the extractor and
//!   matcher type enums, and the keypoint / match matrix conversions.
//! - [`utils`]: `utils.h` / `utils.cc`, keypoints to points, descriptor normalization and
//!   quantization, and top-scale feature selection.
//! - [`vlfeat`]: VLFeat's SIFT filter (`thirdparty/VLFeat`), which COLMAP's CPU SIFT
//!   extractor drives.

pub mod types;
pub mod utils;
pub mod vlfeat;
