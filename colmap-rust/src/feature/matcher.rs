//! Port of `colmap/feature/matcher.h` / `matcher.cc`: [`FeatureMatchingOptions`] and the
//! [`FeatureMatcher`] trait (`Match`, `MatchGuided`) that the matching controllers drive. Port
//! of colmap-sharp's `Feature/FeatureMatcher.cs`. Neighbors: [`super::sift::cpu_matcher`]
//! (`SiftCPUFeatureMatcher`, the only matcher in scope), [`super::sift::SiftMatchingOptions`],
//! [`crate::scene::two_view_geometry`] (`MatchGuided`'s input and output).
//! Tests: `tests/feature/matcher.rs` (`matcher_test.cc`).
//!
//! The trait shape, which later phases (the matching controllers, the app) build on:
//! - `Match(image1, image2, matches*)` becomes [`FeatureMatcher::match_features`] (`match` is
//!   a Rust keyword) and `MatchGuided(max_error, image1, image2, two_view_geometry*)` becomes
//!   [`FeatureMatcher::match_guided`], both with `&mut` outputs that are cleared first, as in
//!   COLMAP.
//! - Both take a [`CancelToken`], which COLMAP's interface does not have: the app must be able
//!   to stop a long match (an 8k x 8k pair) in the middle. A cancelled call returns `Err` with
//!   `ErrorKind::Cancelled` and leaves the output matches empty; cancellation never changes
//!   the result of a call that runs to completion.
//! - Both return `Result<()>`: `Err` where COLMAP throws (a failed `THROW_CHECK`, e.g. a
//!   descriptor type mismatch) or the token was cancelled.
//! - `&mut self`: a matcher keeps per-pair state (the SIFT CPU matcher keeps the last two
//!   images' descriptor indexes), so one instance serves one thread, as in COLMAP.
//! - [`FeatureMatcherImage`] borrows the image's camera, keypoints and descriptors where
//!   COLMAP holds raw and shared pointers; `None` is COLMAP's null pointer.
//!
//! Out of scope (docs/LICENSE_AUDIT.md): the GPU matcher (SiftGPU, excluded by licence) and
//! the ONNX matchers (LightGlue, ALIKED, LoMa). So the options have no `use_gpu` / `gpu_index`
//! and no `aliked` / `loma` sub-options, `Check` fails for the ONNX matcher types, and
//! [`create_feature_matcher`] returns `Err` for them. `RequiresOpenGL` is kept for every type.
//!
//! Translation notes:
//! - `FeatureMatchingTypeOptions`' `shared_ptr` sub-options with their deep-copying copy
//!   constructor and assignment become an owned [`SiftMatchingOptions`] field: `Clone` is the
//!   deep copy (the descriptor index cache it points to stays shared, as in COLMAP).
//! - `ThrowUnknownFeatureMatcherType` (a `std::runtime_error`) is `Err` with
//!   `ErrorKind::RuntimeError` and COLMAP's message, for `UNDEFINED`.
//! - `Check` is COLMAP's `CHECK_OPTION_*`, which logs and returns false, so it returns false
//!   on a violation.

use super::sift::cpu_matcher::create_sift_feature_matcher;
use super::sift::SiftMatchingOptions;
use super::types::{FeatureDescriptors, FeatureKeypoint, FeatureMatcherType, FeatureMatches};
use crate::geometry::pose_prior::PosePrior;
use crate::scene::camera::Camera;
use crate::scene::two_view_geometry::TwoViewGeometry;
use crate::util::cancellation::CancelToken;
use crate::util::check::{ColmapError, ErrorKind, Result};
use crate::util::types::{ImageId, INVALID_IMAGE_ID};

/// Port of `colmap::FeatureMatchingOptions` (the fields of the in-scope CPU matcher).
#[derive(Debug, Clone)]
pub struct FeatureMatchingOptions {
    /// The matcher type.
    pub matcher_type: FeatureMatcherType,
    /// SIFT matching options (`FeatureMatchingTypeOptions::sift`).
    pub sift: SiftMatchingOptions,
    /// Number of threads for feature matching and geometric verification (-1: all cores).
    pub num_threads: i32,
    /// Maximum number of matches.
    pub max_num_matches: i32,
    /// Whether to perform guided matching.
    pub guided_matching: bool,
    /// Skips the geometric verification stage and forwards matches unchanged. This option is
    /// ignored when guided matching is enabled, because guided matching depends on the
    /// two-view geometry produced by geometric verification.
    pub skip_geometric_verification: bool,
    /// Whether to perform geometric verification using rig constraints between pairs of
    /// non-trivial frames. If disabled, performs geometric two-view verification for
    /// non-trivial frames without rig constraints. This option is ignored when
    /// `skip_geometric_verification` is true.
    pub rig_verification: bool,
    /// Whether to skip matching images within the same frame. This is useful for the case of
    /// non-overlapping cameras in a rig.
    pub skip_image_pairs_in_same_frame: bool,
}

impl Default for FeatureMatchingOptions {
    fn default() -> Self {
        Self::new(FeatureMatcherType::SiftBruteforce)
    }
}

impl FeatureMatchingOptions {
    /// `FeatureMatchingOptions(type)`: default options for the given matcher type.
    pub fn new(matcher_type: FeatureMatcherType) -> Self {
        Self {
            matcher_type,
            sift: SiftMatchingOptions::default(),
            num_threads: -1,
            max_num_matches: 32768,
            guided_matching: false,
            skip_geometric_verification: false,
            rig_verification: false,
            skip_image_pairs_in_same_frame: false,
        }
    }

    /// Port of `FeatureMatchingOptions::RequiresOpenGL`. Only the excluded SiftGPU matcher
    /// (`use_gpu`) needs OpenGL, so every known type answers false.
    pub fn requires_opengl(&self) -> Result<bool> {
        match self.matcher_type {
            FeatureMatcherType::Undefined => Err(unknown_type(self.matcher_type)),
            _ => Ok(false),
        }
    }

    /// Port of `FeatureMatchingOptions::Check`: false on a violated option. The ONNX matcher
    /// types (SIFT LightGlue, ALIKED, LoMa) are not available in this port, so they fail the
    /// check, as does `UNDEFINED` (COLMAP's "Unknown feature matcher type").
    pub fn check(&self) -> bool {
        if self.max_num_matches < 0 {
            return false;
        }
        match self.matcher_type {
            FeatureMatcherType::SiftBruteforce => self.sift.check(),
            _ => false,
        }
    }
}

/// Port of `colmap::FeatureMatcher::Image`: one image's features as seen by a matcher.
#[derive(Debug, Clone, Copy)]
pub struct FeatureMatcherImage<'a> {
    /// Unique identifier for the image. Allows a matcher to cache some computations per image
    /// in consecutive calls to matching.
    pub image_id: ImageId,
    /// The image's camera (required by guided matching).
    pub camera: Option<&'a Camera>,
    /// The keypoints (required by guided matching).
    pub keypoints: Option<&'a [FeatureKeypoint]>,
    /// The descriptors, one row per keypoint.
    pub descriptors: Option<&'a FeatureDescriptors>,
    /// The image's pose prior, if any.
    pub pose_prior: Option<&'a PosePrior>,
}

impl Default for FeatureMatcherImage<'_> {
    /// COLMAP's member initializers: an invalid image id and null pointers.
    fn default() -> Self {
        Self {
            image_id: INVALID_IMAGE_ID,
            camera: None,
            keypoints: None,
            descriptors: None,
            pose_prior: None,
        }
    }
}

/// Port of `colmap::FeatureMatcher`.
pub trait FeatureMatcher {
    /// `Match`: matches the features of two images into `matches` (cleared first).
    fn match_features(
        &mut self,
        image1: &FeatureMatcherImage<'_>,
        image2: &FeatureMatcherImage<'_>,
        matches: &mut FeatureMatches,
        cancel: &CancelToken,
    ) -> Result<()>;

    /// `MatchGuided`: matches the features of two images restricted to pairs consistent with
    /// the two-view geometry within `max_error` pixels, into its `inlier_matches` (cleared
    /// first).
    fn match_guided(
        &mut self,
        max_error: f64,
        image1: &FeatureMatcherImage<'_>,
        image2: &FeatureMatcherImage<'_>,
        two_view_geometry: &mut TwoViewGeometry,
        cancel: &CancelToken,
    ) -> Result<()>;
}

/// Port of `FeatureMatcher::Create`: the SIFT CPU matcher for `SIFT_BRUTEFORCE`. `Err` for the
/// ONNX matchers (LightGlue, ALIKED, LoMa), which this port excludes, and for `UNDEFINED`.
pub fn create_feature_matcher(options: &FeatureMatchingOptions) -> Result<Box<dyn FeatureMatcher>> {
    match options.matcher_type {
        FeatureMatcherType::SiftBruteforce | FeatureMatcherType::SiftLightglue => {
            create_sift_feature_matcher(options)
        }
        FeatureMatcherType::Undefined => Err(unknown_type(options.matcher_type)),
        other => Err(ColmapError::new(
            ErrorKind::RuntimeError,
            format!(
                "Feature matcher {other} needs ONNX models, which colmap-rust does not include."
            ),
        )),
    }
}

/// `ThrowUnknownFeatureMatcherType`: COLMAP's `std::runtime_error`.
fn unknown_type(matcher_type: FeatureMatcherType) -> ColmapError {
    ColmapError::new(
        ErrorKind::RuntimeError,
        format!("Unknown feature matcher type: {matcher_type}"),
    )
}
