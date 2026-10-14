//! The parts of VLFeat (`thirdparty/VLFeat`, BSD-2-Clause, as vendored by COLMAP) that COLMAP's
//! CPU SIFT runs, ported from colmap-sharp's `Feature/VLFeat/` (MIT, itself ported from
//! VLFeat; notice in `THIRD_PARTY_NOTICES.md`). VLFeat is C, so it is ported, not wrapped.
//! - [`mathop`]: the `mathop.h` approximations SIFT uses and `sift.c`'s `fast_expn` table.
//! - [`imopv`]: `vl_imconvcol_vf`, the separable column convolution of the Gaussian smoothing.
//! - [`imopv_smooth`]: `vl_imsmooth_f` and the image gradients, for the covariant detector.
//! - [`linear`]: `mathop.c`'s 2x2 SVD and small Gaussian elimination, for the covariant detector.
//! - [`scalespace`]: `VlScaleSpace`, the Gaussian scale space of the covariant detector.
//! - [`covdet`]: `VlCovDet`, the covariant (DoG) detector: detection and non-extrema
//!   suppression; [`covdet_shape`] holds its patch extraction, affine shape adaptation and
//!   orientation assignment.
//! - [`sift_filter`]: `VlSiftFilt` and its Gaussian scale space (`vl_sift_new`,
//!   `vl_sift_process_first_octave` / `_next_octave`); [`sift_detect`] holds `vl_sift_detect`
//!   and [`sift_descriptor`] the gradients, orientations and descriptors;
//!   [`sift_raw_descriptor`] is `vl_sift_calc_raw_descriptor`, the covariant extractor's.
//!
//! Tier A (exact): every keypoint field, orientation and float descriptor value matches
//! VLFeat compiled without FMA contraction, bit for bit
//! (`tests/feature/rust_only_vl_sift_filter.rs`, fixture `sift_vlfeat.json`). So does every
//! covariant-detector feature (frame, octave, level, scores) and the raw descriptor of each of
//! its patches (`tests/feature/rust_only_vl_covdet.rs`, fixture `covariant_sift_vlfeat.json`);
//! its pieces ([`imopv_smooth`], [`linear`], [`scalespace`], [`sift_raw_descriptor`]) also have
//! `rust_only_*` tests of VLFeat's behaviour.

pub mod covdet;
pub mod covdet_shape;
pub mod imopv;
pub mod imopv_smooth;
pub mod linear;
pub mod mathop;
pub mod scalespace;
pub mod sift_descriptor;
pub mod sift_detect;
pub mod sift_filter;
pub mod sift_raw_descriptor;
