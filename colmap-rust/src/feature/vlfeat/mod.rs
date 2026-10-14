//! The parts of VLFeat (`thirdparty/VLFeat`, BSD-2-Clause, as vendored by COLMAP) that COLMAP's
//! CPU SIFT runs, ported from colmap-sharp's `Feature/VLFeat/` (MIT, itself ported from
//! VLFeat; notice in `THIRD_PARTY_NOTICES.md`). VLFeat is C, so it is ported, not wrapped.
//! - [`mathop`]: the `mathop.h` approximations SIFT uses and `sift.c`'s `fast_expn` table.
//! - [`imopv`]: `vl_imconvcol_vf`, the separable column convolution of the Gaussian smoothing.
//! - [`sift_filter`]: `VlSiftFilt` and its Gaussian scale space (`vl_sift_new`,
//!   `vl_sift_process_first_octave` / `_next_octave`); [`sift_detect`] holds `vl_sift_detect`
//!   and [`sift_descriptor`] the gradients, orientations and descriptors.
//!
//! Tier A (exact): every keypoint field, orientation and float descriptor value matches
//! VLFeat compiled without FMA contraction, bit for bit
//! (`tests/feature/rust_only_vl_sift_filter.rs`, fixture `sift_vlfeat.json`).

pub mod imopv;
pub mod mathop;
pub mod sift_descriptor;
pub mod sift_detect;
pub mod sift_filter;
