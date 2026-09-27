//! colmap-rust: a pure-Rust port of COLMAP 4.2.0 (Structure-from-Motion and Multi-View
//! Stereo). Pure Rust (std plus the `libm` crate), no GUI and no GPU, and it builds for
//! `wasm32-unknown-unknown`.
//!
//! Modules mirror COLMAP's `src/colmap/` tree. A module appears here only once it has real,
//! ported content (see `PORTING_PLAN.md` for the order the rest arrive in):
//!
//! - [`util`]: COLMAP's `util/` — the `THROW_CHECK` family ([`check!`] and friends, returning
//!   [`util::check::ColmapError`]) and C++ stream formatting of doubles.
//! - [`math`]: COLMAP's `math/` — so far [`math::fns`], the single choke point for
//!   transcendental functions.

pub mod math;
pub mod util;

pub use util::check::{ColmapError, ErrorKind, Result};
