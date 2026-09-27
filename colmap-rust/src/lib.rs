//! colmap-rust: a pure-Rust port of COLMAP 4.2.0 (Structure-from-Motion and Multi-View
//! Stereo). Pure Rust (std plus the `libm` crate), no GUI and no GPU, and it builds for
//! `wasm32-unknown-unknown`.
//!
//! Modules mirror COLMAP's `src/colmap/` tree. A module appears here only once it has real,
//! ported content (see `PORTING_PLAN.md` for the order the rest arrive in):
//!
//! - [`util`]: COLMAP's `util/` — the `THROW_CHECK` family ([`check!`] and friends, returning
//!   [`util::check::ColmapError`]), C++ stream formatting of doubles, the id types, string,
//!   CSV, endian and path helpers, the timer, and cancellation/progress.
//! - [`math`]: COLMAP's `math/` — [`math::fns`] (the single choke point
//!   for transcendental functions), `math.h`'s scalar helpers, the mt19937 PRNG with libc++'s
//!   distributions ([`math::random`], [`math::random_eigen`]), union find, connected
//!   components, spanning trees, graph cuts, polynomials and the RQ decomposition
//!   ([`math::matrix`]).
//! - [`linalg`]: the Eigen replacement's fixed-size types (vectors, matrices up to 6x6,
//!   [`linalg::Quaterniond`], [`linalg::AngleAxisd`], [`linalg::AlignedBox3d`]), dynamic-size
//!   matrices and the dense decompositions, a port of colmap-sharp's `LinearAlgebra/`.
//! - [`geometry`]: COLMAP's `geometry/` — [`geometry::Rigid3d`], [`geometry::Sim3d`], pose
//!   helpers, pose priors, GPS conversions, normalization, boxes, essential and homography
//!   matrices, and triangulation.
//! - [`sensor`]: COLMAP's `sensor/` — camera models and their Jacobians, rigs, the bitmap pixel
//!   buffer with EXIF, and the camera sensor-width database.
//! - [`optim`]: COLMAP's `optim/` robust-estimation framework — the [`optim::Estimator`] and
//!   [`optim::Sampler`] traits, the random, PROSAC and combination samplers, support
//!   measurement, [`optim::Ransac`], [`optim::LoRansac`] and SPRT.

pub mod geometry;
pub mod linalg;
pub mod math;
pub mod optim;
pub mod sensor;
pub mod util;

pub use util::check::{ColmapError, ErrorKind, Result};
