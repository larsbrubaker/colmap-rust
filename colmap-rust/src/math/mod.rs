//! Port of COLMAP's `src/colmap/math/` (the parts that need no linear algebra):
//!
//! - [`fns`]: the single choke point for transcendental functions.
//! - [`utils`]: `math.h`/`math.cc`'s scalar helpers, re-exported here (`colmap::Median`
//!   is `math::median`, and so on).
//! - [`random`]: `random.h`/`random.cc`, COLMAP's thread-local mt19937 PRNG with libc++'s
//!   distributions.
//! - [`union_find`], [`connected_components`], [`spanning_tree`]: the graph utilities.
//!
//! `graph_cut` is not ported yet. `polynomial`, `matrix.h` and `random_eigen.h` need `linalg`
//! and arrive with it.

pub mod connected_components;
pub mod fns;
pub mod random;
pub mod spanning_tree;
pub mod union_find;
pub mod utils;

pub use utils::*;
