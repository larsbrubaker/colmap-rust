//! Port of COLMAP's `src/colmap/math/` (the parts that need no linear algebra):
//!
//! - [`fns`]: the single choke point for transcendental functions.
//! - [`utils`]: `math.h`/`math.cc`'s scalar helpers, re-exported here (`colmap::Median`
//!   is `math::median`, and so on).
//! - [`random`]: `random.h`/`random.cc`, COLMAP's thread-local mt19937 PRNG with libc++'s
//!   distributions.
//! - [`random_eigen`]: `random_eigen.h`, fixed-size random vectors, matrices and unit
//!   quaternions drawn from [`random`].
//! - [`union_find`], [`connected_components`], [`spanning_tree`]: the graph utilities.
//! - [`graph_cut`] (Stoer-Wagner min cut, normalized k-way cut), [`graph_cut_min_st`]
//!   (`MinSTGraphCut`, Boykov-Kolmogorov max-flow) and [`graph_cut_partitioner`] (the
//!   multilevel partitioner that replaces METIS): `graph_cut.h`/`graph_cut.cc`.
//!
//! `polynomial` and `matrix.h` need the dynamic-size linear algebra and arrive with it.

pub mod connected_components;
pub mod fns;
pub mod graph_cut;
pub mod graph_cut_min_st;
pub mod graph_cut_partitioner;
pub mod random;
pub mod random_eigen;
pub mod spanning_tree;
pub mod union_find;
pub mod utils;

pub use utils::*;
