//! Port of `colmap/feature/index.h` / `index.cc`: [`FeatureDescriptorIndex`], the
//! k-nearest-neighbour descriptor index the SIFT CPU matcher searches when
//! `cpu_brute_force_matcher` is off (the default). Port of colmap-sharp's
//! `Feature/FeatureDescriptorIndex.cs`. Neighbors: `super::sift::match_kernels` (the integer
//! dot product both share), [`super::sift::cpu_matcher`] (the caller, through a
//! [`FeatureDescriptorIndexCache`]). Tests: `tests/feature/index.rs` (`index_test.cc`).
//!
//! COLMAP's only implementation wraps faiss (native, so excluded by CLAUDE.md contract 1).
//! Here the index is an exact k-nearest-neighbour search written from the definition: it
//! returns exactly what COLMAP's `IndexFlatL2` returns for fewer than 512 descriptors, and for
//! 512 or more the true nearest neighbours where faiss's IVF index (`nprobe = 8`) is
//! approximate (docs/CPP_DIVERGENCES.md entry 231).
//!
//! Tier A below 512 descriptors, and exact nearest neighbours in general:
//! - SIFT descriptors whose float values are integers in [0, 255] (what
//!   [`FeatureDescriptors::to_float`](super::types::FeatureDescriptors::to_float) produces) are
//!   searched as bytes with integer squared distances, which are exact in `f32` (at most
//!   128 * 255^2 < 2^24).
//! - Anything else (learned descriptors) sums `(a - b)^2` in `f32`, left to right.
//! - Neighbours are ordered by (distance, index). faiss leaves the order of equal distances
//!   unspecified; COLMAP's ratio test rejects a tie for the best anyway.
//!
//! The search runs sequentially: the core crate has no `parallel` feature yet, and the result
//! never depends on how query rows would be split across threads (each writes its own row).
//! `num_threads` is kept for COLMAP's `Create` signature.
//!
//! The trait shape, which the matching controllers (Phase 11) build on:
//! - `Build` / `Search` return `Result` (`Search`'s `THROW_CHECK_EQ`s are `Err` with
//!   `ErrorKind::InvalidArgument`); `Search` writes into `&mut` outputs as COLMAP does, and
//!   leaves them untouched when there are no queries.
//! - `Search` takes a [`CancelToken`], polled once per query row, because the matcher's
//!   default path spends its time here (CLAUDE.md "Cancellation and progress").
//! - Indexes are shared between matchers through [`FeatureDescriptorIndexCache`], the seam
//!   for COLMAP's `ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>`: the cache builds each
//!   image's index once and hands out `Arc`s. [`ThreadSafeLRUCache`] over
//!   `dyn FeatureDescriptorIndex` implements it, which is what COLMAP's option points at; the
//!   trait keeps the matcher options independent of the cache's key and load-function types.

use std::sync::Arc;

use super::sift::match_kernels::{dot_product, squared_norms, SIFT_DIM};
use super::types::{FeatureDescriptorsFloat, FeatureExtractorType};
use crate::check_eq;
use crate::linalg::RowMajorMatrix;
use crate::util::cache::ThreadSafeLRUCache;
use crate::util::cancellation::CancelToken;
use crate::util::check::Result;
use crate::util::types::ImageId;

/// Port of `FeatureDescriptorIndex::Type`. COLMAP's `DEFAULT` and `FAISS` are both 1; the only
/// index here is the exact search that replaces faiss.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum FeatureDescriptorIndexType {
    /// `DEFAULT` (= `FAISS`): the exact k-NN search.
    #[default]
    Default = 1,
}

impl FeatureDescriptorIndexType {
    /// `Type::FAISS`, the same value as `Type::DEFAULT` in COLMAP.
    pub const FAISS: FeatureDescriptorIndexType = FeatureDescriptorIndexType::Default;
}

/// Port of `colmap::FeatureDescriptorIndex`: a k-nearest-neighbour descriptor index.
pub trait FeatureDescriptorIndex: Send + Sync {
    /// `Build`: indexes `descriptors` (one per row), replacing any previous contents.
    fn build(&mut self, descriptors: &FeatureDescriptorsFloat) -> Result<()>;

    /// `Search`: the `num_neighbors` nearest indexed descriptors of each query row (fewer when
    /// the index is smaller), nearest first, with their squared L2 distances. Both outputs are
    /// 0 x 0 when `num_neighbors` is not positive or the index is empty, and left untouched
    /// when there are no queries, like COLMAP. `Err` when the query type differs from the
    /// indexed type or the widths differ, or when `cancel` was cancelled mid-search (the
    /// outputs are then left as they were).
    fn search(
        &self,
        num_neighbors: i32,
        query_descriptors: &FeatureDescriptorsFloat,
        indices: &mut RowMajorMatrix<i32>,
        l2_dists: &mut RowMajorMatrix<f32>,
        cancel: &CancelToken,
    ) -> Result<()>;
}

/// The seam for `ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>`
/// (`SiftMatchingOptions::cpu_descriptor_index_cache`): the built index of an image.
pub trait FeatureDescriptorIndexCache: Send + Sync {
    /// `Get(image_id)`: the image's index, built on first use.
    fn get(&self, image_id: ImageId) -> Result<Arc<dyn FeatureDescriptorIndex>>;
}

/// COLMAP's `ThreadSafeLRUCache<image_t, FeatureDescriptorIndex>` itself.
impl FeatureDescriptorIndexCache for ThreadSafeLRUCache<ImageId, dyn FeatureDescriptorIndex> {
    fn get(&self, image_id: ImageId) -> Result<Arc<dyn FeatureDescriptorIndex>> {
        ThreadSafeLRUCache::get(self, &image_id)
    }
}

/// Port of `FeatureDescriptorIndex::Create`. `num_threads` is accepted for COLMAP's
/// signature; the search is sequential (module header).
pub fn create_feature_descriptor_index(
    index_type: FeatureDescriptorIndexType,
    num_threads: i32,
) -> Box<dyn FeatureDescriptorIndex> {
    match index_type {
        FeatureDescriptorIndexType::Default => Box::new(ExactFeatureDescriptorIndex {
            num_threads,
            ..ExactFeatureDescriptorIndex::default()
        }),
    }
}

/// The indexed rows: bytes (SIFT with integral values) with their squared norms, or floats.
#[derive(Debug, Clone)]
enum IndexedRows {
    Empty,
    Bytes {
        rows: RowMajorMatrix<u8>,
        norms: Vec<i32>,
    },
    Floats(RowMajorMatrix<f32>),
}

/// Exact k-NN replacement of COLMAP's `FaissFeatureDescriptorIndex` (module header).
#[derive(Debug, Clone)]
struct ExactFeatureDescriptorIndex {
    #[allow(dead_code)] // COLMAP's thread count; the search is sequential (module header).
    num_threads: i32,
    extractor_type: FeatureExtractorType,
    num_indexed: usize,
    dim: usize,
    rows: IndexedRows,
}

impl Default for ExactFeatureDescriptorIndex {
    fn default() -> Self {
        Self {
            num_threads: 1,
            extractor_type: FeatureExtractorType::Undefined,
            num_indexed: 0,
            dim: 0,
            rows: IndexedRows::Empty,
        }
    }
}

/// Neighbour counts up to which a sorted insertion beats sorting all distances.
const MAX_INSERTION_NEIGHBORS: usize = 8;

impl FeatureDescriptorIndex for ExactFeatureDescriptorIndex {
    fn build(&mut self, descriptors: &FeatureDescriptorsFloat) -> Result<()> {
        self.extractor_type = descriptors.extractor_type;
        self.num_indexed = descriptors.data.rows();
        self.dim = descriptors.data.cols();
        self.rows = if self.num_indexed == 0 {
            IndexedRows::Empty
        } else if let Some(rows) = try_to_bytes(descriptors) {
            let norms = squared_norms(&rows);
            IndexedRows::Bytes { rows, norms }
        } else {
            IndexedRows::Floats(descriptors.data.clone())
        };
        Ok(())
    }

    fn search(
        &self,
        num_neighbors: i32,
        query_descriptors: &FeatureDescriptorsFloat,
        indices: &mut RowMajorMatrix<i32>,
        l2_dists: &mut RowMajorMatrix<f32>,
        cancel: &CancelToken,
    ) -> Result<()> {
        check_eq!(query_descriptors.extractor_type, self.extractor_type);

        if num_neighbors <= 0 || self.num_indexed == 0 {
            *indices = RowMajorMatrix::new(0, 0);
            *l2_dists = RowMajorMatrix::new(0, 0);
            return Ok(());
        }

        check_eq!(query_descriptors.data.cols(), self.dim);
        let num_queries = query_descriptors.data.rows();
        if num_queries == 0 {
            return Ok(());
        }

        // `num_neighbors` is positive here, so the cast is exact.
        let k = (num_neighbors as usize).min(self.num_indexed);
        let mut out_indices = RowMajorMatrix::<i32>::new(num_queries, k);
        let mut out_dists = RowMajorMatrix::<f32>::new(num_queries, k);

        // The byte path needs byte queries; otherwise the float path runs on float index
        // rows, converted when the index itself holds bytes.
        let query_bytes = match &self.rows {
            IndexedRows::Bytes { .. } => try_to_bytes(query_descriptors),
            _ => None,
        };
        match (&self.rows, query_bytes) {
            (IndexedRows::Bytes { rows, norms }, Some(queries)) => {
                let query_norms = squared_norms(&queries);
                let mut dists = vec![0.0f32; self.num_indexed];
                for (q, &query_norm) in query_norms.iter().enumerate() {
                    cancel.check_cancelled()?;
                    let a = queries.row(q);
                    for (i, dist) in dists.iter_mut().enumerate() {
                        let dot = dot_product(a, rows.row(i));
                        // An integer below 2^24 (module header), so the cast is exact.
                        *dist = (query_norm + norms[i] - 2 * dot) as f32;
                    }
                    select_nearest(&dists, k, out_indices.row_mut(q), out_dists.row_mut(q));
                }
            }
            (IndexedRows::Bytes { rows, .. }, None) => {
                let floats = RowMajorMatrix::from_vec(
                    rows.rows(),
                    rows.cols(),
                    rows.as_slice().iter().map(|&v| f32::from(v)).collect(),
                );
                search_floats(
                    query_descriptors,
                    &floats,
                    k,
                    &mut out_indices,
                    &mut out_dists,
                    cancel,
                )?;
            }
            (IndexedRows::Floats(floats), _) => {
                search_floats(
                    query_descriptors,
                    floats,
                    k,
                    &mut out_indices,
                    &mut out_dists,
                    cancel,
                )?;
            }
            // `num_indexed > 0` here, so the rows are never empty.
            (IndexedRows::Empty, _) => {}
        }

        *indices = out_indices;
        *l2_dists = out_dists;
        Ok(())
    }
}

/// The float path: `(a - b)^2` summed in `f32`, left to right, for every query row.
fn search_floats(
    queries: &FeatureDescriptorsFloat,
    floats: &RowMajorMatrix<f32>,
    k: usize,
    out_indices: &mut RowMajorMatrix<i32>,
    out_dists: &mut RowMajorMatrix<f32>,
    cancel: &CancelToken,
) -> Result<()> {
    let mut dists = vec![0.0f32; floats.rows()];
    for q in 0..queries.data.rows() {
        cancel.check_cancelled()?;
        let a = queries.data.row(q);
        for (i, dist) in dists.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (&x, &y) in a.iter().zip(floats.row(i)) {
                let d = x - y;
                sum += d * d;
            }
            *dist = sum;
        }
        select_nearest(&dists, k, out_indices.row_mut(q), out_dists.row_mut(q));
    }
    Ok(())
}

/// Offers `(d, index)` to a query's sorted list of its `count` (at most `k`) nearest.
/// Indices arrive in ascending order and a later equal distance goes after, so ties keep
/// ascending index order.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // `!(d < worst)`: a NaN distance never enters.
fn insert_nearest(
    d: f32,
    index: usize,
    k: usize,
    count: &mut usize,
    out_indices: &mut [i32],
    out_dists: &mut [f32],
) {
    if *count == k && !(d < out_dists[k - 1]) {
        return;
    }
    let mut pos = if *count == k {
        k - 1
    } else {
        *count += 1;
        *count - 1
    };
    while pos > 0 && d < out_dists[pos - 1] {
        out_dists[pos] = out_dists[pos - 1];
        out_indices[pos] = out_indices[pos - 1];
        pos -= 1;
    }
    out_dists[pos] = d;
    // An index row count is far below i32::MAX (COLMAP's indices are int).
    out_indices[pos] = index as i32;
}

/// The `k` smallest distances ordered by (distance, index).
fn select_nearest(dists: &[f32], k: usize, out_indices: &mut [i32], out_dists: &mut [f32]) {
    if k <= MAX_INSERTION_NEIGHBORS {
        let mut count = 0;
        for (i, &d) in dists.iter().enumerate() {
            insert_nearest(d, i, k, &mut count, out_indices, out_dists);
        }
        return;
    }
    let mut order: Vec<usize> = (0..dists.len()).collect();
    // Explicit index tie-break: equal distances keep ascending index order (module header).
    order.sort_unstable_by(|&x, &y| dists[x].total_cmp(&dists[y]).then(x.cmp(&y)));
    for j in 0..k {
        out_indices[j] = order[j] as i32;
        out_dists[j] = dists[order[j]];
    }
}

/// SIFT rows of integral values in [0, 255] (what `ToFloat` produces), 128 wide, as bytes;
/// `None` for anything else.
fn try_to_bytes(descriptors: &FeatureDescriptorsFloat) -> Option<RowMajorMatrix<u8>> {
    if descriptors.extractor_type != FeatureExtractorType::Sift
        || descriptors.data.cols() != SIFT_DIM
    {
        return None;
    }
    let mut converted = Vec::with_capacity(descriptors.data.len());
    for &v in descriptors.data.as_slice() {
        if !(0.0..=255.0).contains(&v) || v != v.floor() {
            return None;
        }
        // An integer in [0, 255], so the cast is exact.
        converted.push(v as u8);
    }
    Some(RowMajorMatrix::from_vec(
        descriptors.data.rows(),
        descriptors.data.cols(),
        converted,
    ))
}
