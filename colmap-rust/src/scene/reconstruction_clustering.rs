//! Port of `colmap/scene/reconstruction_clustering.h` / `.cc` (and colmap-sharp's
//! `Scene/ReconstructionClustering.cs`): groups the frames of a [`Reconstruction`] by how
//! many 3D points they share: a covisibility graph, an adaptive median-minus-MAD edge
//! threshold ([`crate::math::utils::median_absolute_deviation`]) and union-find over the
//! strong edges ([`crate::math::union_find`]). Tests: `tests/scene/reconstruction_clustering.rs`
//! (`reconstruction_clustering_test.cc` 1:1) and
//! `tests/scene/rust_only_reconstruction_clustering.rs`.
//!
//! Tier A (exact) for the memberships and for the ids of clusters of different sizes.
//! Translation notes:
//! - Hash-container order. COLMAP counts covisibility in `NodeHashMap`s and walks them; that
//!   order only feeds the union-find and the median, neither of which depends on it (here
//!   the maps are `BTreeMap`s). The one place it leaks is the id of clusters of equal size:
//!   COLMAP sorts the clusters by size with `std::sort`, so equal-size clusters are numbered
//!   in an unspecified (hash-map) order. Here they are numbered by ascending smallest frame
//!   id, a deliberate divergence: the translation rules require a deterministic tie-break
//!   wherever tie order reaches an output, and reproducing absl's or Boost's hash order would
//!   tie the result to one library version. Only equal-size clusters can get swapped ids.
//!   Evidence: all nine `reconstruction_clustering_test.cc` cases
//!   (`tests/scene/reconstruction_clustering.rs`) pass;
//!   `tests/scene/rust_only_reconstruction_clustering.rs`
//!   `rust_only_equal_size_clusters_numbered_by_smallest_frame_id` pins the numbering (and
//!   fails when the tie-break is reversed). Same as colmap-sharp entry 93.
//! - The result is a `BTreeMap` (COLMAP's `NodeHashMap`, whose order carries no meaning).
//! - `ClusterReconstructionFrames` takes the reconstruction by `const` reference here; COLMAP's
//!   non-const reference is never written through.
//! - The `LOG(INFO)` / `LOG(WARNING)` lines are dropped, like logging elsewhere in the library.

use std::collections::{BTreeMap, BTreeSet};

use crate::math::union_find::UnionFind;
use crate::math::utils::median_absolute_deviation;
use crate::scene::reconstruction::Reconstruction;
use crate::util::types::{image_pair_to_pair_id, pair_id_to_image_pair, FrameId, ImagePairId};
use crate::Result;

/// Port of `colmap::ReconstructionClusteringOptions`.
#[derive(Clone, Debug, PartialEq)]
pub struct ReconstructionClusteringOptions {
    /// Minimum number of shared 3D points between two frames to consider them connected in
    /// the covisibility graph.
    pub min_covisibility_count: i32,
    /// Minimum edge weight threshold for clustering. If the adaptive threshold
    /// (median - MAD) falls below this, this value is used instead.
    pub min_edge_weight_threshold: f64,
    /// Minimum number of registered frames required for a cluster to be kept. Clusters with
    /// fewer frames will be discarded.
    pub min_num_reg_frames: i32,
}

impl Default for ReconstructionClusteringOptions {
    fn default() -> Self {
        Self {
            min_covisibility_count: 5,
            min_edge_weight_threshold: 20.0,
            min_num_reg_frames: 3,
        }
    }
}

impl ReconstructionClusteringOptions {
    /// `Check()` (`THROW_CHECK_*`: an error on a violation).
    pub fn check(&self) -> Result<()> {
        crate::check_ge!(self.min_covisibility_count, 1);
        crate::check_gt!(self.min_edge_weight_threshold, 0.0);
        crate::check_ge!(self.min_num_reg_frames, 2);
        Ok(())
    }
}

/// Port of `colmap::ClusterReconstructionFrames`: clusters frames based on 3D point
/// covisibility and marks weakly connected frames.
///
/// Covisibility is the number of 3D points visible in both frames. Frames with high
/// covisibility likely have reliable relative pose estimates, while weakly connected frames
/// may have less reliable geometry. Algorithm: (1) build a covisibility graph where edges
/// connect frames sharing at least `min_covisibility_count` points (counted over the points
/// whose track is longer than 2); (2) compute an adaptive edge weight threshold using median
/// minus median absolute deviation (MAD); (3) cluster frames using union-find, merging
/// strongly connected frames; (4) assign cluster ids sorted by number of frames in descending
/// order (cluster 0 is the largest).
///
/// Returns the cluster id of every registered frame (and every frame seen in a track of
/// length > 2); -1 marks a frame in a cluster smaller than `min_num_reg_frames`. Empty when
/// no frame pair passes `min_covisibility_count`.
pub fn cluster_reconstruction_frames(
    options: &ReconstructionClusteringOptions,
    reconstruction: &Reconstruction,
) -> Result<BTreeMap<FrameId, i32>> {
    options.check()?;

    // Step 1: Compute covisibility counts between all frame pairs. For each 3D point,
    // increment the count for every pair of frames that sees it.
    let mut frame_covisibility_count: BTreeMap<ImagePairId, i32> = BTreeMap::new();
    // Insert all registered frames to the nodes set.
    let mut nodes: BTreeSet<FrameId> = reconstruction.reg_frame_ids().iter().copied().collect();

    for point3d in reconstruction.points3d().values() {
        let track = &point3d.track;
        if track.length() <= 2 {
            continue;
        }
        let frame_ids = track
            .elements()
            .iter()
            .map(|track_el| Ok(reconstruction.image(track_el.image_id)?.frame_id()))
            .collect::<Result<Vec<FrameId>>>()?;
        for (i, &frame_id1) in frame_ids.iter().enumerate() {
            nodes.insert(frame_id1);
            // COLMAP's TODO: this may over-count frame pairs when multiple images from the
            // same frame appear in a track (e.g., rig cameras).
            for &frame_id2 in &frame_ids[i + 1..] {
                if frame_id1 == frame_id2 {
                    continue;
                }
                // frame_t pairs share image_t's pair id encoding (frame_pair_t).
                let pair_id = image_pair_to_pair_id(frame_id1, frame_id2)?;
                *frame_covisibility_count.entry(pair_id).or_insert(0) += 1;
            }
        }
    }

    // Filter edges to keep only reliable connections.
    let edge_weights: BTreeMap<ImagePairId, i32> = frame_covisibility_count
        .into_iter()
        .filter(|&(_, count)| count >= options.min_covisibility_count)
        .collect();

    if edge_weights.is_empty() {
        // COLMAP logs "No valid frame pairs found for clustering".
        return Ok(BTreeMap::new());
    }

    // Compute adaptive threshold using median minus median absolute deviation (MAD).
    let mut weight_values: Vec<i32> = edge_weights.values().copied().collect();
    let (median, mad) = median_absolute_deviation(&mut weight_values)?;
    let threshold = (median - mad).max(options.min_edge_weight_threshold);

    // Cluster frames based on covisibility weights.
    establish_strong_clusters(options, &nodes, &edge_weights, threshold)
}

// EstablishStrongClusters: clusters nodes using union-find, merging nodes connected by
// strong edges (weight >= threshold).
fn establish_strong_clusters(
    options: &ReconstructionClusteringOptions,
    nodes: &BTreeSet<FrameId>,
    edge_weights: &BTreeMap<ImagePairId, i32>,
    edge_weight_threshold: f64,
) -> Result<BTreeMap<FrameId, i32>> {
    let mut uf = UnionFind::new();
    uf.reserve(nodes.len());

    // Create initial clusters from strong edges (weight > threshold). COLMAP's TODO: use
    // different thresholds for different edges based on local statistics.
    for (&pair_id, &weight) in edge_weights {
        if f64::from(weight) >= edge_weight_threshold {
            let (frame_id1, frame_id2) = pair_id_to_image_pair(pair_id)?;
            uf.union(&frame_id1, &frame_id2);
        }
    }

    // Collect nodes by their union-find roots.
    uf.compress();
    let mut root_to_nodes: BTreeMap<FrameId, Vec<FrameId>> = BTreeMap::new();
    for (&node, &root) in uf.parents() {
        root_to_nodes.entry(root).or_default().push(node);
    }

    // Sort by number of frames (largest first); equal sizes by ascending smallest frame id,
    // where COLMAP's order is its hash map's (see the module header).
    let mut sorted_clusters: Vec<Vec<FrameId>> = root_to_nodes.into_values().collect();
    sorted_clusters.sort_by_key(|cluster_nodes| {
        (
            std::cmp::Reverse(cluster_nodes.len()),
            cluster_nodes.iter().min().copied(),
        )
    });

    // Assign cluster ids based on sorted order.
    let mut cluster_ids = BTreeMap::new();
    let mut num_valid_clusters = 0;
    for cluster_nodes in &sorted_clusters {
        // size_t(min_num_reg_frames): Check() guarantees it is at least 2.
        if cluster_nodes.len() >= options.min_num_reg_frames as usize {
            for &node in cluster_nodes {
                cluster_ids.insert(node, num_valid_clusters);
            }
            num_valid_clusters += 1;
        } else {
            // Clusters smaller than min_num_reg_frames are discarded.
            for &node in cluster_nodes {
                cluster_ids.insert(node, -1);
            }
        }
    }

    // Ensure all nodes are assigned a cluster id.
    for &node in nodes {
        if uf.find_if_exists(&node).is_none() {
            cluster_ids.insert(node, -1);
        }
    }

    Ok(cluster_ids)
}
