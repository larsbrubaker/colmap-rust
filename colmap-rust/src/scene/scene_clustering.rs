//! Port of `colmap/scene/scene_clustering.h` / `.cc` (and colmap-sharp's
//! `Scene/SceneClustering.cs`): splits the scene graph (images linked by their number of
//! inlier matches) into overlapping clusters with the normalized graph cut of
//! [`crate::math::graph_cut`], either hierarchically until every leaf is small enough or flat
//! into `branching` clusters. Tests: `tests/scene/scene_clustering.rs`
//! (`scene_clustering_test.cc` 1:1) and `tests/scene/rust_only_scene_clustering.rs`.
//!
//! Tier C (outcome): the partition itself comes from our multilevel partitioner, not METIS
//! (see `math/graph_cut_partitioner.rs`), so the memberships match COLMAP's tests but the labels
//! and the order of the child clusters can differ.
//!
//! Translation notes:
//! - Ties (entry 284). COLMAP sorts the overlap-candidate edges and each image's related
//!   images by descending weight with `std::sort`, which leaves equal weights in an
//!   unspecified order. Here equal weights keep their edge order (a stable sort).
//! - The flat child-cluster sort (entry 290). COLMAP's comparator ("size >= other size and
//!   min id < other min id") is not a strict weak ordering and dereferences `min_element` of
//!   an empty cluster. Here children are ordered by what its comment says it intends:
//!   descending size, then ascending smallest image id, with empty clusters last.
//! - COLMAP's `image_t` -> `int` narrowing of the edge endpoints is kept (`as`, which wraps
//!   as the C++ conversion does), so the graph cut sees the same vertex ids.
//! - `Create`'s pair order (entry 291). COLMAP iterates a hash map of image pairs; here the
//!   pairs are ordered by Boost.Unordered's `mulx` mix of the pair id (ported from Boost,
//!   BSL-1.0, see THIRD_PARTY_NOTICES.md), modelling COLMAP's default Boost backend, so ties
//!   do not all break toward low image ids.
//! - The other hash containers COLMAP uses here (the `FlatHashSet` of a child's images, the
//!   `NodeHashMap` of related images) are only looked up, never iterated in a way that
//!   reaches an output.
//! - `LOG(WARNING)` for an image the graph cut did not label is dropped, like logging
//!   elsewhere in the library; the image still goes to child 0.

use std::collections::{BTreeSet, HashMap, HashSet};

use crate::math::graph_cut::compute_normalized_min_graph_cut;
use crate::scene::database_cache::DatabaseCache;
use crate::util::types::{pair_id_to_image_pair, ImageId, ImagePairId};
use crate::Result;

/// Port of `SceneClustering::Options`.
#[derive(Clone, Debug, PartialEq)]
pub struct SceneClusteringOptions {
    /// Flag for hierarchical vs flat clustering.
    pub is_hierarchical: bool,
    /// The branching factor of the hierarchical clustering.
    pub branching: i32,
    /// The number of overlapping images between child clusters.
    pub image_overlap: i32,
    /// The max related images matches to look for in a flat cluster.
    pub num_image_matches: i32,
    /// The maximum number of images in a leaf node cluster, otherwise the cluster is further
    /// partitioned using the given branching factor. Note that a cluster leaf node will have
    /// at most `leaf_max_num_images + overlap` images to satisfy the overlap constraint.
    pub leaf_max_num_images: i32,
}

impl Default for SceneClusteringOptions {
    fn default() -> Self {
        Self {
            is_hierarchical: true,
            branching: 2,
            image_overlap: 50,
            num_image_matches: 20,
            leaf_max_num_images: 500,
        }
    }
}

impl SceneClusteringOptions {
    /// `Check()` (`CHECK_OPTION_*`: false on a violation).
    pub fn check(&self) -> bool {
        self.branching > 0 && self.image_overlap >= 0
    }
}

/// Port of `SceneClustering::Cluster`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Cluster {
    /// The images of this cluster, including the overlap images.
    pub image_ids: Vec<ImageId>,
    /// The child clusters; empty for a leaf.
    pub child_clusters: Vec<Cluster>,
}

/// Port of `colmap::SceneClustering`: scene clustering using normalized cuts on the scene
/// graph. The scene is hierarchically partitioned into overlapping clusters until a maximum
/// number of images is in a leaf node.
#[derive(Clone, Debug)]
pub struct SceneClustering {
    options: SceneClusteringOptions,
    root_cluster: Option<Cluster>,
}

// Boost.Unordered's mulx_mix (boost/unordered/detail/mulx.hpp, Boost 1.87, BSL-1.0), which
// its open-addressing tables apply to a non-avalanching hash such as the identity
// std::hash<uint64_t> of COLMAP's pair ids: the 128-bit product with the golden-ratio
// constant, low half xor high half. A table of 2^k groups puts the key in group
// mix >> (64 - k) and iterates groups in ascending order, so ascending mixed value is the
// table's order up to the order of keys within one group.
fn boost_mulx_mix(pair_id: ImagePairId) -> u64 {
    let product = u128::from(pair_id) * u128::from(0x9E37_79B9_7F4A_7C15_u64);
    (product as u64) ^ ((product >> 64) as u64)
}

// `labels.at(id)` of the graph cut's labels, as a child index.
fn label_of(labels: &std::collections::BTreeMap<i32, i32>, id: i32) -> Result<usize> {
    let label = labels.get(&id);
    crate::check!(label.is_some());
    Ok(label.copied().unwrap_or_default() as usize)
}

impl SceneClustering {
    /// Creates an unpartitioned clustering; fails if the options are invalid.
    pub fn new(options: SceneClusteringOptions) -> Result<Self> {
        crate::check!(options.check());
        Ok(Self {
            options,
            root_cluster: None,
        })
    }

    /// `Partition`: builds the cluster tree of the scene graph whose edges are `image_pairs`
    /// weighted by `num_inliers`. May only be called once.
    pub fn partition(
        &mut self,
        image_pairs: &[(ImageId, ImageId)],
        num_inliers: &[i32],
    ) -> Result<()> {
        crate::check!(self.root_cluster.is_none());
        crate::check_eq!(image_pairs.len(), num_inliers.len());

        let mut image_ids = BTreeSet::new();
        let mut edges = Vec::with_capacity(image_pairs.len());
        for &(image_id1, image_id2) in image_pairs {
            image_ids.insert(image_id1);
            image_ids.insert(image_id2);
            // image_t -> int, wrapping as COLMAP's narrowing does.
            edges.push((image_id1 as i32, image_id2 as i32));
        }

        let mut root_cluster = Cluster {
            image_ids: image_ids.into_iter().collect(),
            child_clusters: Vec::new(),
        };
        if self.options.is_hierarchical {
            self.partition_hierarchical_cluster(&edges, num_inliers, &mut root_cluster)?;
        } else {
            self.partition_flat_cluster(&edges, num_inliers, &mut root_cluster)?;
        }
        self.root_cluster = Some(root_cluster);
        Ok(())
    }

    /// `GetRootCluster()`: the root of the cluster tree, or `None` before [`Self::partition`].
    pub fn root_cluster(&self) -> Option<&Cluster> {
        self.root_cluster.as_ref()
    }

    /// `GetLeafClusters()`: the clusters without children, in COLMAP's depth-first
    /// (last-pushed first) visiting order. The root is its own only leaf when it has no
    /// children. Fails before [`Self::partition`].
    pub fn leaf_clusters(&self) -> Result<Vec<&Cluster>> {
        crate::check!(self.root_cluster.is_some());
        let mut leaf_clusters = Vec::new();
        let Some(root_cluster) = self.root_cluster.as_ref() else {
            return Ok(leaf_clusters);
        };
        if root_cluster.child_clusters.is_empty() {
            leaf_clusters.push(root_cluster);
            return Ok(leaf_clusters);
        }

        let mut non_leaf_clusters = vec![root_cluster];
        while let Some(cluster) = non_leaf_clusters.pop() {
            for child_cluster in &cluster.child_clusters {
                if child_cluster.child_clusters.is_empty() {
                    leaf_clusters.push(child_cluster);
                } else {
                    non_leaf_clusters.push(child_cluster);
                }
            }
        }
        Ok(leaf_clusters)
    }

    /// `Create(options, database_cache)`: partitions the scene graph of the cache's
    /// correspondence graph, with the number of matches between two images as the edge
    /// weight. Pairs are taken in the order of Boost's mixed hash of their pair ids,
    /// modelling COLMAP's Boost hash map backend (entry 291).
    pub fn create(options: SceneClusteringOptions, database_cache: &DatabaseCache) -> Result<Self> {
        let mut num_matches_between_images = database_cache
            .correspondence_graph()
            .num_matches_between_all_images();
        // COLMAP walks a hash map here, so the edge order, which breaks every weight tie in
        // the graph cut and the overlap selection, has nothing to do with image ids. The
        // correspondence graph's order is ascending pair id, which breaks all ties toward the
        // same few low image ids. Those are usually the images of one rig frame, so sibling
        // clusters could overlap in a single frame, too few to align zero-baseline
        // (panoramic) rigs. The mix is not a bijection, so equal mixed values fall back to
        // the pair id (Boost keeps them in insertion order).
        num_matches_between_images.sort_by_key(|&(pair_id, _)| (boost_mulx_mix(pair_id), pair_id));

        let mut all_image_pairs = Vec::with_capacity(num_matches_between_images.len());
        let mut all_num_inliers = Vec::with_capacity(num_matches_between_images.len());
        for (pair_id, num_matches) in num_matches_between_images {
            all_image_pairs.push(pair_id_to_image_pair(pair_id)?);
            // point2D_t -> int, as COLMAP's static_cast.
            all_num_inliers.push(num_matches as i32);
        }

        let mut scene_clustering = Self::new(options)?;
        scene_clustering.partition(&all_image_pairs, &all_num_inliers)?;
        Ok(scene_clustering)
    }

    fn partition_hierarchical_cluster(
        &self,
        edges: &[(i32, i32)],
        weights: &[i32],
        cluster: &mut Cluster,
    ) -> Result<()> {
        crate::check_eq!(edges.len(), weights.len());

        // If the cluster is small enough, we return from the recursive clustering. COLMAP
        // adds the two limits as size_t, so a negative leaf_max_num_images wraps.
        let max_size = (self.options.leaf_max_num_images as i64 as u64)
            .wrapping_add(self.options.image_overlap as i64 as u64);
        if edges.is_empty() || cluster.image_ids.len() as u64 <= max_size {
            return Ok(());
        }

        let branching = self.options.branching as usize;

        // Partition the cluster using a normalized cut on the scene graph.
        let labels = compute_normalized_min_graph_cut(edges, weights, self.options.branching)?;

        // Assign the images to the clustered child clusters.
        cluster.child_clusters = vec![Cluster::default(); branching];
        for &image_id in &cluster.image_ids {
            // An unlabeled image goes to child 0 (COLMAP logs a warning).
            let label = labels.get(&(image_id as i32)).map_or(0, |&l| l as usize);
            crate::check_lt!(label, branching);
            cluster.child_clusters[label].image_ids.push(image_id);
        }

        // Collect the edges between child clusters as overlap candidates.
        let mut overlapping_edges: Vec<Vec<((i32, i32), i32)>> = vec![Vec::new(); branching];
        for (&edge, &weight) in edges.iter().zip(weights) {
            let label1 = label_of(&labels, edge.0)?;
            let label2 = label_of(&labels, edge.1)?;
            if label1 != label2 {
                overlapping_edges[label1].push((edge, weight));
                overlapping_edges[label2].push((edge, weight));
            }
        }

        // Add overlapping images between sibling clusters before partitioning.
        if self.options.image_overlap > 0 {
            for (i, candidates) in overlapping_edges.iter_mut().enumerate() {
                // Descending weight; equal weights keep their edge order (entry 284).
                candidates.sort_by_key(|edge| std::cmp::Reverse(edge.1));

                let mut overlapping_image_ids = BTreeSet::new();
                for &((image1, image2), _) in candidates.iter() {
                    let image_id = if label_of(&labels, image1)? == i {
                        image2
                    } else {
                        image1
                    };
                    overlapping_image_ids.insert(image_id as ImageId);
                    if overlapping_image_ids.len() >= self.options.image_overlap as usize {
                        break;
                    }
                }

                cluster.child_clusters[i]
                    .image_ids
                    .extend(overlapping_image_ids);
            }
        }

        // Retain the full subgraph induced by every expanded child cluster. Edges incident
        // to inherited overlap images allow the graph cut to assign those images based on
        // all their connections at every subsequent level.
        let mut child_edges = vec![Vec::new(); branching];
        let mut child_weights = vec![Vec::new(); branching];
        for i in 0..branching {
            // Lookup only, never iterated.
            let child_image_ids: HashSet<ImageId> = cluster.child_clusters[i]
                .image_ids
                .iter()
                .copied()
                .collect();
            for (&edge, &weight) in edges.iter().zip(weights) {
                if child_image_ids.contains(&(edge.0 as ImageId))
                    && child_image_ids.contains(&(edge.1 as ImageId))
                {
                    child_edges[i].push(edge);
                    child_weights[i].push(weight);
                }
            }
        }

        // Recursively partition all the child clusters.
        let num_images = cluster.image_ids.len();
        for (i, child_cluster) in cluster.child_clusters.iter_mut().enumerate() {
            // Skip empty clusters or clusters where the current cluster has as many images
            // as its child to avoid infinite loops. This can happen because the normalized
            // cut sometimes decides to put all images into one cluster.
            if child_cluster.image_ids.is_empty() || child_cluster.image_ids.len() == num_images {
                continue;
            }
            self.partition_hierarchical_cluster(&child_edges[i], &child_weights[i], child_cluster)?;
        }

        // Remove empty clusters.
        cluster
            .child_clusters
            .retain(|child_cluster| !child_cluster.image_ids.is_empty());

        // If the child cluster is the same as the current cluster, it is redundant and we can
        // remove it.
        if cluster.child_clusters.len() == 1
            && cluster.image_ids.len() == cluster.child_clusters[0].image_ids.len()
        {
            cluster.child_clusters.clear();
        }
        Ok(())
    }

    fn partition_flat_cluster(
        &self,
        edges: &[(i32, i32)],
        weights: &[i32],
        root_cluster: &mut Cluster,
    ) -> Result<()> {
        crate::check_eq!(edges.len(), weights.len());
        let branching = self.options.branching as usize;

        // Partition the cluster using a normalized cut on the scene graph.
        let labels = compute_normalized_min_graph_cut(edges, weights, self.options.branching)?;

        // Assign the images to the clustered child clusters.
        root_cluster.child_clusters = vec![Cluster::default(); branching];
        for &image_id in &root_cluster.image_ids {
            if let Some(&label) = labels.get(&(image_id as i32)) {
                let label = label as usize;
                crate::check_lt!(label, branching);
                root_cluster.child_clusters[label].image_ids.push(image_id);
            }
        }

        // Sort child clusters by descending size of images and secondarily by lowest image
        // id; empty clusters last (entry 290). The sort is stable, so equal keys keep label
        // order.
        root_cluster.child_clusters.sort_by_key(|child| {
            (
                std::cmp::Reverse(child.image_ids.len()),
                child
                    .image_ids
                    .iter()
                    .min()
                    .map_or(u64::MAX, |&id| u64::from(id)),
            )
        });

        // For each image find all related images with their weights. Lookup only, never
        // iterated in a way that reaches an output: each list is sorted on its own.
        let mut related_images: HashMap<i32, Vec<(i32, i32)>> = HashMap::new();
        for (&(image1, image2), &weight) in edges.iter().zip(weights) {
            related_images
                .entry(image1)
                .or_default()
                .push((image2, weight));
            related_images
                .entry(image2)
                .or_default()
                .push((image1, weight));
        }

        // Sort related images by decreasing weights; equal weights keep their edge order
        // (entry 284).
        for related in related_images.values_mut() {
            related.sort_by_key(|image| std::cmp::Reverse(image.1));
        }

        // For each cluster add as many of the needed matching images up to the max image
        // overlap allowance. We do the process sequentially for each image to ensure that at
        // least we get the best matches first.
        // COLMAP compares j with num_image_matches as size_t, so a negative value means "no
        // limit".
        let num_image_matches = self.options.num_image_matches as i64 as u64;
        for child_cluster in root_cluster.child_clusters.iter_mut() {
            let orig_image_ids = &mut child_cluster.image_ids;
            let mut cluster_images: BTreeSet<i32> =
                orig_image_ids.iter().map(|&id| id as i32).collect();
            let max_size = cluster_images.len() + self.options.image_overlap as usize;
            // Past the longest related-image list no round can add an image, so stopping
            // there gives COLMAP's result without spinning to a "no limit" bound.
            let longest_related = orig_image_ids
                .iter()
                .filter_map(|&id| related_images.get(&(id as i32)))
                .map(Vec::len)
                .max()
                .unwrap_or(0);
            // Check up to all the desired matches.
            let mut j = 0usize;
            while (j as u64) < num_image_matches
                && j < longest_related
                && cluster_images.len() < max_size
            {
                for &image_id in orig_image_ids.iter() {
                    let Some(images) = related_images.get(&(image_id as i32)) else {
                        continue;
                    };
                    if j >= images.len() {
                        continue;
                    }
                    // An image not yet in the cluster goes into the overlap set.
                    cluster_images.insert(images[j].0);
                    if cluster_images.len() >= max_size {
                        break;
                    }
                }
                j += 1;
            }
            *orig_image_ids = cluster_images.into_iter().map(|id| id as ImageId).collect();
        }
        Ok(())
    }
}
