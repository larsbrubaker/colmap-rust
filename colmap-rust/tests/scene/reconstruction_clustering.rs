// Port of colmap/scene/reconstruction_clustering_test.cc, 1:1 (same test names, values and
// checks), for colmap_rust::scene::reconstruction_clustering::cluster_reconstruction_frames.
// COLMAP's gtest main seeds the PRNG with 0 before every test, so every case calls
// `set_prng_seed(0)` first. The helper `PartitionFramesIntoClusters` walks the points in
// ascending id order and the observing clusters in ascending cluster id order where COLMAP
// walks hash maps, so its random keep/delete draws land on other observations than in COLMAP;
// the checks are on outcomes (memberships), which do not depend on that.

use std::collections::{BTreeMap, HashSet};

use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::reconstruction_clustering::{
    cluster_reconstruction_frames, ReconstructionClusteringOptions,
};
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::scene::track::TrackElement;
use colmap_rust::util::types::FrameId;

// Partitions frames into clusters based on assigned cluster ids. Each cluster will only share
// 3D points among its member frames, and observations connecting frames from different
// clusters will be removed, except for a `keep_ratio` fraction of them. This effectively
// creates isolated connected components in the covisibility graph.
pub(crate) fn partition_frames_into_clusters(
    reconstruction: &mut Reconstruction,
    frame_to_cluster: &BTreeMap<FrameId, i32>,
    keep_ratio: f64,
) {
    // Collect all observations to delete first, because deleting while iterating is unsafe.
    let mut observations_to_delete = Vec::new();

    for (&point3d_id, point3d) in reconstruction.points3d() {
        // Determine which clusters observe this 3D point.
        let mut cluster_observations: BTreeMap<i32, Vec<TrackElement>> = BTreeMap::new();
        for track_el in point3d.track.elements() {
            let frame_id = reconstruction.image(track_el.image_id).unwrap().frame_id();
            if let Some(&cluster_id) = frame_to_cluster.get(&frame_id) {
                cluster_observations
                    .entry(cluster_id)
                    .or_default()
                    .push(*track_el);
            }
        }

        // If multiple clusters observe this point, pick one cluster to keep this point and
        // remove observations from all other clusters, but keep a fraction (keep_ratio) of
        // cross-cluster observations.
        if cluster_observations.len() > 1 {
            let cluster_ids_vec: Vec<i32> = cluster_observations.keys().copied().collect();
            let chosen_cluster =
                cluster_ids_vec[(point3d_id % cluster_ids_vec.len() as u64) as usize];

            for (&cluster_id, observations) in &cluster_observations {
                if cluster_id == chosen_cluster {
                    continue;
                }
                for observation in observations {
                    // Use a random decision to decide whether to keep this observation.
                    let random_value = random_uniform_real(0.0, 1.0);
                    if random_value >= keep_ratio {
                        observations_to_delete
                            .push((observation.image_id, observation.point2d_idx));
                    }
                }
            }
        }
    }

    // Delete the collected observations. An observation may already be gone if its 3D point
    // was removed due to its track becoming too short.
    for (image_id, point2d_idx) in observations_to_delete {
        let image = reconstruction.image(image_id).unwrap();
        if point2d_idx < image.num_points2d() && image.point2d(point2d_idx).has_point3d() {
            reconstruction
                .delete_observation(image_id, point2d_idx)
                .unwrap();
        }
    }
}

// All registered frame ids, sorted for deterministic test behavior.
pub(crate) fn extract_sorted_frame_ids(reconstruction: &Reconstruction) -> Vec<FrameId> {
    let mut frame_ids: Vec<FrameId> = reconstruction
        .frames()
        .iter()
        .filter(|(_, frame)| frame.has_pose())
        .map(|(&frame_id, _)| frame_id)
        .collect();
    frame_ids.sort();
    frame_ids
}

// result[cluster_id] holds the frame ids in that cluster.
pub(crate) fn build_clusters_from_output(
    cluster_ids: &BTreeMap<FrameId, i32>,
) -> Vec<HashSet<FrameId>> {
    let mut clusters: Vec<HashSet<FrameId>> = Vec::new();
    for (&frame_id, &cluster_id) in cluster_ids {
        if cluster_id == -1 {
            continue;
        }
        if clusters.len() <= cluster_id as usize {
            clusters.resize(cluster_id as usize + 1, HashSet::new());
        }
        clusters[cluster_id as usize].insert(frame_id);
    }
    clusters
}

fn synthesize(
    num_cameras_per_rig: i32,
    num_frames_per_rig: usize,
    num_points3d: i32,
) -> (Reconstruction, SyntheticDatasetOptions) {
    set_prng_seed(0);
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig,
        num_frames_per_rig: num_frames_per_rig as i32,
        num_points3d,
        num_points2d_without_point3d: 0,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();
    (reconstruction, synthetic_dataset_options)
}

// Assigns the sorted frames to consecutive clusters of the given sizes: cluster c gets the
// next sizes[c] frames. Returns the frame -> cluster map and each cluster's frames.
fn assign_clusters(
    all_frame_ids: &[FrameId],
    sizes: &[usize],
) -> (BTreeMap<FrameId, i32>, Vec<HashSet<FrameId>>) {
    let mut frame_to_cluster = BTreeMap::new();
    let mut expected_clusters = vec![HashSet::new(); sizes.len()];
    let mut next = 0;
    for (cluster_id, &size) in sizes.iter().enumerate() {
        for &frame_id in &all_frame_ids[next..next + size] {
            frame_to_cluster.insert(frame_id, cluster_id as i32);
            expected_clusters[cluster_id].insert(frame_id);
        }
        next += size;
    }
    (frame_to_cluster, expected_clusters)
}

#[test]
fn cluster_reconstruction_frames_empty() {
    set_prng_seed(0);
    let reconstruction = Reconstruction::new();
    let options = ReconstructionClusteringOptions::default();
    let cluster_ids = cluster_reconstruction_frames(&options, &reconstruction).unwrap();
    assert!(cluster_ids.is_empty());
}

#[test]
fn cluster_reconstruction_frames_well_connected_reconstruction() {
    let (reconstruction, _) = synthesize(1, 5, 100);

    let initial_num_reg_frames = reconstruction.num_reg_frames();
    assert_eq!(initial_num_reg_frames, 5);

    let options = ReconstructionClusteringOptions::default();
    let cluster_ids = cluster_reconstruction_frames(&options, &reconstruction).unwrap();

    // All frames should be assigned cluster 0
    for &cluster_id in cluster_ids.values() {
        assert_eq!(cluster_id, 0);
    }
}

#[test]
fn cluster_reconstruction_frames_weakly_connected_reconstruction() {
    // Create a reconstruction with very few 3D points (10 total). With so few shared
    // observations, the covisibility between frames is weak and each frame should be
    // assigned to its own cluster.
    let (reconstruction, synthetic_dataset_options) = synthesize(1, 10, 10);

    let k_num_frames = reconstruction.num_reg_frames();
    assert_eq!(k_num_frames, 10);

    let options = ReconstructionClusteringOptions {
        min_edge_weight_threshold: f64::from(synthetic_dataset_options.num_points3d + 1),
        ..ReconstructionClusteringOptions::default()
    };
    let cluster_ids = cluster_reconstruction_frames(&options, &reconstruction).unwrap();

    // All frames should be assigned to clusters.
    assert_eq!(cluster_ids.len(), k_num_frames);

    // Each frame should get id -1
    for &cluster_id in cluster_ids.values() {
        assert_eq!(cluster_id, -1);
    }
}

#[test]
fn cluster_reconstruction_frames_one_major_connected_component() {
    // Create a reconstruction with 10 frames, all initially well-connected.
    let (mut reconstruction, _) = synthesize(1, 10, 250);

    let initial_num_reg_frames = reconstruction.num_reg_frames();
    assert_eq!(initial_num_reg_frames, 10);

    // Get all frame IDs sorted for deterministic behavior.
    let all_frame_ids = extract_sorted_frame_ids(&reconstruction);

    // Partition into one cluster and independent frames: first 8 frames in cluster 0, other
    // frames are independent
    let k_large_cluster_size = 8;
    let mut frame_to_cluster = BTreeMap::new();
    for &frame_id in &all_frame_ids[..k_large_cluster_size] {
        frame_to_cluster.insert(frame_id, 0);
    }
    for (i, &frame_id) in all_frame_ids
        .iter()
        .enumerate()
        .take(10)
        .skip(k_large_cluster_size)
    {
        frame_to_cluster.insert(frame_id, (i - k_large_cluster_size + 1) as i32);
    }

    // Partition the reconstruction to disconnect the clusters. With keep_ratio=0.1, some
    // weak connections may remain.
    partition_frames_into_clusters(&mut reconstruction, &frame_to_cluster, 0.1);

    assert_eq!(reconstruction.num_reg_frames(), initial_num_reg_frames);

    let options = ReconstructionClusteringOptions::default();
    let cluster_ids = cluster_reconstruction_frames(&options, &reconstruction).unwrap();

    // All frames should be assigned to clusters.
    assert_eq!(cluster_ids.len(), initial_num_reg_frames);

    // Build the resulting clusters.
    let clusters = build_clusters_from_output(&cluster_ids);

    // Find the largest cluster.
    let mut largest_cluster_idx = 0;
    for i in 1..clusters.len() {
        if clusters[i].len() > clusters[largest_cluster_idx].len() {
            largest_cluster_idx = i;
        }
    }
    // The largest cluster should be cluster 0.
    assert_eq!(largest_cluster_idx, 0);

    // The largest cluster should have exactly kLargeClusterSize frames.
    assert_eq!(clusters[largest_cluster_idx].len(), k_large_cluster_size);

    // Other clusters should be single-frame clusters.
    for (i, cluster) in clusters.iter().enumerate() {
        if i != largest_cluster_idx {
            assert_eq!(cluster.len(), 1);
        }
    }
}

// The shared body of the three-cluster cases: partition the sorted frames into clusters of
// `sizes`, cut the cross-cluster observations with `keep_ratio`, and expect exactly those
// clusters back, largest first.
fn check_three_clusters(reconstruction: &mut Reconstruction, sizes: [usize; 3], keep_ratio: f64) {
    let total_frames: usize = sizes.iter().sum();
    let all_frame_ids = extract_sorted_frame_ids(reconstruction);
    let (frame_to_cluster, mut expected_clusters) = assign_clusters(&all_frame_ids, &sizes);

    partition_frames_into_clusters(reconstruction, &frame_to_cluster, keep_ratio);

    // All frames should still be registered.
    assert_eq!(reconstruction.num_reg_frames(), total_frames);

    let options = ReconstructionClusteringOptions::default();
    let cluster_ids = cluster_reconstruction_frames(&options, reconstruction).unwrap();

    // All frames should be assigned to clusters.
    assert_eq!(cluster_ids.len(), total_frames);

    // Build the resulting clusters.
    let result_clusters = build_clusters_from_output(&cluster_ids);

    // Should have exactly 3 clusters
    assert_eq!(result_clusters.len(), 3);

    // Sort expected cluster by size for comparison. Result clusters are already sorted by
    // size due to implementation.
    expected_clusters.sort_by_key(|cluster| std::cmp::Reverse(cluster.len()));

    // Verify that the clusters match exactly.
    assert_eq!(result_clusters[0], expected_clusters[0]);
    assert_eq!(result_clusters[1], expected_clusters[1]);
    assert_eq!(result_clusters[2], expected_clusters[2]);
}

#[test]
fn cluster_reconstruction_frames_multiple_weakly_connected_clusters() {
    // Create a reconstruction with frames that will be partitioned into weakly connected
    // clusters (some cross-cluster connections remain).
    let sizes = [25, 5, 4];
    let k_total_frames: usize = sizes.iter().sum();
    let (mut reconstruction, _) = synthesize(1, k_total_frames, 400);
    assert_eq!(reconstruction.num_reg_frames(), k_total_frames);

    // Partition with keep_ratio=0.1 to leave some weak connections.
    check_three_clusters(&mut reconstruction, sizes, 0.1);
}

#[test]
fn cluster_reconstruction_frames_multiple_disjoint_clusters() {
    // Create a reconstruction with completely disjoint clusters (no shared observations
    // between clusters). Verify that the clustering exactly matches the original partition.
    let sizes = [10, 8, 6];
    let k_total_frames: usize = sizes.iter().sum();
    let (mut reconstruction, _) = synthesize(1, k_total_frames, 500);
    assert_eq!(reconstruction.num_reg_frames(), k_total_frames);

    // Use keep_ratio=0.0 to create completely disjoint clusters with no shared observations
    // between them.
    check_three_clusters(&mut reconstruction, sizes, 0.0);
}

// Tests with non-trivial rigs (multiple cameras per frame).

#[test]
fn cluster_reconstruction_frames_rig_one_major_connected_component() {
    // Create a reconstruction with 10 frames from a rig with 3 cameras each. Test that
    // clustering correctly handles multi-camera rigs. Note: With multi-camera rigs,
    // covisibility between frames is stronger because 3D points are often visible from
    // multiple cameras in the same frame.
    let (mut reconstruction, _) = synthesize(3, 10, 300);

    let initial_num_reg_frames = reconstruction.num_reg_frames();
    assert_eq!(initial_num_reg_frames, 10);
    // Should have 30 images (10 frames * 3 cameras).
    assert_eq!(reconstruction.num_reg_images(), 30);

    // Get all frame IDs sorted for deterministic behavior.
    let all_frame_ids = extract_sorted_frame_ids(&reconstruction);

    // Partition into one large cluster and independent frames: first 7 frames in cluster 0,
    // other 3 frames are independent.
    let k_large_cluster_size = 7;
    let mut frame_to_cluster = BTreeMap::new();
    let mut expected_large_cluster = HashSet::new();
    for &frame_id in &all_frame_ids[..k_large_cluster_size] {
        frame_to_cluster.insert(frame_id, 0);
        expected_large_cluster.insert(frame_id);
    }
    for (i, &frame_id) in all_frame_ids
        .iter()
        .enumerate()
        .take(10)
        .skip(k_large_cluster_size)
    {
        frame_to_cluster.insert(frame_id, (i - k_large_cluster_size + 1) as i32);
    }

    // Use keep_ratio=0.0 to completely disconnect clusters. With multi-camera rigs, weak
    // connections are harder to break.
    partition_frames_into_clusters(&mut reconstruction, &frame_to_cluster, 0.0);

    assert_eq!(reconstruction.num_reg_frames(), initial_num_reg_frames);

    let options = ReconstructionClusteringOptions::default();
    let cluster_ids = cluster_reconstruction_frames(&options, &reconstruction).unwrap();

    // All frames should be assigned to clusters.
    assert_eq!(cluster_ids.len(), initial_num_reg_frames);

    // Build the resulting clusters.
    let clusters = build_clusters_from_output(&cluster_ids);

    // Should be only 1 large cluster
    assert_eq!(clusters.len(), 1);
    // The largest cluster (cluster 0) should have exactly kLargeClusterSize.
    assert_eq!(clusters[0].len(), k_large_cluster_size);
    assert_eq!(clusters[0], expected_large_cluster);

    for &frame_id in &all_frame_ids[k_large_cluster_size..10] {
        // Other frames should not be in any cluster.
        assert_eq!(cluster_ids[&frame_id], -1);
    }
}

#[test]
fn cluster_reconstruction_frames_rig_multiple_weakly_connected_clusters() {
    // Create a reconstruction with frames from a rig with 2 cameras each. Partition into 3
    // clusters. With multi-camera rigs, the covisibility between frames is higher, so we use
    // very low keep_ratio to ensure cluster separation while still leaving some weak
    // connections.
    let sizes = [30, 4, 3];
    let k_total_frames: usize = sizes.iter().sum();
    let (mut reconstruction, _) = synthesize(2, k_total_frames, 500);

    assert_eq!(reconstruction.num_reg_frames(), k_total_frames);
    // Should have 60 images (30 frames * 2 cameras).
    assert_eq!(reconstruction.num_reg_images(), k_total_frames * 2);

    // Use keep_ratio=0.05 to create disjoint clusters.
    check_three_clusters(&mut reconstruction, sizes, 0.05);
}

#[test]
fn cluster_reconstruction_frames_rig_multiple_disjoint_clusters() {
    // Create a reconstruction with frames from a rig with 4 cameras each. Partition into 3
    // completely disjoint clusters.
    let sizes = [8, 6, 4];
    let k_total_frames: usize = sizes.iter().sum();
    let k_cameras_per_rig = 4;
    let (mut reconstruction, _) = synthesize(k_cameras_per_rig, k_total_frames, 600);

    assert_eq!(reconstruction.num_reg_frames(), k_total_frames);
    // Should have 72 images (18 frames * 4 cameras).
    assert_eq!(
        reconstruction.num_reg_images(),
        k_total_frames * k_cameras_per_rig as usize
    );

    // Use keep_ratio=0.0 to create completely disjoint clusters.
    check_three_clusters(&mut reconstruction, sizes, 0.0);
}
