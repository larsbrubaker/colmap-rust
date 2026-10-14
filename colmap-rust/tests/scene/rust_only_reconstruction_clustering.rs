// Rust-only checks of colmap_rust::scene::reconstruction_clustering beyond
// reconstruction_clustering_test.cc: equally large clusters are numbered by ascending smallest
// frame id (docs/CPP_DIVERGENCES.md entry 284), and invalid options are rejected.

use std::collections::{BTreeMap, HashSet};

use colmap_rust::math::random::set_prng_seed;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::reconstruction_clustering::{
    cluster_reconstruction_frames, ReconstructionClusteringOptions,
};
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};

use crate::reconstruction_clustering::{
    build_clusters_from_output, extract_sorted_frame_ids, partition_frames_into_clusters,
};

#[test]
fn rust_only_equal_size_clusters_numbered_by_smallest_frame_id() {
    set_prng_seed(0);
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 8,
        num_points3d: 400,
        num_points2d_without_point3d: 0,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    let frame_ids = extract_sorted_frame_ids(&reconstruction);

    // Two disjoint clusters of four frames; the one with the larger frame ids gets the
    // smaller test-side id, so the numbering cannot come from it.
    let mut frame_to_cluster = BTreeMap::new();
    for (i, &frame_id) in frame_ids.iter().enumerate() {
        frame_to_cluster.insert(frame_id, if i < 4 { 1 } else { 0 });
    }
    partition_frames_into_clusters(&mut reconstruction, &frame_to_cluster, 0.0);

    let cluster_ids =
        cluster_reconstruction_frames(&ReconstructionClusteringOptions::default(), &reconstruction)
            .unwrap();
    let clusters = build_clusters_from_output(&cluster_ids);
    assert_eq!(clusters.len(), 2);
    assert_eq!(
        clusters[0],
        frame_ids[..4].iter().copied().collect::<HashSet<_>>()
    );
    assert_eq!(
        clusters[1],
        frame_ids[4..].iter().copied().collect::<HashSet<_>>()
    );
}

#[test]
fn rust_only_invalid_options_are_rejected() {
    let reconstruction = Reconstruction::new();
    for options in [
        ReconstructionClusteringOptions {
            min_covisibility_count: 0,
            ..ReconstructionClusteringOptions::default()
        },
        ReconstructionClusteringOptions {
            min_edge_weight_threshold: 0.0,
            ..ReconstructionClusteringOptions::default()
        },
        ReconstructionClusteringOptions {
            min_num_reg_frames: 1,
            ..ReconstructionClusteringOptions::default()
        },
    ] {
        assert!(cluster_reconstruction_frames(&options, &reconstruction).is_err());
    }
}
