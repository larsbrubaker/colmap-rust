// Rust-only checks of colmap_rust::scene::scene_clustering beyond scene_clustering_test.cc:
// the flat child order and Create's pair order (both described in the
// `scene/scene_clustering.rs` header).

use std::collections::{BTreeSet, HashSet};

use colmap_rust::math::random::set_prng_seed;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::database_cache::{DatabaseCache, DatabaseCacheOptions};
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::scene_clustering::{SceneClustering, SceneClusteringOptions};
use colmap_rust::scene::synthetic::{synthesize_dataset_with_database, SyntheticDatasetOptions};

fn flat_children(image_pairs: &[(u32, u32)], num_inliers: &[i32], branching: i32) -> Vec<Vec<u32>> {
    let options = SceneClusteringOptions {
        branching,
        image_overlap: 0,
        is_hierarchical: false,
        ..SceneClusteringOptions::default()
    };
    let mut scene_clustering = SceneClustering::new(options).unwrap();
    scene_clustering
        .partition(image_pairs, num_inliers)
        .unwrap();
    scene_clustering
        .root_cluster()
        .unwrap()
        .child_clusters
        .iter()
        .map(|child| child.image_ids.clone())
        .collect()
}

#[test]
fn rust_only_flat_child_clusters_ordered_by_size_then_smallest_id() {
    // Equal sizes: ascending smallest image id (scene_clustering_test.cc's one-level graph).
    let one_level = [
        (0, 1),
        (2, 3),
        (4, 5),
        (1, 2),
        (3, 4),
        (5, 0),
        (0, 3),
        (2, 5),
        (4, 1),
    ];
    assert_eq!(
        flat_children(&one_level, &[100, 100, 100, 10, 10, 10, 1, 1, 1], 3),
        vec![vec![0, 1], vec![2, 3], vec![4, 5]]
    );

    // Descending size first, even when the larger child has the larger smallest id (the case
    // where libc++ keeps COLMAP's order unchanged).
    assert_eq!(
        flat_children(&[(0, 1), (2, 3), (3, 4), (2, 4)], &[5, 5, 5, 5], 2),
        vec![vec![2, 3, 4], vec![0, 1]]
    );

    // More parts than images: the empty children come last.
    assert_eq!(
        flat_children(&[(0, 1), (1, 2)], &[1, 1], 5),
        vec![vec![0], vec![1], vec![2], vec![], vec![]]
    );
}

// In a synthetic dataset every image pair has the same number of matches, so every
// clustering tie falls to the order in which Create hands over the pairs. Taken in ascending
// pair id, the root split's overlap was images 1-3 and 31-33: one frame per side. For a
// panoramic rig (zero sensor translation) every image of a frame has the same projection
// center, so the two halves would share only two distinct centers, and a similarity that
// aligns them needs at least three.
#[test]
fn rust_only_create_spreads_tied_overlap_across_frames() {
    set_prng_seed(0);
    let database = Database::new();
    let synthetic_options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 3,
        num_frames_per_rig: 10,
        num_points3d: 100,
        sensor_from_rig_translation_stddev: 0.0,
        sensor_from_rig_rotation_stddev: 30.0,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&synthetic_options, &mut Reconstruction::new(), &database)
        .unwrap();
    let database_cache =
        DatabaseCache::create(&database, &DatabaseCacheOptions::default()).unwrap();

    let options = SceneClusteringOptions {
        leaf_max_num_images: 10,
        image_overlap: 3,
        ..SceneClusteringOptions::default()
    };
    let scene_clustering = SceneClustering::create(options, &database_cache).unwrap();

    let children = &scene_clustering.root_cluster().unwrap().child_clusters;
    assert_eq!(children.len(), 2);
    let second: BTreeSet<u32> = children[1].image_ids.iter().copied().collect();
    let shared_frame_ids: HashSet<u32> = children[0]
        .image_ids
        .iter()
        .filter(|image_id| second.contains(image_id))
        .map(|&image_id| database_cache.image(image_id).unwrap().frame_id())
        .collect();
    assert!(
        shared_frame_ids.len() >= 3,
        "shared frames: {shared_frame_ids:?}"
    );
}
