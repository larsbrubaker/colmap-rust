// Port of colmap/scene/scene_clustering_test.cc, 1:1 (same test names, graphs and expected
// memberships), for colmap_rust::scene::scene_clustering::SceneClustering. Tier C (outcome):
// the partitions come from our graph cut, not METIS (docs/CPP_DIVERGENCES.md entry 46), so,
// as in COLMAP's test, clusters compare as sets of image-id sets.

use std::collections::BTreeSet;

use colmap_rust::scene::scene_clustering::{Cluster, SceneClustering, SceneClusteringOptions};

struct SceneGraph {
    image_pairs: Vec<(u32, u32)>,
    num_inliers: Vec<i32>,
}

// Image connectivity graph:
//
//               100           10
//        (0) -------- (1) -------- (2)
//         |                         |
//        10                        100
//         |                         |
//        (5) -------- (4) -------- (3)
//              100           10
//
// Weak cross-cluster connections (1 inlier):
//        (0) -------- (3)
//        (2) -------- (5)
//        (4) -------- (1)
fn make_scene_graph_one_level() -> SceneGraph {
    SceneGraph {
        image_pairs: vec![
            (0, 1),
            (2, 3),
            (4, 5),
            (1, 2),
            (3, 4),
            (5, 0),
            (0, 3),
            (2, 5),
            (4, 1),
        ],
        num_inliers: vec![100, 100, 100, 10, 10, 10, 1, 1, 1],
    }
}

// Image connectivity graph:
//
//               100          50          100
//        (0) -------- (1) -------- (2) -------- (3)
//         |                                      |
//        10                                      10
//         |                                      |
//        (7) -------- (6) -------- (5) -------- (4)
//              100          50          100
//
// Weak cross-cluster connections (1 inlier):
//        (0) -------- (4)
//        (1) -------- (6)
//        (2) -------- (5)
//        (3) -------- (7)
fn make_scene_graph_two_level() -> SceneGraph {
    SceneGraph {
        image_pairs: vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (4, 5),
            (5, 6),
            (6, 7),
            (0, 7),
            (3, 4),
            (0, 4),
            (1, 6),
            (2, 5),
            (3, 7),
        ],
        num_inliers: vec![100, 50, 100, 100, 50, 100, 10, 10, 1, 1, 1, 1],
    }
}

// Path graph with three levels of progressively weaker connections.
fn make_scene_graph_three_level() -> SceneGraph {
    SceneGraph {
        image_pairs: (0..15).map(|i| (i, i + 1)).collect(),
        num_inliers: vec![
            1000, 100, 1000, 10, 1000, 100, 1000, 1, 1000, 100, 1000, 10, 1000, 100, 1000,
        ],
    }
}

// The first six images form one top-level cluster and the last six images form the other.
// Image 6 overlaps the first cluster. Its strongest individual connection is to image 0, but
// its combined connection to images 3 and 4 is stronger.
fn make_scene_graph_aggregate_overlap() -> SceneGraph {
    SceneGraph {
        image_pairs: vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (6, 7),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
            (6, 0),
            (6, 3),
            (6, 4),
        ],
        num_inliers: vec![
            10000, 10000, 1, 10000, 10000, 20000, 20000, 20000, 20000, 20000, 100, 90, 90,
        ],
    }
}

// UnorderedClustersEq: the clusters as sets, compared as a sorted list of sets.
fn unordered_clusters(clusters: Vec<BTreeSet<u32>>) -> Vec<BTreeSet<u32>> {
    let mut clusters = clusters;
    clusters.sort();
    clusters
}

fn expected(clusters: &[&[u32]]) -> Vec<BTreeSet<u32>> {
    unordered_clusters(
        clusters
            .iter()
            .map(|cluster| cluster.iter().copied().collect())
            .collect(),
    )
}

fn get_leaf_image_sets(leaves: &[&Cluster]) -> Vec<BTreeSet<u32>> {
    unordered_clusters(
        leaves
            .iter()
            .map(|leaf| leaf.image_ids.iter().copied().collect())
            .collect(),
    )
}

fn get_child_image_sets(cluster: &Cluster) -> Vec<BTreeSet<u32>> {
    cluster
        .child_clusters
        .iter()
        .map(|child| child.image_ids.iter().copied().collect())
        .collect()
}

fn options(branching: i32, image_overlap: i32) -> SceneClusteringOptions {
    SceneClusteringOptions {
        branching,
        image_overlap,
        ..SceneClusteringOptions::default()
    }
}

#[test]
fn scene_clustering_empty() {
    let options = SceneClusteringOptions {
        leaf_max_num_images: 2,
        ..options(2, 0)
    };
    let mut scene_clustering = SceneClustering::new(options).unwrap();
    assert!(scene_clustering.root_cluster().is_none());
    scene_clustering.partition(&[], &[]).unwrap();
    assert_eq!(scene_clustering.root_cluster().unwrap().image_ids.len(), 0);
    assert_eq!(
        scene_clustering
            .root_cluster()
            .unwrap()
            .child_clusters
            .len(),
        0
    );
    assert_eq!(scene_clustering.leaf_clusters().unwrap().len(), 1);
}

#[test]
fn scene_clustering_one_level() {
    let options = SceneClusteringOptions {
        leaf_max_num_images: 2,
        ..options(2, 0)
    };
    let mut scene_clustering = SceneClustering::new(options).unwrap();
    assert!(scene_clustering.root_cluster().is_none());
    scene_clustering.partition(&[(0, 1)], &[10]).unwrap();
    let root = scene_clustering.root_cluster().unwrap();
    assert_eq!(root.image_ids.len(), 2);
    assert_eq!(
        root.image_ids.iter().copied().collect::<BTreeSet<_>>(),
        BTreeSet::from([0, 1])
    );
    assert_eq!(root.child_clusters.len(), 0);
    let leaves = scene_clustering.leaf_clusters().unwrap();
    assert_eq!(leaves.len(), 1);
    assert!(std::ptr::eq(root, leaves[0]));
}

#[test]
fn scene_clustering_three_flat_clusters() {
    let graph = make_scene_graph_one_level();

    let options = SceneClusteringOptions {
        is_hierarchical: false,
        ..options(3, 0)
    };
    let mut scene_clustering = SceneClustering::new(options).unwrap();
    assert!(scene_clustering.root_cluster().is_none());
    scene_clustering
        .partition(&graph.image_pairs, &graph.num_inliers)
        .unwrap();

    let leaves = scene_clustering.leaf_clusters().unwrap();
    assert_eq!(leaves.len(), 3);
    assert_eq!(
        get_leaf_image_sets(&leaves),
        expected(&[&[0, 1], &[2, 3], &[4, 5]])
    );
}

#[test]
fn scene_clustering_three_flat_clusters_two_overlap() {
    let graph = make_scene_graph_one_level();

    let options = SceneClusteringOptions {
        is_hierarchical: false,
        ..options(3, 2)
    };
    let mut scene_clustering = SceneClustering::new(options).unwrap();
    assert!(scene_clustering.root_cluster().is_none());
    scene_clustering
        .partition(&graph.image_pairs, &graph.num_inliers)
        .unwrap();

    let leaves = scene_clustering.leaf_clusters().unwrap();
    assert_eq!(leaves.len(), 3);
    assert_eq!(
        get_leaf_image_sets(&leaves),
        expected(&[&[0, 1, 2, 5], &[1, 2, 3, 4], &[0, 3, 4, 5]])
    );
}

#[test]
fn scene_clustering_hierarchical_two_levels_no_overlap() {
    let graph = make_scene_graph_two_level();

    let options = SceneClusteringOptions {
        leaf_max_num_images: 2,
        ..options(2, 0)
    };
    let mut scene_clustering = SceneClustering::new(options).unwrap();
    assert!(scene_clustering.root_cluster().is_none());
    scene_clustering
        .partition(&graph.image_pairs, &graph.num_inliers)
        .unwrap();

    let leaves = scene_clustering.leaf_clusters().unwrap();
    assert_eq!(leaves.len(), 4);
    assert_eq!(
        get_leaf_image_sets(&leaves),
        expected(&[&[0, 1], &[2, 3], &[4, 5], &[6, 7]])
    );
}

#[test]
fn scene_clustering_hierarchical_two_levels_with_overlap() {
    let graph = make_scene_graph_two_level();

    let options = SceneClusteringOptions {
        leaf_max_num_images: 3,
        ..options(2, 2)
    };
    let mut scene_clustering = SceneClustering::new(options).unwrap();
    assert!(scene_clustering.root_cluster().is_none());
    scene_clustering
        .partition(&graph.image_pairs, &graph.num_inliers)
        .unwrap();

    let leaves = scene_clustering.leaf_clusters().unwrap();
    assert_eq!(leaves.len(), 4);
    assert_eq!(
        get_leaf_image_sets(&leaves),
        expected(&[
            &[0, 1, 2, 3, 4],
            &[0, 1, 2, 4, 7],
            &[0, 3, 4, 5, 6],
            &[0, 4, 5, 6, 7]
        ])
    );
}

#[test]
fn scene_clustering_hierarchical_three_levels_with_overlap() {
    let graph = make_scene_graph_three_level();

    let options = SceneClusteringOptions {
        leaf_max_num_images: 3,
        ..options(2, 1)
    };
    let mut scene_clustering = SceneClustering::new(options).unwrap();
    scene_clustering
        .partition(&graph.image_pairs, &graph.num_inliers)
        .unwrap();

    // Top-level overlap images are partitioned twice after they are inherited, exercising
    // connectivity beyond the single generation covered above.
    let leaves = scene_clustering.leaf_clusters().unwrap();
    assert_eq!(leaves.len(), 8);
    assert_eq!(
        get_leaf_image_sets(&leaves),
        expected(&[
            &[0, 1, 2],
            &[1, 2, 3, 4],
            &[3, 4, 5, 6],
            &[5, 6, 7, 8],
            &[7, 8, 9, 10],
            &[9, 10, 11, 12],
            &[11, 12, 13, 14],
            &[13, 14, 15]
        ])
    );
}

#[test]
fn scene_clustering_hierarchical_overlap_uses_all_connections() {
    let graph = make_scene_graph_aggregate_overlap();

    let options = SceneClusteringOptions {
        leaf_max_num_images: 4,
        ..options(2, 1)
    };
    let mut scene_clustering = SceneClustering::new(options).unwrap();
    scene_clustering
        .partition(&graph.image_pairs, &graph.num_inliers)
        .unwrap();

    let target_cluster = scene_clustering
        .root_cluster()
        .unwrap()
        .child_clusters
        .iter()
        .find(|child| {
            let image_ids: BTreeSet<u32> = child.image_ids.iter().copied().collect();
            (0..=5).all(|id| image_ids.contains(&id))
        })
        .expect("a child holding images 0-5");
    let child_image_sets = get_child_image_sets(target_cluster);
    assert_eq!(child_image_sets.len(), 2);

    // Image 6 does not simply follow its strongest connection to image 0. Its full
    // connectivity to images 3 and 4 participates in the graph cut. The exact partition is
    // implementation-dependent in METIS.
    assert!(child_image_sets
        .iter()
        .any(|image_ids| image_ids.contains(&3)
            && image_ids.contains(&4)
            && image_ids.contains(&6)));
}
