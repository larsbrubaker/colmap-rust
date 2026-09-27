// Port of COLMAP's src/colmap/math/spanning_tree_test.cc (1:1, same test names in
// snake_case). COLMAP's default `root = 0` is passed explicitly.
#![allow(clippy::float_cmp)]

use colmap_rust::math::spanning_tree::{
    compute_maximum_spanning_tree, compute_minimum_spanning_tree, SpanningTree,
};

fn compute_tree_weight(tree: &SpanningTree, edges: &[(i32, i32)], weights: &[f32]) -> f32 {
    let mut total = 0.0f32;
    for (i, &(u, v)) in edges.iter().enumerate() {
        // Check if this edge is in the tree (either direction).
        if tree.parents[u as usize] == v || tree.parents[v as usize] == u {
            total += weights[i];
        }
    }
    total
}

#[test]
fn spanning_tree_nominal() {
    // Triangle: edges with weights 1, 2, 3.
    // Max spanning tree uses edges 2+3=5, min uses 1+2=3.
    let edges = [(0, 1), (1, 2), (0, 2)];
    let weights = [1.0f32, 2.0, 3.0];

    let max_tree = compute_maximum_spanning_tree(3, &edges, &weights, 0).unwrap();
    let min_tree = compute_minimum_spanning_tree(3, &edges, &weights, 0).unwrap();

    assert_eq!(compute_tree_weight(&max_tree, &edges, &weights), 5.0f32);
    assert_eq!(compute_tree_weight(&min_tree, &edges, &weights), 3.0f32);
}

#[test]
fn spanning_tree_disconnected_graph() {
    // Two components: {0,1} and {2,3}. Only component containing root is
    // included.
    let edges = [(0, 1), (2, 3)];
    let weights = [1.0f32, 2.0];

    // Root at 0: includes {0,1}, excludes {2,3}.
    let tree0 = compute_maximum_spanning_tree(4, &edges, &weights, 0).unwrap();
    assert_eq!(tree0.root, 0);
    assert_eq!(tree0.parents[0], 0);
    assert_eq!(tree0.parents[1], 0);
    assert_eq!(tree0.parents[2], -1);
    assert_eq!(tree0.parents[3], -1);

    // Root at 2: includes {2,3}, excludes {0,1}.
    let tree2 = compute_maximum_spanning_tree(4, &edges, &weights, 2).unwrap();
    assert_eq!(tree2.root, 2);
    assert_eq!(tree2.parents[0], -1);
    assert_eq!(tree2.parents[1], -1);
    assert_eq!(tree2.parents[2], 2);
    assert_eq!(tree2.parents[3], 2);
}

#[test]
fn spanning_tree_empty_graph() {
    let tree = compute_maximum_spanning_tree(0, &[], &[], 0).unwrap();
    assert!(!tree.is_valid());
}

// Rust-only: among equal costs the earlier input edge wins (docs/CPP_DIVERGENCES.md,
// entry 43), so a tied triangle always keeps edges 0 and 1.
#[test]
fn rust_only_spanning_tree_ties_take_earlier_edge() {
    let edges = [(0, 1), (1, 2), (0, 2)];
    let weights = [2.0f32, 2.0, 2.0];
    for tree in [
        compute_maximum_spanning_tree(3, &edges, &weights, 0).unwrap(),
        compute_minimum_spanning_tree(3, &edges, &weights, 0).unwrap(),
    ] {
        assert_eq!(tree.parents, vec![0, 0, 1]);
    }
}

// Rust-only: out-of-range input fails a check instead of indexing out of bounds (entry 43).
#[test]
fn rust_only_spanning_tree_rejects_out_of_range_input() {
    assert!(compute_maximum_spanning_tree(2, &[(0, 2)], &[1.0], 0).is_err());
    assert!(compute_maximum_spanning_tree(2, &[(0, 1)], &[], 0).is_err());
    assert!(compute_maximum_spanning_tree(2, &[(0, 1)], &[1.0], 2).is_err());
}

// Rust-only: NaN weights never panic; a NaN cost sorts after every number, so NaN edges are
// taken only to connect what the numeric edges cannot (docs/CPP_DIVERGENCES.md, entry 43).
#[test]
fn rust_only_spanning_tree_with_nan_weights() {
    let n = 200;
    let mut edges = Vec::new();
    let mut weights = Vec::new();
    for i in 0..n - 1 {
        edges.push((i, i + 1));
        weights.push(if i % 5 == 0 { f32::NAN } else { (i % 7) as f32 });
        edges.push((i, (i * 13 + 7) % n));
        weights.push(if i % 5 == 1 {
            f32::NAN
        } else {
            (i % 11) as f32
        });
    }
    for tree in [
        compute_maximum_spanning_tree(n, &edges, &weights, 0).unwrap(),
        compute_minimum_spanning_tree(n, &edges, &weights, 0).unwrap(),
    ] {
        assert!(tree.parents.iter().all(|&p| p >= 0));
    }
    // Triangle: the NaN edge loses to the two numeric ones either way.
    let edges = [(0, 1), (1, 2), (0, 2)];
    let weights = [f32::NAN, 1.0, 2.0];
    for tree in [
        compute_maximum_spanning_tree(3, &edges, &weights, 0).unwrap(),
        compute_minimum_spanning_tree(3, &edges, &weights, 0).unwrap(),
    ] {
        assert_eq!(tree.parents, vec![0, 2, 0]);
    }
}
