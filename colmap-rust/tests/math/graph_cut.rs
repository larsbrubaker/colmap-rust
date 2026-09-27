// Port of COLMAP's src/colmap/math/graph_cut_test.cc (1:1, same test names in snake_case,
// same values). `ComputeMinGraphCutStoerWagner`'s out-parameters become the returned
// `(cut_weight, cut_labels)`; `MinSTGraphCut<int, int>` is `MinSTGraphCut::<i32>`; the
// C++'s returned `NodeHashMap` is a `BTreeMap` (docs/CPP_DIVERGENCES.md, entry 46). The
// Rust-only cross-check, scaling and partition tests are in rust_only_graph_cut.rs.

use colmap_rust::math::graph_cut::{
    compute_min_graph_cut_stoer_wagner, compute_normalized_min_graph_cut,
};
use colmap_rust::math::graph_cut_min_st::MinSTGraphCut;
use std::collections::BTreeMap;

const EDGES: [(i32, i32); 16] = [
    (3, 4),
    (3, 6),
    (3, 5),
    (0, 4),
    (0, 1),
    (0, 6),
    (0, 7),
    (0, 5),
    (0, 2),
    (4, 1),
    (1, 6),
    (1, 5),
    (6, 7),
    (7, 5),
    (5, 2),
    (3, 4),
];
const WEIGHTS: [i32; 16] = [0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1, 1, 4];

fn duplicate_edge_graph() -> (Vec<(i32, i32)>, Vec<i32>) {
    let mut edges = EDGES.to_vec();
    edges.push((3, 4));
    let mut weights = WEIGHTS.to_vec();
    weights.push(4);
    (edges, weights)
}

fn missing_vertex_graph() -> (Vec<(i32, i32)>, Vec<i32>) {
    let edges = vec![
        (3, 4),
        (3, 6),
        (3, 5),
        (0, 1),
        (0, 6),
        (0, 7),
        (0, 5),
        (0, 2),
        (4, 1),
        (1, 6),
        (1, 5),
        (6, 7),
        (7, 5),
        (5, 2),
    ];
    let weights = vec![0, 3, 1, 3, 1, 2, 6, 1, 8, 1, 1, 80, 2, 1];
    (edges, weights)
}

const DISCONNECTED_EDGES: [(i32, i32); 3] = [(0, 1), (1, 2), (3, 4)];
const DISCONNECTED_WEIGHTS: [i32; 3] = [1, 3, 1];

// EXPECT_GE(label, 0) holds by type (u8); EXPECT_LT(label, 2) is checked.
fn expect_binary_labels(cut_labels: &[u8]) {
    for &label in cut_labels {
        assert!(label < 2);
    }
}

fn expect_two_nonempty_parts(cut_labels: &BTreeMap<i32, i32>) {
    let mut num_labels = [0usize; 2];
    for &label in cut_labels.values() {
        assert!(label >= 0);
        assert!(label < 2);
        num_labels[label as usize] += 1;
    }
    assert!(num_labels[0] > 0);
    assert!(num_labels[1] > 0);
}

#[test]
fn graph_cut_compute_min_graph_cut_stoer_wagner() {
    let (cut_weight, cut_labels) = compute_min_graph_cut_stoer_wagner(&EDGES, &WEIGHTS).unwrap();
    assert_eq!(cut_weight, 7);
    assert_eq!(cut_labels.len(), 8);
    expect_binary_labels(&cut_labels);
}

#[test]
fn graph_cut_compute_min_graph_cut_stoer_wagner_duplicate_edge() {
    let (edges, weights) = duplicate_edge_graph();
    let (cut_weight, cut_labels) = compute_min_graph_cut_stoer_wagner(&edges, &weights).unwrap();
    assert_eq!(cut_weight, 7);
    assert_eq!(cut_labels.len(), 8);
    expect_binary_labels(&cut_labels);
}

#[test]
fn graph_cut_compute_min_graph_cut_stoer_wagner_missing_vertex() {
    let (edges, weights) = missing_vertex_graph();
    let (cut_weight, cut_labels) = compute_min_graph_cut_stoer_wagner(&edges, &weights).unwrap();
    assert_eq!(cut_weight, 2);
    assert_eq!(cut_labels.len(), 8);
    expect_binary_labels(&cut_labels);
}

#[test]
fn graph_cut_compute_min_graph_cut_stoer_wagner_disconnected() {
    let (cut_weight, cut_labels) =
        compute_min_graph_cut_stoer_wagner(&DISCONNECTED_EDGES, &DISCONNECTED_WEIGHTS).unwrap();
    assert_eq!(cut_weight, 0);
    assert_eq!(cut_labels.len(), 5);
    expect_binary_labels(&cut_labels);
}

#[test]
fn graph_cut_compute_normalized_min_graph_cut() {
    let cut_labels = compute_normalized_min_graph_cut(&EDGES, &WEIGHTS, 2).unwrap();
    assert_eq!(cut_labels.len(), 8);
    expect_two_nonempty_parts(&cut_labels);
}

#[test]
fn graph_cut_compute_normalized_min_graph_cut_duplicate_edge() {
    let (edges, weights) = duplicate_edge_graph();
    let cut_labels = compute_normalized_min_graph_cut(&edges, &weights, 2).unwrap();
    assert_eq!(cut_labels.len(), 8);
    expect_two_nonempty_parts(&cut_labels);
}

#[test]
fn graph_cut_compute_normalized_min_graph_cut_missing_vertex() {
    let (edges, weights) = missing_vertex_graph();
    let cut_labels = compute_normalized_min_graph_cut(&edges, &weights, 2).unwrap();
    assert_eq!(cut_labels.len(), 8);
    expect_two_nonempty_parts(&cut_labels);
}

#[test]
fn graph_cut_compute_normalized_min_graph_cut_disconnected() {
    let cut_labels =
        compute_normalized_min_graph_cut(&DISCONNECTED_EDGES, &DISCONNECTED_WEIGHTS, 2).unwrap();
    assert_eq!(cut_labels.len(), 5);
    assert_eq!(cut_labels[&0], cut_labels[&1]);
    assert_eq!(cut_labels[&1], cut_labels[&2]);
    assert_ne!(cut_labels[&2], cut_labels[&3]);
    assert_eq!(cut_labels[&3], cut_labels[&4]);
}

#[test]
fn graph_cut_min_st_graph_cut1() {
    let mut graph = MinSTGraphCut::<i32>::new(2);
    assert_eq!(graph.num_nodes(), 2);
    assert_eq!(graph.num_edges(), 0);
    graph.add_node(0, 5, 1).unwrap();
    graph.add_node(1, 2, 6).unwrap();
    graph.add_edge(0, 1, 3, 4).unwrap();
    assert_eq!(graph.num_edges(), 10);
    assert_eq!(graph.compute(), 6);
    assert!(graph.is_connected_to_source(0).unwrap());
    assert!(graph.is_connected_to_sink(1).unwrap());
}

#[test]
fn graph_cut_min_st_graph_cut2() {
    let mut graph = MinSTGraphCut::<i32>::new(2);
    graph.add_node(0, 1, 5).unwrap();
    graph.add_node(1, 2, 6).unwrap();
    graph.add_edge(0, 1, 3, 4).unwrap();
    assert_eq!(graph.num_edges(), 10);
    assert_eq!(graph.compute(), 3);
    assert!(graph.is_connected_to_sink(0).unwrap());
    assert!(graph.is_connected_to_sink(1).unwrap());
}

#[test]
fn graph_cut_min_st_graph_cut3() {
    let mut graph = MinSTGraphCut::<i32>::new(3);
    graph.add_node(0, 6, 4).unwrap();
    graph.add_node(2, 3, 6).unwrap();
    graph.add_edge(0, 1, 2, 4).unwrap();
    graph.add_edge(1, 2, 3, 5).unwrap();
    assert_eq!(graph.num_edges(), 12);
    assert_eq!(graph.compute(), 9);
    assert!(graph.is_connected_to_source(0).unwrap());
    assert!(graph.is_connected_to_sink(1).unwrap());
    assert!(graph.is_connected_to_sink(2).unwrap());
}
