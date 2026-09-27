// Port of COLMAP's src/colmap/math/connected_components_test.cc (1:1, same test names in
// snake_case). COLMAP's `FlatHashSet` of nodes becomes a slice; gmock's
// `UnorderedElementsAre` becomes a sorted comparison.

use colmap_rust::math::connected_components::{
    find_connected_components, find_largest_connected_component,
};

fn sorted<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort();
    v
}

#[test]
fn find_connected_components_empty() {
    let nodes: Vec<i32> = vec![];
    let edges: Vec<(i32, i32)> = vec![];
    let components = find_connected_components(&nodes, &edges);
    assert!(components.is_empty());
}

#[test]
fn find_connected_components_single_node() {
    let nodes = vec![1];
    let edges: Vec<(i32, i32)> = vec![];
    let components = find_connected_components(&nodes, &edges);
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].len(), 1);
    assert_eq!(components[0][0], 1);
}

#[test]
fn find_connected_components_two_connected_nodes() {
    let nodes = vec![1, 2];
    let edges = vec![(1, 2)];
    let components = find_connected_components(&nodes, &edges);
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].len(), 2);
}

#[test]
fn find_connected_components_two_disconnected_nodes() {
    let nodes = vec![1, 2];
    let edges: Vec<(i32, i32)> = vec![];
    let components = find_connected_components(&nodes, &edges);
    assert_eq!(components.len(), 2);
    assert_eq!(components[0].len(), 1);
    assert_eq!(components[1].len(), 1);
}

#[test]
fn find_connected_components_three_components() {
    let nodes = vec![1, 2, 3, 4, 5, 6];
    let edges = vec![(1, 2), (3, 4), (5, 6)];
    let components = find_connected_components(&nodes, &edges);
    assert_eq!(components.len(), 3);
    for comp in &components {
        assert_eq!(comp.len(), 2);
    }
}

#[test]
fn find_connected_components_chain() {
    let nodes = vec![1, 2, 3, 4, 5];
    let edges = vec![(1, 2), (2, 3), (3, 4), (4, 5)];
    let components = find_connected_components(&nodes, &edges);
    assert_eq!(components.len(), 1);
    assert_eq!(components[0].len(), 5);
}

#[test]
fn find_largest_connected_component_empty() {
    let nodes: Vec<i32> = vec![];
    let edges: Vec<(i32, i32)> = vec![];
    let largest = find_largest_connected_component(&nodes, &edges);
    assert!(largest.is_empty());
}

#[test]
fn find_largest_connected_component_single_node() {
    let nodes = vec![42];
    let edges: Vec<(i32, i32)> = vec![];
    let largest = find_largest_connected_component(&nodes, &edges);
    assert_eq!(sorted(largest), vec![42]);
}

#[test]
fn find_largest_connected_component_all_connected() {
    let nodes = vec![1, 2, 3, 4];
    let edges = vec![(1, 2), (2, 3), (3, 4)];
    let largest = find_largest_connected_component(&nodes, &edges);
    assert_eq!(sorted(largest), vec![1, 2, 3, 4]);
}

#[test]
fn find_largest_connected_component_two_components_different_sizes() {
    // Component 1: {1, 2, 3} (size 3)
    // Component 2: {10, 20} (size 2)
    let nodes = vec![1, 2, 3, 10, 20];
    let edges = vec![(1, 2), (2, 3), (10, 20)];
    let largest = find_largest_connected_component(&nodes, &edges);
    assert_eq!(sorted(largest), vec![1, 2, 3]);
}

#[test]
fn find_largest_connected_component_many_small_onelarger() {
    // 5 isolated nodes + 1 component of 3
    let nodes = vec![1, 2, 3, 4, 5, 100, 200, 300];
    let edges = vec![(100, 200), (200, 300)];
    let largest = find_largest_connected_component(&nodes, &edges);
    assert_eq!(sorted(largest), vec![100, 200, 300]);
}

#[test]
fn find_largest_connected_component_string_type() {
    let s = |x: &str| x.to_string();
    let nodes = vec![s("a"), s("b"), s("c"), s("x"), s("y")];
    let edges = vec![(s("a"), s("b")), (s("b"), s("c"))];
    let largest = find_largest_connected_component(&nodes, &edges);
    assert_eq!(sorted(largest), vec![s("a"), s("b"), s("c")]);
}

// Rust-only: components come out by first node in `nodes` order, members in `nodes` order,
// and ties for the largest go to the first (docs/CPP_DIVERGENCES.md, entry 42).
#[test]
fn rust_only_connected_components_follow_node_order() {
    let nodes = vec![4, 1, 3, 2];
    let edges = vec![(1, 2), (3, 4)];
    assert_eq!(
        find_connected_components(&nodes, &edges),
        vec![vec![4, 3], vec![1, 2]]
    );
    assert_eq!(find_largest_connected_component(&nodes, &edges), vec![4, 3]);
}

// Rust-only: COLMAP takes the nodes as a set; a node repeated in the slice counts once, at
// its first occurrence (docs/CPP_DIVERGENCES.md, entry 42).
#[test]
fn rust_only_connected_components_dedup_nodes() {
    assert_eq!(
        find_connected_components(&[1, 1, 2], &[]),
        vec![vec![1], vec![2]]
    );
    assert_eq!(
        find_connected_components(&[2, 1, 2, 1], &[(1, 2)]),
        vec![vec![2, 1]]
    );
    assert_eq!(find_largest_connected_component(&[1, 1, 2], &[]), vec![1]);
}
