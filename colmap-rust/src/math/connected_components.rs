//! Port of COLMAP's `colmap/math/connected_components.h`: the connected components of an
//! undirected graph given as a node set and an edge list, built on [`super::union_find`].
//! Port of colmap-sharp's `Mathematics/ConnectedComponents.cs`. Tests:
//! `tests/math/connected_components.rs` (connected_components_test.cc 1:1).
//!
//! Tier A for the component *sets*. The order of the components, the order of the nodes
//! inside each, and which of two equally large components `find_largest_connected_component`
//! returns follow COLMAP's hash-container iteration there; here they follow the order of the
//! caller's `nodes` slice (COLMAP takes a `FlatHashSet`; repeated nodes in the slice count
//! once, at their first occurrence). See docs/CPP_DIVERGENCES.md, entry 42.

use super::union_find::UnionFind;
use std::collections::{HashMap, HashSet};
use std::hash::Hash;

/// `FindConnectedComponents(nodes, edges)`: all connected components. Each component lists
/// its nodes in `nodes` order, and components are ordered by their first node in that order.
/// A node repeated in `nodes` is reported once. Nodes that appear only in `edges` are not
/// reported.
pub fn find_connected_components<T: Clone + Eq + Hash>(
    nodes: &[T],
    edges: &[(T, T)],
) -> Vec<Vec<T>> {
    group_by_root(nodes, edges)
}

/// `FindLargestConnectedComponent(nodes, edges)`: the largest connected component. Of several
/// equally large components, returns the one whose first node comes first in `nodes`. Returns
/// an empty vector for an empty graph.
pub fn find_largest_connected_component<T: Clone + Eq + Hash>(
    nodes: &[T],
    edges: &[(T, T)],
) -> Vec<T> {
    // COLMAP starts from an empty component and keeps the first strictly larger one.
    let mut largest = Vec::new();
    for members in group_by_root(nodes, edges) {
        if members.len() > largest.len() {
            largest = members;
        }
    }
    largest
}

// Unites every edge, then buckets each node under its root. Components are kept in a Vec in
// first-seen order; the HashMap only maps a root to its bucket and is never iterated.
fn group_by_root<T: Clone + Eq + Hash>(nodes: &[T], edges: &[(T, T)]) -> Vec<Vec<T>> {
    let mut uf = UnionFind::new();
    uf.reserve(nodes.len());
    for (node1, node2) in edges {
        uf.union(node1, node2);
    }
    let mut bucket_of_root: HashMap<T, usize> = HashMap::new();
    let mut components: Vec<Vec<T>> = Vec::new();
    // COLMAP's nodes are a set: a node repeated in the slice counts once, at its first
    // occurrence (entry 42). Lookup only, never iterated.
    let mut seen: HashSet<&T> = HashSet::with_capacity(nodes.len());
    for node in nodes {
        if !seen.insert(node) {
            continue;
        }
        let root = uf.find(node);
        let bucket = *bucket_of_root.entry(root).or_insert_with(|| {
            components.push(Vec::new());
            components.len() - 1
        });
        components[bucket].push(node.clone());
    }
    components
}
