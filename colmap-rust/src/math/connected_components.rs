//! Port of COLMAP's `colmap/math/connected_components.h`: the connected components of an
//! undirected graph given as a node set and an edge list, built on [`super::union_find`].
//! Port of colmap-sharp's `Mathematics/ConnectedComponents.cs`. Tests:
//! `tests/math/connected_components.rs` (connected_components_test.cc 1:1).
//!
//! Tier A for the component *sets*, and for every root `find` returns for the same call
//! sequence.
//!
//! Deliberate divergence in iteration order, here and in [`UnionFind::parents`] (which
//! iterates in insertion order): `find_connected_components` returns components ordered by
//! their first node in the caller's `nodes` slice, each listing its nodes in that order, and
//! `find_largest_connected_component` breaks ties between equally large components the same
//! way. COLMAP's orders come from `boost::unordered_node_map` / `unordered_flat_set`
//! (`colmap/util/hash_containers.h`), and it takes the nodes as a `FlatHashSet` where the port
//! takes a slice; to keep the set semantics a node repeated in the slice is reported once, at
//! its first occurrence (`[1, 1, 2]` with no edges gives `[[1], [2]]`). Reproducing COLMAP's
//! orders would mean porting Boost.Unordered's bucket layout and `boost::hash` mixing for every
//! key type, and the order is not part of COLMAP's contract: `connected_components_test.cc`
//! compares with `UnorderedElementsAre`, and `union_find_test.cc` never inspects it. Insertion
//! order is deterministic and the same on every platform (a std `HashMap` is used only for
//! lookups, never iterated). `rust_only_union_find_parents_in_insertion_order`,
//! `rust_only_connected_components_follow_node_order` and
//! `rust_only_connected_components_dedup_nodes` pin the orders and the repeated-node handling.
//! Callers that need a reproducible component order pass nodes in a deterministic order (for
//! example sorted ids). Same as colmap-sharp entry 2.

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
    // occurrence (see the module header). Lookup only, never iterated.
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
