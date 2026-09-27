//! Port of COLMAP's `colmap/math/spanning_tree.h` and `spanning_tree.cc`: minimum and maximum
//! spanning trees of an undirected weighted graph, returned as parent pointers rooted at a
//! chosen node. Rotation averaging initializes from the maximum spanning tree of the pose
//! graph. Port of colmap-sharp's `Mathematics/SpanningTree.cs`. Tests:
//! `tests/math/spanning_tree.rs` (spanning_tree_test.cc 1:1).
//!
//! COLMAP runs `boost::kruskal_minimum_spanning_tree`. Boost is not ported
//! (docs/LICENSE_AUDIT.md); this is Kruskal's algorithm written from its published description
//! (J. B. Kruskal, "On the shortest spanning subtree of a graph and the traveling salesman
//! problem", Proc. AMS 7(1), 1956) with [`super::union_find`] as the disjoint-set forest.
//!
//! Tier A when the edge costs (after the max-weight negation below) are distinct: the
//! minimum spanning tree is then unique, so the parents match COLMAP exactly. Among equal
//! costs Boost's priority queue picks in heap order; here the earlier edge in the input wins.
//! See docs/CPP_DIVERGENCES.md, entry 43.

use super::union_find::UnionFind;
use super::utils::nan_last_cmp;
use crate::check;
use std::collections::VecDeque;

/// Port of `colmap::SpanningTree`: a rooted spanning tree as a parent map. For each node
/// index i, `parents[i]` is its parent; the root has `parents[root] == root`, and nodes
/// outside the root's component have -1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpanningTree {
    /// The root node, or -1 for an empty tree.
    pub root: i32,
    /// Parent of each node (see the struct docs).
    pub parents: Vec<i32>,
}

impl Default for SpanningTree {
    fn default() -> Self {
        Self {
            root: -1,
            parents: Vec::new(),
        }
    }
}

impl SpanningTree {
    /// `IsValid()`: the tree has a root and non-empty parents.
    pub fn is_valid(&self) -> bool {
        self.root >= 0 && !self.parents.is_empty()
    }

    /// `NumNodes()`.
    pub fn num_nodes(&self) -> usize {
        self.parents.len()
    }
}

/// `ComputeMaximumSpanningTree(num_nodes, edges, weights, root)`: the maximum spanning tree
/// of the graph with nodes `0..num_nodes` (higher weight preferred). If the graph is
/// disconnected, only the component containing `root` gets parents. COLMAP's default root is
/// 0.
pub fn compute_maximum_spanning_tree(
    num_nodes: i32,
    edges: &[(i32, i32)],
    weights: &[f32],
    root: i32,
) -> crate::Result<SpanningTree> {
    compute_spanning_tree(num_nodes, edges, weights, root, true)
}

/// `ComputeMinimumSpanningTree`: same interface as [`compute_maximum_spanning_tree`].
pub fn compute_minimum_spanning_tree(
    num_nodes: i32,
    edges: &[(i32, i32)],
    weights: &[f32],
    root: i32,
) -> crate::Result<SpanningTree> {
    compute_spanning_tree(num_nodes, edges, weights, root, false)
}

fn compute_spanning_tree(
    num_nodes: i32,
    edges: &[(i32, i32)],
    weights: &[f32],
    root: i32,
    maximize: bool,
) -> crate::Result<SpanningTree> {
    if num_nodes <= 0 {
        return Ok(SpanningTree::default());
    }
    // boost::add_edge on a vecS graph would silently grow the vertex set for an out-of-range
    // endpoint, and COLMAP then indexes past its adjacency list; fail with a check instead
    // (entry 43).
    check!(edges.len() == weights.len());
    check!(root >= 0 && root < num_nodes);

    // For the maximum spanning tree COLMAP maps w to (max_weight - w) in float, with
    // max_weight starting at 0, and finds the minimum. The same float arithmetic keeps the
    // same ties. std::max(a, b) is (a < b) ? b : a, so a NaN weight never becomes max_weight
    // (and a NaN weight gives a NaN cost).
    let mut max_weight = 0.0f32;
    if maximize {
        for &w in weights {
            if max_weight < w {
                max_weight = w;
            }
        }
    }
    let n = num_nodes as usize;
    let mut costs = Vec::with_capacity(edges.len());
    for (i, &(node1, node2)) in edges.iter().enumerate() {
        check!(node1 >= 0 && node1 < num_nodes && node2 >= 0 && node2 < num_nodes);
        costs.push(if maximize {
            max_weight - weights[i]
        } else {
            weights[i]
        });
    }

    // Kruskal: take edges by increasing cost, keeping each that joins two components. The
    // stable sort keeps input order among equal costs, and NaN costs sort after every number
    // (entry 43).
    let mut order: Vec<usize> = (0..edges.len()).collect();
    order.sort_by(|&a, &b| nan_last_cmp(&costs[a], &costs[b]));

    let mut components = UnionFind::new();
    components.reserve(n);
    let mut adjacency: Vec<Vec<i32>> = vec![Vec::new(); n];
    for edge_idx in order {
        let (source, target) = edges[edge_idx];
        if components.find(&source) != components.find(&target) {
            components.union(&source, &target);
            adjacency[source as usize].push(target);
            adjacency[target as usize].push(source);
        }
    }

    Ok(SpanningTree {
        root,
        parents: build_parents_from_adjacency_list(&adjacency, root),
    })
}

// Breadth-first search from the root turns the undirected tree into parent pointers. Nodes
// the search never reaches (other components) keep -1.
fn build_parents_from_adjacency_list(adjacency: &[Vec<i32>], root: i32) -> Vec<i32> {
    let mut parents = vec![-1; adjacency.len()];
    parents[root as usize] = root;
    let mut visited = vec![false; adjacency.len()];
    visited[root as usize] = true;
    let mut queue = VecDeque::new();
    queue.push_back(root);
    while let Some(current) = queue.pop_front() {
        for &neighbor in &adjacency[current as usize] {
            if !visited[neighbor as usize] {
                visited[neighbor as usize] = true;
                parents[neighbor as usize] = current;
                queue.push_back(neighbor);
            }
        }
    }
    parents
}
