//! Port of the `MinSTGraphCut` class template in COLMAP's `colmap/math/graph_cut.h`: the
//! minimum S-T cut of a directed graph by max-flow. Port of colmap-sharp's
//! `Mathematics/MinSTGraphCut.cs` (our own MIT code). Its neighbor [`super::graph_cut`] holds
//! the free functions of the same header. Delaunay meshing (`mvs/delaunay_meshing`) labels
//! cells inside/outside with it. Tests: `tests/math/graph_cut.rs`.
//!
//! COLMAP calls `boost::boykov_kolmogorov_max_flow`. Neither Boost's implementation nor
//! Kolmogorov's own maxflow library (GPL / research-only) was read or transcribed; colmap-sharp
//! wrote the algorithm from its paper: Y. Boykov and V. Kolmogorov, "An Experimental
//! Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision", IEEE PAMI
//! 26(9), 2004 — search trees S and T grown from the terminals, augmentation along the path
//! where they meet, and adoption of orphans, with the paper's timestamp/distance heuristic for
//! choosing parents.
//!
//! Terminal links are not stored as edges. As the paper's implementation section describes,
//! each node keeps one residual terminal capacity (positive: to the source, negative: to the
//! sink). The flow s -> v -> t that a node can carry on its own is pushed up front, and every
//! node left with terminal capacity starts in its tree and active. This matters for scale:
//! Delaunay meshing gives nearly every cell a terminal link, and treating those as out-edges
//! of a terminal made the terminal rescan all of them after each augmentation (quadratic).
//! Growth also keeps a current arc per active node, so a node that found a path resumes its
//! scan where it stopped instead of rescanning arcs it already classified.
//!
//! Tier A for integer capacities: the flow value is the unique max-flow, and the labels are
//! too, because at termination the S tree is exactly the set of nodes reachable from the
//! source in the residual graph and the T tree the set that can reach the sink; both sets are
//! the same for every maximum flow. Boost colors free nodes (in neither tree) gray, which
//! COLMAP reports as connected to the source; so does this. With float capacities it is
//! Tier B: see docs/CPP_DIVERGENCES.md, entry 45.
//!
//! Translation notes:
//! - The C++ template parameter `node_t` becomes `usize` (node indices address arrays here);
//!   `value_t` becomes the [`FlowValue`] trait (i32, i64, f32, f64).
//! - Non-terminal edges are stored in pairs, so the reverse of edge e is `e ^ 1`; Boost's
//!   explicit reverse edge map is not needed.
//! - COLMAP's index checks admit the two terminal indices (num_nodes and num_nodes + 1), so an
//!   edge may touch a terminal. Such an edge becomes terminal capacity (source -> v, v ->
//!   sink) or direct flow (source -> sink); edges into the source or out of the sink can carry
//!   no s-t flow and are only counted.

use crate::util::check::CheckOpValue;
use crate::{check, check_ge, check_le, check_lt};
use std::collections::VecDeque;
/// A capacity/flow type for [`MinSTGraphCut`] (COLMAP's `value_t`).
///
/// Flow arithmetic goes through [`Self::add`], [`Self::sub`] and [`Self::neg`]. For the
/// integer types these wrap on overflow: COLMAP's Boost max-flow sums `value_t` unchecked
/// (signed overflow is undefined behavior there) and colmap-sharp's C# wraps, so wrapping is
/// the defined stand-in, and a debug build never panics on huge capacities. The float types
/// use the plain IEEE operations.
///
/// NaN capacities are rejected by `add_node`/`add_edge` (the `>= 0` check fails, as COLMAP's
/// `THROW_CHECK_GE` does). Infinite capacities are accepted, as in COLMAP, and can produce a
/// NaN flow (`inf - inf`), the same as Boost.
pub trait FlowValue: Copy + PartialOrd + CheckOpValue {
    /// The additive identity.
    const ZERO: Self;

    /// `self + b`, wrapping for integers.
    fn add(self, b: Self) -> Self;

    /// `self - b`, wrapping for integers.
    fn sub(self, b: Self) -> Self;

    /// `-self`, wrapping for integers.
    fn neg(self) -> Self;

    /// The smaller of the two (`b` when `b < a`, else `a`).
    #[inline]
    fn min_value(self, b: Self) -> Self {
        if b < self {
            b
        } else {
            self
        }
    }
}

macro_rules! impl_flow_value_int {
    ($($t:ty),*) => {$(
        impl FlowValue for $t {
            const ZERO: Self = 0;
            #[inline]
            fn add(self, b: Self) -> Self { self.wrapping_add(b) }
            #[inline]
            fn sub(self, b: Self) -> Self { self.wrapping_sub(b) }
            #[inline]
            fn neg(self) -> Self { self.wrapping_neg() }
        }
    )*};
}

macro_rules! impl_flow_value_float {
    ($($t:ty),*) => {$(
        impl FlowValue for $t {
            const ZERO: Self = 0.0;
            #[inline]
            fn add(self, b: Self) -> Self { self + b }
            #[inline]
            fn sub(self, b: Self) -> Self { self - b }
            #[inline]
            fn neg(self) -> Self { -self }
        }
    )*};
}

impl_flow_value_int!(i32, i64);
impl_flow_value_float!(f32, f64);

const FREE_NODE: u8 = 0;
const SOURCE_TREE: u8 = 1;
const SINK_TREE: u8 = 2;

// Parent-edge markers for nodes that have no parent edge (real edges are >= 0).
const TERMINAL_PARENT: isize = -1;
const ORPHAN_PARENT: isize = -2;
const NO_PARENT: isize = -3;

/// Port of `colmap::MinSTGraphCut`: the minimum cut of a directed S-T graph using the
/// Boykov-Kolmogorov max-flow min-cut algorithm.
#[derive(Clone, Debug)]
pub struct MinSTGraphCut<V: FlowValue> {
    num_nodes: usize,
    // Non-terminal edge e runs from tail[e] to head[e]; its reverse is e ^ 1.
    tail: Vec<usize>,
    head: Vec<usize>,
    capacity: Vec<V>,
    // Terminal link capacities per node, and the capacity of edges straight from source to
    // sink.
    source_capacity: Vec<V>,
    sink_capacity: Vec<V>,
    direct_flow_capacity: V,
    // Edges as COLMAP's Boost graph counts them, terminal edges and their reverses included.
    num_edges: usize,
    // Filled by compute: tree membership per node (FREE_NODE, SOURCE_TREE, SINK_TREE).
    trees: Vec<u8>,
    last_compute_work: u64,
}

impl<V: FlowValue> MinSTGraphCut<V> {
    /// Creates a graph with `num_nodes` nodes plus the two terminals.
    pub fn new(num_nodes: usize) -> Self {
        Self {
            num_nodes,
            tail: Vec::new(),
            head: Vec::new(),
            capacity: Vec::new(),
            source_capacity: vec![V::ZERO; num_nodes],
            sink_capacity: vec![V::ZERO; num_nodes],
            direct_flow_capacity: V::ZERO,
            num_edges: 0,
            trees: Vec::new(),
            last_compute_work: 0,
        }
    }

    /// `NumNodes()`: the number of nodes, not counting the terminals.
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// `NumEdges()`: the number of directed edges, each reverse edge counted.
    pub fn num_edges(&self) -> usize {
        self.num_edges
    }

    /// Diagnostic for tests: the steps the last [`Self::compute`] took, counting every arc
    /// examined while growing trees and adopting orphans and every parent link followed while
    /// augmenting or measuring distances. Unlike wall-clock time it does not depend on
    /// machine load, so a scaling test can bound it to catch quadratic behavior.
    pub fn last_compute_work(&self) -> u64 {
        self.last_compute_work
    }

    fn source_node(&self) -> usize {
        self.num_nodes
    }

    fn sink_node(&self) -> usize {
        self.num_nodes + 1
    }

    /// `AddNode`: connects node `node_idx` to the source and sink terminals. A zero capacity
    /// adds no edge.
    pub fn add_node(
        &mut self,
        node_idx: usize,
        source_capacity: V,
        sink_capacity: V,
    ) -> crate::Result<()> {
        check_le!(node_idx, self.num_nodes + 2);
        check_ge!(source_capacity, V::ZERO);
        check_ge!(sink_capacity, V::ZERO);

        if source_capacity > V::ZERO {
            self.add_edge_pair(self.source_node(), node_idx, source_capacity, V::ZERO)?;
        }
        if sink_capacity > V::ZERO {
            self.add_edge_pair(node_idx, self.sink_node(), sink_capacity, V::ZERO)?;
        }
        Ok(())
    }

    /// `AddEdge`: adds an edge from `node_idx1` to `node_idx2` with `capacity`, and the
    /// reverse edge with `reverse_capacity`.
    pub fn add_edge(
        &mut self,
        node_idx1: usize,
        node_idx2: usize,
        capacity: V,
        reverse_capacity: V,
    ) -> crate::Result<()> {
        check_le!(node_idx1, self.num_nodes + 2);
        check_le!(node_idx2, self.num_nodes + 2);
        check_ge!(capacity, V::ZERO);
        check_ge!(reverse_capacity, V::ZERO);
        self.add_edge_pair(node_idx1, node_idx2, capacity, reverse_capacity)
    }

    /// `Compute`: computes the min-cut with the max-flow algorithm and returns the flow.
    pub fn compute(&mut self) -> V {
        let mut solver = Solver::new(self);
        let flow = solver.run();
        self.last_compute_work = solver.work;
        self.trees = solver.tree;
        flow
    }

    /// `IsConnectedToSource`: after [`Self::compute`], whether the node is on the source side
    /// of the cut. Nodes in neither search tree count as source side, as in COLMAP. Fails a
    /// check (COLMAP's `colors_.at` throws) for an index out of range or before `compute`.
    pub fn is_connected_to_source(&self, node_idx: usize) -> crate::Result<bool> {
        Ok(self.tree_of(node_idx)? != SINK_TREE)
    }

    /// `IsConnectedToSink`: after [`Self::compute`], whether the node is on the sink side of
    /// the cut.
    pub fn is_connected_to_sink(&self, node_idx: usize) -> crate::Result<bool> {
        Ok(self.tree_of(node_idx)? == SINK_TREE)
    }

    fn tree_of(&self, node_idx: usize) -> crate::Result<u8> {
        if node_idx == self.source_node() {
            return Ok(SOURCE_TREE);
        }
        if node_idx == self.sink_node() {
            return Ok(SINK_TREE);
        }
        check_lt!(node_idx, self.trees.len());
        Ok(self.trees[node_idx])
    }

    fn add_edge_pair(
        &mut self,
        from: usize,
        to: usize,
        forward: V,
        backward: V,
    ) -> crate::Result<()> {
        // The check_le calls admit the index one past the last terminal, as COLMAP's
        // THROW_CHECK_LE does; Boost would grow the graph there. Refuse it rather than
        // corrupt.
        check!(from < self.num_nodes + 2 && to < self.num_nodes + 2);
        self.num_edges += 2;

        if from < self.num_nodes && to < self.num_nodes && from != to {
            self.tail.push(from);
            self.head.push(to);
            self.capacity.push(forward);
            self.tail.push(to);
            self.head.push(from);
            self.capacity.push(backward);
            return Ok(());
        }

        self.add_terminal_arc(from, to, forward);
        self.add_terminal_arc(to, from, backward);
        Ok(())
    }

    /// One direction of an edge that touches a terminal, or a self-loop (which carries
    /// nothing).
    fn add_terminal_arc(&mut self, from: usize, to: usize, arc_capacity: V) {
        let (source, sink, n) = (self.source_node(), self.sink_node(), self.num_nodes);
        // Wrapping sums for integers: see FlowValue.
        if from == source && to == sink {
            self.direct_flow_capacity = self.direct_flow_capacity.add(arc_capacity);
        } else if from == source && to < n {
            self.source_capacity[to] = self.source_capacity[to].add(arc_capacity);
        } else if from < n && to == sink {
            self.sink_capacity[from] = self.sink_capacity[from].add(arc_capacity);
        }
    }
}

/// One max-flow run over a snapshot of the graph, so `compute` can be called again.
struct Solver<V: FlowValue> {
    head: Vec<usize>,
    residual: Vec<V>,
    // Node p's arcs are arcs[arc_start[p] .. arc_start[p + 1]].
    arc_start: Vec<usize>,
    arcs: Vec<usize>,
    // Residual terminal capacity: > 0 source, < 0 sink.
    terminal: Vec<V>,
    tree: Vec<u8>,
    // Edge from the node to its parent in its tree, or one of the *_PARENT markers.
    parent_edge: Vec<isize>,
    current_arc: Vec<usize>,
    timestamp: Vec<u32>,
    distance: Vec<usize>,
    is_active: Vec<bool>,
    active: VecDeque<usize>,
    orphans: VecDeque<usize>,
    flow: V,
    time: u32,
    // Steps taken, for MinSTGraphCut::last_compute_work.
    work: u64,
}

impl<V: FlowValue> Solver<V> {
    fn new(graph: &MinSTGraphCut<V>) -> Self {
        let n = graph.num_nodes;
        let head = graph.head.clone();

        // Compressed adjacency, each node's arcs in insertion order.
        let mut arc_start = vec![0usize; n + 1];
        for &from in &graph.tail {
            arc_start[from + 1] += 1;
        }
        for v in 0..n {
            arc_start[v + 1] += arc_start[v];
        }
        let mut arcs = vec![0usize; head.len()];
        let mut fill = arc_start[..n].to_vec();
        for (e, &from) in graph.tail.iter().enumerate() {
            arcs[fill[from]] = e;
            fill[from] += 1;
        }

        let mut solver = Self {
            head,
            residual: graph.capacity.clone(),
            arc_start,
            arcs,
            terminal: vec![V::ZERO; n],
            tree: vec![FREE_NODE; n],
            parent_edge: vec![NO_PARENT; n],
            current_arc: vec![0; n],
            timestamp: vec![0; n],
            distance: vec![0; n],
            is_active: vec![false; n],
            active: VecDeque::new(),
            orphans: VecDeque::new(),
            flow: graph.direct_flow_capacity,
            time: 0,
            work: 0,
        };

        // Push s -> v -> t directly, then seed both trees with the nodes that still have
        // terminal capacity.
        for v in 0..n {
            let to_source = graph.source_capacity[v];
            let to_sink = graph.sink_capacity[v];
            solver.flow = solver.flow.add(to_source.min_value(to_sink));
            solver.terminal[v] = to_source.sub(to_sink);
            if solver.terminal[v] > V::ZERO {
                solver.add_to_tree(v, SOURCE_TREE, TERMINAL_PARENT);
            } else if solver.terminal[v] < V::ZERO {
                solver.add_to_tree(v, SINK_TREE, TERMINAL_PARENT);
            }
        }
        solver
    }

    fn run(&mut self) -> V {
        while let Some(meeting_edge) = self.grow() {
            let bottleneck = self.augment(meeting_edge);
            self.flow = self.flow.add(bottleneck);
            self.adopt();
        }
        self.flow
    }

    fn add_to_tree(&mut self, node: usize, which_tree: u8, parent: isize) {
        self.tree[node] = which_tree;
        self.parent_edge[node] = parent;
        self.distance[node] = 1;
        self.activate(node);
    }

    /// Queues the node if it is not queued, and restarts its arc scan in any case: arcs it
    /// already passed may lead somewhere new (a neighbor that was freed, for example).
    fn activate(&mut self, node: usize) {
        self.current_arc[node] = self.arc_start[node];
        if !self.is_active[node] {
            self.is_active[node] = true;
            self.active.push_back(node);
        }
    }

    /// The node at the parent end of a real parent edge.
    fn parent_of(&self, node: usize) -> usize {
        self.head[self.parent_edge[node] as usize]
    }

    /// Growth stage: expand the trees from active nodes until they touch. Returns the edge
    /// from the source-tree side to the sink-tree side where they met, or `None` when no
    /// active node is left, which means the flow is maximal.
    fn grow(&mut self) -> Option<usize> {
        while let Some(&p) = self.active.front() {
            self.work += 1;
            if self.tree[p] == FREE_NODE {
                // Became free during adoption after it was queued.
                self.active.pop_front();
                self.is_active[p] = false;
                continue;
            }

            let end = self.arc_start[p + 1];
            for k in self.current_arc[p]..end {
                self.work += 1;
                let e = self.arcs[k];
                let q = self.head[e];
                // Tree capacity: parent-to-child residual in S, child-to-parent in T.
                let tree_edge = if self.tree[p] == SOURCE_TREE {
                    e
                } else {
                    e ^ 1
                };
                if self.residual[tree_edge] <= V::ZERO {
                    continue;
                }

                if self.tree[q] == FREE_NODE {
                    self.add_to_tree(q, self.tree[p], (e ^ 1) as isize);
                    self.distance[q] = self.distance[p] + 1;
                    self.timestamp[q] = self.timestamp[p];
                } else if self.tree[q] != self.tree[p] {
                    // p stays active and resumes at this arc: it may still have capacity once
                    // this path is saturated.
                    self.current_arc[p] = k;
                    return Some(tree_edge);
                }
            }

            self.active.pop_front();
            self.is_active[p] = false;
        }
        None
    }

    /// Augmentation stage: push the bottleneck along source -> ... -> meeting edge -> ... ->
    /// sink, and make orphans of the nodes whose parent link saturates.
    fn augment(&mut self, meeting_edge: usize) -> V {
        let mut bottleneck = self.residual[meeting_edge];
        let mut v = self.head[meeting_edge ^ 1];
        while self.parent_edge[v] != TERMINAL_PARENT {
            self.work += 1;
            let to_parent = self.parent_edge[v] as usize;
            bottleneck = bottleneck.min_value(self.residual[to_parent ^ 1]);
            v = self.head[to_parent];
        }
        bottleneck = bottleneck.min_value(self.terminal[v]);

        v = self.head[meeting_edge];
        while self.parent_edge[v] != TERMINAL_PARENT {
            self.work += 1;
            let to_parent = self.parent_edge[v] as usize;
            bottleneck = bottleneck.min_value(self.residual[to_parent]);
            v = self.head[to_parent];
        }
        bottleneck = bottleneck.min_value(self.terminal[v].neg());

        self.push(meeting_edge, bottleneck);

        v = self.head[meeting_edge ^ 1];
        while self.parent_edge[v] != TERMINAL_PARENT {
            let to_parent = self.parent_edge[v] as usize;
            self.push(to_parent ^ 1, bottleneck);
            if self.residual[to_parent ^ 1] <= V::ZERO {
                self.make_orphan(v);
            }
            v = self.head[to_parent];
        }
        self.terminal[v] = self.terminal[v].sub(bottleneck);
        if self.terminal[v] <= V::ZERO {
            self.make_orphan(v);
        }

        v = self.head[meeting_edge];
        while self.parent_edge[v] != TERMINAL_PARENT {
            let to_parent = self.parent_edge[v] as usize;
            self.push(to_parent, bottleneck);
            if self.residual[to_parent] <= V::ZERO {
                self.make_orphan(v);
            }
            v = self.head[to_parent];
        }
        self.terminal[v] = self.terminal[v].add(bottleneck);
        if self.terminal[v] >= V::ZERO {
            self.make_orphan(v);
        }

        bottleneck
    }

    fn push(&mut self, edge: usize, amount: V) {
        self.residual[edge] = self.residual[edge].sub(amount);
        self.residual[edge ^ 1] = self.residual[edge ^ 1].add(amount);
    }

    fn make_orphan(&mut self, node: usize) {
        self.parent_edge[node] = ORPHAN_PARENT;
        self.orphans.push_back(node);
    }

    /// Adoption stage: give each orphan a new parent in its own tree whose path reaches the
    /// terminal, or free it and orphan its children.
    fn adopt(&mut self) {
        self.time += 1;
        while let Some(p) = self.orphans.pop_front() {
            self.work += 1;
            let own_tree = self.tree[p];

            // Terminal capacity in the tree's direction reconnects it directly. (Terminal
            // capacity never changes sign, so an orphan normally has none left.)
            let has_terminal = if own_tree == SOURCE_TREE {
                self.terminal[p] > V::ZERO
            } else {
                self.terminal[p] < V::ZERO
            };
            if has_terminal {
                self.parent_edge[p] = TERMINAL_PARENT;
                self.timestamp[p] = self.time;
                self.distance[p] = 1;
                continue;
            }

            let mut best_edge: Option<usize> = None;
            let mut best_distance = usize::MAX;
            let (start, end) = (self.arc_start[p], self.arc_start[p + 1]);
            for k in start..end {
                self.work += 1;
                let e = self.arcs[k];
                let q = self.head[e];
                let into_p = if own_tree == SOURCE_TREE { e ^ 1 } else { e };
                if self.tree[q] != own_tree || self.residual[into_p] <= V::ZERO {
                    continue;
                }
                if let Some(d) = self.distance_to_terminal(q) {
                    if d < best_distance {
                        best_edge = Some(e);
                        best_distance = d;
                    }
                }
            }

            if let Some(e) = best_edge {
                self.parent_edge[p] = e as isize;
                self.timestamp[p] = self.time;
                self.distance[p] = best_distance + 1;
                continue;
            }

            for k in start..end {
                self.work += 1;
                let e = self.arcs[k];
                let q = self.head[e];
                if self.tree[q] != own_tree {
                    continue;
                }
                let into_p = if own_tree == SOURCE_TREE { e ^ 1 } else { e };
                if self.residual[into_p] > V::ZERO {
                    self.activate(q);
                }
                if self.parent_edge[q] >= 0 && self.parent_of(q) == p {
                    self.make_orphan(q);
                }
            }

            self.tree[p] = FREE_NODE;
            self.parent_edge[p] = NO_PARENT;
        }
    }

    /// Follows parent edges from q. Returns the distance of q from the terminal (a node with a
    /// terminal link is at 1), or `None` if the path runs into an orphan. Nodes verified
    /// during this adoption stage are timestamped with their distance, so later walks stop
    /// there (the paper's heuristic).
    fn distance_to_terminal(&mut self, q: usize) -> Option<usize> {
        let mut steps = 0;
        let mut v = q;
        let total = loop {
            self.work += 1;
            if self.timestamp[v] == self.time {
                break steps + self.distance[v];
            }
            let to_parent = self.parent_edge[v];
            if to_parent == TERMINAL_PARENT {
                self.timestamp[v] = self.time;
                self.distance[v] = 1;
                break steps + 1;
            }
            if to_parent < 0 {
                return None;
            }
            steps += 1;
            v = self.head[to_parent as usize];
        };

        let mut remaining = total;
        v = q;
        while self.timestamp[v] != self.time {
            self.timestamp[v] = self.time;
            self.distance[v] = remaining;
            remaining -= 1;
            v = self.parent_of(v);
        }
        Some(total)
    }
}
