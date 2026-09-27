//! The k-way graph partitioner behind [`super::graph_cut::compute_normalized_min_graph_cut`].
//! COLMAP calls `METIS_PartGraphKway` there; METIS is not ported (docs/CPP_DIVERGENCES.md,
//! entry 46). Port of colmap-sharp's `Mathematics/MultilevelPartitioner.cs` (our own MIT
//! code), which was written from the published multilevel scheme, not from METIS's code:
//!
//!   - B. Hendrickson and R. Leland, "A Multilevel Algorithm for Partitioning Graphs",
//!     Supercomputing 1995 (coarsen, partition the coarsest graph, project and refine).
//!   - G. Karypis and V. Kumar, "A Fast and High Quality Multilevel Scheme for Partitioning
//!     Irregular Graphs", SIAM J. Sci. Comput. 20(1), 1998 (heavy-edge matching, greedy graph
//!     growing for the initial bisection, recursive bisection for k parts).
//!   - C. M. Fiduccia and R. M. Mattheyses, "A Linear-Time Heuristic for Improving Network
//!     Partitions", DAC 1982 (the boundary refinement with rollback to the best prefix).
//!
//! k parts come from recursive bisection: floor(k/2) parts on one side, the rest on the
//! other, with the target weights split in the same ratio (as `METIS_PartGraphRecursive`
//! splits them). There is no randomness: every choice breaks ties by vertex index, so the
//! output is a pure function of the input graph, and matches colmap-sharp's. Tier C: COLMAP's
//! contract is "a balanced partition with a small cut", and graph_cut_test.cc checks the
//! label range, that both parts are used, and that the disconnected graph is split along its
//! components. Tests: `tests/math/graph_cut.rs`.
//!
//! Translation notes: C# `int` edge and vertex weights stay `i32`, with `wrapping_add` where
//! C# sums them unchecked; gains, cuts and side weights are `i64` as C#'s `long`. C#'s
//! `SortedSet<(long, int)>` is a `BTreeSet<(i64, usize)>` and its min-`PriorityQueue` a
//! `BinaryHeap` of `Reverse` entries; both order by the full tuple, so ties go to the lowest
//! vertex index exactly as in C#.

use crate::check;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap};

/// Allowed imbalance per side: METIS's k-way default ufactor is 30, i.e. 1.03.
const IMBALANCE_TOLERANCE: f64 = 1.03;

/// Graphs this small are bisected directly instead of coarsened further.
const COARSEN_TO: usize = 20;

/// Coarsening stops when a level shrinks the graph by less than this.
const MIN_COARSENING_RATIO: f64 = 0.95;

const MAX_REFINEMENT_PASSES: usize = 10;

const MAX_INITIAL_SEEDS: usize = 8;

/// Fraction of vertices heavy-edge matching may leave unmatched before two-hop matching pairs
/// up the rest.
const UNMATCHED_FOR_TWO_HOP: f64 = 0.1;

/// Partitions a weighted undirected graph given in CSR form (`xadj`, `adjncy`, `adjwgt`, each
/// edge stored in both directions, unit vertex weights) into `num_parts` parts. Returns one
/// label in `[0, num_parts)` per vertex. Parallel edges add up and self-loops are ignored.
/// Fails a check for `num_parts == 0` or a malformed CSR graph (see [`check_csr`]).
pub fn partition(
    xadj: &[usize],
    adjncy: &[usize],
    adjwgt: &[i32],
    num_parts: usize,
) -> crate::Result<Vec<usize>> {
    Ok(partition_with_work(xadj, adjncy, adjwgt, num_parts)?.0)
}

/// [`partition`], also returning a count of the elementary steps taken (adjacency entries,
/// vertices and queue entries visited). Tests use it to pin the running time's growth
/// without timing anything.
pub fn partition_with_work(
    xadj: &[usize],
    adjncy: &[usize],
    adjwgt: &[i32],
    num_parts: usize,
) -> crate::Result<(Vec<usize>, u64)> {
    // num_parts == 0 would recurse forever (0 / 2 parts on each side).
    check!(num_parts >= 1);
    check_csr(xadj, adjncy, adjwgt)?;
    let mut partitioner = Partitioner { work: 0 };
    let num_vertices = xadj.len() - 1;
    let input = Graph {
        xadj: xadj.to_vec(),
        adjncy: adjncy.to_vec(),
        adjwgt: adjwgt.to_vec(),
        vertex_weights: vec![1; num_vertices],
    };
    let identity: Vec<usize> = (0..num_vertices).collect();

    // Contracting by the identity map merges parallel edges and drops self-loops.
    let graph = partitioner.contract(&input, &identity, num_vertices);
    let mut labels = vec![0usize; num_vertices];
    partitioner.partition_recursive(&graph, &identity, num_parts, 0, &mut labels);
    Ok((labels, partitioner.work))
}

/// Greedy graph growing from `seed` for a graph in CSR form with vertex weights; returns 0
/// for the grown side and 1 for the rest. Exposed for tests. Fails a check for a malformed
/// CSR graph, `vertex_weights` not one per vertex, or `seed` out of range.
pub fn grow_region(
    xadj: &[usize],
    adjncy: &[usize],
    adjwgt: &[i32],
    vertex_weights: &[i32],
    seed: usize,
    target0: f64,
    max_weight0: i64,
) -> crate::Result<Vec<usize>> {
    check_csr(xadj, adjncy, adjwgt)?;
    let num_vertices = xadj.len() - 1;
    check!(vertex_weights.len() == num_vertices);
    check!(seed < num_vertices);
    let graph = Graph {
        xadj: xadj.to_vec(),
        adjncy: adjncy.to_vec(),
        adjwgt: adjwgt.to_vec(),
        vertex_weights: vertex_weights.to_vec(),
    };
    Ok(Partitioner { work: 0 }.grow_region(&graph, Some(seed), target0, max_weight0))
}

/// Checks that (`xadj`, `adjncy`, `adjwgt`) is a well-formed CSR graph, so the partitioner
/// can index it without panicking: `xadj` holds n + 1 non-decreasing offsets from 0 to
/// `adjncy.len()`, `adjwgt` has one weight per entry, and every neighbor is below n.
fn check_csr(xadj: &[usize], adjncy: &[usize], adjwgt: &[i32]) -> crate::Result<()> {
    check!(!xadj.is_empty());
    check!(xadj[0] == 0);
    check!(xadj[xadj.len() - 1] == adjncy.len());
    check!(xadj.windows(2).all(|w| w[0] <= w[1]));
    check!(adjwgt.len() == adjncy.len());
    let num_vertices = xadj.len() - 1;
    check!(adjncy.iter().all(|&u| u < num_vertices));
    Ok(())
}

/// A graph in CSR form with vertex weights.
struct Graph {
    xadj: Vec<usize>,
    adjncy: Vec<usize>,
    adjwgt: Vec<i32>,
    vertex_weights: Vec<i32>,
}

impl Graph {
    fn num_vertices(&self) -> usize {
        self.vertex_weights.len()
    }

    fn total_weight(&self) -> i64 {
        self.vertex_weights.iter().map(|&w| i64::from(w)).sum()
    }

    fn edges(&self, v: usize) -> std::ops::Range<usize> {
        self.xadj[v]..self.xadj[v + 1]
    }
}

/// `(excess weight over the balance limits, cut)`: states compare by excess first.
type Score = (i64, i64);

/// One partitioning run; `work` counts elementary steps (see [`partition_with_work`]).
struct Partitioner {
    work: u64,
}

impl Partitioner {
    fn partition_recursive(
        &mut self,
        graph: &Graph,
        original_ids: &[usize],
        num_parts: usize,
        label_offset: usize,
        labels: &mut [usize],
    ) {
        if graph.num_vertices() == 0 {
            return;
        }

        if num_parts == 1 {
            for &id in original_ids {
                labels[id] = label_offset;
            }
            return;
        }

        let left_parts = num_parts / 2;
        let side = self.bisect(graph, left_parts as f64 / num_parts as f64);
        for s in 0..2 {
            let (subgraph, sub_ids) = self.extract_side(graph, original_ids, &side, s);
            self.partition_recursive(
                &subgraph,
                &sub_ids,
                if s == 0 {
                    left_parts
                } else {
                    num_parts - left_parts
                },
                if s == 0 {
                    label_offset
                } else {
                    label_offset + left_parts
                },
                labels,
            );
        }
    }

    /// Multilevel bisection: coarsen by heavy-edge matching, bisect the coarsest graph, then
    /// project back level by level with FM refinement at each one.
    fn bisect(&mut self, graph: &Graph, left_fraction: f64) -> Vec<usize> {
        let total = graph.total_weight();
        let targets = [
            left_fraction * total as f64,
            (1.0 - left_fraction) * total as f64,
        ];
        // The ceiling keeps an exact split feasible for small graphs of unit weights.
        let max_weights = targets.map(|t| (t * IMBALANCE_TOLERANCE).floor().max(t.ceil()) as i64);

        let mut levels: Vec<Graph> = Vec::new();
        let mut coarse_maps: Vec<Vec<usize>> = Vec::new();
        let max_coarse_vertex_weight = 1.max((1.5 * total as f64 / COARSEN_TO as f64) as i32);
        loop {
            let current = levels.last().unwrap_or(graph);
            if current.num_vertices() <= COARSEN_TO {
                break;
            }
            let (coarse_map, num_coarse) =
                self.heavy_edge_matching(current, max_coarse_vertex_weight);
            if num_coarse as f64 > MIN_COARSENING_RATIO * current.num_vertices() as f64 {
                break;
            }
            let coarse = self.contract(current, &coarse_map, num_coarse);
            coarse_maps.push(coarse_map);
            levels.push(coarse);
        }

        let mut side =
            self.initial_bisection(levels.last().unwrap_or(graph), targets[0], &max_weights);
        // Level 0 is the input graph; level i + 1 is levels[i].
        for level in (0..coarse_maps.len()).rev() {
            let coarse_map = &coarse_maps[level];
            let mut fine = vec![0usize; coarse_map.len()];
            for (v, f) in fine.iter_mut().enumerate() {
                self.work += 1;
                *f = side[coarse_map[v]];
            }
            side = fine;
            let fine_graph = if level == 0 {
                graph
            } else {
                &levels[level - 1]
            };
            self.refine_fm(fine_graph, &mut side, &max_weights);
        }
        side
    }

    /// Visits vertices by increasing degree (index breaks ties) and matches each unmatched one
    /// with the unmatched neighbor across its heaviest edge, as long as the merged weight
    /// stays under the cap that keeps coarse vertices from dominating the balance.
    fn heavy_edge_matching(
        &mut self,
        graph: &Graph,
        max_vertex_weight: i32,
    ) -> (Vec<usize>, usize) {
        let n = graph.num_vertices();
        let mut order: Vec<usize> = (0..n).collect();
        let degrees: Vec<usize> = (0..n).map(|v| graph.xadj[v + 1] - graph.xadj[v]).collect();
        self.work += n as u64;
        order.sort_by_key(|&v| (degrees[v], v));

        const UNMATCHED: usize = usize::MAX;
        let mut matched = vec![UNMATCHED; n];
        for &v in &order {
            self.work += 1;
            if matched[v] != UNMATCHED {
                continue;
            }

            let mut best = v;
            let mut best_weight = i32::MIN;
            for e in graph.edges(v) {
                self.work += 1;
                let u = graph.adjncy[e];
                let w = graph.adjwgt[e];
                if matched[u] != UNMATCHED
                    || graph.vertex_weights[u].wrapping_add(graph.vertex_weights[v])
                        > max_vertex_weight
                {
                    continue;
                }
                if w > best_weight || (w == best_weight && u < best) {
                    best = u;
                    best_weight = w;
                }
            }

            matched[v] = best;
            matched[best] = v;
        }

        let num_unmatched = (0..n).filter(|&v| matched[v] == v).count();
        self.work += n as u64;

        // Heavy-edge matching stalls on stars and hub-dominated graphs (one hub matches, its
        // leaves have no one left) and on isolated vertices, which would leave the coarsest
        // graph nearly as large as the input and everything after it quadratic. When over
        // UNMATCHED_FOR_TWO_HOP of the vertices are left unmatched, pair them up further.
        if num_unmatched as f64 > UNMATCHED_FOR_TWO_HOP * n as f64 {
            self.match_two_hop(graph, &mut matched, max_vertex_weight);
        }

        let mut coarse_map = vec![usize::MAX; n];
        let mut num_coarse = 0;
        for v in 0..n {
            self.work += 1;
            if coarse_map[v] == usize::MAX {
                coarse_map[v] = num_coarse;
                coarse_map[matched[v]] = num_coarse;
                num_coarse += 1;
            }
        }
        (coarse_map, num_coarse)
    }

    /// Pairs still-unmatched vertices (`matched[v] == v`) that share a neighbor: for each
    /// vertex h in index order, its unmatched neighbors pair off in adjacency order. Then
    /// unmatched isolated vertices, which share the empty neighborhood, pair off in index
    /// order. Both respect the coarse weight cap. Linear in the number of edges.
    fn match_two_hop(&mut self, graph: &Graph, matched: &mut [usize], max_vertex_weight: i32) {
        let n = graph.num_vertices();
        let vw = &graph.vertex_weights;
        let fits = |a: usize, b: usize| vw[a].wrapping_add(vw[b]) <= max_vertex_weight;
        for h in 0..n {
            self.work += 1;
            let mut pending: Option<usize> = None;
            for e in graph.edges(h) {
                self.work += 1;
                let u = graph.adjncy[e];
                if matched[u] != u {
                    continue;
                }
                match pending {
                    Some(p) if fits(p, u) => {
                        matched[p] = u;
                        matched[u] = p;
                        pending = None;
                    }
                    Some(p) if vw[u] < vw[p] => pending = Some(u),
                    Some(_) => {}
                    None => pending = Some(u),
                }
            }
        }

        let mut pending_isolated: Option<usize> = None;
        for v in 0..n {
            self.work += 1;
            if matched[v] != v || graph.xadj[v + 1] != graph.xadj[v] {
                continue;
            }
            match pending_isolated {
                Some(p) if fits(p, v) => {
                    matched[p] = v;
                    matched[v] = p;
                    pending_isolated = None;
                }
                Some(p) if vw[v] < vw[p] => pending_isolated = Some(v),
                Some(_) => {}
                None => pending_isolated = Some(v),
            }
        }
    }

    /// Builds the graph whose vertex c is the union of the fine vertices mapped to c. Edge
    /// weights between merged vertices add up; edges inside a merged vertex disappear.
    fn contract(&mut self, graph: &Graph, coarse_map: &[usize], num_coarse: usize) -> Graph {
        let mut members: Vec<Vec<usize>> = vec![Vec::new(); num_coarse];
        for v in 0..graph.num_vertices() {
            self.work += 1;
            members[coarse_map[v]].push(v);
        }

        let mut xadj = vec![0usize; num_coarse + 1];
        let mut adjncy: Vec<usize> = Vec::new();
        let mut adjwgt: Vec<i32> = Vec::new();
        let mut vertex_weights = vec![0i32; num_coarse];
        let mut slot = vec![usize::MAX; num_coarse];
        for c in 0..num_coarse {
            let start = adjncy.len();
            for &v in &members[c] {
                self.work += 1;
                vertex_weights[c] = vertex_weights[c].wrapping_add(graph.vertex_weights[v]);
                for e in graph.edges(v) {
                    self.work += 1;
                    let cu = coarse_map[graph.adjncy[e]];
                    if cu == c {
                        continue;
                    }
                    if slot[cu] == usize::MAX {
                        slot[cu] = adjncy.len();
                        adjncy.push(cu);
                        adjwgt.push(graph.adjwgt[e]);
                    } else {
                        adjwgt[slot[cu]] = adjwgt[slot[cu]].wrapping_add(graph.adjwgt[e]);
                    }
                }
            }
            for &cu in &adjncy[start..] {
                slot[cu] = usize::MAX;
            }
            xadj[c + 1] = adjncy.len();
        }

        Graph {
            xadj,
            adjncy,
            adjwgt,
            vertex_weights,
        }
    }

    /// Greedy graph growing from several evenly spaced seeds: side 0 grows by the vertex that
    /// most reduces the cut among those adjacent to it (the lowest-index remaining vertex when
    /// none is adjacent, so disconnected graphs still grow) until it reaches its target
    /// weight. A vertex too heavy to fit is skipped, not an end to growth. Each candidate is
    /// FM-refined and the best one kept.
    fn initial_bisection(
        &mut self,
        graph: &Graph,
        target0: f64,
        max_weights: &[i64; 2],
    ) -> Vec<usize> {
        let n = graph.num_vertices();
        let num_seeds = n.min(MAX_INITIAL_SEEDS);
        let mut best: Option<(Vec<usize>, Score)> = None;
        for t in 0..num_seeds {
            let mut side =
                self.grow_region(graph, Some(t * n / num_seeds), target0, max_weights[0]);
            self.refine_fm(graph, &mut side, max_weights);
            let score = self.score(graph, &side, max_weights);
            if best
                .as_ref()
                .is_none_or(|(_, best_score)| score < *best_score)
            {
                best = Some((side, score));
            }
        }
        best.map(|(side, _)| side).unwrap_or_default()
    }

    fn grow_region(
        &mut self,
        graph: &Graph,
        seed: Option<usize>,
        target0: f64,
        max_weight0: i64,
    ) -> Vec<usize> {
        let n = graph.num_vertices();
        let vw = &graph.vertex_weights;
        let mut side = vec![1usize; n];
        let mut connection = vec![0i64; n];
        let mut degree = vec![0i64; n];
        let mut touched = vec![false; n];

        // A vertex too heavy for side 0 now never fits later, since side 0 only grows.
        let mut rejected = vec![false; n];
        for (v, d) in degree.iter_mut().enumerate() {
            self.work += 1;
            for e in graph.edges(v) {
                self.work += 1;
                *d += i64::from(graph.adjwgt[e]);
            }
        }

        // Touched candidates by largest gain, then lowest index. Gains only rise as side 0
        // grows, so an entry whose gain no longer matches is a stale duplicate.
        let mut queue: BinaryHeap<Reverse<(i64, usize)>> = BinaryHeap::new();
        let mut cursor = 0;
        let mut weight0: i64 = 0;
        let mut next = seed;
        while (weight0 as f64) < target0 {
            if next.is_none() {
                while let Some(Reverse((neg_gain, v))) = queue.pop() {
                    self.work += 1;
                    if side[v] == 0 || rejected[v] || -neg_gain != 2 * connection[v] - degree[v] {
                        continue;
                    }
                    if weight0 + i64::from(vw[v]) > max_weight0 {
                        rejected[v] = true;
                        continue;
                    }
                    next = Some(v);
                    break;
                }
            }

            // No touched vertex fits: continue from the lowest-index untouched vertex, so
            // disconnected graphs still grow.
            while next.is_none() && cursor < n {
                self.work += 1;
                let v = cursor;
                cursor += 1;
                if side[v] == 0 || touched[v] || rejected[v] {
                    continue;
                }
                if weight0 + i64::from(vw[v]) > max_weight0 {
                    rejected[v] = true;
                    continue;
                }
                next = Some(v);
            }

            let Some(chosen) = next else {
                break;
            };

            if weight0 + i64::from(vw[chosen]) > max_weight0 {
                rejected[chosen] = true;
                next = None;
                continue;
            }

            side[chosen] = 0;
            weight0 += i64::from(vw[chosen]);
            for e in graph.edges(chosen) {
                self.work += 1;
                let u = graph.adjncy[e];
                connection[u] += i64::from(graph.adjwgt[e]);
                touched[u] = true;
                if side[u] == 1 && !rejected[u] {
                    queue.push(Reverse((-(2 * connection[u] - degree[u]), u)));
                }
            }
            next = None;
        }
        side
    }

    fn score(&mut self, graph: &Graph, side: &[usize], max_weights: &[i64; 2]) -> Score {
        let mut weights = [0i64; 2];
        let mut cut: i64 = 0;
        for v in 0..graph.num_vertices() {
            self.work += 1;
            weights[side[v]] += i64::from(graph.vertex_weights[v]);
            for e in graph.edges(v) {
                self.work += 1;
                if side[graph.adjncy[e]] != side[v] {
                    cut += i64::from(graph.adjwgt[e]);
                }
            }
        }
        (excess(&weights, max_weights), cut / 2)
    }

    /// Fiduccia-Mattheyses passes. Each pass moves every vertex at most once, always taking
    /// the allowed move with the largest cut reduction (even a negative one, to climb out of
    /// local minima), then rolls back to the best state seen. States compare by excess weight
    /// over the balance limits first and cut second, so an unbalanced projection is repaired.
    /// A move is allowed when the source is over its limit, or when the destination stays
    /// within its limit plus one heaviest vertex. That slack lets a pass swap vertices when
    /// both sides sit exactly at their limits (an exact split of unit weights leaves no other
    /// move); the rollback only keeps a state that is back within the limits, or as close as
    /// it got.
    fn refine_fm(&mut self, graph: &Graph, side: &mut [usize], max_weights: &[i64; 2]) {
        let n = graph.num_vertices();
        let mut gains = vec![0i64; n];
        let mut locked = vec![false; n];
        let mut moves: Vec<usize> = Vec::with_capacity(n);
        let max_non_improving = (n / 100).clamp(15, 100);
        let min_vertex_weight = graph.vertex_weights.iter().copied().min().unwrap_or(0);
        let slack = graph.vertex_weights.iter().copied().max().unwrap_or(0);
        for _ in 0..MAX_REFINEMENT_PASSES {
            let (excess0, mut cut) = self.score(graph, side, max_weights);
            let mut weights = [0i64; 2];
            let mut queues: [BTreeSet<(i64, usize)>; 2] = [BTreeSet::new(), BTreeSet::new()];
            for v in 0..n {
                self.work += 1;
                weights[side[v]] += i64::from(graph.vertex_weights[v]);
                gains[v] = self.gain(graph, side, v);
                locked[v] = false;
                queues[side[v]].insert((-gains[v], v));
            }

            let mut best_score = (excess0, cut);
            let mut best_count = 0;
            let mut non_improving = 0;
            moves.clear();
            while let Some(v) = self.select_move(
                graph,
                &queues,
                &weights,
                max_weights,
                min_vertex_weight,
                slack,
            ) {
                let from = side[v];
                queues[from].remove(&(-gains[v], v));
                locked[v] = true;
                side[v] = 1 - from;
                weights[from] -= i64::from(graph.vertex_weights[v]);
                weights[1 - from] += i64::from(graph.vertex_weights[v]);
                cut -= gains[v];
                for e in graph.edges(v) {
                    self.work += 1;
                    let u = graph.adjncy[e];
                    if locked[u] {
                        continue;
                    }
                    queues[side[u]].remove(&(-gains[u], u));
                    let w = 2 * i64::from(graph.adjwgt[e]);
                    gains[u] += if side[u] == from { w } else { -w };
                    queues[side[u]].insert((-gains[u], u));
                }

                moves.push(v);
                let score = (excess(&weights, max_weights), cut);
                if score < best_score {
                    best_score = score;
                    best_count = moves.len();
                    non_improving = 0;
                } else {
                    non_improving += 1;
                    if non_improving > max_non_improving {
                        break;
                    }
                }
            }

            for &v in moves[best_count..].iter().rev() {
                side[v] = 1 - side[v];
            }

            if best_count == 0 {
                break;
            }
        }
    }

    /// The cut reduction from moving v to the other side.
    fn gain(&mut self, graph: &Graph, side: &[usize], v: usize) -> i64 {
        let mut gain = 0i64;
        for e in graph.edges(v) {
            self.work += 1;
            let w = i64::from(graph.adjwgt[e]);
            gain += if side[graph.adjncy[e]] != side[v] {
                w
            } else {
                -w
            };
        }
        gain
    }

    /// The best allowed move from each side's queue; between the two, the larger gain wins,
    /// then the move out of the side further over its limit, then the lower vertex index.
    fn select_move(
        &mut self,
        graph: &Graph,
        queues: &[BTreeSet<(i64, usize)>; 2],
        weights: &[i64; 2],
        max_weights: &[i64; 2],
        min_vertex_weight: i32,
        slack: i32,
    ) -> Option<usize> {
        let mut candidates: [Option<(i64, usize)>; 2] = [None, None];
        for from in 0..2 {
            let to = 1 - from;
            let source_over = weights[from] > max_weights[from];
            let limit = max_weights[to] + i64::from(slack);
            if !source_over && weights[to] + i64::from(min_vertex_weight) > limit {
                // Nothing fits; skip the scan that would reject every queued vertex.
                continue;
            }
            for &entry in &queues[from] {
                self.work += 1;
                if source_over || weights[to] + i64::from(graph.vertex_weights[entry.1]) <= limit {
                    candidates[from] = Some(entry);
                    break;
                }
            }
        }

        match candidates {
            [None, c1] => c1.map(|c| c.1),
            [Some(c0), None] => Some(c0.1),
            [Some(c0), Some(c1)] => {
                if c0.0 != c1.0 {
                    return Some(if c0.0 < c1.0 { c0.1 } else { c1.1 });
                }
                let over0 = weights[0] - max_weights[0];
                let over1 = weights[1] - max_weights[1];
                if over0 != over1 {
                    return Some(if over0 > over1 { c0.1 } else { c1.1 });
                }
                Some(c0.1.min(c1.1))
            }
        }
    }

    /// The subgraph induced by the vertices on side s, renumbered in increasing order.
    fn extract_side(
        &mut self,
        graph: &Graph,
        original_ids: &[usize],
        side: &[usize],
        s: usize,
    ) -> (Graph, Vec<usize>) {
        let n = graph.num_vertices();
        let mut new_index = vec![usize::MAX; n];
        let mut ids: Vec<usize> = Vec::new();
        let mut vertex_weights: Vec<i32> = Vec::new();
        for v in 0..n {
            self.work += 1;
            if side[v] == s {
                new_index[v] = ids.len();
                ids.push(original_ids[v]);
                vertex_weights.push(graph.vertex_weights[v]);
            }
        }

        let mut xadj = vec![0usize; ids.len() + 1];
        let mut adjncy: Vec<usize> = Vec::new();
        let mut adjwgt: Vec<i32> = Vec::new();
        let mut index = 0;
        for v in 0..n {
            self.work += 1;
            if side[v] != s {
                continue;
            }
            for e in graph.edges(v) {
                self.work += 1;
                let u = graph.adjncy[e];
                if side[u] == s {
                    adjncy.push(new_index[u]);
                    adjwgt.push(graph.adjwgt[e]);
                }
            }
            index += 1;
            xadj[index] = adjncy.len();
        }

        let subgraph = Graph {
            xadj,
            adjncy,
            adjwgt,
            vertex_weights,
        };
        (subgraph, ids)
    }
}

/// Weight over the balance limits, summed over both sides.
fn excess(weights: &[i64; 2], max_weights: &[i64; 2]) -> i64 {
    (weights[0] - max_weights[0]).max(0) + (weights[1] - max_weights[1]).max(0)
}
