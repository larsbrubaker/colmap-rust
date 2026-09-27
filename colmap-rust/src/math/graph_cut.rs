//! Port of the free functions of COLMAP's `colmap/math/graph_cut.h` and `graph_cut.cc`:
//! [`compute_min_graph_cut_stoer_wagner`], the global min-cut of an undirected graph, and
//! [`compute_normalized_min_graph_cut`], the balanced k-way partition. Port of colmap-sharp's
//! `Mathematics/GraphCut.cs`. The S-T min-cut class from the same header is in
//! [`super::graph_cut_min_st`]; the partitioner that stands in for METIS is in
//! [`super::graph_cut_partitioner`]. Tests: `tests/math/graph_cut.rs` (graph_cut_test.cc 1:1,
//! plus colmap-sharp's C#-only cross-check, scaling and partition tests as `rust_only_*`).
//!
//! COLMAP calls `boost::stoer_wagner_min_cut`. Boost is not ported (docs/LICENSE_AUDIT.md);
//! this is the algorithm written from its paper: M. Stoer and F. Wagner, "A Simple Min-Cut
//! Algorithm", Journal of the ACM 44(4), 1997.
//!
//! Tier A for the cut weight (the minimum is unique). When several cuts share that weight,
//! the side reported can differ from Boost's, and which side is labeled 1 is this code's
//! choice (the vertices merged into the last-added vertex of the best phase).
//! graph_cut_test.cc only pins the weight and the label range. See docs/CPP_DIVERGENCES.md,
//! entry 44.
//!
//! `compute_normalized_min_graph_cut` builds the same CSR graph as COLMAP's `MetisGraph`
//! wrapper but partitions it with the multilevel partitioner instead of
//! `METIS_PartGraphKway`. Tier C: the parts are balanced with a small cut, but need not be
//! METIS's parts. See docs/CPP_DIVERGENCES.md, entry 46.

use super::graph_cut_partitioner;
use crate::check;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

/// Port of `colmap::ComputeMinGraphCutStoerWagner`: the minimum cut of an undirected graph
/// with vertices `0..=max vertex id`. Returns `(cut_weight, cut_labels)`, where `cut_labels`
/// has one entry per vertex, 0 or 1 by side of the cut. Parallel edges add up; self-loops
/// never cross a cut.
pub fn compute_min_graph_cut_stoer_wagner(
    edges: &[(i32, i32)],
    weights: &[i32],
) -> crate::Result<(i32, Vec<u8>)> {
    check!(edges.len() == weights.len());
    check!(edges.len() >= 2);

    let mut max_vertex_index = 0;
    for &(v1, v2) in edges {
        check!(v1 >= 0);
        check!(v2 >= 0);
        max_vertex_index = max_vertex_index.max(v1);
        max_vertex_index = max_vertex_index.max(v2);
    }

    let num_vertices = max_vertex_index as usize + 1;

    // boost::stoer_wagner_min_cut throws bad_graph for fewer than two vertices.
    check!(num_vertices >= 2);

    // Merged-graph adjacency: neighbor -> total weight. A BTreeMap keeps the iteration order
    // fixed; it only decides the order in which keys are raised, which cannot change a key
    // (integer sums) or the queue's pick (a total order), so any map would give the same cut.
    // Sums stay i32 as Boost's do; wrapping_add stands in for C++'s (UB) signed overflow.
    let mut adjacency: Vec<BTreeMap<usize, i32>> = vec![BTreeMap::new(); num_vertices];
    let mut members: Vec<Vec<usize>> = (0..num_vertices).map(|v| vec![v]).collect();
    for (&(v1, v2), &weight) in edges.iter().zip(weights) {
        if v1 == v2 {
            continue;
        }
        let (v1, v2) = (v1 as usize, v2 as usize);
        let entry = adjacency[v1].entry(v2).or_insert(0);
        *entry = entry.wrapping_add(weight);
        let entry = adjacency[v2].entry(v1).or_insert(0);
        *entry = entry.wrapping_add(weight);
    }

    let mut active: Vec<usize> = (0..num_vertices).collect();
    let mut best_weight = i32::MAX;
    let mut best_side: Vec<usize> = Vec::new();

    let mut key = vec![0i32; num_vertices];
    let mut in_a = vec![false; num_vertices];

    // Each phase orders the remaining (merged) vertices by "most tightly connected to the set
    // so far". The weight between the last vertex and the rest is a minimum cut separating
    // the last two, which are then merged. The best such cut is the global one. The queue
    // pops the largest key, then the lowest vertex index.
    let mut queue: BinaryHeap<(i32, Reverse<usize>)> = BinaryHeap::new();
    while active.len() > 1 {
        queue.clear();
        for &v in &active {
            key[v] = 0;
            in_a[v] = false;
            queue.push((0, Reverse(v)));
        }

        let mut previous = usize::MAX;
        let mut last = usize::MAX;
        for _ in 0..active.len() {
            // Lazy deletion: skip entries for vertices already added or with a stale key.
            let v = loop {
                let (k, Reverse(v)) = queue
                    .pop()
                    .expect("every vertex not yet added has a live queue entry");
                if !in_a[v] && k == key[v] {
                    break v;
                }
            };

            in_a[v] = true;
            previous = last;
            last = v;
            for (&neighbor, &weight) in &adjacency[v] {
                if !in_a[neighbor] {
                    key[neighbor] = key[neighbor].wrapping_add(weight);
                    queue.push((key[neighbor], Reverse(neighbor)));
                }
            }
        }

        let cut_of_the_phase = key[last];
        if cut_of_the_phase < best_weight {
            best_weight = cut_of_the_phase;
            best_side = members[last].clone();
        }

        merge(&mut adjacency, &mut members, previous, last);
        if let Some(pos) = active.iter().position(|&v| v == last) {
            active.remove(pos);
        }
    }

    let mut cut_labels = vec![0u8; num_vertices];
    for v in best_side {
        cut_labels[v] = 1;
    }
    Ok((best_weight, cut_labels))
}

/// Contracts vertex `from` into vertex `into`, summing parallel edges and dropping the edge
/// between them.
fn merge(
    adjacency: &mut [BTreeMap<usize, i32>],
    members: &mut [Vec<usize>],
    into: usize,
    from: usize,
) {
    let from_edges = std::mem::take(&mut adjacency[from]);
    for (neighbor, weight) in from_edges {
        adjacency[neighbor].remove(&from);
        if neighbor == into {
            continue;
        }
        let entry = adjacency[into].entry(neighbor).or_insert(0);
        *entry = entry.wrapping_add(weight);
        let entry = adjacency[neighbor].entry(into).or_insert(0);
        *entry = entry.wrapping_add(weight);
    }

    let from_members = std::mem::take(&mut members[from]);
    members[into].extend(from_members);
}

/// Port of `colmap::ComputeNormalizedMinGraphCut`: splits the graph into `num_parts` parts of
/// about equal vertex count with a small total weight of cut edges. Returns vertex id -> part
/// label in `[0, num_parts)`. Only vertices that appear in an edge are labeled. Parallel
/// edges add up.
///
/// COLMAP returns a `NodeHashMap<int, int>`; this returns a `BTreeMap` so that iterating the
/// result is deterministic (entry 46).
pub fn compute_normalized_min_graph_cut(
    edges: &[(i32, i32)],
    weights: &[i32],
    num_parts: i32,
) -> crate::Result<BTreeMap<i32, i32>> {
    check!(!edges.is_empty());
    check!(edges.len() == weights.len());
    check!(num_parts > 0);

    // As COLMAP's MetisGraph: vertex indices in order of first appearance, and each vertex's
    // neighbors in edge order. The id -> index lookup is only queried, never iterated.
    let mut id_to_index: BTreeMap<i32, usize> = BTreeMap::new();
    let mut index_to_id: Vec<i32> = Vec::new();
    let mut adjacency: Vec<Vec<(usize, i32)>> = Vec::new();
    let mut vertex_index = |id: i32, adjacency: &mut Vec<Vec<(usize, i32)>>| -> usize {
        *id_to_index.entry(id).or_insert_with(|| {
            index_to_id.push(id);
            adjacency.push(Vec::new());
            index_to_id.len() - 1
        })
    };

    for (&(id1, id2), &weight) in edges.iter().zip(weights) {
        let index1 = vertex_index(id1, &mut adjacency);
        let index2 = vertex_index(id2, &mut adjacency);
        adjacency[index1].push((index2, weight));
        adjacency[index2].push((index1, weight));
    }

    let num_vertices = adjacency.len();
    let mut xadj = Vec::with_capacity(num_vertices + 1);
    let mut adjncy = Vec::with_capacity(2 * edges.len());
    let mut adjwgt = Vec::with_capacity(2 * edges.len());
    for neighbors in &adjacency {
        xadj.push(adjncy.len());
        for &(neighbor, weight) in neighbors {
            adjncy.push(neighbor);
            adjwgt.push(weight);
        }
    }
    xadj.push(adjncy.len());
    check!(adjncy.len() == 2 * edges.len());

    let cut_labels = graph_cut_partitioner::partition(&xadj, &adjncy, &adjwgt, num_parts as usize);

    Ok(index_to_id
        .iter()
        .zip(cut_labels)
        .map(|(&id, label)| (id, label as i32))
        .collect())
}
