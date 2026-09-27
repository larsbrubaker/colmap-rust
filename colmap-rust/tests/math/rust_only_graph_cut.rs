// Rust-only (not COLMAP tests): ports of colmap-sharp's C#-only GraphCutCrossCheckTests,
// MinSTGraphCutScalingTests and NormalizedMinGraphCutTests. The graph cuts are written from
// their papers in place of Boost and METIS (docs/CPP_DIVERGENCES.md, entries 44-46), so
// graph_cut_test.cc's tiny graphs (ported 1:1 in graph_cut.rs) are not enough to trust them:
// these check Stoer-Wagner and MinSTGraphCut against exhaustive search, MinSTGraphCut and the
// partitioner for linear work at scale, and the partitioner's outcome (Tier C) on larger
// graphs. Random inputs come from our mt19937 with libc++'s uniform_int_distribution, so the
// draws differ from the C# tests' System.Random ones; the properties checked are the same.

use colmap_rust::math::graph_cut::{
    compute_min_graph_cut_stoer_wagner, compute_normalized_min_graph_cut,
};
use colmap_rust::math::graph_cut_min_st::MinSTGraphCut;
use colmap_rust::math::graph_cut_partitioner::{grow_region, partition_with_work};
use colmap_rust::math::random::libcxx::{generate_canonical, uniform_int};
use colmap_rust::math::random::Mt19937;
use std::collections::BTreeSet;

/// C#'s `Random.Next(lo, hi)`: uniform in `[lo, hi)`.
fn next(g: &mut Mt19937, lo: i32, hi: i32) -> i32 {
    uniform_int(g, lo, hi - 1)
}

fn cut_weight(edges: &[(i32, i32)], weights: &[i32], side: impl Fn(i32) -> bool) -> i32 {
    edges
        .iter()
        .zip(weights)
        .filter(|(&(a, b), _)| side(a) != side(b))
        .map(|(_, &w)| w)
        .sum()
}

#[test]
fn rust_only_stoer_wagner_matches_exhaustive_min_cut() {
    let mut g = Mt19937::new(1234);
    for _ in 0..200 {
        let num_vertices = next(&mut g, 2, 9);
        let num_edges = next(&mut g, 2, 20);
        // The first edge pins the vertex count (Stoer-Wagner needs at least two vertices).
        let mut edges = vec![(0, num_vertices - 1)];
        let mut weights = vec![next(&mut g, 0, 10)];
        for _ in 1..num_edges {
            let a = next(&mut g, 0, num_vertices);
            let b = next(&mut g, 0, num_vertices);
            edges.push((a, b));
            weights.push(next(&mut g, 0, 10));
        }

        let (weight, labels) = compute_min_graph_cut_stoer_wagner(&edges, &weights).unwrap();
        let n = labels.len();
        let best = (1..(1u32 << n) - 1)
            .map(|mask| cut_weight(&edges, &weights, |v| (mask >> v) & 1 == 1))
            .min()
            .unwrap();
        let labels_one_side = labels.iter().filter(|&&l| l == 1).count();
        assert_eq!(weight, best);
        assert_eq!(
            cut_weight(&edges, &weights, |v| labels[v as usize] == 1),
            weight
        );
        assert!(labels_one_side > 0);
        assert!(labels_one_side < n);
    }
}

#[test]
fn rust_only_min_st_graph_cut_matches_exhaustive_min_cut() {
    let mut g = Mt19937::new(4321);
    for _ in 0..300 {
        let num_nodes = next(&mut g, 1, 9) as usize;
        let mut graph = MinSTGraphCut::<i32>::new(num_nodes);
        let mut source_capacities = vec![0; num_nodes];
        let mut sink_capacities = vec![0; num_nodes];
        let mut arcs: Vec<(usize, usize, i32)> = Vec::new();
        for v in 0..num_nodes {
            source_capacities[v] = if next(&mut g, 0, 3) == 0 {
                0
            } else {
                next(&mut g, 0, 10)
            };
            sink_capacities[v] = if next(&mut g, 0, 3) == 0 {
                0
            } else {
                next(&mut g, 0, 10)
            };
            graph
                .add_node(v, source_capacities[v], sink_capacities[v])
                .unwrap();
        }

        let num_edges = next(&mut g, 0, 16);
        for _ in 0..num_edges {
            // Occasionally an endpoint is a terminal (index num_nodes or num_nodes + 1), which
            // COLMAP's index checks allow.
            let endpoint = |g: &mut Mt19937| {
                if next(g, 0, 10) == 0 {
                    num_nodes + next(g, 0, 2) as usize
                } else {
                    next(g, 0, num_nodes as i32) as usize
                }
            };
            let a = endpoint(&mut g);
            let b = endpoint(&mut g);
            let forward = next(&mut g, 0, 10);
            let backward = next(&mut g, 0, 10);
            graph.add_edge(a, b, forward, backward).unwrap();
            arcs.push((a, b, forward));
            arcs.push((b, a, backward));
        }

        // Capacity of the cut whose source side is `on_source_side`.
        let cut_capacity = |on_source_side: &dyn Fn(usize) -> bool| -> i32 {
            let mut total = 0;
            for v in 0..num_nodes {
                total += if on_source_side(v) {
                    sink_capacities[v]
                } else {
                    source_capacities[v]
                };
            }
            for &(from, to, capacity) in &arcs {
                let from_source = from == num_nodes || (from < num_nodes && on_source_side(from));
                let to_source = to == num_nodes || (to < num_nodes && on_source_side(to));
                if from_source && !to_source {
                    total += capacity;
                }
            }
            total
        };

        let mut best = i32::MAX;
        let mut minimal_sink_side = 0u32;
        for mask in 0..(1u32 << num_nodes) {
            let capacity = cut_capacity(&|v| (mask >> v) & 1 == 1);
            // Among min cuts, the one with the largest source side has the smallest sink
            // side: the nodes that can reach the sink in every max flow's residual graph.
            if capacity < best
                || (capacity == best && mask.count_ones() > minimal_sink_side.count_ones())
            {
                best = capacity;
                minimal_sink_side = mask;
            }
        }

        let flow = graph.compute();
        assert_eq!(flow, best);
        assert_eq!(
            cut_capacity(&|v| graph.is_connected_to_source(v).unwrap()),
            best
        );
        for v in 0..num_nodes {
            // Tier A: the sink side is the unique minimal one, whatever the flow.
            assert_eq!(
                graph.is_connected_to_sink(v).unwrap(),
                (minimal_sink_side >> v) & 1 == 0
            );
        }
    }
}

#[test]
fn rust_only_min_st_graph_cut_large_grid_with_terminals_on_every_node_is_not_quadratic() {
    // Delaunay meshing hands MinSTGraphCut millions of cells, each with terminal capacities.
    // Storing those as ordinary edges out of a terminal rescans all of them after every
    // augmentation (quadratic). Work is counted, not timed, so machine load cannot flake it.
    const SIDE: usize = 448; // 200,704 nodes
    let mut g = Mt19937::new(1);
    let uniform = |g: &mut Mt19937| generate_canonical::<f64>(g) as f32;
    let mut graph = MinSTGraphCut::<f32>::new(SIDE * SIDE);
    let mut source_capacities = vec![0f32; SIDE * SIDE];
    let mut sink_capacities = vec![0f32; SIDE * SIDE];
    for i in 0..SIDE * SIDE {
        source_capacities[i] = uniform(&mut g);
        sink_capacities[i] = uniform(&mut g);
        graph
            .add_node(i, source_capacities[i], sink_capacities[i])
            .unwrap();
    }

    let mut arcs: Vec<(usize, usize, f32)> = Vec::new();
    for y in 0..SIDE {
        for x in 0..SIDE {
            let i = y * SIDE + x;
            let mut neighbors = Vec::new();
            if x + 1 < SIDE {
                neighbors.push(i + 1);
            }
            if y + 1 < SIDE {
                neighbors.push(i + SIDE);
            }
            for j in neighbors {
                let forward = uniform(&mut g) * 0.5;
                let backward = uniform(&mut g) * 0.5;
                graph.add_edge(i, j, forward, backward).unwrap();
                arcs.push((i, j, forward));
                arcs.push((j, i, backward));
            }
        }
    }

    let flow = graph.compute();

    // The labels must describe a cut whose capacity is the flow (max-flow = min-cut).
    let mut cut_capacity = 0f64;
    for i in 0..SIDE * SIDE {
        cut_capacity += f64::from(if graph.is_connected_to_source(i).unwrap() {
            sink_capacities[i]
        } else {
            source_capacities[i]
        });
    }
    for &(from, to, capacity) in &arcs {
        if graph.is_connected_to_source(from).unwrap() && graph.is_connected_to_sink(to).unwrap() {
            cut_capacity += f64::from(capacity);
        }
    }

    // colmap-sharp measured ~3.4 steps per node and edge; the quadratic version took ~1e10.
    let bound = 10 * (graph.num_nodes() + graph.num_edges()) as u64;
    assert!(
        graph.last_compute_work() < bound,
        "{} >= {bound}",
        graph.last_compute_work()
    );
    let flow = f64::from(flow);
    assert!(
        (cut_capacity - flow).abs() < 1e-3 * flow.max(1.0),
        "{cut_capacity} vs {flow}"
    );
}

fn planted_clusters_case(num_clusters: i32, seed_offset: u32) {
    // Dense clusters of 30 vertices with heavy edges, joined in a ring by single light edges.
    // The balanced min cut is exactly the ring edges.
    const CLUSTER_SIZE: i32 = 30;
    let mut g = Mt19937::new(7 + num_clusters as u32 + 1000 * seed_offset);
    let mut edges = Vec::new();
    let mut weights = Vec::new();
    for c in 0..num_clusters {
        let offset = c * CLUSTER_SIZE;
        for i in 0..CLUSTER_SIZE {
            for j in i + 1..CLUSTER_SIZE {
                if next(&mut g, 0, 3) == 0 || j == i + 1 {
                    edges.push((offset + i, offset + j));
                    weights.push(next(&mut g, 20, 100));
                }
            }
        }
        edges.push((offset, ((c + 1) % num_clusters) * CLUSTER_SIZE + 1));
        weights.push(1);
    }

    let labels = compute_normalized_min_graph_cut(&edges, &weights, num_clusters).unwrap();
    assert_eq!(labels.len(), (num_clusters * CLUSTER_SIZE) as usize);
    let mut cluster_labels = BTreeSet::new();
    for c in 0..num_clusters {
        let label = labels[&(c * CLUSTER_SIZE)];
        cluster_labels.insert(label);
        for i in 0..CLUSTER_SIZE {
            assert_eq!(labels[&(c * CLUSTER_SIZE + i)], label);
        }
    }
    assert_eq!(cluster_labels.len(), num_clusters as usize);
}

#[test]
fn rust_only_normalized_min_graph_cut_planted_clusters_are_each_one_part() {
    for num_clusters in 2..=5 {
        // Ten random draws per cluster count.
        for seed_offset in 0..10 {
            planted_clusters_case(num_clusters, seed_offset);
        }
    }
}

#[test]
fn rust_only_normalized_min_graph_cut_random_graphs_parts_are_balanced_and_deterministic() {
    let mut g = Mt19937::new(4321);
    for num_parts in [1, 2, 3, 5, 8] {
        for _ in 0..5 {
            let num_vertices = next(&mut g, 100, 400);
            let mut edges = Vec::new();
            let mut weights = Vec::new();
            for v in 1..num_vertices {
                edges.push((next(&mut g, 0, v), v));
                weights.push(next(&mut g, 0, 50));
            }
            for _ in 0..3 * num_vertices {
                let a = next(&mut g, 0, num_vertices);
                let b = next(&mut g, 0, num_vertices);
                edges.push((a, b));
                weights.push(next(&mut g, 0, 50));
            }

            let labels = compute_normalized_min_graph_cut(&edges, &weights, num_parts).unwrap();
            let again = compute_normalized_min_graph_cut(&edges, &weights, num_parts).unwrap();
            let mut sizes = vec![0; num_parts as usize];
            for &label in labels.values() {
                sizes[label as usize] += 1;
            }

            // Each bisection allows 3% over its target (plus rounding up to a whole vertex),
            // compounded over at most three levels for eight parts.
            let max_size = (f64::from(num_vertices) / f64::from(num_parts) * 1.1).ceil() + 3.0;
            assert_eq!(labels.len(), num_vertices as usize);
            assert_eq!(again, labels);
            for size in sizes {
                assert!(size > 0);
                assert!(size <= max_size as i32, "{size} > {max_size}");
            }
        }
    }
}

#[test]
fn rust_only_normalized_min_graph_cut_grid_is_bisected_near_the_optimal_cut() {
    // A 100 x 40 grid of unit edges: the best balanced bisection cuts 40 edges.
    const WIDTH: i32 = 100;
    const HEIGHT: i32 = 40;
    let mut edges = Vec::new();
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            let v = y * WIDTH + x;
            if x + 1 < WIDTH {
                edges.push((v, v + 1));
            }
            if y + 1 < HEIGHT {
                edges.push((v, v + WIDTH));
            }
        }
    }
    let weights = vec![1; edges.len()];

    let labels = compute_normalized_min_graph_cut(&edges, &weights, 2).unwrap();
    let cut = edges.iter().filter(|(a, b)| labels[a] != labels[b]).count();

    // Allow 20% over the optimum (the Tier C bar). The partitioner is deterministic and the
    // input involves no random draws, so it must also cut exactly the 43 edges colmap-sharp's
    // partitioner cuts on the same graph: a different count is a bug in one of the two ports.
    assert!(cut <= 48, "cut {cut}");
    assert_eq!(cut, 43);
}

/// Graph shapes whose heavy-edge matching stalls. "star": one hub and n leaves. "hubs": n
/// leaves, each linked to 3 of 20 hubs. "pairs": n/2 disconnected edges.
fn make_graph(shape: &str, n: usize) -> (usize, Vec<(usize, usize, i32)>) {
    let mut edges = Vec::new();
    match shape {
        "star" => {
            for i in 1..=n {
                edges.push((0, i, 1 + (i % 7) as i32));
            }
            (n + 1, edges)
        }
        "hubs" => {
            for i in 0..n {
                let leaf = 20 + i;
                edges.push((i % 20, leaf, 1 + (i % 5) as i32));
                edges.push(((i + 7) % 20, leaf, 1 + (i % 3) as i32));
                edges.push(((i + 13) % 20, leaf, 2));
            }
            (n + 20, edges)
        }
        _ => {
            for i in (0..n.saturating_sub(1)).step_by(2) {
                edges.push((i, i + 1, 1 + (i % 4) as i32));
            }
            (n, edges)
        }
    }
}

fn to_csr(
    num_vertices: usize,
    edges: &[(usize, usize, i32)],
) -> (Vec<usize>, Vec<usize>, Vec<i32>) {
    let mut adjacency: Vec<Vec<(usize, i32)>> = vec![Vec::new(); num_vertices];
    for &(a, b, w) in edges {
        adjacency[a].push((b, w));
        adjacency[b].push((a, w));
    }
    let mut xadj = vec![0];
    let mut adjncy = Vec::new();
    let mut adjwgt = Vec::new();
    for neighbors in adjacency {
        for (u, w) in neighbors {
            adjncy.push(u);
            adjwgt.push(w);
        }
        xadj.push(adjncy.len());
    }
    (xadj, adjncy, adjwgt)
}

#[test]
fn rust_only_multilevel_partitioner_work_grows_near_linearly() {
    // Counts elementary steps rather than timing. Quadrupling n must cost well under the 16x
    // a quadratic step would.
    for shape in ["star", "hubs", "pairs"] {
        let work = |n: usize| {
            let (num_vertices, edges) = make_graph(shape, n);
            let (xadj, adjncy, adjwgt) = to_csr(num_vertices, &edges);
            partition_with_work(&xadj, &adjncy, &adjwgt, 2).1
        };
        let ratio = work(20000) as f64 / work(5000) as f64;
        assert!(ratio < 6.0, "{shape}: {ratio}");
    }
}

#[test]
fn rust_only_multilevel_partitioner_grow_region_skips_a_vertex_that_does_not_fit() {
    // Vertex 0 joins vertex 1 by a heavy edge and vertices 2 and 3 by light ones. Vertex 1
    // weighs 5, more than the grown side may hold, so growth must pass over it and take the
    // lighter vertices instead of stopping.
    let (xadj, adjncy, adjwgt) = to_csr(4, &[(0, 1, 10), (0, 2, 1), (0, 3, 1)]);
    let side = grow_region(&xadj, &adjncy, &adjwgt, &[1, 5, 1, 1], 0, 3.0, 3);
    assert_eq!(side, vec![0, 1, 0, 0]);
}
