# Deliberate divergences from COLMAP

Each entry: what differs, why, and the evidence. Numbered so code comments can cite them
(`docs/CPP_DIVERGENCES.md`, entry N). Remove an entry when the divergence is gone.

Numbers are stable: an entry keeps its number for good, and a removed entry's number is not
reused, so gaps in the sequence are intentional. Never renumber, because code comments and open
branches cite these numbers.

colmap-sharp's `docs/CPP_DIVERGENCES.md` documents ~136 divergences it took. Most will recur here
for the same reason (a replaced dependency, a deterministic tie-break). When the Rust port takes
the same divergence, write a full entry here (don't just point there) and add
"Same as colmap-sharp entry N." so the two can be cross-checked.

## 1. Transcendentals use the `libm` crate, not the platform libm

**What differs.** COLMAP calls `std::sin`, `std::exp`, `std::log`, `std::atan2`, `std::pow`
and friends (and their `float` overloads), which resolve to the platform's C libm; the pycolmap
oracle runs on macOS, so that is Apple libm. colmap-rust routes every transcendental through
`colmap_rust::math::fns`, which calls the pure-Rust `libm` crate (a port of musl's libm).
Results can differ from COLMAP/pycolmap on macOS by 1-2 ulp for the same input.

**Why.** Tier A code must give the same bits natively and in the browser. Rust's std calls the
platform libm natively (Apple libm, glibc, UCRT — all different) and compiler-builtins' musl
port on `wasm32-unknown-unknown`, so std results depend on the target. The `libm` crate is the
same code everywhere. Apple libm is closed source and cannot be reproduced in pure Rust, so
no backend could match the oracle bit for bit anyway; being identical across our own targets
is the property we can have.

**Evidence.** The probe `colmap-rust/tests/math/fns_probe.rs` (7154 f64 probes of sin, cos,
tan, asin, acos, atan, atan2, exp, log, log2, log10, pow, sqrt, cbrt, hypot; special values
and seeded random inputs in COLMAP's ranges), run on macOS aarch64 and on a
`wasm32-unknown-unknown` build under node:
- std on wasm32 vs std on macOS (Apple libm): 490 differ (sin 10, cos 14, tan 101, asin 15,
  acos 45, atan 17, atan2 126, exp 15, log 5, log2 2, log10 2, pow 30, cbrt 18, hypot 90;
  sqrt 0), 1 ulp except tan (up to 2 ulp).
- `libm` crate vs Apple libm: 435 differ (same pattern; hypot 35).
- `libm` crate native aarch64 vs wasm32: 0 differ, for the f64 table and for the 8654-probe f32
  table (`sinf` ... `hypotf`).

The probe tables (`colmap-rust/tests/data/fns_probe_expected{,_f32}.txt`) now pin the `libm`
crate's bits; CI runs them on Linux, macOS and Windows.

**Consequence for tests.** A Tier A oracle comparison of code that calls a transcendental
(camera-model undistortion, SIFT's Gaussian weights, angle conversions) may need a tolerance of
a few ulp instead of bit equality; such a test must cite this entry where it states that
tolerance. Code without transcendentals stays bit-exact.

## 40. Real-valued random draws are not fused (no FMA), unlike the macOS pycolmap wheel

**What differs.** `math::random::random_uniform_real` (for `min != 0`), `random_gaussian` and
`libcxx::NormalDistribution` evaluate libc++'s `(b - a) * u + a`, `u*u + v*v` and
`x * stddev + mean` as separate multiply and add. Apple clang's default `-ffp-contract=on`
fuses these into single-rounding FMAs on arm64, so a libc++ build with default flags can
differ from colmap-rust in the last ulp of those draws. Integer draws, `random_uniform_real`
with `min == 0`, the shuffles and the mt19937 words are unaffected.

**Why.** CLAUDE.md's "No FMA" rule: results must be identical on every platform, and Rust
never contracts, while contraction in the C++ build is a compiler and flags choice, not COLMAP
behavior. Matching it would mean `mul_add` at exactly the sites a particular clang version
chose to fuse.

**Evidence.** `oracle/fixture_random.py` builds `oracle/random_harness.cc` with
`-ffp-contract=off` and with the default; the draws above differ between the two builds
(recorded as `contracted_cases` in `colmap-rust/tests/data/oracle/random.json`), and
colmap-rust matches the `-ffp-contract=off` build bit for bit
(`tests/math/rust_only_random_oracle.rs`; the f64 Gaussian cases within 2e-15 because of
entry 1's `log`, bit for bit when `log` is Apple libm's). The pycolmap 4.2.0 wheel's `_core`
disassembles to about 36,000 `fmadd`/`fmsub` family instructions, so it is built with
contraction on and most likely produces the contracted numbers. Downstream, a pycolmap fixture
that depends on real-valued draws (e.g. `synthesize_dataset` noise) can therefore differ in
the last ulp and is compared at Tier B or C, not Tier A.

Same as colmap-sharp entry 1.

## 41. Percentile selection with NaN or tied signed zeros

**What differs.** `math::percentile` (and `median`, `median_absolute_deviation`) select with
`slice::select_nth_unstable_by` under `partial_cmp` (NaN compares equal to everything) instead
of libc++'s `std::nth_element` under `operator<`. For ordinary input the selected order
statistics, and so the result, are identical. With NaN in the data, or -0.0 and +0.0 tied at
the selected rank, the element picked (a NaN, or the other signed zero) can differ; the order
left in the slice differs too, but it is unspecified in C++ as well.

**Why.** `nth_element`'s partition order is libc++'s introselect detail, not COLMAP's
contract; the value depends only on the order statistics for totally ordered input. NaN input
has no meaningful percentile.

**Evidence.** `tests/math/math.rs` (math_test.cc's `Percentile`, `Median` and
`MedianAbsoluteDeviation` cases, including unsorted inputs) passes with exact equality.

Same as colmap-sharp's documented `Percentile` caveat (its `MathUtils.cs` header).

## 42. Iteration order of UnionFind parents and connected components

**What differs.** `math::union_find::UnionFind::parents` iterates in insertion order.
`find_connected_components` returns components ordered by their first node in the caller's
`nodes` slice, each listing its nodes in that order, and `find_largest_connected_component`
breaks ties between equally large components the same way. COLMAP's orders come from
`boost::unordered_node_map` / `unordered_flat_set` (`colmap/util/hash_containers.h`), and it
takes the nodes as a `FlatHashSet` where the port takes a slice.

**Why.** Reproducing them would mean porting Boost.Unordered's bucket layout and `boost::hash`
mixing for every key type. The order is not part of COLMAP's contract:
`connected_components_test.cc` compares with `UnorderedElementsAre`, and `union_find_test.cc`
never inspects it. Insertion order is deterministic and the same on every platform (a
std `HashMap` is used only for lookups, never iterated).

**Evidence.** The component sets, and every root `find` returns for the same call sequence,
are identical (`tests/math/union_find.rs`, `tests/math/connected_components.rs`, ported 1:1).
`rust_only_union_find_parents_in_insertion_order` and
`rust_only_connected_components_follow_node_order` pin the orders. Callers that need a
reproducible component order pass nodes in a deterministic order (for example sorted ids).

Same as colmap-sharp entry 2.

## 43. Spanning-tree ties between equal edge weights, and input checks

**What differs.** `compute_maximum_spanning_tree` / `compute_minimum_spanning_tree` run
Kruskal with edges stably sorted by cost, so among equal costs the earlier input edge wins.
COLMAP calls `boost::kruskal_minimum_spanning_tree`, which pops edges from a
`std::priority_queue`, so among equal costs it takes them in heap order; with tied weights the
two can choose different (equally optimal) trees. The port also returns a
`Result`: an edge endpoint outside `0..num_nodes`, a root outside it, or `edges` and `weights`
of different lengths fail a check, where Boost's `add_edge` would grow the vertex set and
COLMAP would then index past its adjacency list (undefined behavior).

**Why.** Boost is not ported; the input-index tie-break is the deterministic, documented rule.
The cost transform itself (`max_weight - w` in `f32`, `max_weight` starting at 0) is kept, so
the same weights tie on both sides.

**Evidence.** With distinct costs the minimum spanning tree is unique and the parents match
exactly (`tests/math/spanning_tree.rs`, spanning_tree_test.cc 1:1).
`rust_only_spanning_tree_ties_take_earlier_edge` and
`rust_only_spanning_tree_rejects_out_of_range_input` pin the tie rule and the checks.

Same as colmap-sharp entry 3.

## 44. Stoer-Wagner: which min cut, and which side is labeled 1

**What differs.** `compute_min_graph_cut_stoer_wagner` (`colmap-rust/src/math/graph_cut.rs`)
is written from Stoer & Wagner (JACM 1997) instead of calling `boost::stoer_wagner_min_cut`.
The cut weight is the global minimum on both sides, but when several cuts share it the one
reported can differ, and the side labeled 1 is the set of vertices merged into the last vertex
of the best phase (Boost's parity map labels its own choice). Each phase's queue takes the
largest key, then the lowest vertex index. The port returns `(cut_weight, cut_labels)` instead
of filling out-parameters, and checks for at least two vertices where Boost throws `bad_graph`.

**Why.** Boost is not ported (`docs/LICENSE_AUDIT.md`). COLMAP's contract is the minimum weight
plus a 0/1 label per vertex; `graph_cut_test.cc` checks only the weight and the label range.

**Evidence.** The four ported `graph_cut_compute_min_graph_cut_stoer_wagner*` cases pass
(`tests/math/graph_cut.rs`), and `rust_only_stoer_wagner_matches_exhaustive_min_cut`
(`tests/math/rust_only_graph_cut.rs`) compares the weight and the labeled cut against exhaustive
search on 200 random graphs.

Same as colmap-sharp entry 4.

## 45. Boykov-Kolmogorov max-flow with float capacities

**What differs.** `MinSTGraphCut` (`colmap-rust/src/math/graph_cut_min_st.rs`) is a port of
colmap-sharp's implementation, written from Boykov & Kolmogorov (PAMI 2004), instead of calling
`boost::boykov_kolmogorov_max_flow`, and it pushes each node's direct source -> node -> sink
flow before the search starts. Both change the order in which flow is augmented. With integer
capacities that order cannot show in the result; with float capacities (`MinSTGraphCut<f32>`,
as Delaunay meshing uses it) the returned flow is summed in a different order, so it can differ
from COLMAP's by rounding, and a node whose residual path to a terminal is only a rounding
residue can be labeled on the other side of the cut. `is_connected_to_source` /
`is_connected_to_sink` return a `Result`, failing a check for an index out of range or before
`compute`, where COLMAP's `colors_.at()` throws `std::out_of_range`.

**Why.** Boost is not ported (`docs/LICENSE_AUDIT.md`), and Kolmogorov's own maxflow library is
GPL/research-only, so neither is transcribed. Reproducing Boost's exact float rounding would
mean reproducing its exact augmentation order. The terminal-capacity handling is needed for
scale: storing terminal links as ordinary terminal out-edges made each augmentation rescan
them, which was quadratic on Delaunay-sized graphs.

**Evidence.** The three ported `graph_cut_min_st_graph_cut*` cases pass. For integer
capacities the result is Tier A: `rust_only_min_st_graph_cut_matches_exhaustive_min_cut`
compares the flow and the labels with exhaustive search on 300 random graphs, and the sink-side
labels equal the unique minimal sink-side min cut, which is the same for every maximum flow and
so for Boost too. `rust_only_min_st_graph_cut_large_grid_with_terminals_on_every_node_is_not_quadratic`
checks, for a 200k-node float grid, that the labeled cut's capacity equals the returned flow
within 1e-3 relative and that the solver's step count stays under 10 per node and edge.

Same as colmap-sharp entry 5.

## 46. ComputeNormalizedMinGraphCut partitions with our own multilevel bisection, not METIS

**What differs.** COLMAP's `ComputeNormalizedMinGraphCut` (math/graph_cut.cc) calls
`METIS_PartGraphKway` with default options. The port
(`colmap-rust/src/math/graph_cut.rs`) builds the same CSR graph (vertex indices by first
appearance, parallel edges kept) and hands it to `colmap-rust/src/math/graph_cut_partitioner.rs`,
a port of colmap-sharp's `MultilevelPartitioner.cs`, which was written from the published
multilevel scheme (Hendrickson & Leland 1995; Karypis & Kumar, SIAM J. Sci. Comput. 1998;
Fiduccia & Mattheyses 1982): heavy-edge-matching coarsening (followed, when over 10% of the
vertices stay unmatched, by pairing unmatched vertices that share a neighbor and unmatched
isolated vertices, so stars, hub images and many small components still coarsen), greedy graph
growing from eight seeds on the coarsest graph (a vertex too heavy to fit is skipped), FM
refinement at every level, and recursive bisection for k parts (floor(k/2) parts on the first
side). METIS's k-way path instead refines all k parts at once with its own greedy k-way
refinement and randomizes its visit orders with GKlib's RNG. So the labels, which part gets
which number, and the exact cut can differ from COLMAP's. Each bisection allows 3% over its
target weight (METIS's k-way `ufactor` default of 30), rounded up to a whole vertex so that
small graphs can always be split. Every tie is broken by vertex index, so the output is
deterministic. Self-loops are ignored (they never cross a cut). The result is a
`BTreeMap<i32, i32>` (vertex id -> label) instead of COLMAP's `NodeHashMap<int, int>`, so
iterating it is deterministic.

**Why.** Porting the reached METIS subset (coarsening, recursive-bisection initial
partitioning, 2-way and k-way FM refinement, balancing, GKlib's priority queues and RNG) is
large, and a close match would also need GKlib's random stream reproduced exactly. COLMAP's
contract, and all its callers need (scene clustering), is a balanced partition with a small
cut; graph_cut_test.cc checks the label range, that both parts are used, and the component
split of a disconnected graph. No METIS code was read or transcribed, so METIS's notice is not
needed.

**Evidence.** Tier C. The four ported `graph_cut_compute_normalized_min_graph_cut*` cases pass
(`tests/math/graph_cut.rs`). `tests/math/rust_only_graph_cut.rs` checks that planted clusters
(2-5 dense clusters in a ring of light edges, ten random draws each) come out one part per
cluster; that random graphs of 100-400 vertices split into 1, 2, 3, 5 and 8 parts give
non-empty parts within 10% (plus three vertices) of n/k and identical output on a second call;
that a 100 x 40 unit grid is bisected with 43 cut edges against an optimum of 40 — exactly
colmap-sharp's count on the same graph; and that the partitioner's step count grows less than
6x from n = 5k to 20k on a star, a 20-hub graph and disconnected pairs (a quadratic step would
give 16x).

Same as colmap-sharp entry 77.
