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

## 280. Reading a truncated binary PLY mesh fails inside the texcoord lists

**What differs.** COLMAP's `ReadPlyMesh` (util/ply.cc) checks `file.good()` after every vertex,
face count and face index it reads from a binary file, but not after a face's texcoord count
or its six UVs. A file cut there leaves those values indeterminate: usually the count is not 6
and the read fails with "Expected 6 texture coordinates per triangular face", a cut inside a
middle face's UVs fails at the next face, and a cut inside the last face's UVs loads the mesh
with garbage UVs and no error. `colmap_rust::util::ply::read_ply_mesh` checks the count and
every UV and fails with "Unexpected end of PLY file at face i" for the face whose texcoords are
cut, in all three cases. Same as colmap-sharp entry 113.

**Why.** The same robustness gap as a short binary stream (`util/endian.rs` header): a
half-written mesh must not load with garbage texture coordinates, and the error should name the
cause and the face. Well-formed files read identically.

**Evidence.** Reading of ply.cc (the unchecked `file.read` calls for `num_texcoords` and `uv`).
`tests/util/rust_only_ply.rs` `rust_only_truncated_binary_texcoords_fail` cuts a one-face
textured mesh before the texcoord count and after two of its six UVs (the last-face case); the
ported `ply_test.cc` cases, which read complete files, pass unchanged.

## 281. The color extractors take images from the host instead of reading image files

**What differs.** COLMAP's `Reconstruction::ExtractColorsForImage(image_id, path)` and
`ExtractColorsForAllImages(path, num_threads)` read each image from `path / image.Name()` with
`Bitmap::Read(..., as_rgb = true)` (OpenImageIO); a file that cannot be read makes the first
return `false` and is skipped (with a log warning) by the second.
`colmap_rust::scene::reconstruction_ply` takes a `load_image(name) -> Option<Bitmap>` callback
instead: `None` is the unreadable file. `extract_colors_for_all_images` has no `num_threads`
and accumulates each point's color sum in registered-image order, which is COLMAP's result
with one thread (with more, COLMAP adds per-thread partial sums in a scheduling-dependent
order, which can move a mean that lands near a rounding tie).

**Why.** Image file decoding is not ported: the host owns image I/O, natively and in the
browser (as for `SynthesizeImages`, whose doc in `scene/synthetic_noise.rs` holds that
divergence). The core is single-threaded (CLAUDE.md, Threading).

**Evidence.** `tests/scene/reconstruction_ply.rs` `reconstruction_extract_colors_for_all_images`
(COLMAP's test with the deleted image file as a missing map entry) and
`tests/scene/rust_only_reconstruction_ply.rs`
`rust_only_extract_colors_for_image_colors_black_points_only`.

## 282. PoseGraph orders equally large frame components by smallest frame id

**What differs.** COLMAP's `PoseGraph::ConnectedFrameComponents` collects the frames of the
valid edges in a `FlatHashSet`, finds the components (whose order follows that set, entry 42)
and orders them with an unstable `std::sort` by descending size, so the order of equally large
components is unspecified. `LargestConnectedFrameComponent` keeps the first strictly largest
component in the same hash order. `colmap_rust::scene::pose_graph::PoseGraph` hands
`find_connected_components` the frames in ascending id order and sorts stably, so equally large
components come out by their smallest frame id, and `largest_connected_frame_component` returns
that same first one. `connected_image_ids_for_frame_components` follows the same component
order. The edges live in a `BTreeMap` (ascending pair id) where COLMAP's `NodeHashMap` has its
hash order; no output of the pose graph itself depends on it. Same as colmap-sharp entry 38.

**Why.** CLAUDE.md's stable-sort and hash-order rules: the tie order decides which component
global SfM treats as the largest. COLMAP relies on no particular order: `pose_graph_test.cc`
uses components of different sizes and compares sets.

**Evidence.** `tests/scene/pose_graph.rs` (pose_graph_test.cc 1:1) passes with COLMAP's
expectations; `tests/scene/rust_only_pose_graph.rs`
`rust_only_equal_size_components_ordered_by_smallest_frame_id` pins the tie order and that the
largest component equals the first one.

## 283. ReconstructionManager::write breaks point-count ties by index

**What differs.** COLMAP's `ReconstructionManager::Write` orders the models with `std::sort` by
descending `NumPoints3D` and writes them to `0/`, `1/`, ...; the order of models with the same
point count is unspecified. `colmap_rust::scene::reconstruction_manager::ReconstructionManager`
sorts stably, so equal-sized models keep their manager order. Same as colmap-sharp entry 34.

**Why.** CLAUDE.md's stable-sort rule: the tie order decides which folder a model lands in.
libc++'s `std::sort` uses insertion sort for short ranges, which also keeps equal elements in
order, so for the handful of models a mapper produces the result is the same as COLMAP's; only
a long run of equal counts could differ.

**Evidence.** `tests/scene/rust_only_reconstruction_manager.rs`
`rust_only_write_orders_by_point_count_then_index` writes models with point counts 1, 2, 1, 2
and reads back sources 1, 3, 0, 2.

## 284. Scene and reconstruction clustering break sort ties deterministically

**What differs.** Three `std::sort` calls leave equal keys in an unspecified order:
`SceneClustering::PartitionHierarchicalCluster` sorts the overlap-candidate edges of each
child by descending weight, `SceneClustering::PartitionFlatCluster` sorts each image's
related images by descending weight (scene_clustering.cc), and `EstablishStrongClusters`
(reconstruction_clustering.cc) sorts the union-find clusters by descending size after
collecting them from a `NodeHashMap`, so equal-size clusters are numbered in hash order.
`colmap_rust::scene::scene_clustering` keeps equal weights in edge order (a stable sort), and
`colmap_rust::scene::reconstruction_clustering` numbers equal-size clusters by ascending
smallest frame id. When a weight tie straddles the overlap budget, a different overlap image
can be chosen; equal-size reconstruction clusters can get swapped ids. Same as colmap-sharp
entry 93.

**Why.** The translation rules require an explicit, deterministic tie-break wherever tie
order reaches an output. Edge order is what the macOS SDK's libc++ `std::sort` gives for short
lists: lengths 2-5 go through `__sort3/4/5`, which never reorder equal elements under these
`>` comparators; lengths 6-23 go through insertion sort, which is stable; lengths of 24 or more
go through pdqsort, which is not. So an overlap list or related-image list of 24+ entries with
tied weights can come out in a different order than COLMAP's; every such list in COLMAP's tests
has at most 12 entries. Reproducing pdqsort and absl's or Boost's hash order would tie the
result to one library version.

**Evidence.** All eight `scene_clustering_test.cc` cases (`tests/scene/scene_clustering.rs`)
and all nine `reconstruction_clustering_test.cc` cases
(`tests/scene/reconstruction_clustering.rs`) pass;
`tests/scene/rust_only_reconstruction_clustering.rs`
`rust_only_equal_size_clusters_numbered_by_smallest_frame_id` pins the cluster numbering (and
fails when the tie-break is reversed).

## 290. Flat scene clusters are ordered by size, then smallest image id

**What differs.** `SceneClustering::PartitionFlatCluster` sorts the child clusters with the
comparator `size(a) >= size(b) && min(a) < min(b)`, which is not a strict weak ordering
(`std::sort` then has undefined behavior) and dereferences `min_element` of an empty cluster.
`colmap_rust::scene::scene_clustering` orders the children by what COLMAP's comment says it
intends: descending size, then ascending smallest image id, with empty clusters last (stable
among themselves). Only the order of `root_cluster().child_clusters` / `leaf_clusters()` is
affected; each child's overlap images depend only on its own members. In practice the orders do
differ: with 3-5 children libc++'s `__sort3/4/5` only swap when the comparator says so, so a
child of size 2 with smallest id 1 stays ahead of a child of size 3 with smallest id 5 (neither
compares "less" than the other), while the port puts the larger child first. The flat overlap
loop also stops once `j` passes the longest related-image list of the child, where COLMAP keeps
counting up to `num_image_matches` (a negative value, "no limit" as `size_t`, would spin for
2^64 rounds); those rounds cannot add an image, so the result is the same. Same as colmap-sharp
entry 94.

**Why.** Undefined behavior cannot be ported; the comment states the intent.

**Evidence.** `tests/scene/rust_only_scene_clustering.rs`
`rust_only_flat_child_clusters_ordered_by_size_then_smallest_id` pins the order;
`scene_clustering_three_flat_clusters` and `_three_flat_clusters_two_overlap` (1:1,
order-insensitive as in COLMAP) pass.

## 291. SceneClustering::create hands the image pairs over in Boost hash order

**What differs.** COLMAP's `SceneClustering::Create` builds the edge list by iterating the
`NodeHashMap` from `NumMatchesBetweenAllImages()`, so the edges come in hash order.
`colmap_rust::scene::scene_clustering::SceneClustering::create` models the order of COLMAP's
default backend, which CMake picks when Boost >= 1.84 is found (`cmake/FindDependencies.cmake`):
`boost::unordered_node_map` with the identity `std::hash<uint64_t>`. Boost marks that hash
non-avalanching (`hash_traits.hpp`), so its table mixes it with `mulx(h) = lo ^ hi` of
`h * 0x9E3779B97F4A7C15` (`detail/mulx.hpp`, ported under BSL-1.0, `THIRD_PARTY_NOTICES.md`),
puts the key in group `mix >> size_index` and iterates groups in ascending order. The port
orders the pairs by that mixed value (then by pair id) instead of the correspondence graph's
order, which is ascending pair id. This is Boost's order up to the order of the keys that share
a group (Boost keeps insertion order within a group and moves overflow to later groups), which
depends on the table size and the insertion history. Same as colmap-sharp entry 120.

**Why.** The edge order decides every weight tie: the vertex numbering of the graph cut
(entry 46) and which overlap images each child cluster gets (entry 284). COLMAP relies, without
saying so, on that order being unrelated to image ids. In ascending order every tie breaks
toward the same few low image ids, and a database numbers the images of one rig frame
consecutively, so sibling clusters can overlap in a single frame. A zero-baseline rig's images
of one frame share a projection center, so the two halves have too few distinct centers in
common to estimate the similarity that merges them (colmap-sharp's hierarchical pipeline test
on a panoramic rig ended with two reconstructions until this order). Modelling the default
Boost backend reproduces the property, and most of the order, and stays deterministic.

**Evidence.** Tier C. `tests/scene/rust_only_scene_clustering.rs`
`rust_only_create_spreads_tied_overlap_across_frames` gets at least three shared frames with
the Boost order and fails with the ascending order (two shared frames, 1 and 11). The ported
`scene_clustering_test.cc` cases call `partition` directly and are unaffected.

## 295. The covariant detector rejects an image too small for its octaves

**What differs.** `VlCovDet::put_image` (`feature/vlfeat/covdet.rs`) returns `Err` for an image
whose shorter side is below 16 pixels with the default first octave (-1), or too small for the
configured first octave; VLFeat's `vl_covdet_put_image` goes on: a 1-pixel side converts
`log2(0) = -inf` to an integer, and with a last octave below 0 `vl_scalespace_put_image`
upsamples octave -1 from an octave 0 the scale space never allocated (`scalespace.rs`'s
`put_image` checks this). It also returns `Err` for an image buffer of the wrong length, which
VLFeat cannot check.

**Why.** VLFeat's behaviour there is undefined (out-of-bounds writes), so there is no result to
match; an error the caller can report is the safe equivalent. COLMAP's covariant extractor
never gets there: it rejects such images first (colmap-sharp's `CovariantSift.cs` throws for a
side below `MinimumImageSide`), so no COLMAP result is lost.

**Evidence.** `tests/feature/rust_only_vl_covdet.rs` `rust_only_vl_covdet_rejects_bad_input`.

## 300. The small dense kernels use Ceres' naive loops for every shape

**What differs.** Ceres' `small_blas.h` products (`MatrixMatrixMultiply`,
`MatrixTransposeMatrixMultiply`, `MatrixVectorMultiply`, `MatrixTransposeVectorMultiply`) use
Eigen's fixed-size kernels when all sizes are compile-time constants (the template
specializations the Schur eliminator picks from the Jacobian's static block sizes, for COLMAP's
bundle adjustment typically `<2, 3, 6>`) and the "Naive" loops otherwise.
`colmap_rust::solver::small_blas` always computes each output element as a sum over k in
increasing order into a temporary that starts at 0.0, then stores it with the operation
(=, +=, -=): the naive loops' rounding. The bundle adjustment shapes (2 or 3 inner rows,
inner dimension 3) go through `small_blas_fixed.rs`, whose const-generic kernels unroll the k
loop but keep that order, so every shape rounds exactly as the naive loops do. Eigen's
fixed-size products may sum in another order (vectorized pairs), so results agree with Ceres'
to rounding, not bit for bit. Same as colmap-sharp entry 35 (item 2, the kernel part).

**Why.** Eigen is replaced rather than ported (CLAUDE.md contract 2). One summation order for
every shape makes the Schur solvers' results independent of which kernel a shape dispatches
to, which the later determinism tests rely on. Bundle adjustment is Tier C.

**Evidence.** `tests/solver/rust_only_small_blas.rs` (Ceres' `small_blas_test.cc` via
colmap-sharp: every product, operation, stride and start offset against MatrixXd within
5 * epsilon) and `tests/solver/rust_only_small_blas_fixed.rs` (the fixed kernels give the
naive loops' bits for every operation, with negative zeros and strided, offset output blocks;
reversing the fixed kernels' k order makes it fail).

## 301. Inverting a PSD block that is not positive definite gives NaN

**What differs.** Ceres' `InvertPSDMatrix` (assume_full_rank = true) is
`selfadjointView<Upper>().llt().solve(Identity)`; for a block that is not positive definite,
Eigen's LLT stops and the solve returns unspecified values. `small_blas::invert_upper_psd`
(and `_in_place`, which `BlockRandomAccessDiagonalMatrix::invert` uses for the SCHUR_JACOBI
preconditioner, and which the Schur eliminator will use for E'E) fills the whole inverse with
NaN as soon as a Cholesky pivot is not positive or is NaN. Same as colmap-sharp entry 35
(item 5).

**Why.** Eigen's partial result cannot be reproduced without porting Eigen, and it is garbage
either way. NaN makes the linear solve, and so the trust-region step, invalid; the minimizer
then rejects the step and shrinks the radius, which is what Ceres' minimizer does when the
garbage happens to be non-finite.

**Evidence.** `tests/solver/rust_only_sparse_matrix.rs`
`rust_only_invert_upper_psd_matches_llt_and_rejects_indefinite` (an indefinite 2x2 block
gives all NaN; an SPD block with garbage below the diagonal matches `linalg::Llt` within
1e-12, in place bit-identical to out of place). The ported
`BlockRandomAccessDiagonalMatrixTest.Invert` checks the positive definite case at 1e-14.

## 305. The Ceres manifolds and polynomial balancing sum norms left to right

**What differs.** Ceres evaluates the norms in its manifolds through Eigen
(`delta.norm()`, `x.norm()`, the Householder vector's `squaredNorm()` and dot product in
`sphere_manifold_functions.h` / `householder_vector.h`) and the 1-norms of
`BalanceCompanionMatrix` through `lpNorm<1>()`, all of which Eigen may vectorize and reduce in
packet lanes. `colmap_rust::solver::{sphere_manifold, ceres_polynomial}` sum the squares and
absolute values left to right in coefficient order. The quaternion manifolds' norms are
`std::hypot`'s three-argument form, which the port follows exactly (`math::fns::hypot3`). Results
can differ from COLMAP's Ceres in the last bits. Tier C for the manifolds, like the solver that
consumes them; Tier B for the polynomial roots, which already go through `linalg::EigenSolver`.

**Why.** Eigen is MPL-2.0 and not ported (contract rule 2), and its packet order depends on the
target's SIMD width and the compiler, so there is no single order to match. A sequential order
gives the same bits on every target, native and wasm. Same as colmap-sharp entry 115 (its
`Solver/Manifolds.cs` part).

**Evidence.** `tests/solver/rust_only_manifold.rs` (colmap-sharp's `ManifoldTests`, plus the
sphere's poles) checks Plus/Minus round trips to 1e-12 and the Jacobians against finite
differences to 1e-8; `tests/solver/ceres_polynomial.rs` (Ceres' `polynomial_test.cc`) passes at
Ceres' own tolerances.

## 306. A Problem owns its parameter values and names blocks by id

**What differs.** Ceres' `Problem` keys a parameter block by the user's `double*`, reads the
user's memory during evaluation and writes the solution back into it. `colmap_rust::solver::Problem`
copies the values into one arena when a block is added (`add_parameter_block` returns a
`ParameterBlockId`), evaluates and solves on that arena, and the caller reads the result back
with `Problem::parameter_block`. Every other call that takes a `double*` in Ceres takes the id.
Consequences: blocks cannot alias or be re-added with another size, so Ceres' aliasing and
same-pointer checks do not exist; `AddResidualBlock` does not add unknown blocks on the fly (a
residual block names ids the problem handed out, an unknown id is an error);
`GetParameterBlocks` lists blocks in the order they were added, not in address order (the one
COLMAP caller, `GetOtherParams` in `estimators/covariance.cc`, uses that order only to lay out
the factorization; same as colmap-sharp entry 53); messages that print a block's pointer print
its id. Cost functions, losses and manifolds are `Arc`s, so the ownership options
(`Problem::Options::*_ownership`) have nothing to control. Where Ceres CHECK-fails or
`LOG(FATAL)`s (unknown block, duplicate blocks, size mismatch, a manifold of the wrong size),
the call returns the error with Ceres' message. Block removal, `EvaluateResidualBlock` and the
evaluation callback are not ported: COLMAP never calls them.

**Why.** Rust gives a block no stable address to key on without `unsafe`, and a problem that
borrowed the caller's arrays would need every block borrowed mutably for the problem's whole
life, which COLMAP's shared camera and rig blocks rule out. Owning the values costs one copy in
and one copy out per solve (Ceres copies them into its state vector anyway) and keeps the
solver free of lifetimes. Insertion order is deterministic, where address order is an accident
of the allocator.

**Evidence.** `tests/solver/problem.rs` (Ceres' `problem_test.cc`, the cases that apply; the
skipped ones are listed in its header with their reason) and `tests/solver/problem_evaluate.rs`
(`ProblemEvaluateTest`, exact values) pass.

## 307. A ParameterBlockOrdering lists a group's blocks in the order they were added to the problem

**What differs.** Ceres' `ParameterBlockOrdering` (`OrderedGroups<double*>`) keeps each group
in a `std::set<double*>`, so a group's blocks come out in heap-address order, and the Schur
solvers lay out the eliminated blocks (and the reduced camera system's columns) in that order.
`colmap_rust::solver::OrderedGroups` is Ceres' class over `ParameterBlockId`, so a group's
blocks come out in id order: the order they were added to the problem. Groups are still visited
in increasing id order, and every other rule (moving an element, `Reverse`'s numbering, the
counts) is Ceres'. colmap-sharp keeps each group in the order its blocks were added to the
group (its entry 36); id order is the same wherever a caller adds blocks to the problem and to
the ordering in the same order, as COLMAP's `GlobalPositioner` does.

**Why.** Address order is an accident of the allocator, so no run-to-run order exists to match;
the ids make Ceres' own `std::set` rule deterministic. The effect on results is rounding-level
(the order of additions into the reduced system), within the Tier C outcome bar.

**Evidence.** `tests/solver/ordered_groups.rs` (Ceres' `ordered_groups_test.cc`, unchanged
values) passes; none of its cases depends on the order inside a group.
