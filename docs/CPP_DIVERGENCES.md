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

## 2. No FMA contraction in the quaternion-vector rotation (the macOS pycolmap wheel fuses it)

**What differs.** `Quaterniond * Vector3d` (Eigen's quaternion-vector rotation,
`colmap-rust/src/linalg/quaternion.rs`) differs from the pycolmap 4.2.0 macOS arm64 wheel in
the last bits on about half of the inputs. Code built on it (the Rigid3d/Sim3d point
transform, the translations of composition and inverse) inherits the difference when it is
ported.

**Why.** The wheel is built with floating-point contraction on and evaluates the cross
products inside the rotation, `a1*b2 - a2*b1`, as `fma(a1, b2, -(a2*b1))`. colmap-rust never
uses FMA in math paths (CLAUDE.md, "No FMA"), so its results are the same on every target;
they are expected to match a C++ build that does not contract. Same as colmap-sharp entry 6.

**Evidence.** `oracle/linear_algebra_rotations.py` prints it: re-deriving `q * v` with our
formula gives 69/138 mismatching cases against the wheel with plain cross products and 0/138
with the cross products fused as above. `tests/linalg/rust_only_rotation_oracle.rs`
(`rust_only_tolerance_fields`, field `rotated`) pins the Rust result at 1e-14 relative; the
product, norm, inverse and matrix conversions in the same fixture are bit-identical
(`rust_only_exact_fields`).

## 3. sin(a/2) in the angle-axis to quaternion conversion

**What differs.** `AngleAxisd::to_quaternion` / `Quaterniond::from_angle_axis`
(`colmap-rust/src/linalg/quaternion.rs`) can differ from the wheel by 1-2 ulp in any
coefficient.

**Why.** Two causes. Our `sin`/`cos` are the `libm` crate's (entry 1), which differ from Apple
libm by an ulp on some inputs. And even Apple libm's `sin` does not always reproduce the
wheel's `sin(a/2)`: on some inputs the wheel rounds one ulp away from it. That cause is not
established; one hypothesis is that the compiler fused the adjacent `sin` and `cos` of the same
argument into a `sincos` call that rounds differently. We do not emulate a compiler's choice of
math routine. Same as colmap-sharp entry 7, plus entry 1.

**Evidence.** On the 138 cases of `tests/data/oracle/linear_algebra_rotations.json`, the
quaternion coefficients differ from the wheel on 15 coefficients with `fns::sin`/`fns::cos`
(at most 2 ulp) and on 3 with std's (Apple libm) `sin`/`cos`; the oracle script shows the
Apple-libm mismatch disappears when `sin(a/2)` moves one ulp.
`tests/linalg/rust_only_rotation_oracle.rs` (`rust_only_tolerance_fields`, field
`from_axis_angle`) pins it at 1e-14 relative.

## 4. from_two_vectors handles nearly opposite vectors with its own half-turn construction

**What differs.** `Quaterniond::from_two_vectors` (`colmap-rust/src/linalg/quaternion.rs`,
the replacement for Eigen's `Quaternion::FromTwoVectors`, which COLMAP's `SynthesizeDataset`
uses to aim frames) uses Melax's shortest-arc formula like Eigen does in general, but when the
two directions are nearly opposite (1 + c < 1e-8, c the cosine between them) it composes a
half turn about an axis perpendicular to the first vector (built from the least-aligned
coordinate axis) with the well-conditioned short arc from the negated first vector to the
second. Eigen switches branch at a different threshold (1 + c < 1e-12) and picks its
perpendicular axis another way, so for 1 + c < 1e-8 the returned rotation can differ from
COLMAP's: for exactly opposite vectors any half turn about a perpendicular axis is correct and
the two libraries pick different ones; for nearly opposite vectors both map the first
direction onto the second, but COLMAP's general formula there carries errors up to ~1e-8 that
ours does not.

**Why.** Eigen is MPL-2.0 and not ported (contract rule 2), so this branch is written from
first principles; the threshold is where Melax's formula loses more than ~5e-9 relative
accuracy (s = sqrt(2 (1 + c)) with 1 + c known only to ~1e-16 absolute). The general branch,
where all practical inputs land, is unchanged. Same as colmap-sharp entry 28.

**Evidence.** `tests/linalg/rust_only_quaternion.rs` (`rust_only_from_two_vectors_opposite`)
checks exactly and nearly opposite inputs (1 + c from 0 to ~5e-9, every least-aligned axis)
map the first direction onto the second within 8e-16 with unit norm. colmap-sharp's synthetic
oracle matches pycolmap's frame rotations bit for bit with the general branch (none of those
inputs is nearly opposite; view directions are uniform random, so 1 + c < 1e-8 has
probability ~5e-9 per frame).

## 5. Eigen's SIMD evaluation order is not reproduced outside the oracle-pinned cases

**What differs.** Eigen vectorizes fixed-size expressions: a norm or dot product is summed in
packet lanes and then reduced horizontally, and matrix products use vectorized kernels. The
fixed-size types in `colmap-rust/src/linalg/` evaluate these as left-to-right sums in
coefficient order, except where the oracle showed Eigen's order and the port copies it
(`Vector4d`'s reductions, `Quaterniond`'s product, `Matrix3d::trace`). The remaining sites can
differ from COLMAP in the last bits: the 3x3, 3x4, 4x4 and 6x6 matrix products, the Frobenius
norms, `Matrix4d::trace`, and the closed-form 3x3 and 4x4 determinants and inverses. Tier B.

**Why.** Eigen is MPL-2.0 and not ported (contract rule 2), and its packet order depends on the
target's SIMD width and the compiler (NEON, SSE and AVX builds reduce differently), so there
is no single order to match. A plain sequential order gives the same result on every target,
native and wasm. Same as colmap-sharp entry 115 (its fixed-size part).

**Evidence.** No pycolmap 4.2.0 binding exposes these products, norms or inverses directly, so
there is no fixture to pin them; `tests/linalg/rust_only_matrix.rs` checks them against exact
values and algebraic identities. Where a binding does reach Eigen's order (quaternion norm,
product, matrix conversions), `tests/linalg/rust_only_rotation_oracle.rs` pins it bit for bit.

## 20. Dynamic-size products and norms are left-to-right sums, not Eigen's blocked kernels

**What differs.** Eigen evaluates `MatrixXd` / `VectorXd` products, dot products and norms with
blocked, vectorized kernels (packet lanes reduced horizontally, cache-blocked GEMM). The
dynamic-size types in `colmap-rust/src/linalg/` (`matrix_x.rs`, `matrix_x_ops.rs`,
`vector_x.rs`) compute every product coefficient, dot product and squared norm as a plain
left-to-right sum seeded with the first term, no FMA (`linalg::dot`, `linalg::product`). The
transposed products `transpose_times`, `transpose_times_vector` and `transpose_times_self`
use the same order as the explicit `transpose() * b`, so they are bit-identical to it. Results
can differ from COLMAP in the last bits wherever a dynamic product is formed. Tier B.

**Why.** Eigen is MPL-2.0 and not ported (contract rule 2), and its blocking and packet order
depend on the matrix size, the target's SIMD width and the compiler, so there is no single
order to match. A sequential order gives the same bits on every target, native and wasm.
Same as colmap-sharp entry 115 (its dynamic-size part).

**Evidence.** No pycolmap 4.2.0 binding exposes a dynamic product directly.
`tests/linalg/rust_only_dynamic_matrix.rs` checks products against hand-computed values and
the transposed products bit for bit against the explicit transpose; the decomposition oracle
tests (`rust_only_decomposition_oracle.rs`, `rust_only_spectral_oracle.rs`) compare the
decompositions built on these products against numpy within their stated tolerances.

## 30. SVDs are two-sided Jacobi; singular-vector signs are ours, not Eigen's

**What differs.** COLMAP decomposes with `Eigen::JacobiSVD` (dynamic and fixed 3x3/4x4).
`colmap-rust/src/linalg/jacobi_svd.rs`, `svd_fixed.rs` and `jacobi_svd_kernel.rs` run a
two-sided (Kogbetliantz) cyclic Jacobi SVD written from Golub & Van Loan §8.6.3 and
Brent-Luk-Van Loan (1985), with Demmel-Veselić's relative stopping test plus an absolute floor
eps*||A||_F, preceded for rows > cols by a column-pivoted Householder QR (Eigen's documented
default preconditioner). Eigen's documented contract is kept: singular values non-negative and
decreasing, thin/full U and V on request, `rank()`/`solve()` with the threshold
max(1, min(rows, cols)) * eps * s_max. What can differ: the sign of each singular vector pair
(arbitrary in both), the basis chosen inside a repeated singular value's subspace or a null
space, and the last bits of every value. Ties keep their diagonal order (stable insertion
sort); a negative diagonal entry flips its U column. A 2x2 block with an exactly zero column or
row is rotated from one side only, so an exact null vector (a zero column of A) comes out
exact, which COLMAP's `TriangulatePoint` parallel-ray test (`V(3,3) == 0`) relies on.
`Svd3d`/`Svd4d` are bit-identical to `JacobiSvd` on the same matrix. Tier B.

**Why.** Eigen is MPL-2.0 and not ported (contract rule 2); its JacobiSVD's 2x2 step and sweep
order are implementation details with no published convention precise enough to reproduce its
signs or bits. COLMAP uses singular vectors only up to sign and null spaces only as spaces.
Port of colmap-sharp's `LinearAlgebra/JacobiSVD.cs`, `JacobiSvdKernel.cs` and `SvdFixed.cs`;
colmap-sharp documents this in those file headers and has no separate entry.

**Evidence.** `tests/linalg/rust_only_spectral_oracle.rs` checks 18 numpy (LAPACK dgesdd) cases
in COLMAP's shapes (3x3 to 20x9, 4x30, rank-deficient, repeated values, zero): singular values
within 1e-12 * s_max, rank exact, U/V orthogonal and U S V^T == A within 1e-12, simple singular
vectors within 1e-9 up to sign, null columns ||A v|| <= 1e-12, minimum-norm solve within 1e-10
of numpy's pinv, and Svd3d/Svd4d bit-identical to JacobiSvd. `rust_only_spectral_svd.rs` pins
sorting and signs on a diagonal, shapes, non-finite input, underflow at 1e-170, convergence in
<= 8 sweeps on exactly rank-deficient inputs, and the exact zero-column null vector.

## 31. SelfAdjointEigenSolver is cyclic Jacobi; eigenvector signs are ours

**What differs.** COLMAP's `Eigen::SelfAdjointEigenSolver` (the 4x4 normal matrix in
`geometry/triangulation.cc`) uses tridiagonalization + QL.
`colmap-rust/src/linalg/self_adjoint_eigen_solver.rs` runs cyclic Jacobi (Golub & Van Loan
Algorithms 8.5.1/8.5.3) on the input divided by its largest |entry|. Eigen's documented
contract is kept: only the lower triangle is read, eigenvalues increase (ties keep diagonal
order), eigenvectors are normalized columns. Eigenvector signs, the basis inside a repeated
eigenvalue's eigenspace and the last bits can differ. Tier B.

**Why.** Eigen is MPL-2.0 and not ported; Jacobi is simple, accurate to high relative precision
on the small matrices COLMAP decomposes, and has no Eigen-specific convention to match. COLMAP
reads the smallest eigenvalue's eigenvector up to scale. Port of colmap-sharp's
`LinearAlgebra/SelfAdjointEigenSolver.cs` (documented in its header; no separate colmap-sharp
entry).

**Evidence.** `rust_only_spectral_oracle.rs`: 4 numpy (dsyevd) cases including repeated and
zero eigenvalues: eigenvalues within 4e-12, simple eigenvectors within 1e-9 up to sign, V
orthonormal and V D V^T == A within 1e-12. `rust_only_spectral_eigen.rs` pins the lower-triangle
read, ascending order and scales 1e-170 / 1e160.

## 32. EigenSolver: eigenvalues in our Schur-block order, complex eigenvectors in our phase

**What differs.** COLMAP's `Eigen::EigenSolver` (polynomial roots via the companion matrix; the
4x4 in generalized_relative_pose.cc) is replaced by `colmap-rust/src/linalg/eigen_solver.rs` +
`eigen_solver_vectors.rs`: Householder Hessenberg reduction and Francis double-shift QR (Golub &
Van Loan 7.4.2 / 7.5.1, Wilkinson's exceptional shift at 10 and 20 iterations), real 2x2 blocks
split by a rotation, eigenvectors by quasi-triangular back-substitution with EISPACK hqr2's
eps*||T|| zero-pivot perturbation. Eigen's documented contract is kept: eigenvalues in the
order of T's diagonal blocks (not sorted), a complex pair as (re + i im, re - i im) with im > 0
first, eigenvectors unit-norm columns, real for a real eigenvalue. What can differ: the order of
the eigenvalues (our deflation order vs Eigen's), each eigenvector's sign and, for complex
vectors, its phase (the block eigenvector starts as (b, lambda - a) and is scaled to unit norm
without rotation), and last bits. The eigenvalues-only solve is bit-identical to the full
solve. Complex arithmetic (`linalg::Complex`) follows .NET's `System.Numerics.Complex`
operators (Smith division), so results match colmap-sharp. Tier B.

**Why.** Eigen is MPL-2.0 and not ported; its Schur deflation order and eigenvector
normalization are not a documented convention. No COLMAP caller depends on them: the
polynomial-root callers treat the roots as a set, GR8P's G is symmetric (real eigenvectors,
`hnormalized()` removes scale and sign), GR6P uses eigenvalues only. Port of colmap-sharp's
`LinearAlgebra/EigenSolver.cs` / `EigenSolver.Vectors.cs` (documented in their headers;
colmap-sharp entry 30 notes the Schur order only as it affects the six-point solvers).

**Evidence.** `rust_only_spectral_oracle.rs`: 10 numpy (dgeev) cases (random 3x3..8x8, four
companion matrices, a defective Jordan-block matrix): eigenvalues matched as a multiset within
1e-9 (1e-7 defective), ||A v - lambda v|| within the same, unit norm within 1e-12, real vectors
exactly real. `rust_only_spectral_eigen.rs` pins pair order, eigenvalues-only == full solve
bitwise on 64x64/17x17/8x8 random, a 9x9 mixed-block and a 7x7 companion matrix, and
1e200-scaled input.

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
`slice::select_nth_unstable_by` under `math::nan_last_cmp` instead of libc++'s
`std::nth_element` under `operator<`. `nan_last_cmp` is a total order: numbers compare by
value (-0.0 and +0.0 equivalent, as under `operator<`) and every NaN sorts after every number.
For NaN-free input the selected order statistics, and so the result, are identical to
COLMAP's. With NaN input the result is exactly that of the data sorted with the NaNs last: a
percentile whose interpolation ranks both fall on numbers is computed from the numbers only,
and one that touches a NaN rank returns NaN (the largest-below maximum skips NaN, since
`a < NaN` is false). With -0.0 and +0.0 tied at the selected rank, the signed zero returned
can differ. The order left in the slice differs too, but it is unspecified in C++ as well.

**Why.** With NaN, `std::nth_element` under `operator<` violates its strict-weak-ordering
precondition, so COLMAP's result is undefined (in practice it depends on libc++'s introselect
pivots); there is nothing definable to match. A comparator that is not a total order (such as
`partial_cmp(..).unwrap_or(Equal)`) may make std's selection panic, so a total order is
required for library code never to panic on NaN. NaN-last is the order `std::sort` of IEEE
values most often approximates and it is deterministic.

**Evidence.** `tests/math/math.rs` (math_test.cc's `Percentile`, `Median` and
`MedianAbsoluteDeviation` cases, including unsorted inputs) passes with exact equality.
`rust_only_percentile_with_nan_orders_nan_last` checks 200 values with 20% NaN: ranks on
numbers match the NaN-free sorted numbers, the 100th percentile is NaN, nothing panics.

Same as colmap-sharp's documented `Percentile` caveat (its `MathUtils.cs` header).

## 42. Iteration order of UnionFind parents and connected components

**What differs.** `math::union_find::UnionFind::parents` iterates in insertion order.
`find_connected_components` returns components ordered by their first node in the caller's
`nodes` slice, each listing its nodes in that order, and `find_largest_connected_component`
breaks ties between equally large components the same way. COLMAP's orders come from
`boost::unordered_node_map` / `unordered_flat_set` (`colmap/util/hash_containers.h`), and it
takes the nodes as a `FlatHashSet` where the port takes a slice. To keep the set semantics a
node repeated in the slice is reported once, at its first occurrence (`[1, 1, 2]` with no
edges gives `[[1], [2]]`).

**Why.** Reproducing them would mean porting Boost.Unordered's bucket layout and `boost::hash`
mixing for every key type. The order is not part of COLMAP's contract:
`connected_components_test.cc` compares with `UnorderedElementsAre`, and `union_find_test.cc`
never inspects it. Insertion order is deterministic and the same on every platform (a
std `HashMap` is used only for lookups, never iterated).

**Evidence.** The component sets, and every root `find` returns for the same call sequence,
are identical (`tests/math/union_find.rs`, `tests/math/connected_components.rs`, ported 1:1).
`rust_only_union_find_parents_in_insertion_order` and
`rust_only_connected_components_follow_node_order` pin the orders, and
`rust_only_connected_components_dedup_nodes` the handling of repeated nodes. Callers that need a
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

NaN weights: `max_weight` is computed as `std::max` does (`(a < b) ? b : a`), so a NaN weight
never becomes it, and a NaN weight gives a NaN cost. Costs are sorted with `math::nan_last_cmp`
(numbers by value, every NaN after every number, NaNs tied and kept in input order), so NaN
edges are taken only when they join components the numeric edges left apart, in input order.
COLMAP's Boost priority queue with NaN keys violates its ordering precondition, so its choice
is undefined.

**Why.** Boost is not ported; the input-index tie-break is the deterministic, documented rule.
The cost transform itself (`max_weight - w` in `f32`, `max_weight` starting at 0) is kept, so
the same weights tie on both sides. The sort needs a total order so that NaN input cannot make
std's sort panic.

**Evidence.** With distinct costs the minimum spanning tree is unique and the parents match
exactly (`tests/math/spanning_tree.rs`, spanning_tree_test.cc 1:1).
`rust_only_spanning_tree_ties_take_earlier_edge` and
`rust_only_spanning_tree_rejects_out_of_range_input` pin the tie rule and the checks, and
`rust_only_spanning_tree_with_nan_weights` (200 nodes, 20% NaN weights, and a triangle whose
NaN edge loses both ways) the NaN rule.

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
`compute`, where COLMAP's `colors_.at()` throws `std::out_of_range`. Integer flow sums that overflow wrap (`FlowValue`), where Boost's signed overflow is undefined behavior.

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

## 60. StringToDouble parses with Rust's parser and rejects non-decimal spellings

**What differs.** `util::string::string_to_double` (COLMAP's `StringToDouble`, also behind
`CSVToVector<float/double>`) parses the white-space-trimmed token with Rust's `f64::from_str`
instead of a classic-locale `std::istringstream >> double`. Both accept decimal and exponent
notation ("1", "-0.5", ".5", "1e-3") and reject words and trailing characters. Where they could
disagree, the Rust port rejects: a token with any character outside `0-9 . e E + -` (so
`inf`, `nan`, `infinity` and hexadecimal floats such as `0x1p3` fail), and a value that
overflows to infinity (libc++ sets `failbit` on `ERANGE`). Underflow to a subnormal or zero is
accepted, where libc++ may set `failbit`.

**Why.** Reproducing libc++'s `num_get` exactly would mean porting a C++ standard library for
inputs COLMAP never writes: every string that reaches this parser in COLMAP's own formats is
decimal output of its writers (`%g`-style or `precision(17)`), which both parsers read to the
same, correctly rounded double. Same as colmap-sharp entry 20.

**Evidence.** `tests/util/string.rs` (`string_to_double_nominal`,
`string_to_double_locale_independence`, `rust_only_string_to_double_rejects`) and the
`CSVToVector` cases of `tests/util/misc.rs` pass 1:1.

## 61. Little-endian binary reads fail on a short stream

**What differs.** COLMAP's `ReadBinaryLittleEndian<T>` reads `sizeof(T)` bytes with
`std::istream::read` and returns whatever is in its buffer when the stream ends early (the
stream's failbit is set, and callers do not check it per value). colmap-rust's
`util::endian::read_binary_little_endian` returns the `std::io::Error` (`UnexpectedEof`), so
a truncated `cameras.bin` / `images.bin` / `points3D.bin` or depth map is reported instead of
read as garbage.

**Why.** Rust's `Read::read_exact` reports the short read, and silently continuing with an
unspecified value is not a behavior worth reproducing; on complete input the two are
identical byte for byte. colmap-sharp made the same choice for its MVS reader (its entry 62).

**Evidence.** `tests/util/endian.rs`: the ported round trips pass 1:1, and
`rust_only_little_endian_wire_bytes_and_short_read` pins the wire bytes and the error.

## 62. The timer's clock comes from the host on wasm32-unknown-unknown

**What differs.** COLMAP's `Timer` reads `std::chrono::high_resolution_clock`.
colmap-rust's `util::timer` reads `std::time::Instant` natively, but on
`wasm32-unknown-unknown` std has no clock (`Instant::now()` panics there), so it reads a
monotonic source (nanoseconds) that the host must install with
`util::timer::set_clock_source`, e.g. from `performance.now()`. With none installed on that
target, the clock stands still and every elapsed time reads 0. Elapsed microseconds are truncated from nanoseconds as COLMAP's `duration_cast` does.

**Why.** The core crate must run in the browser without JavaScript bindings (no
`wasm-bindgen` in the core, CLAUDE.md contract 1), and elapsed times only feed progress
reports, never results.

**Evidence.** `tests/util/timer.rs` passes 1:1 natively; the core crate builds for
`wasm32-unknown-unknown`.

## 63. File-extension helpers split paths only at '/'

**What differs.** COLMAP's `HasFileExtension` takes a `std::filesystem::path`, whose file name
on Windows also ends at '\'. colmap-rust's `util::file::has_file_extension` works on path
strings and treats only '/' as a separator, on every platform. So for the Windows-style name `dir\.jpg`, `has_file_extension(.., ".jpg")`
is true here (the whole string is the file name, and its last '.' is not its first
character), where COLMAP on Windows sees the dot file `.jpg`, which has no extension, and
returns false. `split_file_extension` matches
COLMAP everywhere (COLMAP splits that one at '.' only).

**Why.** The core crate has no file system and must give the same answer natively and in the
browser, so it cannot depend on the host platform's separator rules. Hosts must pass
'/'-normalized names, and on those the two agree.

**Evidence.** `tests/util/file.rs`: the ported `file_test.cc` cases pass 1:1, and
`rust_only_has_file_extension_edge_cases` pins the '/' rules.

## 100. FMA contraction in the camera models

**What differs.** Camera model projection (`camera_model_img_from_cam`) and ray unprojection
(`camera_model_cam_ray_from_img`) of every perspective model, and `camera_model_cam_from_img`
of the fisheye, division, FOV and EUCM models, differ from the pycolmap 4.2.0 macOS arm64
wheel by a few ulps on part of the inputs (at most 2e-14 relative to max(1, |value|) in the
fixture). Which calls succeed or fail never differs. Same as colmap-sharp entry 12.

**Why.** The wheel is built with contraction on and fuses multiply-adds that sit in one C++
statement, e.g. `*x = f * *x + c1` in every model's `ImgFromCam` and `u * u + v * v + 1.0`
in `CamRayFromImg`. colmap-rust never uses FMA in math paths (CLAUDE.md, "No FMA"), so its
results are the same on every platform. The iterative undistortion runs its distortion on
`ceres::Jet`, whose operators are separate function calls that clang does not contract, and
matches the wheel bit for bit. The models that call `sin`/`cos`/`tan`/`atan`/`atan2` (the
fisheye models, FOV, EQUIRECTANGULAR) also go through the `libm` crate rather than Apple libm
(entry 1), which can move the same outputs by an ulp; colmap-sharp holds EQUIRECTANGULAR
bit-exact because .NET calls the platform libm, colmap-rust does not.

**Evidence.** `oracle/camera_models.py` prints it: re-deriving SIMPLE_RADIAL's projected x with
the unfused formula matches the wheel on 64/75 and 60/75 points of the two parameter sets,
and on 75/75 with only `f * x + c1` fused; PINHOLE's ray z matches on 98/101 plain and
101/101 with `u*u + v*v` fused. `tests/sensor/rust_only_camera_model_oracle.rs` requires
bit-identical `CamFromImg` for the plain pinholes and the models that unproject through the
iterative undistortion (SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV) and the pixel threshold,
and pins everything else at 2e-14 relative to max(1, |value|).

**Related, not observed here.** C++ `EquirectangularCameraModel` evaluates
`2.0 * EIGEN_PI * (...)` with `EIGEN_PI` a `long double` literal, so on x86-64 Linux (80-bit
long double) its results may differ from both the macOS wheel (where long double is double)
and colmap-rust, which uses `std::f64::consts::PI`.

## 101. An unknown camera model id: an error at the boundary functions, a panic on the per-point path

**What differs.** COLMAP's camera model dispatch functions all throw
`std::domain_error("Camera model does not exist")` for an id outside `CAMERA_MODEL_CASES`.
colmap-rust splits them:
- The metadata and validation functions that COLMAP calls at input boundaries
  (`camera_model_initialize_params`, `camera_model_params_info`, the four index-group getters,
  `camera_model_num_params`, `camera_model_verify_params`, `camera_model_has_bogus_params`)
  return `Err` with `ErrorKind::DomainError` and COLMAP's message, the Rust form of the throw.
- The per-point functions (`camera_model_img_from_cam`, `camera_model_cam_from_img`,
  `camera_model_cam_ray_from_img`, `camera_model_cam_from_img_threshold`,
  `camera_model_rescale`, `camera_model_is_perspective`, `..._is_perspective_pinhole`,
  `..._is_spherical`) stay infallible and panic with the same message.
- The functions COLMAP answers without throwing (`CameraModelNameToId` returns `kInvalid`,
  `CameraModelIdToName` returns "", `ExistsCameraModelWithId`,
  `CameraModelIsPerspectiveFisheye`) answer the same way here.
Separately, a C++ enum class can hold any integer (`static_cast<CameraModelId>(123456789)`); a
Rust `CameraModelId` holds only named enumerators, so a raw value is checked once, at
`CameraModelId::from_i32`, which returns `None` for an unnamed value. The only unknown id that
reaches a dispatch function is therefore `CameraModelId::Invalid`.

**Why.** COLMAP does pass `kInvalid` into dispatch at its input boundaries: the text reader
(`scene/reconstruction_io_text.cc`, around line 140) looks a model name up with
`CameraModelNameToId` and calls `CameraModelNumParams` / `CameraModelVerifyParams` on the
result without an existence check, and `camera_test.cc` expects `VerifyParams` on a default
(`kInvalid`) `Camera` to throw `domain_error`. A misspelled model in a user's `cameras.txt`
must be a recoverable error (in the web app a panic would abort the app), so those functions
return `Result`. The per-point functions run once per point in every projection, residual and
undistortion loop, where a `Result` would cost every caller for a condition that cannot occur
once the camera is validated, so they keep a panic: an internal invariant, where CLAUDE.md's
error rule allows one.

**Obligation on later phases.** Every path that creates a camera from outside data must
validate the id through the fallible functions before any per-point call: Phase 4's
`Camera::verify_params` (returning the error, as `camera_test.cc` expects), the camera
readers (text, binary, database) and `Camera::create_from_model_name` /
`create_from_model_id` propagate the `DomainError`; nothing may call a per-point function on
an unverified `Camera`.

**Evidence.** `tests/sensor/models.rs` ports `models_test.cc` 1:1, including the
`ExistsCameraModelWithId(static_cast<CameraModelId>(123456789))` check through `from_i32`.
`tests/sensor/rust_only_models.rs` checks that every boundary function returns
`Err(DomainError, "Camera model does not exist")` for `Invalid` and that projection and
unprojection panic with that message.

## 102. The analytic projection Jacobians are unfused and use pure-Rust libm

**What differs.** The `ImgFromCamWithJac` kernels (`sensor/models/jacobian*.rs`, port of
`models_jacobian.h`) and `camera_model_img_from_cam_with_jac` keep COLMAP's operation order
but evaluate every multiply-add unfused, and the fisheye, FOV and EQUIRECTANGULAR kernels call
`atan`/`tan`/`atan2` through the `libm` crate. A clang build with contraction on (the pycolmap
macOS wheel) can therefore differ in the last bits of the pixel and of the Jacobian entries.
EQUIRECTANGULAR's `kInv2Pi = 1.0 / (2.0 * EIGEN_PI)` and `kInvPi` are `long double`
expressions in C++; here they are computed from `f64` pi, which is the same value wherever
long double is double (macOS arm64), not on x86-64 Linux.

**Why.** Same causes as entries 1 and 100: colmap-rust never uses FMA in math paths and routes
transcendentals through `math::fns` so native and wasm agree bit for bit.

**Evidence.** `src/sensor/models/jacobian_tests.rs` ports `models_jacobian_test.cc` 1:1: every
kernel matches `ImgFromCam` differentiated with the Jet to COLMAP's 1e-10, and the dispatch
matches the typed kernel. pycolmap exposes these kernels only inside its essential-matrix
estimators (through `Camera::CamRayFromImgWithJac`), so there is no
oracle fixture for these kernels.
