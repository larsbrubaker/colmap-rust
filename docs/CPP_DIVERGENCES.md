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
eigenvalue's eigenspace and the last bits can differ. On non-finite input (`info()` is
`InvalidInput`) the eigenvalues are NaN and, when eigenvectors were requested, the
eigenvector matrix is NaN-filled, as `JacobiSvd` fills U and V, where Eigen leaves them
unspecified; `eigenvectors()` panics only when they were not requested. Tier B.
eigenvalue's eigenspace and the last bits can differ. Tier B.

**Why.** Eigen is MPL-2.0 and not ported; Jacobi is simple, accurate to high relative precision
on the small matrices COLMAP decomposes, and has no Eigen-specific convention to match. COLMAP
reads the smallest eigenvalue's eigenvector up to scale. Port of colmap-sharp's
`LinearAlgebra/SelfAdjointEigenSolver.cs` (documented in its header; no separate colmap-sharp
entry).

**Evidence.** `rust_only_spectral_oracle.rs`: 4 numpy (dsyevd) cases including repeated and
zero eigenvalues: eigenvalues within 4e-12, simple eigenvectors within 1e-9 up to sign, V
orthonormal and V D V^T == A within 1e-12. `rust_only_spectral_eigen.rs` pins the lower-triangle
read, ascending order, scales 1e-170 / 1e160, and the NaN-filled eigenvectors on 1x1 NaN
input.
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
solve. On non-finite input (`InvalidInput`) or no convergence (`NoConvergence`) the eigenvalues
are NaN and requested eigenvectors are NaN-filled (Eigen leaves them unspecified);
`eigenvectors()` panics only when they were not requested. Complex arithmetic (`linalg::Complex`) follows .NET's `System.Numerics.Complex`
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
bitwise on 64x64/17x17/8x8 random, a 9x9 mixed-block and a 7x7 companion matrix,
1e200-scaled input, and the NaN-filled eigenvectors on 1x1 NaN input.
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

## 50. Durand-Kerner divides complex numbers with Smith's algorithm, not libc++'s

**What differs.** `FindPolynomialRootsDurandKerner` divides `std::complex<double>` values;
libc++'s `operator/` rescales both operands by `logb`/`scalbn` of the divisor's larger part and
then applies the textbook formula (which Apple clang may also FMA-contract).
`colmap-rust/src/math/polynomial.rs` uses `linalg::Complex`'s division, Smith's algorithm in
.NET's branch order, so each Newton-like update can differ in the last bits, and after up to 100
iterations the roots can differ at the 1e-10 convergence threshold. The `+ double` step of the
Horner evaluation adds to the real part only, as libc++ does. Tier B. Same as colmap-sharp's
choice (documented in its `Mathematics/Polynomial.cs` header rather than as a numbered entry).

**Why.** Reproducing libc++'s division bit for bit would also need Apple clang's contraction
decisions; the iteration's contract is convergence to the roots, which both divisions meet.
Sharing `linalg::Complex` keeps colmap-rust and colmap-sharp on identical bits.

**Evidence.** `tests/math/polynomial.rs`: `find_polynomial_roots_durand_kerner_nominal` matches
COLMAP's reference roots within 1e-6 in COLMAP's order, and
`find_cubic_polynomial_roots_multi_root` agrees with the closed form within 1e-4.
`rust_only_companion_matrix_agrees_with_durand_kerner_on_real_roots` checks a quartic with
known roots to 1e-8.

## 51. Companion-matrix roots come in our EigenSolver's order

**What differs.** `FindPolynomialRootsCompanionMatrix` returns the companion matrix's
eigenvalues in the order the eigen solver produces them. COLMAP's come from Eigen's
`EigenSolver`; ours from `linalg::EigenSolver` (entry 32), whose Schur deflation order can
differ from Eigen's for some inputs, and whose values can differ in the last bits. A complex
pair still lists `+imag` first, and the zero root added for trailing zero coefficients is still
last. Tier B. Same as colmap-sharp (its `Mathematics/Polynomial.cs` header).

**Why.** Eigen is not ported (entry 32). COLMAP's callers (the minimal solvers) treat the roots
as a set.

**Evidence.** `tests/math/polynomial.rs`: `find_polynomial_roots_companion_matrix_nominal` and
`_zero_solution` match COLMAP's reference roots within 1e-6 in COLMAP's listed order, so on
these cases the order is Eigen's.

## 52. Polynomial root outputs are returned, not written through nullable pointers

**What differs.** COLMAP's root finders take `Eigen::VectorXd* real, Eigen::VectorXd* imag`
(either may be null) and return `bool`; `colmap-rust/src/math/polynomial.rs` returns
`Result<Option<PolynomialRoots>>` with both parts always computed (`None` for `false`, `Err`
for a failed `THROW_CHECK`). On `false`, COLMAP leaves the caller's vectors untouched; there is
nothing to leave here. `FindCubicPolynomialRoots` writes only its first `num_roots` entries of
the caller's `Vector3d`; `find_cubic_polynomial_roots` returns `(num_roots, Vector3d)` with the
unused entries zero. Same as colmap-sharp's port (C# `out` parameters, always filled, unused
cubic entries zero).

**Why.** Rust has no nullable output references worth mirroring, and computing both parts
costs nothing measurable. No COLMAP caller reads the untouched entries.

**Evidence.** `rust_only_cubic_leaves_unused_entries_zero` and
`rust_only_polynomial_constant_has_no_roots` in `tests/math/rust_only_polynomial_matrix.rs`;
the ported `CHECK_EQUAL_RESULT` cases compare the full outputs.

## 53. DecomposeMatrixRQ goes through our Householder QR

**What differs.** `DecomposeMatrixRQ` factors the flipped transpose with
`Eigen::HouseholderQR`; `colmap-rust/src/math/matrix.rs` uses `linalg::HouseholderQr`
(colmap-sharp's port, not Eigen's blocked implementation), so R and Q can differ from COLMAP's
in the last bits. The algorithm around the QR (flips, zeroing loop, `det(Q) > 0`
normalization with the caller's matrix type's determinant) is COLMAP's. The template becomes
three entry points: `decompose_matrix_rq` (`MatrixXd`, `Err` unless square),
`decompose_matrix_rq_3d` and `decompose_matrix_rq_4d`. Tier B. Same as colmap-sharp
(`Mathematics/MatrixUtils.cs` header).

**Why.** Eigen is MPL-2.0 and not ported.

**Evidence.** `tests/math/matrix.rs`: `decompose_matrix_rq_nominal` (COLMAP's 1e-6 tolerances,
upper-triangular and unitary at Eigen's default precision).
`rust_only_decompose_matrix_rq_dynamic_and_3d` covers 2x2/3x3/5x5 at 1e-9 and pins the 3x3
entry point to the dynamic one bit for bit.

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

## 80. No FMA contraction in the pose/covariance products and the GPS conversions (the macOS pycolmap wheel fuses them)

**What differs.** Beyond the quaternion-vector rotation of entry 2, the pycolmap 4.2.0 macOS
arm64 wheel differs from colmap-rust in the last bits on `Rigid3d::adjoint_inverse` and
`get_covariance_for_rigid3d_inverse` (Eigen's 3x3 and 6x6 products,
`colmap-rust/src/geometry/rigid3.rs`) and on the `GpsTransform` ellipsoid/ECEF/ENU/UTM
conversions (`colmap-rust/src/geometry/gps.rs`), up to 9.3e-10 m on ECEF-scale coordinates.

**Why.** Same cause as entry 2: the wheel is built with contraction on and fuses multiply-adds
(for example `N * (1 - e2) + alt` in `EllipsoidToECEF`). colmap-rust never uses FMA in math
paths (CLAUDE.md, "No FMA"), so its results are the same on every target and are expected to
match a C++ build that does not contract. The GPS conversions also go through `math::fns`
(entry 1). Same as colmap-sharp entry 6.

**Evidence.** colmap-sharp's `oracle/geometry_transforms.py` (copied here) shows that fusing
`N * (1 - e2) + alt` takes the `EllipsoidToECEF` z coordinate from 13/80 to 3/80 mismatches
while x and y (no multiply-add) match on every case.
`tests/geometry/rust_only_transforms_oracle.rs` pins these fields at 1e-14 relative (1e-13 for
the 6x6 covariance, and 1e-8 m for GPS coordinates in meters) in
`rust_only_transforms_oracle_tolerance_fields`; the operations that do not multiply-add
(composition and inverse rotations, `to_matrix`, `from_matrix`, `adjoint`, the UTM zone, the
strings) are bit-identical in `rust_only_transforms_oracle_exact_fields` and
`rust_only_transforms_oracle_strings`.

## 81. ComputeBoundingBoxAndCentroid sorts instead of std::nth_element

**What differs.** `geometry::normalization::compute_bounding_box_and_centroid` fully sorts
each coordinate list (`sort_by(f64::total_cmp)`) where COLMAP partitions it with two
`std::nth_element` calls. The bounding box (the elements at the two percentile positions) is
the same value, and so is the multiset of elements the centroid averages, but the order they
are summed in differs: COLMAP's is whatever libc++'s `nth_element` leaves between the two
positions. The centroid can therefore differ from COLMAP's in the last bits.

**Why.** The element order after `nth_element` is an unspecified implementation detail of the
C++ standard library; reproducing it would mean porting libc++'s introselect for one
rounding-level effect. Sorting satisfies every `nth_element` postcondition. Same as
colmap-sharp entry 15.

**Evidence.** `tests/geometry/normalization.rs`: normalization_test.cc passes 1:1, including
the exact bounding boxes and the 1e-6 centroid checks.

## 82. EllipsoidToUTM rejects longitude 180 instead of writing out of bounds

**What differs.** `GpsTransform::ellipsoid_to_utm` returns a "Check failed" error for a point
at longitude exactly 180. COLMAP accepts it (its range check is `lon <= 180`), maps it to zone
61 and increments `zone_counts[60]` of a 60-element `std::array`, which is undefined behavior.

**Why.** There is no defined COLMAP behavior to match, and Rust would panic on the
out-of-bounds index. An error keeps library code panic-free and tells the caller. Longitude
-180 (zone 1) is unaffected. colmap-sharp throws `IndexOutOfRangeException` in the same place.

**Evidence.** `tests/geometry/gps.rs`, `rust_only_gps_utm_rejects_out_of_range_input`.

## 83. UTMToEllipsoid latitude can differ by 1 ulp

**What differs.** `GpsTransform::utm_to_ellipsoid` returns a latitude one ulp away from the
pycolmap 4.2.0 macOS arm64 wheel on some points of `geometry_transforms.json` (colmap-sharp
sees 2 of 80); longitude and altitude match.

**Why.** The cause is not established. colmap-sharp re-derived the conversion in Python with
the platform libm and reproduced its (and our) result exactly, and fusing the series'
multiply-adds did not remove the mismatch, so it is neither a port bug nor the contraction of
entry 80. We do not emulate it. Same as colmap-sharp entry 11.

**Evidence.** `tests/geometry/rust_only_transforms_oracle.rs` pins `utm_to_ellipsoid` at
1e-14 relative in `rust_only_transforms_oracle_tolerance_fields`; the observed gap is about
7e-15 degrees.

## 84. GravityFromExifOrientation does not log

**What differs.** `geometry::pose_prior::gravity_from_exif_orientation` returns `None` for a
mirrored EXIF orientation (2, 4, 5, 7) or an unknown value, as COLMAP returns
`std::nullopt`, but it does not emit COLMAP's `LOG(WARNING)` / `LOG(ERROR)` line.

**Why.** The core crate has no logging facility yet (COLMAP's glog is not ported). The
return value, which is the function's contract, is unchanged; the log line is diagnostic only.
When a logger lands, this entry goes away.

**Evidence.** `tests/geometry/pose_prior.rs`, `pose_prior_gravity_from_exif_orientation`
(pose_prior_test.cc 1:1) checks every `None` case.

## 85. Essential-matrix candidate order and the epipole's sign follow our SVD's signs

**What differs.** `geometry::essential_matrix::decompose_essential_matrix` builds its two
rotations and the translation from the SVD of `E` (entry 30), whose singular-vector signs are
ours, not Eigen's. After COLMAP's own determinant fix-up of U and V, the candidate *set*
{(R1, t), (R2, t), (R1, -t), (R2, -t)} is the same for any valid SVD (Hartley and Zisserman,
"Multiple View Geometry", 2nd ed., Result 9.19), but which rotation is called R1 and the sign
of t can differ from COLMAP. `pose_from_essential_matrix` keeps the *last* candidate with the
most points in front of both cameras, as COLMAP does, so its result can differ from COLMAP's
only when two candidates tie on that count. `epipole_from_essential_matrix` returns the null
vector as our SVD gives it, so its overall sign can differ (it is arbitrary in COLMAP too; an
epipole is a homogeneous point).

**Why.** Eigen's JacobiSVD is not ported (MPL-2.0) and its sign convention is an
implementation detail. COLMAP's contract is the relative pose with the most cheiral points,
and the epipole as a projective point; both are kept.

**Evidence.** `tests/geometry/essential_matrix.rs` (essential_matrix_test.cc 1:1:
`decompose_essential_matrix_nominal` accepts either rotation and either sign of t, as COLMAP's
test does; `pose_from_essential_matrix_nominal` recovers the pose at 1e-12). The pycolmap
fixture `geometry_two_view.json` deliberately carries no epipoles. colmap-sharp documents the
same behavior in its `Geometry/EssentialMatrix.cs` header (no numbered entry there).

## 86. PoseFromHomographyMatrix on noise-free planar data can pick the other valid candidate

**What differs.** For exact correspondences of a plane, two of the four candidates of
`geometry::homography_matrix::decompose_homography_matrix` usually both triangulate every
point in front of both cameras, with angular reprojection sums at rounding level (~1e-16).
`pose_from_homography_matrix` breaks that tie by the smaller sum, as COLMAP does, so which of
the two it returns depends on last-bit rounding (our 3x3 inverse and products, our SVD's
middle singular value, and `triangulate_mid_point`'s SVD, entries 20 and 30) and can differ
from COLMAP. With any noise on the correspondences the sums separate and the choice matches.

**Why.** Reproducing Eigen's rounding exactly is not possible without porting Eigen
(contract rule 2); the tie is a property of the exact data, not of the algorithm.

**Evidence.** `tests/geometry/rust_only_two_view_oracle.rs`,
`rust_only_pose_from_homography_matrix_matches_pycolmap`: on 20 noisy scenes the pose,
normal and points agree with pycolmap within 1e-12 / 1e-11 / 1e-10 relative.
`tests/geometry/homography_matrix.rs` (homography_matrix_test.cc 1:1) passes, including the
noise-free `pose_from_homography_matrix_nominal`. colmap-sharp documents the same behavior in
its `Geometry/HomographyMatrix.cs` header and fixture generator (no numbered entry there).

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
expressions in C++; computed from `f64` pi here, they round to the same doubles on every
platform (checked at 64-bit and 113-bit long double mantissas), so they are not a source of
difference.

**Why.** Same causes as entries 1 and 100: colmap-rust never uses FMA in math paths and routes
transcendentals through `math::fns` so native and wasm agree bit for bit.

**Evidence.** `src/sensor/models/jacobian_tests.rs` ports `models_jacobian_test.cc` 1:1: every
kernel matches `ImgFromCam` differentiated with the Jet to COLMAP's 1e-10, and the dispatch
matches the typed kernel. pycolmap exposes these kernels only inside its essential-matrix
estimators (through `Camera::CamRayFromImgWithJac`), so there is no
oracle fixture for these kernels.

## 120. CameraDatabase iterates the sensor-width table in specs.cc order, not hash order

Same as colmap-sharp entry 8.

**What differs.** COLMAP's `camera_specs_t` is a `NodeHashMap`, and
`CameraDatabase::QuerySensorWidth` iterates it, writing the output width on every match and
stopping after the second non-exact match per make. `sensor::specs::initialize_camera_specs`
returns the makes in `specs.cc` source order, so when a cleaned EXIF make matches more than one
table make (a substring match either way round; an empty make matches all of them), which
widths are seen, and so the width left behind and whether a unique match is found, can differ
from a given COLMAP build.

**Why.** Hash iteration order is unspecified and differs between standard libraries and Boost,
so there is no single COLMAP behavior to match; CLAUDE.md asks for deterministic order.

**Evidence.** `tests/sensor/database.rs` ports `database_test.cc` 1:1; its cases match a
single make and pass. Queries whose make matches one table make are unaffected.

## 121. Bitmap::rescale is a resampler written here, not OpenImageIO's resize

Same as colmap-sharp entry 9.

**What differs.** COLMAP's `Bitmap::Rescale` calls `OIIO::ImageBufAlgo::resize` with a
"triangle" (kBilinear) or "box" (kBox) filter. `src/sensor/bitmap/resize.rs` (a port of
colmap-sharp's `BitmapResize.cs`) reimplements the model OIIO's output follows (filter widened
by the downsampling ratio, clamp-to-edge samples, separable, round half away from zero,
accumulation in double). Bilinear results are within one gray level of pycolmap's; box results
agree except where a source pixel center lies exactly on the box edge at a non-integer ratio,
where OIIO's inclusion rule is not reproduced and a destination pixel can average one source
pixel more or fewer.

**Why.** OpenImageIO is native (`docs/LICENSE_AUDIT.md`). Its resize accumulates in float with
its own filter evaluation, so bit-exact output would need a port of OIIO's resampling code
(Apache-2.0, allowed but not done).

**Evidence.** `oracle/fixture_bitmap_rescale.py` (copied from colmap-sharp) records pycolmap
4.2.0's bilinear output on seeded grey and RGB images, up and down;
`tests/sensor/rust_only_bitmap_rescale_oracle.rs` checks every pixel within one gray level.
colmap-sharp's impulse probes matched pycolmap exactly for both filters at ratios 4, 8, 3, 1.5,
5/3, 2/3 and 3/5; the box tie case is 23 -> 10 pixels, destination pixel 5.

## 122. The EXIF reader leaves rationals with a zero denominator unset

Same as colmap-sharp entry 10.

**What differs.** An EXIF RATIONAL with denominator 0 (FocalLength, FocalPlaneXResolution,
GPSLatitude/Longitude, GPSAltitude) is not stored in the Bitmap's metadata by
`src/sensor/exif_reader.rs`. OpenImageIO, through which COLMAP reads EXIF, most likely stores
the float quotient (inf, or NaN for 0/0), which COLMAP's getters would see.

**Why.** An inf/NaN focal length or GPS coordinate carries no information, and cameras write
0/0 to mean "unknown", so "absent" is the faithful reading. It is visible only through the
getters: `exif_latitude` / `exif_longitude` / `exif_altitude` return `None` where COLMAP could
return NaN or inf, and `exif_focal_length` returns `None` (or a later fallback's value) where
COLMAP could return inf or NaN from a zero-denominator FocalLength.

**Evidence.** Not verified against OIIO: no oracle fixture carries a zero-denominator tag. The
reader's behavior is stated in its file header.

## 123. Bitmap interpolation treats points beyond int range and NaN as outside the image

Same as colmap-sharp entry 117.

**What differs.** `Bitmap::InterpolateBilinear` computes `x0 = static_cast<int>(std::floor(x))`,
`x1 = x0 + 1` and rejects the point when `x0 < 0 || x1 >= width_` (likewise for y);
`InterpolateNearestNeighbor` casts `std::round(x)`. For a point beyond int range or NaN the
cast is undefined behavior in C++. The port tests the doubles first (`x >= 0 && x < width - 1`,
the same check for every finite x) and returns `None` for NaN in both methods, so such points
are outside the image.

**Why.** Rust's `as i32` saturates and maps NaN to 0, so the literal translation would sample
pixel (0, 0) for NaN, and `floor(x) = i32::MAX` would make `x1` overflow. On x86 COLMAP's cast
yields `INT_MIN`, which its check rejects, so `None` is what COLMAP does there; on arm64 COLMAP
reads out of bounds. colmap-sharp found this through intermittent stereo-rectification failures
(source points beyond int range for about 2.5% of PRNG seeds).

**Evidence.** `tests/sensor/rust_only_bitmap.rs`,
`rust_only_interpolate_far_outside_or_nan_returns_none`.

## 124. Bitmap has no file I/O, and its metadata is a typed store that always exists

**What differs.** COLMAP's `Bitmap::Read` / `Write` decode and encode image files through
OpenImageIO; the port has neither: the host decodes pixels into `row_major_data_mut()` and passes
the file's bytes to `sensor::exif_reader` for the EXIF metadata. COLMAP's metadata is an OIIO
`ImageSpec` behind a pointer that is null for a default-constructed bitmap or a copy of an
empty one, and is addressed by OIIO type strings (`SetMetaData(name, "float", &value)`). Here it
is a `MetaDataValue` enum with typed getters (int, float, point, string). Only the conversions
the EXIF getters use are implemented (an int reads as float and as its decimal string); other
conversions read as absent, where OIIO may convert more. Names are compared
ASCII case-insensitively like OIIO's default `getattribute`, and the store always exists: where
COLMAP dereferences the null pointer (metadata access, `Rescale`, `Rot90` or `CloneMetadata` on
a bitmap without metadata), the port reads absent values and writes create entries. A copy of an
empty bitmap still drops the metadata, as in COLMAP.

**Why.** OpenImageIO is native and not ported (`docs/LICENSE_AUDIT.md`); the app hosts decode
images (the browser through its own decoders), so the core takes pixel buffers. The null
dereference is a crash in COLMAP, not a contract, and a defined result is the safe reading.

**Evidence.** The file I/O cases of `bitmap_test.cc` are listed as skipped in
`PORTING_PLAN.md`; the metadata cases (`SetGetMetaData`, `CloneMetaData`, the EXIF getters) are
ported in `tests/sensor/bitmap_exif.rs`, and `tests/sensor/rust_only_bitmap.rs` pins the type
conversions, case-insensitivity and the copy-of-empty behavior.

## 140. PROSAC's out-of-range sample index fails a check instead of reading past the data

**What differs.** COLMAP's `ProgressiveSampler` (ported faithfully in
`optim/progressive_sampler.rs`) makes index `n` the mandatory element of a progressive sample,
and `n` can equal `total_num_samples`: on the first sample when `num_samples ==
total_num_samples`, and in general on the sample where the growth schedule reaches the last
element. COLMAP's `Sampler::SampleXY` then reads `X[total_num_samples]`, past the end of the
`std::vector` (undefined behavior: garbage or a crash). `Sampler::sample_x` / `sample_xy` in
`optim/sampler.rs` check every sampled index against the data length and return COLMAP's
"Check failed" error instead.

**Why.** Undefined behavior has no Rust equivalent to match (an out-of-bounds index panics),
and silently clamping or shifting the index would change the sampler's Tier A sequence that
`progressive_sampler_test.cc` pins. COLMAP 4.2.0 instantiates `ProgressiveSampler` nowhere
outside its own test, so no pipeline reaches this path. Same as colmap-sharp entry 16.

**Evidence.** `tests/optim/progressive_sampler.rs` (1:1) and the seeded sequence in
`tests/optim/rust_only_sampler_sequence.rs` pass unchanged;
`tests/optim/rust_only_ransac.rs`, `rust_only_progressive_sampler_index_past_end_fails_check`
runs RANSAC with PROSAC on exactly `MIN_NUM_SAMPLES` pairs and expects the check failure.

## 141. RANSAC and LO-RANSAC always run their trial loop serially

**What differs.** With `RANSACOptions::num_threads > 1` (or -1), COLMAP built with OpenMP runs
the trial loop on several threads, each with its own sampler seeded `random_seed + thread
index`, sharing an atomic trial counter and a mutex-guarded best model. `optim/ransac.rs` and
`optim/loransac.rs` validate `num_threads` exactly as COLMAP does (`Check()`, and "Parallel
RANSAC only supports RandomSampler" for any other sampler with more than one effective thread)
and then run the loop once on the calling thread with the thread-0 seed, which is what COLMAP
itself does when built without OpenMP ("the block runs once serially").

**Why.** COLMAP's parallel result depends on thread scheduling (which thread claims which trial
index, and which thread's model reaches the shared best first), so it is not reproducible even
against itself; CLAUDE.md requires sequential and parallel runs to give the same result. The
serial loop is COLMAP's own `num_threads == 1` behavior. A deterministic parallel scheme (for
example fixed per-trial seeds) would be a different algorithm from both COLMAP builds and is
left for when profiling shows RANSAC is a bottleneck. Same as colmap-sharp entry 17.

**Evidence.** `tests/optim/rust_only_ransac.rs`: `rust_only_ransac_parallel_line_fit` and
`rust_only_loransac_parallel_line_fit` (`num_threads = 4`) pass the same checks as the serial
runs, and `rust_only_parallel_requires_random_sampler` pins the kept validation. The 1:1
`ParallelSimilarityTransform` cases join them with Phase 6's `SimilarityTransformEstimator`.
