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

## 33. Sparse Cholesky: simplicial LLT/LDLT with our AMD instead of CHOLMOD and Eigen

**What differs.** `SparseCholeskyWithFallbackSolver`'s first stage is CHOLMOD's supernodal
LLT in COLMAP and the simplicial LLT of `colmap-rust/src/linalg/simplicial_cholesky.rs` here;
its fallback and `LeastAbsoluteDeviationSolver`'s `SimplicialLLT` are Eigen's simplicial
factorizations in COLMAP and ours here. `LeastAbsoluteDeviationSolverType::SupernodalCholmodLlt`
keeps its name but selects the fallback solver. The fill-reducing ordering is our AMD
(`linalg/amd_ordering.rs`, from the Amestoy-Davis-Duff paper, without dense-row deferral,
initial supervariable detection or mass elimination), so the permutation, and with it the
order of floating-point operations and the last bits of every solution, differ from CHOLMOD's
and Eigen's. Accept/reject decisions follow the same rules (LLT stops at the first pivot <= 0,
LDLT at an exactly zero pivot), but a pivot within rounding of zero can land on the other side
under a different ordering.

Also, once the fallback solver has switched to LDLT, `analyze_pattern`/`compute` with a new
pattern re-analyzes the LDLT stage here. COLMAP re-analyzes only the supernodal stage, so its
next `ldlt_.factorize` runs on a stale symbolic analysis (an upstream bug with undefined
results in Eigen). And the two `LOG(WARNING)`s (the LLT-to-LDLT fallback, and a failed A^T A
factorization in the LAD solver) are not emitted, because the core library has no logger; the
same facts are visible through `SparseCholeskyWithFallbackSolver::uses_ldlt` and
`LeastAbsoluteDeviationSolver::valid`.

**Why.** CHOLMOD and CSparse are GPL/LGPL and Eigen is MPL-2.0 (`docs/LICENSE_AUDIT.md`); none
of them can be ported. A supernodal code is a performance refinement over the simplicial one,
not a different result, and can replace it later behind the same API. Same as colmap-sharp
entry 13.

**Evidence.** `sparse_cholesky_test.cc` and `least_absolute_deviations_test.cc` pass 1:1
(`tests/optim/sparse_cholesky.rs`, `tests/optim/least_absolute_deviations.rs`), including the
singular, indefinite and ridge cases whose outcome depends on the pivot rules;
`rust_only_compute_new_pattern_after_fallback` pins the re-analysis. The Rust-only
`tests/linalg/rust_only_simplicial_cholesky.rs` pins the solutions to the dense LLT within 1e-10
relative error, and `rust_only_100k_grid_laplacian_work_counts_match_colmap_sharp` shows the
ordering and elimination are colmap-sharp's step for step (same nnz(L), AMD work and
factorization work on a 100k-unknown grid Laplacian).

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

**The same split on `Camera`** (`scene/camera.rs`). The methods over model metadata
(`verify_params`, `has_bogus_params`, `params_info`, the four index-group getters, the focal
length and principal point accessors and setters, `calibration_matrix`, `mean_focal_length`,
`is_undistorted`, `set_params_from_string`, `rescale(scale)`) return the `DomainError`;
`create_from_model_id` / `create_from_model_name` return COLMAP's `THROW_CHECK` error for an
unknown model. The per-point methods (`cam_from_img`, `cam_ray_from_img`,
`cam_from_img_threshold`, `img_from_cam`, `img_from_cam_with_jac`,
`cam_ray_from_img_with_jac`, the `is_*` model-kind queries, `rescale_to`) and the projection
functions in `scene/projection.rs` panic. `camera_test.cc`'s `EXPECT_THROW(.., domain_error)`
becomes an `Err` check on the first group and a caught panic on the second.

**Obligation on later phases.** Every path that creates a camera from outside data must
validate the id through the fallible functions before any per-point call: the camera readers
(text, binary, database) propagate the `DomainError`; nothing may call a per-point function on
an unverified `Camera`.

**Evidence.** `tests/sensor/models.rs` ports `models_test.cc` 1:1, including the
`ExistsCameraModelWithId(static_cast<CameraModelId>(123456789))` check through `from_i32`.
`tests/sensor/rust_only_models.rs` checks that every boundary function returns
`Err(DomainError, "Camera model does not exist")` for `Invalid` and that projection and
unprojection panic with that message.
`tests/scene/camera.rs` ports `camera_test.cc` 1:1 on that split, and
`tests/scene/rust_only_scene_types.rs` checks every metadata method of `Camera` for the
`DomainError`.

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

## 160. Track element access out of range panics instead of throwing `std::out_of_range`

**What differs.** COLMAP's `Track::Element(idx)` and `SetElement(idx, ..)` go through
`std::vector::at`, which throws `std::out_of_range` for a bad index. `Track::element`,
`element_mut` and `set_element` (`scene/track.rs`) index the `Vec` and panic.
`DeleteElement(idx)`, a `THROW_CHECK_LT` in COLMAP, stays an `Err`.

**Why.** No COLMAP code catches that exception: callers iterate `0..Length()`, so an
out-of-range index is a bug in the caller, an internal invariant where CLAUDE.md's error rule
allows a panic. A `Result` on every element access would burden the per-observation loops of
triangulation and bundle adjustment for a condition that cannot occur in correct code.

**Evidence.** `tests/scene/rust_only_scene_types.rs`:
`rust_only_track_element_out_of_range_panics` and
`rust_only_track_delete_element_out_of_range_fails_check`.

## 161. VisibilityPyramid defines its two undefined-behavior cases

**What differs.** Two inputs are undefined behavior in COLMAP's `VisibilityPyramid` and
defined in `scene/visibility_pyramid.rs`:
- `max_score_ += dim * dim * dim * dim` is `int` arithmetic and overflows from 8 levels on
  (dim 256, 2^32); here it is `usize`, so `max_score` is the sum the formula means.
- `CellForPoint` converts `max_dim * x / width` to `size_t` before clamping; for a negative or
  NaN coordinate that conversion is undefined. Rust's `as usize` saturates both to 0, so the
  point lands in the first cell (and a coordinate beyond the range in the last, as in COLMAP).

**Why.** Undefined behavior has no result to match; these are the values a caller would
expect. COLMAP's only user, the incremental mapper's observation manager, builds pyramids
with `kNumPoint3DVisibilityPyramidLevels = 6` levels and the image's size, where both versions
agree.

**Evidence.** `tests/scene/visibility_pyramid.rs` (1:1, 1 to 7 levels) passes unchanged;
`tests/scene/rust_only_scene_types.rs`: `rust_only_visibility_pyramid_eight_levels_max_score`
and `rust_only_visibility_pyramid_out_of_range_points_clamp`.

## 170. Frames and images store ids, not pointers to their rig, camera and frame

**What differs.** COLMAP's `Frame` holds a raw `Rig* rig_ptr_`, and its `Image` holds
`Camera* camera_ptr_` and `Frame* frame_ptr_`; the reconstruction sets them (`SetRigPtr`,
`SetCameraPtr`, `SetFramePtr`) and clears them (`Reset*Ptr`), and methods such as
`Frame::SensorFromWorld`, `Image::CamFromWorld` and `Image::ProjectPoint` follow them.
`scene/frame.rs` and `scene/image.rs` store only the ids; the owner of the maps passes the
objects to the methods that need them, and each method checks it was handed the right one:
- `Frame::sensor_from_world(&rig, ..)`, `set_cam_from_world(&rig, ..)`;
  `Image::cam_from_world(&frame, &rig)`, `projection_center`, `viewing_direction`,
  `is_ref_in_frame` (same parameters), `project_point(&frame, &rig, &camera, ..)`. Each fails
  unless the frame / camera has the image's id and the rig has the frame's rig id, where
  COLMAP fails on a null pointer.
- `SetRigPtr`, `SetCameraPtr`, `SetFramePtr` become `Frame::verify_rig`,
  `Image::verify_camera`, `Image::verify_frame`: the same checks, with nothing stored.
  Rebinding to another rig / camera / frame (COLMAP's `Set*Ptr` on an already bound object,
  which adopts the new id) is setting the id, then verifying.
- `AddDataId` on a frame with a rig (which checks the rig has the sensor) is
  `Frame::add_data_id_in_rig(&rig, ..)`; `add_data_id` is the unbound form.
- `Image::HasPose` takes the image's frame as an `Option` (`frames.get(&image.frame_id())`);
  `operator<<` is `Image::display(frame)`. `Image`'s `PartialEq` compares the image's own data;
  `Image::eq_with_frames` is COLMAP's `operator==`, which also compares the frames' poses.
- `RigPtr`, `CameraPtr`, `FramePtr`, `Has*Ptr` and `Reset*Ptr` have no counterpart, and
  `Frame::set_rig_id` / `Image::set_camera_id` / `Image::set_frame_id` drop COLMAP's
  `THROW_CHECK(!Has*Ptr())`: there is no binding to protect, and a frame or image inside a
  reconstruction is only reached through it.

**Why.** CLAUDE.md's translation rule: pointer graphs become ids into owning maps. Shared
mutable pointers between the reconstruction's maps cannot be expressed in safe Rust without
`Rc<RefCell<..>>` everywhere, which would make every pose read in bundle adjustment and the
mapper a runtime borrow check. Passing the owner keeps frames and images plain data that
clone, move between threads and serialize without fix-ups (COLMAP's reconstruction re-links
every pointer after a copy; with ids there is nothing to re-link).

**Evidence.** `tests/scene/frame.rs` and `tests/scene/image.rs` port `frame_test.cc` and
`image_test.cc` 1:1, with the pointer tests (`SetResetRigPtr`, `CameraPtr`, `FramePtr`,
`ConstructCopy`, `AssignCopy`) as behaviour tests of the id-based API under the C++ names;
`tests/scene/rust_only_frame_image.rs` checks that every owner-taking method rejects the
wrong rig, frame or camera, and the two equality forms.

## 171. Image point access out of range panics instead of throwing `std::out_of_range`

**What differs.** COLMAP's `Image::Point2D(idx)`, `SetPoint3DForPoint2D` and
`ResetPoint3DForPoint2D` go through `std::vector::at`, which throws `std::out_of_range` for a
bad index. `Image::point2d`, `point2d_mut`, `set_point3d_for_point2d` and
`reset_point3d_for_point2d` (`scene/image.rs`) index the `Vec` and panic.
`SetPoint3DForPoint2D`'s `THROW_CHECK_NE(point3D_id, kInvalidPoint3DId)` stays an `Err`.

**Why.** The same reason as entry 160 for `Track`: no COLMAP code catches the exception;
callers index with feature indices taken from the image itself or from its correspondences,
so an out-of-range index is a caller bug, an internal invariant where CLAUDE.md's error rule
allows a panic, and a `Result` on every point access would burden triangulation and bundle
adjustment.

**Evidence.** `tests/scene/rust_only_frame_image.rs`:
`rust_only_image_point2d_out_of_range_panics` and
`rust_only_image_set_point3d_out_of_range_panics`.

## 180. SIFT descriptors converted from float saturate to 0..255

**What differs.** `FeatureDescriptors::FromFloat` casts each SIFT value with
`static_cast<uint8_t>`, which is undefined behavior for a value outside `[0, 256)` or NaN.
`FeatureDescriptors::from_float` (`feature/types.rs`) uses Rust's `as u8`: it truncates toward
zero like `static_cast` for in-range values, clamps out-of-range values to 0 or 255, and maps
NaN to 0.

**Why.** Undefined behavior has no result to match. Saturation is the value a caller would
expect from a byte descriptor, and SIFT's own descriptors are already quantized to 0..255
before they reach this conversion, so in-range inputs (every input COLMAP produces) give
COLMAP's bytes.

**Evidence.** `tests/feature/types.rs` `feature_descriptors_sift_conversion` (1:1) passes;
`tests/feature/rust_only_feature_types.rs`
`rust_only_feature_descriptors_sift_cast_saturates` pins `[12.9, 255, 300, -4, NaN, 0.5]` →
`[12, 255, 255, 0, 0, 0]`.

## 181. The keypoint and match matrix conversions check their column count at run time

**What differs.** COLMAP's `KeypointsFromMatrix` and `MatchesFromMatrix` take
`Eigen::Ref<const Matrix<.., Dynamic, 4 or 2, RowMajor>>`, so the column count is part of the
type: a fixed-size source with the wrong count does not compile, and a dynamic one trips an
Eigen assertion (undefined behavior in a release build). `keypoints_from_matrix` and
`matches_from_matrix` (`feature/types.rs`) take a dynamic `RowMajorMatrix` and return the
`THROW_CHECK_EQ`-style `Err` for any other column count. `KeypointsFromMatrix` also returns the
keypoint constructor's `Err` for a negative scale, where COLMAP throws.

**Why.** The Rust matrix type has no compile-time column count; a checked error is the closest
equivalent and never reads past a row. Same as colmap-sharp, which checks the column count
the same way (`FeatureKeypoints.KeypointsFromMatrix` / `MatchesFromMatrix`).

**Evidence.** `tests/feature/rust_only_feature_types.rs`
`rust_only_feature_matrix_conversions_check_columns`.

## 200. ExtractTopScaleFeatures keeps equal-scale keypoints in input order

**What differs.** COLMAP's `ExtractTopScaleFeatures` (feature/utils.cc) selects the largest
scales with `std::partial_sort`, which leaves keypoints of equal scale in an unspecified,
implementation-defined order (libc++'s heap-based selection). `extract_top_scale_features`
(`feature/utils.rs`) sorts by scale, largest first, with the input index as the tie-break
(`f32::total_cmp`, so a NaN scale also has a fixed place, where COLMAP's `>` comparator is not
a strict weak order).

**Why.** Equal scales are common: the SIFT extractor emits one keypoint per orientation, and
all of them share the scale. Reproducing libc++'s heap order would tie the result to one
standard library; the index tie-break is deterministic and stable. Only the order among equal
scales differs, and *which* keypoints are kept differs only when a tie straddles the cut,
where COLMAP's choice is arbitrary too. Same as colmap-sharp entry 40.

**Evidence.** `tests/feature/utils.rs` `extract_top_scale_features_nominal` (1:1, distinct
scales) passes unchanged; `tests/feature/rust_only_feature_utils.rs`
`rust_only_extract_top_scale_features_ties_keep_input_order`.

## 201. SIFT never reuses a previous image's gradient

**What differs.** VLFeat caches the polar gradient of the current octave and records which
octave it holds in `grad_o`, which only `vl_sift_new` resets. A filter reused for a second
image of the same size keeps the first image's `grad_o`: when the first image's last octave
with keypoints is the second image's first octave, VLFeat's orientations and descriptors of the
second image's first octave are computed from the *first* image's gradient. COLMAP's
`SiftCPUFeatureExtractor` reuses its filter across images of equal size, so its results can
depend on the order images are processed. `VlSiftFilter::process_first_octave`
(`feature/vlfeat/sift_filter.rs`) resets `grad_o`, so every image gets its own gradient.

**Why.** It is a bug in VLFeat's reuse path: a feature's descriptor should depend only on its
own image. A fresh filter (what VLFeat computes for the first image) is the correct result.
Same as colmap-sharp entry 43 (the gradient part; its cancellation part is a Rust
`CancelToken` here and changes no numbers).

**Evidence.** `tests/feature/rust_only_vl_sift_filter.rs`
`rust_only_vl_sift_filter_reused_filter_matches_fresh_one` (fails with the reset removed).

## 202. SIFT's `fast_expn` table is a constant of correctly rounded values

**What differs.** VLFeat fills its 257-entry `exp(-x)` interpolation table (`sift.c`'s
`fast_expn_init`) by calling the platform `exp`. colmap-rust's rule (entry 1) routes every
transcendental through the `libm` crate, whose `exp` is 1 ulp from the correctly rounded value
for some of these 257 arguments. `feature/vlfeat/mathop.rs` instead holds the table as a
constant: each entry the correctly rounded double of `exp(-k * 25 / 256)`, written as bits.

**Why.** The table weights every orientation-histogram vote and descriptor sample, and a 1 ulp
difference in it moved SIFT orientations by an ulp against the VLFeat fixture. Apple libm,
which the pycolmap wheel and the fixture run, returns exactly the correctly rounded values for
all 257 arguments (checked against Python's `decimal` at 80 digits), so the constant table gives
VLFeat-on-macOS's bits on every target, native and wasm, which is what entry 1 aims for.

**Evidence.** `tests/feature/rust_only_vl_sift_filter.rs`: the six
`rust_only_vl_sift_filter_matches_unfused_vlfeat_*` tests are bit-exact with the table (with
`libm`'s `exp`, four of them fail on an orientation's last bit);
`rust_only_vl_sift_expn_table_is_exp` checks every entry within 1 ulp of `fns::exp`.

## 210. CorrespondenceGraph lists image pairs in insertion order

**What differs.** `CorrespondenceGraph::image_pairs` and `num_matches_between_all_images`
(`scene/correspondence_graph.rs`) enumerate the image pairs in the order
`add_two_view_geometry` added them. COLMAP iterates its `FlatHashMap<image_pair_t, ImagePair>`,
whose order is the hash table's: `std::unordered_map` or `boost::unordered_flat_map`, whichever
`COLMAP_HASH_MAP_BACKEND` the build picked, so it is not even the same across COLMAP builds.
`NumMatchesBetweenAllImages` returns a `NodeHashMap`; here it is a `Vec<(pair id, number of
matches)>`, since every COLMAP caller (`ObservationManager`, `PoseGraph::Load`,
`SceneClustering`, `DatabaseCache`) only iterates it.

**Why.** CLAUDE.md requires deterministic iteration, and reproducing either backend's bucket
layout would mean porting it; the order is not part of COLMAP's contract
(`correspondence_graph_test.cc` compares `ImagePairs()` with `UnorderedElementsAre` and reads
`NumMatchesBetweenAllImages()` by key). Insertion order rather than ascending pair id keeps
colmap-rust and colmap-sharp walking the pairs in the same order, so pipelines that iterate
them (the pose graph and the global mapper) agree between the two ports. Every per-point
correspondence list, `extract_matches_between_images` and `extract_transitive_correspondences`
are in COLMAP's order already (they come from vectors, not hash iteration).
Same as colmap-sharp entry 14.

**Evidence.** `tests/scene/correspondence_graph.rs` (`correspondence_graph_test.cc` 1:1)
passes; `tests/scene/rust_only_correspondence_graph.rs`
`rust_only_correspondence_graph_pairs_in_insertion_order` pins the order.

## 211. CorrespondenceGraph does not log dropped matches

**What differs.** COLMAP's `AddTwoViewGeometry` emits a `LOG(WARNING)` line for a self-match
(`image_id1 == image_id2`), for each match whose point index is out of bounds and for each
duplicate match. `CorrespondenceGraph::add_two_view_geometry` drops and counts them exactly as
COLMAP does, without the log lines.

**Why.** The core crate has no logging facility yet (COLMAP's glog is not ported); the lines are
diagnostic only, and the counts, which are the function's contract, are unchanged. When a
logger lands, this entry goes away (as entry 84).

**Evidence.** `tests/scene/correspondence_graph.rs` `correspondence_graph_out_of_bounds` and
`correspondence_graph_duplicate` (1:1) check the counts; `tests/scene/rust_only_correspondence_graph.rs`
`rust_only_correspondence_graph_self_match_is_ignored`.

## 212. A rejected AddTwoViewGeometry leaves the correspondence graph unchanged

**What differs.** COLMAP's `AddTwoViewGeometry` adds the match count to both images'
`num_correspondences` before its `THROW_CHECK(inserted)` for a pair that was already added
(and before `ImagePairToPairId` rejects an id beyond `kMaxNumImages`), so the throw leaves
both totals inflated. `CorrespondenceGraph::add_two_view_geometry` checks the
pair (and that both images exist) first, so an `Err` leaves the graph as it was.

**Why.** A failed call that half-applies is a bug: after it, the counts the incremental mapper
uses to pick its initial pair no longer match the stored correspondences. No caller relies on
the inflated totals.

**Evidence.** `tests/scene/rust_only_correspondence_graph.rs`
`rust_only_correspondence_graph_rejected_duplicate_pair_leaves_graph_unchanged` and
`rust_only_correspondence_graph_missing_image_is_out_of_range`.

## 213. Reconstruction iterates its objects in ascending id order

**What differs.** COLMAP's `Reconstruction` keeps rigs, cameras, frames, images and 3D points
in `NodeHashMap`s, whose iteration order is unspecified (it depends on the hash-map backend and
its history). `scene/reconstruction.rs` keeps them in `BTreeMap`s, which iterate in ascending
id order. Where that order reaches an output, the result can differ from a given COLMAP build:
- `find_image_with_name` returns the smallest-id image when several share a name, and so
  `find_common_reg_image_ids` pairs with that one.
- `rigs()` / `cameras()` / `frames()` / `images()` / `points3d()` and `point3d_ids()` enumerate
  in ascending id order for every caller; `compute_mean_reprojection_error` sums in that order.

`reg_frame_ids()` and `reg_image_ids()` are COLMAP's vector, in registration order, unchanged.

**Why.** CLAUDE.md requires deterministic iteration. COLMAP relies on no particular order (its
tests only use order-free properties), so ascending id is the natural deterministic choice and
needs no extra state. Same as colmap-sharp entry 21.

**Evidence.** `tests/scene/reconstruction.rs` (`reconstruction_test.cc` 1:1 for the ported
methods) passes; `tests/scene/rust_only_reconstruction.rs`
`rust_only_reconstruction_iterates_in_ascending_id_order` pins the order.

## 214. Changing an attached image's or frame's ids is caught by IsValid, not by the setter

**What differs.** In COLMAP, `Image::SetCameraId` and `Frame::SetRigId` throw once the object is
bound to a camera / rig pointer (`THROW_CHECK(!HasCameraPtr())`), so the image or frame inside a
reconstruction cannot be moved to another camera or rig by accident; `SetFrameId` has no such
guard. `Reconstruction::image_mut` / `frame_mut` hand out `&mut Image` / `&mut Frame`, and the
setters (entry 170) do not know the object is in a reconstruction, so such a change is not
rejected when made. `Reconstruction::is_valid` reports it (the camera, rig or frame must exist
and hold the object), and every owner-taking method of `Frame` / `Image` rejects mismatched
objects, so a broken association never yields a wrong pose silently.

**Why.** With ids into owning maps there is no pointer to guard; a guarded mutable view (a
wrapper type forwarding every safe setter) would double the `Image` / `Frame` API for a
mistake no COLMAP code makes (COLMAP's own code changes these ids only on objects outside a
reconstruction, or in `SetRigsAndFrames`, which re-verifies them). Flagged for review: if a
later port needs the guard, `image_mut` becomes that view.

**Evidence.** `tests/scene/rust_only_reconstruction.rs`
`rust_only_reconstruction_changed_ids_are_invalid`; `tests/scene/reconstruction.rs`
`reconstruction_is_valid` (the "empty frame pointer" case clears the image's frame id).

## 215. DeleteAllPoints2DAndPoints3D resets each image's triangulated count

**What differs.** COLMAP's `DeleteAllPoints2DAndPoints3D` clears the 3D points and calls
`image.SetPoints2D(std::vector<Eigen::Vector2d>(0))`, which resizes the image points but does not
touch `num_points3D_`, so every image that had triangulated points keeps a stale count: its
`NumPoints3D()` is still non-zero with no points left, and `IsValid()` is false.
`Reconstruction::delete_all_points2d_and_points3d` leaves each image's count at 0.

**Why.** A count of triangulated points on an image without points is a bug; it makes the
reconstruction fail its own validity check and inflates `ComputeNumObservations`. Nothing
relies on the stale value.

**Evidence.** `tests/scene/rust_only_reconstruction.rs`
`rust_only_reconstruction_delete_all_points2d_and_points3d_resets_counts` (COLMAP would leave
`num_points3d` at 1 and `is_valid` false).

## 216. Rejected Reconstruction operations leave the reconstruction unchanged

**What differs.** Several COLMAP `Reconstruction` operations throw part-way, after they changed
something:
- `AddPoint3D(point3D_id, point3D)` raises `max_point3D_id_` and marks the track's image points
  one by one before it fails on a missing image, an image point observing another 3D point, or
  an id already in use.
- `AddObservation` marks the image point before it looks up the 3D point.
- `DeletePoint3D` resets image points one by one before it fails on a missing track image.

Here each checks everything it can fail on first, so an `Err` leaves the reconstruction as it
was (including the next id `add_point3d` hands out).

**Why.** A failed call that half-applies is a bug: afterwards image points claim a 3D point
that does not exist (or no longer claim one that does), and `IsValid()` is false. No caller
relies on the partial state.

**Evidence.** `tests/scene/rust_only_reconstruction.rs`
`rust_only_reconstruction_failed_operations_leave_it_unchanged`.

## 220. re3q3's random change of variables uses a fixed-seed mt19937, not std::rand

**What differs.** PoseLib's `re3q3` (P4Pf's three-quadratics solver,
`estimators/solvers/poselib/re3q3.rs`, with the retry in `re3q3_var_change.rs`) retries a
near-degenerate system (all three elimination determinants below 1e-10) after a random affine
change of variables. PoseLib draws that from Eigen's `Quaternion::UnitRandom()` and
`setRandom()`, which read the C library's global `std::rand()`. The port draws a uniform
rotation (Shoemake, Graphics Gems III) and a shift in [-1, 1]^3 from a fresh `Mt19937` seeded
with 0 on every call, through libc++'s `uniform_real_distribution`
(`math::random::libcxx::uniform_real`).

**Why.** `std::rand()` is a process-global stream whose sequence depends on the C library and
on every earlier call in the process, so COLMAP's result in this branch is not reproducible
even against itself; porting it would add hidden global state shared across threads, which
CLAUDE.md rules out (sequential and parallel runs must agree). Eigen's `UnitRandom` code is
MPL-2.0 and is not ported, so the rotation follows the published method it cites. A fixed
per-call seed keeps the result a pure function of the input. Only the degenerate branch is
affected; the regular path is a straight port. Same as colmap-sharp entry 26.

**Evidence.** A change of variables followed by `refine_3q3` against the original
coefficients targets the same system, so the solution set agrees up to solver tolerance
whichever random matrix is drawn. `tests/estimators/rust_only_re3q3.rs`,
`rust_only_degenerate_elimination_uses_random_var_change` (x^2 = 1, y^2 = 4, z^2 = 9: every
elimination determinant is 0) finds all eight solutions, each satisfying the original system
to 1e-8. colmap-sharp cross-checked C++ PoseLib fa7280f (clang++, Eigen 3.4) in a scratch
harness: for `std::srand` seeds 1..20 it also returns 8 solutions with max residual below
3e-9, and for x^2 = 1, y^2 = 4, z = x + y (a linear third equation) it returns 0 solutions for
every seed, as the port does (upstream behavior, pinned by
`rust_only_linear_equation_returns_no_solutions_like_poselib`).

## 221. re3q3_rotation's pre-rotation uses a fixed-seed mt19937, not std::rand

**What differs.** PoseLib's `re3q3_rotation` (used by `gp3p`, which COLMAP's `GP3PEstimator`
calls; `estimators/solvers/poselib/re3q3.rs`, `solve_rotation`) rotates every problem by a
random rotation R0 before solving in Cayley parameters, so that no solution lands on the
Cayley transform's singularity (a rotation by pi). Unlike re3q3's change of variables (entry
220), this draw happens on *every* call, not only for degenerate input. PoseLib takes R0 from
Eigen's `Quaternion::UnitRandom()`, i.e. from the process-global `std::rand()`. The port draws
it (Shoemake's uniform rotation) from a fresh `Mt19937` seeded with 1 on every call.

**Why.** The same as entry 220: `std::rand()` is hidden global state whose sequence depends on
the C library and on every earlier call, so COLMAP's gp3p is not reproducible even against
itself, and porting it would make sequential and parallel runs disagree. Eigen's `UnitRandom`
is MPL-2.0 and not ported. The seed differs from entry 220's so the two draws are not the same
rotation. Consequence: for a fixed input the port always uses the same R0, so an input whose
true rotation is (close to) R0 composed with a rotation by pi is always ill-conditioned here,
where PoseLib would be only for an unlucky draw. That set has measure zero and RANSAC draws
other samples. Same as colmap-sharp entry 29.

**Evidence.** colmap-sharp ran PoseLib fa7280f's gp3p (clang++, Eigen 3.4) in a scratch
harness: it returns the same four poses for `std::srand` seeds 1..5 on a fixed three-ray
input, agreeing to about 1e-10 across seeds. `tests/estimators/rust_only_gp3p.rs`,
`rust_only_matches_poselib_solution_set` returns those same four poses to 1e-8.

## 230. CPU SIFT matches the macOS pycolmap wheel within FMA drift, not bit for bit

**What differs.** COLMAP's `SiftCPUFeatureExtractor` runs VLFeat, which the macOS arm64
pycolmap wheel compiles with Apple clang's default `-ffp-contract=on`: multiply-adds in the
Gaussian convolution, the DoG refinement and the descriptor binning are fused into FMAs that
round once. colmap-rust never fuses (CLAUDE.md "No FMA"), so against that wheel the extractor
(`feature/sift/cpu_extractor.rs`) gives the same keypoints, in the same order and count, with
last-bit drift: keypoint values differ by up to about 1e-3 and descriptor bytes by at most one
grey level, on at most 0.1% of the bytes.

**Why.** Fusing depends on the compiler and target, so it cannot be reproduced on every
platform (wasm has no guaranteed FMA); unfused VLFeat is the portable reference, and against it
the port is bit-exact. Same as colmap-sharp entry 41.

**Evidence.** `tests/feature/rust_only_vl_sift_filter.rs`: bit-exact against VLFeat compiled
with `-ffp-contract=off` (fixture `sift_vlfeat.json`). `tests/feature/rust_only_sift.rs`
`rust_only_sift_matches_pycolmap_*`: the nine `sift.json` cases against the wheel within
2e-3 per keypoint value, 1 per descriptor byte and 0.1% mismatched bytes.

## 231. FeatureDescriptorIndex is an exact nearest-neighbour search, not faiss's IVF index

**What differs.** COLMAP's `FeatureDescriptorIndex` (`feature/index.cc`) wraps faiss: an exact
`IndexFlatL2` for fewer than 512 indexed descriptors, and for 512 or more an inverted-file
index (`IndexIVFScalarQuantizer` with `QT_8bit_direct` for SIFT, `IndexIVFFlat` otherwise;
4 sqrt(N) k-means centroids, searched with `nprobe = 8`). The IVF search is approximate: it
only visits the 8 nearest clusters, so it can miss a query's true nearest or second-nearest
neighbour. The port (`colmap-rust/src/feature/index.rs`) always returns the exact k nearest
neighbours, ordered by (squared distance, index). SIFT descriptors are compared as integers
(exact); other descriptor types sum (a - b)^2 in `f32`, left to right, where faiss's flat index
uses the ||a||^2 + ||b||^2 - 2ab expansion.

**Why.** faiss is native code (CLAUDE.md contract 1). Porting it, k-means training included,
would reproduce a randomized approximation whose exact output also depends on faiss's OpenMP
partitioning and BLAS. The exact search is the result the IVF index approximates, and it makes
the default (index) matching path agree with COLMAP's own brute-force matcher, which is what
`SiftCPUFeatureMatcherFaissVsBruteForce` asserts. On an image pair with 512+ features, the SIFT
matches can therefore differ from COLMAP's default CPU matcher wherever faiss's approximation
missed a neighbour (COLMAP returns fewer or different matches there); they equal COLMAP's
`cpu_brute_force_matcher` matches except where acos-of-dot and sqrt-of-L2 round a threshold
comparison differently, which COLMAP's two paths also do. Ties in distance are ordered by
index; faiss leaves them unspecified, and the ratio test rejects a tie for the best neighbour
either way. Same as colmap-sharp entry 42.

**Evidence.** `tests/feature/index.rs` (`index_test.cc` 1:1, including the 1000-descriptor rows,
where COLMAP uses IVF) and `tests/feature/sift_matching.rs`
`sift_cpu_feature_matcher_faiss_vs_brute_force_nominal` (`sift_test.cc` 1:1) pass;
`rust_only_index_float_queries_on_byte_index_and_large_k_order` checks the order against the
definition.

## 240. Synthetic datasets visit points, images and chained pairs in ascending id order

**What differs.** COLMAP's `SynthesizeDataset` projects the reconstruction's 3D points into
each image, and `track_length` pruning visits them, in `Reconstruction::Points3D()` order,
which is the order of COLMAP's hash map; its CHAINED match configuration writes the pairs of a
`NodeHashMap`; `SynthesizeNoise` adds 2D and 3D point noise visiting `Images()` and
`Points3D()`, and `SynthesizeImages` renders the images in `Images()` order. `scene/synthetic.rs`,
`scene/synthetic_matches.rs` and `scene/synthetic_noise.rs` visit all of them in ascending id
order (the reconstruction's `BTreeMap`s, entry 213; a `BTreeMap` of pairs for CHAINED). The
PRNG draws are the same and in the same sequence; what differs is the order of each image's 2D
points before the shuffle, which point's track receives which pruning shuffle, which pair
receives which outlier-match shuffle, which point receives which noise draws, and the
descriptor seeds `SynthesizeImages` gives points without a 3D point. So a seeded run gives the
same cameras, rigs, frame poses and 3D point positions as pycolmap's, but different 2D point
orders, pruned tracks and noise assignment.

**Why.** CLAUDE.md requires deterministic iteration; COLMAP's tests rely only on order-free
properties (counts, track lengths, error bounds, connectivity, equality of two seeded runs).
Same as colmap-sharp entry 31.

**Evidence.** `tests/scene/synthetic.rs`, `synthetic_database.rs` and `synthetic_noise.rs`
(`synthetic_test.cc` 1:1, all 18 cases) and every Phase 4 test that synthesizes its scene pass.

## 241. Model file readers reject input COLMAP's streams misread

**What differs.** COLMAP's model readers use `std::istream`, which parses a value
character by character and does not check every read. The Rust readers:
- parse each text value as a whole whitespace-separated token (`util/line_tokens.rs`), so
  "12abc" fails where libc++ reads 12 and leaves "abc" for the next value;
- reject a sensor type number in `rigs.bin` / `frames.bin` that names no `SensorType`, and a
  camera model number that names no model (a Rust enum cannot hold an unnamed value);
- reject an image name or text line that is not UTF-8 (a Rust `String` cannot hold it);
- fail when a binary stream ends early (as entry 61), where COLMAP reads unspecified values
  or, for an image name, loops on the last character.

Valid files, including every file COLMAP writes, read to the same model.

**Why.** Each case is malformed input where COLMAP's result is unspecified or silently wrong;
an error is the only safe answer. Same as colmap-sharp entry 25 for the token parsing.

**Evidence.** `tests/scene/rust_only_reconstruction_io.rs`
`rust_only_reconstruction_io_rejects_what_cpp_misreads`; the pycolmap fixtures in
`tests/data/oracle/reconstruction_io` read and re-write byte for byte
(`rust_only_reconstruction_io_rewrites_pycolmap_files_byte_for_byte`).

## 242. Reading a model resets the reconstruction's registration

**What differs.** COLMAP's `Reconstruction::ReadText` / `ReadBinary` clear the rig, camera,
frame, image and 3D point maps, but keep the registered-frame list, the registered-image count
and the largest 3D point id. Reading into a reconstruction that already held registered frames
leaves frames that no longer exist registered, so `NumRegFrames()` is wrong and `IsValid()`
fails. Here `read_text` / `read_binary` start from an empty reconstruction.

**Why.** Stale registration of frames that are gone is a bug; nothing relies on it (COLMAP
reads into fresh reconstructions in practice).

**Evidence.** `tests/scene/rust_only_reconstruction_io.rs`
`rust_only_reconstruction_read_resets_registration` (COLMAP would report 5 registered frames
for a file with 3).

## 250. The five-point solver takes its null space from unpivoted Householder QR

**What differs.** PoseLib's `relpose_5pt` (the minimal case of COLMAP's
`EssentialMatrixFivePointEstimator`) takes the 4D null space of its five epipolar constraints
from the last four columns of the full Q of `epipolar_constraints.fullPivHouseholderQr()` (a
9 x 5 matrix). `estimators/solvers/poselib/relpose_5pt.rs` takes the same four columns from an
unpivoted Householder QR (`linalg::householder::factor_in_place` / `apply_q`). Both are
orthonormal bases of the same space; only the basis inside it differs, and with it the
intermediate polynomial coefficients (`compute_trace_constraints`, the degree-10 determinant
polynomial and its roots, which parametrize the solutions in basis coordinates).

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no published
convention precise enough to reproduce Eigen's Q column for column. The solution set does not
depend on the basis: the solver returns every unit-norm E = x N0 + y N1 + z N2 + N3 in the
null space that meets the cubic constraints, and a change of orthonormal basis maps that set
onto itself (up to the sign of each E). What can differ is numerical conditioning near the
back substitution's 1e-6 switch to the three-row QR solve and the Sturm bracketing tolerance,
where a near-degenerate root can be kept on one side and dropped on the other. Same as
colmap-sharp entry 27.

**Evidence.** `tests/estimators/rust_only_relpose_5pt.rs`,
`rust_only_recovers_true_essential_matrix`: on 100 random minimal problems every solution has
unit norm and satisfies the five epipolar constraints to 1e-8, and the true E is among the
solutions up to sign. colmap-sharp's 1:1 `EssentialMatrixFivePointEstimatorTests_Nominal`
cases pass on the same construction. Not compared against C++ PoseLib intermediates.

## 251. TinySphereManifold's tangent basis uses Hughes & Moller, not Eigen's unitOrthogonal()

**What differs.** COLMAP's `SphereManifold<3>` (`estimators/cost_functions/tiny_manifold.h`)
builds the tangent basis at x as `b1 = x_hat.unitOrthogonal()`, `b2 = x_hat.cross(b1)`.
`estimators/cost_functions/tiny_manifold.rs` (`TinySphereManifold`) builds `b1` with the
construction of Hughes & Moller, "Building an Orthonormal Basis from a Unit Vector" (JGT 4(4),
1999): zero the component of smallest magnitude (the first on a tie), swap the other two and
negate one, normalize. `b2` is the same cross product. The two constructions can pick a
different unit vector in the tangent plane, so the tangent coordinates of a step, and with them
TinySolver's Jacobi scaling and iterates, can differ from COLMAP's while spanning the same
plane.

**Why.** Eigen documents only that `unitOrthogonal()` returns some unit vector orthogonal to
the input; its choice for 3-vectors is an implementation detail of MPL-2.0 code we may not
transcribe (CLAUDE.md contract 2). Any orthonormal tangent basis gives a valid retraction and
Plus Jacobian; TinySolver is Tier C, so what must agree is the converged result, which does not
depend on the basis. Same as colmap-sharp entry 19.

**Evidence.** `tests/estimators/tiny_manifold.rs`, the 1:1
`sphere_manifold_plus_stays_on_unit_sphere` and `sphere_manifold_plus_jacobian_matches_finite_diff`
(orthogonality to x within 1e-12, the finite-difference Jacobian within 1e-6) and the product
manifold tests pass. colmap-sharp records the downstream effect of the same basis: where
TinySolver stops before converging (the 25-iteration cap of
`EssentialMatrixTangentSampsonEstimator`'s refinement, LO-RANSAC's ten refit rounds), the
relative poses its global-SfM oracle test re-estimates agree with pycolmap to 6.5e-7 deg.

## 252. TinyFocalSampsonErrorCostFunctor's Jacobian is closed-form, not Ceres autodiff

**What differs.** COLMAP's `TinyFocalSampsonErrorCostFunctor`
(`estimators/cost_functions/tiny_sampson_error.h`), the refiner of
`RelativePoseSharedFocalEstimator::Refine`, is a templated functor that TinySolver
differentiates through `ceres::TinySolverAutoDiffFunction` (Jets).
`estimators/cost_functions/tiny_sampson_error.rs` evaluates the same residual,
`SampsonError(F, point1, point2)` with F = diag(1/f, 1/f, 1) E diag(1/f, 1/f, 1), in the same
arithmetic order, and computes its 8-parameter Jacobian in closed form: dr/dF contracted with
dF/d[q, t, log_f], where the pose columns are COLMAP's dE/d[q, t] carried through the focal
scaling and d F/d(log_f) negates F with the top-left 2x2 block doubled and F(2, 2) zeroed. It
is the exact derivative of the same function, so it differs from autodiff's only by rounding;
TinySolver's iterates can therefore differ in the last bits.

**Why.** The closed form is the pattern COLMAP itself uses for the neighboring tangent,
one-sided focal and fundamental functors in the same header, and it is cheaper than autodiff.
TinySolver is Tier C, so what must agree is the converged result.

**Evidence.** `tests/estimators/tiny_sampson_error.rs`:
`tiny_focal_sampson_error_cost_functor_matches_squared_sampson_error` (the 1:1
`TinyFocalSampsonErrorCostFunctor.MatchesSquaredSampsonError`, residuals within 1e-9) and
`rust_only_tiny_focal_sampson_error_cost_functor_jacobian_matches_finite_difference` (every
column, log-focal included, against central differences within 1e-5 at two poses and focals;
scaling the log-focal column wrongly fails it). `tests/estimators/relpose_shared_focal.rs`'s
1:1 `relative_pose_shared_focal_estimator_refine_from_initial_model` converges from a 10%
focal error to within 1% on all 50 draws, and fails with that same mutation.

## 255. The six-point focal relative pose solvers take their null space from unpivoted Householder QR

**What differs.** PoseLib's `relpose_6pt_shared_focal` and `relpose_6pt_onesided_focal` (the
minimal solvers of COLMAP's `RelativePoseSharedFocalEstimator` and
`RelativePoseOneSidedFocalEstimator`) take the 3D null space of their six epipolar
constraints from the last three columns of the full Q of
`epipolar_constraints.fullPivHouseholderQr()` (a 9 x 6 matrix).
`estimators/solvers/poselib/relpose_6pt_shared_focal.rs` (`null_space3`, used by both) takes
the same three columns from an unpivoted Householder QR (`linalg::householder::factor_in_place`
/ `apply_q`). Both are orthonormal bases of the same space; only the basis inside it differs,
and with it the 280 (shared) or 190 (one-sided) template coefficients, the action matrix and
its eigenvalues, which parametrize F = N0 + x N1 + y N2 in basis coordinates. The one-sided
solver also orders its nine solutions by the eigenvalue order of `linalg::EigenSolver` (Schur
order), which may differ from Eigen's.

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR has no published
convention precise enough to reproduce Eigen's Q column for column (entry 250 is the same
choice for the five-point solver). In exact arithmetic the solution set does not depend on the
basis (apart from solutions whose F has no N0 component, which the affine parametrization
cannot represent; a measure-zero configuration that moves with the basis). In floating point
it can: the shared-focal solver brackets the real roots of an ill-conditioned degree-15
characteristic polynomial with Sturm sequences, and which spurious (non-physical) real roots
survive that bracketing depends on the basis, so extra models can appear or vanish on either
side. The output order is not part of the contract: COLMAP treats the models as a set. What
can differ is numerical conditioning near the solvers' thresholds (the Sturm bracketing
tolerance, the 1e-8 tests on imaginary parts and on q = 1/f^2). Same as colmap-sharp entry 30.

**Evidence.** `tests/estimators/rust_only_relpose_6pt_focal.rs`: on 50 random minimal problems
each, both solvers return the true pose and focal (relative error below 1e-4; well-conditioned
problems agree to 1e-12, one shared-focal problem was 5e-6 off). colmap-sharp's differential
run against the real PoseLib (its entry 30) found the one-sided models identical and the
shared-focal models matching except for spurious roots. Not compared against C++ PoseLib here.

## 256. The generalized six-point solver returns its poses in Schur eigenvalue order

**What differs.** PoseLib's `gen_relpose_6pt` (the minimal solver of COLMAP's `GR6PEstimator`)
takes the real eigenvalues (|imag| < 1e-6) of its 64 x 64 action matrix from
`Eigen::EigenSolver` and returns one pose per eigenvalue that passes cheirality, in Eigen's
eigenvalue order. `estimators/solvers/poselib/gen_relpose_6pt.rs` takes them from
`linalg::EigenSolver` (colmap-sharp's Hessenberg + Francis QR port), whose eigenvalue order
(Schur order) and last-bit rounding may differ from Eigen's. It also computes
`C0^-1 C1` column by column through `linalg::PartialPivLu` rather than colmap-sharp's fused
augmented elimination, which colmap-sharp documents as bit-identical.

**Why.** Eigen (MPL-2.0) is not ported. Only eigenvalues are used, never eigenvectors, so the
eigenvector phase freedom cannot reach the result; the solution set is the same up to
rounding. The order is not part of the contract (RANSAC treats the models as a set). What can
differ is an eigenvalue whose imaginary part sits at the 1e-6 threshold, kept on one side and
dropped on the other. Same choice as colmap-sharp's `GenRelpose6pt.cs` (which records it in
its file header, not as a numbered entry).

**Evidence.** `tests/estimators/rust_only_gen_relpose_6pt.rs`
`rust_only_gen_relpose_6pt_recovers_true_pose`: on 50 random minimal problems between two
generalized cameras the true metric pose is among the solutions to 1e-6. Not compared against
C++ PoseLib.

## 260. The 7-point fundamental solver takes its null space from unpivoted Householder QR

**What differs.** `FundamentalMatrixSevenPointEstimator::Estimate` gets the 2D null space of
its 7 x 9 constraint system from columns 7 and 8 of the full Q of `A.fullPivHouseholderQr()`
(A is 9 x 7). `estimators/solvers/fundamental_matrix.rs` takes the same two columns from an
unpivoted Householder QR of A (`linalg::householder::factor_in_place` / `apply_q`,
allocation-free). Both pairs are orthonormal bases of the same space: in any factorization
A = Q R, the first seven columns of Q contain A's column space, so the last two are orthogonal
to every constraint, whatever the pivoting and whatever A's rank. Only the basis inside that
plane differs.

**Why.** Eigen (MPL-2.0) is not ported, and full-pivoting Householder QR (which also swaps
rows) has no published convention precise enough to reproduce Eigen's Q column for column
(entries 250 and 255 are the same choice for the five- and six-point solvers). The basis does
not reach the solutions: the solver finds the rank-2 members of the pencil {a f1 + b f2},
normalized to unit norm, and that set of matrices (up to sign, which the Sampson error and
COLMAP's F / F(2, 2) comparisons ignore) does not depend on the basis. The one place it shows
is the parametrization lambda f1 + f2 with f1 = q7 - q8: when |det(f1)| < 1e-16 COLMAP returns
no model, and which member of the pencil f1 is depends on the basis. For a generic sample that
is a measure-zero event on either side, so the two can disagree only on a sample that sits on
that numeric knife edge, and then one of them emits no model for that RANSAC hypothesis. Same
as colmap-sharp entry 24.

**Evidence.** `tests/estimators/fundamental_matrix.rs`:
`fundamental_seven_point_estimator_reference` (COLMAP's Matlab reference values, 1e-6) and
`fundamental_seven_point_estimator_nominal` (100 random problems, at least one model equal to
the true F up to scale) pass 1:1.

## 261. GR8P returns its four candidate poses in Schur eigenvector order

**What differs.** `GR8PEstimator::Estimate` reads the translations of its four candidate poses
from the eigenvectors of the 4 x 4 matrix G (`Eigen::EigenSolver<Matrix4d>`,
`V.real().colwise().hnormalized()`) and returns them in Eigen's eigenvector order.
`estimators/solvers/generalized_relative_pose_gr8p.rs` takes them from `linalg::EigenSolver`
(colmap-sharp's Hessenberg + Francis QR port), whose eigenvalue order (Schur order) and
last-bit rounding may differ from Eigen's, so the four candidates can come in a different
order.

**Why.** Eigen (MPL-2.0) is not ported, and its deflation order is not a documented
convention. G is symmetric (`ComposeG` mirrors every off-diagonal entry), so every eigenvalue
is real and so is its eigenvector, and `hnormalized()` divides out the scale and sign that are
the only freedom left: the candidate set is the same up to rounding. RANSAC scores all four,
so the order only matters when two candidates have exactly the same support and score, where
the first one wins. Same choice as colmap-sharp's `GeneralizedRelativePose.GR8P.cs` (which
records it in its file header, not as a numbered entry).

**Evidence.** `tests/estimators/generalized_relative_pose.rs`: the seven
`..._gr8p_*` instantiations of COLMAP's `ParameterizedGRNPEstimatorTests.GR8P` pass 1:1
(RANSAC over 16 correspondences finds the true `rig2_from_rig1` within rtol and ttol 1e-2 in
one of ten trials). Not compared against C++ Eigen.

## 262. IsPanoramicRig compares rig camera origins to the smallest camera index

**What differs.** `IsPanoramicRig` (estimators/generalized_pose.cc) decides whether all
cameras used by a set of correspondences share one optical center: it puts the camera indices
in a `FlatHashSet`, takes the set's first element as the reference, and checks every other
camera's origin in the rig with `isApprox(reference, 1e-6)`. The set's iteration order is
unspecified. `estimators/generalized_pose.rs` takes the smallest camera index as the
reference. Same as colmap-sharp entry 46.

**Why.** Hash iteration order is not reproducible across containers (CLAUDE.md, hash
iteration order). The reference only matters when the origins agree to about 1e-6 relative
with one camera but not with another (`isApprox` is not transitive), which never happens for a
real rig, whose cameras are either co-centered (panoramic) or centimeters apart.

**Evidence.** `tests/estimators/generalized_pose.rs`: COLMAP's
`EstimateGeneralizedRelativePose.Nominal`, which covers panoramic and non-panoramic rigs, and
`EstimateStructureLessAbsolutePose.PanoramicWorldCameras` pass 1:1.

## 263. The generalized pose estimators return no pose for empty or misaligned input

**What differs.** In estimators/generalized_pose.cc, `ThrowCheckCameras` dereferences
`std::minmax_element` of the camera index list, and `EstimateStructureLessAbsolutePose` calls
`IsPanoramicRig`, which dereferences the first element of the index set: both are undefined
behavior for an empty list. `EstimateGeneralizedRelativePose` also reads `camera_idxs1[i]` and
`camera_idxs2[i]` for every point without checking their lengths. In
`estimators/generalized_pose.rs`, an empty index list has nothing to check,
`estimate_structure_less_absolute_pose` returns `Ok(None)` (COLMAP's `false`) for no
correspondences, as the other two estimators already do, and
`estimate_generalized_relative_pose` fails a check (`camera_idxs1.len() == num_points`) when
an index list is shorter or longer than the points.

**Why.** An out-of-range read is a bug, not behavior to reproduce. Returning "no pose" for no
correspondences matches what COLMAP's other two estimators do, and colmap-sharp makes the same
choice (in comments, not as a numbered entry). Results for valid input are unchanged.

**Evidence.** `tests/estimators/rust_only_generalized_pose.rs`
(`rust_only_generalized_pose_empty_correspondences_find_no_pose`,
`rust_only_generalized_relative_pose_short_camera_idxs_is_an_error`).

## 264. Two-view geometry estimation does not log

**What differs.** COLMAP's `EstimateTwoViewGeometry` emits a one-time `LOG_FIRST_N(WARNING)`
when it marks a pair degenerate because `force_H_use` was asked of a non-pinhole camera, and
another when it marks an uncalibrated fisheye pair degenerate (asking for a focal length prior);
`MaybeDecomposeRelativePoses` logs `LOG(INFO)` lines with its decomposed and failed counts and
its run time. `colmap_rust::estimators::two_view_geometry` returns the same geometries without
these lines, and drops the `Timer` that only fed the last one.

**Why.** The core crate has no logging facility yet (COLMAP's glog is not ported); the lines are
diagnostic only and every returned geometry, configuration and stored pose is unchanged. When a
logger lands, this entry goes away (as entries 84 and 211).

**Evidence.** `tests/estimators/two_view_geometry.rs`
`estimate_two_view_geometry_uncalibrated_fisheye_is_degenerate` and
`estimate_two_view_geometry_force_h_use_with_fisheye_is_degenerate`, and
`tests/estimators/two_view_geometry_misc.rs` `maybe_decompose_relative_poses_*`
(`two_view_geometry_test.cc` 1:1) pass.

## 265. The database is one in-memory type, not a factory over SQLite files

**What differs.** COLMAP's `Database` (scene/database.h) is an abstract class whose only
implementation is `SqliteDatabase` (database_sqlite.cc), opened through a path-keyed factory
registry (`Database::Open`, `Database::Register`); `kInMemorySqliteDatabasePath` opens an
in-memory SQLite database. `colmap_rust::scene::database::Database` is one concrete type that
keeps its tables in memory: `Database::new()` stands for `Open(kInMemorySqliteDatabasePath)`,
and there is no `Open(path)`, no `Register` and no file. Its methods take `&self` (the tables
sit behind a mutex, so a database can be shared between threads like COLMAP's `shared_ptr`) and
return `Result`, so a closed database fails every call as COLMAP's does. `BeginTransaction` /
`EndTransaction` only check that the database is open: every call is applied at once, and
COLMAP never rolls a transaction back. `DatabaseTransaction` still holds the transaction lock,
so only one is active at a time.

**Why.** SQLite database files are out of scope (PORTING_PLAN.md: SQLite is native code), and a
trait with a single implementation would only add indirection. Everything the SQLite schema
makes observable is reproduced: AUTOINCREMENT ids for rigs, cameras, frames and images (one more
than the largest id the table ever held, explicit ids included; clearing does not reset it), the
pose priors' plain INTEGER PRIMARY KEY (one more than the largest current id), the UNIQUE
indices (image names, a sensor in at most one rig as reference and one as non-reference sensor,
a data id in at most one frame, one pose prior per data id, one keypoints / descriptors /
matches / two-view geometry row per image or pair), the foreign keys with their cascades
(clearing rigs drops their frames, clearing images their keypoints and descriptors, clearing
cameras fails while images exist), rows read in id order and pairs in pair id order, UPDATE of a
missing row doing nothing, reads of a missing row returning defaults, and the two-view geometry
columns (`tri_angle` is not stored, so a read returns its default).

**Evidence.** `tests/scene/database.rs` (database_test.cc 1:1; the three `OpenFile*` cases are
skipped as file-based) and `tests/scene/rust_only_database.rs`. Same as colmap-sharp's
`InMemoryDatabase` (its `Scene/InMemoryDatabase.cs` header).

## 266. A database write that violates a constraint stores nothing

**What differs.** COLMAP's `WriteRig` / `UpdateRig` and `WriteFrame` / `UpdateFrame` issue one
INSERT for the rig or frame row and then one per sensor or data id, each in SQLite's autocommit
mode; when a later INSERT violates a constraint (say, a sensor already in another rig), COLMAP
throws after the earlier rows were committed, leaving a rig or frame with part of its sensors or
data. Here every write checks all constraints first, so a failing call leaves the database
unchanged. Consequences:
- On successful writes the assigned ids, what is stored and every read are the same.
- After such a partially failed write COLMAP has consumed the new id (the row exists), so its
  next automatic id is one higher; here the id was never used and the next write gets it.
- The same calls fail, but the messages differ: COLMAP's `std::runtime_error` reads "SQLite
  error: constraint failed" (`sqlite3_errstr`); here it is `ErrorKind::RuntimeError` worded like
  SQLite's extended messages ("SQLite error: UNIQUE constraint failed: images.name", "SQLite
  error: FOREIGN KEY constraint failed"). `THROW_CHECK` sites keep COLMAP's check messages.
- An automatic id past `u32::MAX` fails ("the next id does not fit a 32-bit id") where COLMAP
  truncates SQLite's 64-bit rowid to the 32-bit id.

**Why.** The partially written state is an accident of statement-at-a-time execution, not
behavior any COLMAP caller relies on (all of them treat the exception as fatal), and reproducing
it would mean emulating SQLite's statement boundaries. A truncated id would silently alias an
existing row.

**Evidence.** `tests/scene/rust_only_database.rs` `rust_only_failed_rig_write_stores_nothing`
and `rust_only_image_names_are_unique_and_need_a_camera`; database_test.cc's cases that expect a
constraint violation to fail (`PosePrior`, `TwoViewGeometry`) pass. Same as colmap-sharp entry
23.

## 267. DatabaseCache iterates its objects in ascending id order

**What differs.** COLMAP's `DatabaseCache` keeps rigs, cameras, frames and images in
`NodeHashMap`s, whose iteration order depends on the hash-map backend of the build.
`colmap_rust::scene::database_cache::DatabaseCache` keeps them in `BTreeMap`s, so `rigs()`,
`cameras()`, `frames()` and `images()` enumerate in ascending id order, like `Reconstruction`
(entry 213). Where that order reaches an output: `find_image_with_name` returns the
smallest-id image when several share a name (possible only through `add_image`; database names
are unique), and the correspondence graph receives its images in ascending id order (the order
of `AddImage` calls is not observable). Pairs are added in the database's pair id order in
`load` and in the source graph's insertion order (entry 210) in `create_from_cache`, both as in
COLMAP. `Load` keeps COLMAP's `emplace` semantics: an object already in the cache is not
replaced.

**Why.** CLAUDE.md requires deterministic iteration. COLMAP relies on no particular order here
(its tests use order-free properties), so ascending id is the natural choice.

**Evidence.** `tests/scene/database_cache.rs` (database_cache_test.cc 1:1) passes with COLMAP's
expectations; `rust_only_create_from_cache_keeps_whole_frames` pins the filtering. Same as
colmap-sharp entry 33.

## 268. SynthesizeImages hands bitmaps to a sink instead of writing image files

**What differs.** COLMAP's `SynthesizeImages(options, reconstruction, image_path)` writes each
image to `image_path / image.Name()` with `Bitmap::Write` (OpenImageIO) and logs a failed
write. `colmap_rust::scene::synthetic_noise::synthesize_images(options, reconstruction,
write_image)` calls `write_image(image.name(), bitmap)` for each rendered RGB bitmap; the
pixels are the same.

**Why.** Image file encoding is not ported: the host owns image I/O (PORTING_PLAN.md, the
skipped `bitmap_test.cc` file cases). A sink lets a caller encode files with its own encoder or
keep the bitmaps in memory, natively and in the browser. Same as colmap-sharp entry 32.

**Evidence.** `tests/scene/synthetic_noise.rs` `synthesize_images_nominal` checks every image
reaches the sink with its camera's width and height (COLMAP's test reads each PNG back and
checks the same).

## 269. SynthesizeDataset rejects pose priors without a database

**What differs.** COLMAP's `SynthesizeDataset` with `prior_position` or `prior_gravity` set
writes each image's pose prior through `database->WritePosePrior` without checking the
pointer, so calling it without a database dereferences null (undefined behavior, in practice a
crash). `colmap_rust::scene::synthetic::synthesize_dataset` (the database-free form) returns
an `InvalidArgument` error, "SynthesizeDataset: pose priors need a database", at the first
image instead.

**Why.** A null dereference is a bug, not behavior to reproduce; an error is the only safe
answer, and the database form behaves exactly as COLMAP's.

**Evidence.** `tests/scene/synthetic_database.rs`
`rust_only_synthesize_dataset_pose_priors_need_a_database`.

## 273. TranscribeImageIdsToDatabase changes nothing when it fails

**What differs.** COLMAP's `Reconstruction::TranscribeImageIdsToDatabase` renames the images
one by one while it looks their names up in the database, so when a name is missing
(`LOG(FATAL_THROW)`) or two images map to one id (`THROW_CHECK`), the images before it already
carry their new ids while the frames and tracks still refer to the old ones: the reconstruction
is left broken. `colmap_rust::scene::reconstruction_database::transcribe_image_ids_to_database`
looks every id up, and checks every frame data id and track element has a mapping, before it
changes anything, so a failure returns the error and leaves the reconstruction as it was. The
successful result is the same.

**Why.** A half-applied renumbering is a bug (`IsValid` fails afterwards); checking first costs
one extra pass and makes the error recoverable.

**Evidence.** `tests/scene/reconstruction_database.rs`
`rust_only_transcribe_image_ids_to_database_unknown_name_changes_nothing`; COLMAP's
TranscribeImageIdsToDatabase test passes 1:1.

## 274. ReadRigConfig requires exactly 4 rotation and 3 translation entries

**What differs.** COLMAP's `ReadRigConfig` (scene/rig.cc) writes the entries of
`cam_from_rig_rotation` and `cam_from_rig_translation` into a fixed-size `Eigen::Vector4d` and
`Rigid3d::translation()` by running index, without checking the count.
`colmap_rust::scene::rig::read_rig_config` returns a "Check failed" error ("... must have 4
entries" / "... must have 3 entries") unless there are exactly 4 and 3 entries.

**Why.** Fewer entries leave COLMAP with uninitialized coefficients and more entries write out
of bounds; both are undefined behavior with no result to match. Every well-formed config reads
the same. Same as colmap-sharp entry 80.

**Evidence.** `tests/scene/rig.rs` (rig_test.cc 1:1) passes, including
`read_rig_config_nominal`, which reads a 4-entry rotation and a 3-entry translation;
`tests/scene/rust_only_rig.rs` `rust_only_read_rig_config_pose_needs_four_and_three_entries`.

## 275. The rig config JSON reader limits nesting depth

**What differs.** COLMAP reads rig configs with `boost::property_tree::read_json`, whose parser
recurses once per nested array or object without a limit, so a deeply nested file overflows the
stack. `colmap_rust::util::json::parse_json`, the core's own reader that stands in for it,
returns an error ("nesting deeper than 512 levels") past `MAX_DEPTH` (512) nested arrays and
objects. Every document within the limit parses as Boost's parser does (strict JSON).

**Why.** A stack overflow is a crash, natively and in the browser (whose wasm stack is small);
an error is the only safe answer. Real rig configs nest three levels.

**Evidence.** `tests/util/rust_only_json.rs` `rust_only_json_nesting_limit` (512 levels parse,
513 fail, a million fail without a crash).

## 276. ApplyRigConfig hands the reconstruction its rigs and frames in id order

**What differs.** COLMAP's `UpdateRigsAndFramesFromDatabase` (scene/rig.cc) collects the
reconstruction's rigs and frames in `NodeHashMap`s and passes them to
`Reconstruction::SetRigsAndFrames` in the maps' iteration order, and `AddFrame` registers posed
frames in that order, so it decides the order of `RegFrameIds()`.
`colmap_rust::scene::rig_apply::apply_rig_config` collects them in `BTreeMap`s, so the frames
are registered in ascending frame id order.

**Why.** In COLMAP 4.2.0 `NodeHashMap` is `boost::unordered_node_map` or `std::unordered_map`,
depending on the build (util/hash_containers.h). Either way its iteration order is
implementation-defined: it depends on the build configuration, the hash and the standard
library, so there is no single order to match, and CLAUDE.md requires a deterministic order.
The sets of rigs, frames and registered frames do not depend on it. Same as colmap-sharp
entry 81.

**Evidence.** `tests/scene/rig.rs` `apply_rig_config_*` (rig_test.cc 1:1) pass with COLMAP's
counts of rigs, frames and registered frames, none of which depends on the order;
`tests/scene/rust_only_rig.rs`
`rust_only_apply_rig_config_derives_sensor_from_rig_from_reconstruction` pins the ascending
registration order.

## 277. Reconstruction alignment and merging use deterministic orders and do not log

**What differs.** Three choices in `estimators/alignment.cc` depend on hash-map iteration order
in COLMAP:
- `AlignReconstructionsViaPoints` counts, per source 3D point, how often each target 3D point is
  seen along its track in a `FlatHashMap` and takes `std::max_element` over it, so among target
  points with the same count the hash order picks one.
  `colmap_rust::estimators::alignment::align_reconstructions_via_points` keeps the counts in
  first-seen order along the track and takes the first maximum.
- `MergeReconstructions` copies the missing source images in the iteration order of a
  `FlatHashSet`, which decides the order their frames are registered in the target
  (`RegFrameIds()`). `estimators::alignment::merge_reconstructions` copies them in the source's
  `RegImageIds()` order. It merges the source points in ascending id order (entry 213), which
  decides the ids the new target points get.
- `AlignReconstructionToOrigRigScales` sums the per-rig scales over a `NodeHashMap` of rigs;
  `align_reconstruction_to_orig_rig_scales` takes a `BTreeMap` and sums in ascending rig id
  order, so the floating-point sum is the same on every run.

COLMAP's `LOG(WARNING)` (too few pose priors, no prior covariance), `LOG(ERROR)` (inconsistent
image id and name in a merge), `LOG(INFO)` and `VLOG` lines are not emitted (the core has no
logger); the return values are COLMAP's.

**Why.** The hash orders are implementation-defined (they depend on the build's hash-map
backend and its history), so there is no single COLMAP order to match, and CLAUDE.md requires a
deterministic order. The selected point, the set of merged images and points and the scale
mean do not depend on the order beyond ties and summation rounding. Same as colmap-sharp
entry 49.

**Evidence.** `tests/estimators/alignment.rs` (`alignment_test.cc` 1:1) passes with COLMAP's
values and tolerances, including `MergeReconstructions`' counts of rigs, cameras, frames,
images, points and observations and `AlignReconstructionToOrigRigScales`' recovered rig
extrinsics.

## 280. Reading a truncated binary PLY mesh fails inside the texcoord lists

**What differs.** COLMAP's `ReadPlyMesh` (util/ply.cc) checks `file.good()` after every vertex,
face count and face index it reads from a binary file, but not after a face's texcoord count
or its six UVs. A file cut there leaves those values indeterminate: usually the count is not 6
and the read fails with "Expected 6 texture coordinates per triangular face", a cut inside a
middle face's UVs fails at the next face, and a cut inside the last face's UVs loads the mesh
with garbage UVs and no error. `colmap_rust::util::ply::read_ply_mesh` checks the count and
every UV and fails with "Unexpected end of PLY file at face i" for the face whose texcoords are
cut, in all three cases. Same as colmap-sharp entry 113.

**Why.** The same robustness gap as entry 61: a half-written mesh must not load with garbage
texture coordinates, and the error should name the cause and the face. Well-formed files read
identically.

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
browser (as for `SynthesizeImages`, entry 268). The core is single-threaded (CLAUDE.md,
Threading).

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
