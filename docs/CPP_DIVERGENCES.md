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
