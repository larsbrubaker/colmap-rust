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
