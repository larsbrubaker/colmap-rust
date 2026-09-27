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
