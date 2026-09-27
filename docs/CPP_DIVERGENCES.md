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
