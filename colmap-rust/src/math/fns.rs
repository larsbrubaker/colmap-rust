//! The single choke point for transcendental functions (CLAUDE.md, "Transcendentals").
//!
//! Every `sin`/`cos`/`exp`/`log`/`atan2`/`pow`/... in the port calls these instead of the
//! `f64`/`f32` methods. std's versions are not the same code on every target: native targets
//! call the platform C libm (Apple libm on macOS, glibc on Linux, the UCRT on Windows), while
//! `wasm32-unknown-unknown` uses compiler-builtins' Rust port of musl's libm, and they differ
//! in the last bit or two. Tier A code must give the same bits natively and in the browser,
//! so every function here is the pure-Rust `libm` crate's, which is the same code on every
//! target.
//!
//! Deliberate divergence from COLMAP: COLMAP calls `std::sin`, `std::exp`, `std::log`,
//! `std::atan2`, `std::pow` and friends (and their `float` overloads), which resolve to the
//! platform's C libm; the pycolmap oracle runs on macOS, so that is Apple libm. Results here can
//! differ from COLMAP/pycolmap on macOS by 1-2 ulp for the same input. Apple libm is closed
//! source and cannot be reproduced in pure Rust, so no backend could match the oracle bit for
//! bit anyway; being identical across our own targets is the property we can have.
//!
//! Evidence: the probe `tests/math/fns_probe.rs` (7154 f64 probes of sin, cos, tan, asin, acos,
//! atan, atan2, exp, log, log2, log10, pow, sqrt, cbrt, hypot; special values and seeded random
//! inputs in COLMAP's ranges), run on macOS aarch64 and on a `wasm32-unknown-unknown` build
//! under node:
//! - std on wasm32 vs std on macOS (Apple libm): 490 differ (sin 10, cos 14, tan 101, asin 15,
//!   acos 45, atan 17, atan2 126, exp 15, log 5, log2 2, log10 2, pow 30, cbrt 18, hypot 90;
//!   sqrt 0), 1 ulp except tan (up to 2 ulp).
//! - `libm` crate vs Apple libm: 435 differ (same pattern; hypot 35).
//! - `libm` crate native aarch64 vs wasm32: 0 differ, for the f64 table and for the 8654-probe
//!   f32 table (`sinf` ... `hypotf`).
//!
//! The probe tables (`tests/data/fns_probe_expected{,_f32}.txt`) pin the `libm` crate's bits;
//! CI runs them on Linux, macOS and Windows, natively and as a `wasm32-wasip1` build.
//!
//! Consequence for tests: a Tier A oracle comparison of code that calls a transcendental
//! (camera-model undistortion, SIFT's Gaussian weights, angle conversions) may need a tolerance
//! of a few ulp instead of bit equality, and such a test cites this module where it states that
//! tolerance. Code without transcendentals stays bit-exact.
//!
//! The f64 functions use Rust's names (`ln` is C's `log`); the f32 ones use C's `<math.h>`
//! names (`sinf`, `logf`, ...), since that is how COLMAP's float code spells them.
//! `sqrt`/`sqrtf` are IEEE-754 correctly rounded everywhere; they are here so callers have
//! one import for all of `<cmath>`.

/// `std::sin`.
#[inline]
pub fn sin(x: f64) -> f64 {
    libm::sin(x)
}

/// `std::cos`.
#[inline]
pub fn cos(x: f64) -> f64 {
    libm::cos(x)
}

/// `std::tan`.
#[inline]
pub fn tan(x: f64) -> f64 {
    libm::tan(x)
}

/// `std::asin`.
#[inline]
pub fn asin(x: f64) -> f64 {
    libm::asin(x)
}

/// `std::acos`.
#[inline]
pub fn acos(x: f64) -> f64 {
    libm::acos(x)
}

/// `std::atan`.
#[inline]
pub fn atan(x: f64) -> f64 {
    libm::atan(x)
}

/// `std::atan2(y, x)`.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    libm::atan2(y, x)
}

/// `std::exp`.
#[inline]
pub fn exp(x: f64) -> f64 {
    libm::exp(x)
}

/// `std::log` (natural logarithm).
#[inline]
pub fn ln(x: f64) -> f64 {
    libm::log(x)
}

/// `std::log2`.
#[inline]
pub fn log2(x: f64) -> f64 {
    libm::log2(x)
}

/// `std::log10`.
#[inline]
pub fn log10(x: f64) -> f64 {
    libm::log10(x)
}

/// `std::pow(x, y)`.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// `std::sqrt`.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    libm::sqrt(x)
}

/// `std::cbrt`.
#[inline]
pub fn cbrt(x: f64) -> f64 {
    libm::cbrt(x)
}

/// `std::hypot(x, y)`.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    libm::hypot(x, y)
}

/// C++17's three-argument `std::hypot(x, y, z)` as libc++ computes it (the library the
/// macOS pycolmap wheel links; libm has no three-argument form): sqrt(x^2 + y^2 + z^2) on
/// arguments pre-scaled by a power of two when the largest magnitude could overflow (above
/// 2^512) or underflow (below 2^-512) the squares. Power-of-two scaling is exact, so in range
/// this is the plain square root of the left-to-right sum of squares. Ceres' rotation
/// conversions (`AngleAxisToQuaternion`, `QuaternionToAngleAxis`) call it. Port of
/// colmap-sharp's `ScalarMath.Hypot`.
pub fn hypot3(x: f64, y: f64, z: f64) -> f64 {
    // numeric_limits<double>::max_exponent / 2 = 1024 / 2. Exact powers of two, built from
    // their bits: 2^512 and 2^-(512 + 20).
    let overflow_threshold = f64::from_bits((1023 + 512) << 52);
    let overflow_scale = f64::from_bits((1023 - 532) << 52);
    let max_abs = x.abs().max(y.abs()).max(z.abs());
    let scale = if max_abs > overflow_threshold {
        overflow_scale
    } else if max_abs < 1.0 / overflow_threshold {
        1.0 / overflow_scale
    } else {
        1.0
    };
    let (x, y, z) = (x * scale, y * scale, z * scale);
    sqrt(x * x + y * y + z * z) / scale
}

/// `std::sinh`.
#[inline]
pub fn sinh(x: f64) -> f64 {
    libm::sinh(x)
}

/// `std::cosh`.
#[inline]
pub fn cosh(x: f64) -> f64 {
    libm::cosh(x)
}

/// `std::atanh`.
#[inline]
pub fn atanh(x: f64) -> f64 {
    libm::atanh(x)
}

/// `sinf`.
#[inline]
pub fn sinf(x: f32) -> f32 {
    libm::sinf(x)
}

/// `cosf`.
#[inline]
pub fn cosf(x: f32) -> f32 {
    libm::cosf(x)
}

/// `tanf`.
#[inline]
pub fn tanf(x: f32) -> f32 {
    libm::tanf(x)
}

/// `asinf`.
#[inline]
pub fn asinf(x: f32) -> f32 {
    libm::asinf(x)
}

/// `acosf`.
#[inline]
pub fn acosf(x: f32) -> f32 {
    libm::acosf(x)
}

/// `atanf`.
#[inline]
pub fn atanf(x: f32) -> f32 {
    libm::atanf(x)
}

/// `atan2f(y, x)`.
#[inline]
pub fn atan2f(y: f32, x: f32) -> f32 {
    libm::atan2f(y, x)
}

/// `expf`.
#[inline]
pub fn expf(x: f32) -> f32 {
    libm::expf(x)
}

/// `logf`.
#[inline]
pub fn logf(x: f32) -> f32 {
    libm::logf(x)
}

/// `log2f`.
#[inline]
pub fn log2f(x: f32) -> f32 {
    libm::log2f(x)
}

/// `log10f`.
#[inline]
pub fn log10f(x: f32) -> f32 {
    libm::log10f(x)
}

/// `powf(x, y)`.
#[inline]
pub fn powf(x: f32, y: f32) -> f32 {
    libm::powf(x, y)
}

/// `sqrtf`.
#[inline]
pub fn sqrtf(x: f32) -> f32 {
    libm::sqrtf(x)
}

/// `cbrtf`.
#[inline]
pub fn cbrtf(x: f32) -> f32 {
    libm::cbrtf(x)
}

/// `hypotf(x, y)`.
#[inline]
pub fn hypotf(x: f32, y: f32) -> f32 {
    libm::hypotf(x, y)
}
