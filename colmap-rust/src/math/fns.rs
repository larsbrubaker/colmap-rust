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
//! The cost is that results can differ from COLMAP/pycolmap on macOS (Apple libm) by 1-2 ulp:
//! docs/CPP_DIVERGENCES.md, entry 1, which records the probe evidence. The guarantee is pinned
//! by `tests/math/fns_probe.rs` against `tests/data/fns_probe_expected{,_f32}.txt`.
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
