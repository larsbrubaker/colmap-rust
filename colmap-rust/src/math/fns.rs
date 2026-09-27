//! The single choke point for transcendental functions (CLAUDE.md, "Transcendentals").
//!
//! Every `sin`/`cos`/`exp`/`log`/`atan2`/`pow`/... in the port calls these instead of the
//! `f64` methods, so the implementation behind them can be pinned in one place. That matters
//! because std's math functions are not the same code on every target: native targets call
//! the platform's C libm (Apple's libm on macOS, glibc on Linux, the UCRT on Windows), while
//! `wasm32-unknown-unknown` uses compiler-builtins' Rust port of musl's libm. None of them
//! promises correct rounding, so the last bit can differ between targets, and Tier A code
//! must give the same bits natively and in the browser.
//!
//! Backend: std for now. The Phase 0 probe (`tests/math/fns_probe.rs`, expected bits in
//! `tests/data/fns_probe_expected.txt`, generated on macOS aarch64) measures where the
//! targets disagree. Measured when it was written: a `wasm32-unknown-unknown` build of these
//! functions differs from the macOS table on 490 of 7154 probes (every function but `sqrt`,
//! 1-2 ulp); the pure-Rust `libm` crate gives identical bits native and on wasm, but differs
//! from Apple libm on 435 of them. Switching every function to one implementation is a
//! one-file change here.
//!
//! `sqrt` is IEEE-754 correctly rounded on every target; it is here only so callers have one
//! import for all of `<cmath>`.

/// `std::sin`.
#[inline]
pub fn sin(x: f64) -> f64 {
    x.sin()
}

/// `std::cos`.
#[inline]
pub fn cos(x: f64) -> f64 {
    x.cos()
}

/// `std::tan`.
#[inline]
pub fn tan(x: f64) -> f64 {
    x.tan()
}

/// `std::asin`.
#[inline]
pub fn asin(x: f64) -> f64 {
    x.asin()
}

/// `std::acos`.
#[inline]
pub fn acos(x: f64) -> f64 {
    x.acos()
}

/// `std::atan`.
#[inline]
pub fn atan(x: f64) -> f64 {
    x.atan()
}

/// `std::atan2(y, x)`.
#[inline]
pub fn atan2(y: f64, x: f64) -> f64 {
    y.atan2(x)
}

/// `std::exp`.
#[inline]
pub fn exp(x: f64) -> f64 {
    x.exp()
}

/// `std::log` (natural logarithm).
#[inline]
pub fn ln(x: f64) -> f64 {
    x.ln()
}

/// `std::log2`.
#[inline]
pub fn log2(x: f64) -> f64 {
    x.log2()
}

/// `std::log10`.
#[inline]
pub fn log10(x: f64) -> f64 {
    x.log10()
}

/// `std::pow(x, y)` for a double exponent. (Rust's `powi` is not `std::pow(double, int)`:
/// it may use repeated multiplication and differ in the last bit, so ports use this.)
#[inline]
pub fn pow(x: f64, y: f64) -> f64 {
    x.powf(y)
}

/// `std::sqrt`: correctly rounded everywhere; a passthrough for one-stop imports.
#[inline]
pub fn sqrt(x: f64) -> f64 {
    x.sqrt()
}

/// `std::cbrt`.
#[inline]
pub fn cbrt(x: f64) -> f64 {
    x.cbrt()
}

/// `std::hypot(x, y)`.
#[inline]
pub fn hypot(x: f64, y: f64) -> f64 {
    x.hypot(y)
}
