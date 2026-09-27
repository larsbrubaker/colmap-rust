//! Port of COLMAP's `src/colmap/util/timer.h/.cc`: a pausable stopwatch used for the
//! elapsed-time reports of the reconstruction pipelines. Tests: `tests/util/timer.rs`
//! (`timer_test.cc`).
//!
//! `std::chrono::high_resolution_clock` becomes [`now_nanos`]: `std::time::Instant` natively.
//! `wasm32-unknown-unknown` has no clock without JavaScript (`Instant::now()` panics there),
//! so on that target the host must install a monotonic source with [`set_clock_source`]
//! (e.g. from `performance.now()`); with none installed there, the clock stands still and
//! every elapsed time reads 0. `docs/CPP_DIVERGENCES.md` entry 62.
//!
//! Not ported: `PrintSeconds` / `PrintMinutes` / `PrintHours`, which only `LOG(INFO)` the
//! elapsed time; the core crate has no logger. Callers format
//! `format!("Elapsed time: {:.5} [seconds]", timer.elapsed_seconds())` themselves.

use std::sync::OnceLock;

static CLOCK_SOURCE: OnceLock<fn() -> u64> = OnceLock::new();

/// Installs the monotonic clock (nanoseconds since any fixed origin) that [`now_nanos`]
/// reads. Needed only on `wasm32-unknown-unknown`, where std has no clock; on other targets
/// an installed source replaces `std::time::Instant` as well. The first installed source
/// wins; returns `false` if one was already set.
pub fn set_clock_source(source: fn() -> u64) -> bool {
    CLOCK_SOURCE.set(source).is_ok()
}

/// The current time of the monotonic clock, in nanoseconds since an arbitrary origin.
pub fn now_nanos() -> u64 {
    if let Some(source) = CLOCK_SOURCE.get() {
        return source();
    }
    platform_now_nanos()
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn platform_now_nanos() -> u64 {
    use std::time::Instant;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = *EPOCH.get_or_init(Instant::now);
    // u64 nanoseconds cover 584 years of process uptime.
    epoch.elapsed().as_nanos() as u64
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn platform_now_nanos() -> u64 {
    // No clock source installed and no std clock on this target: time stands still.
    0
}

/// Port of `colmap::Timer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Timer {
    started: bool,
    paused: bool,
    start_time: u64,
    pause_time: u64,
}

impl Timer {
    /// A timer that has not been started: every elapsed time is 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Port of `Timer::Start`.
    pub fn start(&mut self) {
        self.started = true;
        self.paused = false;
        self.start_time = now_nanos();
    }

    /// Port of `Timer::Restart`.
    pub fn restart(&mut self) {
        self.started = false;
        self.start();
    }

    /// Port of `Timer::Pause`.
    pub fn pause(&mut self) {
        self.paused = true;
        self.pause_time = now_nanos();
    }

    /// Port of `Timer::Resume`: shifts the start by the paused duration.
    pub fn resume(&mut self) {
        self.paused = false;
        self.start_time = self
            .start_time
            .wrapping_add(now_nanos().wrapping_sub(self.pause_time));
    }

    /// Port of `Timer::Reset`.
    pub fn reset(&mut self) {
        self.started = false;
        self.paused = false;
    }

    /// Port of `Timer::ElapsedMicroSeconds`: whole microseconds (`duration_cast` truncates).
    pub fn elapsed_micro_seconds(&self) -> f64 {
        if !self.started {
            return 0.0;
        }
        let end = if self.paused {
            self.pause_time
        } else {
            now_nanos()
        };
        (end.saturating_sub(self.start_time) / 1000) as f64
    }

    /// Port of `Timer::ElapsedSeconds`.
    pub fn elapsed_seconds(&self) -> f64 {
        self.elapsed_micro_seconds() / 1e6
    }

    /// Port of `Timer::ElapsedMinutes`.
    pub fn elapsed_minutes(&self) -> f64 {
        self.elapsed_seconds() / 60.0
    }

    /// Port of `Timer::ElapsedHours`.
    pub fn elapsed_hours(&self) -> f64 {
        self.elapsed_minutes() / 60.0
    }
}
