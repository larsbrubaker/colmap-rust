//! The parts of COLMAP's `src/colmap/util/threading.h/.cc` the port needs: so far
//! `GetEffectiveNumThreads`. COLMAP's `Thread`, `ThreadPool`, `JobQueue` and
//! `ThreadSafeQueue` are not ported: parallel loops use rayon behind the `parallel` feature
//! at each call site (CLAUDE.md "Threading"), as colmap-sharp uses `Parallel.For`
//! (`ColmapSharp/Util/Threading.cs`), and cancellation is [`super::cancellation`].
//! Tests: `tests/util/threading.rs` (the `GetEffectiveNumThreads` case of
//! `threading_test.cc`).

/// Port of `colmap::GetEffectiveNumThreads`: a non-positive count means all hardware
/// threads, and the result is at least 1. `std::thread::available_parallelism` stands in
/// for `std::thread::hardware_concurrency`; where it is unknown (e.g. `wasm32-unknown-unknown`)
/// it counts as 0, which COLMAP also maps to 1.
pub fn get_effective_num_threads(num_threads: i32) -> i32 {
    let mut num_effective_threads = num_threads;
    if num_threads <= 0 {
        num_effective_threads = std::thread::available_parallelism()
            .map(|n| i32::try_from(n.get()).unwrap_or(i32::MAX))
            .unwrap_or(0);
    }
    if num_effective_threads <= 0 {
        num_effective_threads = 1;
    }
    num_effective_threads
}
