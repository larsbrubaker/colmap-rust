// Port of the CancellationToken case of COLMAP's src/colmap/util/cancellation_test.cc,
// testing colmap_rust::util::cancellation. The five ScopedSignalHandler cases
// (RecordsFirstSignal, SecondSignalTerminatesImmediately,
// RestoresPreviousHandlerAndClearsState, RejectsNestedInstances, ThreadHandlesSignal) are
// not ported: ScopedSignalHandler installs process signal handlers for COLMAP's CLI and is
// not part of the library (src/util/cancellation.rs header).

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Mutex;

use colmap_rust::util::cancellation::{CancelToken, Progress};
use colmap_rust::ErrorKind;

#[test]
fn cancellation_token_cancel() {
    let token = CancelToken::new();
    assert!(!token.is_cancelled());
    token.cancel();
    assert!(token.is_cancelled());
}

// Rust-only: clones share the flag across threads, and check_cancelled turns it into an
// error that `?` propagates.
#[test]
fn rust_only_cancel_token_shared_across_threads() {
    let token = CancelToken::new();
    let worker_token = token.clone();
    let polls = AtomicU32::new(0);
    std::thread::scope(|scope| {
        let worker = scope.spawn(|| -> colmap_rust::Result<()> {
            loop {
                polls.fetch_add(1, Ordering::SeqCst);
                worker_token.check_cancelled()?;
                std::thread::yield_now();
            }
        });
        while polls.load(Ordering::SeqCst) == 0 {
            std::thread::yield_now();
        }
        token.cancel();
        let err = worker.join().unwrap().unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Cancelled);
    });
    assert!(CancelToken::new().check_cancelled().is_ok());
}

// Rust-only: a progress callback can be passed as `&Progress<'_>` and called from any thread.
#[test]
fn rust_only_progress_callback() {
    let seen = Mutex::new(Vec::new());
    let record = |fraction: f64| seen.lock().unwrap().push(fraction);
    let progress: &Progress<'_> = &record;
    std::thread::scope(|scope| {
        scope.spawn(|| progress(0.5)).join().unwrap();
    });
    progress(1.0);
    assert_eq!(*seen.lock().unwrap(), [0.5, 1.0]);
}
