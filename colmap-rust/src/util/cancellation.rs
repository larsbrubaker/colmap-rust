//! Cancellation and progress for long-running work: port of COLMAP's
//! `src/colmap/util/cancellation.h/.cc` `CancellationToken`, plus the progress callback
//! type that CLAUDE.md ("Cancellation and progress") requires alongside it, because the app
//! shows progress and lets the user cancel, and the browser must stay responsive.
//! colmap-sharp uses .NET's `CancellationToken` and `IProgress<double>` for the same job.
//! Tests: `tests/util/cancellation.rs` (`cancellation_test.cc`).
//!
//! Not ported: `ScopedSignalHandler`, which installs process-wide SIGINT/SIGTERM handlers
//! for COLMAP's command-line tool. It needs libc signal APIs (not in std, and absent on
//! wasm) and belongs to a CLI host, which can cancel a [`CancelToken`] from its own handler.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use super::check::{ColmapError, ErrorKind, Result};

/// Port of `colmap::CancellationToken`. Clones share one flag, so the token handed to a
/// worker is cancelled from the UI by cancelling any clone.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    is_cancelled: Arc<AtomicBool>,
}

impl CancelToken {
    /// A token that is not cancelled.
    pub fn new() -> Self {
        Self::default()
    }

    /// Port of `CancellationToken::Cancel`.
    pub fn cancel(&self) {
        // COLMAP's `store(true)` is sequentially consistent.
        self.is_cancelled.store(true, Ordering::SeqCst);
    }

    /// Port of `CancellationToken::IsCancelled`.
    pub fn is_cancelled(&self) -> bool {
        self.is_cancelled.load(Ordering::SeqCst)
    }

    /// `Err` with [`ErrorKind::Cancelled`] once cancelled, so a cancelled pipeline unwinds
    /// with `?` at its polling points.
    pub fn check_cancelled(&self) -> Result<()> {
        if self.is_cancelled() {
            return Err(ColmapError::new(
                ErrorKind::Cancelled,
                "Operation cancelled",
            ));
        }
        Ok(())
    }
}

/// A progress callback: receives the fraction done, in `[0, 1]`, from whichever thread runs
/// the work. Long-running entry points take `Option<&Progress<'_>>`; the lifetime lets the
/// callback borrow the caller's state.
pub type Progress<'a> = dyn Fn(f64) + Send + Sync + 'a;
