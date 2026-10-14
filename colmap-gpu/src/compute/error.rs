//! The compute seam's error: the .NET exceptions colmap-sharp's `IComputeDevice` documents
//! (`ArgumentException`, `InvalidOperationException`, `ArgumentOutOfRangeException`,
//! `ObjectDisposedException`, `OperationCanceledException`) become [`ComputeErrorKind`]s, so a
//! caller (and a test) can tell a broken rule from a GPU fault from a cancellation from a machine
//! with no GPU.

use std::fmt;

/// Which rule a [`ComputeError`] reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeErrorKind {
    /// A broken argument rule: misalignment, a range past a buffer's end, a foreign handle, a
    /// binding that does not match the layout, WGSL that does not compile (`ArgumentException`).
    InvalidArgument,
    /// A broken state rule or a device limit, and every GPU fault: a write to a buffer a pending
    /// dispatch binds, a buffer over `maxBufferSize`, a GPU error, a lost device
    /// (`InvalidOperationException`).
    InvalidOperation,
    /// A workgroup count over the device's per-dimension limit (`ArgumentOutOfRangeException`).
    OutOfRange,
    /// A bind group whose buffer has been dropped (`ObjectDisposedException`). Other handles cannot
    /// be used once dropped: Rust's ownership rules that out at compile time.
    Disposed,
    /// The call's [`colmap_rust::util::cancellation::CancelToken`] was already cancelled: the
    /// pending work was submitted, but not waited for (`OperationCanceledException`).
    Cancelled,
    /// No GPU adapter could be opened, so this machine (or browser) cannot run compute work at
    /// all; a caller falls back to the CPU. C# threw `InvalidOperationException` here too; the
    /// kind of its own lets a caller (and a GPU test) tell "no GPU" from a broken device.
    NoAdapter,
}

/// An error from a compute device call: what kind of rule it broke, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeError {
    pub kind: ComputeErrorKind,
    pub message: String,
}

impl ComputeError {
    pub fn new(kind: ComputeErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    pub(crate) fn argument(message: impl Into<String>) -> Self {
        Self::new(ComputeErrorKind::InvalidArgument, message)
    }

    pub(crate) fn operation(message: impl Into<String>) -> Self {
        Self::new(ComputeErrorKind::InvalidOperation, message)
    }
}

impl fmt::Display for ComputeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ComputeError {}

/// The result of a compute device call.
pub type ComputeResult<T> = Result<T, ComputeError>;
