//! The handles `WebGpuComputeDevice` (`device.rs`) hands out: port of MatterCAD's
//! `Tests/ColmapGpuTests/WebGpuComputeHandles.cs`. Each wraps the wgpu object it stands for and
//! remembers what the seam's rules need that the wgpu object does not carry: the owning device,
//! the buffer kind, the kernel's per-group layout, the bind group's entries.
//!
//! Dropping a handle releases the caller's reference only; wgpu objects are reference counted,
//! so recorded or in-flight work that binds it keeps it alive until it has run, as the seam
//! promises. A dropped kernel, bind group or buffer cannot be named in a later call (ownership
//! rules that out); a bind group whose buffer was dropped is refused at dispatch, as C#'s
//! `ObjectDisposedException` was.

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use crate::compute::{ComputeBufferKind, ComputeKernelBinding};

/// What a buffer's handle and every bind group that binds it share.
pub(crate) struct BufferShared {
    pub(crate) id: u64,
    pub(crate) device_id: u64,
    pub(crate) buffer: wgpu::Buffer,
    pub(crate) kind: ComputeBufferKind,
    pub(crate) size: u64,
    pub(crate) label: Option<String>,
    pub(crate) dropped: AtomicBool,
}

impl BufferShared {
    pub(crate) fn is_dropped(&self) -> bool {
        self.dropped.load(Ordering::SeqCst)
    }
}

impl fmt::Display for BufferShared {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "buffer '{}' ({:?}, {} bytes)",
            self.label.as_deref().unwrap_or("unlabeled"),
            self.kind,
            self.size
        )
    }
}

/// A storage or uniform buffer. Port of `WebGpuComputeBuffer`.
pub struct WebGpuComputeBuffer {
    pub(crate) shared: Arc<BufferShared>,
}

impl WebGpuComputeBuffer {
    /// Size in bytes.
    pub fn size(&self) -> u64 {
        self.shared.size
    }

    /// What the buffer can be bound as.
    pub fn kind(&self) -> ComputeBufferKind {
        self.shared.kind
    }

    /// The debug name it was created with.
    pub fn label(&self) -> Option<&str> {
        self.shared.label.as_deref()
    }
}

impl Drop for WebGpuComputeBuffer {
    fn drop(&mut self) {
        self.shared.dropped.store(true, Ordering::SeqCst);
    }
}

impl fmt::Display for WebGpuComputeBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.shared.fmt(f)
    }
}

/// A compiled compute pipeline with its explicit layout. Port of `WebGpuComputeKernel`.
pub struct WebGpuComputeKernel {
    pub(crate) device_id: u64,
    pub(crate) pipeline: wgpu::ComputePipeline,
    pub(crate) label: String,
    /// Each group's bindings, sorted by binding index.
    pub(crate) group_layouts: Vec<Vec<ComputeKernelBinding>>,
    pub(crate) bind_group_layouts: Vec<wgpu::BindGroupLayout>,
}

impl WebGpuComputeKernel {
    /// The kernel's debug name.
    pub fn label(&self) -> &str {
        &self.label
    }
}

impl fmt::Display for WebGpuComputeKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "kernel '{}'", self.label)
    }
}

/// One bound buffer range of a bind group.
pub(crate) struct BoundEntry {
    pub(crate) buffer: Arc<BufferShared>,
}

/// One bind group of a kernel's layout. Port of `WebGpuComputeBindGroup`.
pub struct WebGpuComputeBindGroup {
    pub(crate) device_id: u64,
    pub(crate) group: u32,
    pub(crate) bind_group: wgpu::BindGroup,
    /// The layout the group was made for, sorted by binding; compared by value at dispatch.
    pub(crate) layout: Vec<ComputeKernelBinding>,
    /// The entries, sorted by binding so entry i matches `layout[i]`.
    pub(crate) entries: Vec<BoundEntry>,
}

impl WebGpuComputeBindGroup {
    /// The group index it binds as.
    pub fn group(&self) -> u32 {
        self.group
    }
}

impl fmt::Display for WebGpuComputeBindGroup {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "bind group {}", self.group)
    }
}
