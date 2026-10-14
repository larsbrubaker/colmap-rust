//! The compute device contract: port of colmap-sharp `ColmapSharp/Compute/IComputeDevice.cs`.
//! The value types it takes are in `descriptors.rs`, its limits in `limits.rs`, its errors in
//! `error.rs`; `crate::webgpu::WebGpuComputeDevice` implements it over wgpu.

use std::future::Future;

use colmap_rust::util::cancellation::CancelToken;

use super::{
    ComputeBufferBinding, ComputeBufferKind, ComputeDeviceLimits, ComputeKernelDescriptor,
    ComputeResult,
};

/// A WebGPU-shaped compute device: storage and uniform buffers, WGSL kernels, bind groups,
/// recorded dispatches, and asynchronous flush and readback. Implemented by the host.
///
/// **Recording and ordering.** [`dispatch`](Self::dispatch) only records. Recorded dispatches
/// reach the GPU at the next [`flush`](Self::flush) (or [`read_buffer`](Self::read_buffer), which
/// flushes first) and run in recording order: every dispatch sees the storage writes of every
/// dispatch recorded before it, with no barrier to spell out. Workgroups within one dispatch run
/// in any order.
///
/// **Writes are not ordered against dispatches.** [`write_buffer`](Self::write_buffer) is a
/// queue write: it lands before the next flush executes *any* of its dispatches, including
/// dispatches recorded before the write. So never write a buffer that a dispatch recorded since
/// the last flush binds; every such dispatch would see the new bytes. Per-dispatch parameters get
/// their own buffer (created with initial data) and bind group. The device refuses the hazard
/// rather than letting it corrupt results silently.
///
/// **Writable bindings are exclusive.** WebGPU checks buffer usage per whole buffer for the whole
/// dispatch, so a buffer bound read_write in any of a dispatch's groups may not be bound again
/// anywhere in that dispatch, not even read-only over a disjoint range. Read-only and uniform
/// bindings may share a buffer freely.
///
/// **Buffers.** Both kinds can be written and read back.
///
/// **Alignment.** Buffer sizes, write and read offsets and lengths, and binding sizes are
/// multiples of 4. Binding offsets are multiples of
/// [`ComputeDeviceLimits::min_storage_buffer_offset_alignment`].
///
/// **Limits.** Every call that would cross one of [`limits`](Self::limits) fails instead of
/// failing on the GPU, where WebGPU would only report it out of band and a readback would return
/// stale or zero bytes.
///
/// **Errors.** Argument and state errors return `Err` from the call. GPU execution and map
/// failures, including a WGSL compile or pipeline error the device learns of only asynchronously
/// (browser WebGPU), fail the next [`flush`](Self::flush) or [`read_buffer`](Self::read_buffer).
///
/// **Async.** `flush` and `read_buffer` complete when the GPU has finished. On a desktop device
/// they complete at once; in the browser they genuinely wait for the JS event loop, so when
/// [`supports_blocking_wait`](Self::supports_blocking_wait) is false a caller must `.await` them
/// and never block on them. Cancellation is observed only on entry: a call whose token is already
/// cancelled still submits every pending write and dispatch, then fails with
/// [`super::ComputeErrorKind::Cancelled`] (a cancelled read copies nothing into its destination).
/// So after any flush or read returns, nothing is pending and the device is reusable.
///
/// **Ownership.** The host owns the device; callers drop only the handles they created. Dropping
/// a handle that recorded or in-flight work binds is allowed: that work keeps it alive until it
/// has run.
pub trait ComputeDevice {
    /// A buffer this device created.
    type Buffer;
    /// A compiled kernel.
    type Kernel;
    /// One bind group of a kernel's layout.
    type BindGroup;

    /// The limits this device enforces, fixed at creation.
    fn limits(&self) -> &ComputeDeviceLimits;

    /// True when a caller may block a thread until `flush` or `read_buffer` completes (a desktop
    /// device). False in the browser, where blocking would deadlock.
    fn supports_blocking_wait(&self) -> bool;

    /// Creates a buffer of `size` bytes (a positive multiple of 4, at most `max_buffer_size`),
    /// filled with `initial` and zero after it.
    fn create_buffer(
        &self,
        kind: ComputeBufferKind,
        size: u64,
        initial: &[u8],
        label: Option<&str>,
    ) -> ComputeResult<Self::Buffer>;

    /// Writes `data` at `offset` as a queue write; see the ordering rule above.
    fn write_buffer(
        &mut self,
        buffer: &Self::Buffer,
        offset: u64,
        data: &[u8],
    ) -> ComputeResult<()>;

    /// Compiles a kernel from WGSL with an explicit bind group layout.
    fn create_kernel(&self, descriptor: &ComputeKernelDescriptor) -> ComputeResult<Self::Kernel>;

    /// Creates group `group` of `kernel`'s layout, one entry per declared binding in any order.
    fn create_bind_group(
        &self,
        kernel: &Self::Kernel,
        group: u32,
        entries: &[ComputeBufferBinding<'_, Self::Buffer>],
    ) -> ComputeResult<Self::BindGroup>;

    /// Records one dispatch of `x * y * z` workgroups, with `groups[i]` bound as group `i`.
    fn dispatch(
        &mut self,
        kernel: &Self::Kernel,
        groups: &[&Self::BindGroup],
        x: u32,
        y: u32,
        z: u32,
    ) -> ComputeResult<()>;

    /// Submits everything recorded and completes once the GPU has run it.
    fn flush<'a>(
        &'a mut self,
        cancel: Option<&'a CancelToken>,
    ) -> impl Future<Output = ComputeResult<()>> + 'a;

    /// Flushes, then copies `destination.len()` bytes of `buffer` from `offset` into
    /// `destination`.
    fn read_buffer<'a>(
        &'a mut self,
        buffer: &'a Self::Buffer,
        offset: u64,
        destination: &'a mut [u8],
        cancel: Option<&'a CancelToken>,
    ) -> impl Future<Output = ComputeResult<()>> + 'a;
}
