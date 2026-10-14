//! colmap-gpu: wgpu compute for colmap-rust.
//!
//! - [`compute`]: the compute seam, a port of colmap-sharp's `ColmapSharp.Compute`
//!   (`IComputeDevice.cs`, `ComputeDescriptors.cs`, `ComputeDeviceLimits.cs`): a WebGPU-shaped
//!   device of storage and uniform buffers, WGSL kernels, bind groups, recorded dispatches and
//!   asynchronous flush and readback, with every rule checked in the call.
//! - [`webgpu`]: [`webgpu::WebGpuComputeDevice`], the seam over a dedicated wgpu device: a port of
//!   MatterCAD's `Tests/ColmapGpuTests/WebGpuComputeDevice.cs` and `WebGpuComputeHandles.cs`,
//!   which implemented the same seam over agg-sharp's WebGPU device.
//!
//! The WGSL PatchMatch kernels (colmap-sharp `Mvs/PatchMatchGpu*.cs`, `Mvs/Shaders/`) run through
//! this seam. Tests: `tests/compute.rs` (`WebGpuComputeDeviceTests`).

pub mod compute;
pub mod webgpu;

/// The wgpu this crate is built on, so a host passes limits and reads adapter info in the same
/// version.
pub use wgpu;
