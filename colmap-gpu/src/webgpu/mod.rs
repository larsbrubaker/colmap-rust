//! The compute seam over wgpu: port of MatterCAD's `Tests/ColmapGpuTests/WebGpuComputeDevice.cs`
//! (`device.rs`, `submit.rs`), `WebGpuComputeHandles.cs` (`handles.rs`) and the device's rule
//! checks (`validate.rs`). `signal.rs` is the one-shot future the GPU callbacks complete.
//! Every rule the seam states is checked before wgpu sees the call, because WebGPU reports most
//! broken rules only out of band and a readback then returns stale or zero bytes.

mod device;
mod handles;
mod signal;
mod submit;
mod validate;

pub use device::{map_limits, WebGpuComputeDevice};
pub use handles::{WebGpuComputeBindGroup, WebGpuComputeBuffer, WebGpuComputeKernel};
