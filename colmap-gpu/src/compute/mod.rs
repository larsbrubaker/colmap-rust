//! The compute seam: port of colmap-sharp's `ColmapSharp/Compute/` (`IComputeDevice.cs`,
//! `ComputeDescriptors.cs`, `ComputeDeviceLimits.cs`). The descriptors and limits are in
//! `descriptors.rs` and `limits.rs`, the error type in `error.rs`, and the device contract in
//! `device.rs`. The wgpu implementation is `crate::webgpu`.

mod descriptors;
mod device;
mod error;
mod limits;

pub use descriptors::{
    ComputeBindingType, ComputeBufferBinding, ComputeBufferKind, ComputeKernelBinding,
    ComputeKernelDescriptor,
};
pub use device::ComputeDevice;
pub use error::{ComputeError, ComputeErrorKind, ComputeResult};
pub use limits::ComputeDeviceLimits;
