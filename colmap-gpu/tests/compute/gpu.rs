//! Opening a real GPU device for the compute tests, and the byte helpers they share. A machine
//! with no GPU adapter skips each GPU test with a printed reason, as the C# tests' GpuTestGate
//! does; any other failure to open a device fails the test. Tests in one binary are serialized
//! by `GPU_LOCK`, so two tests never submit at once (C#'s `[NotInParallel]`).

use std::sync::{Mutex, MutexGuard};

use colmap_gpu::compute::{ComputeDevice, ComputeErrorKind};
use colmap_gpu::webgpu::{WebGpuComputeBuffer, WebGpuComputeDevice};

static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Serializes GPU tests; a test that panicked does not block the rest.
pub fn gate() -> MutexGuard<'static, ()> {
    GPU_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// A dedicated device, or `None` (after printing why) on a machine with no GPU adapter.
pub fn device_or_skip(raise_compute_limits: bool, test: &str) -> Option<WebGpuComputeDevice> {
    match WebGpuComputeDevice::create(raise_compute_limits, None) {
        Ok(device) => Some(device),
        Err(e) if e.kind == ComputeErrorKind::NoAdapter => {
            println!("{test}: skipped, no GPU adapter on this machine ({e}).");
            None
        }
        Err(e) => panic!("{test}: opening the GPU device failed: {e}"),
    }
}

pub fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

pub fn to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn to_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

pub fn read_bytes(
    device: &mut WebGpuComputeDevice,
    buffer: &WebGpuComputeBuffer,
    offset: u64,
    length: usize,
) -> Vec<u8> {
    let mut bytes = vec![0u8; length];
    pollster::block_on(device.read_buffer(buffer, offset, &mut bytes, None))
        .expect("the read succeeds");
    bytes
}

pub fn read_f32(
    device: &mut WebGpuComputeDevice,
    buffer: &WebGpuComputeBuffer,
    offset: u64,
    count: usize,
) -> Vec<f32> {
    to_f32(&read_bytes(device, buffer, offset, count * 4))
}

pub fn read_u32(
    device: &mut WebGpuComputeDevice,
    buffer: &WebGpuComputeBuffer,
    offset: u64,
    count: usize,
) -> Vec<u32> {
    to_u32(&read_bytes(device, buffer, offset, count * 4))
}
