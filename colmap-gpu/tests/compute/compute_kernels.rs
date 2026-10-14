//! The small WGSL kernels `web_gpu_compute_device_tests.rs` runs through the seam, with their bind
//! group layouts. Port of MatterCAD's `Tests/ColmapGpuTests/ComputeKernels.cs`. Every value they
//! produce is exactly representable, so the tests compare exactly.

use colmap_gpu::compute::{ComputeBindingType, ComputeKernelBinding, ComputeKernelDescriptor};

/// The workgroup size every kernel here declares.
pub const WORKGROUP_SIZE: u32 = 64;

/// c[i] = a[i] + b[i] over f32 arrays: group 0 bindings 0 and 1 read-only, 2 read_write.
pub fn add() -> ComputeKernelDescriptor {
    ComputeKernelDescriptor {
        label: "add".into(),
        source: r#"
@group(0) @binding(0) var<storage, read> a : array<f32>;
@group(0) @binding(1) var<storage, read> b : array<f32>;
@group(0) @binding(2) var<storage, read_write> c : array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
	let i = id.x;
	if (i < arrayLength(&c))
	{
		c[i] = a[i] + b[i];
	}
}
"#
        .into(),
        entry_point: "main".into(),
        bindings: vec![
            ComputeKernelBinding::new(0, 0, ComputeBindingType::ReadOnlyStorage),
            ComputeKernelBinding::new(0, 1, ComputeBindingType::ReadOnlyStorage),
            ComputeKernelBinding::new(0, 2, ComputeBindingType::Storage),
        ],
    }
}

/// dst[i] = src[n - 1 - i] + 1 over u32 arrays. Nearly every element reads one a different
/// workgroup of the previous dispatch wrote, so dispatches that ran out of order would give wrong
/// values.
pub fn reverse() -> ComputeKernelDescriptor {
    ComputeKernelDescriptor {
        label: "reverse".into(),
        source: r#"
@group(0) @binding(0) var<storage, read> src : array<u32>;
@group(0) @binding(1) var<storage, read_write> dst : array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
	let n = arrayLength(&dst);
	let i = id.x;
	if (i < n)
	{
		dst[i] = src[n - 1u - i] + 1u;
	}
}
"#
        .into(),
        entry_point: "main".into(),
        bindings: vec![
            ComputeKernelBinding::new(0, 0, ComputeBindingType::ReadOnlyStorage),
            ComputeKernelBinding::new(0, 1, ComputeBindingType::Storage),
        ],
    }
}

/// output[i] = input[i] * scale + offset for i < count, the parameters in a uniform in group 1 so
/// a dispatch can swap them without touching the data group.
pub fn scale() -> ComputeKernelDescriptor {
    ComputeKernelDescriptor {
        label: "scale".into(),
        source: r#"
struct Params
{
	scale : f32,
	offset : f32,
	count : u32,
	pad : u32,
};

@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;
@group(1) @binding(0) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id : vec3<u32>)
{
	let i = id.x;
	if (i < params.count)
	{
		output[i] = input[i] * params.scale + params.offset;
	}
}
"#
        .into(),
        entry_point: "main".into(),
        bindings: vec![
            ComputeKernelBinding::new(0, 0, ComputeBindingType::ReadOnlyStorage),
            ComputeKernelBinding::new(0, 1, ComputeBindingType::Storage),
            ComputeKernelBinding::new(1, 0, ComputeBindingType::Uniform),
        ],
    }
}

/// The workgroups a one-dimensional dispatch over `count` items needs.
pub fn workgroups_for(count: usize) -> u32 {
    (count as u32).div_ceil(WORKGROUP_SIZE)
}
