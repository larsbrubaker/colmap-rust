//! The limits a compute device enforces: port of colmap-sharp
//! `ColmapSharp/Compute/ComputeDeviceLimits.cs`. [`ComputeDeviceLimits::DEFAULTS`] are WebGPU's
//! default limits, which every adapter grants.

/// The limits a compute device enforces, fixed when the device is created.
/// Port of `ComputeDeviceLimits`; the field names are WebGPU's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeDeviceLimits {
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
    pub max_storage_buffers_per_shader_stage: u32,
    pub max_uniform_buffer_binding_size: u64,
    pub max_compute_workgroups_per_dimension: u32,
    pub max_compute_invocations_per_workgroup: u32,
    /// The alignment of every binding offset, storage and uniform alike: a positive power of two.
    pub min_storage_buffer_offset_alignment: u32,
    pub max_bind_groups: u32,
    pub max_uniform_buffers_per_shader_stage: u32,
    pub max_bindings_per_bind_group: u32,
    pub max_compute_workgroup_storage_size: u32,
}

impl ComputeDeviceLimits {
    /// WebGPU's default limits. Port of `ComputeDeviceLimits.Defaults`.
    pub const DEFAULTS: ComputeDeviceLimits = ComputeDeviceLimits {
        max_buffer_size: 268_435_456,
        max_storage_buffer_binding_size: 134_217_728,
        max_storage_buffers_per_shader_stage: 8,
        max_uniform_buffer_binding_size: 65_536,
        max_compute_workgroups_per_dimension: 65_535,
        max_compute_invocations_per_workgroup: 256,
        min_storage_buffer_offset_alignment: 256,
        max_bind_groups: 4,
        max_uniform_buffers_per_shader_stage: 12,
        max_bindings_per_bind_group: 1000,
        max_compute_workgroup_storage_size: 16_384,
    };
}

impl Default for ComputeDeviceLimits {
    fn default() -> Self {
        Self::DEFAULTS
    }
}
