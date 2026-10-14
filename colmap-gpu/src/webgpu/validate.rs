//! The seam's rule checks that `WebGpuComputeDevice` (`device.rs`) runs before wgpu sees a call:
//! port of the validation half of MatterCAD's `Tests/ColmapGpuTests/WebGpuComputeDevice.cs`
//! (`ValidateKernel`, `ValidateEntry`, `RejectWritableAliasing`, `ValidateRange`,
//! `ValidateWorkgroups`). The messages are C#'s.

use std::collections::{BTreeMap, HashSet};

use super::handles::{
    BufferShared, WebGpuComputeBindGroup, WebGpuComputeBuffer, WebGpuComputeKernel,
};
use crate::compute::{
    ComputeBindingType, ComputeBufferBinding, ComputeDeviceLimits, ComputeError, ComputeErrorKind,
    ComputeKernelBinding, ComputeKernelDescriptor, ComputeResult,
};

/// Checks a kernel descriptor against the seam's rules and the device's limits, and returns each
/// group's bindings sorted by binding index.
pub(crate) fn validate_kernel(
    limits: &ComputeDeviceLimits,
    descriptor: &ComputeKernelDescriptor,
) -> ComputeResult<Vec<Vec<ComputeKernelBinding>>> {
    require_text(&descriptor.label, "label")?;
    require_text(&descriptor.source, "WGSL source")?;
    require_text(&descriptor.entry_point, "entry point")?;
    let label = &descriptor.label;

    let mut group_count = 0u32;
    let mut storage_count = 0u32;
    let mut uniform_count = 0u32;
    let mut seen = HashSet::new();
    for binding in &descriptor.bindings {
        if !seen.insert((binding.group, binding.binding)) {
            return Err(ComputeError::argument(format!(
                "Kernel '{label}' declares @group({}) @binding({}) twice.",
                binding.group, binding.binding
            )));
        }

        if binding.binding >= limits.max_bindings_per_bind_group {
            return Err(ComputeError::argument(format!(
                "Kernel '{label}' declares @binding({}); this device allows binding indices below {}.",
                binding.binding, limits.max_bindings_per_bind_group
            )));
        }

        group_count = group_count.max(binding.group.saturating_add(1));
        if binding.ty == ComputeBindingType::Uniform {
            uniform_count += 1;
        } else {
            storage_count += 1;
        }
    }

    if group_count > limits.max_bind_groups {
        return Err(ComputeError::argument(format!(
            "Kernel '{label}' uses {group_count} bind groups; this device allows at most {}.",
            limits.max_bind_groups
        )));
    }

    let mut layouts = Vec::with_capacity(group_count as usize);
    for group in 0..group_count {
        let mut layout: Vec<ComputeKernelBinding> = descriptor
            .bindings
            .iter()
            .copied()
            .filter(|b| b.group == group)
            .collect();
        // (group, binding) pairs are unique, so the order is total.
        layout.sort_by_key(|b| b.binding);
        if layout.is_empty() {
            return Err(ComputeError::argument(format!(
                "Kernel '{label}' declares no binding in group {group} but uses group {}; groups must run 0, 1, ... with no gap.",
                group_count - 1
            )));
        }
        layouts.push(layout);
    }

    if storage_count > limits.max_storage_buffers_per_shader_stage {
        return Err(ComputeError::argument(format!(
            "Kernel '{label}' declares {storage_count} storage buffers; this device allows at most {} per shader stage.",
            limits.max_storage_buffers_per_shader_stage
        )));
    }

    if uniform_count > limits.max_uniform_buffers_per_shader_stage {
        return Err(ComputeError::argument(format!(
            "Kernel '{label}' declares {uniform_count} uniform buffers; this device allows at most {} per shader stage.",
            limits.max_uniform_buffers_per_shader_stage
        )));
    }

    Ok(layouts)
}

/// Checks one bind group entry against its declared binding: buffer kind, offset alignment, size
/// and range, and the binding size limit.
pub(crate) fn validate_entry(
    limits: &ComputeDeviceLimits,
    kernel: &WebGpuComputeKernel,
    group: u32,
    entry: &ComputeBufferBinding<'_, WebGpuComputeBuffer>,
    ty: ComputeBindingType,
) -> ComputeResult<()> {
    let (binding, offset, size) = (entry.binding, entry.offset, entry.size);
    let buffer: &BufferShared = &entry.buffer.shared;
    let place = format!("{kernel} group {group} binding {binding}");
    let want_kind = ty.buffer_kind();
    if buffer.kind != want_kind {
        return Err(ComputeError::argument(format!(
            "{place} is declared {ty:?} and needs a {want_kind:?} buffer; {buffer} is {:?}.",
            buffer.kind
        )));
    }

    let alignment = u64::from(limits.min_storage_buffer_offset_alignment);
    if !offset.is_multiple_of(alignment) {
        return Err(ComputeError::argument(format!(
            "{place}: offset {offset} is not a multiple of minStorageBufferOffsetAlignment ({alignment})."
        )));
    }

    if size == 0 || !size.is_multiple_of(4) {
        return Err(ComputeError::argument(format!(
            "{place}: size {size} is not a positive multiple of 4."
        )));
    }

    if offset > buffer.size || size > buffer.size - offset {
        return Err(ComputeError::argument(format!(
            "{place}: {size} bytes at offset {offset} run past the end of {buffer}."
        )));
    }

    let limit = if ty == ComputeBindingType::Uniform {
        limits.max_uniform_buffer_binding_size
    } else {
        limits.max_storage_buffer_binding_size
    };
    if size > limit {
        return Err(ComputeError::argument(format!(
            "{place}: a {size} byte {ty:?} binding exceeds this device's limit of {limit} bytes."
        )));
    }

    Ok(())
}

/// WebGPU tracks usage per whole buffer per dispatch, so a buffer bound read_write may not be
/// bound anywhere else in the same dispatch, even over a disjoint range.
pub(crate) fn reject_writable_aliasing(
    kernel: &WebGpuComputeKernel,
    bound: &[&WebGpuComputeBindGroup],
) -> ComputeResult<()> {
    // Keyed by buffer id; a BTreeMap so nothing depends on hash order.
    let mut binding_counts: BTreeMap<u64, u32> = BTreeMap::new();
    let mut writable: Vec<&BufferShared> = Vec::new();
    for (group, bind_group) in bound.iter().enumerate() {
        let layout = &kernel.group_layouts[group];
        for (index, entry) in bind_group.entries.iter().enumerate() {
            *binding_counts.entry(entry.buffer.id).or_insert(0) += 1;
            if layout[index].ty == ComputeBindingType::Storage {
                writable.push(&entry.buffer);
            }
        }
    }

    for buffer in writable {
        if binding_counts.get(&buffer.id).copied().unwrap_or(0) > 1 {
            return Err(ComputeError::argument(format!(
                "{kernel} binds {buffer} read_write and binds it again in the same dispatch. WebGPU forbids any other binding of a buffer bound read_write, even of a disjoint range; split it into separate buffers."
            )));
        }
    }

    Ok(())
}

/// A write or read range: offset and length multiples of 4, inside the buffer.
pub(crate) fn validate_range(
    buffer: &BufferShared,
    offset: u64,
    length: usize,
    verb: &str,
) -> ComputeResult<()> {
    let length = length as u64;
    if !offset.is_multiple_of(4) || !length.is_multiple_of(4) {
        return Err(ComputeError::argument(format!(
            "A buffer {verb}'s offset ({offset}) and length ({length}) must be multiples of 4."
        )));
    }

    if offset > buffer.size || length > buffer.size - offset {
        return Err(ComputeError::argument(format!(
            "A {verb} of {length} bytes at offset {offset} runs past the end of {buffer}."
        )));
    }

    Ok(())
}

/// One dispatch dimension against `maxComputeWorkgroupsPerDimension`.
pub(crate) fn validate_workgroups(
    limits: &ComputeDeviceLimits,
    count: u32,
    name: &str,
) -> ComputeResult<()> {
    if count > limits.max_compute_workgroups_per_dimension {
        return Err(ComputeError::new(
            ComputeErrorKind::OutOfRange,
            format!(
                "{name} = {count}: a dispatch may have at most {} workgroups along one dimension.",
                limits.max_compute_workgroups_per_dimension
            ),
        ));
    }

    Ok(())
}

fn require_text(value: &str, what: &str) -> ComputeResult<()> {
    if value.is_empty() {
        return Err(ComputeError::argument(format!("A kernel needs a {what}.")));
    }

    Ok(())
}
