//! `WebGpuComputeDevice`: the compute seam (`crate::compute::ComputeDevice`) over a dedicated wgpu
//! device. Port of MatterCAD's `Tests/ColmapGpuTests/WebGpuComputeDevice.cs`, which implemented
//! the same seam over agg-sharp's WebGPU device. This file opens the device and records work
//! (buffers, kernels, bind groups, dispatches); `submit.rs` submits it and waits (flush, read,
//! GPU faults); `validate.rs` holds the rule checks; `handles.rs` the handle types.

use std::collections::HashSet;
use std::future::Future;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use colmap_rust::util::cancellation::CancelToken;

use super::handles::{
    BoundEntry, BufferShared, WebGpuComputeBindGroup, WebGpuComputeBuffer, WebGpuComputeKernel,
};
use super::validate;
use crate::compute::{
    ComputeBindingType, ComputeBufferBinding, ComputeBufferKind, ComputeDevice,
    ComputeDeviceLimits, ComputeError, ComputeErrorKind, ComputeKernelDescriptor, ComputeResult,
};

/// Device and buffer ids, unique in the process, so a handle can tell which device made it.
static NEXT_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    NEXT_ID.fetch_add(1, Ordering::Relaxed)
}

/// A [`ComputeDevice`] over a wgpu device it opens and owns.
///
/// **One compute pass per dispatch.** Each dispatch opens a compute pass on the pending command
/// encoder, records the dispatch and ends the pass. WebGPU orders dispatches across passes exactly
/// as within one, and beginning a pass is a CPU-side encoder call.
///
/// **GPU errors.** wgpu reports validation and WGSL errors through the device's uncaptured-error
/// callback rather than failing the call. This device collects them: one raised while a kernel is
/// being created fails that call with [`ComputeErrorKind::InvalidArgument`]; any other fails the
/// next flush or read with [`ComputeErrorKind::InvalidOperation`], as does a lost device. The
/// callback cannot say which call caused an error, so the attribution is only right because this
/// device always owns a dedicated wgpu device that nothing else records on.
pub struct WebGpuComputeDevice {
    pub(crate) id: u64,
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    adapter_limits: wgpu::Limits,
    limits: ComputeDeviceLimits,
    supports_blocking_wait: bool,
    pub(crate) errors: Arc<Mutex<Vec<String>>>,
    pub(crate) lost: Arc<Mutex<Option<String>>>,
    /// Dispatches recorded since the last submit.
    pub(crate) encoder: Option<wgpu::CommandEncoder>,
    /// Ids of buffers bound by dispatches recorded since the last flush: the ones a write must
    /// refuse.
    pub(crate) pending_bound: HashSet<u64>,
}

impl WebGpuComputeDevice {
    /// Opens a dedicated wgpu device, blocking, on the desktop. In the browser use
    /// [`create_async`](Self::create_async).
    ///
    /// `raise_compute_limits` asks the adapter for its maximum `maxBufferSize`,
    /// `maxStorageBufferBindingSize`, `maxStorageBuffersPerShaderStage` and
    /// `maxComputeInvocationsPerWorkgroup` instead of the WebGPU defaults, which is what PatchMatch
    /// on full-size photos needs; what was granted is in [`ComputeDevice::limits`]. Fails with
    /// [`ComputeErrorKind::NoAdapter`] when the machine has no GPU adapter.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn create(raise_compute_limits: bool, label: Option<&str>) -> ComputeResult<Self> {
        pollster::block_on(Self::create_async(raise_compute_limits, || false, label))
    }

    /// Opens a dedicated wgpu device anywhere, the browser included, where the adapter and device
    /// requests settle only once control returns to the JS event loop. `is_browser` says whether a
    /// thread must never block on a GPU wait ([`ComputeDevice::supports_blocking_wait`] is its
    /// negation); it is passed in so a desktop test can build the browser answer.
    pub async fn create_async(
        raise_compute_limits: bool,
        is_browser: impl FnOnce() -> bool,
        label: Option<&str>,
    ) -> ComputeResult<Self> {
        let supports_blocking_wait = !is_browser();
        let label = label.unwrap_or("colmap compute");
        let mut instance_descriptor = wgpu::InstanceDescriptor::new_without_display_handle();
        instance_descriptor.backends = wgpu::Backends::PRIMARY;
        let instance = wgpu::Instance::new(instance_descriptor);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| {
                ComputeError::new(
                    ComputeErrorKind::NoAdapter,
                    format!("No WebGPU adapter could be opened: {e}"),
                )
            })?;
        let adapter_limits = adapter.limits();
        let required_limits = required_limits(raise_compute_limits, &adapter_limits);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some(label),
                required_features: wgpu::Features::empty(),
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::default(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|e| {
                ComputeError::operation(format!("No WebGPU device could be opened: {e}"))
            })?;

        let errors = Arc::new(Mutex::new(Vec::new()));
        let lost = Arc::new(Mutex::new(None));
        {
            let errors = Arc::clone(&errors);
            device.on_uncaptured_error(Arc::new(move |error: wgpu::Error| {
                if let Ok(mut errors) = errors.lock() {
                    errors.push(error.to_string());
                }
            }));
            let lost = Arc::clone(&lost);
            device.set_device_lost_callback(move |reason, message| {
                // Dropping the device reports `Destroyed`; only a real loss faults later calls.
                if reason != wgpu::DeviceLostReason::Destroyed {
                    if let Ok(mut lost) = lost.lock() {
                        *lost = Some(message);
                    }
                }
            });
        }

        Ok(Self {
            id: next_id(),
            limits: map_limits(&device.limits()),
            device,
            queue,
            adapter_info: adapter.get_info(),
            adapter_limits,
            supports_blocking_wait,
            errors,
            lost,
            encoder: None,
            pending_bound: HashSet::new(),
        })
    }

    /// The adapter the device was opened on, for diagnostics.
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// The adapter's own limits, the most a raised request can be granted.
    pub fn adapter_limits(&self) -> &wgpu::Limits {
        &self.adapter_limits
    }

    /// The limits the wgpu device was granted.
    pub fn granted_limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    fn own_buffer<'b>(&self, buffer: &'b WebGpuComputeBuffer) -> ComputeResult<&'b BufferShared> {
        if buffer.shared.device_id != self.id {
            return Err(ComputeError::argument(
                "The buffer was not created by this device.",
            ));
        }
        Ok(&buffer.shared)
    }

    fn own_kernel(&self, kernel: &WebGpuComputeKernel) -> ComputeResult<()> {
        if kernel.device_id != self.id {
            return Err(ComputeError::argument(
                "The kernel was not created by this device.",
            ));
        }
        Ok(())
    }

    fn create_buffer_impl(
        &self,
        kind: ComputeBufferKind,
        size: u64,
        initial: &[u8],
        label: Option<&str>,
    ) -> ComputeResult<WebGpuComputeBuffer> {
        if size == 0 || !size.is_multiple_of(4) {
            return Err(ComputeError::argument(format!(
                "A buffer's size ({size}) must be a positive multiple of 4."
            )));
        }

        if initial.len() as u64 > size {
            return Err(ComputeError::argument(format!(
                "{} bytes of initial data do not fit a {size} byte buffer.",
                initial.len()
            )));
        }

        if size > self.limits.max_buffer_size {
            return Err(ComputeError::operation(format!(
                "A {size} byte buffer exceeds this device's maxBufferSize of {} bytes.",
                self.limits.max_buffer_size
            )));
        }

        // Every buffer can be written and read back, whatever it is bound as (the seam's rule).
        let usage = match kind {
            ComputeBufferKind::Uniform => wgpu::BufferUsages::UNIFORM,
            ComputeBufferKind::Storage => wgpu::BufferUsages::STORAGE,
        } | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label,
            size,
            usage,
            mapped_at_creation: !initial.is_empty(),
        });
        if !initial.is_empty() {
            // Mapped-at-creation memory starts zeroed, so the bytes after `initial` read as zero.
            buffer
                .slice(..)
                .get_mapped_range_mut()
                .slice(..initial.len())
                .copy_from_slice(initial);
            buffer.unmap();
        }

        Ok(WebGpuComputeBuffer {
            shared: Arc::new(BufferShared {
                id: next_id(),
                device_id: self.id,
                buffer,
                kind,
                size,
                label: label.map(str::to_owned),
                dropped: AtomicBool::new(false),
            }),
        })
    }

    fn create_kernel_impl(
        &self,
        descriptor: &ComputeKernelDescriptor,
    ) -> ComputeResult<WebGpuComputeKernel> {
        let group_layouts = validate::validate_kernel(&self.limits, descriptor)?;
        let errors_before = self.errors.lock().map(|e| e.len()).unwrap_or(0);

        let bind_group_layouts: Vec<wgpu::BindGroupLayout> = group_layouts
            .iter()
            .enumerate()
            .map(|(group, layout)| {
                let entries: Vec<wgpu::BindGroupLayoutEntry> = layout
                    .iter()
                    .map(|b| wgpu::BindGroupLayoutEntry {
                        binding: b.binding,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: match b.ty {
                                ComputeBindingType::Uniform => wgpu::BufferBindingType::Uniform,
                                ComputeBindingType::ReadOnlyStorage => {
                                    wgpu::BufferBindingType::Storage { read_only: true }
                                }
                                ComputeBindingType::Storage => {
                                    wgpu::BufferBindingType::Storage { read_only: false }
                                }
                            },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    })
                    .collect();
                self.device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some(&format!("{} group {group}", descriptor.label)),
                        entries: &entries,
                    })
            })
            .collect();
        let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> =
            bind_group_layouts.iter().map(Some).collect();
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&descriptor.label),
                bind_group_layouts: &layout_refs,
                immediate_size: 0,
            });
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&descriptor.label),
                source: wgpu::ShaderSource::Wgsl(descriptor.source.as_str().into()),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&descriptor.label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(&descriptor.entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        // wgpu reports a WGSL or pipeline error through the uncaptured-error callback during the
        // create call (natively) and hands back an invalid handle. Report it here, where the
        // caller can still tell which kernel it was, rather than as a fault of a later flush.
        if let Ok(mut errors) = self.errors.lock() {
            if errors.len() > errors_before {
                let messages = errors.split_off(errors_before).join("\n");
                return Err(ComputeError::argument(format!(
                    "Kernel '{}' does not compile: {messages}",
                    descriptor.label
                )));
            }
        }

        Ok(WebGpuComputeKernel {
            device_id: self.id,
            pipeline,
            label: descriptor.label.clone(),
            group_layouts,
            bind_group_layouts,
        })
    }

    fn create_bind_group_impl(
        &self,
        kernel: &WebGpuComputeKernel,
        group: u32,
        entries: &[ComputeBufferBinding<'_, WebGpuComputeBuffer>],
    ) -> ComputeResult<WebGpuComputeBindGroup> {
        self.own_kernel(kernel)?;
        let Some(layout) = kernel.group_layouts.get(group as usize) else {
            return Err(ComputeError::argument(format!(
                "{kernel} declares no group {group}."
            )));
        };
        if entries.len() != layout.len() {
            return Err(ComputeError::argument(format!(
                "{kernel} declares {} bindings in group {group}, but {} entries were given.",
                layout.len(),
                entries.len()
            )));
        }

        let mut sorted: Vec<ComputeBufferBinding<'_, WebGpuComputeBuffer>> = entries.to_vec();
        // Stable, and a binding given twice is refused just below, so ties never survive.
        sorted.sort_by_key(|e| e.binding);
        let mut bound = Vec::with_capacity(sorted.len());
        for (entry, declared) in sorted.iter().zip(layout) {
            if entry.binding != declared.binding {
                let declared_list: Vec<String> =
                    layout.iter().map(|b| b.binding.to_string()).collect();
                return Err(ComputeError::argument(format!(
                    "{kernel} group {group} has no binding {}, or it was given twice; the layout declares bindings {}.",
                    entry.binding,
                    declared_list.join(", ")
                )));
            }

            self.own_buffer(entry.buffer)?;
            validate::validate_entry(&self.limits, kernel, group, entry, declared.ty)?;
            bound.push(BoundEntry {
                buffer: Arc::clone(&entry.buffer.shared),
            });
        }

        let wgpu_entries: Vec<wgpu::BindGroupEntry<'_>> = sorted
            .iter()
            .map(|e| wgpu::BindGroupEntry {
                binding: e.binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &e.buffer.shared.buffer,
                    offset: e.offset,
                    size: NonZeroU64::new(e.size),
                }),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{} group {group}", kernel.label)),
            layout: &kernel.bind_group_layouts[group as usize],
            entries: &wgpu_entries,
        });
        Ok(WebGpuComputeBindGroup {
            device_id: self.id,
            group,
            bind_group,
            layout: layout.clone(),
            entries: bound,
        })
    }

    fn dispatch_impl(
        &mut self,
        kernel: &WebGpuComputeKernel,
        groups: &[&WebGpuComputeBindGroup],
        workgroups: [u32; 3],
    ) -> ComputeResult<()> {
        self.own_kernel(kernel)?;
        if groups.len() != kernel.group_layouts.len() {
            return Err(ComputeError::argument(format!(
                "{kernel} declares {} bind groups, but {} were given.",
                kernel.group_layouts.len(),
                groups.len()
            )));
        }

        for (index, bind_group) in groups.iter().enumerate() {
            if bind_group.device_id != self.id {
                return Err(ComputeError::argument(
                    "The bind group was not created by this device.",
                ));
            }
            if bind_group.group as usize != index
                || bind_group.layout != kernel.group_layouts[index]
            {
                return Err(ComputeError::argument(format!(
                    "The bind group at index {index} ({bind_group}) does not match {kernel}'s layout for group {index}."
                )));
            }
            for entry in &bind_group.entries {
                if entry.buffer.is_dropped() {
                    return Err(ComputeError::new(
                        ComputeErrorKind::Disposed,
                        format!("{} has been dropped; {bind_group} binds it.", entry.buffer),
                    ));
                }
            }
        }

        validate::reject_writable_aliasing(kernel, groups)?;
        for (count, name) in workgroups.iter().zip(["x", "y", "z"]) {
            validate::validate_workgroups(&self.limits, *count, name)?;
        }

        let device = &self.device;
        let encoder = self.encoder.get_or_insert_with(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("colmap compute"),
            })
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&kernel.label),
                timestamp_writes: None,
            });
            pass.set_pipeline(&kernel.pipeline);
            for (index, bind_group) in groups.iter().enumerate() {
                pass.set_bind_group(index as u32, &bind_group.bind_group, &[]);
            }
            pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        }

        for bind_group in groups {
            for entry in &bind_group.entries {
                self.pending_bound.insert(entry.buffer.id);
            }
        }
        Ok(())
    }

    fn write_buffer_impl(
        &mut self,
        buffer: &WebGpuComputeBuffer,
        offset: u64,
        data: &[u8],
    ) -> ComputeResult<()> {
        let target = self.own_buffer(buffer)?;
        validate::validate_range(target, offset, data.len(), "write")?;
        if self.pending_bound.contains(&target.id) {
            return Err(ComputeError::operation(format!(
                "WriteBuffer to {target}, which a dispatch recorded since the last flush binds. A write is a queue write that lands before the flush runs any dispatch, so that dispatch would read the new bytes; give per-dispatch data its own buffer, or flush before writing."
            )));
        }

        if !data.is_empty() {
            self.queue.write_buffer(&target.buffer, offset, data);
        }
        Ok(())
    }
}

/// The limits a device asks for: WebGPU's defaults, with the four compute limits raised to the
/// adapter's maxima when asked (as agg-sharp's `WebGpuRenderDevice.RequiredLimits` does).
fn required_limits(raise: bool, adapter: &wgpu::Limits) -> wgpu::Limits {
    let mut required = wgpu::Limits::default();
    if raise {
        required.max_buffer_size = adapter.max_buffer_size;
        required.max_storage_buffer_binding_size = adapter.max_storage_buffer_binding_size;
        required.max_storage_buffers_per_shader_stage =
            adapter.max_storage_buffers_per_shader_stage;
        required.max_compute_invocations_per_workgroup =
            adapter.max_compute_invocations_per_workgroup;
    }
    required
}

/// The seam's limits from a wgpu device's. Port of `WebGpuComputeDevice.MapLimits`. The binding
/// offset alignment is the larger of the storage and uniform ones, because the seam applies one
/// alignment to both kinds of binding.
pub fn map_limits(limits: &wgpu::Limits) -> ComputeDeviceLimits {
    ComputeDeviceLimits {
        max_buffer_size: limits.max_buffer_size,
        max_storage_buffer_binding_size: limits.max_storage_buffer_binding_size,
        max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
        max_uniform_buffer_binding_size: limits.max_uniform_buffer_binding_size,
        max_compute_workgroups_per_dimension: limits.max_compute_workgroups_per_dimension,
        max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
        min_storage_buffer_offset_alignment: limits
            .min_storage_buffer_offset_alignment
            .max(limits.min_uniform_buffer_offset_alignment),
        max_bind_groups: limits.max_bind_groups,
        max_uniform_buffers_per_shader_stage: limits.max_uniform_buffers_per_shader_stage,
        max_bindings_per_bind_group: limits.max_bindings_per_bind_group,
        max_compute_workgroup_storage_size: limits.max_compute_workgroup_storage_size,
    }
}

impl ComputeDevice for WebGpuComputeDevice {
    type Buffer = WebGpuComputeBuffer;
    type Kernel = WebGpuComputeKernel;
    type BindGroup = WebGpuComputeBindGroup;

    fn limits(&self) -> &ComputeDeviceLimits {
        &self.limits
    }

    fn supports_blocking_wait(&self) -> bool {
        self.supports_blocking_wait
    }

    fn create_buffer(
        &self,
        kind: ComputeBufferKind,
        size: u64,
        initial: &[u8],
        label: Option<&str>,
    ) -> ComputeResult<WebGpuComputeBuffer> {
        self.create_buffer_impl(kind, size, initial, label)
    }

    fn write_buffer(
        &mut self,
        buffer: &WebGpuComputeBuffer,
        offset: u64,
        data: &[u8],
    ) -> ComputeResult<()> {
        self.write_buffer_impl(buffer, offset, data)
    }

    fn create_kernel(
        &self,
        descriptor: &ComputeKernelDescriptor,
    ) -> ComputeResult<WebGpuComputeKernel> {
        self.create_kernel_impl(descriptor)
    }

    fn create_bind_group(
        &self,
        kernel: &WebGpuComputeKernel,
        group: u32,
        entries: &[ComputeBufferBinding<'_, WebGpuComputeBuffer>],
    ) -> ComputeResult<WebGpuComputeBindGroup> {
        self.create_bind_group_impl(kernel, group, entries)
    }

    fn dispatch(
        &mut self,
        kernel: &WebGpuComputeKernel,
        groups: &[&WebGpuComputeBindGroup],
        x: u32,
        y: u32,
        z: u32,
    ) -> ComputeResult<()> {
        self.dispatch_impl(kernel, groups, [x, y, z])
    }

    fn flush<'a>(
        &'a mut self,
        cancel: Option<&'a CancelToken>,
    ) -> impl Future<Output = ComputeResult<()>> + 'a {
        self.flush_impl(cancel)
    }

    fn read_buffer<'a>(
        &'a mut self,
        buffer: &'a WebGpuComputeBuffer,
        offset: u64,
        destination: &'a mut [u8],
        cancel: Option<&'a CancelToken>,
    ) -> impl Future<Output = ComputeResult<()>> + 'a {
        self.read_buffer_impl(buffer, offset, destination, cancel)
    }
}
