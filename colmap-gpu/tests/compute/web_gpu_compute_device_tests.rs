//! Port of MatterCAD's `Tests/ColmapGpuTests/WebGpuComputeDeviceTests.cs`: each rule the compute
//! seam states, exercised through `WebGpuComputeDevice` on a real GPU. The kernels are in
//! `compute_kernels.rs`; opening the device (and skipping without an adapter) is `gpu.rs`.
//! `handles_disposed_while_their_work_is_pending_still_run_it` is converted: C# also checked that a
//! disposed kernel, bind group or buffer handle is refused in a later call, which Rust's ownership
//! rules out at compile time; what remains reachable, a bind group whose buffer was dropped, is
//! checked instead.

use colmap_gpu::compute::{
    ComputeBufferBinding as Binding, ComputeBufferKind, ComputeDevice, ComputeDeviceLimits,
    ComputeErrorKind,
};
use colmap_gpu::webgpu::{map_limits, WebGpuComputeDevice};
use colmap_gpu::wgpu;
use colmap_rust::util::cancellation::CancelToken;

use crate::compute_kernels::{self as kernels, workgroups_for};
use crate::gpu::{self, device_or_skip, f32_bytes, read_f32, read_u32, u32_bytes};

const STORAGE: ComputeBufferKind = ComputeBufferKind::Storage;
const UNIFORM: ComputeBufferKind = ComputeBufferKind::Uniform;

#[test]
fn a_kernel_adds_two_arrays_through_the_seam() {
    // Not a multiple of the workgroup size, so the kernel's bounds check is exercised too.
    const COUNT: usize = 1000;
    let a: Vec<f32> = (0..COUNT).map(|i| i as f32).collect();
    let b: Vec<f32> = (0..COUNT).map(|i| (2 * i) as f32 + 0.5).collect();

    let _gate = gpu::gate();
    let Some(mut device) = device_or_skip(false, "a_kernel_adds_two_arrays_through_the_seam")
    else {
        return;
    };
    let size = (COUNT * 4) as u64;
    let kernel = device.create_kernel(&kernels::add()).unwrap();
    let buffer_a = device
        .create_buffer(STORAGE, size, &f32_bytes(&a), Some("a"))
        .unwrap();
    let buffer_b = device
        .create_buffer(STORAGE, size, &f32_bytes(&b), Some("b"))
        .unwrap();
    let buffer_c = device.create_buffer(STORAGE, size, &[], Some("c")).unwrap();
    let group = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(2, &buffer_c, 0, size),
                Binding::new(0, &buffer_a, 0, size),
                Binding::new(1, &buffer_b, 0, size),
            ],
        )
        .unwrap();

    device
        .dispatch(&kernel, &[&group], workgroups_for(COUNT), 1, 1)
        .unwrap();
    let c = read_f32(&mut device, &buffer_c, 0, COUNT);

    for i in 0..COUNT {
        assert_eq!(c[i], a[i] + b[i], "element {i}");
    }
}

#[test]
fn dispatches_recorded_together_run_in_recording_order() {
    // A chain of eight reversals in one flush, 1024 workgroups each: every dispatch reads what the
    // one before it wrote, almost always from another workgroup, so the result is only right if
    // every dispatch saw all of every earlier dispatch's writes. Big enough that dispatches
    // overlapping on the GPU would show.
    const COUNT: usize = 65536;
    const STEPS: usize = 8;
    let input: Vec<u32> = (0..COUNT).map(|i| (i * 7) as u32).collect();

    let _gate = gpu::gate();
    let Some(mut device) =
        device_or_skip(false, "dispatches_recorded_together_run_in_recording_order")
    else {
        return;
    };
    let size = (COUNT * 4) as u64;
    let kernel = device.create_kernel(&kernels::reverse()).unwrap();
    let mut buffers = vec![device
        .create_buffer(STORAGE, size, &u32_bytes(&input), None)
        .unwrap()];
    for _ in 0..STEPS {
        buffers.push(device.create_buffer(STORAGE, size, &[], None).unwrap());
    }

    for step in 0..STEPS {
        let group = device
            .create_bind_group(
                &kernel,
                0,
                &[
                    Binding::new(0, &buffers[step], 0, size),
                    Binding::new(1, &buffers[step + 1], 0, size),
                ],
            )
            .unwrap();
        device
            .dispatch(&kernel, &[&group], workgroups_for(COUNT), 1, 1)
            .unwrap();
    }

    let result = read_u32(&mut device, &buffers[STEPS], 0, COUNT);

    // An even number of reversals is none, plus one per step.
    for i in 0..COUNT {
        assert_eq!(result[i], input[i] + STEPS as u32, "element {i}");
    }
}

fn scale_params(scale: f32, offset: f32, count: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(16);
    bytes.extend_from_slice(&scale.to_le_bytes());
    bytes.extend_from_slice(&offset.to_le_bytes());
    bytes.extend_from_slice(&count.to_le_bytes());
    bytes.extend_from_slice(&[0; 4]);
    bytes
}

#[test]
fn each_dispatch_sees_the_uniform_it_was_recorded_with() {
    // Two dispatches in one flush with different parameters, each in its own uniform buffer
    // created with its data - the seam's recipe for per-dispatch parameters.
    const COUNT: usize = 100;
    let input: Vec<f32> = (0..COUNT).map(|i| i as f32).collect();

    let _gate = gpu::gate();
    let Some(mut device) =
        device_or_skip(false, "each_dispatch_sees_the_uniform_it_was_recorded_with")
    else {
        return;
    };
    let size = (COUNT * 4) as u64;
    let kernel = device.create_kernel(&kernels::scale()).unwrap();
    let source = device
        .create_buffer(STORAGE, size, &f32_bytes(&input), None)
        .unwrap();
    let first = device.create_buffer(STORAGE, size, &[], None).unwrap();
    let second = device.create_buffer(STORAGE, size, &[], None).unwrap();
    let double_it = device
        .create_buffer(UNIFORM, 16, &scale_params(2.0, 1.0, COUNT as u32), None)
        .unwrap();
    let halve_it = device
        .create_buffer(
            UNIFORM,
            16,
            &scale_params(0.5, -3.0, (COUNT / 2) as u32),
            None,
        )
        .unwrap();

    let to_first = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(0, &source, 0, size),
                Binding::new(1, &first, 0, size),
            ],
        )
        .unwrap();
    let to_second = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(0, &first, 0, size),
                Binding::new(1, &second, 0, size),
            ],
        )
        .unwrap();
    let double_group = device
        .create_bind_group(&kernel, 1, &[Binding::new(0, &double_it, 0, 16)])
        .unwrap();
    let halve_group = device
        .create_bind_group(&kernel, 1, &[Binding::new(0, &halve_it, 0, 16)])
        .unwrap();

    device
        .dispatch(
            &kernel,
            &[&to_first, &double_group],
            workgroups_for(COUNT),
            1,
            1,
        )
        .unwrap();
    device
        .dispatch(
            &kernel,
            &[&to_second, &halve_group],
            workgroups_for(COUNT),
            1,
            1,
        )
        .unwrap();
    pollster::block_on(device.flush(None)).unwrap();

    let result = read_f32(&mut device, &second, 0, COUNT);
    for i in 0..COUNT {
        // The second dispatch covers only the first half; the rest stays zero.
        let expected = if i < COUNT / 2 {
            ((input[i] * 2.0 + 1.0) * 0.5) - 3.0
        } else {
            0.0
        };
        assert_eq!(result[i], expected, "element {i}");
    }
}

#[test]
fn a_read_at_an_offset_returns_that_range_and_a_write_after_a_flush_lands() {
    let initial: Vec<u32> = (0..16).map(|i| 1000 + i).collect();

    let _gate = gpu::gate();
    let Some(mut device) = device_or_skip(
        false,
        "a_read_at_an_offset_returns_that_range_and_a_write_after_a_flush_lands",
    ) else {
        return;
    };
    // Initial data shorter than the buffer: the rest reads as zero.
    let buffer = device
        .create_buffer(STORAGE, 128, &u32_bytes(&initial), None)
        .unwrap();

    assert_eq!(
        read_u32(&mut device, &buffer, 12 * 4, 5),
        vec![1012, 1013, 1014, 1015, 0]
    );

    device
        .write_buffer(&buffer, 4, &u32_bytes(&[7, 8]))
        .unwrap();
    assert_eq!(read_u32(&mut device, &buffer, 0, 4), vec![1000, 7, 8, 1003]);

    // A uniform buffer can be read back too.
    let uniform = device
        .create_buffer(UNIFORM, 16, &u32_bytes(&[1, 2, 3, 4]), None)
        .unwrap();
    assert_eq!(read_u32(&mut device, &uniform, 8, 2), vec![3, 4]);
}

#[test]
fn writing_a_buffer_a_pending_dispatch_binds_is_refused_until_the_flush() {
    let _gate = gpu::gate();
    let Some(mut device) = device_or_skip(
        false,
        "writing_a_buffer_a_pending_dispatch_binds_is_refused_until_the_flush",
    ) else {
        return;
    };
    let kernel = device.create_kernel(&kernels::reverse()).unwrap();
    let source = device.create_buffer(STORAGE, 256, &[], None).unwrap();
    let target = device.create_buffer(STORAGE, 256, &[], None).unwrap();
    let group = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(0, &source, 0, 256),
                Binding::new(1, &target, 0, 256),
            ],
        )
        .unwrap();
    device.dispatch(&kernel, &[&group], 1, 1, 1).unwrap();

    let data = [0u8; 4];
    let refused = device.write_buffer(&source, 0, &data).unwrap_err();
    assert_eq!(refused.kind, ComputeErrorKind::InvalidOperation);

    pollster::block_on(device.flush(None)).unwrap();
    device.write_buffer(&source, 0, &data).unwrap();
}

#[test]
fn a_pre_canceled_call_still_submits_its_pending_work_and_leaves_the_device_usable() {
    const COUNT: usize = 64;
    let input: Vec<u32> = (0..COUNT as u32).collect();
    let canceled = CancelToken::new();
    canceled.cancel();

    let _gate = gpu::gate();
    let Some(mut device) = device_or_skip(
        false,
        "a_pre_canceled_call_still_submits_its_pending_work_and_leaves_the_device_usable",
    ) else {
        return;
    };
    let size = (COUNT * 4) as u64;
    let kernel = device.create_kernel(&kernels::reverse()).unwrap();
    let source = device
        .create_buffer(STORAGE, size, &u32_bytes(&input), None)
        .unwrap();
    let target = device.create_buffer(STORAGE, size, &[], None).unwrap();
    let group = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(0, &source, 0, size),
                Binding::new(1, &target, 0, size),
            ],
        )
        .unwrap();

    device.dispatch(&kernel, &[&group], 1, 1, 1).unwrap();
    let flushed = pollster::block_on(device.flush(Some(&canceled))).unwrap_err();
    assert_eq!(flushed.kind, ComputeErrorKind::Cancelled);

    // Nothing is pending any more, so the source may be written again at once.
    device.write_buffer(&source, 0, &[0u8; 4]).unwrap();

    // A canceled read also submits what is pending, but copies nothing into its destination.
    let second = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(0, &target, 0, size),
                Binding::new(1, &source, 0, size),
            ],
        )
        .unwrap();
    device.dispatch(&kernel, &[&second], 1, 1, 1).unwrap();
    let mut untouched = vec![0xABu8; COUNT * 4];
    let read = pollster::block_on(device.read_buffer(&target, 0, &mut untouched, Some(&canceled)))
        .unwrap_err();
    assert_eq!(read.kind, ComputeErrorKind::Cancelled);
    assert!(untouched.iter().all(|&b| b == 0xAB));

    // Both canceled calls' dispatches ran: target = reverse(input) + 1, source = input + 2.
    let reversed = read_u32(&mut device, &target, 0, COUNT);
    let round_trip = read_u32(&mut device, &source, 0, COUNT);
    for i in 0..COUNT {
        assert_eq!(reversed[i], input[COUNT - 1 - i] + 1);
        assert_eq!(round_trip[i], input[i] + 2);
    }
}

#[test]
fn handles_disposed_while_their_work_is_pending_still_run_it() {
    const COUNT: usize = 256;
    let a: Vec<f32> = (0..COUNT).map(|i| i as f32).collect();

    let _gate = gpu::gate();
    let Some(mut device) = device_or_skip(
        false,
        "handles_disposed_while_their_work_is_pending_still_run_it",
    ) else {
        return;
    };
    let size = (COUNT * 4) as u64;
    let kernel = device.create_kernel(&kernels::add()).unwrap();
    let buffer_a = device
        .create_buffer(STORAGE, size, &f32_bytes(&a), None)
        .unwrap();
    let buffer_b = device
        .create_buffer(STORAGE, size, &f32_bytes(&a), None)
        .unwrap();
    let buffer_c = device.create_buffer(STORAGE, size, &[], None).unwrap();
    let group = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(0, &buffer_a, 0, size),
                Binding::new(1, &buffer_b, 0, size),
                Binding::new(2, &buffer_c, 0, size),
            ],
        )
        .unwrap();
    device
        .dispatch(&kernel, &[&group], workgroups_for(COUNT), 1, 1)
        .unwrap();

    drop(group);
    drop(kernel);
    drop(buffer_a);
    drop(buffer_b);

    let c = read_f32(&mut device, &buffer_c, 0, COUNT);
    for (i, value) in c.iter().enumerate() {
        assert_eq!(*value, 2.0 * i as f32);
    }

    // A dropped kernel, bind group or buffer cannot be named in a later call; a bind group that
    // outlives a buffer it binds is refused at dispatch.
    let kernel = device.create_kernel(&kernels::add()).unwrap();
    let buffer_a = device.create_buffer(STORAGE, size, &[], None).unwrap();
    let buffer_b = device.create_buffer(STORAGE, size, &[], None).unwrap();
    let group = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(0, &buffer_a, 0, size),
                Binding::new(1, &buffer_b, 0, size),
                Binding::new(2, &buffer_c, 0, size),
            ],
        )
        .unwrap();
    drop(buffer_a);
    let refused = device.dispatch(&kernel, &[&group], 1, 1, 1).unwrap_err();
    assert_eq!(refused.kind, ComputeErrorKind::Disposed);
}

#[test]
fn without_the_opt_in_the_limits_are_the_web_gpu_defaults() {
    let _gate = gpu::gate();
    let Some(device) = device_or_skip(
        false,
        "without_the_opt_in_the_limits_are_the_web_gpu_defaults",
    ) else {
        return;
    };
    assert_eq!(*device.limits(), ComputeDeviceLimits::DEFAULTS);
    assert!(device.supports_blocking_wait());
}

#[test]
fn the_opt_in_reports_what_the_adapter_granted() {
    let _gate = gpu::gate();
    let Some(device) = device_or_skip(true, "the_opt_in_reports_what_the_adapter_granted") else {
        return;
    };
    let granted = device.granted_limits();
    let adapter = device.adapter_limits();
    let info = device.adapter_info();
    println!(
        "Adapter '{}' ({:?}): {:?}",
        info.name,
        info.backend,
        device.limits()
    );

    let limits = *device.limits();
    assert_eq!(limits, map_limits(&granted));
    assert_eq!(limits.max_buffer_size, adapter.max_buffer_size);
    assert_eq!(
        limits.max_storage_buffer_binding_size,
        adapter.max_storage_buffer_binding_size
    );
    assert_eq!(
        limits.max_storage_buffers_per_shader_stage,
        adapter.max_storage_buffers_per_shader_stage
    );
    assert_eq!(
        limits.max_compute_invocations_per_workgroup,
        adapter.max_compute_invocations_per_workgroup
    );

    let defaults = ComputeDeviceLimits::DEFAULTS;
    assert!(limits.max_buffer_size >= defaults.max_buffer_size);
    assert!(limits.max_storage_buffer_binding_size >= defaults.max_storage_buffer_binding_size);
    assert_eq!(limits.max_bind_groups, defaults.max_bind_groups);
    assert_eq!(
        limits.max_uniform_buffers_per_shader_stage,
        defaults.max_uniform_buffers_per_shader_stage
    );
    assert_eq!(
        limits.max_bindings_per_bind_group,
        defaults.max_bindings_per_bind_group
    );
    assert_eq!(
        limits.max_compute_workgroup_storage_size,
        defaults.max_compute_workgroup_storage_size
    );
}

#[test]
fn the_offset_alignment_is_the_larger_of_storage_and_uniform() {
    let limits = map_limits(&wgpu_limits(32, 128));
    assert_eq!(limits.min_storage_buffer_offset_alignment, 128);
}

fn wgpu_limits(storage_alignment: u32, uniform_alignment: u32) -> wgpu::Limits {
    wgpu::Limits {
        min_storage_buffer_offset_alignment: storage_alignment,
        min_uniform_buffer_offset_alignment: uniform_alignment,
        ..wgpu::Limits::default()
    }
}

#[test]
fn the_browser_predicate_decides_whether_a_caller_may_block() {
    // create_async is the browser's way in; on the desktop it completes at once, and the
    // predicate alone decides what the device tells its callers.
    let _gate = gpu::gate();
    let open = |is_browser: bool| match pollster::block_on(WebGpuComputeDevice::create_async(
        false,
        || is_browser,
        None,
    )) {
        Ok(device) => Some(device),
        Err(e) if e.kind == ComputeErrorKind::NoAdapter => {
            println!("the_browser_predicate_decides_whether_a_caller_may_block: skipped ({e}).");
            None
        }
        Err(e) => panic!("opening the GPU device failed: {e}"),
    };
    let Some(mut browser) = open(true) else {
        return;
    };
    assert!(!browser.supports_blocking_wait());
    pollster::block_on(browser.flush(None)).unwrap();
    drop(browser);

    let desktop = open(false).expect("an adapter that opened once opens again");
    assert!(desktop.supports_blocking_wait());
}

#[test]
fn wgsl_that_does_not_compile_fails_the_create_and_the_device_carries_on() {
    let _gate = gpu::gate();
    let Some(mut device) = device_or_skip(
        false,
        "wgsl_that_does_not_compile_fails_the_create_and_the_device_carries_on",
    ) else {
        return;
    };
    let mut broken = kernels::add();
    broken.label = "broken".into();
    broken.source = broken
        .source
        .replace("a[i] + b[i]", "a[i] + undefined_name");
    // The message is the adapter's own, so the refusal came from wgpu's compile error and not
    // from a descriptor check.
    let refused = device
        .create_kernel(&broken)
        .err()
        .expect("broken WGSL is refused");
    assert_eq!(refused.kind, ComputeErrorKind::InvalidArgument);
    assert!(
        refused.message.contains("does not compile"),
        "{}",
        refused.message
    );

    // Nothing of the failure is left to fault a later flush.
    pollster::block_on(device.flush(None)).unwrap();
    device.create_kernel(&kernels::add()).unwrap();
}

#[test]
fn broken_rules_throw_from_the_call() {
    let _gate = gpu::gate();
    let Some(mut device) = device_or_skip(false, "broken_rules_throw_from_the_call") else {
        return;
    };
    let other = device_or_skip(false, "broken_rules_throw_from_the_call")
        .expect("an adapter that opened once opens again");
    let kernel = device.create_kernel(&kernels::reverse()).unwrap();
    let buffer = device.create_buffer(STORAGE, 256, &[], None).unwrap();
    let foreign = other.create_buffer(STORAGE, 256, &[], None).unwrap();
    let kind = |result: Result<(), colmap_gpu::compute::ComputeError>| result.unwrap_err().kind;

    let max = device.limits().max_buffer_size;
    assert_eq!(
        kind(device.create_buffer(STORAGE, 6, &[], None).map(drop)),
        ComputeErrorKind::InvalidArgument
    );
    assert_eq!(
        kind(device.create_buffer(STORAGE, max + 4, &[], None).map(drop)),
        ComputeErrorKind::InvalidOperation
    );
    assert_eq!(
        kind(device.write_buffer(&foreign, 0, &[0; 4])),
        ComputeErrorKind::InvalidArgument
    );
    assert_eq!(
        kind(device.write_buffer(&buffer, 254, &[0; 4])),
        ComputeErrorKind::InvalidArgument
    );

    // read_write bound again in the same dispatch, even as read-only.
    let aliased = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(0, &buffer, 0, 256),
                Binding::new(1, &buffer, 0, 256),
            ],
        )
        .unwrap();
    assert_eq!(
        kind(device.dispatch(&kernel, &[&aliased], 1, 1, 1)),
        ComputeErrorKind::InvalidArgument
    );

    let target = device.create_buffer(STORAGE, 256, &[], None).unwrap();
    let group = device
        .create_bind_group(
            &kernel,
            0,
            &[
                Binding::new(0, &buffer, 0, 256),
                Binding::new(1, &target, 0, 256),
            ],
        )
        .unwrap();
    let too_many = device.limits().max_compute_workgroups_per_dimension + 1;
    assert_eq!(
        kind(device.dispatch(&kernel, &[&group], too_many, 1, 1)),
        ComputeErrorKind::OutOfRange
    );

    // A misaligned binding offset.
    assert_eq!(
        kind(
            device
                .create_bind_group(
                    &kernel,
                    0,
                    &[
                        Binding::new(0, &buffer, 4, 128),
                        Binding::new(1, &target, 0, 256)
                    ],
                )
                .map(drop)
        ),
        ComputeErrorKind::InvalidArgument
    );

    // None of that reached the GPU.
    pollster::block_on(device.flush(None)).unwrap();
}
