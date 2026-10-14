//! The compute seam's test binary: the files under `compute/`, one per C# test class (plus the
//! shared kernels and GPU helper). Run with `cargo test -p colmap-gpu --test compute`.

#[path = "compute/compute_kernels.rs"]
mod compute_kernels;
#[path = "compute/gpu.rs"]
mod gpu;
#[path = "compute/web_gpu_compute_device_tests.rs"]
mod web_gpu_compute_device_tests;
