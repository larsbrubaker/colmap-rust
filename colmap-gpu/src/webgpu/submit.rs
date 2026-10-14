//! Submitting and waiting for `WebGpuComputeDevice` (`device.rs`): flush, readback, the
//! pre-cancelled path and GPU fault reporting. Port of `FlushAsync`, `ReadBufferAsync`,
//! `SubmitCanceled`, `SubmitAndWaitAsync` and `ThrowIfGpuFaulted` in MatterCAD's
//! `Tests/ColmapGpuTests/WebGpuComputeDevice.cs`.
//!
//! **Flush is a fence.** A flush submits everything recorded and completes once the queue
//! reports the submitted work done (`Queue::on_submitted_work_done`). On the desktop the device
//! is polled to completion first, so the wait is already over when the future is first polled; in
//! the browser the poll does nothing and the callback arrives from the JS event loop.

use colmap_rust::util::cancellation::CancelToken;

use super::device::WebGpuComputeDevice;
use super::handles::WebGpuComputeBuffer;
use super::signal::Signal;
use super::validate;
use crate::compute::{ComputeError, ComputeErrorKind, ComputeResult};

fn cancelled_error() -> ComputeError {
    ComputeError::new(
        ComputeErrorKind::Cancelled,
        "The operation was cancelled before it waited for the GPU.",
    )
}

impl WebGpuComputeDevice {
    pub(crate) async fn flush_impl(&mut self, cancel: Option<&CancelToken>) -> ComputeResult<()> {
        if cancel.is_some_and(CancelToken::is_cancelled) {
            return self.submit_cancelled();
        }

        self.pending_bound.clear();
        self.submit(None);
        self.wait_for_queue().await
    }

    pub(crate) async fn read_buffer_impl(
        &mut self,
        buffer: &WebGpuComputeBuffer,
        offset: u64,
        destination: &mut [u8],
        cancel: Option<&CancelToken>,
    ) -> ComputeResult<()> {
        if buffer.shared.device_id != self.id {
            return Err(ComputeError::argument(
                "The buffer was not created by this device.",
            ));
        }
        validate::validate_range(&buffer.shared, offset, destination.len(), "read")?;
        if cancel.is_some_and(CancelToken::is_cancelled) {
            return self.submit_cancelled();
        }

        self.pending_bound.clear();
        if destination.is_empty() {
            self.submit(None);
            return self.wait_for_queue().await;
        }

        let length = destination.len() as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("colmap compute readback"),
            size: length,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.submit(Some((&buffer.shared.buffer, offset, &staging, length)));
        let (mapped, send) = Signal::new();
        staging.map_async(wgpu::MapMode::Read, .., send);
        self.poll()?;
        let map_result = mapped.await;
        // A GPU fault explains a failed map better than the map error itself.
        self.throw_if_gpu_faulted()?;
        map_result.map_err(|e| {
            ComputeError::operation(format!("Reading the buffer back from the GPU failed: {e}"))
        })?;
        destination.copy_from_slice(&staging.slice(..).get_mapped_range());
        staging.unmap();
        Ok(())
    }

    /// Everything recorded still goes to the GPU (so nothing is left pending), but the call does
    /// not wait for it. A GPU error already known wins over the cancellation.
    fn submit_cancelled(&mut self) -> ComputeResult<()> {
        self.pending_bound.clear();
        self.submit(None);
        self.throw_if_gpu_faulted()?;
        Err(cancelled_error())
    }

    /// Submits the recorded dispatches, followed by a copy into `readback` if given.
    fn submit(&mut self, readback: Option<(&wgpu::Buffer, u64, &wgpu::Buffer, u64)>) {
        let mut encoder = self.encoder.take().unwrap_or_else(|| {
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("colmap compute"),
                })
        });
        if let Some((source, offset, destination, length)) = readback {
            encoder.copy_buffer_to_buffer(source, offset, destination, 0, length);
        }
        self.queue.submit(Some(encoder.finish()));
    }

    /// Waits until the queue has run everything submitted, then reports any GPU fault.
    async fn wait_for_queue(&mut self) -> ComputeResult<()> {
        let (done, send) = Signal::new();
        self.queue.on_submitted_work_done(move || send(()));
        self.poll()?;
        done.await;
        self.throw_if_gpu_faulted()
    }

    /// Drives the native device until the submitted work and its callbacks have run; a no-op
    /// on browser WebGPU, whose callbacks come from the event loop.
    fn poll(&self) -> ComputeResult<()> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map(|_| ())
            .map_err(|e| {
                self.throw_if_gpu_faulted().err().unwrap_or_else(|| {
                    ComputeError::operation(format!("Waiting for the GPU failed: {e}"))
                })
            })
    }

    fn throw_if_gpu_faulted(&self) -> ComputeResult<()> {
        if let Some(message) = self.lost.lock().ok().and_then(|lost| lost.clone()) {
            return Err(ComputeError::operation(format!(
                "The GPU device was lost: {message}"
            )));
        }

        if let Ok(mut errors) = self.errors.lock() {
            if !errors.is_empty() {
                let messages = errors.join("\n");
                errors.clear();
                return Err(ComputeError::operation(format!(
                    "The GPU reported an error running compute work: {messages}"
                )));
            }
        }

        Ok(())
    }
}
