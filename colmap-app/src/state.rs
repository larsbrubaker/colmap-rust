// Application state shared by the widget tree, the shells and the headless test harness.
// Every field is a shared cell: widgets hold clones and read them on each layout / paint, and
// tests mutate the same cells directly (colmap-app-test), so there is one source of truth.
//
// `Rc` rather than `Arc`: the UI lives on one thread (the browser main thread on wasm). Long
// pipeline work will run off-thread and report back through a channel the UI drains, which is
// a later phase's decision.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::camera::OrbitCamera;
use crate::pipeline::PipelineStatus;
use crate::ready::FirstPaintGate;
use crate::scene::Scene;

/// The GPU the viewport renders with, as wgpu reports it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BackendInfo {
    /// Adapter name, e.g. "Apple M2 Pro".
    pub adapter_name: String,
    /// Graphics API, e.g. "Metal", "Vulkan", "BrowserWebGpu".
    pub backend: String,
    /// Adapter class, e.g. "IntegratedGpu".
    pub device_type: String,
}

impl BackendInfo {
    /// One line for the About panel.
    pub fn summary(&self) -> String {
        format!(
            "{} ({}, {})",
            self.adapter_name, self.backend, self.device_type
        )
    }
}

/// Everything the UI reads and writes.
#[derive(Clone, Default)]
pub struct AppState {
    /// Status of each reconstruction stage (Pipeline panel).
    pub pipeline: Rc<RefCell<PipelineStatus>>,
    /// The 3D viewport's camera.
    pub camera: Rc<RefCell<OrbitCamera>>,
    /// What the 3D viewport draws.
    pub scene: Rc<RefCell<Scene>>,
    /// Whether the About / diagnostics sheet is showing.
    pub about_open: Rc<Cell<bool>>,
    /// Filled by the viewport the first time it paints on a wgpu backend; `None` headless.
    pub backend: Rc<RefCell<Option<BackendInfo>>>,
    /// Set by the host after the first presented frame; the app's ready signal.
    pub first_paint: Rc<FirstPaintGate>,
}

impl AppState {
    /// Fresh state: every stage "Not yet ported", default camera and scene, About closed.
    pub fn new() -> Self {
        Self::default()
    }

    /// True once a frame has been painted and presented (see [`FirstPaintGate`]).
    pub fn is_ready(&self) -> bool {
        self.first_paint.has_painted()
    }
}
