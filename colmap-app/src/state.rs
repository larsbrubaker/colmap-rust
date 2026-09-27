// Application state shared by the widget tree, the shells and the headless test harness.
// Every field is a shared cell: widgets hold clones and read them on each layout / paint, and
// tests mutate the same cells directly (colmap-app-test), so there is one source of truth.
//
// `Rc` rather than `Arc`: the UI lives on one thread (the browser main thread on wasm). Long
// pipeline work will run off-thread and report back through a channel the UI drains, which is
// a later phase's decision.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::backend_info::BackendInfo;
use crate::camera::OrbitCamera;
use crate::pipeline::PipelineStatus;
use crate::scene::Scene;

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
    /// The viewport renderer's current adapter, republished whenever it rebuilds its GPU
    /// objects on a new device (device loss, backend switch); `None` headless.
    pub backend: Rc<RefCell<Option<BackendInfo>>>,
    /// Set by the shell once a frame is on screen; the app's ready signal (see
    /// [`AppState::mark_presented`]).
    pub presented: Rc<Cell<bool>>,
}

impl AppState {
    /// Fresh state: every stage "Not yet ported", default camera and scene, About closed.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record that a frame has been presented. Each shell calls it from the point where its
    /// frame is on screen: colmap-web from the web shell's `after_present`, colmap-native from
    /// `after_paint` (agg-gui-shell has no post-present hook; `after_paint` runs after
    /// `end_frame`, immediately before `present`). A tick that bails before presenting never
    /// reaches either hook, so the flag stays false. The shells themselves own "always paint
    /// the first frame" (agg-gui-shell / agg-gui-web-shell), so no gate logic lives here.
    pub fn mark_presented(&self) {
        self.presented.set(true);
    }

    /// True once a frame has been presented.
    pub fn is_ready(&self) -> bool {
        self.presented.get()
    }
}
