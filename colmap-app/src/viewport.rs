// The 3D viewport widget (id `viewport-3d`): maps pointer gestures onto the shared
// `OrbitCamera` (left drag orbits, right/middle drag pans, wheel zooms) and, on a wgpu backend,
// queues `viewport_render::ViewportRenderer` into the frame through agg-gui-wgpu's custom-render
// hook. On any other DrawCtx (the headless harness's software context) it paints only its 2D
// background, so the widget tree and its input handling work with no GPU at all.

use std::cell::RefCell;
use std::rc::Rc;

use agg_gui::{
    Color, DrawCtx, Event, EventResult, HAnchor, MouseButton, Point, Rect, Size, TransAffine,
    VAnchor, Widget,
};
use agg_gui_wgpu::{SharedCustomRenderer, WgpuGfxCtx};

use crate::state::AppState;
use crate::viewport_render::ViewportRenderer;

/// Widget id of the viewport.
pub const VIEWPORT_ID: &str = "viewport-3d";

/// Background behind the scene (dark neutral, like most reconstruction viewers).
const BACKGROUND: Color = Color::rgb(0.11, 0.115, 0.13);

/// What a held button does while dragging.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DragMode {
    Orbit,
    Pan,
}

pub struct Viewport3d {
    bounds: Rect,
    children: Vec<Box<dyn Widget>>,
    state: AppState,
    renderer: Rc<RefCell<ViewportRenderer>>,
    drag: Option<(DragMode, Point)>,
}

impl Viewport3d {
    pub fn new(state: &AppState) -> Self {
        let renderer = ViewportRenderer::new(
            state.scene.clone(),
            state.camera.clone(),
            state.backend.clone(),
        );
        Self {
            bounds: Rect::default(),
            children: Vec::new(),
            state: state.clone(),
            renderer: Rc::new(RefCell::new(renderer)),
            drag: None,
        }
    }
}

/// Axis-aligned bounds of a `width` x `height` widget under transform `t` (target pixels).
fn transformed_rect(t: &TransAffine, width: f64, height: f64) -> Rect {
    let mut min = (f64::INFINITY, f64::INFINITY);
    let mut max = (f64::NEG_INFINITY, f64::NEG_INFINITY);
    for (mut x, mut y) in [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)] {
        t.transform(&mut x, &mut y);
        min = (min.0.min(x), min.1.min(y));
        max = (max.0.max(x), max.1.max(y));
    }
    Rect::new(min.0, min.1, max.0 - min.0, max.1 - min.1)
}

impl Widget for Viewport3d {
    fn type_name(&self) -> &'static str {
        "Viewport3d"
    }
    fn id(&self) -> Option<&str> {
        Some(VIEWPORT_ID)
    }
    fn bounds(&self) -> Rect {
        self.bounds
    }
    fn set_bounds(&mut self, bounds: Rect) {
        self.bounds = bounds;
    }
    fn children(&self) -> &[Box<dyn Widget>] {
        &self.children
    }
    fn children_mut(&mut self) -> &mut Vec<Box<dyn Widget>> {
        &mut self.children
    }
    fn h_anchor(&self) -> HAnchor {
        HAnchor::STRETCH
    }
    fn v_anchor(&self) -> VAnchor {
        VAnchor::STRETCH
    }
    fn layout(&mut self, available: Size) -> Size {
        available
    }

    fn paint(&mut self, ctx: &mut dyn DrawCtx) {
        ctx.set_fill_color(BACKGROUND);
        ctx.begin_path();
        ctx.rect(0.0, 0.0, self.bounds.width, self.bounds.height);
        ctx.fill();

        let screen_rect = transformed_rect(&ctx.transform(), self.bounds.width, self.bounds.height);
        let Some(wgpu_ctx) = ctx
            .as_any_mut()
            .and_then(|any| any.downcast_mut::<WgpuGfxCtx>())
        else {
            return;
        };
        // Annotated so the `Rc<RefCell<ViewportRenderer>>` unsizes to the trait object.
        let shared: SharedCustomRenderer = self.renderer.clone();
        wgpu_ctx.push_custom_render(shared, screen_rect);
    }

    fn on_event(&mut self, event: &Event) -> EventResult {
        match event {
            Event::MouseDown { pos, button, .. } => {
                let mode = match button {
                    MouseButton::Left => DragMode::Orbit,
                    MouseButton::Right | MouseButton::Middle => DragMode::Pan,
                    MouseButton::Other(_) => return EventResult::Ignored,
                };
                self.drag = Some((mode, *pos));
                EventResult::Consumed
            }
            Event::MouseMove { pos } => {
                let Some((mode, last)) = self.drag else {
                    return EventResult::Ignored;
                };
                let (dx, dy) = ((pos.x - last.x) as f32, (pos.y - last.y) as f32);
                {
                    let mut camera = self.state.camera.borrow_mut();
                    match mode {
                        DragMode::Orbit => camera.orbit(dx, dy),
                        DragMode::Pan => camera.pan(dx, dy, self.bounds.height as f32),
                    }
                }
                self.drag = Some((mode, *pos));
                EventResult::Consumed
            }
            Event::MouseUp { .. } => {
                if self.drag.take().is_some() {
                    EventResult::Consumed
                } else {
                    EventResult::Ignored
                }
            }
            Event::MouseWheel { delta_y, .. } => {
                self.state.camera.borrow_mut().zoom(*delta_y as f32);
                EventResult::Consumed
            }
            _ => EventResult::Ignored,
        }
    }

    fn properties(&self) -> Vec<(&'static str, String)> {
        let camera = self.state.camera.borrow();
        vec![
            ("yaw", format!("{:.4}", camera.yaw)),
            ("pitch", format!("{:.4}", camera.pitch)),
            ("distance", format!("{:.4}", camera.distance)),
            ("layers", self.state.scene.borrow().layer_names().join(",")),
        ]
    }
}
