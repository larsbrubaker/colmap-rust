//! Headless UI test harness for COLMAP Rust (testing net 3, CLAUDE.md).
//!
//! [`TestHarness`] builds the real production tree (`colmap_app::build_app`) over a fresh
//! [`AppState`] after the shells' own startup (`install_theme_and_fonts`), with no window and
//! no GPU. Tests drive it with synthetic mouse / wheel / key events that go through agg-gui's
//! `App` dispatch exactly as a shell's would, re-laying out after every event, then assert on
//! the live `AppState` and on the widget tree through agg-gui reflection (`find_widget_by_id`,
//! `find_widget_screen_rect`, `Widget::properties`).
//!
//! Modeled on AtomArtist's `atomartist-ui-test` harness.
//!
//! Coordinates: event helpers take agg-gui's physical-pixel screen space (origin top-left,
//! Y down), like `App::on_mouse_*`; the harness runs at device scale 1, so physical equals
//! logical. Widget rects from reflection are Y-up; [`TestHarness::center_of`] converts.

use agg_gui::widget::{find_widget_by_id, find_widget_screen_rect};
use agg_gui::{App, Key, Modifiers, MouseButton, Rect, Size, Widget};
use colmap_app::{build_app, install_theme_and_fonts, AppState};

/// Default harness viewport, matching the native shell's default logical size.
pub const DEFAULT_WIDTH: f64 = 1280.0;
pub const DEFAULT_HEIGHT: f64 = 800.0;

/// State + driver for one UI test scenario.
pub struct TestHarness {
    state: AppState,
    app: App,
    cursor: (f64, f64),
    modifiers: Modifiers,
    size: (f64, f64),
}

impl TestHarness {
    /// Fresh state, real widget tree, laid out at the default size.
    pub fn new() -> Self {
        Self::with_state(AppState::new())
    }

    /// Boot the real tree over a caller-seeded state.
    pub fn with_state(state: AppState) -> Self {
        agg_gui::set_device_scale(1.0);
        // The shells' own startup, not a harness copy, so a registration startup stops
        // performing fails here too.
        let font = install_theme_and_fonts(1.0).expect("bundled fonts load");
        let mut app = App::new(build_app(&state, font));
        app.layout(Size::new(DEFAULT_WIDTH, DEFAULT_HEIGHT));
        Self {
            state,
            app,
            cursor: (0.0, 0.0),
            modifiers: Modifiers::default(),
            size: (DEFAULT_WIDTH, DEFAULT_HEIGHT),
        }
    }

    /// The live app state the widgets read and write.
    pub fn state(&self) -> &AppState {
        &self.state
    }

    pub fn app(&self) -> &App {
        &self.app
    }

    pub fn app_mut(&mut self) -> &mut App {
        &mut self.app
    }

    /// Harness viewport `(width, height)`.
    pub fn size(&self) -> (f64, f64) {
        self.size
    }

    /// Resize and re-layout.
    pub fn resize(&mut self, width: f64, height: f64) -> &mut Self {
        self.size = (width, height);
        self.frame()
    }

    /// A frame boundary without an event: re-run layout so state changes reach the tree.
    pub fn frame(&mut self) -> &mut Self {
        self.app.layout(Size::new(self.size.0, self.size.1));
        self
    }

    /// Paint once into a throwaway software framebuffer. Exercises every widget's `paint` on
    /// the non-wgpu path (the viewport must degrade to its 2D background); pixels are dropped.
    pub fn paint_once(&mut self) -> &mut Self {
        let mut fb = agg_gui::Framebuffer::new(self.size.0 as u32, self.size.1 as u32);
        let mut ctx = agg_gui::GfxCtx::new(&mut fb);
        self.app.paint(&mut ctx);
        self
    }

    // ── Reflection ────────────────────────────────────────────────────────

    /// First widget (DFS) whose `id()` is `id`.
    pub fn find_by_id(&self, id: &str) -> Option<&dyn Widget> {
        find_widget_by_id(self.app.root(), id)
    }

    /// Y-up screen rect of the visible widget with `id`.
    pub fn screen_rect(&self, id: &str) -> Option<Rect> {
        find_widget_screen_rect(self.app.root(), id)
    }

    /// Y-down screen point at the center of the visible widget with `id`.
    pub fn center_of(&self, id: &str) -> Option<(f64, f64)> {
        self.screen_rect(id)
            .map(|r| (r.x + r.width * 0.5, self.size.1 - (r.y + r.height * 0.5)))
    }

    /// Value of property `key` on the widget with `id` (`Widget::properties`).
    pub fn property(&self, id: &str, key: &str) -> Option<String> {
        self.find_by_id(id)?
            .properties()
            .into_iter()
            .find(|(k, _)| *k == key)
            .map(|(_, v)| v)
    }

    // ── Input ─────────────────────────────────────────────────────────────

    pub fn set_modifiers(&mut self, modifiers: Modifiers) -> &mut Self {
        self.modifiers = modifiers;
        self
    }

    /// Move the cursor (Y-down screen pixels).
    pub fn mouse_move(&mut self, x: f64, y: f64) -> &mut Self {
        self.cursor = (x, y);
        self.app.on_mouse_move(x, y);
        self.frame()
    }

    pub fn mouse_down(&mut self, button: MouseButton) -> &mut Self {
        let (x, y) = self.cursor;
        self.app.on_mouse_down(x, y, button, self.modifiers);
        self.frame()
    }

    pub fn mouse_up(&mut self, button: MouseButton) -> &mut Self {
        let (x, y) = self.cursor;
        self.app.on_mouse_up(x, y, button, self.modifiers);
        self.frame()
    }

    /// Move, press and release at one point.
    pub fn click(&mut self, x: f64, y: f64, button: MouseButton) -> &mut Self {
        self.mouse_move(x, y);
        self.mouse_down(button);
        self.mouse_up(button)
    }

    /// Click the center of the widget with `id`. Panics if it is not visible.
    pub fn click_id(&mut self, id: &str) -> &mut Self {
        let (x, y) = self
            .center_of(id)
            .unwrap_or_else(|| panic!("no visible widget with id {id:?}"));
        self.click(x, y, MouseButton::Left)
    }

    /// Press at `from`, move to `to` in `steps` equal moves, release.
    pub fn drag(
        &mut self,
        from: (f64, f64),
        to: (f64, f64),
        button: MouseButton,
        steps: u32,
    ) -> &mut Self {
        self.mouse_move(from.0, from.1);
        self.mouse_down(button);
        let steps = steps.max(1);
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            self.mouse_move(from.0 + (to.0 - from.0) * t, from.1 + (to.1 - from.1) * t);
        }
        self.mouse_up(button)
    }

    /// Wheel at the cursor, in agg-gui notches. Positive `delta_y` is a forward (zoom-in) step.
    pub fn scroll(&mut self, delta_y: f64) -> &mut Self {
        let (x, y) = self.cursor;
        self.app.on_mouse_wheel(x, y, delta_y);
        self.frame()
    }

    pub fn key_down(&mut self, key: Key) -> &mut Self {
        self.app.on_key_down(key, self.modifiers);
        self.frame()
    }
}

impl Default for TestHarness {
    fn default() -> Self {
        Self::new()
    }
}
