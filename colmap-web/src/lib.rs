//! colmap-web: the browser shell for COLMAP Rust (wasm32 only; natively this crate is empty).
//!
//! Boots `agg-gui-web-shell` on `<canvas id="canvas">` with the WebGPU backend — required, not
//! preferred, because the 3D viewport and the later PatchMatch compute kernels need WebGPU. A
//! browser without it gets the shell's fatal panel in place of the canvas (and `web/index.html`
//! checks `navigator.gpu` first, so the common case shows the app's own message).
//!
//! Like `colmap-native`, it only wires the shell: installs the shared theme and fonts at the
//! device scale and builds `colmap_app::build_app`. From the shell's `after_present` hook (the
//! frame is on the canvas) it marks the app presented (`AppState::mark_presented`) and, once,
//! tells the page for the Playwright smoke test (`web/tests/smoke.spec.ts`):
//! `window.__colmapReady = true` and `data-ready="1"` on the canvas. Everything the user sees
//! lives in colmap-app.
//!
//! With `?test=1` in the page URL it also renders through the shell's offscreen scene and installs
//! `window.__colmap_frame_stats()` (`probe.rs`), which reads the rendered frame back from the GPU
//! and returns its pixel statistics (`frame_stats.rs`) — what the smoke test asserts on.

pub mod frame_stats;

#[cfg(target_arch = "wasm32")]
mod probe;

#[cfg(target_arch = "wasm32")]
mod web {
    use agg_gui_web_shell::agg_gui::App;
    use agg_gui_web_shell::{
        start, web_sys, Backend, Frame, RedrawPolicy, WebShellConfig, WebShellError, WebShellHost,
        WgpuGfxCtx,
    };
    use colmap_app::{build_app, install_theme_and_fonts, AppState};

    use crate::frame_stats::test_hooks_requested;
    use crate::probe::{self, SharedProbe};
    use wasm_bindgen::prelude::*;

    /// `id` of the canvas in `web/index.html`.
    const CANVAS_ID: &str = "canvas";
    /// The page-level ready flag the smoke test waits for.
    const READY_FLAG: &str = "__colmapReady";

    /// App-side shell hooks.
    struct WebHost {
        state: AppState,
        canvas: web_sys::HtmlCanvasElement,
        announced: bool,
        /// The smoke test's frame probe, present only with `?test=1`.
        probe: Option<SharedProbe>,
    }

    impl WebShellHost for WebHost {
        fn after_paint(&mut self, ctx: &mut WgpuGfxCtx, _frame: &Frame) {
            if let Some(p) = &self.probe {
                probe::after_paint(p, ctx);
            }
        }

        fn after_present(&mut self, _app: &mut App, _frame: &Frame) {
            // The frame is on the canvas: the app is ready, and the page hears it once.
            self.state.mark_presented();
            if !self.announced {
                announce_ready(&self.canvas);
                self.announced = true;
            }
        }
    }

    /// Publish readiness to the page: `window.__colmapReady = true` and `data-ready="1"`.
    fn announce_ready(canvas: &web_sys::HtmlCanvasElement) {
        let _ = canvas.set_attribute("data-ready", "1");
        if let Some(window) = web_sys::window() {
            let _ = js_sys::Reflect::set(&window, &JsValue::from_str(READY_FLAG), &JsValue::TRUE);
        }
    }

    /// wasm entry point, run by wasm-bindgen when the module is instantiated.
    #[wasm_bindgen(start)]
    pub fn start_colmap_web() -> Result<(), JsValue> {
        let test_hooks = web_sys::window()
            .and_then(|w| w.location().search().ok())
            .is_some_and(|search| test_hooks_requested(&search));
        let probe = if test_hooks {
            Some(probe::install()?)
        } else {
            None
        };
        let config = WebShellConfig::new(CANVAS_ID)
            .with_backend(Backend::WebGpu)
            .with_app_name("COLMAP Rust")
            .with_device_label("colmap-rust")
            .with_redraw_policy(RedrawPolicy::Reactive)
            .with_offscreen_scene(test_hooks);
        let state = AppState::new();
        start(config, move |init| {
            let font = install_theme_and_fonts(init.device_scale()).map_err(WebShellError::app)?;
            let app = App::new(build_app(&state, font));
            let host = WebHost {
                state,
                canvas: init.canvas().clone(),
                announced: false,
                probe,
            };
            Ok((app, host))
        })
        .map_err(|e| JsValue::from_str(&e.to_string()))
    }
}
