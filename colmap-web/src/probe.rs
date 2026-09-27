//! The smoke test's frame probe (wasm32 only): `window.__colmap_frame_stats()` returns a Promise
//! of the [`crate::frame_stats::FrameStats`] of the next rendered frame, read back from the GPU.
//!
//! Installed only when the page URL carries `?test=1` (`crate::frame_stats::test_hooks_requested`),
//! which also turns on the shell's offscreen scene (`WebShellConfig::with_offscreen_scene`): the
//! app renders into a copyable texture that is then blitted to the canvas, because a WebGPU canvas
//! texture itself can't be copied from. That costs a full-size texture and a blit per frame, so the
//! production page (no flag) doesn't pay it.
//!
//! Flow, all on the browser main thread: the JS call parks the Promise's resolve/reject here and
//! marks the shell dirty; the next painted frame's `after_paint` snapshots the scene texture
//! (`capture_screenshot`) and starts a non-blocking readback (`begin_capture_readback`); following
//! frames (kept coming with `mark_dirty`) poll it until the pixels arrive, then every waiter is
//! resolved with the stats computed in Rust.

use std::cell::RefCell;
use std::rc::Rc;

use agg_gui_web_shell::agg_gui::DrawCtx;
use agg_gui_web_shell::{mark_dirty, web_sys, WgpuGfxCtx};
use wasm_bindgen::prelude::*;

use crate::frame_stats::{frame_stats, FrameStats};

/// Name of the page-level probe function.
const PROBE_FN: &str = "__colmap_frame_stats";

type Waiter = (js_sys::Function, js_sys::Function);

/// Probe state shared by the JS-facing closure and the shell host.
#[derive(Default)]
pub struct FrameProbe {
    /// Promises waiting for the next readback: (resolve, reject).
    waiting: Vec<Waiter>,
    /// A readback has been started and not harvested yet.
    in_flight: bool,
}

pub type SharedProbe = Rc<RefCell<FrameProbe>>;

/// Create the probe and expose `window.__colmap_frame_stats`.
pub fn install() -> Result<SharedProbe, JsValue> {
    let probe: SharedProbe = Rc::default();
    let for_js = Rc::clone(&probe);
    let request = Closure::<dyn Fn() -> js_sys::Promise>::new(move || {
        let probe = Rc::clone(&for_js);
        let promise = js_sys::Promise::new(&mut |resolve, reject| {
            probe.borrow_mut().waiting.push((resolve, reject));
        });
        mark_dirty();
        promise
    });
    let window = web_sys::window().ok_or("no window")?;
    js_sys::Reflect::set(&window, &JsValue::from_str(PROBE_FN), request.as_ref())?;
    // The page keeps the function for its whole life.
    request.forget();
    Ok(probe)
}

/// Drive the probe from the shell's `after_paint` (the scene texture holds this frame).
pub fn after_paint(probe: &SharedProbe, ctx: &mut WgpuGfxCtx) {
    let mut p = probe.borrow_mut();
    if p.in_flight {
        if let Some((rgba, w, h)) = ctx.poll_capture_readback() {
            p.in_flight = false;
            let waiters = std::mem::take(&mut p.waiting);
            match frame_stats(&rgba, w, h) {
                Some(stats) => settle(waiters, Ok(stats_object(&stats))),
                None => settle(waiters, Err("frame readback had the wrong size")),
            }
        } else if ctx.has_pending_readback() {
            mark_dirty(); // not mapped yet: look again next frame
        } else {
            p.in_flight = false;
            let waiters = std::mem::take(&mut p.waiting);
            settle(waiters, Err("frame readback failed to map"));
        }
    } else if !p.waiting.is_empty() {
        if ctx.capture_screenshot() && ctx.begin_capture_readback() {
            p.in_flight = true;
            mark_dirty();
        } else {
            let waiters = std::mem::take(&mut p.waiting);
            settle(
                waiters,
                Err("frame readback unavailable (offscreen scene off?)"),
            );
        }
    }
}

fn settle(waiters: Vec<Waiter>, result: Result<JsValue, &str>) {
    for (resolve, reject) in waiters {
        let _ = match &result {
            Ok(v) => resolve.call1(&JsValue::NULL, v),
            Err(msg) => reject.call1(&JsValue::NULL, &js_sys::Error::new(msg)),
        };
    }
}

/// `{ width, height, distinct, dominantFraction, red, green, blue }` — the smoke test's shape.
fn stats_object(s: &FrameStats) -> JsValue {
    let obj = js_sys::Object::new();
    let fields: [(&str, f64); 7] = [
        ("width", s.width as f64),
        ("height", s.height as f64),
        ("distinct", s.distinct as f64),
        ("dominantFraction", s.dominant_fraction),
        ("red", s.red as f64),
        ("green", s.green as f64),
        ("blue", s.blue as f64),
    ];
    for (k, v) in fields {
        let _ = js_sys::Reflect::set(&obj, &JsValue::from_str(k), &JsValue::from_f64(v));
    }
    obj.into()
}
