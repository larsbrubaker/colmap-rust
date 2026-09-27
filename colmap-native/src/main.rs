// colmap-native: the desktop shell for COLMAP Rust (binary `colmap-rust`). Opens an agg-gui-shell
// window, installs the shared theme/fonts, builds `colmap_app::build_app`, and marks the app
// presented (its ready signal) after each painted frame. Everything the user sees lives in
// colmap-app.
//
// Usage: `colmap-rust [--screenshot <file.png>]` — the flag paints a few settle frames, writes
// the window to a PNG through agg-gui-shell's deterministic capture, and exits.

mod bounds_store;

use agg_gui::App;
use agg_gui_shell::{run, Frame, ShellConfig, ShellError, ShellHost, WgpuGfxCtx};
use colmap_app::{build_app, install_theme_and_fonts, AppState, APP_TITLE};

use bounds_store::{default_bounds_path, FileBoundsStore};

/// App-side shell hooks.
struct NativeHost {
    state: AppState,
}

impl ShellHost for NativeHost {
    fn after_paint(&mut self, _ctx: &mut WgpuGfxCtx, _frame: &Frame) {
        // Runs after `end_frame`, right before `present`: the frame is complete. agg-gui-shell
        // has no `after_present` hook (agg-gui-web-shell does), so this is the closest point.
        self.state.mark_presented();
    }
}

fn screenshot_arg() -> Option<String> {
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == "--screenshot" {
            return args.next();
        }
    }
    None
}

fn main() -> Result<(), ShellError> {
    let mut config = ShellConfig::new(APP_TITLE)
        .with_logical_size(1280.0, 800.0)
        .with_min_logical_size(640.0, 400.0)
        .with_device_label("colmap-rust")
        // Not FIFO: with a reactive redraw loop, FIFO can block surface acquisition for several
        // vblanks on Windows (measured in AtomArtist). AutoNoVsync picks Mailbox/Immediate.
        .with_present_mode(wgpu::PresentMode::AutoNoVsync);
    if let Some(path) = default_bounds_path() {
        config = config.with_bounds_store(FileBoundsStore::new(path));
    }
    if let Some(path) = screenshot_arg() {
        config = config.with_screenshot(path, 3);
    }

    let state = AppState::new();
    run(config, move |init| {
        let font = install_theme_and_fonts(init.device_scale()).map_err(ShellError::app)?;
        let app = App::new(build_app(&state, font));
        Ok((app, NativeHost { state }))
    })
}
