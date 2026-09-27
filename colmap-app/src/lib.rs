//! colmap-app: the COLMAP Rust application, built on agg-gui.
//!
//! [`build_app`] returns the production widget tree. The native shell (`colmap-native`), the
//! web shell (`colmap-web`) and the headless UI harness (`colmap-app-test`) all build exactly this
//! tree over an [`AppState`], after calling [`install_theme_and_fonts`].
//!
//! Layout: a top bar (title + About button), then a row with the Pipeline panel on the left and
//! the 3D viewport filling the rest, with the About sheet stacked over everything.
//!
//! Modules: `camera` (orbit camera math), `scene` (drawable layers), `viewport` +
//! `viewport_render` (the wgpu custom-render widget), `pipeline` + `pipeline_panel` (stage
//! status), `chrome` (top bar, About), `backend_info` (GPU adapter labels), `state` (incl. the
//! presented / ready flag), `fonts`, `widgets`.

pub mod backend_info;
pub mod camera;
pub mod chrome;
pub mod fonts;
pub mod pipeline;
pub mod pipeline_panel;
pub mod scene;
pub mod state;
pub mod viewport;
pub mod viewport_render;
pub mod widgets;

use std::sync::Arc;

use agg_gui::text::Font;
use agg_gui::{FlexColumn, FlexRow, Stack, Widget};

pub use backend_info::BackendInfo;
pub use chrome::APP_TITLE;
pub use fonts::install_theme_and_fonts;
pub use pipeline::{PipelineStatus, Stage, StageStatus};
pub use state::AppState;

/// Build the production widget tree over `state`. `font` is the UI font returned by
/// [`install_theme_and_fonts`].
pub fn build_app(state: &AppState, font: Arc<Font>) -> Box<dyn Widget> {
    // Zero gaps: the panels butt against each other and the viewport, no seams.
    let main_row = FlexRow::new()
        .with_gap(0.0)
        .add(pipeline_panel::build_pipeline_panel(state, &font))
        .add_flex(Box::new(viewport::Viewport3d::new(state)), 1.0);
    let column = FlexColumn::new()
        .with_gap(0.0)
        .with_top_anchor(true)
        .add(chrome::build_top_bar(state, &font))
        .add_flex(Box::new(main_row), 1.0);
    Box::new(
        Stack::new()
            .add(Box::new(column))
            .add(chrome::build_about_sheet(state, &font)),
    )
}
