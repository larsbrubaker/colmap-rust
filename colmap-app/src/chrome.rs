// App chrome around the viewport: the top bar (title + Font Awesome info button) and the
// About / diagnostics sheet that button opens (version, pinned COLMAP reference, GPU backend).
// The sheet is an agg-gui `ModalSheet` driven by `AppState::about_open`.

use std::sync::Arc;

use agg_gui::text::Font;
use agg_gui::{
    current_visuals, Button, FlexColumn, FlexRow, HAnchor, Insets, Label, ModalSheet, Separator,
    Size, Spacer, VAnchor, Widget,
};

use crate::fonts::fa;
use crate::state::AppState;
use crate::widgets::{row_filler, BoundLabel, Tagged};

/// Window / top-bar title.
pub const APP_TITLE: &str = "COLMAP Rust";
/// Widget id of the info button in the top bar.
pub const ABOUT_BUTTON_ID: &str = "about-button";
/// Widget id of the About sheet's content.
pub const ABOUT_PANEL_ID: &str = "about-panel";
/// Widget id of the About sheet's Close button.
pub const ABOUT_CLOSE_ID: &str = "about-close";
/// Widget id of the backend line in the About sheet.
pub const ABOUT_BACKEND_ID: &str = "about-backend";

/// The `REFERENCE` file: "<COLMAP version> <commit sha>".
const REFERENCE: &str = include_str!("../../REFERENCE");

/// COLMAP reference version and commit this build ports, from `REFERENCE`.
pub fn colmap_reference() -> (&'static str, &'static str) {
    let mut parts = REFERENCE.split_whitespace();
    let version = parts.next().unwrap_or("unknown");
    let commit = parts.next().unwrap_or("unknown");
    (version, commit)
}

/// Text of the About sheet's backend line.
pub fn backend_line(state: &AppState) -> String {
    match state.backend.borrow().as_ref() {
        Some(info) => format!("Renderer: {}", info.summary()),
        None => "Renderer: no GPU adapter reported yet".to_string(),
    }
}

/// The top bar: title on the left, info button on the right.
pub fn build_top_bar(state: &AppState, font: &Arc<Font>) -> Box<dyn Widget> {
    let about_open = state.about_open.clone();
    let info = Button::new(fa::INFO_CIRCLE.to_string(), Arc::clone(font))
        .with_font_size(16.0)
        .with_ghost()
        .with_tooltip("About COLMAP Rust")
        .on_click(move || about_open.set(true));
    Box::new(
        FlexRow::new()
            .with_gap(8.0)
            .with_inner_padding(Insets::symmetric(12.0, 6.0))
            .with_background(current_visuals().top_bar_bg)
            .with_h_anchor(HAnchor::STRETCH)
            .add(Box::new(
                Label::new(APP_TITLE, Arc::clone(font))
                    .with_font_size(16.0)
                    .with_strong(true)
                    .with_v_anchor(VAnchor::CENTER),
            ))
            .add_flex(Box::new(row_filler()), 1.0)
            .add(Box::new(Tagged::new(ABOUT_BUTTON_ID, Box::new(info)))),
    )
}

/// The About / diagnostics sheet (hidden until `about_open` is set).
pub fn build_about_sheet(state: &AppState, font: &Arc<Font>) -> Box<dyn Widget> {
    let (reference, commit) = colmap_reference();
    let short_commit = commit.get(..12).unwrap_or(commit);
    let line = |text: String| -> Box<dyn Widget> {
        Box::new(Label::new(text, Arc::clone(font)).with_font_size(13.0))
    };
    let backend_state = state.clone();
    let backend = BoundLabel::new(
        Arc::clone(font),
        move || backend_line(&backend_state),
        |l| l.with_font_size(13.0),
    )
    .with_id(ABOUT_BACKEND_ID);
    let about_open = state.about_open.clone();
    let close = Button::new(format!("{}  Close", fa::TIMES), Arc::clone(font))
        .with_cancel_action()
        .on_click(move || about_open.set(false));
    let content = FlexColumn::new()
        .with_gap(8.0)
        .with_inner_padding(Insets::symmetric(18.0, 16.0))
        .with_top_anchor(true)
        .add(Box::new(
            Label::new(APP_TITLE, Arc::clone(font))
                .with_font_size(18.0)
                .with_strong(true),
        ))
        .add(line(
            "Pure-Rust port of COLMAP (Structure-from-Motion and Multi-View Stereo)".to_string(),
        ))
        .add(Box::new(Separator::horizontal()))
        .add(line(format!("Version {}", env!("CARGO_PKG_VERSION"))))
        .add(line(format!(
            "COLMAP reference: {reference} ({short_commit})"
        )))
        .add(Box::new(backend))
        .add_flex(Box::new(Spacer::new()), 1.0)
        .add(Box::new(Tagged::new(ABOUT_CLOSE_ID, Box::new(close))));
    Box::new(
        ModalSheet::new(
            state.about_open.clone(),
            Box::new(Tagged::new(ABOUT_PANEL_ID, Box::new(content))),
        )
        .with_panel_size(Size::new(520.0, 260.0)),
    )
}
