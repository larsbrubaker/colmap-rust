// The left "Pipeline" panel: one row per reconstruction stage (`pipeline::Stage::ALL`) with its
// title and a status label bound to `AppState::pipeline`. Rows are `Tagged` with
// `Stage::row_id()` and the status label has id `<row id>-status`, so tests and later phases
// can find them by reflection.

use std::sync::Arc;

use agg_gui::text::Font;
use agg_gui::{
    current_visuals, FlexColumn, FlexRow, HAnchor, Insets, Label, Separator, Size, VAnchor, Widget,
};

use crate::pipeline::Stage;
use crate::state::AppState;
use crate::widgets::{row_filler, BoundLabel, Tagged};

/// Widget id of the panel.
pub const PIPELINE_PANEL_ID: &str = "pipeline-panel";
/// Fixed panel width, logical pixels.
pub const PANEL_WIDTH: f64 = 260.0;

/// Widget id of a stage's status label.
pub fn status_label_id(stage: Stage) -> String {
    format!("{}-status", stage.row_id())
}

fn stage_row(state: &AppState, font: &Arc<Font>, stage: Stage) -> Box<dyn Widget> {
    let pipeline = state.pipeline.clone();
    let status = BoundLabel::new(
        Arc::clone(font),
        move || pipeline.borrow().status(stage).label(),
        |l| l.with_font_size(12.0).with_dim(true),
    )
    .with_id(status_label_id(stage));
    let row = FlexRow::new()
        .with_gap(8.0)
        .with_h_anchor(HAnchor::STRETCH)
        .add(Box::new(
            Label::new(stage.title(), Arc::clone(font)).with_font_size(13.0),
        ))
        .add_flex(Box::new(row_filler()), 1.0)
        .add(Box::new(status));
    Box::new(Tagged::new(stage.row_id(), Box::new(row)))
}

/// Build the panel.
pub fn build_pipeline_panel(state: &AppState, font: &Arc<Font>) -> Box<dyn Widget> {
    let visuals = current_visuals();
    let mut column = FlexColumn::new()
        .with_gap(6.0)
        .with_inner_padding(Insets::symmetric(12.0, 10.0))
        .with_background(visuals.panel_fill)
        .with_top_anchor(true)
        .with_v_anchor(VAnchor::STRETCH)
        .with_min_size(Size::new(PANEL_WIDTH, 0.0))
        .with_max_size(Size::new(PANEL_WIDTH, f64::MAX))
        .add(Box::new(
            Label::new("Pipeline", Arc::clone(font))
                .with_font_size(15.0)
                .with_strong(true),
        ))
        .add(Box::new(Separator::horizontal()));
    for stage in Stage::ALL {
        column = column.add(stage_row(state, font, stage));
    }
    Box::new(Tagged::new(PIPELINE_PANEL_ID, Box::new(column)))
}
