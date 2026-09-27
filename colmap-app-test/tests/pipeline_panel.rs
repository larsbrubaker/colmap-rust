// Headless UI tests for the Pipeline panel (colmap-app `pipeline_panel.rs`): every stage is
// listed with the status held in `AppState::pipeline`, and the labels follow state changes.

use colmap_app::pipeline_panel::{status_label_id, PIPELINE_PANEL_ID};
use colmap_app::{Stage, StageStatus};
use colmap_app_test::TestHarness;

#[test]
fn pipeline_panel_lists_every_stage_with_its_status() {
    let h = TestHarness::new();
    assert!(
        h.screen_rect(PIPELINE_PANEL_ID).is_some(),
        "panel is on screen"
    );
    let mut last_top = f64::INFINITY;
    for stage in Stage::ALL {
        let row = h
            .screen_rect(&stage.row_id())
            .unwrap_or_else(|| panic!("row for {stage:?} is on screen"));
        // Rows run top to bottom in pipeline order (Y-up: each row's top is lower).
        assert!(
            row.y + row.height < last_top,
            "{stage:?} is below the previous stage"
        );
        last_top = row.y + row.height;
        let status = h.property(&status_label_id(stage), "text");
        let expected = h.state().pipeline.borrow().status(stage).label();
        assert_eq!(status.as_deref(), Some(expected.as_str()), "{stage:?}");
        assert_eq!(expected, "Not yet ported");
    }
}

#[test]
fn flipping_a_stage_status_updates_its_label() {
    let mut h = TestHarness::new();
    h.state()
        .pipeline
        .borrow_mut()
        .set_status(Stage::Features, StageStatus::Running { progress: 0.25 });
    h.frame();
    assert_eq!(
        h.property(&status_label_id(Stage::Features), "text")
            .as_deref(),
        Some("Running 25%")
    );
    // Other stages are untouched.
    assert_eq!(
        h.property(&status_label_id(Stage::Matching), "text")
            .as_deref(),
        Some("Not yet ported")
    );
    h.state()
        .pipeline
        .borrow_mut()
        .set_status(Stage::Features, StageStatus::Done);
    h.frame();
    assert_eq!(
        h.property(&status_label_id(Stage::Features), "text")
            .as_deref(),
        Some("Done")
    );
}
