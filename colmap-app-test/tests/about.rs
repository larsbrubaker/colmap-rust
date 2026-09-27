// Headless UI tests for the top bar's info button and the About / diagnostics sheet
// (colmap-app `chrome.rs`).

use agg_gui::Key;
use colmap_app::chrome::{
    colmap_reference, ABOUT_BACKEND_ID, ABOUT_BUTTON_ID, ABOUT_CLOSE_ID, ABOUT_PANEL_ID,
};
use colmap_app_test::TestHarness;

#[test]
fn info_button_opens_the_about_panel() {
    let mut h = TestHarness::new();
    assert!(!h.state().about_open.get());
    assert!(h.screen_rect(ABOUT_PANEL_ID).is_none(), "hidden at start");
    h.click_id(ABOUT_BUTTON_ID);
    assert!(h.state().about_open.get(), "info button sets about_open");
    assert!(
        h.screen_rect(ABOUT_PANEL_ID).is_some(),
        "About panel visible"
    );
    assert_eq!(
        h.property(ABOUT_BACKEND_ID, "text").as_deref(),
        Some("Renderer: no GPU adapter reported yet")
    );
}

#[test]
fn close_button_and_escape_dismiss_the_about_panel() {
    let mut h = TestHarness::new();
    h.click_id(ABOUT_BUTTON_ID);
    h.click_id(ABOUT_CLOSE_ID);
    assert!(!h.state().about_open.get(), "Close hides the sheet");
    assert!(h.screen_rect(ABOUT_PANEL_ID).is_none());

    h.click_id(ABOUT_BUTTON_ID);
    assert!(h.state().about_open.get());
    h.key_down(Key::Escape);
    assert!(!h.state().about_open.get(), "Escape hides the sheet");
}

#[test]
fn about_shows_the_pinned_colmap_reference() {
    let (version, commit) = colmap_reference();
    let reference = include_str!("../../REFERENCE");
    assert!(reference.starts_with(version));
    assert!(reference.contains(commit));
    assert_eq!(version, "4.2.0");
}

#[test]
fn backend_line_follows_state() {
    let mut h = TestHarness::new();
    *h.state().backend.borrow_mut() = Some(colmap_app::BackendInfo {
        adapter_name: "Test Adapter".to_string(),
        backend: "Metal".to_string(),
        device_type: "integrated GPU".to_string(),
        driver: String::new(),
    });
    h.click_id(ABOUT_BUTTON_ID);
    assert_eq!(
        h.property(ABOUT_BACKEND_ID, "text").as_deref(),
        Some("Renderer: Test Adapter, Metal, integrated GPU")
    );
}

#[test]
fn backend_line_for_a_browser_adapter_omits_empty_fields() {
    let mut h = TestHarness::new();
    *h.state().backend.borrow_mut() = Some(colmap_app::BackendInfo {
        adapter_name: String::new(),
        backend: "WebGPU (browser)".to_string(),
        device_type: String::new(),
        driver: String::new(),
    });
    h.click_id(ABOUT_BUTTON_ID);
    assert_eq!(
        h.property(ABOUT_BACKEND_ID, "text").as_deref(),
        Some("Renderer: WebGPU (browser)")
    );
}
