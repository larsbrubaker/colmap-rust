// Headless UI test for the app's ready signal (`AppState::mark_presented` / `is_ready`): not
// ready until a shell reports a presented frame; painting alone (no present) does not flip it.

use colmap_app_test::TestHarness;

#[test]
fn app_is_ready_only_after_a_presented_frame() {
    let mut h = TestHarness::new();
    assert!(!h.state().is_ready());
    h.paint_once();
    assert!(!h.state().is_ready(), "painting is not presenting");
    h.state().mark_presented();
    assert!(h.state().is_ready());
}
