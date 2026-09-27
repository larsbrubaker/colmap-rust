// Headless UI tests for the 3D viewport widget (colmap-app `viewport.rs`): pointer gestures on
// the real widget tree drive the shared orbit camera, and the widget paints without a GPU.

use agg_gui::MouseButton;
use colmap_app::camera::{MAX_DISTANCE, MIN_DISTANCE, ZOOM_STEP_PER_NOTCH};
use colmap_app::viewport::VIEWPORT_ID;
use colmap_app_test::TestHarness;

fn viewport_center(h: &TestHarness) -> (f64, f64) {
    h.center_of(VIEWPORT_ID).expect("viewport is on screen")
}

#[test]
fn viewport_fills_the_space_right_of_the_panel() {
    let h = TestHarness::new();
    let r = h.screen_rect(VIEWPORT_ID).expect("viewport is on screen");
    assert!(r.width > h.size().0 * 0.6, "viewport width {}", r.width);
    assert!(r.height > h.size().1 * 0.8, "viewport height {}", r.height);
    assert_eq!(
        h.property(VIEWPORT_ID, "layers").as_deref(),
        Some("grid,axis-gizmo")
    );
}

#[test]
fn left_drag_orbits_the_camera() {
    let mut h = TestHarness::new();
    let before = *h.state().camera.borrow();
    let (x, y) = viewport_center(&h);
    h.drag((x, y), (x + 80.0, y - 40.0), MouseButton::Left, 4);
    let after = *h.state().camera.borrow();
    assert_ne!(after.yaw, before.yaw, "yaw changed");
    assert_ne!(after.pitch, before.pitch, "pitch changed");
    assert_eq!(after.distance, before.distance, "orbit does not zoom");
    assert_eq!(after.target, before.target, "orbit does not pan");
}

#[test]
fn right_drag_pans_the_camera() {
    let mut h = TestHarness::new();
    let before = *h.state().camera.borrow();
    let (x, y) = viewport_center(&h);
    h.drag((x, y), (x + 50.0, y + 30.0), MouseButton::Right, 2);
    let after = *h.state().camera.borrow();
    assert_ne!(after.target, before.target, "target moved");
    assert_eq!(after.yaw, before.yaw);
    assert_eq!(after.pitch, before.pitch);
}

#[test]
fn wheel_zooms_and_clamps_distance() {
    let mut h = TestHarness::new();
    let (x, y) = viewport_center(&h);
    h.mouse_move(x, y);
    let d0 = h.state().camera.borrow().distance;
    h.scroll(1.0);
    let d1 = h.state().camera.borrow().distance;
    assert!(
        d1 < d0,
        "forward wheel (away from the user) zooms in: {d0} -> {d1}"
    );
    assert!(
        (d1 - d0 * ZOOM_STEP_PER_NOTCH).abs() < 1e-4 * d0,
        "one notch is one AtomArtist-sized step: {d0} -> {d1}"
    );
    h.scroll(-2.0);
    assert!(
        h.state().camera.borrow().distance > d1,
        "backward wheel zooms out"
    );
    for _ in 0..200 {
        h.scroll(10.0);
    }
    assert_eq!(h.state().camera.borrow().distance, MIN_DISTANCE);
    for _ in 0..400 {
        h.scroll(-10.0);
    }
    assert_eq!(h.state().camera.borrow().distance, MAX_DISTANCE);
}

#[test]
fn wheel_over_the_pipeline_panel_does_not_zoom() {
    let mut h = TestHarness::new();
    let (x, y) = h
        .center_of(colmap_app::pipeline_panel::PIPELINE_PANEL_ID)
        .expect("panel on screen");
    h.mouse_move(x, y);
    let d0 = h.state().camera.borrow().distance;
    h.scroll(1.0);
    assert_eq!(h.state().camera.borrow().distance, d0);
}

#[test]
fn tree_paints_without_a_gpu() {
    let mut h = TestHarness::new();
    h.paint_once();
    // No wgpu context headless, so the viewport never reports a backend.
    assert!(h.state().backend.borrow().is_none());
}
