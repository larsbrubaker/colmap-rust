// Integration-test binary for COLMAP's `scene/` module (layout convention: see the header of
// `tests/math.rs`): one file per colmap/scene/*_test.cc, ported 1:1, plus
// `scene/rust_only_scene_types.rs` (error paths and the documented divergences).
//
// Run: `cargo test -p colmap-rust --test scene`.

#[path = "scene/camera.rs"]
mod camera;
#[path = "scene/point2d.rs"]
mod point2d;
#[path = "scene/point3d.rs"]
mod point3d;
#[path = "scene/projection.rs"]
mod projection;
#[path = "scene/rust_only_scene_types.rs"]
mod rust_only_scene_types;
#[path = "scene/rust_only_two_view_geometry.rs"]
mod rust_only_two_view_geometry;
#[path = "scene/track.rs"]
mod track;
#[path = "scene/two_view_geometry.rs"]
mod two_view_geometry;
#[path = "scene/visibility_pyramid.rs"]
mod visibility_pyramid;
