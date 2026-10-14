// Integration-test binary for COLMAP's `scene/` module (layout convention: see the header of
// `tests/math.rs`): one file per colmap/scene/*_test.cc, ported 1:1, plus the `rust_only_*`
// files (error paths and the documented divergences).
//
// Run: `cargo test -p colmap-rust --test scene`.

#[path = "support/eigen_matchers.rs"]
mod eigen_matchers;
#[path = "support/transform_matchers.rs"]
mod transform_matchers;

#[path = "scene/camera.rs"]
mod camera;
#[path = "scene/correspondence_graph.rs"]
mod correspondence_graph;
#[path = "scene/database.rs"]
mod database;
#[path = "scene/database_cache.rs"]
mod database_cache;
#[path = "scene/frame.rs"]
mod frame;
#[path = "scene/image.rs"]
mod image;
#[path = "scene/point2d.rs"]
mod point2d;
#[path = "scene/point3d.rs"]
mod point3d;
#[path = "scene/projection.rs"]
mod projection;
#[path = "scene/reconstruction.rs"]
mod reconstruction;
#[path = "scene/reconstruction_io.rs"]
mod reconstruction_io;
#[path = "scene/reconstruction_queries.rs"]
mod reconstruction_queries;
#[path = "scene/reconstruction_synthetic.rs"]
mod reconstruction_synthetic;
#[path = "scene/reconstruction_transform.rs"]
mod reconstruction_transform;
#[path = "scene/rust_only_correspondence_graph.rs"]
mod rust_only_correspondence_graph;
#[path = "scene/rust_only_database.rs"]
mod rust_only_database;
#[path = "scene/rust_only_frame_image.rs"]
mod rust_only_frame_image;
#[path = "scene/rust_only_reconstruction.rs"]
mod rust_only_reconstruction;
#[path = "scene/rust_only_reconstruction_io.rs"]
mod rust_only_reconstruction_io;
#[path = "scene/rust_only_scene_types.rs"]
mod rust_only_scene_types;
#[path = "scene/rust_only_two_view_geometry.rs"]
mod rust_only_two_view_geometry;
#[path = "scene/synthetic.rs"]
mod synthetic;
#[path = "scene/synthetic_database.rs"]
mod synthetic_database;
#[path = "scene/synthetic_noise.rs"]
mod synthetic_noise;
#[path = "scene/track.rs"]
mod track;
#[path = "scene/two_view_geometry.rs"]
mod two_view_geometry;
#[path = "scene/visibility_pyramid.rs"]
mod visibility_pyramid;
