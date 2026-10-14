//! COLMAP's `scene/` module. Tests: `colmap-rust/tests/scene.rs`.
//!
//! Ownership: where COLMAP's scene types hold raw pointers to shared objects (`Frame` to its
//! `Rig`, `Image` to its `Camera` and `Frame`), the Rust types store only the ids, and the
//! owner of the maps (the reconstruction) passes the referenced objects to the methods that
//! need them, e.g. `frame.sensor_from_world(&rig, sensor_id)` or
//! `image.cam_from_world(&frame, &rig)`; docs/CPP_DIVERGENCES.md entry 170.
//!
//! - [`point2d`]: `point2d.h` / `point2d.cc`, a feature and the 3D point it observes;
//! - [`point3d`]: `point3d.h` / `point3d.cc`, a triangulated point with its track;
//! - [`track`]: `track.h` / `track.cc`, the observations of a 3D point;
//! - [`camera`]: `camera.h` / `camera.cc`, a camera model with its parameters;
//! - [`frame`]: `frame.h` / `frame.cc`, a posed instance of a rig with its measurements;
//! - [`image`]: `image.h` / `image.cc`, one exposure: its features, camera and frame;
//! - [`projection`]: `projection.h` / `projection.cc`, reprojection errors and cheirality;
//! - [`visibility_pyramid`]: `visibility_pyramid.h` / `.cc`, how evenly points cover an image;
//! - [`two_view_geometry`]: `two_view_geometry.h` / `.cc`, the verified geometry of an image
//!   pair.

pub mod camera;
pub mod frame;
pub mod image;
pub mod point2d;
pub mod point3d;
pub mod projection;
pub mod track;
pub mod two_view_geometry;
pub mod visibility_pyramid;
