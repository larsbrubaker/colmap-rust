//! COLMAP's `scene/` module. Tests: `colmap-rust/tests/scene.rs`.
//! - [`point2d`]: `point2d.h` / `point2d.cc`, a feature and the 3D point it observes;
//! - [`point3d`]: `point3d.h` / `point3d.cc`, a triangulated point with its track;
//! - [`track`]: `track.h` / `track.cc`, the observations of a 3D point;
//! - [`camera`]: `camera.h` / `camera.cc`, a camera model with its parameters;
//! - [`projection`]: `projection.h` / `projection.cc`, reprojection errors and cheirality;
//! - [`visibility_pyramid`]: `visibility_pyramid.h` / `.cc`, how evenly points cover an image.

pub mod camera;
pub mod point2d;
pub mod point3d;
pub mod projection;
pub mod track;
pub mod visibility_pyramid;
