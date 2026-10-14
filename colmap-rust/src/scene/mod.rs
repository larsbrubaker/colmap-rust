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
//! - [`correspondence_graph`]: `correspondence_graph.h` / `.cc`, the feature correspondences
//!   between all images;
//! - [`database`]: `database.h` / `database.cc`, the in-memory store of rigs, cameras,
//!   frames, images, pose priors, features, matches and two-view geometries, with its queries;
//!   [`database_write`] its writes, updates, deletes and clears, and [`database_merge`]
//!   `Merge`, `LoadRandomDatabaseDescriptors` and the keypoint/match blobs;
//! - [`database_cache`]: `database_cache.h` / `.cc`, what a mapper loads from a database;
//! - [`frame`]: `frame.h` / `frame.cc`, a posed instance of a rig with its measurements;
//! - [`image`]: `image.h` / `image.cc`, one exposure: its features, camera and frame;
//! - [`reconstruction`]: `reconstruction.h` / `.cc`, the owner of the rigs, cameras, frames,
//!   images and 3D points of a model, with [`reconstruction_queries`] its whole-model queries
//!   and [`reconstruction_transform`] its bounds, normalization, transform and crop,
//!   [`reconstruction_files`] its model-folder read and write, and [`reconstruction_database`]
//!   `Load` from a database cache and `TranscribeImageIdsToDatabase`;
//! - [`reconstruction_io_binary`], [`reconstruction_io_text`] (readers),
//!   [`reconstruction_io_text_write`] (writers) and [`reconstruction_io_utils`]:
//!   `reconstruction_io_binary.*`, `reconstruction_io_text.*` and `reconstruction_io_utils.*`,
//!   COLMAP's binary and text model files;
//! - [`projection`]: `projection.h` / `projection.cc`, reprojection errors and cheirality;
//! - [`visibility_pyramid`]: `visibility_pyramid.h` / `.cc`, how evenly points cover an image;
//! - [`synthetic`]: `synthetic.h` / `.cc`, synthetic datasets for tests, with
//!   [`synthetic_matches`] the matches it writes to a database and [`synthetic_noise`]
//!   `SynthesizeNoise` and `SynthesizeImages`;
//! - [`two_view_geometry`]: `two_view_geometry.h` / `.cc`, the verified geometry of an image
//!   pair.

pub mod camera;
pub mod correspondence_graph;
pub mod database;
pub mod database_cache;
pub mod database_merge;
pub mod database_write;
pub mod frame;
pub mod image;
pub mod point2d;
pub mod point3d;
pub mod projection;
pub mod reconstruction;
pub mod reconstruction_database;
pub mod reconstruction_files;
pub mod reconstruction_io_binary;
pub mod reconstruction_io_text;
pub mod reconstruction_io_text_write;
pub mod reconstruction_io_utils;
pub mod reconstruction_queries;
pub mod reconstruction_transform;
pub mod synthetic;
pub mod synthetic_matches;
pub mod synthetic_noise;
pub mod track;
pub mod two_view_geometry;
pub mod visibility_pyramid;
