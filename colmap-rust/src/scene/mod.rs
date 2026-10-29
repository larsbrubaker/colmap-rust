//! COLMAP's `scene/` module. Tests: `colmap-rust/tests/scene.rs`.
//!
//! Ownership: where COLMAP's scene types hold raw pointers to shared objects (`Frame` to its
//! `Rig`, `Image` to its `Camera` and `Frame`), the Rust types store only the ids, and the
//! owner of the maps (the reconstruction) passes the referenced objects to the methods that
//! need them, e.g. `frame.sensor_from_world(&rig, sensor_id)` or
//! `image.cam_from_world(&frame, &rig)`.
//!
//! That is a deliberate divergence. COLMAP's reconstruction sets the pointers (`SetRigPtr`,
//! `SetCameraPtr`, `SetFramePtr`) and clears them (`Reset*Ptr`), and methods such as
//! `Frame::SensorFromWorld`, `Image::CamFromWorld` and `Image::ProjectPoint` follow them. Here
//! each owner-taking method fails unless the frame / camera has the image's id and the rig has
//! the frame's rig id, where COLMAP fails on a null pointer; `Set*Ptr` become `verify_*`
//! methods with the same checks and nothing stored, and rebinding (COLMAP's `Set*Ptr` on a
//! bound object, which adopts the new id) is setting the id, then verifying. The `frame` and
//! `image` headers list each method's counterpart. `RigPtr`, `CameraPtr`, `FramePtr`,
//! `Has*Ptr` and `Reset*Ptr` have none, and `Frame::set_rig_id` / `Image::set_camera_id` /
//! `Image::set_frame_id` drop COLMAP's `THROW_CHECK(!Has*Ptr())`: there is no binding to
//! protect, and a frame or image inside a reconstruction is only reached through it (what the
//! reconstruction checks instead is in the `reconstruction` header).
//!
//! Why: CLAUDE.md's translation rule makes pointer graphs ids into owning maps. Shared mutable
//! pointers between the reconstruction's maps cannot be expressed in safe Rust without
//! `Rc<RefCell<..>>` everywhere, which would make every pose read in bundle adjustment and the
//! mapper a runtime borrow check. Passing the owner keeps frames and images plain data that
//! clone, move between threads and serialize without fix-ups (COLMAP's reconstruction
//! re-links every pointer after a copy; with ids there is nothing to re-link).
//! `tests/scene/frame.rs` and `tests/scene/image.rs` port `frame_test.cc` and
//! `image_test.cc` 1:1, with the pointer tests (`SetResetRigPtr`, `CameraPtr`, `FramePtr`,
//! `ConstructCopy`, `AssignCopy`) as behaviour tests of the id-based API under the C++ names;
//! `tests/scene/rust_only_frame_image.rs` checks that every owner-taking method rejects the
//! wrong rig, frame or camera, and the two equality forms.
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
//!   `Load` from a database cache and `TranscribeImageIdsToDatabase`, and
//!   [`reconstruction_ply`] its PLY import/export, color extraction and `CreateImageDirs`;
//! - [`reconstruction_io_binary`], [`reconstruction_io_text`] (readers),
//!   [`reconstruction_io_text_write`] (writers) and [`reconstruction_io_utils`]:
//!   `reconstruction_io_binary.*`, `reconstruction_io_text.*` and `reconstruction_io_utils.*`,
//!   COLMAP's binary and text model files, and [`reconstruction_export`] its exporters;
//! - [`pose_graph`]: `pose_graph.h` / `.cc`, relative poses between image pairs and their
//!   connected frame components;
//! - [`reconstruction_manager`]: `reconstruction_manager.h` / `.cc`, the models a mapper
//!   produces, read from and written to model folders;
//! - [`reconstruction_pruning`]: `reconstruction_pruning.h` / `.cc`, the 3D points that add
//!   no image coverage;
//! - [`projection`]: `projection.h` / `projection.cc`, reprojection errors and cheirality;
//! - [`rig`]: `rig.h` / `rig.cc`, rig configurations (`RigConfig`, `ReadRigConfig`), with
//!   [`rig_apply`] `ApplyRigConfig`, which writes them into a database and reconstruction;
//! - [`visibility_pyramid`]: `visibility_pyramid.h` / `.cc`, how evenly points cover an image;
//! - [`scene_clustering`]: `scene_clustering.h` / `.cc`, the scene graph split into
//!   overlapping image clusters by normalized graph cuts;
//! - [`reconstruction_clustering`]: `reconstruction_clustering.h` / `.cc`, the frames of a
//!   model grouped by covisibility;
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
pub mod pose_graph;
pub mod projection;
pub mod reconstruction;
pub mod reconstruction_clustering;
pub mod reconstruction_database;
pub mod reconstruction_export;
mod reconstruction_export_recon3d_vrml;
pub mod reconstruction_files;
pub mod reconstruction_io_binary;
pub mod reconstruction_io_text;
pub mod reconstruction_io_text_write;
pub mod reconstruction_io_utils;
pub mod reconstruction_manager;
pub mod reconstruction_ply;
pub mod reconstruction_pruning;
pub mod reconstruction_queries;
pub mod reconstruction_transform;
pub mod rig;
pub mod rig_apply;
pub mod scene_clustering;
pub mod synthetic;
pub mod synthetic_matches;
pub mod synthetic_noise;
pub mod track;
pub mod two_view_geometry;
pub mod visibility_pyramid;
