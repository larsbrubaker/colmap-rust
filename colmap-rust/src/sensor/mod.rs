//! COLMAP's `sensor/` module. Tests: `colmap-rust/tests/sensor.rs`.
//! - [`models`]: the camera models of `colmap/sensor/models.h` / `models.cc`;
//! - [`rig`]: `rig.h` / `rig.cc`, rigidly mounted sensors ([`rig::Rig`]);
//! - [`bitmap`]: `bitmap.h` / `bitmap.cc` without file I/O (the host decodes images), plus
//!   `JetColormap`;
//! - [`exif_reader`]: the EXIF parser that stands in for OpenImageIO's EXIF decoding;
//! - [`specs`] and [`database`]: `specs.h` / `specs.cc` (the camera sensor-width table) and
//!   `database.h` / `database.cc` (its make/model lookup).

pub mod bitmap;
pub mod database;
pub mod exif_reader;
pub mod models;
pub mod rig;
pub mod specs;
