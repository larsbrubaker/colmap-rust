//! Port of `colmap/sensor/specs.h` / `specs.cc`: the database of camera sensor widths (mm)
//! keyed by make and model. [`super::database::CameraDatabase`] queries it to turn an EXIF
//! focal length in mm into pixels when the EXIF has no focal-plane resolution
//! ([`super::bitmap::Bitmap::exif_focal_length`]).
//!
//! The table itself is GENERATED from the C++ reference by `scripts/generate-camera-specs.py`
//! into `data1.rs` .. `data9.rs` (split so each file stays under the 800-line limit); this
//! file holds the types and the entry point. COLMAP has no `specs_test.cc`; the table is tested
//! through `database_test.cc` (`tests/sensor/database.rs`).
//!
//! Translation note: COLMAP's `camera_specs_t` is a `NodeHashMap` (hash-ordered). Here it is
//! a static slice in `specs.cc` source order, so the query iterates deterministically
//! (`docs/CPP_DIVERGENCES.md`, entry 120).

mod data1;
mod data2;
mod data3;
mod data4;
mod data5;
mod data6;
mod data7;
mod data8;
mod data9;

/// One key/value pair of `camera_specs_t`: a make and its models' sensor widths in mm
/// (`camera_make_specs_t`), in `specs.cc` order. Makes and models are lower case without
/// separators.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraMakeSpecs {
    /// Camera make, e.g. "canon".
    pub make: &'static str,
    /// `(model, sensor width in mm)` pairs.
    pub models: &'static [(&'static str, f32)],
}

/// Port of `colmap::InitializeCameraSpecs`: every make with its models' sensor widths, in the
/// order `specs.cc` lists them. Makes are unique.
pub fn initialize_camera_specs() -> Vec<CameraMakeSpecs> {
    [
        data1::MAKES,
        data2::MAKES,
        data3::MAKES,
        data4::MAKES,
        data5::MAKES,
        data6::MAKES,
        data7::MAKES,
        data8::MAKES,
        data9::MAKES,
    ]
    .concat()
}
