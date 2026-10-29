//! Port of `colmap/sensor/database.h` / `database.cc`: [`CameraDatabase`], the fuzzy
//! make/model lookup over the sensor-width table of [`super::specs`].
//! [`super::bitmap::Bitmap::exif_focal_length`] falls back to it when the EXIF has a focal
//! length in mm but no focal-plane resolution. Port of colmap-sharp's
//! `ColmapSharp/Sensor/CameraDatabase.cs`. Tests: `tests/sensor/database.rs`
//! (`database_test.cc` 1:1).
//!
//! Tier A: same strings in, same answer out, except when the cleaned make matches more than
//! one make of the table; then COLMAP's answer depends on its hash map's iteration order and
//! ours on `specs.cc` source order.
//!
//! That is a deliberate divergence: COLMAP's `camera_specs_t` is a `NodeHashMap`, and
//! `QuerySensorWidth` iterates it, writing the output width on every match and stopping after
//! the second non-exact match per make. [`initialize_camera_specs`] returns the makes in
//! `specs.cc` source order, so when a cleaned EXIF make matches more than one table make (a
//! substring match either way round; an empty make matches all of them), which widths are
//! seen, and so the width left behind and whether a unique match is found, can differ from a
//! given COLMAP build. Hash iteration order is unspecified and differs between standard
//! libraries and Boost, so there is no single COLMAP behavior to match, and CLAUDE.md asks for
//! deterministic order. `database_test.cc`'s cases match a single make and pass; queries whose
//! make matches one table make are unaffected. Same as colmap-sharp entry 8.

use std::sync::OnceLock;

use super::specs::{initialize_camera_specs, CameraMakeSpecs};
use crate::util::string::{string_contains, string_replace};

/// COLMAP's `static const camera_specs_t specs_`, built once on first use.
fn specs() -> &'static [CameraMakeSpecs] {
    static SPECS: OnceLock<Vec<CameraMakeSpecs>> = OnceLock::new();
    SPECS.get_or_init(initialize_camera_specs)
}

/// Port of `colmap::CameraDatabase`: sensor widths for many cameras, which is useful to
/// automatically extract the focal length if EXIF information is incomplete.
#[derive(Debug, Default, Clone, Copy)]
pub struct CameraDatabase;

impl CameraDatabase {
    /// `CameraDatabase()`.
    pub fn new() -> Self {
        CameraDatabase
    }

    /// `NumEntries`: the number of camera makes in the database.
    pub fn num_entries(&self) -> usize {
        specs().len()
    }

    /// Port of `CameraDatabase::QuerySensorWidth`. Returns true when the model matches
    /// exactly, or exactly one model matches as a substring (either way round).
    /// `sensor_width_mm` is written on every match, as in COLMAP, so after an ambiguous query
    /// it holds a width even though the result is false.
    pub fn query_sensor_width(&self, make: &str, model: &str, sensor_width_mm: &mut f64) -> bool {
        // Clean the strings from all separators. StringToLower is ::tolower in the C locale,
        // which changes only ASCII A-Z.
        let cleaned_make =
            string_replace(&string_replace(make, " ", ""), "-", "").to_ascii_lowercase();
        let cleaned_model =
            string_replace(&string_replace(model, " ", ""), "-", "").to_ascii_lowercase();

        // Make sure that make name is not duplicated.
        let cleaned_model = string_replace(&cleaned_model, &cleaned_make, "");

        // Check if cleaned_make exists in database: Test whether EXIF string is substring of
        // database entry and vice versa.
        let mut spec_matches = 0usize;
        for make_specs in specs() {
            let make_name = make_specs.make;
            if string_contains(&cleaned_make, make_name)
                || string_contains(make_name, &cleaned_make)
            {
                for &(model_name, sensor_width) in make_specs.models {
                    if string_contains(&cleaned_model, model_name)
                        || string_contains(model_name, &cleaned_model)
                    {
                        *sensor_width_mm = f64::from(sensor_width);
                        if cleaned_model == model_name {
                            // Model exactly matches, return immediately.
                            return true;
                        }
                        spec_matches += 1;
                        if spec_matches > 1 {
                            break;
                        }
                    }
                }
            }
        }

        // Only return unique results, if model does not exactly match.
        spec_matches == 1
    }
}
