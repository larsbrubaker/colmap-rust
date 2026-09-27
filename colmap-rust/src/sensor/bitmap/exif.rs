//! The metadata half of `colmap/sensor/bitmap.cc`: `Set/GetMetaData`, `CloneMetadata`,
//! `SetJpegQuality` and the EXIF getters (orientation, camera model string, focal length in
//! pixels, GPS latitude/longitude/altitude). Part of [`super::Bitmap`]; port of colmap-sharp's
//! `ColmapSharp/Sensor/Bitmap.Exif.cs`.
//!
//! COLMAP keeps metadata in an OpenImageIO `ImageSpec`'s attribute list and names EXIF values
//! the way OIIO's EXIF decoder does ("Make", "Exif:FocalLength", "GPS:Latitude", ...).
//! [`BitmapMetaData`] is a stand-in for that list with the same names and the value types
//! COLMAP asks for: int, float, point (3 floats) and string. `sensor/exif_reader.rs` fills it
//! from a JPEG's EXIF block. COLMAP's type-string API (`SetMetaData(name, "float", &value)`)
//! becomes the [`MetaDataValue`] enum and typed getters. Type conversion on read follows the
//! subset of OIIO's `convert_type` the getters rely on: an int reads as float (EXIF SHORTs such
//! as FocalLengthIn35mmFilm are queried as float) and as its decimal string (GPS:AltitudeRef is
//! a BYTE queried as "0"/"1"); every other type mismatch reads as absent. Names are
//! case-insensitive, like OIIO's `getattribute` by default.
//!
//! Metadata access on a bitmap without a metadata store (default-constructed, or a copy of an
//! empty bitmap) dereferences a null pointer in COLMAP; here it reads as absent and writes
//! create the store (`docs/CPP_DIVERGENCES.md`, entry 124).
//!
//! Tier A: the EXIF getters are COLMAP's arithmetic verbatim.

use super::Bitmap;
use crate::math::fns;
use crate::sensor::database::CameraDatabase;
use crate::{check_gt, check_le, Result};

/// One metadata attribute value: the OIIO types COLMAP uses.
#[derive(Debug, Clone, PartialEq)]
pub enum MetaDataValue {
    /// OIIO `int`.
    Int(i32),
    /// OIIO `float`.
    Float(f32),
    /// OIIO `point`: three floats, e.g. a GPS coordinate as degrees, minutes and seconds.
    Point([f32; 3]),
    /// OIIO `string`.
    String(String),
}

/// Stand-in for the OIIO `ImageSpec` attribute list behind COLMAP's Bitmap metadata. Setting a
/// name replaces its previous value and type, as `ImageSpec::attribute` does. Kept in
/// insertion order (a `Vec`), so nothing depends on hash order.
#[derive(Debug, Clone, Default, PartialEq)]
pub(super) struct BitmapMetaData {
    values: Vec<(String, MetaDataValue)>,
}

impl BitmapMetaData {
    pub(super) fn set(&mut self, name: &str, value: MetaDataValue) {
        match self
            .values
            .iter_mut()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
        {
            Some(entry) => entry.1 = value,
            None => self.values.push((name.to_string(), value)),
        }
    }

    fn get(&self, name: &str) -> Option<&MetaDataValue> {
        self.values
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v)
    }
}

impl Bitmap {
    /// Port of `Bitmap::SetMetaData(name, type, value)` and `SetMetaData(name, string)`.
    pub fn set_meta_data(&mut self, name: &str, value: MetaDataValue) {
        self.meta_data.set(name, value);
    }

    /// `GetMetaData(name, "int", &value)`.
    pub fn get_meta_data_int(&self, name: &str) -> Option<i32> {
        match self.meta_data.get(name) {
            Some(MetaDataValue::Int(v)) => Some(*v),
            _ => None,
        }
    }

    /// `GetMetaData(name, "float", &value)`. An int attribute converts, as OIIO's
    /// `getattribute` does.
    pub fn get_meta_data_float(&self, name: &str) -> Option<f32> {
        match self.meta_data.get(name) {
            Some(MetaDataValue::Float(v)) => Some(*v),
            Some(MetaDataValue::Int(v)) => Some(*v as f32),
            _ => None,
        }
    }

    /// `GetMetaData(name, "point", &value)`: three floats.
    pub fn get_meta_data_point(&self, name: &str) -> Option<[f32; 3]> {
        match self.meta_data.get(name) {
            Some(MetaDataValue::Point(v)) => Some(*v),
            _ => None,
        }
    }

    /// `GetMetaData(name)`: the string value, or an int attribute in decimal (OIIO's string
    /// conversion); `None` if absent.
    pub fn get_meta_data(&self, name: &str) -> Option<String> {
        match self.meta_data.get(name) {
            Some(MetaDataValue::String(v)) => Some(v.clone()),
            Some(MetaDataValue::Int(v)) => Some(v.to_string()),
            _ => None,
        }
    }

    /// Port of `Bitmap::CloneMetadata`: copy this bitmap's metadata to the target.
    pub fn clone_metadata(&self, target: &mut Bitmap) {
        target.meta_data = self.meta_data.clone();
    }

    /// Port of `Bitmap::SetJpegQuality`: records the JPEG compression quality in [1, 100] as
    /// "Compression" metadata, for the host's encoder.
    pub fn set_jpeg_quality(&mut self, quality: i32) -> Result<()> {
        check_gt!(quality, 0);
        check_le!(quality, 100);
        self.set_meta_data(
            "Compression",
            MetaDataValue::String(format!("jpeg:{quality}")),
        );
        Ok(())
    }

    /// Port of `Bitmap::ExifOrientation`: the EXIF orientation tag (1..8), if present.
    pub fn exif_orientation(&self) -> Option<i32> {
        self.get_meta_data_int("Orientation")
    }

    /// Port of `Bitmap::ExifCameraModel`: `"make-model-focal-WxH"`, which groups images that
    /// can share intrinsics. `None` unless make, model and a focal length are present.
    pub fn exif_camera_model(&self) -> Option<String> {
        // Read camera make and model
        let make = self.get_meta_data("Make")?;
        let model = self.get_meta_data("Model")?;
        let focal_length = self
            .get_meta_data_float("Exif:FocalLengthIn35mmFilm")
            .or_else(|| self.get_meta_data_float("Exif:FocalLength"))?;
        // StringPrintf("%s-%s-%.6f-%dx%d", ...). Rust's {:.6} rounds the exact decimal
        // expansion like printf for finite values.
        Some(format!(
            "{make}-{model}-{:.6}-{}x{}",
            f64::from(focal_length),
            self.width,
            self.height
        ))
    }

    /// Port of `Bitmap::ExifFocalLength`: the focal length in pixels, from the 35 mm
    /// equivalent focal length, else from the focal length in mm and the focal-plane
    /// resolution, else from the focal length in mm and the camera database's sensor width.
    pub fn exif_focal_length(&self) -> Option<f64> {
        let max_size = f64::from(self.width.max(self.height));

        if let Some(focal_length_35mm) = self.get_meta_data_float("Exif:FocalLengthIn35mmFilm") {
            if focal_length_35mm > 0.0 {
                // Based on https://en.wikipedia.org/wiki/35_mm_equivalent_focal_length
                // According to CIPA guidelines, 35 mm equivalent focal length is to be
                // calculated like this:
                // "focal length in 35 mm camera" =
                //   (Diagonal distance of image area in the 35 mm camera (43.27 mm) /
                //    Diagonal distance of image area on the image sensor of the DSC)
                //    * focal length of the lens of the DSC.
                // C++ computes width_ * width_ + height_ * height_ in int.
                let diagonal = fns::sqrt(f64::from(
                    self.width
                        .wrapping_mul(self.width)
                        .wrapping_add(self.height.wrapping_mul(self.height)),
                ));
                return Some(f64::from(focal_length_35mm) / 43.27 * diagonal);
            }
        }

        if let Some(focal_length_mm) = self.get_meta_data_float("Exif:FocalLength") {
            if let (Some(focal_x_res), Some(focal_x_res_unit)) = (
                self.get_meta_data_float("Exif:FocalPlaneXResolution"),
                self.get_meta_data_int("Exif:FocalPlaneResolutionUnit"),
            ) {
                if focal_length_mm > 0.0 && focal_x_res_unit > 1 && focal_x_res_unit <= 5 {
                    let focal_x_res = f64::from(focal_x_res);
                    let pixels_per_mm = match focal_x_res_unit {
                        2 => focal_x_res / 25.4,   // inches
                        3 => focal_x_res / 10.0,   // cm
                        4 => focal_x_res * 1.0,    // mm
                        _ => focal_x_res * 1000.0, // um (5; the range is checked above)
                    };
                    return Some(f64::from(focal_length_mm) * pixels_per_mm);
                }
            }

            // Lookup sensor width in database.
            if let (Some(make), Some(model)) =
                (self.get_meta_data("Make"), self.get_meta_data("Model"))
            {
                let database = CameraDatabase::new();
                let mut sensor_width_mm = 0.0;
                if database.query_sensor_width(&make, &model, &mut sensor_width_mm) {
                    return Some(f64::from(focal_length_mm) / sensor_width_mm * max_size);
                }
            }
        }

        None
    }

    /// Port of `Bitmap::ExifLatitude`: decimal degrees, negative for "S".
    pub fn exif_latitude(&self) -> Option<f64> {
        let sign = match self.get_meta_data("GPS:LatitudeRef").as_deref() {
            Some("S") | Some("s") => -1.0,
            _ => 1.0,
        };
        self.gps_degrees("GPS:Latitude", sign)
    }

    /// Port of `Bitmap::ExifLongitude`: decimal degrees, negative for "W".
    pub fn exif_longitude(&self) -> Option<f64> {
        let sign = match self.get_meta_data("GPS:LongitudeRef").as_deref() {
            Some("W") | Some("w") => -1.0,
            _ => 1.0,
        };
        self.gps_degrees("GPS:Longitude", sign)
    }

    /// The shared tail of `ExifLatitude` / `ExifLongitude`.
    fn gps_degrees(&self, name: &str, sign: f64) -> Option<f64> {
        let deg_min_sec = self.get_meta_data_point(name)?;
        let mut degrees = f64::from(deg_min_sec[0])
            + f64::from(deg_min_sec[1]) / 60.0
            + f64::from(deg_min_sec[2]) / 3600.0;
        if degrees > 0.0 && sign < 0.0 {
            degrees *= sign;
        }
        Some(degrees)
    }

    /// Port of `Bitmap::ExifAltitude`: meters, negative below sea level (ref "1").
    pub fn exif_altitude(&self) -> Option<f64> {
        let sign = match self.get_meta_data("GPS:AltitudeRef").as_deref() {
            Some("1") => -1.0,
            _ => 1.0,
        };
        let mut altitude = f64::from(self.get_meta_data_float("GPS:Altitude")?);
        if altitude > 0.0 && sign < 0.0 {
            altitude *= sign;
        }
        Some(altitude)
    }
}
