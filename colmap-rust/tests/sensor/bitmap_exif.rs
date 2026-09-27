// Port of colmap/sensor/bitmap_test.cc, second half: the Bitmap cases from Rot90 through the
// EXIF getters (same test names, values and checks). See bitmap.rs for the first half and the
// cases that are not ported.
//
// COLMAP's metadata API takes an OIIO type name ("int", "float", "point"); here the value is
// a MetaDataValue and the getters are typed. SetGetMetaData's probes with type "int8" (a type
// the value does not have) become probes with get_meta_data_int, which is likewise not the
// stored type.

use colmap_rust::sensor::bitmap::{Bitmap, BitmapColor, MetaDataValue};

#[test]
fn bitmap_rot90() {
    let mut bitmap = Bitmap::new(10, 5, false);
    bitmap.set_pixel(0, 0, BitmapColor::gray(255));
    bitmap.set_pixel(9, 0, BitmapColor::gray(128));
    bitmap.set_pixel(9, 4, BitmapColor::gray(64));

    let mut rotated1 = bitmap.clone();
    rotated1.rot90(1); // 90 CCW
    assert_eq!(rotated1.width(), 5);
    assert_eq!(rotated1.height(), 10);
    // Top-left (0,0) -> Bottom-left (0,9)
    assert_eq!(rotated1.get_pixel(0, 9).unwrap().r, 255);
    // Top-right (9,0) -> Top-left (0,0)
    assert_eq!(rotated1.get_pixel(0, 0).unwrap().r, 128);
    // Bottom-right (9,4) -> Top-right (4,0)
    assert_eq!(rotated1.get_pixel(4, 0).unwrap().r, 64);

    let mut rotated2 = bitmap.clone();
    rotated2.rot90(2); // 180 CCW
    assert_eq!(rotated2.width(), 10);
    assert_eq!(rotated2.height(), 5);
    assert_eq!(rotated2.get_pixel(9, 4).unwrap().r, 255);

    let mut rotated3 = bitmap.clone();
    rotated3.rot90(3); // 270 CCW
    assert_eq!(rotated3.width(), 5);
    assert_eq!(rotated3.height(), 10);
    // Top-left (0,0) -> Top-right (4,0)
    assert_eq!(rotated3.get_pixel(4, 0).unwrap().r, 255);
}

#[test]
fn bitmap_rot90_empty() {
    let mut bitmap = Bitmap::empty();
    bitmap.rot90(1);
    assert!(bitmap.is_empty());
    assert_eq!(bitmap.width(), 0);
    assert_eq!(bitmap.height(), 0);
}

#[test]
fn bitmap_rot90_negative_k() {
    let mut bitmap = Bitmap::new(10, 5, false);
    bitmap.fill(BitmapColor::gray(0)).unwrap();
    bitmap.set_pixel(0, 0, BitmapColor::gray(255));
    bitmap.set_pixel(9, 0, BitmapColor::gray(128));

    // Rot90(-1) should be equivalent to Rot90(3) (270 CCW = 90 CW).
    let mut rotated_neg = bitmap.clone();
    rotated_neg.rot90(-1);

    let mut rotated_3 = bitmap.clone();
    rotated_3.rot90(3);

    assert_eq!(rotated_neg.width(), rotated_3.width());
    assert_eq!(rotated_neg.height(), rotated_3.height());
    assert_eq!(rotated_neg.row_major_data(), rotated_3.row_major_data());
}

#[test]
fn bitmap_rot90_no_op() {
    let mut bitmap = Bitmap::new(10, 5, false);
    bitmap.fill(BitmapColor::gray(0)).unwrap();
    bitmap.set_pixel(0, 0, BitmapColor::gray(255));
    let original_data = bitmap.row_major_data().to_vec();

    let mut rotated0 = bitmap.clone();
    rotated0.rot90(0);
    assert_eq!(rotated0.width(), 10);
    assert_eq!(rotated0.height(), 5);
    assert_eq!(rotated0.row_major_data(), original_data.as_slice());

    let mut rotated4 = bitmap.clone();
    rotated4.rot90(4);
    assert_eq!(rotated4.width(), 10);
    assert_eq!(rotated4.height(), 5);
    assert_eq!(rotated4.row_major_data(), original_data.as_slice());
}

#[test]
fn bitmap_clone() {
    let mut bitmap = Bitmap::new(100, 80, true);
    bitmap.fill(BitmapColor::new(0, 0, 0)).unwrap();
    bitmap.set_pixel(0, 0, BitmapColor::new(10, 20, 30));
    let cloned_bitmap = bitmap.clone();
    assert_eq!(cloned_bitmap.width(), 100);
    assert_eq!(cloned_bitmap.height(), 80);
    assert_eq!(cloned_bitmap.channels(), 3);
    assert_eq!(
        cloned_bitmap.get_pixel(0, 0),
        Some(BitmapColor::new(10, 20, 30))
    );
}

#[test]
fn bitmap_clone_as_rgb() {
    let mut bitmap = Bitmap::new(100, 80, false);
    bitmap.fill(BitmapColor::new(0, 0, 0)).unwrap();
    bitmap.set_pixel(0, 0, BitmapColor::new(10, 0, 0));
    let cloned_bitmap = bitmap.clone_as_rgb().unwrap();
    assert_eq!(cloned_bitmap.width(), 100);
    assert_eq!(cloned_bitmap.height(), 80);
    assert_eq!(cloned_bitmap.channels(), 3);
    assert_eq!(
        cloned_bitmap.get_pixel(0, 0),
        Some(BitmapColor::new(10, 10, 10))
    );
    // The PNG Write/Read round trip that follows in COLMAP is not ported (file I/O).
}

#[test]
fn bitmap_clone_as_rgb_from_rgb() {
    let mut bitmap = Bitmap::new(4, 3, true);
    bitmap.fill(BitmapColor::new(0, 0, 0)).unwrap();
    bitmap.set_pixel(0, 0, BitmapColor::new(10, 20, 30));
    bitmap.set_pixel(1, 1, BitmapColor::new(40, 50, 60));

    let cloned = bitmap.clone_as_rgb().unwrap();
    assert_eq!(cloned.width(), 4);
    assert_eq!(cloned.height(), 3);
    assert_eq!(cloned.channels(), 3);
    assert!(cloned.is_rgb());
    assert_eq!(cloned.row_major_data(), bitmap.row_major_data());
}

#[test]
fn bitmap_clone_as_grey() {
    let mut bitmap = Bitmap::new(100, 80, true);
    bitmap.fill(BitmapColor::new(0, 0, 0)).unwrap();
    bitmap.set_pixel(0, 0, BitmapColor::new(10, 20, 30));
    bitmap.set_pixel(1, 0, BitmapColor::new(1, 0, 4));
    bitmap.set_pixel(2, 0, BitmapColor::new(0, 0, 6));
    bitmap.set_pixel(3, 0, BitmapColor::new(255, 255, 255));
    let cloned_bitmap = bitmap.clone_as_grey();
    assert_eq!(cloned_bitmap.width(), 100);
    assert_eq!(cloned_bitmap.height(), 80);
    assert_eq!(cloned_bitmap.channels(), 1);
    assert_eq!(
        cloned_bitmap.get_pixel(0, 0),
        Some(BitmapColor::new(19, 19, 19))
    );
    assert_eq!(
        cloned_bitmap.get_pixel(1, 0),
        Some(BitmapColor::new(1, 1, 1))
    );
    assert_eq!(
        cloned_bitmap.get_pixel(2, 0),
        Some(BitmapColor::new(0, 0, 0))
    );
    assert_eq!(
        cloned_bitmap.get_pixel(3, 0),
        Some(BitmapColor::new(255, 255, 255))
    );
    // The PNG Write/Read round trip that follows in COLMAP is not ported (file I/O).
}

#[test]
fn bitmap_clone_as_grey_from_grey() {
    let mut bitmap = Bitmap::new(4, 3, false);
    bitmap.fill(BitmapColor::gray(0)).unwrap();
    bitmap.set_pixel(0, 0, BitmapColor::gray(42));
    bitmap.set_pixel(1, 1, BitmapColor::gray(100));

    let cloned = bitmap.clone_as_grey();
    assert_eq!(cloned.width(), 4);
    assert_eq!(cloned.height(), 3);
    assert_eq!(cloned.channels(), 1);
    assert!(cloned.is_grey());
    assert_eq!(cloned.row_major_data(), bitmap.row_major_data());
}

#[test]
fn bitmap_set_get_meta_data() {
    let mut bitmap = Bitmap::new(100, 80, true);
    let k_value = 1.0f32;
    bitmap.set_meta_data("foobar", MetaDataValue::Float(k_value));
    assert_eq!(bitmap.get_meta_data_float("foobar"), Some(k_value));
    assert_eq!(bitmap.get_meta_data_float("does_not_exist"), None);
    assert_eq!(bitmap.get_meta_data_int("foobar"), None);
    bitmap.set_meta_data("foobar_str", MetaDataValue::String("string".to_string()));
    assert_eq!(bitmap.get_meta_data("foobar_str").unwrap(), "string");
    assert_eq!(bitmap.get_meta_data_int("foobar_str"), None);
    assert_eq!(bitmap.get_meta_data_float("foobar_str"), None);
    assert!(bitmap.get_meta_data("does_not_exist").is_none());
}

#[test]
fn bitmap_clone_meta_data() {
    let mut bitmap = Bitmap::new(100, 80, true);
    bitmap.set_meta_data("foobar", MetaDataValue::Float(1.0));

    let mut bitmap2 = Bitmap::new(100, 80, true);
    assert_eq!(bitmap2.get_meta_data_float("foobar"), None);
    bitmap.clone_metadata(&mut bitmap2);
    assert_eq!(bitmap2.get_meta_data_float("foobar"), Some(1.0));
}

#[test]
fn bitmap_exif_orientation() {
    let mut bitmap = Bitmap::new(100, 80, true);
    assert!(bitmap.exif_orientation().is_none());
    bitmap.set_meta_data("Orientation", MetaDataValue::Int(6));
    assert_eq!(bitmap.exif_orientation(), Some(6));
}

fn set_string(bitmap: &mut Bitmap, name: &str, value: &str) {
    bitmap.set_meta_data(name, MetaDataValue::String(value.to_string()));
}

#[test]
fn bitmap_exif_camera_model() {
    let mut bitmap = Bitmap::new(100, 80, true);
    assert!(bitmap.exif_camera_model().is_none());
    set_string(&mut bitmap, "Make", "make");
    set_string(&mut bitmap, "Model", "model");
    bitmap.set_meta_data("Exif:FocalLengthIn35mmFilm", MetaDataValue::Float(50.0));
    assert_eq!(
        bitmap.exif_camera_model().as_deref(),
        Some("make-model-50.000000-100x80")
    );
}

#[test]
fn bitmap_exif_camera_model_no_model() {
    let mut bitmap = Bitmap::new(100, 80, true);
    set_string(&mut bitmap, "Make", "make");
    // Do not set Model.
    bitmap.set_meta_data("Exif:FocalLengthIn35mmFilm", MetaDataValue::Float(50.0));
    assert!(bitmap.exif_camera_model().is_none());
}

#[test]
fn bitmap_exif_camera_model_no_focal_length() {
    let mut bitmap = Bitmap::new(100, 80, true);
    set_string(&mut bitmap, "Make", "make");
    set_string(&mut bitmap, "Model", "model");
    // Do not set any focal length metadata.
    assert!(bitmap.exif_camera_model().is_none());
}

#[test]
fn bitmap_exif_focal_length_in35mm() {
    let mut bitmap = Bitmap::new(100, 80, true);
    assert!(bitmap.exif_focal_length().is_none());
    bitmap.set_meta_data("Exif:FocalLengthIn35mmFilm", MetaDataValue::Float(70.0));
    let focal_length = bitmap.exif_focal_length().unwrap();
    assert!((focal_length - 207.17).abs() <= 0.1);
}

#[test]
fn bitmap_exif_focal_length_with_plane() {
    let mut bitmap = Bitmap::new(100, 80, true);
    assert!(bitmap.exif_focal_length().is_none());
    bitmap.set_meta_data("Exif:FocalLength", MetaDataValue::Float(72.0));
    set_string(&mut bitmap, "Make", "canon");
    set_string(&mut bitmap, "Model", "eos1dsmarkiii");
    assert_eq!(bitmap.exif_focal_length(), Some(200.0));
}

#[test]
fn bitmap_exif_focal_length_with_database_lookup() {
    let mut bitmap = Bitmap::new(100, 80, true);
    assert!(bitmap.exif_focal_length().is_none());
    bitmap.set_meta_data("Exif:FocalLength", MetaDataValue::Float(120.0));
    bitmap.set_meta_data("Exif:PixelXDimension", MetaDataValue::Int(100));
    bitmap.set_meta_data("Exif:FocalPlaneXResolution", MetaDataValue::Float(1.0));
    bitmap.set_meta_data("Exif:FocalPlaneResolutionUnit", MetaDataValue::Int(4));
    assert_eq!(bitmap.exif_focal_length(), Some(120.0));
}

#[test]
fn bitmap_exif_focal_length_units() {
    // Initialize a dummy bitmap
    let mut bitmap = Bitmap::new(100, 80, true);

    // Set the base focal length and resolution values
    bitmap.set_meta_data("Exif:FocalLength", MetaDataValue::Float(50.0));
    bitmap.set_meta_data("Exif:FocalPlaneXResolution", MetaDataValue::Float(100.0));

    let mut check_unit = |unit: i32, expected: f64| {
        bitmap.set_meta_data("Exif:FocalPlaneResolutionUnit", MetaDataValue::Int(unit));
        let focal_length = bitmap.exif_focal_length().unwrap();
        assert!((focal_length - expected).abs() <= 1e-4, "unit {unit}");
    };
    // Case 2: Inches (25.4 mm per inch)
    check_unit(2, 50.0 * (100.0 / 25.4));
    // Case 3: Centimeters (10 mm per cm)
    check_unit(3, 50.0 * (100.0 / 10.0));
    // Case 4: Millimeters (1 mm per mm)
    check_unit(4, 50.0 * (100.0 * 1.0));
    // Case 5: Micrometers (1000 um per mm)
    check_unit(5, 50.0 * (100.0 * 1000.0));
}

#[test]
fn bitmap_exif_latitude() {
    let mut bitmap = Bitmap::new(100, 80, true);
    assert!(bitmap.exif_latitude().is_none());

    set_string(&mut bitmap, "GPS:LatitudeRef", "N");
    bitmap.set_meta_data("GPS:Latitude", MetaDataValue::Point([46.0, 30.0, 900.0]));
    assert_eq!(bitmap.exif_latitude(), Some(46.75));

    set_string(&mut bitmap, "GPS:LatitudeRef", "S");
    assert_eq!(bitmap.exif_latitude(), Some(-46.75));
}

#[test]
fn bitmap_exif_longitude() {
    let mut bitmap = Bitmap::new(100, 80, true);
    assert!(bitmap.exif_longitude().is_none());

    set_string(&mut bitmap, "GPS:LongitudeRef", "W");
    bitmap.set_meta_data("GPS:Longitude", MetaDataValue::Point([92.0, 30.0, 900.0]));
    assert_eq!(bitmap.exif_longitude(), Some(-92.75));

    set_string(&mut bitmap, "GPS:LongitudeRef", "E");
    assert_eq!(bitmap.exif_longitude(), Some(92.75));
}

#[test]
fn bitmap_exif_altitude() {
    let mut bitmap = Bitmap::new(100, 80, true);
    assert!(bitmap.exif_altitude().is_none());

    set_string(&mut bitmap, "GPS:AltitudeRef", "0");
    let k_altitude_val = 123.456f32;
    bitmap.set_meta_data("GPS:Altitude", MetaDataValue::Float(k_altitude_val));
    assert_eq!(bitmap.exif_altitude(), Some(f64::from(k_altitude_val)));

    set_string(&mut bitmap, "GPS:AltitudeRef", "1");
    assert_eq!(bitmap.exif_altitude(), Some(-f64::from(k_altitude_val)));
}
