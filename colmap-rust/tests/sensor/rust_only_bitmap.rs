// Rust-only tests of src/sensor/bitmap/ beyond bitmap_test.cc (ports of colmap-sharp's C#-only
// Bitmap cases plus the behaviors this port pins on its own):
// - interpolation of points far outside the image or NaN (docs/CPP_DIVERGENCES.md, entry 123);
// - case-insensitive metadata names (OIIO's getattribute default) and the metadata store of
//   a copied empty bitmap (entry 124);
// - BitmapColor::cast's NaN and float-limit behavior (BitmapColorCast verbatim);
// - JetColormap, which bitmap_test.cc does not cover;
// - try_new's rejection of invalid dimensions, and printf's non-finite spelling in
//   exif_camera_model.

use colmap_rust::sensor::bitmap::{Bitmap, BitmapColor, JetColormap, MetaDataValue};

#[test]
fn rust_only_interpolate_far_outside_or_nan_returns_none() {
    for as_rgb in [true, false] {
        let mut bitmap = Bitmap::new(11, 10, as_rgb);
        bitmap.fill(BitmapColor::new(1, 2, 3)).unwrap();
        let bad = [
            3e9,
            -3e9,
            f64::from(i32::MAX),
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ];
        for v in bad {
            assert!(bitmap.interpolate_bilinear(v, 5.0).is_none(), "{v}");
            assert!(bitmap.interpolate_bilinear(5.0, v).is_none(), "{v}");
            assert!(bitmap.interpolate_nearest_neighbor(v, 5.0).is_none(), "{v}");
            assert!(bitmap.interpolate_nearest_neighbor(5.0, v).is_none(), "{v}");
        }
    }
}

#[test]
fn rust_only_meta_data_names_are_case_insensitive() {
    let mut bitmap = Bitmap::new(10, 8, true);
    bitmap.set_meta_data("Exif:FocalLength", MetaDataValue::Float(24.0));
    assert_eq!(bitmap.get_meta_data_float("exif:focallength"), Some(24.0));
    bitmap.set_meta_data("EXIF:FOCALLENGTH", MetaDataValue::Float(35.0));
    assert_eq!(bitmap.get_meta_data_float("Exif:FocalLength"), Some(35.0));
    bitmap.set_meta_data("make", MetaDataValue::String("Canon".to_string()));
    assert_eq!(bitmap.get_meta_data("Make").as_deref(), Some("Canon"));
}

#[test]
fn rust_only_meta_data_type_conversions() {
    let mut bitmap = Bitmap::new(10, 8, true);
    // The constructor records the colorspace, as COLMAP's does.
    assert_eq!(
        bitmap.get_meta_data("oiio:ColorSpace").as_deref(),
        Some("sRGB")
    );
    let linear = Bitmap::with_colorspace(2, 2, false, true);
    assert_eq!(
        linear.get_meta_data("oiio:ColorSpace").as_deref(),
        Some("linear")
    );
    // An int reads as float and as its decimal string; nothing else converts.
    bitmap.set_meta_data("GPS:AltitudeRef", MetaDataValue::Int(1));
    assert_eq!(bitmap.get_meta_data_float("GPS:AltitudeRef"), Some(1.0));
    assert_eq!(
        bitmap.get_meta_data("GPS:AltitudeRef").as_deref(),
        Some("1")
    );
    bitmap.set_meta_data("p", MetaDataValue::Point([1.0, 2.0, 3.0]));
    assert_eq!(bitmap.get_meta_data_float("p"), None);
    assert_eq!(bitmap.get_meta_data("p"), None);
    bitmap.set_jpeg_quality(90).unwrap();
    assert_eq!(
        bitmap.get_meta_data("Compression").as_deref(),
        Some("jpeg:90")
    );
    assert!(bitmap.set_jpeg_quality(0).is_err());
    assert!(bitmap.set_jpeg_quality(101).is_err());
}

#[test]
fn rust_only_copy_of_empty_bitmap_has_no_meta_data() {
    // COLMAP's copy constructor leaves meta_data_ null for an empty source: a 0x0 bitmap has
    // metadata, but its copy does not.
    let mut with_meta = Bitmap::new(0, 0, true);
    with_meta.set_meta_data("Make", MetaDataValue::String("x".to_string()));
    assert!(with_meta.is_empty());
    let copy = with_meta.clone();
    assert_eq!(copy.get_meta_data("Make"), None);
    // A non-empty bitmap's clone keeps it.
    let mut full = Bitmap::new(1, 1, true);
    full.set_meta_data("Make", MetaDataValue::String("x".to_string()));
    assert_eq!(full.clone().get_meta_data("Make").as_deref(), Some("x"));
}

#[test]
fn rust_only_bitmap_color_cast_limits() {
    // std::round is half away from zero; values clamp to [0, 255].
    let color = BitmapColor::<f32>::new(0.5, 254.5, 1000.0).cast::<u8>();
    assert_eq!(color, BitmapColor::new(1, 255, 255));
    // NaN turns into the lower limit (std::max(low, NaN) is low).
    assert_eq!(
        BitmapColor::<f32>::gray(f32::NAN).cast::<u8>(),
        BitmapColor::gray(0)
    );
    // numeric_limits<float>::min() is the smallest positive normal, so a float 0 cast to float
    // maps to FLT_MIN; from u8 the limit is first cast to uint8_t (0), so 0 stays 0.
    let float_to_float = BitmapColor::<f32>::new(0.0, 7.4, -3.0).cast::<f32>();
    assert_eq!(
        float_to_float,
        BitmapColor::new(f32::MIN_POSITIVE, 7.0, f32::MIN_POSITIVE)
    );
    let u8_to_float = BitmapColor::<u8>::new(0, 7, 255).cast::<f32>();
    assert_eq!(u8_to_float, BitmapColor::new(0.0, 7.0, 255.0));
}

#[test]
fn rust_only_jet_colormap() {
    // Expected values follow bitmap.cc's piecewise-linear Base in single precision.
    assert_eq!(JetColormap::red(0.0), 0.0);
    assert_eq!(JetColormap::green(0.0), 0.0);
    assert_eq!(JetColormap::blue(0.0), 0.5);
    assert_eq!(JetColormap::red(0.5), 0.5);
    assert_eq!(JetColormap::green(0.5), 1.0);
    assert_eq!(JetColormap::blue(0.5), 0.5);
    assert_eq!(JetColormap::red(1.0), 0.5);
    assert_eq!(JetColormap::green(1.0), 0.0);
    assert_eq!(JetColormap::blue(1.0), 0.0);
}

#[test]
fn rust_only_try_new_rejects_invalid_dimensions() {
    let ok = Bitmap::try_new(4, 3, true).unwrap();
    assert_eq!(ok.num_bytes(), 36);
    assert!(Bitmap::try_new(0, 0, false).unwrap().is_empty());
    assert!(Bitmap::try_new(-1, 3, true).is_err());
    assert!(Bitmap::try_new(3, -1, false).is_err());
    // The scan line (Pitch, a C++ int) overflows i32.
    assert!(Bitmap::try_new(i32::MAX / 2, 1, true).is_err());
    assert!(Bitmap::try_with_colorspace(i32::MAX, i32::MAX, true, true).is_err());
}

#[test]
#[should_panic(expected = "Invalid bitmap dimensions")]
fn rust_only_new_panics_on_negative_dimensions() {
    let _ = Bitmap::new(-2, 2, false);
}

#[test]
fn rust_only_exif_camera_model_non_finite_focal_uses_printf_spelling() {
    let mut bitmap = Bitmap::new(10, 8, true);
    bitmap.set_meta_data("Make", MetaDataValue::String("m".to_string()));
    bitmap.set_meta_data("Model", MetaDataValue::String("x".to_string()));
    for (focal, text) in [
        (f32::NAN, "nan"),
        (-f32::NAN, "nan"),
        (f32::INFINITY, "inf"),
        (f32::NEG_INFINITY, "-inf"),
        (1.0 / 3.0, "0.333333"),
    ] {
        bitmap.set_meta_data("Exif:FocalLength", MetaDataValue::Float(focal));
        assert_eq!(
            bitmap.exif_camera_model().as_deref(),
            Some(format!("m-x-{text}-10x8").as_str())
        );
    }
}
