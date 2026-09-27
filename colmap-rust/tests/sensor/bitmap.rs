// Port of colmap/sensor/bitmap_test.cc, first half: the BitmapColor cases and the Bitmap cases
// from Empty through ThumbnailNoOp (same test names, values and checks). The second half
// (Rot90 through the EXIF getters) is in bitmap_exif.rs.
//
// Translation notes:
// - C++ move construction/assignment (which leaves the source empty) is `std::mem::take`, so
//   the Move* cases are ported with the same checks on both bitmaps.
// - Copy construction/assignment is `Clone`.
// - Not ported (OpenImageIO file I/O; the host decodes and encodes images, see
//   PORTING_PLAN.md "Skipped tests" -> sensor/): ReadWriteAsRGB, ReadWriteUnicodePath, ReadWriteAsGrey,
//   ReadWriteAsGreyNonLinear, ReadWriteLinearColorspace, WriteJpegWithQuality,
//   WriteInvalidFormat, ReadNonImageFile, ReadNonExistentFile, ReadUnsupportedChannels, all
//   ParameterizedBitmapFormatTests, and the PNG round-trip tails of CloneAsRGB / CloneAsGrey.

use colmap_rust::sensor::bitmap::{Bitmap, BitmapColor, RescaleFilter};

#[test]
fn bitmap_color_empty() {
    let color = BitmapColor::<u8>::default();
    assert_eq!(color.r, 0);
    assert_eq!(color.g, 0);
    assert_eq!(color.b, 0);
    assert_eq!(color, BitmapColor::gray(0));
    assert_eq!(color, BitmapColor::new(0, 0, 0));
}

#[test]
fn bitmap_color_gray() {
    let color = BitmapColor::<u8>::gray(5);
    assert_eq!(color.r, 5);
    assert_eq!(color.g, 5);
    assert_eq!(color.b, 5);
}

#[test]
fn bitmap_color_rgb() {
    let color = BitmapColor::<u8>::new(1, 2, 3);
    assert_eq!(color.r, 1);
    assert_eq!(color.g, 2);
    assert_eq!(color.b, 3);
}

#[test]
fn bitmap_color_cast() {
    let color1 = BitmapColor::<f32>::new(1.1, 2.9, -3.0);
    let color2: BitmapColor<u8> = color1.cast::<u8>();
    assert_eq!(color2.r, 1);
    assert_eq!(color2.g, 3);
    assert_eq!(color2.b, 0);
}

#[test]
fn bitmap_color_print_uint8() {
    let color = BitmapColor::<u8>::new(1, 2, 3);
    assert_eq!(color.to_string(), "RGB(1, 2, 3)");
}

#[test]
fn bitmap_color_print_float() {
    let color = BitmapColor::<f32>::new(1.3, 2.4, 3.5);
    assert_eq!(color.to_string(), "RGB(1.3, 2.4, 3.5)");
}

#[test]
fn bitmap_empty() {
    let bitmap = Bitmap::empty();
    assert_eq!(bitmap.width(), 0);
    assert_eq!(bitmap.height(), 0);
    assert_eq!(bitmap.channels(), 0);
    assert!(!bitmap.is_rgb());
    assert!(!bitmap.is_grey());
    assert!(bitmap.is_empty());
}

#[test]
fn bitmap_print() {
    let bitmap = Bitmap::new(100, 80, true);
    assert_eq!(
        bitmap.to_string(),
        "Bitmap(width=100, height=80, channels=3)"
    );
}

#[test]
fn bitmap_allocate_rgb() {
    let bitmap = Bitmap::new(100, 80, true);
    assert_eq!(bitmap.width(), 100);
    assert_eq!(bitmap.height(), 80);
    assert_eq!(bitmap.channels(), 3);
    assert!(bitmap.is_rgb());
    assert!(!bitmap.is_grey());
    assert!(!bitmap.is_empty());
}

#[test]
fn bitmap_allocate_grey() {
    let bitmap = Bitmap::new(100, 80, false);
    assert_eq!(bitmap.width(), 100);
    assert_eq!(bitmap.height(), 80);
    assert_eq!(bitmap.channels(), 1);
    assert!(!bitmap.is_rgb());
    assert!(bitmap.is_grey());
    assert!(!bitmap.is_empty());
}

#[test]
fn bitmap_move_construct_empty() {
    let mut bitmap = Bitmap::empty();
    let moved_bitmap = std::mem::take(&mut bitmap);
    assert_eq!(moved_bitmap.width(), 0);
    assert_eq!(moved_bitmap.height(), 0);
    assert_eq!(moved_bitmap.channels(), 0);
    assert!(moved_bitmap.is_empty());
}

#[test]
fn bitmap_move_construct() {
    let mut bitmap = Bitmap::new(2, 1, true);
    let moved_bitmap = std::mem::take(&mut bitmap);
    assert_eq!(moved_bitmap.width(), 2);
    assert_eq!(moved_bitmap.height(), 1);
    assert_eq!(moved_bitmap.channels(), 3);
    assert_eq!(bitmap.width(), 0);
    assert_eq!(bitmap.height(), 0);
    assert_eq!(bitmap.channels(), 0);
    assert!(bitmap.is_empty());
}

#[test]
fn bitmap_move_assign_empty() {
    let mut bitmap = Bitmap::empty();
    let moved_bitmap = std::mem::take(&mut bitmap);
    assert_eq!(moved_bitmap.width(), 0);
    assert_eq!(moved_bitmap.height(), 0);
    assert_eq!(moved_bitmap.channels(), 0);
}

#[test]
fn bitmap_move_assign() {
    let mut bitmap = Bitmap::new(2, 1, true);
    let mut moved_bitmap = Bitmap::empty();
    assert!(moved_bitmap.is_empty());
    moved_bitmap = std::mem::take(&mut bitmap);
    assert_eq!(moved_bitmap.width(), 2);
    assert_eq!(moved_bitmap.height(), 1);
    assert_eq!(moved_bitmap.channels(), 3);
    assert_eq!(bitmap.width(), 0);
    assert_eq!(bitmap.height(), 0);
    assert_eq!(bitmap.channels(), 0);
    assert!(bitmap.is_empty());
}

#[test]
fn bitmap_construct_copy_empty() {
    let bitmap = Bitmap::empty();
    let copied_bitmap = bitmap.clone();
    assert_eq!(copied_bitmap.width(), 0);
    assert_eq!(copied_bitmap.height(), 0);
    assert_eq!(copied_bitmap.channels(), 0);
}

#[test]
fn bitmap_construct_copy() {
    let bitmap = Bitmap::new(2, 1, true);
    let copied_bitmap = bitmap.clone();
    assert_eq!(copied_bitmap.width(), 2);
    assert_eq!(copied_bitmap.height(), 1);
    assert_eq!(copied_bitmap.channels(), 3);
    assert_eq!(bitmap.width(), 2);
    assert_eq!(bitmap.height(), 1);
    assert_eq!(bitmap.channels(), 3);
}

#[test]
fn bitmap_assign_copy_empty() {
    let bitmap = Bitmap::empty();
    let mut copied_bitmap = Bitmap::new(1, 1, true);
    assert!(!copied_bitmap.is_empty());
    copied_bitmap.clone_from(&bitmap);
    assert_eq!(copied_bitmap.width(), 0);
    assert_eq!(copied_bitmap.height(), 0);
    assert_eq!(copied_bitmap.channels(), 0);
}

#[test]
fn bitmap_assign_copy() {
    let bitmap = Bitmap::new(2, 1, true);
    let mut copied_bitmap = Bitmap::empty();
    copied_bitmap.clone_from(&bitmap);
    assert_eq!(copied_bitmap.width(), 2);
    assert_eq!(copied_bitmap.height(), 1);
    assert_eq!(copied_bitmap.channels(), 3);
    assert_eq!(bitmap.width(), 2);
    assert_eq!(bitmap.height(), 1);
    assert_eq!(bitmap.channels(), 3);
}

#[test]
fn bitmap_bits_per_pixel() {
    let mut bitmap = Bitmap::new(1, 1, true);
    assert_eq!(bitmap.bits_per_pixel(), 24);
    bitmap = Bitmap::new(1, 1, false);
    assert_eq!(bitmap.bits_per_pixel(), 8);
}

#[test]
fn bitmap_num_bytes() {
    let mut bitmap = Bitmap::empty();
    assert_eq!(bitmap.num_bytes(), 0);
    bitmap = Bitmap::new(100, 80, true);
    assert_eq!(bitmap.num_bytes(), 3 * 100 * 80);
    bitmap = Bitmap::new(100, 80, false);
    assert_eq!(bitmap.num_bytes(), 100 * 80);
}

fn set_column_pattern(bitmap: &mut Bitmap) {
    bitmap.set_pixel(0, 0, BitmapColor::new(0, 0, 0));
    bitmap.set_pixel(0, 1, BitmapColor::new(1, 0, 0));
    bitmap.set_pixel(0, 2, BitmapColor::new(2, 0, 0));
    bitmap.set_pixel(1, 0, BitmapColor::new(3, 0, 0));
    bitmap.set_pixel(1, 1, BitmapColor::new(4, 0, 0));
    bitmap.set_pixel(1, 2, BitmapColor::new(5, 0, 0));
}

#[test]
fn bitmap_row_major_data_rgb() {
    let mut bitmap = Bitmap::new(2, 3, true);
    set_column_pattern(&mut bitmap);
    assert_eq!(
        bitmap.row_major_data(),
        [0, 0, 0, 3, 0, 0, 1, 0, 0, 4, 0, 0, 2, 0, 0, 5, 0, 0]
    );
}

#[test]
fn bitmap_row_major_data_grey() {
    let mut bitmap = Bitmap::new(2, 3, false);
    set_column_pattern(&mut bitmap);
    assert_eq!(bitmap.row_major_data(), [0, 3, 1, 4, 2, 5]);
}

#[test]
fn bitmap_get_and_set_pixel_rgb() {
    let mut bitmap = Bitmap::new(2, 3, true);
    bitmap.set_pixel(1, 1, BitmapColor::new(1, 2, 3));
    let color = bitmap.get_pixel(1, 1);
    assert_eq!(color, Some(BitmapColor::new(1, 2, 3)));
}

#[test]
fn bitmap_get_and_set_pixel_grey() {
    let mut bitmap = Bitmap::new(2, 3, false);
    bitmap.set_pixel(1, 1, BitmapColor::new(0, 2, 3));
    let mut color = bitmap.get_pixel(1, 1);
    assert_eq!(color, Some(BitmapColor::new(0, 0, 0)));
    bitmap.set_pixel(1, 1, BitmapColor::new(1, 2, 3));
    color = bitmap.get_pixel(1, 1);
    assert_eq!(color, Some(BitmapColor::new(1, 1, 1)));
}

#[test]
fn bitmap_fill() {
    let mut bitmap = Bitmap::new(100, 100, true);
    bitmap.fill(BitmapColor::new(1, 2, 3)).unwrap();
    for y in 0..bitmap.height() {
        for x in 0..bitmap.width() {
            assert_eq!(bitmap.get_pixel(x, y), Some(BitmapColor::new(1, 2, 3)));
        }
    }
}

#[test]
fn bitmap_fill_grey() {
    let mut bitmap = Bitmap::new(10, 10, false);
    bitmap.fill(BitmapColor::gray(42)).unwrap();
    for y in 0..bitmap.height() {
        for x in 0..bitmap.width() {
            assert_eq!(bitmap.get_pixel(x, y).unwrap().r, 42);
        }
    }
}

#[test]
fn bitmap_interpolate_nearest_neighbor() {
    let mut bitmap = Bitmap::new(11, 10, true);
    bitmap.fill(BitmapColor::new(0, 0, 0)).unwrap();
    bitmap.set_pixel(5, 4, BitmapColor::new(1, 2, 3));
    let mut color = bitmap.interpolate_nearest_neighbor(5.0, 4.0);
    assert_eq!(color, Some(BitmapColor::new(1, 2, 3)));
    color = bitmap.interpolate_nearest_neighbor(5.4999, 4.4999);
    assert_eq!(color, Some(BitmapColor::new(1, 2, 3)));
    color = bitmap.interpolate_nearest_neighbor(5.5, 4.5);
    assert_eq!(color, Some(BitmapColor::new(0, 0, 0)));
    color = bitmap.interpolate_nearest_neighbor(4.5, 4.4999);
    assert_eq!(color, Some(BitmapColor::new(1, 2, 3)));
}

#[test]
fn bitmap_interpolate_bilinear() {
    let mut bitmap = Bitmap::new(11, 10, true);
    bitmap.fill(BitmapColor::new(0, 0, 0)).unwrap();
    bitmap.set_pixel(5, 4, BitmapColor::new(1, 2, 3));
    let mut color = bitmap.interpolate_bilinear(5.0, 4.0);
    assert_eq!(color, Some(BitmapColor::new(1.0, 2.0, 3.0)));
    color = bitmap.interpolate_bilinear(5.5, 4.0);
    assert_eq!(color, Some(BitmapColor::new(0.5, 1.0, 1.5)));
    color = bitmap.interpolate_bilinear(5.5, 4.5);
    assert_eq!(color, Some(BitmapColor::new(0.25, 0.5, 0.75)));
}

#[test]
fn bitmap_interpolate_bilinear_grey() {
    let mut bitmap = Bitmap::new(11, 10, false);
    bitmap.fill(BitmapColor::gray(0)).unwrap();
    bitmap.set_pixel(5, 4, BitmapColor::gray(100));
    let mut color = bitmap.interpolate_bilinear(5.0, 4.0).unwrap();
    assert_eq!(color.r, 100.0f32);
    color = bitmap.interpolate_bilinear(5.5, 4.0).unwrap();
    assert!((color.r - 50.0).abs() <= 1e-5);
    color = bitmap.interpolate_bilinear(5.5, 4.5).unwrap();
    assert!((color.r - 25.0).abs() <= 1e-5);
}

#[test]
fn bitmap_interpolate_bilinear_out_of_bounds() {
    let mut bitmap = Bitmap::new(11, 10, true);
    bitmap.fill(BitmapColor::new(1, 2, 3)).unwrap();
    // x at the right boundary: x0=10, x1=11 >= width_=11
    assert!(bitmap.interpolate_bilinear(10.0, 5.0).is_none());
    // y at the bottom boundary: y0=9, y1=10 >= height_=10
    assert!(bitmap.interpolate_bilinear(5.0, 9.0).is_none());
    // Negative x: x0=-1 < 0
    assert!(bitmap.interpolate_bilinear(-0.5, 5.0).is_none());
    // Negative y: y0=-1 < 0
    assert!(bitmap.interpolate_bilinear(5.0, -0.5).is_none());
}

fn check_rescale(as_rgb: bool, channels: i32) {
    let bitmap = Bitmap::new(100, 80, as_rgb);
    let mut bitmap1 = bitmap.clone();
    bitmap1.rescale(50, 25, RescaleFilter::default());
    assert_eq!(bitmap1.width(), 50);
    assert_eq!(bitmap1.height(), 25);
    assert_eq!(bitmap1.channels(), channels);
    let mut bitmap2 = bitmap.clone();
    bitmap2.rescale(150, 20, RescaleFilter::default());
    assert_eq!(bitmap2.width(), 150);
    assert_eq!(bitmap2.height(), 20);
    assert_eq!(bitmap2.channels(), channels);
}

#[test]
fn bitmap_rescale_rgb() {
    check_rescale(true, 3);
}

#[test]
fn bitmap_rescale_grey() {
    check_rescale(false, 1);
}

#[test]
fn bitmap_rescale_filters() {
    let mut bitmap = Bitmap::new(4, 4, false);
    bitmap.fill(BitmapColor::gray(0)).unwrap();
    bitmap.set_pixel(0, 0, BitmapColor::gray(255));

    let mut bilinear = bitmap.clone();
    bilinear.rescale(1, 1, RescaleFilter::Bilinear);
    let mut boxed = bitmap.clone();
    boxed.rescale(1, 1, RescaleFilter::Box);

    assert_ne!(
        bilinear.get_pixel(0, 0).unwrap().r,
        boxed.get_pixel(0, 0).unwrap().r
    );
}

#[test]
fn bitmap_thumbnail() {
    let bitmap = Bitmap::new(100, 80, true);

    // Landscape: width is the limiting dimension.
    let mut landscape = bitmap.clone();
    let landscape_scale = landscape.thumbnail(50, RescaleFilter::default()).unwrap();
    assert_eq!(landscape_scale, 0.5);
    assert_eq!(landscape.width(), 50);
    assert_eq!(landscape.height(), 40);

    // Portrait: height is the limiting dimension.
    let mut portrait = Bitmap::new(80, 100, true);
    let portrait_scale = portrait.thumbnail(50, RescaleFilter::default()).unwrap();
    assert_eq!(portrait_scale, 0.5);
    assert_eq!(portrait.width(), 40);
    assert_eq!(portrait.height(), 50);

    // Non-integer scale: dimensions are rounded to the nearest integer, so the limiting
    // dimension fits exactly into max_image_size (300 * 100/300 rounds to 100 rather than
    // truncating to 99) and the other is rounded (200 * 100/300 = 66.67 rounds to 67).
    let mut non_integer = Bitmap::new(300, 200, true);
    non_integer
        .thumbnail(100, RescaleFilter::default())
        .unwrap();
    assert_eq!(non_integer.width(), 100);
    assert_eq!(non_integer.height(), 67);
}

#[test]
fn bitmap_thumbnail_no_op() {
    let bitmap = Bitmap::new(100, 80, false);

    // Bound larger than both dimensions leaves the image unchanged.
    let mut larger = bitmap.clone();
    assert_eq!(
        larger.thumbnail(200, RescaleFilter::default()).unwrap(),
        1.0
    );
    assert_eq!(larger.width(), 100);
    assert_eq!(larger.height(), 80);

    // Bound equal to the largest dimension is also a no-op.
    let mut equal = bitmap.clone();
    assert_eq!(equal.thumbnail(100, RescaleFilter::default()).unwrap(), 1.0);
    assert_eq!(equal.width(), 100);
    assert_eq!(equal.height(), 80);
}
