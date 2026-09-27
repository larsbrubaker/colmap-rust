// Rust-only tests (COLMAP reads EXIF through OpenImageIO and has no parser test) for
// src/sensor/exif_reader.rs; port of colmap-sharp's ExifReaderTests.cs. Each test hand-builds
// a minimal JPEG whose APP1 segment carries a TIFF/EXIF block laid out per TIFF 6.0 and
// Exif 2.3, in both byte orders, and checks the values land where Bitmap's ported EXIF
// getters read them.

use colmap_rust::sensor::bitmap::Bitmap;
use colmap_rust::sensor::exif_reader::{read_jpeg_exif, read_tiff_exif};

const BYTE: u16 = 1;
const ASCII: u16 = 2;
const SHORT: u16 = 3;
const LONG: u16 = 4;
const RATIONAL: u16 = 5;

/// An IFD entry; `points_to_ifd` makes it a LONG pointer to that IFD.
struct Entry {
    tag: u16,
    value_type: u16,
    count: u32,
    value: Vec<u8>,
    points_to_ifd: Option<usize>,
}

fn entry(tag: u16, value_type: u16, count: u32, value: Vec<u8>) -> Entry {
    Entry {
        tag,
        value_type,
        count,
        value,
        points_to_ifd: None,
    }
}

fn pointer(tag: u16, ifd: usize) -> Entry {
    Entry {
        tag,
        value_type: LONG,
        count: 1,
        value: Vec::new(),
        points_to_ifd: Some(ifd),
    }
}

fn u16_bytes(value: u16, little_endian: bool) -> Vec<u8> {
    if little_endian {
        value.to_le_bytes().to_vec()
    } else {
        value.to_be_bytes().to_vec()
    }
}

fn u32_bytes(value: u32, little_endian: bool) -> Vec<u8> {
    if little_endian {
        value.to_le_bytes().to_vec()
    } else {
        value.to_be_bytes().to_vec()
    }
}

fn rationals(little_endian: bool, values: &[(u32, u32)]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&(n, d)| [u32_bytes(n, little_endian), u32_bytes(d, little_endian)].concat())
        .collect()
}

fn ascii_entry(tag: u16, text: &str) -> Entry {
    let mut value = text.as_bytes().to_vec();
    value.push(0);
    entry(tag, ASCII, value.len() as u32, value)
}

fn build_sample_tiff(le: bool) -> Vec<u8> {
    let ifd0 = vec![
        ascii_entry(0x010F, "Canon"),
        ascii_entry(0x0110, "Canon EOS 5D"),
        entry(0x0112, SHORT, 1, u16_bytes(6, le)),
        pointer(0x8769, 1),
        pointer(0x8825, 2),
    ];
    let exif = vec![
        entry(0x920A, RATIONAL, 1, rationals(le, &[(49, 2)])),
        entry(0xA002, LONG, 1, u32_bytes(4000, le)),
        entry(0xA003, SHORT, 1, u16_bytes(3000, le)),
        entry(0xA20E, RATIONAL, 1, rationals(le, &[(3000, 1)])),
        entry(0xA210, SHORT, 1, u16_bytes(2, le)),
        entry(0xA405, SHORT, 1, u16_bytes(35, le)),
    ];
    let gps = vec![
        ascii_entry(1, "S"),
        entry(2, RATIONAL, 3, rationals(le, &[(46, 1), (30, 1), (900, 1)])),
        ascii_entry(3, "E"),
        entry(
            4,
            RATIONAL,
            3,
            rationals(le, &[(92, 1), (60, 2), (1800, 2)]),
        ),
        entry(5, BYTE, 1, vec![1]),
        entry(6, RATIONAL, 1, rationals(le, &[(25, 2)])),
    ];
    build_tiff(le, &[ifd0, exif, gps])
}

/// Lays out a TIFF block: 8-byte header, then each IFD followed by the values of its entries
/// that do not fit in four bytes.
fn build_tiff(le: bool, ifds: &[Vec<Entry>]) -> Vec<u8> {
    let mut ifd_offsets = Vec::new();
    let mut offset = 8usize;
    for ifd in ifds {
        ifd_offsets.push(offset);
        offset += 2 + 12 * ifd.len() + 4;
        for e in ifd {
            if e.value.len() > 4 {
                offset += (e.value.len() + 1) & !1;
            }
        }
    }

    let mut tiff = vec![0u8; offset];
    let marker = if le { b'I' } else { b'M' };
    tiff[0] = marker;
    tiff[1] = marker;
    fn put(tiff: &mut [u8], at: usize, bytes: &[u8]) {
        tiff[at..at + bytes.len()].copy_from_slice(bytes);
    }
    put(&mut tiff, 2, &u16_bytes(42, le));
    put(&mut tiff, 4, &u32_bytes(ifd_offsets[0] as u32, le));
    for (i, ifd) in ifds.iter().enumerate() {
        let mut position = ifd_offsets[i];
        let mut data_position = position + 2 + 12 * ifd.len() + 4;
        put(&mut tiff, position, &u16_bytes(ifd.len() as u16, le));
        position += 2;
        for e in ifd {
            let value = match e.points_to_ifd {
                Some(target) => u32_bytes(ifd_offsets[target] as u32, le),
                None => e.value.clone(),
            };
            put(&mut tiff, position, &u16_bytes(e.tag, le));
            put(&mut tiff, position + 2, &u16_bytes(e.value_type, le));
            put(&mut tiff, position + 4, &u32_bytes(e.count, le));
            if value.len() <= 4 {
                put(&mut tiff, position + 8, &value);
            } else {
                put(
                    &mut tiff,
                    position + 8,
                    &u32_bytes(data_position as u32, le),
                );
                put(&mut tiff, data_position, &value);
                data_position += (value.len() + 1) & !1;
            }
            position += 12;
        }
        // Next-IFD offset stays 0: no chained IFD (thumbnail).
    }
    tiff
}

fn build_jpeg(tiff: &[u8]) -> Vec<u8> {
    let mut jpeg = vec![0xFF, 0xD8];
    // An APP0 segment first, as most cameras write, so the reader has to skip it.
    jpeg.extend_from_slice(&[0xFF, 0xE0, 0x00, 0x07, b'J', b'F', b'I', b'F', 0]);
    let mut payload = b"Exif\0\0".to_vec();
    payload.extend_from_slice(tiff);
    jpeg.extend_from_slice(&[0xFF, 0xE1]);
    jpeg.extend_from_slice(&u16_bytes(payload.len() as u16 + 2, false));
    jpeg.extend_from_slice(&payload);
    jpeg.extend_from_slice(&[0xFF, 0xD9]);
    jpeg
}

#[test]
fn rust_only_read_jpeg_extracts_camera_focal_and_gps() {
    for little_endian in [true, false] {
        let mut bitmap = Bitmap::new(4000, 3000, true);
        let jpeg = build_jpeg(&build_sample_tiff(little_endian));

        assert!(read_jpeg_exif(&jpeg, &mut bitmap));
        assert_eq!(bitmap.get_meta_data("Make").as_deref(), Some("Canon"));
        assert_eq!(
            bitmap.get_meta_data("Model").as_deref(),
            Some("Canon EOS 5D")
        );
        assert_eq!(bitmap.exif_orientation(), Some(6));
        assert_eq!(bitmap.get_meta_data_float("Exif:FocalLength"), Some(24.5));
        assert_eq!(
            bitmap.get_meta_data_int("Exif:FocalLengthIn35mmFilm"),
            Some(35)
        );
        assert_eq!(
            bitmap.get_meta_data_float("Exif:FocalPlaneXResolution"),
            Some(3000.0)
        );
        assert_eq!(
            bitmap.get_meta_data_int("Exif:FocalPlaneResolutionUnit"),
            Some(2)
        );
        assert_eq!(bitmap.get_meta_data_int("Exif:PixelXDimension"), Some(4000));
        assert_eq!(bitmap.get_meta_data_int("Exif:PixelYDimension"), Some(3000));

        // The getters see what OIIO would have stored: the 35 mm focal length wins.
        assert_eq!(bitmap.exif_focal_length(), Some(35.0 / 43.27 * 5000.0));
        assert_eq!(
            bitmap.exif_camera_model().as_deref(),
            Some("Canon-Canon EOS 5D-35.000000-4000x3000")
        );
        assert_eq!(
            bitmap.exif_latitude(),
            Some(-(46.0 + 30.0 / 60.0 + 900.0 / 3600.0))
        );
        assert_eq!(bitmap.exif_longitude(), Some(92.75));
        assert_eq!(bitmap.exif_altitude(), Some(-f64::from(12.5f32)));
    }
}

#[test]
fn rust_only_read_jpeg_without_exif_returns_false() {
    let mut bitmap = Bitmap::new(2, 2, false);
    // SOI, an APP0 (JFIF) segment, then SOS: no APP1.
    let jpeg = [
        0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x07, b'J', b'F', b'I', b'F', 0, 0xFF, 0xDA, 0x00, 0x02,
    ];
    assert!(!read_jpeg_exif(&jpeg, &mut bitmap));
    assert!(!read_jpeg_exif(&[0x89, b'P', b'N', b'G'], &mut bitmap));
    assert!(bitmap.get_meta_data("Make").is_none());
}

#[test]
fn rust_only_read_jpeg_truncated_input_never_panics() {
    let jpeg = build_jpeg(&build_sample_tiff(true));
    for length in 0..jpeg.len() {
        let mut bitmap = Bitmap::new(4, 3, true);
        // Must not panic for any prefix; the result may be true or false.
        read_jpeg_exif(&jpeg[..length], &mut bitmap);
    }

    // A TIFF block whose IFD offsets point past its end reads nothing but still parses.
    let tiff = build_sample_tiff(true);
    let mut short_bitmap = Bitmap::new(4, 3, true);
    assert!(read_tiff_exif(&tiff[..12], &mut short_bitmap));
    assert!(short_bitmap.get_meta_data("Make").is_none());
}
