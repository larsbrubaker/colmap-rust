//! A small EXIF parser, a port of colmap-sharp's `ColmapSharp/Sensor/ExifReader.cs`, which was
//! written from the published specifications (JEITA CP-3451 "Exif 2.3" and Adobe "TIFF
//! Revision 6.0"), not ported from any library. It replaces the EXIF half of OpenImageIO's
//! image reading, which COLMAP's `Bitmap::Read` relies on and which is native and not ported
//! (`docs/LICENSE_AUDIT.md`). The host decodes the pixels itself and passes the file bytes
//! here; the values COLMAP uses land in the [`Bitmap`]'s metadata under OpenImageIO's
//! attribute names, so the ported EXIF getters read them unchanged.
//! Tests: `tests/sensor/rust_only_exif_reader.rs` (COLMAP's EXIF tests set metadata directly
//! and are ported in `tests/sensor/bitmap_exif.rs`).
//!
//! Extracted (tag, EXIF type -> metadata name, type):
//! - IFD0: Make, Model (ASCII -> string), Orientation (SHORT -> "Orientation", int)
//! - Exif IFD: FocalLength (RATIONAL -> "Exif:FocalLength", float), FocalPlaneXResolution
//!   (RATIONAL -> float), FocalPlaneResolutionUnit (SHORT -> int), FocalLengthIn35mmFilm
//!   (SHORT -> int), PixelXDimension/PixelYDimension (SHORT or LONG -> int)
//! - GPS IFD: GPSLatitudeRef/GPSLongitudeRef (ASCII -> "GPS:LatitudeRef", string),
//!   GPSLatitude/GPSLongitude (3 RATIONAL -> "GPS:Latitude", point), GPSAltitudeRef
//!   (BYTE -> int), GPSAltitude (RATIONAL -> float)
//!
//! Malformed input never panics: an entry that points outside the block, has an unexpected
//! type or count, or a rational with a zero denominator is skipped. EXIF allows n/0 (cameras
//! write 0/0 for "unknown"); OIIO most likely stores the float quotient (inf or NaN), while
//! this reader leaves the attribute unset (`docs/CPP_DIVERGENCES.md`, entry 122).

use super::bitmap::{Bitmap, MetaDataValue};

const TAG_MAKE: u16 = 0x010F;
const TAG_MODEL: u16 = 0x0110;
const TAG_ORIENTATION: u16 = 0x0112;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_GPS_IFD: u16 = 0x8825;
const TAG_FOCAL_LENGTH: u16 = 0x920A;
const TAG_PIXEL_X_DIMENSION: u16 = 0xA002;
const TAG_PIXEL_Y_DIMENSION: u16 = 0xA003;
const TAG_FOCAL_PLANE_X_RESOLUTION: u16 = 0xA20E;
const TAG_FOCAL_PLANE_RESOLUTION_UNIT: u16 = 0xA210;
const TAG_FOCAL_LENGTH_IN_35MM_FILM: u16 = 0xA405;
const TAG_GPS_LATITUDE_REF: u16 = 1;
const TAG_GPS_LATITUDE: u16 = 2;
const TAG_GPS_LONGITUDE_REF: u16 = 3;
const TAG_GPS_LONGITUDE: u16 = 4;
const TAG_GPS_ALTITUDE_REF: u16 = 5;
const TAG_GPS_ALTITUDE: u16 = 6;

const TYPE_BYTE: u16 = 1;
const TYPE_ASCII: u16 = 2;
const TYPE_SHORT: u16 = 3;
const TYPE_LONG: u16 = 4;
const TYPE_RATIONAL: u16 = 5;

/// Finds the EXIF APP1 segment of a JPEG file (its raw bytes) and stores its values in
/// `target`'s metadata. Returns false when the data is not a JPEG or has no readable EXIF
/// block.
pub fn read_jpeg_exif(jpeg: &[u8], target: &mut Bitmap) -> bool {
    // SOI marker.
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return false;
    }

    let mut position = 2usize;
    while position + 4 <= jpeg.len() {
        if jpeg[position] != 0xFF {
            return false;
        }
        let marker = jpeg[position + 1];
        if marker == 0xFF {
            // Fill byte before a marker.
            position += 1;
            continue;
        }
        position += 2;
        if marker == 0x01 || (0xD0..=0xD7).contains(&marker) {
            // Stand-alone markers (TEM, RSTn) carry no length.
            continue;
        }
        if marker == 0xD9 || marker == 0xDA {
            // EOI, or SOS after which entropy-coded data follows: APP1 must come before.
            return false;
        }

        // The segment length counts its own two bytes.
        let length = usize::from(u16::from_be_bytes([jpeg[position], jpeg[position + 1]]));
        if length < 2 || position + length > jpeg.len() {
            return false;
        }
        let payload = &jpeg[position + 2..position + length];
        if marker == 0xE1 && payload.len() >= 6 && &payload[..6] == b"Exif\0\0" {
            return read_tiff_exif(&payload[6..], target);
        }
        position += length;
    }
    false
}

/// Reads a TIFF-structured EXIF block (the APP1 payload after `"Exif\0\0"`, starting with
/// `"II*\0"` or `"MM\0*"`) into `target`'s metadata. Returns false when the header is
/// invalid.
pub fn read_tiff_exif(tiff: &[u8], target: &mut Bitmap) -> bool {
    if tiff.len() < 8 {
        return false;
    }
    let little_endian = match &tiff[..2] {
        b"II" => true,
        b"MM" => false,
        _ => return false,
    };
    let reader = TiffReader {
        data: tiff,
        little_endian,
    };
    if reader.u16_at(2) != Some(42) {
        return false;
    }

    let ifd0 = reader.u32_at(4).map_or(u64::MAX, u64::from);
    let mut exif_ifd = None;
    let mut gps_ifd = None;
    for entry in reader.entries(ifd0) {
        match entry.tag {
            TAG_MAKE => set_string(target, "Make", reader.ascii(&entry)),
            TAG_MODEL => set_string(target, "Model", reader.ascii(&entry)),
            TAG_ORIENTATION => set_int(target, "Orientation", reader.integer(&entry)),
            TAG_EXIF_IFD => exif_ifd = reader.integer(&entry),
            TAG_GPS_IFD => gps_ifd = reader.integer(&entry),
            _ => {}
        }
    }

    if let Some(exif_ifd) = exif_ifd {
        for entry in reader.entries(exif_ifd) {
            let name = match entry.tag {
                TAG_FOCAL_LENGTH => {
                    set_float(target, "Exif:FocalLength", reader.rational(&entry, 0));
                    continue;
                }
                TAG_FOCAL_PLANE_X_RESOLUTION => {
                    let value = reader.rational(&entry, 0);
                    set_float(target, "Exif:FocalPlaneXResolution", value);
                    continue;
                }
                TAG_FOCAL_PLANE_RESOLUTION_UNIT => "Exif:FocalPlaneResolutionUnit",
                TAG_FOCAL_LENGTH_IN_35MM_FILM => "Exif:FocalLengthIn35mmFilm",
                TAG_PIXEL_X_DIMENSION => "Exif:PixelXDimension",
                TAG_PIXEL_Y_DIMENSION => "Exif:PixelYDimension",
                _ => continue,
            };
            set_int(target, name, reader.integer(&entry));
        }
    }

    if let Some(gps_ifd) = gps_ifd {
        for entry in reader.entries(gps_ifd) {
            match entry.tag {
                TAG_GPS_LATITUDE_REF => set_string(target, "GPS:LatitudeRef", reader.ascii(&entry)),
                TAG_GPS_LATITUDE => set_point(target, "GPS:Latitude", &reader, &entry),
                TAG_GPS_LONGITUDE_REF => {
                    set_string(target, "GPS:LongitudeRef", reader.ascii(&entry))
                }
                TAG_GPS_LONGITUDE => set_point(target, "GPS:Longitude", &reader, &entry),
                TAG_GPS_ALTITUDE_REF => set_int(target, "GPS:AltitudeRef", reader.integer(&entry)),
                TAG_GPS_ALTITUDE => set_float(target, "GPS:Altitude", reader.rational(&entry, 0)),
                _ => {}
            }
        }
    }
    true
}

fn set_string(target: &mut Bitmap, name: &str, value: Option<String>) {
    if let Some(value) = value {
        target.set_meta_data(name, MetaDataValue::String(value));
    }
}

fn set_int(target: &mut Bitmap, name: &str, value: Option<u64>) {
    if let Some(value) = value.and_then(|v| i32::try_from(v).ok()) {
        target.set_meta_data(name, MetaDataValue::Int(value));
    }
}

fn set_float(target: &mut Bitmap, name: &str, value: Option<f64>) {
    if let Some(value) = value {
        target.set_meta_data(name, MetaDataValue::Float(value as f32));
    }
}

fn set_point(target: &mut Bitmap, name: &str, reader: &TiffReader<'_>, entry: &IfdEntry) {
    if entry.count != 3 {
        return;
    }
    if let (Some(d), Some(m), Some(s)) = (
        reader.rational(entry, 0),
        reader.rational(entry, 1),
        reader.rational(entry, 2),
    ) {
        target.set_meta_data(name, MetaDataValue::Point([d as f32, m as f32, s as f32]));
    }
}

/// One 12-byte IFD entry: tag, type, count, and where its value bytes are.
struct IfdEntry {
    tag: u16,
    value_type: u16,
    count: u32,
    value_offset: u64,
}

/// Bounds-checked, endian-aware reads of a TIFF block.
struct TiffReader<'a> {
    data: &'a [u8],
    little_endian: bool,
}

impl TiffReader<'_> {
    fn bytes<const N: usize>(&self, offset: u64) -> Option<[u8; N]> {
        let start = usize::try_from(offset).ok()?;
        let end = start.checked_add(N)?;
        self.data.get(start..end)?.try_into().ok()
    }

    fn u16_at(&self, offset: u64) -> Option<u16> {
        let b = self.bytes::<2>(offset)?;
        Some(if self.little_endian {
            u16::from_le_bytes(b)
        } else {
            u16::from_be_bytes(b)
        })
    }

    fn u32_at(&self, offset: u64) -> Option<u32> {
        let b = self.bytes::<4>(offset)?;
        Some(if self.little_endian {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn in_range(&self, offset: u64, size: u64) -> bool {
        offset
            .checked_add(size)
            .is_some_and(|end| end <= self.data.len() as u64)
    }

    /// The entries of the IFD at the given offset (empty if it is out of range).
    fn entries(&self, ifd_offset: u64) -> Vec<IfdEntry> {
        let mut entries = Vec::new();
        let Some(count) = self.u16_at(ifd_offset) else {
            return entries;
        };
        for i in 0..u64::from(count) {
            let entry_offset = ifd_offset + 2 + 12 * i;
            if !self.in_range(entry_offset, 12) {
                break;
            }
            // In range, so every read below succeeds.
            let tag = self.u16_at(entry_offset).unwrap_or(0);
            let value_type = self.u16_at(entry_offset + 2).unwrap_or(0);
            let count = self.u32_at(entry_offset + 4).unwrap_or(0);
            let value_size = type_size(value_type) * u64::from(count);
            // Values of up to four bytes are stored in the offset field itself.
            let value_offset = if value_size <= 4 {
                entry_offset + 8
            } else {
                u64::from(self.u32_at(entry_offset + 8).unwrap_or(0))
            };
            entries.push(IfdEntry {
                tag,
                value_type,
                count,
                value_offset,
            });
        }
        entries
    }

    /// An ASCII value without its NUL terminator(s), or `None`.
    fn ascii(&self, entry: &IfdEntry) -> Option<String> {
        if entry.value_type != TYPE_ASCII
            || !self.in_range(entry.value_offset, u64::from(entry.count))
        {
            return None;
        }
        let start = entry.value_offset as usize;
        let bytes = &self.data[start..start + entry.count as usize];
        let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
        Some(String::from_utf8_lossy(&bytes[..end]).into_owned())
    }

    /// The first value of a BYTE, SHORT or LONG entry, or `None`.
    fn integer(&self, entry: &IfdEntry) -> Option<u64> {
        if entry.count < 1 {
            return None;
        }
        match entry.value_type {
            TYPE_BYTE => self.bytes::<1>(entry.value_offset).map(|b| u64::from(b[0])),
            TYPE_SHORT => self.u16_at(entry.value_offset).map(u64::from),
            TYPE_LONG => self.u32_at(entry.value_offset).map(u64::from),
            _ => None,
        }
    }

    /// The `index`-th value of a RATIONAL entry, or `None` (also for a zero denominator).
    fn rational(&self, entry: &IfdEntry, index: u32) -> Option<f64> {
        if entry.value_type != TYPE_RATIONAL || index >= entry.count {
            return None;
        }
        let offset = entry.value_offset + 8 * u64::from(index);
        let numerator = self.u32_at(offset)?;
        let denominator = self.u32_at(offset + 4)?;
        if denominator == 0 {
            return None;
        }
        Some(f64::from(numerator) / f64::from(denominator))
    }
}

/// Size in bytes of one value of a TIFF field type (0 for unknown types).
fn type_size(value_type: u16) -> u64 {
    match value_type {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 => 4,
        5 | 10 | 12 => 8,
        _ => 0,
    }
}
