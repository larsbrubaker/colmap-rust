//! Port of COLMAP's `src/colmap/util/ply.h/.cc`, part 1 (and colmap-sharp's `Util/Ply.cs`,
//! `Util/Ply.Mesh.cs`, `Util/PlyTypes.cs`): the PLY data types and the readers — `ReadPly`
//! (point clouds), `ReadPlyMesh` (triangle meshes, optionally textured) and
//! `HasPlyMeshFaces`. The writers are in `util/ply_write.rs`. `scene::reconstruction_ply`
//! (`ConvertToPLY` / `ImportPLY`) builds on both. Tests: `tests/util/ply.rs` (`ply_test.cc`).
//!
//! Every reader takes a `Read` stream, with a `_path` form that opens the file, as the
//! reconstruction readers do. The header is parsed line by line exactly as COLMAP does
//! (`std::getline`, `StringTrim`, `StringSplit` on ' ', `std::stoll` for element counts), and
//! the body starts at the byte after the "end_header" line. Values are read with the C++ types
//! and conversions: a double coordinate is narrowed to `f32`, an ASCII color goes through
//! `std::stoi` and wraps into a byte. Tier A (exact).
//!
//! Malformed input fails with an error, never with a huge allocation: counts are checked
//! against COLMAP's limit (2^32), and the vectors start at most `MAX_INITIAL_CAPACITY` large
//! and grow with the data actually read (COLMAP reserves the declared count). A binary mesh
//! whose texcoord lists are cut short fails with "Unexpected end of PLY file at face i", where
//! COLMAP reads them unchecked (`docs/CPP_DIVERGENCES.md` entry 280). COLMAP's `LOG(WARNING)`s
//! (an ignored non-vertex element, a malformed face count in `HasPlyMeshFaces`) are not
//! emitted: the core has no logger.

use std::io::{BufRead, BufReader, Read};
use std::path::Path;

use crate::util::check::{ColmapError, ErrorKind, Result};
use crate::util::file::open_read;
use crate::util::string::{string_split, string_to_double, string_trim, C_ISSPACE};

pub use super::ply_write::{
    write_binary_ply_mesh, write_binary_ply_mesh_path, write_binary_ply_points,
    write_binary_ply_points_path, write_text_ply_mesh, write_text_ply_mesh_path,
    write_text_ply_points, write_text_ply_points_path,
};

/// `kMaxPlyVertices` / `kMaxPlyFaces`.
pub(super) const MAX_PLY_ELEMENTS: u64 = 1 << 32;

/// The most a reader reserves before it has read any data.
pub(super) const MAX_INITIAL_CAPACITY: usize = 1 << 16;

/// Port of `PlyPoint`: a point with normal and color, all zero by default.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PlyPoint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

/// Port of `PlyMeshVertex`: a position and a color, gray (200, 200, 200) by default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlyMeshVertex {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub r: u8,
    pub g: u8,
    pub b: u8,
}

impl Default for PlyMeshVertex {
    fn default() -> Self {
        Self::new(0.0, 0.0, 0.0)
    }
}

impl PlyMeshVertex {
    /// `PlyMeshVertex(x, y, z)`: the default gray color.
    pub fn new(x: f32, y: f32, z: f32) -> Self {
        Self::with_color(x, y, z, 200, 200, 200)
    }

    /// `PlyMeshVertex(x, y, z, r, g, b)`.
    pub fn with_color(x: f32, y: f32, z: f32, r: u8, g: u8, b: u8) -> Self {
        Self { x, y, z, r, g, b }
    }
}

/// Port of `PlyMeshFace`: a triangle's three vertex indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PlyMeshFace {
    pub vertex_idx1: usize,
    pub vertex_idx2: usize,
    pub vertex_idx3: usize,
}

impl PlyMeshFace {
    /// `PlyMeshFace(vertex_idx1, vertex_idx2, vertex_idx3)`.
    pub fn new(vertex_idx1: usize, vertex_idx2: usize, vertex_idx3: usize) -> Self {
        Self {
            vertex_idx1,
            vertex_idx2,
            vertex_idx3,
        }
    }
}

/// Port of `PlyMesh`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlyMesh {
    pub vertices: Vec<PlyMeshVertex>,
    pub faces: Vec<PlyMeshFace>,
}

/// Port of `PlyTexturedMesh`: a mesh with optional per-face UVs.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct PlyTexturedMesh {
    pub mesh: PlyMesh,
    /// Per-face UV coordinates: 6 floats per face (u1,v1, u2,v2, u3,v3).
    pub face_uvs: Vec<f32>,
    /// Texture image filename referenced via "comment TextureFile ..." in the header.
    pub texture_file: String,
}

impl From<PlyMesh> for PlyTexturedMesh {
    /// `PlyTexturedMesh{mesh}`: no UVs and no texture file.
    fn from(mesh: PlyMesh) -> Self {
        Self {
            mesh,
            ..Self::default()
        }
    }
}

pub(crate) fn ply_io_error(error: std::io::Error) -> ColmapError {
    ColmapError::new(
        ErrorKind::RuntimeError,
        format!("PLY stream I/O failed: {error}"),
    )
}

/// A whole PLY file and the read position: `std::getline` for the header and the ASCII
/// body, `istream::read` for the binary body.
pub(super) struct PlyBytes {
    data: Vec<u8>,
    pos: usize,
}

impl PlyBytes {
    pub(super) fn new<R: Read + ?Sized>(stream: &mut R) -> Result<Self> {
        let mut data = Vec::new();
        stream.read_to_end(&mut data).map_err(ply_io_error)?;
        Ok(Self { data, pos: 0 })
    }

    /// `std::getline`: the next line without its '\n', or `None` at the end of the data.
    pub(super) fn getline(&mut self) -> Option<String> {
        if self.pos >= self.data.len() {
            return None;
        }
        let rest = &self.data[self.pos..];
        let len = rest.iter().position(|&b| b == b'\n').unwrap_or(rest.len());
        let line = String::from_utf8_lossy(&rest[..len]).into_owned();
        self.pos = (self.pos + len + 1).min(self.data.len());
        Some(line)
    }

    /// `file.read(buffer, n)` followed by `file.good()`: the next `n` bytes, or `None` when
    /// fewer are left.
    pub(super) fn read(&mut self, n: usize) -> Option<&[u8]> {
        let end = self.pos.checked_add(n)?;
        let bytes = self.data.get(self.pos..end)?;
        self.pos = end;
        Some(bytes)
    }
}

/// The digits `strtoll` accepts: C whitespace, an optional sign, decimal digits. `None` when
/// there are no digits; saturated magnitudes are reported as out of range by the caller.
fn parse_c_integer(s: &str) -> Option<(bool, i128)> {
    let t = s.trim_start_matches(C_ISSPACE);
    let (negative, digits) = match t.as_bytes().first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    let digits: &str = &digits[..digits
        .bytes()
        .position(|b| !b.is_ascii_digit())
        .unwrap_or(digits.len())];
    if digits.is_empty() {
        return None;
    }
    let mut value: i128 = 0;
    for b in digits.bytes() {
        // Saturates far above every C integer range, so overflow is still detected.
        value = (value * 10 + i128::from(b - b'0')).min(i128::from(u64::MAX) * 4);
    }
    Some((negative, if negative { -value } else { value }))
}

/// `std::stoll` / `std::stoi`: libc++'s `invalid_argument` ("<name>: no conversion") or
/// `out_of_range` ("<name>: out of range").
fn cpp_sto(s: &str, name: &str, min: i128, max: i128) -> Result<i128> {
    let Some((_, value)) = parse_c_integer(s) else {
        return Err(ColmapError::new(
            ErrorKind::InvalidArgument,
            format!("{name}: no conversion"),
        ));
    };
    if value < min || value > max {
        return Err(ColmapError::new(
            ErrorKind::OutOfRange,
            format!("{name}: out of range"),
        ));
    }
    Ok(value)
}

/// `std::stoll(s)`.
pub(crate) fn cpp_stoll(s: &str) -> Result<i64> {
    cpp_sto(s, "stoll", i128::from(i64::MIN), i128::from(i64::MAX)).map(|v| v as i64)
}

/// `std::stoi(s)`.
pub(super) fn cpp_stoi(s: &str) -> Result<i32> {
    cpp_sto(s, "stoi", i128::from(i32::MIN), i128::from(i32::MAX)).map(|v| v as i32)
}

/// `std::stoll` stored into a `size_t` count: a negative count wraps, as in C++, and then
/// fails the 2^32 limit.
pub(super) fn element_count(s: &str) -> Result<u64> {
    Ok(cpp_stoll(s)? as u64)
}

/// The tokens `line_stream >> item` extracts from one line.
pub(super) fn tokens(line: &str) -> Vec<&str> {
    line.split(C_ISSPACE).filter(|t| !t.is_empty()).collect()
}

/// `items.at(index)`: libc++'s `out_of_range` ("vector") for a missing value.
pub(super) fn item<'a>(items: &[&'a str], index: i32) -> Result<&'a str> {
    usize::try_from(index)
        .ok()
        .and_then(|i| items.get(i).copied())
        .ok_or_else(|| ColmapError::new(ErrorKind::OutOfRange, "vector"))
}

/// The trimmed `getline` line, `StringTrim` applied.
pub(super) fn trimmed(mut line: String) -> String {
    string_trim(&mut line);
    line
}

/// The format line's effect on `(is_binary, is_little_endian)`.
pub(super) fn parse_format(line: &str, is_binary: &mut bool, is_little_endian: &mut bool) {
    if line.len() >= 6 && line.starts_with("format") {
        match line {
            "format ascii 1.0" => *is_binary = false,
            "format binary_little_endian 1.0" => {
                *is_binary = true;
                *is_little_endian = true;
            }
            "format binary_big_endian 1.0" => {
                *is_binary = true;
                *is_little_endian = false;
            }
            _ => {}
        }
    }
}

/// A float or double property at `byte_pos`, narrowed to `f32` (`ReadCoord`).
pub(super) fn read_coord(buf: &[u8], is_double: bool, byte_pos: i32, little_endian: bool) -> f32 {
    let p = byte_pos as usize;
    if is_double {
        let bytes: [u8; 8] = buf[p..p + 8].try_into().unwrap_or([0; 8]);
        let v = if little_endian {
            f64::from_le_bytes(bytes)
        } else {
            f64::from_be_bytes(bytes)
        };
        v as f32
    } else {
        let bytes: [u8; 4] = buf[p..p + 4].try_into().unwrap_or([0; 4]);
        if little_endian {
            f32::from_le_bytes(bytes)
        } else {
            f32::from_be_bytes(bytes)
        }
    }
}

/// `CheckBytePos`: a property's bytes lie inside one record.
pub(super) fn check_byte_pos(
    byte_pos: i32,
    type_size: usize,
    record_size: usize,
    what: &str,
) -> Result<()> {
    if byte_pos >= 0 {
        let end = byte_pos as usize + type_size;
        crate::check_le!(
            end,
            record_size,
            "{} byte position exceeds {} buffer size",
            what,
            if what.starts_with("PLY mesh") {
                "vertex"
            } else {
                "line"
            }
        );
    }
    Ok(())
}

/// One vertex property of `ReadPly`: its ASCII index, binary byte position and precision.
#[derive(Clone, Copy)]
struct Prop {
    index: i32,
    byte_pos: i32,
    double: bool,
}

const NO_PROP: Prop = Prop {
    index: -1,
    byte_pos: -1,
    double: false,
};

pub(super) fn is_double_type(dtype: &str) -> bool {
    dtype == "double" || dtype == "float64"
}

/// Port of `ReadPly`: a PLY point cloud from a text or binary stream.
pub fn read_ply<R: Read + ?Sized>(stream: &mut R) -> Result<Vec<PlyPoint>> {
    let mut file = PlyBytes::new(stream)?;
    // x, y, z, nx, ny, nz, r, g, b.
    let mut props = [NO_PROP; 9];
    const NAMES: [&str; 9] = ["x", "y", "z", "nx", "ny", "nz", "r", "g", "b"];
    let mut in_vertex_section = false;
    let mut is_binary = false;
    let mut is_little_endian = false;
    let mut num_bytes_per_line: usize = 0;
    let mut num_vertices: u64 = 0;
    let mut index: i32 = 0;

    while let Some(line) = file.getline() {
        let line = trimmed(line);
        if line.is_empty() {
            continue;
        }
        if line == "end_header" {
            break;
        }
        parse_format(&line, &mut is_binary, &mut is_little_endian);

        let line_elems = string_split(&line, " ");
        if line_elems.len() >= 3 && line_elems[0] == "element" {
            in_vertex_section = false;
            if line_elems[1] == "vertex" {
                num_vertices = element_count(&line_elems[2])?;
                in_vertex_section = true;
            } else {
                // COLMAP logs "Only vertex elements supported; ignoring <name>" when the
                // count is positive; the count is still parsed (and can throw).
                cpp_stoll(&line_elems[2])?;
            }
        }
        if !in_vertex_section {
            continue;
        }

        // Show diffuse, ambient, specular colors as regular colors.
        if line_elems.len() >= 3 && line_elems[0] == "property" {
            let dtype = line_elems[1].as_str();
            crate::check!(
                matches!(dtype, "float" | "float32" | "double" | "float64" | "uchar"),
                "PLY import only supports float, double, and uchar data types"
            );
            let slot = match line.as_str() {
                "property float x" | "property float32 x" | "property double x"
                | "property float64 x" => Some(0),
                "property float y" | "property float32 y" | "property double y"
                | "property float64 y" => Some(1),
                "property float z" | "property float32 z" | "property double z"
                | "property float64 z" => Some(2),
                "property float nx"
                | "property float32 nx"
                | "property double nx"
                | "property float64 nx" => Some(3),
                "property float ny"
                | "property float32 ny"
                | "property double ny"
                | "property float64 ny" => Some(4),
                "property float nz"
                | "property float32 nz"
                | "property double nz"
                | "property float64 nz" => Some(5),
                "property uchar r"
                | "property uchar red"
                | "property uchar diffuse_red"
                | "property uchar ambient_red"
                | "property uchar specular_red" => Some(6),
                "property uchar g"
                | "property uchar green"
                | "property uchar diffuse_green"
                | "property uchar ambient_green"
                | "property uchar specular_green" => Some(7),
                "property uchar b"
                | "property uchar blue"
                | "property uchar diffuse_blue"
                | "property uchar ambient_blue"
                | "property uchar specular_blue" => Some(8),
                _ => None,
            };
            if let Some(slot) = slot {
                props[slot] = Prop {
                    index,
                    byte_pos: num_bytes_per_line as i32,
                    double: slot < 6 && is_double_type(dtype),
                };
            }
            index += 1;
            num_bytes_per_line += match dtype {
                "float" | "float32" => 4,
                "double" | "float64" => 8,
                _ => 1, // "uchar", the only type left past the check above.
            };
        }
    }

    let is_normal_missing = props[3..6].iter().any(|p| p.index == -1);
    let is_rgb_missing = props[6..9].iter().any(|p| p.index == -1);
    crate::check!(
        props[..3].iter().all(|p| p.index != -1),
        "Invalid PLY file format: x, y, z properties missing"
    );

    // Validate byte positions against buffer size to prevent out-of-bounds reads from
    // crafted PLY headers.
    if is_binary && num_bytes_per_line > 0 {
        for (slot, p) in props.iter().enumerate() {
            let size = if slot >= 6 {
                1
            } else if p.double {
                8
            } else {
                4
            };
            let what = format!("PLY property {}", NAMES[slot]);
            check_byte_pos(p.byte_pos, size, num_bytes_per_line, &what)?;
        }
    }

    // Sanity-check num_vertices to prevent unbounded memory allocation from a crafted PLY
    // header.
    crate::check_le!(
        num_vertices,
        MAX_PLY_ELEMENTS,
        "PLY file declares too many vertices"
    );
    let mut points = Vec::with_capacity((num_vertices as usize).min(MAX_INITIAL_CAPACITY));

    if is_binary {
        let le = is_little_endian;
        for i in 0..num_vertices {
            let buf = file.read(num_bytes_per_line);
            crate::check!(buf.is_some(), "Unexpected end of PLY file at vertex {}", i);
            let buf = buf.unwrap_or_default();
            let coord = |p: &Prop| read_coord(buf, p.double, p.byte_pos, le);
            let mut point = PlyPoint {
                x: coord(&props[0]),
                y: coord(&props[1]),
                z: coord(&props[2]),
                ..PlyPoint::default()
            };
            if !is_normal_missing {
                point.nx = coord(&props[3]);
                point.ny = coord(&props[4]);
                point.nz = coord(&props[5]);
            }
            if !is_rgb_missing {
                point.r = buf[props[6].byte_pos as usize];
                point.g = buf[props[7].byte_pos as usize];
                point.b = buf[props[8].byte_pos as usize];
            }
            points.push(point);
        }
    } else {
        // Every remaining line is a point, whatever the declared count.
        while let Some(line) = file.getline() {
            let line = trimmed(line);
            let items = tokens(&line);
            let value = |slot: usize| -> Result<f32> {
                Ok(string_to_double(item(&items, props[slot].index)?)? as f32)
            };
            // `static_cast<uint8_t>(std::stoi(...))`: wraps into a byte.
            let color = |slot: usize| -> Result<u8> {
                Ok(cpp_stoi(item(&items, props[slot].index)?)? as u8)
            };
            let mut point = PlyPoint {
                x: value(0)?,
                y: value(1)?,
                z: value(2)?,
                ..PlyPoint::default()
            };
            if !is_normal_missing {
                point.nx = value(3)?;
                point.ny = value(4)?;
                point.nz = value(5)?;
            }
            if !is_rgb_missing {
                point.r = color(6)?;
                point.g = color(7)?;
                point.b = color(8)?;
            }
            points.push(point);
        }
    }

    Ok(points)
}

/// [`read_ply`] from a file.
pub fn read_ply_path(path: &Path) -> Result<Vec<PlyPoint>> {
    read_ply(&mut open_read(path)?)
}

/// Port of `HasPlyMeshFaces`: whether the header declares a positive number of faces. Reads
/// only up to "end_header".
pub fn has_ply_mesh_faces<R: Read + ?Sized>(stream: &mut R) -> Result<bool> {
    let mut reader = BufReader::new(stream);
    let mut bytes = Vec::new();
    loop {
        bytes.clear();
        if reader.read_until(b'\n', &mut bytes).map_err(ply_io_error)? == 0 {
            return Ok(false);
        }
        if bytes.last() == Some(&b'\n') {
            bytes.pop();
        }
        let line = trimmed(String::from_utf8_lossy(&bytes).into_owned());
        if line == "end_header" {
            return Ok(false);
        }
        let line_elems = string_split(&line, " ");
        if line_elems.len() >= 3 && line_elems[0] == "element" && line_elems[1] == "face" {
            // A malformed count is skipped (COLMAP logs a warning).
            if matches!(cpp_stoll(&line_elems[2]), Ok(count) if count > 0) {
                return Ok(true);
            }
        }
    }
}

/// [`has_ply_mesh_faces`] of a file.
pub fn has_ply_mesh_faces_path(path: &Path) -> Result<bool> {
    has_ply_mesh_faces(&mut open_read(path)?)
}

pub use super::ply_mesh_read::{read_ply_mesh, read_ply_mesh_path};
