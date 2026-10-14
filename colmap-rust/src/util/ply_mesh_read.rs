//! Port of `ReadPlyMesh` from COLMAP's `src/colmap/util/ply.cc` (and colmap-sharp's
//! `Util/Ply.Mesh.cs`): a triangle mesh, plain or textured (per-face UVs and a
//! "comment TextureFile" header line), from a text or binary PLY stream. The types and the
//! shared header/byte helpers are in [`super::ply`], which re-exports this reader; the
//! conventions (Tier A, growth instead of reserving, entry 280) are described there.
//! Tests: `tests/util/ply.rs`.

use std::io::Read;
use std::path::Path;

use super::ply::{
    check_byte_pos, cpp_stoi, element_count, is_double_type, item, parse_format, read_coord,
    tokens, trimmed, PlyBytes, PlyMeshFace, PlyMeshVertex, PlyTexturedMesh, MAX_INITIAL_CAPACITY,
    MAX_PLY_ELEMENTS,
};
use crate::scene::reconstruction_io_utils::open_read;
use crate::util::check::{ColmapError, ErrorKind, Result};
use crate::util::line_tokens::LineTokens;
use crate::util::string::{string_split, string_to_double};

/// The byte size of a face list's count or index type (`PlyTypeBytes`).
fn ply_type_bytes(dtype: &str) -> Result<usize> {
    match dtype {
        "int" | "int32" | "uint" | "uint32" => Ok(4),
        "short" | "int16" | "ushort" | "uint16" => Ok(2),
        "char" | "int8" | "uchar" | "uint8" => Ok(1),
        _ => Err(ColmapError::new(
            ErrorKind::RuntimeError,
            format!("Unsupported PLY face data type: {dtype}"),
        )),
    }
}

/// `ReadInt`: a 4-byte value is read as `int32`, a 2-byte one as `int16`, anything else as
/// one unsigned byte, whatever the declared signedness.
fn read_int(buf: &[u8], num_bytes: usize, little_endian: bool) -> i32 {
    match num_bytes {
        4 => {
            let b: [u8; 4] = buf[..4].try_into().unwrap_or([0; 4]);
            if little_endian {
                i32::from_le_bytes(b)
            } else {
                i32::from_be_bytes(b)
            }
        }
        2 => {
            let b: [u8; 2] = buf[..2].try_into().unwrap_or([0; 2]);
            i32::from(if little_endian {
                i16::from_le_bytes(b)
            } else {
                i16::from_be_bytes(b)
            })
        }
        _ => i32::from(buf[0]),
    }
}

/// The vertex layout `ReadPlyMesh` collects from the header.
#[derive(Clone, Copy)]
struct VertexProp {
    index: i32,
    byte_pos: i32,
    double: bool,
}

const NO_PROP: VertexProp = VertexProp {
    index: -1,
    byte_pos: -1,
    double: false,
};

/// `static_cast<int>(num_vertices)`: the bound face indices are checked against (modular,
/// as C++20 defines the narrowing).
fn vertex_bound(num_vertices: u64) -> i32 {
    num_vertices as i32
}

/// Port of `ReadPlyMesh`: a plain or textured triangle mesh from a text or binary stream.
pub fn read_ply_mesh<R: Read + ?Sized>(stream: &mut R) -> Result<PlyTexturedMesh> {
    let mut file = PlyBytes::new(stream)?;
    let mut result = PlyTexturedMesh::default();

    let mut is_binary = false;
    let mut is_little_endian = false;
    let mut num_vertices: u64 = 0;
    let mut num_faces: u64 = 0;
    let mut has_texcoord = false;
    let mut in_vertex_section = false;
    let mut in_face_section = false;
    let mut face_count_type = String::from("uchar");
    let mut face_index_type = String::from("int");
    let mut num_bytes_per_vertex: usize = 0;
    let mut num_vertex_props: i32 = 0;
    // x, y, z, r, g, b.
    let mut props = [NO_PROP; 6];

    while let Some(line) = file.getline() {
        let line = trimmed(line);
        if line.is_empty() {
            continue;
        }
        if line == "end_header" {
            break;
        }
        parse_format(&line, &mut is_binary, &mut is_little_endian);

        if line.len() >= 19 && line.starts_with("comment TextureFile") {
            let elems = string_split(&line, " ");
            if elems.len() >= 3 {
                result.texture_file = elems[2].clone();
            }
        }

        let line_elems = string_split(&line, " ");
        if line_elems.len() >= 3 && line_elems[0] == "element" {
            in_vertex_section = false;
            in_face_section = false;
            if line_elems[1] == "vertex" {
                num_vertices = element_count(&line_elems[2])?;
                in_vertex_section = true;
            } else if line_elems[1] == "face" {
                num_faces = element_count(&line_elems[2])?;
                in_face_section = true;
            }
        }

        // Parse face property lists.
        if in_face_section
            && line_elems.len() >= 5
            && line_elems[0] == "property"
            && line_elems[1] == "list"
        {
            if line_elems[4] == "texcoord" {
                has_texcoord = true;
            } else {
                face_count_type = line_elems[2].clone();
                face_index_type = line_elems[3].clone();
            }
        }

        if in_vertex_section && line_elems.len() >= 3 && line_elems[0] == "property" {
            let dtype = line_elems[1].as_str();
            let slot = match line_elems[2].as_str() {
                "x" => Some(0),
                "y" => Some(1),
                "z" => Some(2),
                "red" | "r" | "diffuse_red" => Some(3),
                "green" | "g" | "diffuse_green" => Some(4),
                "blue" | "b" | "diffuse_blue" => Some(5),
                _ => None,
            };
            if let Some(slot) = slot {
                props[slot] = VertexProp {
                    index: num_vertex_props,
                    byte_pos: num_bytes_per_vertex as i32,
                    double: slot < 3 && is_double_type(dtype),
                };
            }
            num_vertex_props += 1;
            num_bytes_per_vertex += match dtype {
                "float" | "float32" | "int" | "int32" | "uint" | "uint32" => 4,
                "double" | "float64" => 8,
                "short" | "int16" | "ushort" | "uint16" => 2,
                "uchar" | "uint8" | "char" | "int8" => 1,
                _ => {
                    return Err(ColmapError::new(
                        ErrorKind::RuntimeError,
                        format!("Invalid vertex data type: {dtype}"),
                    ))
                }
            };
        }
    }

    crate::check!(
        props[..3].iter().all(|p| p.index != -1),
        "Invalid PLY mesh format: x, y, z properties missing"
    );
    let has_colors = props[3..].iter().all(|p| p.index != -1);

    // Validate byte positions against buffer size for binary PLY mesh files.
    if is_binary && num_bytes_per_vertex > 0 {
        const NAMES: [&str; 6] = ["x", "y", "z", "r", "g", "b"];
        let checked = if has_colors { 6 } else { 3 };
        for (slot, p) in props.iter().enumerate().take(checked) {
            let size = if slot >= 3 {
                1
            } else if p.double {
                8
            } else {
                4
            };
            let what = format!("PLY mesh property {}", NAMES[slot]);
            check_byte_pos(p.byte_pos, size, num_bytes_per_vertex, &what)?;
        }
    }

    // Sanity-check counts to prevent unbounded memory allocation.
    crate::check_le!(
        num_vertices,
        MAX_PLY_ELEMENTS,
        "PLY mesh declares too many vertices"
    );
    crate::check_le!(
        num_faces,
        MAX_PLY_ELEMENTS,
        "PLY mesh declares too many faces"
    );
    let initial = |n: u64| (n as usize).min(MAX_INITIAL_CAPACITY);
    result.mesh.vertices.reserve(initial(num_vertices));
    result.mesh.faces.reserve(initial(num_faces));
    if has_texcoord {
        result.face_uvs.reserve(initial(num_faces) * 6);
    }

    let bound = vertex_bound(num_vertices);
    if is_binary {
        read_binary_body(
            &mut file,
            &mut result,
            &props,
            has_colors,
            has_texcoord,
            is_little_endian,
            (num_vertices, num_bytes_per_vertex, bound, num_faces),
            (&face_count_type, &face_index_type),
        )?;
    } else {
        // Read ASCII vertex data. A missing line reads as empty, as `std::getline` leaves it.
        for _ in 0..num_vertices {
            let line = trimmed(file.getline().unwrap_or_default());
            let items = tokens(&line);
            let coord = |slot: usize| -> Result<f32> {
                Ok(string_to_double(item(&items, props[slot].index)?)? as f32)
            };
            let (x, y, z) = (coord(0)?, coord(1)?, coord(2)?);
            if has_colors {
                // `static_cast<uint8_t>(std::stoi(...))`: wraps into a byte.
                let color = |slot: usize| -> Result<u8> {
                    Ok(cpp_stoi(item(&items, props[slot].index)?)? as u8)
                };
                let (r, g, b) = (color(3)?, color(4)?, color(5)?);
                result
                    .mesh
                    .vertices
                    .push(PlyMeshVertex::with_color(x, y, z, r, g, b));
            } else {
                result.mesh.vertices.push(PlyMeshVertex::new(x, y, z));
            }
        }

        // Read ASCII face data.
        for i in 0..num_faces {
            let line = trimmed(file.getline().unwrap_or_default());
            let mut line_stream = LineTokens::new(&line);
            let num_face_vertices = line_stream.read_i32();
            crate::check!(num_face_vertices.is_some());
            crate::check_eq!(
                num_face_vertices.unwrap_or_default(),
                3,
                "Only triangular faces are supported"
            );
            let idx = [
                line_stream.read_i32(),
                line_stream.read_i32(),
                line_stream.read_i32(),
            ];
            crate::check!(idx.iter().all(Option::is_some));
            let idx = idx.map(Option::unwrap_or_default);
            for v in idx {
                crate::check_ge!(v, 0, "Negative face vertex index at face {}", i);
            }
            for v in idx {
                crate::check_lt!(v, bound, "Face vertex index out of bounds at face {}", i);
            }
            result.mesh.faces.push(PlyMeshFace::new(
                idx[0] as usize,
                idx[1] as usize,
                idx[2] as usize,
            ));

            if has_texcoord {
                let num_texcoords = line_stream.read_i32();
                crate::check!(num_texcoords.is_some());
                crate::check_eq!(
                    num_texcoords.unwrap_or_default(),
                    6,
                    "Expected 6 texture coordinates per triangular face"
                );
                for _ in 0..6 {
                    let uv = line_stream.read_f32();
                    crate::check!(uv.is_some());
                    result.face_uvs.push(uv.unwrap_or_default());
                }
            }
        }
    }

    Ok(result)
}

/// The binary vertex and face records of [`read_ply_mesh`].
#[allow(clippy::too_many_arguments)]
fn read_binary_body(
    file: &mut PlyBytes,
    result: &mut PlyTexturedMesh,
    props: &[VertexProp; 6],
    has_colors: bool,
    has_texcoord: bool,
    le: bool,
    (num_vertices, num_bytes_per_vertex, bound, num_faces): (u64, usize, i32, u64),
    (face_count_type, face_index_type): (&str, &str),
) -> Result<()> {
    for i in 0..num_vertices {
        let buf = file.read(num_bytes_per_vertex);
        crate::check!(buf.is_some(), "Unexpected end of PLY file at vertex {}", i);
        let buf = buf.unwrap_or_default();
        let coord = |p: &VertexProp| read_coord(buf, p.double, p.byte_pos, le);
        let (x, y, z) = (coord(&props[0]), coord(&props[1]), coord(&props[2]));
        let vertex = if has_colors {
            let c = |p: &VertexProp| buf[p.byte_pos as usize];
            PlyMeshVertex::with_color(x, y, z, c(&props[3]), c(&props[4]), c(&props[5]))
        } else {
            PlyMeshVertex::new(x, y, z)
        };
        result.mesh.vertices.push(vertex);
    }

    // Determine byte sizes of the face count and index types from the header.
    let face_count_bytes = ply_type_bytes(face_count_type)?;
    let face_index_bytes = ply_type_bytes(face_index_type)?;
    for i in 0..num_faces {
        let buf = file.read(face_count_bytes);
        crate::check!(buf.is_some(), "Unexpected end of PLY file at face {}", i);
        let num_face_vertices = read_int(buf.unwrap_or_default(), face_count_bytes, le);
        crate::check_eq!(num_face_vertices, 3, "Only triangular faces are supported");

        let mut indices = [0i32; 3];
        for (j, index) in indices.iter_mut().enumerate() {
            let buf = file.read(face_index_bytes);
            crate::check!(
                buf.is_some(),
                "Unexpected end of PLY file at face {} index {}",
                i,
                j
            );
            *index = read_int(buf.unwrap_or_default(), face_index_bytes, le);
            crate::check_ge!(*index, 0, "Negative face vertex index at face {}", i);
            crate::check_lt!(
                *index,
                bound,
                "Face vertex index out of bounds at face {}",
                i
            );
        }
        result.mesh.faces.push(PlyMeshFace::new(
            indices[0] as usize,
            indices[1] as usize,
            indices[2] as usize,
        ));

        if has_texcoord {
            // COLMAP reads the count and the UVs unchecked; a cut file fails here (entry 280).
            let count = file.read(1).map(|b| b[0]);
            crate::check!(count.is_some(), "Unexpected end of PLY file at face {}", i);
            crate::check_eq!(
                count.unwrap_or_default(),
                6u8,
                "Expected 6 texture coordinates per triangular face"
            );
            for _ in 0..6 {
                let uv = file.read(4).map(|b| read_coord(b, false, 0, le));
                crate::check!(uv.is_some(), "Unexpected end of PLY file at face {}", i);
                result.face_uvs.push(uv.unwrap_or_default());
            }
        }
    }
    Ok(())
}

/// [`read_ply_mesh`] from a file.
pub fn read_ply_mesh_path(path: &Path) -> Result<PlyTexturedMesh> {
    read_ply_mesh(&mut open_read(path)?)
}
