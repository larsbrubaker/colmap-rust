//! Port of the writers of COLMAP's `src/colmap/util/ply.cc` (and colmap-sharp's
//! `Util/Ply.Write.cs`): `WriteTextPlyPoints`, `WriteBinaryPlyPoints`, `WriteTextPlyMesh` and
//! `WriteBinaryPlyMesh`, on any `Write` stream or on a path. The types and readers are
//! [`super::ply`], which re-exports these. Tests: `tests/util/ply.rs`.
//!
//! Tier A (exact): text values go through a classic-locale `std::ostream` at its default
//! precision, which is printf's `%.6g` ([`format_double`]) of the float widened to double;
//! binary bodies are little-endian, as COLMAP writes them, after an ASCII header.

use std::io::Write;
use std::path::Path;

use super::ply::{ply_io_error, PlyPoint, PlyTexturedMesh};
use crate::util::check::Result;
use crate::util::file::open_write;
use crate::util::stream_format::{format_double, DEFAULT_PRECISION};

/// `file << value` for a float.
fn num(value: f32) -> String {
    format_double(f64::from(value), DEFAULT_PRECISION)
}

fn put<W: Write + ?Sized>(stream: &mut W, bytes: &[u8]) -> Result<()> {
    stream.write_all(bytes).map_err(ply_io_error)
}

/// The point-cloud header both point writers share.
fn points_header(format: &str, count: usize, write_normal: bool, write_rgb: bool) -> String {
    let mut header = format!("ply\nformat {format} 1.0\nelement vertex {count}\n");
    header.push_str("property float x\nproperty float y\nproperty float z\n");
    if write_normal {
        header.push_str("property float nx\nproperty float ny\nproperty float nz\n");
    }
    if write_rgb {
        header.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    }
    header.push_str("end_header\n");
    header
}

/// The mesh header both mesh writers share, after the UV count check.
fn mesh_header(format: &str, mesh: &PlyTexturedMesh) -> Result<String> {
    let has_texcoords = !mesh.face_uvs.is_empty();
    if has_texcoords {
        crate::check_eq!(
            mesh.face_uvs.len(),
            mesh.mesh.faces.len() * 6,
            "Expected 6 UV coordinates per face"
        );
    }
    let mut header = format!("ply\nformat {format} 1.0\n");
    if !mesh.texture_file.is_empty() {
        header.push_str(&format!("comment TextureFile {}\n", mesh.texture_file));
    }
    header.push_str(&format!("element vertex {}\n", mesh.mesh.vertices.len()));
    header.push_str("property float x\nproperty float y\nproperty float z\n");
    header.push_str(&format!("element face {}\n", mesh.mesh.faces.len()));
    if has_texcoords {
        header.push_str("property list uchar int vertex_indices\n");
        header.push_str("property list uchar float texcoord\n");
    } else {
        header.push_str("property list uchar int vertex_index\n");
    }
    header.push_str("end_header\n");
    Ok(header)
}

/// Port of `WriteTextPlyPoints` (COLMAP's defaults: `write_normal = write_rgb = true`).
pub fn write_text_ply_points<W: Write + ?Sized>(
    stream: &mut W,
    points: &[PlyPoint],
    write_normal: bool,
    write_rgb: bool,
) -> Result<()> {
    let mut text = points_header("ascii", points.len(), write_normal, write_rgb);
    for point in points {
        text.push_str(&format!(
            "{} {} {}",
            num(point.x),
            num(point.y),
            num(point.z)
        ));
        if write_normal {
            text.push_str(&format!(
                " {} {} {}",
                num(point.nx),
                num(point.ny),
                num(point.nz)
            ));
        }
        if write_rgb {
            text.push_str(&format!(" {} {} {}", point.r, point.g, point.b));
        }
        text.push('\n');
    }
    put(stream, text.as_bytes())
}

/// Port of `WriteBinaryPlyPoints`: an ASCII header, then little-endian records.
pub fn write_binary_ply_points<W: Write + ?Sized>(
    stream: &mut W,
    points: &[PlyPoint],
    write_normal: bool,
    write_rgb: bool,
) -> Result<()> {
    let mut bytes = points_header(
        "binary_little_endian",
        points.len(),
        write_normal,
        write_rgb,
    )
    .into_bytes();
    for point in points {
        for v in [point.x, point.y, point.z] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        if write_normal {
            for v in [point.nx, point.ny, point.nz] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        if write_rgb {
            bytes.extend_from_slice(&[point.r, point.g, point.b]);
        }
    }
    put(stream, &bytes)
}

/// Port of `WriteTextPlyMesh`: writes the texture coordinates and the TextureFile comment
/// when the mesh has them. Vertex colors are not written, as in COLMAP.
pub fn write_text_ply_mesh<W: Write + ?Sized>(
    stream: &mut W,
    mesh: &PlyTexturedMesh,
) -> Result<()> {
    let mut text = mesh_header("ascii", mesh)?;
    for vertex in &mesh.mesh.vertices {
        text.push_str(&format!(
            "{} {} {}\n",
            num(vertex.x),
            num(vertex.y),
            num(vertex.z)
        ));
    }
    let has_texcoords = !mesh.face_uvs.is_empty();
    for (i, face) in mesh.mesh.faces.iter().enumerate() {
        text.push_str(&format!(
            "3 {} {} {}",
            face.vertex_idx1, face.vertex_idx2, face.vertex_idx3
        ));
        if has_texcoords {
            text.push_str(" 6");
            for uv in &mesh.face_uvs[i * 6..i * 6 + 6] {
                text.push(' ');
                text.push_str(&num(*uv));
            }
        }
        text.push('\n');
    }
    put(stream, text.as_bytes())
}

/// Port of `WriteBinaryPlyMesh`: an ASCII header, then little-endian vertex and face
/// records. A face index past the vertices fails after the bytes before it are written, as
/// in COLMAP.
pub fn write_binary_ply_mesh<W: Write + ?Sized>(
    stream: &mut W,
    mesh: &PlyTexturedMesh,
) -> Result<()> {
    let mut bytes = mesh_header("binary_little_endian", mesh)?.into_bytes();
    for vertex in &mesh.mesh.vertices {
        for v in [vertex.x, vertex.y, vertex.z] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
    }
    let has_texcoords = !mesh.face_uvs.is_empty();
    let num_vertices = mesh.mesh.vertices.len();
    for (i, face) in mesh.mesh.faces.iter().enumerate() {
        let checked = (|| {
            crate::check_lt!(face.vertex_idx1, num_vertices);
            crate::check_lt!(face.vertex_idx2, num_vertices);
            crate::check_lt!(face.vertex_idx3, num_vertices);
            Ok(())
        })();
        if let Err(error) = checked {
            put(stream, &bytes)?;
            return Err(error);
        }
        bytes.push(3);
        // `WriteBinaryLittleEndian<int>` of a `size_t`: the index narrowed to 32 bits.
        for idx in [face.vertex_idx1, face.vertex_idx2, face.vertex_idx3] {
            bytes.extend_from_slice(&(idx as i32).to_le_bytes());
        }
        if has_texcoords {
            bytes.push(6);
            for uv in &mesh.face_uvs[i * 6..i * 6 + 6] {
                bytes.extend_from_slice(&uv.to_le_bytes());
            }
        }
    }
    put(stream, &bytes)
}

/// Writes `write`'s output to `path` (created or truncated).
fn write_path(path: &Path, write: impl FnOnce(&mut dyn Write) -> Result<()>) -> Result<()> {
    let mut file = open_write(path)?;
    write(&mut file)?;
    file.flush().map_err(ply_io_error)
}

/// [`write_text_ply_points`] to a file.
pub fn write_text_ply_points_path(
    path: &Path,
    points: &[PlyPoint],
    write_normal: bool,
    write_rgb: bool,
) -> Result<()> {
    write_path(path, |s| {
        write_text_ply_points(s, points, write_normal, write_rgb)
    })
}

/// [`write_binary_ply_points`] to a file.
pub fn write_binary_ply_points_path(
    path: &Path,
    points: &[PlyPoint],
    write_normal: bool,
    write_rgb: bool,
) -> Result<()> {
    write_path(path, |s| {
        write_binary_ply_points(s, points, write_normal, write_rgb)
    })
}

/// [`write_text_ply_mesh`] to a file.
pub fn write_text_ply_mesh_path(path: &Path, mesh: &PlyTexturedMesh) -> Result<()> {
    write_path(path, |s| write_text_ply_mesh(s, mesh))
}

/// [`write_binary_ply_mesh`] to a file.
pub fn write_binary_ply_mesh_path(path: &Path, mesh: &PlyTexturedMesh) -> Result<()> {
    write_path(path, |s| write_binary_ply_mesh(s, mesh))
}
