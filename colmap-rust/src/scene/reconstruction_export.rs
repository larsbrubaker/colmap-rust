//! Port of `colmap/scene/reconstruction_io.cc`'s exporters: `ExportPLY`, the reconstruction's
//! 3D points as a binary little-endian PLY point cloud (positions and colors, no normals),
//! on any `Write` stream or on a path. The points come from
//! [`Reconstruction::convert_to_ply`] (`scene::reconstruction_ply`) and the file from
//! [`crate::util::ply`]. Tests: `tests/scene/reconstruction_io.rs` (`ExportPLY.Nominal`).
//! Tier A (exact). COLMAP's other exporters (NVM, Cam, Recon3D, Bundler, VRML) are not
//! ported yet (PORTING_PLAN.md, Phase 4).

use std::io::Write;
use std::path::Path;

use crate::scene::reconstruction::Reconstruction;
use crate::util::ply::{write_binary_ply_points, write_binary_ply_points_path};
use crate::Result;

/// `ExportPLY`'s `kWriteNormal`.
const WRITE_NORMAL: bool = false;
/// `ExportPLY`'s `kWriteRGB`.
const WRITE_RGB: bool = true;

/// Port of `ExportPLY` on a stream.
pub fn export_ply<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    let ply_points = reconstruction.convert_to_ply();
    write_binary_ply_points(stream, &ply_points, WRITE_NORMAL, WRITE_RGB)
}

/// Port of `ExportPLY`.
pub fn export_ply_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    let ply_points = reconstruction.convert_to_ply();
    write_binary_ply_points_path(path, &ply_points, WRITE_NORMAL, WRITE_RGB)
}
