//! Port of `ExportRecon3D` and `ExportVRML` from `colmap/scene/reconstruction_io.cc`: the
//! Recon3D folder (`Recon/synth_0.out`, `Recon/urd-images.txt`, `Recon/imagemap_0.txt`) and the
//! VRML camera and point files. The other exporters are [`super::reconstruction_export`],
//! which re-exports these and whose conventions (Tier A `%g` text, registered image order,
//! ascending point ids, `false` after the lines written so far) apply here too.
//! Tests: `tests/scene/reconstruction_io.rs` (`ExportRecon3D.*`, `ExportVRML.Nominal`).
//!
//! `synth_file << matrix` uses Eigen's default `IOFormat`: every coefficient right-aligned
//! to the widest one's `%g` text (at the stream's precision 17), coefficients separated by
//! ' ' and rows by '\n' ([`eigen_matrix_text`]).

use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::path::Path;

use crate::linalg::Matrix3d;
use crate::scene::reconstruction::Reconstruction;
use crate::scene::reconstruction_export::{
    cam_from_world, drop_last, extra_param, is_pinhole, num, num6, put,
};
use crate::scene::reconstruction_io_utils::{open_write, stream_error};
use crate::sensor::models::{CameraModelId, RadialCameraModel, SimpleRadialCameraModel};
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::ImageId;
use crate::Result;

/// `stream << matrix` with Eigen's default `IOFormat` at precision 17 (see the header).
fn eigen_matrix_text(rows: &[Vec<f64>]) -> String {
    let cells: Vec<Vec<String>> = rows
        .iter()
        .map(|row| row.iter().map(|&v| num(v)).collect())
        .collect();
    let width = cells.iter().flatten().map(String::len).max().unwrap_or(0);
    cells
        .iter()
        .map(|row| {
            row.iter()
                .map(|cell| format!("{cell:>width$}"))
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `CreateDirIfNotExists(dir)` (not recursive).
fn create_dir_if_not_exists(dir: &Path) -> Result<()> {
    if dir.is_dir() {
        return Ok(());
    }
    std::fs::create_dir(dir).map_err(|error| {
        ColmapError::new(
            ErrorKind::RuntimeError,
            format!("Could not create directory {}: {error}", dir.display()),
        )
    })
}

fn matrix_rows(m: &Matrix3d) -> Vec<Vec<f64>> {
    (0..3)
        .map(|r| (0..3).map(|c| m[(r, c)]).collect())
        .collect()
}

/// Port of `ExportRecon3D` (COLMAP's default `skip_distortion` is `false`). Returns `false`
/// for a camera model other than SIMPLE_RADIAL, RADIAL and the pinholes, unless
/// `skip_distortion`.
pub fn export_recon3d_path(
    reconstruction: &Reconstruction,
    path: &Path,
    skip_distortion: bool,
) -> Result<bool> {
    create_dir_if_not_exists(path)?;
    let base_path = path.join("Recon");
    create_dir_if_not_exists(&base_path)?;
    let mut synth_file = open_write(&base_path.join("synth_0.out"))?;
    let mut image_list_file = open_write(&base_path.join("urd-images.txt"))?;
    let mut image_map_file = open_write(&base_path.join("imagemap_0.txt"))?;

    // Write header info.
    let mut synth = format!(
        "colmap 1.0\n{} {}\n",
        reconstruction.num_reg_images(),
        reconstruction.num_points3d()
    );
    let mut image_list = String::new();
    let mut image_map = String::new();
    let mut image_id_to_idx: HashMap<ImageId, usize> = HashMap::new();

    let mut exported = true;
    // Write image/camera info.
    for (image_idx, image_id) in reconstruction.reg_image_ids()?.into_iter().enumerate() {
        let image = reconstruction.image(image_id)?;
        let camera = reconstruction.camera(image.camera_id())?;
        // `-1 * k`, which IEEE negation reproduces exactly.
        let (k1, k2) = if skip_distortion || is_pinhole(camera) {
            (0.0, 0.0)
        } else if camera.model_id == CameraModelId::SimpleRadial {
            (-extra_param::<SimpleRadialCameraModel>(camera, 0), 0.0)
        } else if camera.model_id == CameraModelId::Radial {
            (
                -extra_param::<RadialCameraModel>(camera, 0),
                -extra_param::<RadialCameraModel>(camera, 1),
            )
        } else {
            exported = false;
            break;
        };

        let scale = 1.0 / camera.width.max(camera.height) as f64;
        synth.push_str(&format!(
            "{} {} {}\n",
            num(scale * camera.mean_focal_length()?),
            num(k1),
            num(k2)
        ));
        let cam_from_world = cam_from_world(reconstruction, image)?;
        synth.push_str(&eigen_matrix_text(&matrix_rows(
            &cam_from_world.rotation.to_rotation_matrix(),
        )));
        synth.push('\n');
        let t = cam_from_world.translation;
        synth.push_str(&eigen_matrix_text(&[vec![t.x, t.y, t.z]]));
        synth.push('\n');

        image_id_to_idx.insert(image_id, image_idx);
        image_list.push_str(&format!(
            "{}\n{} {}\n",
            image.name(),
            camera.width,
            camera.height
        ));
        image_map.push_str(&format!("{image_idx}\n"));
    }
    put(&mut image_list_file, &image_list)?;
    put(&mut image_map_file, &image_map)?;
    image_list_file.flush().map_err(stream_error)?;
    image_map_file.flush().map_err(stream_error)?;
    if !exported {
        put(&mut synth_file, &synth)?;
        synth_file.flush().map_err(stream_error)?;
        return Ok(false);
    }

    // Write point info.
    for p in reconstruction.points3d().values() {
        synth.push_str(&format!(
            "{} {} {}\n{} {} {}\n",
            num(p.xyz.x),
            num(p.xyz.y),
            num(p.xyz.z),
            p.color.x,
            p.color.y,
            p.color.z
        ));
        let mut line = String::new();
        let mut image_ids: HashSet<ImageId> = HashSet::new();
        for track_el in p.track.elements() {
            // Make sure that each point only has a single observation per image, since
            // VisualSfM does not support with multiple observations.
            if image_ids.insert(track_el.image_id) {
                let image = reconstruction.image(track_el.image_id)?;
                let camera = reconstruction.camera(image.camera_id())?;
                let point2d = image.point2d(track_el.point2d_idx);
                let scale = 1.0 / camera.width.max(camera.height) as f64;
                // `image_id_to_idx_[id]`: an unregistered image inserts and reads 0.
                let idx = image_id_to_idx
                    .get(&track_el.image_id)
                    .copied()
                    .unwrap_or(0);
                // Use a scale of -1.0 to mark as invalid as it is not needed currently.
                line.push_str(&format!(
                    "{} {} -1.0 {} {} ",
                    idx,
                    track_el.point2d_idx,
                    num6((point2d.xy.x - camera.principal_point_x()?) * scale),
                    num6((point2d.xy.y - camera.principal_point_y()?) * scale)
                ));
            }
        }
        synth.push_str(&format!("{} {}\n", image_ids.len(), drop_last(line)));
    }
    put(&mut synth_file, &synth)?;
    synth_file.flush().map_err(stream_error)?;
    Ok(true)
}

/// The `texCoord` block of every camera shape.
const VRML_TEXCOORDS: &str = " texCoord TextureCoordinate { point [\n  1 1,\n  0 1,\n  0 0,\n  \
                              1 0,\n  0 0,\n  0 0,\n  0 0,\n  0 0,\n  0 0,\n ] }\n} }\n";

/// Port of `ExportVRML`: every posed image as a camera pyramid of `image_rgb` color, scaled by
/// `image_scale` (`images_path`), and the 3D points with their colors (`points3d_path`).
/// Both files are at the stream's default precision 6. The camera vertices are
/// `world_from_cam * point.homogeneous()` summed left to right; Eigen may pair the four
/// products differently, which only a last-bit difference could expose and precision 6 hides.
pub fn export_vrml_path(
    reconstruction: &Reconstruction,
    images_path: &Path,
    points3d_path: &Path,
    image_scale: f64,
    image_rgb: [f64; 3],
) -> Result<()> {
    let mut images_file = open_write(images_path)?;
    let six = image_scale * 0.15;
    let siy = image_scale * 0.1;
    let z = six * 1.0 * 2.0;
    let points = [
        [-six, -siy, z],
        [six, -siy, z],
        [six, siy, z],
        [-six, siy, z],
        [0.0, 0.0, 0.0],
        [-six / 3.0, -siy / 3.0, z],
        [six / 3.0, -siy / 3.0, z],
        [six / 3.0, siy / 3.0, z],
        [-six / 3.0, siy / 3.0, z],
    ];
    let rgb = format!(
        "{} {} {}",
        num6(image_rgb[0]),
        num6(image_rgb[1]),
        num6(image_rgb[2])
    );

    let mut text = String::new();
    for image in reconstruction.images().values() {
        let frame = reconstruction.frame(image.frame_id()).ok();
        if !image.has_pose(frame) {
            continue;
        }
        text.push_str("Shape{\n appearance Appearance {\n");
        text.push_str("  material DEF Default-ffRffGffB Material {\n  ambientIntensity 0\n");
        text.push_str(&format!("  diffuseColor  {rgb}\n"));
        text.push_str("  emissiveColor 0.1 0.1 0.1 } }\n geometry IndexedFaceSet {\n");
        text.push_str(" solid FALSE \n colorPerVertex TRUE \n ccw TRUE \n");
        text.push_str(" coord Coordinate {\n point [\n");

        // Move camera base model to camera pose.
        let world_from_cam = cam_from_world(reconstruction, image)?.inverse().to_matrix();
        for p in &points {
            let coord = |r: usize| {
                world_from_cam[(r, 0)] * p[0]
                    + world_from_cam[(r, 1)] * p[1]
                    + world_from_cam[(r, 2)] * p[2]
                    + world_from_cam[(r, 3)]
            };
            text.push_str(&format!(
                "{} {} {}\n",
                num6(coord(0)),
                num6(coord(1)),
                num6(coord(2))
            ));
        }
        text.push_str(" ] }\ncolor Color {color [\n");
        for _ in &points {
            text.push_str(&format!(" {rgb}\n"));
        }
        text.push_str("\n] }\ncoordIndex [\n 0, 1, 2, 3, -1\n 5, 6, 4, -1\n 6, 7, 4, -1\n");
        text.push_str(" 7, 8, 4, -1\n 8, 5, 4, -1\n \n] \n");
        text.push_str(VRML_TEXCOORDS);
    }
    put(&mut images_file, &text)?;
    images_file.flush().map_err(stream_error)?;

    // Write 3D points.
    let mut points3d_file = open_write(points3d_path)?;
    let mut text = String::from(
        "#VRML V2.0 utf8\nBackground { skyColor [1.0 1.0 1.0] } \nShape{ appearance Appearance {\n \
         material Material {emissiveColor 1 1 1} }\n geometry PointSet {\n coord Coordinate {\n  \
         point [\n",
    );
    for p in reconstruction.points3d().values() {
        text.push_str(&format!(
            "{}, {}, {}\n",
            num6(p.xyz.x),
            num6(p.xyz.y),
            num6(p.xyz.z)
        ));
    }
    text.push_str(" ] }\n color Color { color [\n");
    for p in reconstruction.points3d().values() {
        let c = |v: u8| num6(f64::from(v) / 255.0);
        text.push_str(&format!(
            "{}, {}, {}\n",
            c(p.color.x),
            c(p.color.y),
            c(p.color.z)
        ));
    }
    text.push_str(" ] } } }\n");
    put(&mut points3d_file, &text)?;
    points3d_file.flush().map_err(stream_error)
}
