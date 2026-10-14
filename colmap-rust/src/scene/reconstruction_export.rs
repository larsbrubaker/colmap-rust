//! Port of `colmap/scene/reconstruction_io.cc`'s exporters: `ExportNVM` (VisualSfM),
//! `ExportCam` (one `.cam` file per image, path only), `ExportBundler` (bundle file and image
//! list) and `ExportPLY` (the 3D points as a binary
//! little-endian PLY point cloud, positions and colors, no normals), on `Write` streams or on
//! paths. The PLY points come from [`Reconstruction::convert_to_ply`]
//! (`scene::reconstruction_ply`) and the file from [`crate::util::ply`]. Tests:
//! `tests/scene/reconstruction_io.rs` (the `Export*` cases). COLMAP's Recon3D and VRML exporters
//! are not ported yet (PORTING_PLAN.md, Phase 4).
//!
//! Tier A (exact): COLMAP streams the main file at `precision(17)` and each point's
//! observation line through a fresh `std::ostringstream` at the default precision 6, both
//! `%g` ([`format_double`]). Images are written in registered order and points in ascending
//! id order (entry 213; COLMAP iterates its hash map). An unsupported camera model returns
//! `false` after the lines written so far, as COLMAP's half-written file has them; COLMAP's
//! `LOG(WARNING)` there is not emitted (the core has no logger).

use std::io::Write;
use std::path::Path;

use std::collections::{HashMap, HashSet};

use crate::geometry::rigid3::Rigid3d;
use crate::scene::camera::Camera;
use crate::scene::image::Image;
use crate::scene::reconstruction::Reconstruction;
use crate::scene::reconstruction_io_utils::{open_write, stream_error};
use crate::sensor::models::{
    CameraModel, CameraModelId, RadialCameraModel, SimpleRadialCameraModel,
};
use crate::util::file::split_file_extension;
use crate::util::ply::{write_binary_ply_points, write_binary_ply_points_path};
use crate::util::stream_format::{format_double, DEFAULT_PRECISION};
use crate::util::types::ImageId;
use crate::Result;

/// The main files' `precision(17)`.
fn num(value: f64) -> String {
    format_double(value, 17)
}

/// A fresh `std::ostringstream`'s default precision 6.
fn num6(value: f64) -> String {
    format_double(value, DEFAULT_PRECISION)
}

/// `image.CamFromWorld()` through the image's frame and rig.
fn cam_from_world(reconstruction: &Reconstruction, image: &Image) -> Result<Rigid3d> {
    let frame = reconstruction.frame(image.frame_id())?;
    let rig = reconstruction.rig(frame.rig_id())?;
    image.cam_from_world(frame, rig)
}

/// `camera.params[Model::extra_params_idxs[i]]`.
fn extra_param<M: CameraModel>(camera: &Camera, i: usize) -> f64 {
    camera.params[M::EXTRA_PARAMS_IDXS[i]]
}

fn is_pinhole(camera: &Camera) -> bool {
    matches!(
        camera.model_id,
        CameraModelId::SimplePinhole | CameraModelId::Pinhole
    )
}

/// `line_string.substr(0, line_string.size() - 1)`.
fn drop_last(mut line: String) -> String {
    line.pop();
    line
}

fn put<W: std::io::Write + ?Sized>(stream: &mut W, text: &str) -> Result<()> {
    stream.write_all(text.as_bytes()).map_err(stream_error)
}

/// Port of `ExportNVM` on a stream (COLMAP's default `skip_distortion` is `false`). Returns
/// `false` for a camera model other than SIMPLE_RADIAL and the pinholes, unless
/// `skip_distortion`.
pub fn export_nvm<W: std::io::Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
    skip_distortion: bool,
) -> Result<bool> {
    // White space added for compatibility with Meshlab.
    let mut text = format!("NVM_V3 \n \n{}  \n", reconstruction.num_reg_images());
    let mut image_id_to_idx: HashMap<ImageId, usize> = HashMap::new();
    for (image_idx, image_id) in reconstruction.reg_image_ids()?.into_iter().enumerate() {
        let image = reconstruction.image(image_id)?;
        let camera = reconstruction.camera(image.camera_id())?;
        let k = if skip_distortion || is_pinhole(camera) {
            0.0
        } else if camera.model_id == CameraModelId::SimpleRadial {
            // `-1 * k`, which IEEE negation reproduces exactly.
            -extra_param::<SimpleRadialCameraModel>(camera, 0)
        } else {
            put(stream, &text)?;
            return Ok(false);
        };
        let cam_from_world = cam_from_world(reconstruction, image)?;
        let proj_center = cam_from_world.tgt_origin_in_src();
        let q = cam_from_world.rotation;
        text.push_str(image.name());
        text.push(' ');
        for value in [
            camera.mean_focal_length()?,
            q.w,
            q.x,
            q.y,
            q.z,
            proj_center.x,
            proj_center.y,
            proj_center.z,
            k,
        ] {
            text.push_str(&num(value));
            text.push(' ');
        }
        text.push_str("0\n");
        image_id_to_idx.insert(image_id, image_idx);
    }

    text.push_str(&format!("\n{}\n", reconstruction.num_points3d()));
    for point3d in reconstruction.points3d().values() {
        for value in [point3d.xyz.x, point3d.xyz.y, point3d.xyz.z] {
            text.push_str(&num(value));
            text.push(' ');
        }
        let c = point3d.color;
        text.push_str(&format!("{} {} {} ", c.x, c.y, c.z));

        let mut line = String::new();
        let mut image_ids: HashSet<ImageId> = HashSet::new();
        for track_el in point3d.track.elements() {
            // Make sure that each point only has a single observation per image, since
            // VisualSfM does not support with multiple observations.
            if image_ids.insert(track_el.image_id) {
                let image = reconstruction.image(track_el.image_id)?;
                let point2d = image.point2d(track_el.point2d_idx);
                // `image_id_to_idx_[id]`: an unregistered image inserts and reads 0.
                let idx = image_id_to_idx
                    .get(&track_el.image_id)
                    .copied()
                    .unwrap_or(0);
                line.push_str(&format!(
                    "{} {} {} {} ",
                    idx,
                    track_el.point2d_idx,
                    num6(point2d.xy.x),
                    num6(point2d.xy.y)
                ));
            }
        }
        text.push_str(&format!("{} {}\n", image_ids.len(), drop_last(line)));
    }
    put(stream, &text)?;
    Ok(true)
}

/// Port of `ExportNVM`.
pub fn export_nvm_path(
    reconstruction: &Reconstruction,
    path: &Path,
    skip_distortion: bool,
) -> Result<bool> {
    let mut file = open_write(path)?;
    let exported = export_nvm(reconstruction, &mut file, skip_distortion)?;
    file.flush().map_err(stream_error)?;
    Ok(exported)
}

/// Port of `ExportCam`: creates the image name's directories under `path`, then writes
/// `path / <name without extension>.cam` for every registered image (COLMAP's default
/// `skip_distortion` is `false`). Returns `false` for a camera model other than
/// SIMPLE_RADIAL, RADIAL and the pinholes, unless `skip_distortion`; that image's file is
/// already created (empty) then, as in COLMAP.
pub fn export_cam_path(
    reconstruction: &Reconstruction,
    path: &Path,
    skip_distortion: bool,
) -> Result<bool> {
    reconstruction.create_image_dirs(path)?;
    for image_id in reconstruction.reg_image_ids()? {
        let image = reconstruction.image(image_id)?;
        let camera = reconstruction.camera(image.camera_id())?;
        let (name, _ext) = split_file_extension(image.name())?;
        let name_path = path.join(name + ".cam");
        let mut file = open_write(&name_path)?;

        let (k1, mut k2) = if skip_distortion || is_pinhole(camera) {
            (0.0, 0.0)
        } else if camera.model_id == CameraModelId::SimpleRadial {
            (extra_param::<SimpleRadialCameraModel>(camera, 0), 0.0)
        } else if camera.model_id == CameraModelId::Radial {
            (
                extra_param::<RadialCameraModel>(camera, 0),
                extra_param::<RadialCameraModel>(camera, 1),
            )
        } else {
            file.flush().map_err(stream_error)?;
            return Ok(false);
        };

        // If both k1 and k2 values are non-zero, then the CAM format assumes a Bundler-like
        // radial distortion model, which converts well from COLMAP. However, if k2 is zero,
        // then a different model is used that does not translate as well, so we avoid
        // setting k2 to zero.
        if k1 != 0.0 && k2 == 0.0 {
            k2 = 1e-10;
        }

        let fx = camera.focal_length_x()?;
        let fy = camera.focal_length_y()?;
        let width = camera.width as f64;
        let height = camera.height as f64;
        let focal_length = if width * fy < height * fx {
            fy / height
        } else {
            fx / width
        };

        let cam_from_world = cam_from_world(reconstruction, image)?;
        let r = cam_from_world.rotation.to_rotation_matrix();
        let t = cam_from_world.translation;
        let mut values = vec![t.x, t.y, t.z];
        for row in 0..3 {
            for col in 0..3 {
                values.push(r[(row, col)]);
            }
        }
        let line1: Vec<String> = values.into_iter().map(num).collect();
        let line2: Vec<String> = [
            focal_length,
            k1,
            k2,
            fy / fx,
            camera.principal_point_x()? / width,
            camera.principal_point_y()? / height,
        ]
        .into_iter()
        .map(num)
        .collect();
        put(
            &mut file,
            &format!("{}\n{}\n", line1.join(" "), line2.join(" ")),
        )?;
        file.flush().map_err(stream_error)?;
    }
    Ok(true)
}

/// Port of `ExportBundler` on two streams, the bundle file and the image list (COLMAP's
/// default `skip_distortion` is `false`). Returns `false` for a camera model other than
/// SIMPLE_RADIAL, RADIAL and the pinholes, unless `skip_distortion`. A track observing an
/// unregistered image fails (COLMAP's `image_id_to_idx_.at` throws).
pub fn export_bundler<W: std::io::Write + ?Sized, L: std::io::Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
    list_stream: &mut L,
    skip_distortion: bool,
) -> Result<bool> {
    let mut text = format!(
        "# Bundle file v0.3\n{} {}\n",
        reconstruction.num_reg_images(),
        reconstruction.num_points3d()
    );
    let mut list = String::new();
    let mut image_id_to_idx: HashMap<ImageId, usize> = HashMap::new();
    for (image_idx, image_id) in reconstruction.reg_image_ids()?.into_iter().enumerate() {
        let image = reconstruction.image(image_id)?;
        let camera = reconstruction.camera(image.camera_id())?;
        let (k1, k2) = if skip_distortion || is_pinhole(camera) {
            (0.0, 0.0)
        } else if camera.model_id == CameraModelId::SimpleRadial {
            (extra_param::<SimpleRadialCameraModel>(camera, 0), 0.0)
        } else if camera.model_id == CameraModelId::Radial {
            (
                extra_param::<RadialCameraModel>(camera, 0),
                extra_param::<RadialCameraModel>(camera, 1),
            )
        } else {
            put(stream, &text)?;
            put(list_stream, &list)?;
            return Ok(false);
        };
        text.push_str(&format!(
            "{} {} {}\n",
            num(camera.mean_focal_length()?),
            num(k1),
            num(k2)
        ));
        let cam_from_world = cam_from_world(reconstruction, image)?;
        let r = cam_from_world.rotation.to_rotation_matrix();
        text.push_str(&format!(
            "{} {} {}\n",
            num(r[(0, 0)]),
            num(r[(0, 1)]),
            num(r[(0, 2)])
        ));
        for row in 1..3 {
            text.push_str(&format!(
                "{} {} {}\n",
                num(-r[(row, 0)]),
                num(-r[(row, 1)]),
                num(-r[(row, 2)])
            ));
        }
        let t = cam_from_world.translation;
        text.push_str(&format!("{} {} {}\n", num(t.x), num(-t.y), num(-t.z)));
        list.push_str(image.name());
        list.push('\n');
        image_id_to_idx.insert(image_id, image_idx);
    }

    for point3d in reconstruction.points3d().values() {
        text.push_str(&format!(
            "{} {} {}\n",
            num(point3d.xyz.x),
            num(point3d.xyz.y),
            num(point3d.xyz.z)
        ));
        let c = point3d.color;
        text.push_str(&format!("{} {} {}\n", c.x, c.y, c.z));

        let mut line = format!("{} ", point3d.track.length());
        for track_el in point3d.track.elements() {
            let image = reconstruction.image(track_el.image_id)?;
            let camera = reconstruction.camera(image.camera_id())?;
            // Bundler output assumes image coordinate system origin in the lower left corner
            // of the image with the center of the lower left pixel being (0, 0). Our
            // coordinate system starts in the upper left corner with the center of the upper
            // left pixel being (0.5, 0.5).
            let point2d = image.point2d(track_el.point2d_idx);
            let idx = image_id_to_idx.get(&track_el.image_id).copied();
            crate::check!(
                idx.is_some(),
                "unordered_map::at: key not found (image {})",
                track_el.image_id
            );
            line.push_str(&format!(
                "{} {} {} {} ",
                idx.unwrap_or_default(),
                track_el.point2d_idx,
                num6(point2d.xy.x - camera.principal_point_x()?),
                num6(camera.principal_point_y()? - point2d.xy.y)
            ));
        }
        text.push_str(&drop_last(line));
        text.push('\n');
    }
    put(stream, &text)?;
    put(list_stream, &list)?;
    Ok(true)
}

/// Port of `ExportBundler`.
pub fn export_bundler_path(
    reconstruction: &Reconstruction,
    path: &Path,
    list_path: &Path,
    skip_distortion: bool,
) -> Result<bool> {
    let mut file = open_write(path)?;
    let mut list_file = open_write(list_path)?;
    let exported = export_bundler(reconstruction, &mut file, &mut list_file, skip_distortion)?;
    file.flush().map_err(stream_error)?;
    list_file.flush().map_err(stream_error)?;
    Ok(exported)
}

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
