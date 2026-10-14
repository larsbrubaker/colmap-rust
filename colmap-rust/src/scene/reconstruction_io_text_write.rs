//! Port of `colmap/scene/reconstruction_io_text.h` / `.cc`, part 2 (and colmap-sharp's
//! `Scene/ReconstructionIOText.Write.cs`): the writers of COLMAP's text model files
//! `rigs.txt`, `cameras.txt`, `frames.txt`, `images.txt` and `points3D.txt`, on any `Write`
//! stream or on a path. The readers are [`super::reconstruction_io_text`]. Tests:
//! `tests/scene/reconstruction_io.rs` and `tests/scene/rust_only_reconstruction_io.rs`
//! (byte-exact against pycolmap's files).
//!
//! Tier A: COLMAP streams every double with `precision(17)` in the classic locale, which is
//! printf's `%.17g` ([`crate::util::stream_format::format_double`]), so the files are
//! byte-identical to COLMAP's. Two quirks are kept because they are part of the format COLMAP
//! writes: a `points3D.txt` line always has a space after the error, so a point with an empty
//! track ends "ERROR \n"; and an `images.txt` POINTS2D line keeps its trailing space (COLMAP's
//! `seekp(-1)` moves the put pointer, but `ostringstream::str()` still returns the space).

use std::io::Write;
use std::path::Path;

use crate::geometry::rigid3::Rigid3d;
use crate::scene::reconstruction::Reconstruction;
use crate::scene::reconstruction_io_utils::{open_write, stream_error};
use crate::util::stream_format::format_double;
use crate::Result;

/// "Ensure that we don't loose any precision by storing in text."
const TEXT_PRECISION: usize = 17;

fn num(value: f64) -> String {
    format_double(value, TEXT_PRECISION)
}

/// `QW QX QY QZ TX TY TZ ` (each followed by a space).
fn push_rigid3d(line: &mut String, rigid3d: &Rigid3d) {
    let q = rigid3d.rotation;
    let t = rigid3d.translation;
    for value in [q.w, q.x, q.y, q.z, t.x, t.y, t.z] {
        line.push_str(&num(value));
        line.push(' ');
    }
}

/// `line_string.substr(0, line_string.size() - 1)`: drops the trailing separator (an empty
/// line stays empty, as `substr(0, npos)` keeps it).
fn drop_last(mut line: String) -> String {
    line.pop();
    line
}

fn write_text<W: Write + ?Sized>(stream: &mut W, text: &str) -> Result<()> {
    stream.write_all(text.as_bytes()).map_err(stream_error)
}

fn write_text_path(
    reconstruction: &Reconstruction,
    path: &Path,
    write: fn(&Reconstruction, &mut dyn Write) -> Result<()>,
) -> Result<()> {
    let mut file = open_write(path)?;
    write(reconstruction, &mut file)?;
    file.flush().map_err(stream_error)
}

/// `WriteRigsText(reconstruction, stream)`.
pub fn write_rigs_text<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    let mut text = String::new();
    text.push_str("# Rig calib list with one line of data per calib:\n");
    text.push_str(
        "#   RIG_ID, NUM_SENSORS, REF_SENSOR_TYPE, REF_SENSOR_ID, SENSORS[] as (SENSOR_TYPE, \
         SENSOR_ID, HAS_POSE, [QW, QX, QY, QZ, TX, TY, TZ])\n",
    );
    text.push_str(&format!(
        "# Number of rigs: {}\n",
        reconstruction.num_rigs()
    ));

    for (&rig_id, rig) in reconstruction.rigs() {
        let mut line = format!("{rig_id} {} ", rig.num_sensors());
        if rig.num_sensors() > 0 {
            let ref_sensor_id = rig.ref_sensor_id();
            line.push_str(&format!(
                "{} {} ",
                ref_sensor_id.sensor_type, ref_sensor_id.id
            ));
        }
        for (sensor_id, sensor_from_rig) in rig.non_ref_sensors() {
            line.push_str(&format!("{} {} ", sensor_id.sensor_type, sensor_id.id));
            match sensor_from_rig {
                Some(sensor_from_rig) => {
                    line.push_str("1 ");
                    push_rigid3d(&mut line, sensor_from_rig);
                }
                None => line.push_str("0 "),
            }
        }
        text.push_str(&drop_last(line));
        text.push('\n');
    }
    write_text(stream, &text)
}

/// `WriteRigsText(reconstruction, path)`.
pub fn write_rigs_text_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    write_text_path(reconstruction, path, |r, s| write_rigs_text(r, s))
}

/// `WriteCamerasText(reconstruction, stream)`.
pub fn write_cameras_text<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    let mut text = String::new();
    text.push_str("# Camera list with one line of data per camera:\n");
    text.push_str("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n");
    text.push_str(&format!(
        "# Number of cameras: {}\n",
        reconstruction.num_cameras()
    ));

    for (&camera_id, camera) in reconstruction.cameras() {
        let mut line = format!(
            "{camera_id} {} {} {} ",
            camera.model_name(),
            camera.width,
            camera.height
        );
        for &param in &camera.params {
            line.push_str(&num(param));
            line.push(' ');
        }
        text.push_str(&drop_last(line));
        text.push('\n');
    }
    write_text(stream, &text)
}

/// `WriteCamerasText(reconstruction, path)`.
pub fn write_cameras_text_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    write_text_path(reconstruction, path, |r, s| write_cameras_text(r, s))
}

/// `WriteFramesText(reconstruction, stream)`: the posed frames only.
pub fn write_frames_text<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    let frames: Vec<_> = reconstruction
        .frames()
        .iter()
        .filter(|(_, frame)| frame.has_pose())
        .collect();

    let mut text = String::new();
    text.push_str("# Frame list with one line of data per frame:\n");
    text.push_str(
        "#   FRAME_ID, RIG_ID, RIG_FROM_WORLD[QW, QX, QY, QZ, TX, TY, TZ], NUM_DATA_IDS, \
         DATA_IDS[] as (SENSOR_TYPE, SENSOR_ID, DATA_ID)\n",
    );
    text.push_str(&format!("# Number of frames: {}\n", frames.len()));

    for (&frame_id, frame) in frames {
        text.push_str(&format!("{frame_id} {} ", frame.rig_id()));
        push_rigid3d(&mut text, &frame.rig_from_world()?);
        let data_ids = frame.data_ids();
        text.push_str(&data_ids.len().to_string());
        for data_id in data_ids {
            text.push_str(&format!(
                " {} {} {}",
                data_id.sensor_id.sensor_type, data_id.sensor_id.id, data_id.id
            ));
        }
        text.push('\n');
    }
    write_text(stream, &text)
}

/// `WriteFramesText(reconstruction, path)`.
pub fn write_frames_text_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    write_text_path(reconstruction, path, |r, s| write_frames_text(r, s))
}

/// `WriteImagesText(reconstruction, stream)`: two lines per image of the registered frames,
/// in registration order.
pub fn write_images_text<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    let mut text = String::new();
    text.push_str("# Image list with two lines of data per image:\n");
    text.push_str("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n");
    text.push_str("#   POINTS2D[] as (X, Y, POINT3D_ID)\n");
    text.push_str(&format!(
        "# Number of images: {}, mean observations per image: {}\n",
        reconstruction.num_reg_images(),
        num(reconstruction.compute_mean_observations_per_reg_image()?)
    ));

    for image_id in reconstruction.reg_image_ids()? {
        let image = reconstruction.image(image_id)?;
        let frame = reconstruction.frame(image.frame_id())?;
        let rig = reconstruction.rig(frame.rig_id())?;

        text.push_str(&format!("{image_id} "));
        push_rigid3d(&mut text, &image.cam_from_world(frame, rig)?);
        text.push_str(&format!("{} {}\n", image.camera_id(), image.name()));

        for point2d in image.points2d() {
            text.push_str(&num(point2d.xy.x));
            text.push(' ');
            text.push_str(&num(point2d.xy.y));
            text.push(' ');
            if point2d.has_point3d() {
                text.push_str(&format!("{} ", point2d.point3d_id));
            } else {
                text.push_str("-1 ");
            }
        }
        // COLMAP's `seekp(-1)` here leaves the trailing space in the output (see the header).
        text.push('\n');
    }
    write_text(stream, &text)
}

/// `WriteImagesText(reconstruction, path)`.
pub fn write_images_text_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    write_text_path(reconstruction, path, |r, s| write_images_text(r, s))
}

/// `WritePoints3DText(reconstruction, stream)`.
pub fn write_points3d_text<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    let mut text = String::new();
    text.push_str("# 3D point list with one line of data per point:\n");
    text.push_str("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n");
    text.push_str(&format!(
        "# Number of points: {}, mean track length: {}\n",
        reconstruction.num_points3d(),
        num(reconstruction.compute_mean_track_length()?)
    ));

    for (&point3d_id, point3d) in reconstruction.points3d() {
        text.push_str(&format!(
            "{point3d_id} {} {} {} {} {} {} {} ",
            num(point3d.xyz.x),
            num(point3d.xyz.y),
            num(point3d.xyz.z),
            point3d.color.x,
            point3d.color.y,
            point3d.color.z,
            num(point3d.error)
        ));

        let mut line = String::new();
        for track_el in point3d.track.elements() {
            line.push_str(&format!("{} {} ", track_el.image_id, track_el.point2d_idx));
        }
        text.push_str(&drop_last(line));
        text.push('\n');
    }
    write_text(stream, &text)
}

/// `WritePoints3DText(reconstruction, path)`.
pub fn write_points3d_text_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    write_text_path(reconstruction, path, |r, s| write_points3d_text(r, s))
}
