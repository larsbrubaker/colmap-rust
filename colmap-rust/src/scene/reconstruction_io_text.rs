//! Port of `colmap/scene/reconstruction_io_text.h` / `.cc`, part 1 (and colmap-sharp's
//! `Scene/ReconstructionIOText.cs`): the readers of COLMAP's text model files `rigs.txt`,
//! `cameras.txt`, `frames.txt`, `images.txt` and `points3D.txt`, on any `Read` stream or on
//! a path. The writers are [`super::reconstruction_io_text_write`]; the binary format is
//! [`super::reconstruction_io_binary`]. Tests: `tests/scene/reconstruction_io.rs`
//! (`reconstruction_io_test.cc` 1:1) and `tests/scene/rust_only_reconstruction_io.rs`.
//!
//! Lines are read with `std::getline` + `StringTrim` semantics, blank and '#' lines are
//! skipped, and values are parsed token by token in the classic locale
//! ([`crate::util::line_tokens`]; entry 241 for where a token-whole parse differs from
//! libc++'s character-wise one). A line that is not UTF-8 is an error (entry 241).
//! `THROW_CHECK(line_stream >> a >> b)` fails with "Check failed: line_stream >> value".
//! Tier A: correctly rounded parsing gives the bits COLMAP reads.

use std::io::Read;
use std::path::Path;

use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{Quaterniond, Vector2d, Vector3d, Vector3ub};
use crate::scene::camera::Camera;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::point3d::Point3D;
use crate::scene::reconstruction::Reconstruction;
use crate::scene::reconstruction_io_utils::{
    create_frame_for_image, create_one_rig_per_camera, extract_image_to_frame_id, open_read,
    stream_error,
};
use crate::sensor::models::{camera_model_name_to_id, camera_model_num_params};
use crate::sensor::rig::Rig;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::line_tokens::{read_lines, LineTokens};
use crate::util::string::string_trim;
use crate::util::types::{DataId, Point2DIdx, Point3DId, SensorId, SensorType, INVALID_POINT3D_ID};
use crate::Result;

/// A failed `THROW_CHECK(line_stream >> ...)`.
fn need<T>(value: Option<T>) -> Result<T> {
    value.ok_or_else(|| {
        ColmapError::new(
            ErrorKind::InvalidArgument,
            "Check failed: line_stream >> value",
        )
    })
}

/// The stream's lines, UTF-8 decoded and trimmed (`StringTrim`), blank ones kept.
fn trimmed_lines<R: Read + ?Sized>(stream: &mut R) -> Result<Vec<String>> {
    read_lines(stream)
        .map_err(stream_error)?
        .into_iter()
        .map(|bytes| {
            let mut line = String::from_utf8(bytes).map_err(|_| {
                ColmapError::new(
                    ErrorKind::RuntimeError,
                    "Reconstruction text line is not UTF-8",
                )
            })?;
            string_trim(&mut line);
            Ok(line)
        })
        .collect()
}

fn is_data_line(line: &str) -> bool {
    !line.is_empty() && !line.starts_with('#')
}

fn read_sensor_id(tokens: &mut LineTokens) -> Result<SensorId> {
    let item = need(tokens.read_string())?;
    let id = need(tokens.read_u32())?;
    Ok(SensorId::new(SensorType::from_colmap_str(&item)?, id))
}

/// Reads `QW QX QY QZ TX TY TZ`.
fn read_rigid3d(tokens: &mut LineTokens) -> Result<Rigid3d> {
    let mut values = [0.0; 7];
    for value in &mut values {
        *value = need(tokens.read_f64())?;
    }
    let [qw, qx, qy, qz, tx, ty, tz] = values;
    Ok(Rigid3d::new(
        Quaterniond::new(qw, qx, qy, qz),
        Vector3d::new(tx, ty, tz),
    ))
}

/// `ReadRigsText(reconstruction, stream)`.
pub fn read_rigs_text<R: Read + ?Sized>(
    reconstruction: &mut Reconstruction,
    stream: &mut R,
) -> Result<()> {
    for line in trimmed_lines(stream)? {
        if !is_data_line(&line) {
            continue;
        }
        let mut tokens = LineTokens::new(&line);
        let mut rig = Rig::new();

        // ID, NUM_SENSORS
        let rig_id = need(tokens.read_u32())?;
        let num_sensors = need(tokens.read_u64())?;
        rig.set_rig_id(rig_id);

        if num_sensors > 0 {
            // REF_SENSOR
            rig.add_ref_sensor(read_sensor_id(&mut tokens)?)?;
        }

        // SENSORS
        for _ in 1..num_sensors {
            let sensor_id = read_sensor_id(&mut tokens)?;
            let has_pose = need(tokens.read_i32())? == 1;
            let sensor_from_rig = if has_pose {
                Some(read_rigid3d(&mut tokens)?)
            } else {
                None
            };
            rig.add_sensor(sensor_id, sensor_from_rig)?;
        }

        reconstruction.add_rig(rig)?;
    }
    Ok(())
}

/// `ReadRigsText(reconstruction, path)`.
pub fn read_rigs_text_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_rigs_text(reconstruction, &mut open_read(path)?)
}

/// `ReadCamerasText(reconstruction, stream)`.
pub fn read_cameras_text<R: Read + ?Sized>(
    reconstruction: &mut Reconstruction,
    stream: &mut R,
) -> Result<()> {
    for line in trimmed_lines(stream)? {
        if !is_data_line(&line) {
            continue;
        }
        let mut tokens = LineTokens::new(&line);

        // ID, MODEL, WIDTH, HEIGHT
        let camera_id = need(tokens.read_u32())?;
        let item = need(tokens.read_string())?;
        // `size_t`, read as u64; widths above usize::MAX (wasm32) would truncate.
        let width = need(tokens.read_u64())? as usize;
        let height = need(tokens.read_u64())? as usize;
        let model_id = camera_model_name_to_id(&item);

        // PARAMS
        let mut params = Vec::with_capacity(camera_model_num_params(model_id)?);
        while let Some(param) = tokens.read_f64() {
            params.push(param);
        }

        let camera = Camera {
            camera_id,
            model_id,
            width,
            height,
            params,
            has_prior_focal_length: false,
        };
        crate::check!(camera.verify_params()?);
        reconstruction.add_camera(camera)?;
    }
    Ok(())
}

/// `ReadCamerasText(reconstruction, path)`.
pub fn read_cameras_text_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_cameras_text(reconstruction, &mut open_read(path)?)
}

/// `ReadFramesText(reconstruction, stream)`.
pub fn read_frames_text<R: Read + ?Sized>(
    reconstruction: &mut Reconstruction,
    stream: &mut R,
) -> Result<()> {
    for line in trimmed_lines(stream)? {
        if !is_data_line(&line) {
            continue;
        }
        let mut tokens = LineTokens::new(&line);
        let mut frame = Frame::new();

        // ID, RIG_ID
        let frame_id = need(tokens.read_u32())?;
        let rig_id = need(tokens.read_u32())?;
        frame.set_frame_id(frame_id);
        frame.set_rig_id(rig_id)?;

        // RIG_FROM_WORLD, DATA_IDS
        let rig_from_world = read_rigid3d(&mut tokens)?;
        let num_data_ids = need(tokens.read_u32())?;
        frame.set_rig_from_world(rig_from_world);
        for _ in 0..num_data_ids {
            let sensor_id = read_sensor_id(&mut tokens)?;
            let id = need(tokens.read_u64())?;
            frame.add_data_id(DataId { sensor_id, id })?;
        }

        reconstruction.add_frame(frame)?;
    }
    Ok(())
}

/// `ReadFramesText(reconstruction, path)`.
pub fn read_frames_text_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_frames_text(reconstruction, &mut open_read(path)?)
}

/// `ReadImagesText(reconstruction, stream)`: two lines per image. A model without rigs and
/// frames (written before COLMAP had them) gets one rig per camera and one frame per image.
pub fn read_images_text<R: Read + ?Sized>(
    reconstruction: &mut Reconstruction,
    stream: &mut R,
) -> Result<()> {
    // Handle backwards-compatibility for when we didn't have rigs and frames.
    let is_legacy_reconstruction =
        reconstruction.num_rigs() == 0 && reconstruction.num_frames() == 0;
    if is_legacy_reconstruction {
        create_one_rig_per_camera(reconstruction)?;
    }

    let image_to_frame = extract_image_to_frame_id(reconstruction)?;

    let lines = trimmed_lines(stream)?;
    let mut lines = lines.iter();
    while let Some(line) = lines.next() {
        if !is_data_line(line) {
            continue;
        }
        let mut tokens1 = LineTokens::new(line);

        // ID, CAM_FROM_WORLD, CAMERA_ID
        let image_id = need(tokens1.read_u32())?;
        let cam_from_world = read_rigid3d(&mut tokens1)?;
        let camera_id = need(tokens1.read_u32())?;

        let mut image = Image::new();
        image.set_image_id(image_id);
        image.set_camera_id(camera_id)?;

        if is_legacy_reconstruction {
            create_frame_for_image(&image, &cam_from_world, reconstruction)?;
            image.set_frame_id(image.image_id());
        } else {
            // `image_to_frame.at(image_id)`: std::out_of_range for an image in no frame.
            let frame_id = *image_to_frame.get(&image_id).ok_or_else(|| {
                ColmapError::new(
                    ErrorKind::OutOfRange,
                    format!("Frame for image with ID {image_id} does not exist"),
                )
            })?;
            image.set_frame_id(frame_id);
        }

        // NAME
        let name = need(tokens1.read_string())?;
        image.set_name(&name);

        // POINTS2D
        let Some(line) = lines.next() else {
            break;
        };
        let mut tokens2 = LineTokens::new(line);

        let mut points2d = Vec::new();
        let mut point3d_ids = Vec::new();
        while let Some(x) = tokens2.read_f64() {
            let Some(y) = tokens2.read_f64() else {
                break;
            };
            points2d.push(Vector2d::new(x, y));

            let point3d_id_signed = need(tokens2.read_i64())?;
            point3d_ids.push(if point3d_id_signed == -1 {
                INVALID_POINT3D_ID
            } else {
                // `static_cast<point3D_t>(int64_t)`: other negative ids wrap, as in C++.
                point3d_id_signed as Point3DId
            });
        }

        image.set_points2d_from_xy(&points2d);

        for (point2d_idx, &point3d_id) in point3d_ids.iter().enumerate() {
            if point3d_id != INVALID_POINT3D_ID {
                image.set_point3d_for_point2d(point2d_idx as Point2DIdx, point3d_id)?;
            }
        }

        reconstruction.add_image(image)?;
    }
    Ok(())
}

/// `ReadImagesText(reconstruction, path)`.
pub fn read_images_text_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_images_text(reconstruction, &mut open_read(path)?)
}

/// `ReadPoints3DText(reconstruction, stream)`.
pub fn read_points3d_text<R: Read + ?Sized>(
    reconstruction: &mut Reconstruction,
    stream: &mut R,
) -> Result<()> {
    for line in trimmed_lines(stream)? {
        if !is_data_line(&line) {
            continue;
        }
        let mut tokens = LineTokens::new(&line);

        // ID, XYZ, RGB
        let point3d_id = need(tokens.read_u64())?;
        let x = need(tokens.read_f64())?;
        let y = need(tokens.read_f64())?;
        let z = need(tokens.read_f64())?;
        let r = need(tokens.read_i32())?;
        let g = need(tokens.read_i32())?;
        let b = need(tokens.read_i32())?;

        let mut point3d = Point3D {
            xyz: Vector3d::new(x, y, z),
            // `static_cast<uint8_t>(int)` keeps the low byte.
            color: Vector3ub::new(r as u8, g as u8, b as u8),
            ..Point3D::default()
        };

        // ERROR
        point3d.error = need(tokens.read_f64())?;

        // TRACK
        while let Some(image_id) = tokens.read_u32() {
            let Some(point2d_idx) = tokens.read_u32() else {
                break;
            };
            point3d.track.add_element_ids(image_id, point2d_idx);
        }

        point3d.track.compress();

        reconstruction.add_point3d_with_id(point3d_id, point3d)?;
    }
    Ok(())
}

/// `ReadPoints3DText(reconstruction, path)`.
pub fn read_points3d_text_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_points3d_text(reconstruction, &mut open_read(path)?)
}
