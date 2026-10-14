//! Port of `colmap/scene/reconstruction_io_binary.h` / `.cc` (and colmap-sharp's
//! `Scene/ReconstructionIOBinary.cs`): COLMAP's binary model files `rigs.bin`,
//! `cameras.bin`, `frames.bin`, `images.bin` and `points3D.bin`, read into and written from a
//! [`Reconstruction`], on any `Read` / `Write` stream or on a path. Neighbors:
//! [`super::reconstruction_io_text`] (the text format), [`super::reconstruction_io_utils`]
//! (shared helpers). Tests: `tests/scene/reconstruction_io.rs` (`reconstruction_io_test.cc`
//! 1:1) and `tests/scene/rust_only_reconstruction_io.rs` (byte-exact against
//! pycolmap's files).
//!
//! Tier A: the bytes are COLMAP's, little-endian, in ascending id order (images in
//! registration order, as `RegImageIds`).
//!
//! Translation notes:
//! - A stream that ends early is an error here; COLMAP reads unspecified values (entry 61).
//! - A sensor type or camera model number that names no enumerator, and an image name that
//!   is not UTF-8, are errors (a Rust enum or `String` cannot hold them; entry 241).
//! - Widths and heights are `u64` in the file and `usize` in [`crate::scene::camera::Camera`];
//!   on wasm32 a value above `u32::MAX` would truncate, and none is a real image size.

use std::io::{Read, Write};
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
    open_write, sensor_type_from_i32, stream_error,
};
use crate::sensor::models::{camera_model_num_params, CameraModelId};
use crate::sensor::rig::Rig;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::endian::{read_binary_little_endian, write_binary_little_endian, EndianValue};
use crate::util::types::{
    CameraId, DataId, FrameId, ImageId, Point2DIdx, Point3DId, RigId, SensorId, INVALID_POINT3D_ID,
};
use crate::Result;

fn read<T: EndianValue, R: Read + ?Sized>(stream: &mut R) -> Result<T> {
    read_binary_little_endian(stream).map_err(stream_error)
}

fn write<T: EndianValue, W: Write + ?Sized>(stream: &mut W, value: T) -> Result<()> {
    write_binary_little_endian(stream, value).map_err(stream_error)
}

fn read_sensor_id<R: Read + ?Sized>(stream: &mut R) -> Result<SensorId> {
    let sensor_type = sensor_type_from_i32(read::<i32, _>(stream)?)?;
    let id = read::<u32, _>(stream)?;
    Ok(SensorId::new(sensor_type, id))
}

fn write_sensor_id<W: Write + ?Sized>(stream: &mut W, sensor_id: SensorId) -> Result<()> {
    write::<i32, _>(stream, sensor_id.sensor_type as i32)?;
    write::<u32, _>(stream, sensor_id.id)
}

/// Reads `qw qx qy qz tx ty tz`.
fn read_rigid3d<R: Read + ?Sized>(stream: &mut R) -> Result<Rigid3d> {
    let mut values = [0.0; 7];
    for value in &mut values {
        *value = read::<f64, _>(stream)?;
    }
    let [qw, qx, qy, qz, tx, ty, tz] = values;
    Ok(Rigid3d::new(
        Quaterniond::new(qw, qx, qy, qz),
        Vector3d::new(tx, ty, tz),
    ))
}

/// Writes `qw qx qy qz tx ty tz`.
fn write_rigid3d<W: Write + ?Sized>(stream: &mut W, rigid3d: &Rigid3d) -> Result<()> {
    let q = rigid3d.rotation;
    let t = rigid3d.translation;
    for value in [q.w, q.x, q.y, q.z, t.x, t.y, t.z] {
        write::<f64, _>(stream, value)?;
    }
    Ok(())
}

/// `ReadRigsBinary(reconstruction, stream)`.
pub fn read_rigs_binary<R: Read + ?Sized>(
    reconstruction: &mut Reconstruction,
    stream: &mut R,
) -> Result<()> {
    let num_rigs = read::<u64, _>(stream)?;
    for _ in 0..num_rigs {
        let mut rig = Rig::new();
        rig.set_rig_id(read::<RigId, _>(stream)?);
        let num_sensors = read::<u32, _>(stream)?;

        if num_sensors > 0 {
            rig.add_ref_sensor(read_sensor_id(stream)?)?;
        }

        for _ in 1..num_sensors {
            let sensor_id = read_sensor_id(stream)?;
            let has_pose = read::<u8, _>(stream)? != 0;
            let sensor_from_rig = if has_pose {
                Some(read_rigid3d(stream)?)
            } else {
                None
            };
            rig.add_sensor(sensor_id, sensor_from_rig)?;
        }

        reconstruction.add_rig(rig)?;
    }
    Ok(())
}

/// `ReadRigsBinary(reconstruction, path)`.
pub fn read_rigs_binary_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_rigs_binary(reconstruction, &mut open_read(path)?)
}

/// `ReadCamerasBinary(reconstruction, stream)`.
pub fn read_cameras_binary<R: Read + ?Sized>(
    reconstruction: &mut Reconstruction,
    stream: &mut R,
) -> Result<()> {
    let num_cameras = read::<u64, _>(stream)?;
    for _ in 0..num_cameras {
        let camera_id = read::<CameraId, _>(stream)?;
        let model_value = read::<i32, _>(stream)?;
        let model_id = CameraModelId::from_i32(model_value).ok_or_else(|| {
            ColmapError::new(
                ErrorKind::DomainError,
                format!("Camera model with ID {model_value} does not exist"),
            )
        })?;
        let width = read::<u64, _>(stream)? as usize;
        let height = read::<u64, _>(stream)? as usize;
        let mut params = vec![0.0; camera_model_num_params(model_id)?];
        for param in &mut params {
            *param = read::<f64, _>(stream)?;
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

/// `ReadCamerasBinary(reconstruction, path)`.
pub fn read_cameras_binary_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_cameras_binary(reconstruction, &mut open_read(path)?)
}

/// `ReadFramesBinary(reconstruction, stream)`.
pub fn read_frames_binary<R: Read + ?Sized>(
    reconstruction: &mut Reconstruction,
    stream: &mut R,
) -> Result<()> {
    let num_frames = read::<u64, _>(stream)?;
    for _ in 0..num_frames {
        let mut frame = Frame::new();
        frame.set_frame_id(read::<FrameId, _>(stream)?);
        frame.set_rig_id(read::<RigId, _>(stream)?)?;
        frame.set_rig_from_world(read_rigid3d(stream)?);

        let num_data_ids = read::<u32, _>(stream)?;
        for _ in 0..num_data_ids {
            let sensor_id = read_sensor_id(stream)?;
            let id = read::<u64, _>(stream)?;
            frame.add_data_id(DataId { sensor_id, id })?;
        }

        reconstruction.add_frame(frame)?;
    }
    Ok(())
}

/// `ReadFramesBinary(reconstruction, path)`.
pub fn read_frames_binary_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_frames_binary(reconstruction, &mut open_read(path)?)
}

/// `ReadImagesBinary(reconstruction, stream)`. A model without rigs and frames (written
/// before COLMAP had them) gets one rig per camera and one frame per image.
pub fn read_images_binary<R: Read + ?Sized>(
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

    let num_reg_images = read::<u64, _>(stream)?;
    for _ in 0..num_reg_images {
        let mut image = Image::new();
        image.set_image_id(read::<ImageId, _>(stream)?);
        let cam_from_world = read_rigid3d(stream)?;
        image.set_camera_id(read::<CameraId, _>(stream)?)?;

        if is_legacy_reconstruction {
            create_frame_for_image(&image, &cam_from_world, reconstruction)?;
            image.set_frame_id(image.image_id());
        } else {
            // `image_to_frame.at(image_id)`: std::out_of_range for an image in no frame.
            let frame_id = *image_to_frame.get(&image.image_id()).ok_or_else(|| {
                ColmapError::new(
                    ErrorKind::OutOfRange,
                    format!(
                        "Frame for image with ID {} does not exist",
                        image.image_id()
                    ),
                )
            })?;
            image.set_frame_id(frame_id);
        }

        let mut name = Vec::new();
        loop {
            let name_char = read::<u8, _>(stream)?;
            if name_char == 0 {
                break;
            }
            name.push(name_char);
        }
        let name = String::from_utf8(name).map_err(|_| {
            ColmapError::new(
                ErrorKind::RuntimeError,
                format!("Image {} has a name that is not UTF-8", image.image_id()),
            )
        })?;
        image.set_name(&name);

        let num_points2d = read::<u64, _>(stream)?;
        let mut points2d = Vec::new();
        let mut point3d_ids = Vec::new();
        for _ in 0..num_points2d {
            let x = read::<f64, _>(stream)?;
            let y = read::<f64, _>(stream)?;
            points2d.push(Vector2d::new(x, y));
            point3d_ids.push(read::<Point3DId, _>(stream)?);
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

/// `ReadImagesBinary(reconstruction, path)`.
pub fn read_images_binary_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_images_binary(reconstruction, &mut open_read(path)?)
}

/// `ReadPoints3DBinary(reconstruction, stream)`.
pub fn read_points3d_binary<R: Read + ?Sized>(
    reconstruction: &mut Reconstruction,
    stream: &mut R,
) -> Result<()> {
    let num_points3d = read::<u64, _>(stream)?;
    for _ in 0..num_points3d {
        let mut point3d = Point3D::default();
        let point3d_id = read::<Point3DId, _>(stream)?;

        let x = read::<f64, _>(stream)?;
        let y = read::<f64, _>(stream)?;
        let z = read::<f64, _>(stream)?;
        point3d.xyz = Vector3d::new(x, y, z);
        let r = read::<u8, _>(stream)?;
        let g = read::<u8, _>(stream)?;
        let b = read::<u8, _>(stream)?;
        point3d.color = Vector3ub::new(r, g, b);
        point3d.error = read::<f64, _>(stream)?;

        let track_length = read::<u64, _>(stream)?;
        for _ in 0..track_length {
            let image_id = read::<ImageId, _>(stream)?;
            let point2d_idx = read::<Point2DIdx, _>(stream)?;
            point3d.track.add_element_ids(image_id, point2d_idx);
        }
        point3d.track.compress();

        reconstruction.add_point3d_with_id(point3d_id, point3d)?;
    }
    Ok(())
}

/// `ReadPoints3DBinary(reconstruction, path)`.
pub fn read_points3d_binary_path(reconstruction: &mut Reconstruction, path: &Path) -> Result<()> {
    read_points3d_binary(reconstruction, &mut open_read(path)?)
}

/// `WriteRigsBinary(reconstruction, stream)`.
pub fn write_rigs_binary<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    write::<u64, _>(stream, reconstruction.num_rigs() as u64)?;
    for (&rig_id, rig) in reconstruction.rigs() {
        write::<RigId, _>(stream, rig_id)?;
        write::<u32, _>(stream, rig.num_sensors() as u32)?;
        if rig.num_sensors() > 0 {
            write_sensor_id(stream, rig.ref_sensor_id())?;
        }
        for (&sensor_id, sensor_from_rig) in rig.non_ref_sensors() {
            write_sensor_id(stream, sensor_id)?;
            write::<u8, _>(stream, u8::from(sensor_from_rig.is_some()))?;
            if let Some(sensor_from_rig) = sensor_from_rig {
                write_rigid3d(stream, sensor_from_rig)?;
            }
        }
    }
    Ok(())
}

/// `WriteRigsBinary(reconstruction, path)`.
pub fn write_rigs_binary_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    let mut file = open_write(path)?;
    write_rigs_binary(reconstruction, &mut file)?;
    file.flush().map_err(stream_error)
}

/// `WriteCamerasBinary(reconstruction, stream)`.
pub fn write_cameras_binary<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    write::<u64, _>(stream, reconstruction.num_cameras() as u64)?;
    for (&camera_id, camera) in reconstruction.cameras() {
        write::<CameraId, _>(stream, camera_id)?;
        write::<i32, _>(stream, camera.model_id as i32)?;
        write::<u64, _>(stream, camera.width as u64)?;
        write::<u64, _>(stream, camera.height as u64)?;
        for &param in &camera.params {
            write::<f64, _>(stream, param)?;
        }
    }
    Ok(())
}

/// `WriteCamerasBinary(reconstruction, path)`.
pub fn write_cameras_binary_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    let mut file = open_write(path)?;
    write_cameras_binary(reconstruction, &mut file)?;
    file.flush().map_err(stream_error)
}

/// `WriteFramesBinary(reconstruction, stream)`: the posed frames only.
pub fn write_frames_binary<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    let frames: Vec<(&FrameId, &Frame)> = reconstruction
        .frames()
        .iter()
        .filter(|(_, frame)| frame.has_pose())
        .collect();

    write::<u64, _>(stream, frames.len() as u64)?;
    for (&frame_id, frame) in frames {
        write::<FrameId, _>(stream, frame_id)?;
        write::<RigId, _>(stream, frame.rig_id())?;
        write_rigid3d(stream, &frame.rig_from_world()?)?;

        let data_ids = frame.data_ids();
        write::<u32, _>(stream, data_ids.len() as u32)?;
        for data_id in data_ids {
            write_sensor_id(stream, data_id.sensor_id)?;
            write::<u64, _>(stream, data_id.id)?;
        }
    }
    Ok(())
}

/// `WriteFramesBinary(reconstruction, path)`.
pub fn write_frames_binary_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    let mut file = open_write(path)?;
    write_frames_binary(reconstruction, &mut file)?;
    file.flush().map_err(stream_error)
}

/// `WriteImagesBinary(reconstruction, stream)`: the images of the registered frames, in
/// registration order.
pub fn write_images_binary<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    write::<u64, _>(stream, reconstruction.num_reg_images() as u64)?;
    for image_id in reconstruction.reg_image_ids()? {
        let image = reconstruction.image(image_id)?;
        write::<ImageId, _>(stream, image_id)?;

        let frame = reconstruction.frame(image.frame_id())?;
        let rig = reconstruction.rig(frame.rig_id())?;
        write_rigid3d(stream, &image.cam_from_world(frame, rig)?)?;

        write::<CameraId, _>(stream, image.camera_id())?;

        stream
            .write_all(image.name().as_bytes())
            .map_err(stream_error)?;
        write::<u8, _>(stream, 0)?;

        write::<u64, _>(stream, u64::from(image.num_points2d()))?;
        for point2d in image.points2d() {
            write::<f64, _>(stream, point2d.xy.x)?;
            write::<f64, _>(stream, point2d.xy.y)?;
            write::<Point3DId, _>(stream, point2d.point3d_id)?;
        }
    }
    Ok(())
}

/// `WriteImagesBinary(reconstruction, path)`.
pub fn write_images_binary_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    let mut file = open_write(path)?;
    write_images_binary(reconstruction, &mut file)?;
    file.flush().map_err(stream_error)
}

/// `WritePoints3DBinary(reconstruction, stream)`.
pub fn write_points3d_binary<W: Write + ?Sized>(
    reconstruction: &Reconstruction,
    stream: &mut W,
) -> Result<()> {
    write::<u64, _>(stream, reconstruction.num_points3d() as u64)?;
    for (&point3d_id, point3d) in reconstruction.points3d() {
        write::<Point3DId, _>(stream, point3d_id)?;
        write::<f64, _>(stream, point3d.xyz.x)?;
        write::<f64, _>(stream, point3d.xyz.y)?;
        write::<f64, _>(stream, point3d.xyz.z)?;
        write::<u8, _>(stream, point3d.color.x)?;
        write::<u8, _>(stream, point3d.color.y)?;
        write::<u8, _>(stream, point3d.color.z)?;
        write::<f64, _>(stream, point3d.error)?;

        write::<u64, _>(stream, point3d.track.length() as u64)?;
        for track_el in point3d.track.elements() {
            write::<ImageId, _>(stream, track_el.image_id)?;
            write::<Point2DIdx, _>(stream, track_el.point2d_idx)?;
        }
    }
    Ok(())
}

/// `WritePoints3DBinary(reconstruction, path)`.
pub fn write_points3d_binary_path(reconstruction: &Reconstruction, path: &Path) -> Result<()> {
    let mut file = open_write(path)?;
    write_points3d_binary(reconstruction, &mut file)?;
    file.flush().map_err(stream_error)
}
