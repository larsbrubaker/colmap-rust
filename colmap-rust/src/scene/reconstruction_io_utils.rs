//! Port of `colmap/scene/reconstruction_io_utils.h` / `.cc` (and colmap-sharp's
//! `Scene/ReconstructionIOUtils.cs`): the helpers the binary and text model readers share —
//! the one-rig-per-camera and one-frame-per-image setup for legacy models written before
//! rigs and frames existed, the image-to-frame lookup — plus the stream error plumbing of
//! [`super::reconstruction_io_binary`] and [`super::reconstruction_io_text`]. The file
//! openers live in [`crate::util::file`].
//! Tests: `tests/scene/reconstruction_io.rs`.
//!
//! Translation notes:
//! - `ExtractSortedIds` sorts a hash map's ids so the writers' output does not depend on the
//!   map; the reconstruction's maps are `BTreeMap`s (entry 213), whose keys already come in
//!   ascending order, so the writers iterate them directly.
//! - `ExtractImageToFramePtr` maps image ids to frame pointers; here to frame ids
//!   ([`extract_image_to_frame_id`], entry 170).

use std::collections::HashMap;
use std::path::Path;

use crate::geometry::rigid3::Rigid3d;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::reconstruction::{image_id_of, Reconstruction};
use crate::sensor::rig::Rig;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{FrameId, ImageId, SensorType};
use crate::Result;

/// `CreateOneRigPerCamera`: a rig of the camera's id with the camera as its only sensor,
/// for every camera.
pub fn create_one_rig_per_camera(reconstruction: &mut Reconstruction) -> Result<()> {
    let sensor_ids: Vec<_> = reconstruction
        .cameras()
        .iter()
        .map(|(&camera_id, camera)| (camera_id, camera.sensor_id()))
        .collect();
    for (camera_id, sensor_id) in sensor_ids {
        let mut rig = Rig::new();
        rig.set_rig_id(camera_id);
        rig.add_ref_sensor(sensor_id)?;
        reconstruction.add_rig(rig)?;
    }
    Ok(())
}

/// `CreateFrameForImage`: a frame of the image's id in its camera's trivial rig, holding
/// only the image, posed at `cam_from_world`.
pub fn create_frame_for_image(
    image: &Image,
    cam_from_world: &Rigid3d,
    reconstruction: &mut Reconstruction,
) -> Result<()> {
    let mut frame = Frame::new();
    frame.set_frame_id(image.image_id());
    frame.set_rig_id(image.camera_id())?;
    frame.add_data_id(image.data_id())?;
    frame.set_rig_from_world(*cam_from_world);
    reconstruction.add_frame(frame)
}

/// `ExtractImageToFramePtr`: the frame that holds each image, by id. An image in two frames
/// is an error. Used for lookups only, never iterated.
pub fn extract_image_to_frame_id(
    reconstruction: &Reconstruction,
) -> Result<HashMap<ImageId, FrameId>> {
    let mut image_to_frame = HashMap::new();
    for (&frame_id, frame) in reconstruction.frames() {
        for data_id in frame.image_ids() {
            crate::check!(image_to_frame
                .insert(image_id_of(data_id.id), frame_id)
                .is_none());
        }
    }
    Ok(image_to_frame)
}

/// `static_cast<SensorType>(value)` of a sensor type read from a binary file; a number that
/// names no enumerator is an error (entry 241).
pub(crate) fn sensor_type_from_i32(value: i32) -> Result<SensorType> {
    match value {
        -1 => Ok(SensorType::Invalid),
        0 => Ok(SensorType::Camera),
        1 => Ok(SensorType::Imu),
        _ => Err(ColmapError::new(
            ErrorKind::RuntimeError,
            format!("Unknown value: {value} for enum: SensorType"),
        )),
    }
}

/// The file openers moved to [`crate::util::file`] (the PLY readers in `util` use them too);
/// re-exported here for the readers that import them with the stream helpers below.
pub(crate) use crate::util::file::{open_read, open_write};

/// A stream read or write that failed part-way (COLMAP's unchecked stream state; entry 61).
pub(crate) fn stream_error(error: std::io::Error) -> ColmapError {
    ColmapError::new(
        ErrorKind::RuntimeError,
        format!("Reconstruction stream I/O failed: {error}"),
    )
}

/// `THROW_CHECK_DIR_EXISTS(path)`.
pub(crate) fn check_dir_exists(path: &Path) -> Result<()> {
    crate::check!(
        path.is_dir(),
        "Directory \"{}\" does not exist.",
        path.display()
    );
    Ok(())
}
