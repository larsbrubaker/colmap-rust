//! Port of the `ApplyRigConfig` half of `colmap/scene/rig.cc`, with its helpers
//! `UpdateRigAndCameraCalibsFromReconstruction`, `UpdateRigsAndFramesFromDatabase` and
//! `CopyCameraIntrinsics`: [`apply_rig_config`] replaces the rigs and frames of a
//! [`Database`] (and, if given, of a [`Reconstruction`]) with those of the configs.
//! `scene/rig.rs` holds the config type and its JSON reader. Port of colmap-sharp's
//! `ColmapSharp/Scene/RigConfig.Apply.cs`. Tests: `tests/scene/rig.rs` (`rig_test.cc` 1:1)
//! and `tests/scene/rust_only_rig.rs`.
//!
//! Ordering (CLAUDE.md, hash iteration order):
//! - COLMAP groups images per frame name in a `std::map<std::string>`, so frames are written
//!   in the names' byte order; a `BTreeMap<String, _>` orders the same (UTF-8 bytes).
//! - COLMAP hands the reconstruction its rigs and frames in `NodeHashMap` order, which sets
//!   the registration order of the frames and is implementation-defined per build and
//!   standard library. Here they go in ascending id order (docs/CPP_DIVERGENCES.md entry 276).
//! - The other hash maps are only looked up, never iterated.
//!
//! COLMAP's `LOG(INFO)` ("Configured: ...") and `LOG(WARNING)` (a non-reference camera whose
//! images are not registered) lines are dropped: the core has no logger, and the lines
//! are diagnostic only; the warning's case leaves the sensor's pose unknown, exactly as in COLMAP.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::geometry::pose::average_quaternions;
use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{Quaterniond, Vector3d};
use crate::scene::camera::Camera;
use crate::scene::database::Database;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::reconstruction::{image_id_of, Reconstruction};
use crate::scene::rig::RigConfig;
use crate::sensor::rig::Rig;
use crate::util::string::{string_get_after, string_starts_with};
use crate::util::types::{CameraId, DataId, FrameId, ImageId, RigId, SensorId, SensorType};
use crate::{check, check_eq, Result};

/// Port of `colmap::ApplyRigConfig`. Applies the given rig configuration to the database and
/// optionally derives camera rig extrinsics and intrinsics from the reconstruction, if not
/// defined in the config. If the reconstruction is provided, it is also updated with the
/// provided config and any previous rigs/frames are cleared and overwritten. Existing rigs
/// and frames in the database and reconstruction will be cleared. Any unspecified images in
/// the provided configurations will be configured as trivial rigs/frames.
pub fn apply_rig_config(
    configs: &[RigConfig],
    database: &Database,
    mut reconstruction: Option<&mut Reconstruction>,
) -> Result<()> {
    database.clear_frames()?;
    database.clear_rigs()?;

    let images = database.read_all_images()?;
    let mut configured_image_ids: BTreeSet<ImageId> = BTreeSet::new();

    for config in configs {
        let mut rig = Rig::new();

        let num_cameras = config.cameras.len();

        let mut camera_ids: Vec<Option<CameraId>> = vec![None; num_cameras];
        let mut frame_name_to_images: BTreeMap<String, Vec<&Image>> = BTreeMap::new();
        for image in &images {
            for (camera_idx, config_camera) in config.cameras.iter().enumerate() {
                if !string_starts_with(image.name(), &config_camera.image_prefix) {
                    continue;
                }
                let frame_name = string_get_after(image.name(), &config_camera.image_prefix);
                frame_name_to_images
                    .entry(frame_name)
                    .or_default()
                    .push(image);
                match camera_ids[camera_idx] {
                    Some(camera_id) => {
                        check_eq!(
                            camera_id,
                            image.camera_id(),
                            "Inconsistent cameras for images with prefix: {}. Consider setting \
                             --ImageReader.single_camera_per_folder during feature extraction \
                             or manually assign consistent camera_id's.",
                            config_camera.image_prefix
                        );
                    }
                    None => {
                        camera_ids[camera_idx] = Some(image.camera_id());
                        if let Some(config_intrinsics) = &config_camera.camera {
                            let mut database_camera = database.read_camera(image.camera_id())?;
                            copy_camera_intrinsics(config_intrinsics, &mut database_camera);
                            database.update_camera(&database_camera)?;
                            if let Some(reconstruction) = reconstruction.as_deref_mut() {
                                let reconstruction_camera =
                                    reconstruction.camera_mut(image.camera_id())?;
                                copy_camera_intrinsics(config_intrinsics, reconstruction_camera);
                            }
                        }
                    }
                }
            }
        }

        let mut unique_camera_ids: BTreeSet<CameraId> = BTreeSet::new();
        for (camera_idx, config_camera) in config.cameras.iter().enumerate() {
            check!(
                camera_ids[camera_idx].is_some(),
                "At least one image must exist for each camera in the rig"
            );
            let Some(mut camera_id) = camera_ids[camera_idx] else {
                continue; // Checked just above.
            };
            if !unique_camera_ids.insert(camera_id) {
                // Clone the camera, if multiple cameras in the rig share a camera.
                camera_id = database.write_camera(&database.read_camera(camera_id)?, false)?;
                camera_ids[camera_idx] = Some(camera_id);
            }
            let sensor_id = SensorId::new(SensorType::Camera, camera_id);
            if config_camera.ref_sensor {
                rig.add_ref_sensor(sensor_id)?;
            } else {
                rig.add_sensor(sensor_id, config_camera.cam_from_rig)?;
            }
        }

        let rig_id = database.write_rig(&rig, false)?;
        rig.set_rig_id(rig_id);

        for frame_images in frame_name_to_images.values() {
            let mut frame = Frame::new();
            frame.set_rig_id(rig.rig_id())?;
            for image in frame_images {
                let data_id = image.data_id();
                check!(
                    rig.has_sensor(data_id.sensor_id),
                    "{} must not contain Image(image_id={}, camera_id={}, name={})",
                    rig,
                    image.image_id(),
                    image.camera_id(),
                    image.name()
                );
                frame.add_data_id(data_id)?;
                configured_image_ids.insert(image.image_id());
            }
            database.write_frame(&frame, false)?;
        }

        if let Some(reconstruction) = reconstruction.as_deref() {
            update_rig_and_camera_calibs_from_reconstruction(
                reconstruction,
                &frame_name_to_images,
                &mut rig,
                database,
            )?;
        }
    }

    // Create trivial rigs/frames for images without configuration.
    // This is necessary because we clear rigs/frames above.
    // Lookups only, never iterated.
    let mut camera_to_rig_id: HashMap<CameraId, RigId> = HashMap::new();
    for image in &images {
        if configured_image_ids.contains(&image.image_id()) {
            continue;
        }
        let sensor_id = SensorId::new(SensorType::Camera, image.camera_id());
        let rig_id = match camera_to_rig_id.get(&image.camera_id()) {
            Some(&rig_id) => rig_id,
            None => {
                let mut rig = Rig::new();
                rig.add_ref_sensor(sensor_id)?;
                let rig_id = database.write_rig(&rig, false)?;
                camera_to_rig_id.insert(image.camera_id(), rig_id);
                rig_id
            }
        };
        let mut frame = Frame::new();
        frame.set_rig_id(rig_id)?;
        frame.add_data_id(DataId::new(sensor_id, image.image_id()))?;
        database.write_frame(&frame, false)?;
    }

    if let Some(reconstruction) = reconstruction {
        update_rigs_and_frames_from_database(database, reconstruction)?;
    }
    Ok(())
}

/// `image.HasPose()` for an image of the reconstruction: false without a frame.
fn image_has_pose(reconstruction: &Reconstruction, image: &Image) -> bool {
    image.has_pose(reconstruction.frames().get(&image.frame_id()))
}

/// `image.CamFromWorld()` for an image of the reconstruction, through its frame and rig.
fn image_cam_from_world(reconstruction: &Reconstruction, image: &Image) -> Result<Rigid3d> {
    let frame = reconstruction.frame(image.frame_id())?;
    let rig = reconstruction.rig(frame.rig_id())?;
    image.cam_from_world(frame, rig)
}

/// Port of `UpdateRigAndCameraCalibsFromReconstruction`. Update the database with extracted
/// rig and calibrations from the given reconstruction derived as follows:
///   * Compute the sensor_from_rig poses as the average of the relative poses between
///     registered sensors in the reconstruction.
///   * Set the camera calibration parameters from the first frame with an image of the
///     camera.
fn update_rig_and_camera_calibs_from_reconstruction(
    reconstruction: &Reconstruction,
    frame_name_to_images: &BTreeMap<String, Vec<&Image>>,
    rig: &mut Rig,
    database: &Database,
) -> Result<()> {
    // Lookups only, never iterated.
    let mut rig_from_cams: HashMap<CameraId, (Vec<Quaterniond>, Vector3d)> = HashMap::new();
    let mut updated_cameras: HashSet<CameraId> = HashSet::new();
    for images in frame_name_to_images.values() {
        let mut ref_image: Option<&Image> = None;
        for &image in images {
            if rig.is_ref_sensor(image.data_id().sensor_id) {
                ref_image = Some(image);
            }
        }

        let Some(ref_image) = ref_image else {
            continue;
        };

        let Some(rig_calib_ref_image) = reconstruction.find_image_with_name(ref_image.name())
        else {
            continue;
        };
        if !image_has_pose(reconstruction, rig_calib_ref_image) {
            continue;
        }

        let ref_cam_from_world = image_cam_from_world(reconstruction, rig_calib_ref_image)?;
        if updated_cameras.insert(rig_calib_ref_image.camera_id()) {
            let mut ref_camera = reconstruction
                .camera(rig_calib_ref_image.camera_id())?
                .clone();
            ref_camera.camera_id = ref_image.camera_id();
            database.update_camera(&ref_camera)?;
        }

        for &image in images {
            if image.camera_id() == ref_image.camera_id() {
                continue;
            }
            let Some(rig_calib_image) = reconstruction.find_image_with_name(image.name()) else {
                continue;
            };
            if !image_has_pose(reconstruction, rig_calib_image) {
                continue;
            }
            let rig_from_cam = ref_cam_from_world
                * image_cam_from_world(reconstruction, rig_calib_image)?.inverse();
            if updated_cameras.insert(rig_calib_image.camera_id()) {
                let mut camera = reconstruction.camera(rig_calib_image.camera_id())?.clone();
                camera.camera_id = image.camera_id();
                database.update_camera(&camera)?;
            }
            let (rig_from_cam_rotations, rig_from_cam_translation) = rig_from_cams
                .entry(image.camera_id())
                .or_insert_with(|| (Vec::new(), Vector3d::zeros()));
            if rig_from_cam_rotations.is_empty() {
                *rig_from_cam_translation = rig_from_cam.translation;
            } else {
                *rig_from_cam_translation += rig_from_cam.translation;
            }
            rig_from_cam_rotations.push(rig_from_cam.rotation);
        }
    }

    // Compute the average sensor_from_rig poses over all frames.
    for (sensor_id, sensor_from_rig) in rig.non_ref_sensors_mut().iter_mut() {
        if sensor_from_rig.is_some() {
            // Do not compute it for explicitly provided poses in the config.
            continue;
        }

        let Some((rig_from_cam_rotations, rig_from_cam_translation)) =
            rig_from_cams.get(&sensor_id.id)
        else {
            // COLMAP logs a warning: the camera's images are not registered in the given
            // reconstruction, so its sensor_from_rig cannot be derived.
            continue;
        };

        let weights = vec![1.0; rig_from_cam_rotations.len()];
        let rig_from_cam = Rigid3d::new(
            average_quaternions(rig_from_cam_rotations, &weights)?,
            *rig_from_cam_translation / rig_from_cam_rotations.len() as f64,
        );
        *sensor_from_rig = Some(rig_from_cam.inverse());
    }

    database.update_rig(rig)
}

/// Port of `UpdateRigsAndFramesFromDatabase`: rebuilds the reconstruction's rigs and frames
/// from the database's, matching images by name. Rigs and frames reach `SetRigsAndFrames` in
/// ascending id order (entry 276).
fn update_rigs_and_frames_from_database(
    database: &Database,
    reconstruction: &mut Reconstruction,
) -> Result<()> {
    let database_frames = database.read_all_frames()?;

    // Lookups only, never iterated.
    let mut database_rigs: HashMap<RigId, Rig> = HashMap::new();
    for rig in database.read_all_rigs()? {
        database_rigs.insert(rig.rig_id(), rig);
    }

    // Create O(1) lookup table from image names to images. The first image of a name wins,
    // as with NodeHashMap::emplace. Lookups only, never iterated.
    let mut image_name_to_image: HashMap<&str, &Image> = HashMap::new();
    for image in reconstruction.images().values() {
        image_name_to_image.entry(image.name()).or_insert(image);
    }

    // What each visitor needs of a (database frame, database image, reconstruction image)
    // triple, collected once in COLMAP's visiting order.
    struct FrameData {
        frame_id: FrameId,
        rig_id: RigId,
        database_sensor_id: SensorId,
        reconstruction_sensor_id: SensorId,
        reconstruction_data_id: DataId,
        cam_from_world: Option<Rigid3d>,
    }
    let mut frame_data = Vec::new();
    for database_frame in &database_frames {
        for data_id in database_frame.image_ids() {
            let database_image = database.read_image(image_id_of(data_id.id))?;
            let Some(&reconstruction_image) = image_name_to_image.get(database_image.name()) else {
                continue;
            };
            let cam_from_world = if image_has_pose(reconstruction, reconstruction_image) {
                Some(image_cam_from_world(reconstruction, reconstruction_image)?)
            } else {
                None
            };
            frame_data.push(FrameData {
                frame_id: database_frame.frame_id(),
                rig_id: database_frame.rig_id(),
                database_sensor_id: database_image.data_id().sensor_id,
                reconstruction_sensor_id: reconstruction
                    .camera(reconstruction_image.camera_id())?
                    .sensor_id(),
                reconstruction_data_id: reconstruction_image.data_id(),
                cam_from_world,
            });
        }
    }

    let database_rig = |rig_id: RigId| {
        database_rigs.get(&rig_id).ok_or_else(|| {
            crate::ColmapError::new(
                crate::ErrorKind::OutOfRange,
                format!("Rig with ID {rig_id} does not exist"),
            )
        })
    };

    let mut reconstruction_rigs: BTreeMap<RigId, Rig> = BTreeMap::new();

    // Update reference sensors in reconstruction rigs.
    // (must be done before updating the non-reference sensors).
    for data in &frame_data {
        let database_rig = database_rig(data.rig_id)?;
        let reconstruction_rig = rig_entry(&mut reconstruction_rigs, data.rig_id);
        if !reconstruction_rig.is_ref_sensor(data.reconstruction_sensor_id)
            && database_rig.is_ref_sensor(data.database_sensor_id)
        {
            reconstruction_rig.add_ref_sensor(data.reconstruction_sensor_id)?;
        }
    }

    // Update non-reference sensors in reconstruction rigs.
    for data in &frame_data {
        let database_rig = database_rig(data.rig_id)?;
        let reconstruction_rig = rig_entry(&mut reconstruction_rigs, data.rig_id);
        if !reconstruction_rig
            .non_ref_sensors()
            .contains_key(&data.reconstruction_sensor_id)
            && database_rig
                .non_ref_sensors()
                .contains_key(&data.database_sensor_id)
        {
            // `SensorFromRig` throws for a sensor without a pose, as COLMAP's does.
            reconstruction_rig.add_sensor(
                data.reconstruction_sensor_id,
                Some(database_rig.sensor_from_rig(data.database_sensor_id)?),
            )?;
        }
    }

    // Update reconstruction frames.
    let mut reconstruction_frames: BTreeMap<FrameId, Frame> = BTreeMap::new();
    for data in &frame_data {
        let database_rig = database_rig(data.rig_id)?;
        let reconstruction_frame = reconstruction_frames.entry(data.frame_id).or_default();
        reconstruction_frame.set_frame_id(data.frame_id);
        reconstruction_frame.set_rig_id(data.rig_id)?;
        reconstruction_frame.add_data_id(data.reconstruction_data_id)?;
        if let Some(cam_from_world) = data.cam_from_world {
            if database_rig.is_ref_sensor(data.database_sensor_id) {
                reconstruction_frame.set_rig_from_world(cam_from_world);
            } else {
                reconstruction_frame.set_rig_from_world(
                    database_rig
                        .sensor_from_rig(data.database_sensor_id)?
                        .inverse()
                        * cam_from_world,
                );
            }
        }
    }

    let rigs: Vec<Rig> = reconstruction_rigs.into_values().collect();
    let frames: Vec<Frame> = reconstruction_frames.into_values().collect();
    reconstruction.set_rigs_and_frames(rigs, frames)
}

/// `reconstruction_rigs[rig_id]` followed by `SetRigId(rig_id)`.
fn rig_entry(rigs: &mut BTreeMap<RigId, Rig>, rig_id: RigId) -> &mut Rig {
    let rig = rigs.entry(rig_id).or_default();
    rig.set_rig_id(rig_id);
    rig
}

/// Port of `CopyCameraIntrinsics`: the model, parameters and prior focal length flag.
fn copy_camera_intrinsics(src: &Camera, dst: &mut Camera) {
    dst.model_id = src.model_id;
    dst.params = src.params.clone();
    dst.has_prior_focal_length = src.has_prior_focal_length;
}
