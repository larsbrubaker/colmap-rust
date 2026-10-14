//! The implementation-independent functions of COLMAP's `scene/database.cc` —
//! `Database::Merge` and `LoadRandomDatabaseDescriptors` — plus the keypoint / match blob
//! conversions COLMAP keeps in `database_sqlite.cc` (`FeatureKeypointsToBlob` / `FromBlob`,
//! `SwapFeatureMatchesBlob`), which the store in [`super::database`] /
//! [`super::database_write`] uses. Port of colmap-sharp's `Scene/Database.Merge.cs`.
//! Tests: `tests/scene/database.rs`.
//!
//! Tier A (exact). `load_random_database_descriptors` draws through [`RandomSampler`]
//! (mt19937), like COLMAP. The id maps are only looked up, never iterated.

use std::collections::HashMap;

use crate::feature::types::{
    FeatureDescriptorsFloat, FeatureDescriptorsFloatData, FeatureExtractorType, FeatureKeypoint,
    FeatureKeypoints,
};
use crate::geometry::pose_prior::PosePrior;
use crate::linalg::RowMajorMatrix;
use crate::optim::random_sampler::RandomSampler;
use crate::optim::Sampler;
use crate::scene::frame::Frame;
use crate::sensor::rig::Rig;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{pair_id_to_image_pair, DataId, SensorType};
use crate::{check, check_eq, check_le, check_lt, check_ne, fatal_throw, Result};

use super::database::Database;

/// `FeatureKeypointsBlob`: N x 2, 4 or 6 floats per keypoint (row-major).
pub type FeatureKeypointsBlob = RowMajorMatrix<f32>;

/// `FeatureMatchesBlob`: N x 2 point2D indices (row-major).
pub type FeatureMatchesBlob = RowMajorMatrix<u32>;

/// Port of `FeatureKeypointsToBlob`: N x 6 `[x, y, a11, a12, a21, a22]`.
pub fn feature_keypoints_to_blob(keypoints: &[FeatureKeypoint]) -> FeatureKeypointsBlob {
    let mut blob = FeatureKeypointsBlob::new(keypoints.len(), 6);
    for (i, keypoint) in keypoints.iter().enumerate() {
        blob[(i, 0)] = keypoint.x;
        blob[(i, 1)] = keypoint.y;
        blob[(i, 2)] = keypoint.a11;
        blob[(i, 3)] = keypoint.a12;
        blob[(i, 4)] = keypoint.a21;
        blob[(i, 5)] = keypoint.a22;
    }
    blob
}

/// Port of `FeatureKeypointsFromBlob`: keypoints from an N x 2 (location), 4 (location,
/// scale, orientation) or 6 (location, affine shape) blob. Any other width fails.
pub fn feature_keypoints_from_blob(blob: &FeatureKeypointsBlob) -> Result<FeatureKeypoints> {
    let rows = blob.rows();
    if rows == 0 {
        return Ok(FeatureKeypoints::new());
    }
    match blob.cols() {
        2 => Ok((0..rows)
            .map(|i| FeatureKeypoint::new(blob[(i, 0)], blob[(i, 1)]))
            .collect()),
        4 => (0..rows)
            .map(|i| {
                FeatureKeypoint::from_scale_orientation(
                    blob[(i, 0)],
                    blob[(i, 1)],
                    blob[(i, 2)],
                    blob[(i, 3)],
                )
            })
            .collect(),
        6 => Ok((0..rows)
            .map(|i| {
                FeatureKeypoint::from_affine(
                    blob[(i, 0)],
                    blob[(i, 1)],
                    blob[(i, 2)],
                    blob[(i, 3)],
                    blob[(i, 4)],
                    blob[(i, 5)],
                )
            })
            .collect()),
        _ => fatal_throw!("Keypoint format not supported"),
    }
}

/// Port of `SwapFeatureMatchesBlob`: swaps the two columns in place.
pub fn swap_feature_matches_blob(matches: &mut FeatureMatchesBlob) {
    for i in 0..matches.rows() {
        let row = matches.row_mut(i);
        row.swap(0, 1);
    }
}

/// `std::unordered_map::at`: `std::out_of_range` for a missing key.
fn at(map: &HashMap<u32, u32>, key: u32) -> Result<u32> {
    map.get(&key)
        .copied()
        .ok_or_else(|| ColmapError::new(ErrorKind::OutOfRange, "unordered_map::at: key not found"))
}

/// `at` for a camera data id, whose `u64` id holds an image id.
fn at_image(map: &HashMap<u32, u32>, id: u64) -> Result<u64> {
    let key = u32::try_from(id)
        .map_err(|_| ColmapError::new(ErrorKind::OutOfRange, "unordered_map::at: key not found"))?;
    at(map, key).map(u64::from)
}

/// The "Data type not supported" error of `Merge` (`std::runtime_error`).
fn unsupported_data_type(sensor_type: SensorType) -> ColmapError {
    ColmapError::new(
        ErrorKind::RuntimeError,
        format!("Data type not supported: {sensor_type}"),
    )
}

/// `Merge`'s `update_rig`: the rig with its camera sensors renumbered.
fn update_rig(rig: &Rig, new_camera_ids: &HashMap<u32, u32>) -> Result<Rig> {
    if rig.num_sensors() == 0 {
        return Ok(rig.clone());
    }
    let mut updated_rig = Rig::new();
    updated_rig.set_rig_id(rig.rig_id());
    let mut ref_sensor_id = rig.ref_sensor_id();
    if ref_sensor_id.sensor_type == SensorType::Camera {
        ref_sensor_id.id = at(new_camera_ids, ref_sensor_id.id)?;
    }
    updated_rig.add_ref_sensor(ref_sensor_id)?;
    for (&sensor_id, &sensor_from_rig) in rig.non_ref_sensors() {
        let mut updated_sensor_id = sensor_id;
        if sensor_id.sensor_type == SensorType::Camera {
            updated_sensor_id.id = at(new_camera_ids, sensor_id.id)?;
        }
        updated_rig.add_sensor(updated_sensor_id, sensor_from_rig)?;
    }
    Ok(updated_rig)
}

/// `Merge`'s camera-data renumbering of frames and pose priors.
fn update_data_id(
    data_id: DataId,
    new_camera_ids: &HashMap<u32, u32>,
    new_image_ids: &HashMap<u32, u32>,
) -> Result<DataId> {
    match data_id.sensor_id.sensor_type {
        SensorType::Camera => {
            let mut updated = data_id;
            updated.id = at_image(new_image_ids, data_id.id)?;
            updated.sensor_id.id = at(new_camera_ids, data_id.sensor_id.id)?;
            Ok(updated)
        }
        other => Err(unsupported_data_type(other)),
    }
}

/// The new ids of one source database's objects in the merged database.
struct NewIds {
    cameras: HashMap<u32, u32>,
    rigs: HashMap<u32, u32>,
    images: HashMap<u32, u32>,
}

impl Database {
    /// Port of `Database::Merge`: writes both databases into `merged_database` under new ids
    /// (cameras, rigs, images with their keypoints and descriptors, frames, pose priors,
    /// matches, two-view geometries, in that order). Image names must be unique across the
    /// two databases.
    pub fn merge(
        database1: &Database,
        database2: &Database,
        merged_database: &Database,
    ) -> Result<()> {
        let sources = [database1, database2];
        let mut new_ids = Vec::with_capacity(2);

        // Merge the cameras.
        for database in sources {
            let mut cameras = HashMap::new();
            for camera in database.read_all_cameras()? {
                cameras.insert(
                    camera.camera_id,
                    merged_database.write_camera(&camera, false)?,
                );
            }
            new_ids.push(NewIds {
                cameras,
                rigs: HashMap::new(),
                images: HashMap::new(),
            });
        }

        // Merge the rigs.
        for (database, ids) in sources.iter().zip(new_ids.iter_mut()) {
            for rig in database.read_all_rigs()? {
                let new_rig_id =
                    merged_database.write_rig(&update_rig(&rig, &ids.cameras)?, false)?;
                ids.rigs.insert(rig.rig_id(), new_rig_id);
            }
        }

        // Merge the images.
        for (database, ids) in sources.iter().zip(new_ids.iter_mut()) {
            for mut image in database.read_all_images()? {
                let image_id = image.image_id();
                image.set_camera_id(at(&ids.cameras, image.camera_id())?)?;
                image.set_frame_id(crate::util::types::INVALID_FRAME_ID);
                check!(
                    !merged_database.exists_image_with_name(image.name())?,
                    "The two databases must not contain images with the same name, but there \
                     are images with name {} in both databases",
                    image.name()
                );
                let new_image_id = merged_database.write_image(&image, false)?;
                ids.images.insert(image_id, new_image_id);
                let keypoints = database.read_keypoints(image_id)?;
                let descriptors = database.read_descriptors(image_id)?;
                merged_database.write_keypoints(new_image_id, &keypoints)?;
                merged_database.write_descriptors(new_image_id, &descriptors)?;
            }
        }

        // Merge the frames.
        for (database, ids) in sources.iter().zip(new_ids.iter()) {
            for frame in database.read_all_frames()? {
                let mut updated_frame: Frame = frame.clone();
                updated_frame.set_rig_id(at(&ids.rigs, frame.rig_id())?)?;
                updated_frame.clear_data_ids()?;
                for &data_id in frame.data_ids() {
                    updated_frame.add_data_id(update_data_id(
                        data_id,
                        &ids.cameras,
                        &ids.images,
                    )?)?;
                }
                merged_database.write_frame(&updated_frame, false)?;
            }
        }

        // Merge the pose priors.
        for (database, ids) in sources.iter().zip(new_ids.iter()) {
            for pose_prior in database.read_all_pose_priors()? {
                let updated_pose_prior = PosePrior {
                    corr_data_id: update_data_id(
                        pose_prior.corr_data_id,
                        &ids.cameras,
                        &ids.images,
                    )?,
                    ..pose_prior
                };
                merged_database.write_pose_prior(&updated_pose_prior, false)?;
            }
        }

        // Merge the matches.
        for (database, ids) in sources.iter().zip(new_ids.iter()) {
            for (pair_id, matches) in database.read_all_matches()? {
                let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
                merged_database.write_matches(
                    at(&ids.images, image_id1)?,
                    at(&ids.images, image_id2)?,
                    &matches,
                )?;
            }
        }

        // Merge the two-view geometries.
        for (database, ids) in sources.iter().zip(new_ids.iter()) {
            for (pair_id, two_view_geometry) in database.read_two_view_geometries()? {
                let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
                merged_database.write_two_view_geometry(
                    at(&ids.images, image_id1)?,
                    at(&ids.images, image_id2)?,
                    &two_view_geometry,
                )?;
            }
        }
        Ok(())
    }
}

/// Port of `LoadRandomDatabaseDescriptors`: up to `max_num_descriptors` descriptors drawn
/// uniformly at random over all images (all of them if negative or at least the total), as
/// float, in database order. All images must share one feature type and width.
pub fn load_random_database_descriptors(
    database: &Database,
    max_num_descriptors: i32,
) -> Result<FeatureDescriptorsFloat> {
    let images = database.read_all_images()?;
    let total_num_descriptors = database.num_descriptors()?;

    if total_num_descriptors == 0 {
        return Ok(FeatureDescriptorsFloat::default());
    }

    let mut result = FeatureDescriptorsFloat {
        extractor_type: FeatureExtractorType::Undefined,
        ..FeatureDescriptorsFloat::default()
    };

    let mut descriptor_idxs = Vec::new();
    if max_num_descriptors < 0 || max_num_descriptors as usize >= total_num_descriptors {
        descriptor_idxs.extend(0..total_num_descriptors);
    } else {
        // Random subset of images in the database.
        check_le!(max_num_descriptors as usize, total_num_descriptors);
        let mut random_sampler = RandomSampler::new(max_num_descriptors as usize);
        random_sampler.initialize(total_num_descriptors)?;
        random_sampler.sample(&mut descriptor_idxs)?;
        // Distinct indices: no ties, so the unstable sort is deterministic.
        descriptor_idxs.sort_unstable();
    }

    let mut image_idx: usize = 0;
    let mut image_descriptors = FeatureDescriptorsFloat::default();
    let mut image_descriptor_start = 0;
    let mut image_descriptor_end = 0;

    let mut descriptor_row = 0;
    for &descriptor_idx in &descriptor_idxs {
        while descriptor_idx >= image_descriptor_end {
            // read_next_image.
            check_lt!(image_idx, images.len());
            image_descriptors = database
                .read_descriptors(images[image_idx].image_id())?
                .to_float()?;
            image_idx += 1;
            image_descriptor_start = image_descriptor_end;
            image_descriptor_end = image_descriptor_start + image_descriptors.data.rows();
        }

        // Check that all images have the same feature type.
        if result.extractor_type == FeatureExtractorType::Undefined {
            check_ne!(
                image_descriptors.extractor_type,
                FeatureExtractorType::Undefined
            );
            result.extractor_type = image_descriptors.extractor_type;
            result.data = FeatureDescriptorsFloatData::new(
                descriptor_idxs.len(),
                image_descriptors.data.cols(),
            );
        } else {
            check_eq!(
                result.extractor_type,
                image_descriptors.extractor_type,
                "All images must have the same feature type"
            );
            check_eq!(
                image_descriptors.data.cols(),
                result.data.cols(),
                "All images must have the same descriptor dimensionality"
            );
        }

        result.data.row_mut(descriptor_row).copy_from_slice(
            image_descriptors
                .data
                .row(descriptor_idx - image_descriptor_start),
        );
        descriptor_row += 1;
    }

    check_eq!(descriptor_row, descriptor_idxs.len());

    Ok(result)
}
