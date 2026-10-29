//! Port of `colmap/scene/reconstruction.cc`, part 5: the [`Reconstruction`] methods that
//! connect a model to a database — `Load` (adding a [`DatabaseCache`]'s cameras, rigs, frames
//! and images, checking those already present agree) and `TranscribeImageIdsToDatabase`
//! (renumbering the images, their frames' data ids and the tracks to the database's ids for the
//! same image names). Neighbors: [`super::reconstruction`] (the type and its add/delete
//! methods), [`super::database`] and [`super::database_cache`]. Tests:
//! `tests/scene/reconstruction_database.rs` (`reconstruction_test.cc`
//! TranscribeImageIdsToDatabase 1:1; COLMAP has no direct `Load` test, the mappers exercise
//! it, so `rust_only_*` tests pin it).
//!
//! Tier A (exact): bookkeeping. The cache is visited in ascending id order (the iteration-order
//! divergence in the [`super::database_cache`] header).

use std::collections::{BTreeMap, HashMap};

use crate::scene::database::Database;
use crate::scene::database_cache::DatabaseCache;
use crate::scene::reconstruction::Reconstruction;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{ImageId, SensorType};
use crate::Result;

impl Reconstruction {
    /// `Load(database_cache)`: adds the cache's cameras, rigs, frames and images that the
    /// reconstruction does not hold yet. One it already holds must agree with the cache
    /// (camera model and size; rig sensors; frame rig and data ids; image name), and an
    /// existing image without 2D points takes the cache's.
    pub fn load(&mut self, database_cache: &DatabaseCache) -> Result<()> {
        // Add cameras.
        for (&camera_id, camera) in database_cache.cameras() {
            if self.exists_camera(camera_id) {
                let existing_camera = self.camera(camera_id)?;
                crate::check_eq!(existing_camera.model_id as i32, camera.model_id as i32);
                crate::check_eq!(existing_camera.width, camera.width);
                crate::check_eq!(existing_camera.height, camera.height);
            } else {
                self.add_camera(camera.clone())?;
            }
        }

        // Add rigs.
        for (&rig_id, rig) in database_cache.rigs() {
            if self.exists_rig(rig_id) {
                let existing_rig = self.rig(rig_id)?;
                crate::check!(existing_rig.ref_sensor_id() == rig.ref_sensor_id());
                crate::check!(existing_rig.sensor_ids() == rig.sensor_ids());
            } else {
                self.add_rig(rig.clone())?;
            }
        }

        // Add frames.
        for (&frame_id, frame) in database_cache.frames() {
            if self.exists_frame(frame_id) {
                let existing_frame = self.frame(frame_id)?;
                crate::check!(existing_frame.rig_id() == frame.rig_id());
                crate::check!(existing_frame.data_ids() == frame.data_ids());
            } else {
                self.add_frame(frame.clone())?;
            }
        }

        // Add images.
        for (&image_id, image) in database_cache.images() {
            if self.exists_image(image_id) {
                let existing_image = self.image_mut(image_id)?;
                crate::check_eq!(existing_image.name(), image.name());
                if existing_image.num_points2d() == 0 {
                    existing_image.set_points2d(image.points2d().to_vec())?;
                } else {
                    crate::check_eq!(image.num_points2d(), existing_image.num_points2d());
                }
            } else {
                self.add_image(image.clone())?;
            }
        }
        Ok(())
    }

    /// `TranscribeImageIdsToDatabase(database)`: gives every image the id the database has
    /// for its name, and rewrites the frames' image data ids and the tracks to match. Fails
    /// if an image's name is not in the database or two images map to the same id. The ids
    /// are all looked up before anything changes, so a failure leaves the reconstruction as
    /// it was.
    ///
    /// That is a deliberate divergence. COLMAP renames the images one by one while it looks
    /// their names up in the database, so when a name is missing (`LOG(FATAL_THROW)`) or two
    /// images map to one id (`THROW_CHECK`), the images before it already carry their new ids
    /// while the frames and tracks still refer to the old ones: the reconstruction is left
    /// broken (`IsValid` fails afterwards). This looks every id up, and checks every frame data
    /// id and track element has a mapping, before it changes anything, so a failure returns
    /// the error and leaves the reconstruction as it was; checking first costs one extra pass
    /// and makes the error recoverable. The successful result is the same.
    /// `rust_only_transcribe_image_ids_to_database_unknown_name_changes_nothing`
    /// (`tests/scene/reconstruction_database.rs`) pins it, and COLMAP's
    /// TranscribeImageIdsToDatabase test passes 1:1.
    pub fn transcribe_image_ids_to_database(&mut self, database: &Database) -> Result<()> {
        // Lookups only, never iterated.
        let mut old_to_new_image_ids: HashMap<ImageId, ImageId> =
            HashMap::with_capacity(self.images.len());
        let mut new_image_ids: Vec<(ImageId, ImageId)> = Vec::with_capacity(self.images.len());
        for (&image_id, image) in &self.images {
            let Some(database_image) = database.read_image_with_name(image.name())? else {
                return Err(ColmapError::new(
                    ErrorKind::InvalidArgument,
                    format!(
                        "Image with name {} does not exist in database",
                        image.name()
                    ),
                ));
            };
            old_to_new_image_ids.insert(image_id, database_image.image_id());
            new_image_ids.push((image_id, database_image.image_id()));
        }
        let mut new_images = BTreeMap::new();
        for &(_, new_image_id) in &new_image_ids {
            // `THROW_CHECK(new_images.emplace(new_id, image).second)`, checked up front.
            crate::check!(new_images.insert(new_image_id, ()).is_none());
        }
        let lookup = |image_id: u64| -> Result<ImageId> {
            // `old_to_new_image_ids.at(id)` throws `std::out_of_range` for an unknown id.
            ImageId::try_from(image_id)
                .ok()
                .and_then(|image_id| old_to_new_image_ids.get(&image_id).copied())
                .ok_or_else(|| {
                    ColmapError::new(
                        ErrorKind::OutOfRange,
                        format!("Image {image_id} is not in the reconstruction"),
                    )
                })
        };
        for frame in self.frames.values() {
            for data_id in frame.data_ids() {
                if data_id.sensor_id.sensor_type == SensorType::Camera {
                    lookup(data_id.id)?;
                }
            }
        }
        for point3d in self.points3d.values() {
            for track_el in point3d.track.elements() {
                lookup(u64::from(track_el.image_id))?;
            }
        }

        let mut images = std::mem::take(&mut self.images);
        let mut transcribed = BTreeMap::new();
        for (old_image_id, new_image_id) in new_image_ids {
            if let Some(mut image) = images.remove(&old_image_id) {
                image.set_image_id(new_image_id);
                transcribed.insert(new_image_id, image);
            }
        }
        self.images = transcribed;

        // Transcribe frame data.
        for frame in self.frames.values_mut() {
            let mut new_frame = frame.clone();
            new_frame.clear_data_ids()?;
            for &data_id in frame.data_ids() {
                let mut data_id = data_id;
                if data_id.sensor_id.sensor_type == SensorType::Camera {
                    data_id.id = u64::from(lookup(data_id.id)?);
                }
                new_frame.add_data_id(data_id)?;
            }
            // We finalize the data ids, otherwise some internal bookkeeping (e.g.,
            // counting reg_image_ids_) will be incorrect.
            new_frame.finalize_data_ids();
            *frame = new_frame;
        }

        // Transcribe point tracks.
        for point3d in self.points3d.values_mut() {
            for track_el in point3d.track.elements_mut() {
                track_el.image_id = lookup(u64::from(track_el.image_id))?;
            }
        }
        Ok(())
    }
}
