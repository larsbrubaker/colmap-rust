//! The writing half of [`super::database::Database`] (COLMAP's `scene/database.h`, with the
//! behavior of `database_sqlite.cc`'s INSERT / UPDATE / DELETE statements): `Write*`,
//! `Update*`, `Delete*` and `Clear*`, and the checks that stand in for the SQLite schema's
//! UNIQUE indices, primary keys, foreign keys and the image id CHECK. Every write validates
//! completely before it stores anything (docs/CPP_DIVERGENCES.md entry 266). Port of
//! colmap-sharp's `Scene/InMemoryDatabase.cs` / `.Pairs.cs`. Tests: `tests/scene/database.rs`.

use std::collections::HashSet;

use crate::feature::types::{matches_to_matrix, FeatureDescriptors, FeatureKeypoint, FeatureMatch};
use crate::geometry::pose_prior::PosePrior;
use crate::scene::camera::Camera;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::two_view_geometry::TwoViewGeometry;
use crate::sensor::rig::Rig;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{
    image_pair_to_pair_id, should_swap_image_pair, CameraId, DataId, FrameId, ImageId, PosePriorId,
    RigId, MAX_NUM_IMAGES,
};
use crate::{check, check_eq, Result};

use super::database::{stored_columns, Database, ImageRow, Tables};
use super::database_merge::{
    feature_keypoints_to_blob, swap_feature_matches_blob, FeatureKeypointsBlob, FeatureMatchesBlob,
};

/// SQLite's error for a violated constraint (`std::runtime_error` in COLMAP), worded like
/// SQLite's extended messages.
fn constraint_failed(what: &str) -> ColmapError {
    ColmapError::new(ErrorKind::RuntimeError, format!("SQLite error: {what}"))
}

/// SQLite's "FOREIGN KEY constraint failed".
fn check_foreign_key(condition: bool) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(constraint_failed("FOREIGN KEY constraint failed"))
    }
}

/// SQLite's "UNIQUE constraint failed: <columns>".
fn check_unique(condition: bool, columns: &str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(constraint_failed(&format!(
            "UNIQUE constraint failed: {columns}"
        )))
    }
}

/// The next id of an AUTOINCREMENT table: one more than the largest id it ever held. COLMAP
/// truncates SQLite's 64-bit rowid to the 32-bit id; past `u32::MAX` this fails instead
/// (entry 266).
fn next_auto_increment_id(sequence: u64) -> Result<u32> {
    u32::try_from(sequence + 1).map_err(|_| {
        ColmapError::new(
            ErrorKind::RuntimeError,
            "SQLite error: the next id does not fit a 32-bit id",
        )
    })
}

impl Tables {
    /// UNIQUE index `index_name`, CHECK `image_id_check` and the camera foreign key.
    fn check_image_constraints(
        &self,
        image_id: ImageId,
        name: &str,
        camera_id: CameraId,
    ) -> Result<()> {
        check!(u64::from(image_id) < MAX_NUM_IMAGES, "image_id_check");
        check_unique(
            self.image_ids_by_name
                .get(name)
                .is_none_or(|&other| other == image_id),
            "images.name",
        )?;
        check_foreign_key(self.cameras.contains_key(&camera_id))
    }

    /// UNIQUE indices `rig_ref_sensor_assignment` and `rig_sensor_assignment`.
    fn check_rig_constraints(&self, rig: &Rig, ignore_rig_id: Option<RigId>) -> Result<()> {
        for (&rig_id, other) in &self.rigs {
            if Some(rig_id) == ignore_rig_id {
                continue;
            }
            check_unique(
                other.ref_sensor_id() != rig.ref_sensor_id(),
                "rigs.ref_sensor_id, rigs.ref_sensor_type",
            )?;
            for sensor_id in rig.non_ref_sensors().keys() {
                check_unique(
                    !other.non_ref_sensors().contains_key(sensor_id),
                    "rig_sensors.sensor_id, rig_sensors.sensor_type",
                )?;
            }
        }
        Ok(())
    }

    /// UNIQUE index `frame_sensor_assignment` on `frame_data(data_id, sensor_type)`.
    fn check_frame_data_constraints(
        &self,
        frame: &Frame,
        ignore_frame_id: Option<FrameId>,
    ) -> Result<()> {
        let mut seen = HashSet::new();
        for data_id in frame.data_ids() {
            let key = (data_id.id, data_id.sensor_id.sensor_type);
            let free = self
                .frame_ids_by_data
                .get(&key)
                .is_none_or(|&owner| Some(owner) == ignore_frame_id);
            check_unique(
                free && seen.insert(key),
                "frame_data.data_id, frame_data.sensor_type",
            )?;
        }
        Ok(())
    }

    /// UNIQUE index `pose_prior_data_assignment`.
    fn check_pose_prior_constraints(
        &self,
        corr_data_id: DataId,
        ignore_pose_prior_id: Option<PosePriorId>,
    ) -> Result<()> {
        check_unique(
            self.pose_prior_ids_by_data
                .get(&corr_data_id)
                .is_none_or(|&owner| Some(owner) == ignore_pose_prior_id),
            "pose_priors.corr_data_id, pose_priors.corr_sensor_id, pose_priors.corr_sensor_type",
        )
    }

    fn index_frame_data(&mut self, frame: &Frame) {
        for data_id in frame.data_ids() {
            self.frame_ids_by_data.insert(
                (data_id.id, data_id.sensor_id.sensor_type),
                frame.frame_id(),
            );
        }
    }

    fn unindex_frame_data(&mut self, frame_id: FrameId) {
        if let Some(frame) = self.frames.get(&frame_id) {
            for data_id in frame.data_ids() {
                self.frame_ids_by_data
                    .remove(&(data_id.id, data_id.sensor_id.sensor_type));
            }
        }
    }

    /// The stored frame row: the id, the rig id and the data ids (the pose is no column).
    fn frame_row(frame: &Frame, frame_id: FrameId) -> Result<Frame> {
        let mut row = Frame::new();
        row.set_frame_id(frame_id);
        if frame.has_rig_id() {
            row.set_rig_id(frame.rig_id())?;
        }
        for &data_id in frame.data_ids() {
            row.add_data_id(data_id)?;
        }
        Ok(row)
    }

    /// `WriteImage`'s / `UpdateImage`'s check that the image's frame holds the image.
    fn check_image_frame(&self, image: &Image) -> Result<()> {
        if image.has_frame_id() {
            check!(self
                .read_frame(image.frame_id())
                .has_data_id(image.data_id()));
        }
        Ok(())
    }
}

impl Database {
    /// `WriteRig`: adds a rig with at least one sensor and returns its id (a new one unless
    /// `use_rig_id`).
    pub fn write_rig(&self, rig: &Rig, use_rig_id: bool) -> Result<RigId> {
        check!(rig.num_sensors() > 0, "Rig must have at least one sensor");
        let mut tables = self.open_tables()?;
        if use_rig_id {
            check!(
                !tables.rigs.contains_key(&rig.rig_id()),
                "rig_id must be unique"
            );
        }
        tables.check_rig_constraints(rig, None)?;
        let rig_id = if use_rig_id {
            rig.rig_id()
        } else {
            next_auto_increment_id(tables.rig_sequence)?
        };
        tables.rig_sequence = tables.rig_sequence.max(u64::from(rig_id));
        let mut stored = rig.clone();
        stored.set_rig_id(rig_id);
        tables.rigs.insert(rig_id, stored);
        Ok(rig_id)
    }

    /// `WriteCamera`: adds a camera and returns its id (a new one unless `use_camera_id`).
    pub fn write_camera(&self, camera: &Camera, use_camera_id: bool) -> Result<CameraId> {
        let mut tables = self.open_tables()?;
        if use_camera_id {
            check!(
                !tables.cameras.contains_key(&camera.camera_id),
                "camera_id must be unique"
            );
        }
        let camera_id = if use_camera_id {
            camera.camera_id
        } else {
            next_auto_increment_id(tables.camera_sequence)?
        };
        tables.camera_sequence = tables.camera_sequence.max(u64::from(camera_id));
        let mut stored = camera.clone();
        stored.camera_id = camera_id;
        tables.cameras.insert(camera_id, stored);
        Ok(camera_id)
    }

    /// `WriteFrame`: adds a frame of an existing rig and returns its id (a new one unless
    /// `use_frame_id`).
    pub fn write_frame(&self, frame: &Frame, use_frame_id: bool) -> Result<FrameId> {
        let mut tables = self.open_tables()?;
        if use_frame_id {
            check!(
                !tables.frames.contains_key(&frame.frame_id()),
                "frame_id must be unique"
            );
        }
        check_foreign_key(tables.rigs.contains_key(&frame.rig_id()))?;
        tables.check_frame_data_constraints(frame, None)?;
        let frame_id = if use_frame_id {
            frame.frame_id()
        } else {
            next_auto_increment_id(tables.frame_sequence)?
        };
        tables.frame_sequence = tables.frame_sequence.max(u64::from(frame_id));
        let row = Tables::frame_row(frame, frame_id)?;
        tables.index_frame_data(&row);
        tables.frames.insert(frame_id, row);
        Ok(frame_id)
    }

    /// `WriteImage`: adds an image of an existing camera and returns its id (a new one
    /// unless `use_image_id`). If the image names a frame, that frame must hold it.
    pub fn write_image(&self, image: &Image, use_image_id: bool) -> Result<ImageId> {
        let mut tables = self.open_tables()?;
        tables.check_image_frame(image)?;
        if use_image_id {
            check!(
                !tables.images.contains_key(&image.image_id()),
                "image_id must be unique"
            );
        }
        let image_id = if use_image_id {
            image.image_id()
        } else {
            next_auto_increment_id(tables.image_sequence)?
        };
        tables.check_image_constraints(image_id, image.name(), image.camera_id())?;
        tables.image_sequence = tables.image_sequence.max(u64::from(image_id));
        tables.images.insert(
            image_id,
            ImageRow {
                name: image.name().to_string(),
                camera_id: image.camera_id(),
            },
        );
        tables
            .image_ids_by_name
            .insert(image.name().to_string(), image_id);
        Ok(image_id)
    }

    /// `WritePosePrior`: adds a pose prior and returns its id (a new one unless
    /// `use_pose_prior_id`). At most one pose prior per corresponding data id.
    pub fn write_pose_prior(
        &self,
        pose_prior: &PosePrior,
        use_pose_prior_id: bool,
    ) -> Result<PosePriorId> {
        let mut tables = self.open_tables()?;
        if use_pose_prior_id {
            check!(
                !tables.pose_priors.contains_key(&pose_prior.pose_prior_id),
                "pose_prior_id must be unique"
            );
        }
        tables.check_pose_prior_constraints(pose_prior.corr_data_id, None)?;
        // A plain INTEGER PRIMARY KEY: max(rowid) + 1, or 1 for an empty table.
        let pose_prior_id = if use_pose_prior_id {
            pose_prior.pose_prior_id
        } else {
            let max_id = tables
                .pose_priors
                .keys()
                .next_back()
                .map_or(0, |&id| u64::from(id));
            next_auto_increment_id(max_id)?
        };
        let mut stored = *pose_prior;
        stored.pose_prior_id = pose_prior_id;
        tables.pose_priors.insert(pose_prior_id, stored);
        tables
            .pose_prior_ids_by_data
            .insert(stored.corr_data_id, pose_prior_id);
        Ok(pose_prior_id)
    }

    /// `WriteKeypoints` (stored as an N x 6 blob).
    pub fn write_keypoints(&self, image_id: ImageId, keypoints: &[FeatureKeypoint]) -> Result<()> {
        self.write_keypoints_blob(image_id, &feature_keypoints_to_blob(keypoints))
    }

    /// `WriteKeypoints(image_id, blob)`: an N x 2, 4 or 6 blob of an existing image.
    pub fn write_keypoints_blob(
        &self,
        image_id: ImageId,
        blob: &FeatureKeypointsBlob,
    ) -> Result<()> {
        let mut tables = self.open_tables()?;
        check_unique(
            !tables.keypoints.contains_key(&image_id),
            "keypoints.image_id",
        )?;
        check_foreign_key(tables.images.contains_key(&image_id))?;
        tables.keypoints.insert(image_id, blob.clone());
        Ok(())
    }

    /// `WriteDescriptors` of an existing image.
    pub fn write_descriptors(
        &self,
        image_id: ImageId,
        descriptors: &FeatureDescriptors,
    ) -> Result<()> {
        let mut tables = self.open_tables()?;
        check_unique(
            !tables.descriptors.contains_key(&image_id),
            "descriptors.image_id",
        )?;
        check_foreign_key(tables.images.contains_key(&image_id))?;
        tables.descriptors.insert(image_id, descriptors.clone());
        Ok(())
    }

    /// `WriteMatches`: the raw matches of a pair, given in the pair's order.
    pub fn write_matches(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        matches: &[FeatureMatch],
    ) -> Result<()> {
        self.write_matches_blob(image_id1, image_id2, &matches_to_matrix(matches))
    }

    /// `WriteMatches(image_id1, image_id2, blob)`: an N x 2 blob of a pair.
    pub fn write_matches_blob(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        blob: &FeatureMatchesBlob,
    ) -> Result<()> {
        // FeatureMatchesBlob has two columns at compile time in COLMAP.
        check_eq!(blob.cols(), 2);
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        let mut tables = self.open_tables()?;
        check_unique(!tables.matches.contains_key(&pair_id), "matches.pair_id")?;
        let mut stored = blob.clone();
        if should_swap_image_pair(image_id1, image_id2) {
            swap_feature_matches_blob(&mut stored);
        }
        tables.matches.insert(pair_id, stored);
        Ok(())
    }

    /// `WriteTwoViewGeometry`: the geometry of a pair, given in the pair's order. Fails if
    /// the pair has one already.
    pub fn write_two_view_geometry(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        two_view_geometry: &TwoViewGeometry,
    ) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        let mut tables = self.open_tables()?;
        check!(
            !tables.exists_two_view_geometry(pair_id),
            "Two view geometry between image {} and {} already exists.",
            image_id1,
            image_id2
        );
        // Invert the two-view geometry if the image pair has to be swapped.
        let mut stored = stored_columns(two_view_geometry);
        if should_swap_image_pair(image_id1, image_id2) {
            stored.invert();
        }
        tables.two_view_geometries.insert(pair_id, stored);
        Ok(())
    }

    /// `UpdateRig`: replaces an existing rig's sensors (nothing if the rig is missing, but
    /// re-inserting its sensors then violates the `rig_sensors` foreign key).
    pub fn update_rig(&self, rig: &Rig) -> Result<()> {
        let mut tables = self.open_tables()?;
        if !tables.rigs.contains_key(&rig.rig_id()) {
            return check_foreign_key(rig.non_ref_sensors().is_empty());
        }
        tables.check_rig_constraints(rig, Some(rig.rig_id()))?;
        tables.rigs.insert(rig.rig_id(), rig.clone());
        Ok(())
    }

    /// `UpdateCamera`: replaces an existing camera (nothing if missing).
    pub fn update_camera(&self, camera: &Camera) -> Result<()> {
        let mut tables = self.open_tables()?;
        if let Some(stored) = tables.cameras.get_mut(&camera.camera_id) {
            *stored = camera.clone();
        }
        Ok(())
    }

    /// `UpdateFrame`: replaces an existing frame's rig id and data (nothing if the frame is
    /// missing, but re-inserting its data then violates the `frame_data` foreign key).
    pub fn update_frame(&self, frame: &Frame) -> Result<()> {
        let mut tables = self.open_tables()?;
        let frame_id = frame.frame_id();
        if !tables.frames.contains_key(&frame_id) {
            return check_foreign_key(frame.num_data_ids() == 0);
        }
        check_foreign_key(tables.rigs.contains_key(&frame.rig_id()))?;
        tables.check_frame_data_constraints(frame, Some(frame_id))?;
        let row = Tables::frame_row(frame, frame_id)?;
        tables.unindex_frame_data(frame_id);
        tables.index_frame_data(&row);
        tables.frames.insert(frame_id, row);
        Ok(())
    }

    /// `UpdateImage`: replaces an existing image's name and camera (nothing if missing).
    pub fn update_image(&self, image: &Image) -> Result<()> {
        let mut tables = self.open_tables()?;
        tables.check_image_frame(image)?;
        let image_id = image.image_id();
        let Some(old_name) = tables.images.get(&image_id).map(|row| row.name.clone()) else {
            return Ok(());
        };
        tables.check_image_constraints(image_id, image.name(), image.camera_id())?;
        tables.image_ids_by_name.remove(&old_name);
        tables.images.insert(
            image_id,
            ImageRow {
                name: image.name().to_string(),
                camera_id: image.camera_id(),
            },
        );
        tables
            .image_ids_by_name
            .insert(image.name().to_string(), image_id);
        Ok(())
    }

    /// `UpdatePosePrior`: replaces an existing pose prior (nothing if missing).
    pub fn update_pose_prior(&self, pose_prior: &PosePrior) -> Result<()> {
        let mut tables = self.open_tables()?;
        let id = pose_prior.pose_prior_id;
        let Some(old_data_id) = tables.pose_priors.get(&id).map(|p| p.corr_data_id) else {
            return Ok(());
        };
        tables.check_pose_prior_constraints(pose_prior.corr_data_id, Some(id))?;
        tables.pose_prior_ids_by_data.remove(&old_data_id);
        tables.pose_priors.insert(id, *pose_prior);
        tables
            .pose_prior_ids_by_data
            .insert(pose_prior.corr_data_id, id);
        Ok(())
    }

    /// `UpdateKeypoints`: replaces an existing image's keypoints (nothing if missing).
    pub fn update_keypoints(&self, image_id: ImageId, keypoints: &[FeatureKeypoint]) -> Result<()> {
        self.update_keypoints_blob(image_id, &feature_keypoints_to_blob(keypoints))
    }

    /// `UpdateKeypoints(image_id, blob)`.
    pub fn update_keypoints_blob(
        &self,
        image_id: ImageId,
        blob: &FeatureKeypointsBlob,
    ) -> Result<()> {
        let mut tables = self.open_tables()?;
        if let Some(stored) = tables.keypoints.get_mut(&image_id) {
            *stored = blob.clone();
        }
        Ok(())
    }

    /// `UpdateTwoViewGeometry`: replaces an existing geometry; does nothing if the pair has
    /// none, to align with the UPDATE behavior in SQL.
    pub fn update_two_view_geometry(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        two_view_geometry: &TwoViewGeometry,
    ) -> Result<()> {
        if self.exists_two_view_geometry(image_id1, image_id2)? {
            self.delete_two_view_geometry(image_id1, image_id2)?;
            self.write_two_view_geometry(image_id1, image_id2, two_view_geometry)?;
        }
        Ok(())
    }

    /// `DeleteMatches` of a pair.
    pub fn delete_matches(&self, image_id1: ImageId, image_id2: ImageId) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        self.open_tables()?.matches.remove(&pair_id);
        Ok(())
    }

    /// `DeleteTwoViewGeometry` of a pair.
    pub fn delete_two_view_geometry(&self, image_id1: ImageId, image_id2: ImageId) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        self.open_tables()?.two_view_geometries.remove(&pair_id);
        Ok(())
    }

    /// `DeleteInlierMatches`: clears a pair's inlier matches, keeping the rest of its
    /// geometry.
    pub fn delete_inlier_matches(&self, image_id1: ImageId, image_id2: ImageId) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        if let Some(stored) = self.open_tables()?.two_view_geometries.get_mut(&pair_id) {
            stored.inlier_matches.clear();
        }
        Ok(())
    }

    /// `ClearAllTables`.
    pub fn clear_all_tables(&self) -> Result<()> {
        self.clear_matches()?;
        self.clear_two_view_geometries()?;
        self.clear_descriptors()?;
        self.clear_keypoints()?;
        self.clear_pose_priors()?;
        self.clear_frames()?;
        self.clear_images()?;
        self.clear_rigs()?;
        self.clear_cameras()
    }

    /// `ClearRigs`, and through `frames.rig_id ... ON DELETE CASCADE` their frames.
    pub fn clear_rigs(&self) -> Result<()> {
        let mut tables = self.open_tables()?;
        tables.rigs.clear();
        tables.frames.clear();
        tables.frame_ids_by_data.clear();
        Ok(())
    }

    /// `ClearCameras`. `images.camera_id` references the cameras without a cascade, so this
    /// fails while any image exists.
    pub fn clear_cameras(&self) -> Result<()> {
        let mut tables = self.open_tables()?;
        check_foreign_key(tables.images.is_empty())?;
        tables.cameras.clear();
        Ok(())
    }

    /// `ClearFrames`.
    pub fn clear_frames(&self) -> Result<()> {
        let mut tables = self.open_tables()?;
        tables.frames.clear();
        tables.frame_ids_by_data.clear();
        Ok(())
    }

    /// `ClearImages`, and through `ON DELETE CASCADE` their keypoints and descriptors.
    pub fn clear_images(&self) -> Result<()> {
        let mut tables = self.open_tables()?;
        tables.images.clear();
        tables.image_ids_by_name.clear();
        tables.keypoints.clear();
        tables.descriptors.clear();
        Ok(())
    }

    /// `ClearPosePriors`.
    pub fn clear_pose_priors(&self) -> Result<()> {
        let mut tables = self.open_tables()?;
        tables.pose_priors.clear();
        tables.pose_prior_ids_by_data.clear();
        Ok(())
    }

    /// `ClearDescriptors`.
    pub fn clear_descriptors(&self) -> Result<()> {
        self.open_tables()?.descriptors.clear();
        Ok(())
    }

    /// `ClearKeypoints`.
    pub fn clear_keypoints(&self) -> Result<()> {
        self.open_tables()?.keypoints.clear();
        Ok(())
    }

    /// `ClearMatches`.
    pub fn clear_matches(&self) -> Result<()> {
        self.open_tables()?.matches.clear();
        Ok(())
    }

    /// `ClearTwoViewGeometries`.
    pub fn clear_two_view_geometries(&self) -> Result<()> {
        self.open_tables()?.two_view_geometries.clear();
        Ok(())
    }
}
