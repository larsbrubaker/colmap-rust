//! Port of COLMAP's `scene/database.h` (and the observable behavior of `database_sqlite.cc`):
//! the store of rigs, cameras, frames, images, pose priors, keypoints, descriptors, raw
//! matches and verified two-view geometries that feature extraction and matching fill and
//! the mappers read. Port of colmap-sharp's `Scene/Database.cs` + `Scene/InMemoryDatabase.cs`.
//! This file holds the type, its tables, `Close`, the transactions and every query
//! (`Exists*`, `Num*`, `Read*`); [`super::database_write`] holds the writes, updates,
//! deletes and clears with the constraint checks, and [`super::database_merge`] `Merge`,
//! `LoadRandomDatabaseDescriptors` and the keypoint/match blob conversions.
//! Tests: `tests/scene/database.rs` (`database_test.cc` 1:1).
//!
//! Tier A (exact): pure bookkeeping.
//!
//! Deliberate divergence, one in-memory type (these decisions):
//! - COLMAP's `Database` is an abstract class whose only implementation is `SqliteDatabase`
//!   (`database_sqlite.cc`), opened through a path-keyed factory registry (`Database::Open`,
//!   `Database::Register`); `kInMemorySqliteDatabasePath` opens an in-memory SQLite database.
//!   SQLite database files are out of scope (PORTING_PLAN.md: SQLite is native code), and a
//!   trait with a single implementation would only add indirection, so [`Database`] is one
//!   concrete type that keeps its tables in memory: [`Database::new`] stands for
//!   `Open(kInMemorySqliteDatabasePath)`, and there is no `Open(path)`, no `Register` and no
//!   file.
//! - It reproduces everything COLMAP's SQLite schema and statements make observable: the
//!   AUTOINCREMENT ids of rigs, cameras, frames and images (one more than the largest id the
//!   table ever held, explicit ids included; clearing does not reset it), the pose priors'
//!   plain INTEGER PRIMARY KEY (one more than the largest current id), the UNIQUE indices
//!   (image names, a sensor in at most one rig as reference and one as non-reference sensor,
//!   a data id in at most one frame, one pose prior per data id, one keypoints / descriptors /
//!   matches / two-view geometry row per image or pair), the foreign keys with their cascades
//!   (clearing rigs drops their frames, clearing images their keypoints and descriptors,
//!   clearing cameras fails while images exist), rows read in id order (pairs in pair id
//!   order), UPDATE of a missing row being a no-op, reads of a missing row returning defaults,
//!   and the two-view geometry columns (`tri_angle` is not stored, so a read returns its
//!   default). A write that violates a constraint stores nothing (the divergence in the
//!   [`super::database_write`] header).
//! - Every method takes `&self`: the tables sit behind a `Mutex`, so a `Database` can be
//!   shared between threads (`Arc<Database>`) the way COLMAP shares its `shared_ptr`, and
//!   [`DatabaseTransaction`] holds the transaction lock while its owner keeps writing, so only
//!   one transaction is active at a time. Every call is applied at once, so
//!   `BeginTransaction` / `EndTransaction` only check that the database is open; COLMAP never
//!   rolls a transaction back.
//! - Every method returns `Result`: after [`Database::close`] each call fails, as COLMAP's
//!   do (`OpenCloseInMemory`).
//! - `tests/scene/database.rs` (`database_test.cc` 1:1; the three `OpenFile*` cases are
//!   skipped as file-based) and `tests/scene/rust_only_database.rs` pass. Same as
//!   colmap-sharp's `InMemoryDatabase` (its `Scene/InMemoryDatabase.cs` header).

use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, MutexGuard};

use crate::feature::types::{
    matches_from_matrix, FeatureDescriptors, FeatureKeypoints, FeatureMatches,
};
use crate::geometry::pose_prior::PosePrior;
use crate::scene::camera::Camera;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::two_view_geometry::TwoViewGeometry;
use crate::sensor::models::camera_model_num_params;
use crate::sensor::rig::Rig;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{
    image_pair_to_pair_id, should_swap_image_pair, CameraId, DataId, FrameId, ImageId, ImagePairId,
    PosePriorId, RigId, SensorId, SensorType,
};
use crate::{check_eq, Result};

use super::database_merge::{
    feature_keypoints_from_blob, swap_feature_matches_blob, FeatureKeypointsBlob,
    FeatureMatchesBlob,
};

/// An `images` row: the name and the camera. The frame comes from `frame_data`.
#[derive(Debug, Clone)]
pub(super) struct ImageRow {
    pub(super) name: String,
    pub(super) camera_id: CameraId,
}

/// The tables of a [`Database`], with the indices of its UNIQUE constraints.
#[derive(Debug, Default)]
pub(super) struct Tables {
    pub(super) closed: bool,
    /// `rigs` + `rig_sensors`: the rig with its id set.
    pub(super) rigs: BTreeMap<RigId, Rig>,
    /// `cameras`: the camera with its id set.
    pub(super) cameras: BTreeMap<CameraId, Camera>,
    /// `frames` + `frame_data`: the frame's rig id and data ids (no pose: not a column).
    pub(super) frames: BTreeMap<FrameId, Frame>,
    pub(super) images: BTreeMap<ImageId, ImageRow>,
    pub(super) pose_priors: BTreeMap<PosePriorId, PosePrior>,
    pub(super) keypoints: BTreeMap<ImageId, FeatureKeypointsBlob>,
    pub(super) descriptors: BTreeMap<ImageId, FeatureDescriptors>,
    /// `matches`, stored in the normalized orientation (smaller image id first).
    pub(super) matches: BTreeMap<ImagePairId, FeatureMatchesBlob>,
    /// `two_view_geometries`, stored in the normalized orientation, with only the stored
    /// columns (no `tri_angle`).
    pub(super) two_view_geometries: BTreeMap<ImagePairId, TwoViewGeometry>,
    /// `sqlite_sequence` of the AUTOINCREMENT tables: the largest id ever inserted.
    pub(super) rig_sequence: u64,
    pub(super) camera_sequence: u64,
    pub(super) frame_sequence: u64,
    pub(super) image_sequence: u64,
    // The UNIQUE indices on images(name), frame_data(data_id, sensor_type) and
    // pose_priors(corr_data_id, corr_sensor_id, corr_sensor_type). Lookups only; never
    // iterated, so std's hash order cannot reach an output.
    pub(super) image_ids_by_name: HashMap<String, ImageId>,
    pub(super) frame_ids_by_data: HashMap<(u64, SensorType), FrameId>,
    pub(super) pose_prior_ids_by_data: HashMap<DataId, PosePriorId>,
}

impl Tables {
    /// The images LEFT JOIN frame_data: the frame holding the image as camera data, if any.
    pub(super) fn to_image(&self, image_id: ImageId, row: &ImageRow) -> Result<Image> {
        let mut image = Image::new();
        image.set_image_id(image_id);
        image.set_name(&row.name);
        image.set_camera_id(row.camera_id)?;
        if let Some(&frame_id) = self
            .frame_ids_by_data
            .get(&(u64::from(image_id), SensorType::Camera))
        {
            image.set_frame_id(frame_id);
        }
        Ok(image)
    }

    pub(super) fn read_frame(&self, frame_id: FrameId) -> Frame {
        self.frames.get(&frame_id).cloned().unwrap_or_default()
    }

    pub(super) fn exists_two_view_geometry(&self, pair_id: ImagePairId) -> bool {
        self.two_view_geometries.contains_key(&pair_id)
    }
}

/// `ReadCameraRow`'s check: the stored parameter count must fit the model.
fn read_camera_row(camera: &Camera) -> Result<Camera> {
    check_eq!(
        camera.params.len(),
        camera_model_num_params(camera.model_id)?
    );
    Ok(camera.clone())
}

/// The columns of the `two_view_geometries` table: everything but `tri_angle`, which a
/// read returns at its default, like COLMAP.
pub(super) fn stored_columns(two_view_geometry: &TwoViewGeometry) -> TwoViewGeometry {
    TwoViewGeometry {
        config: two_view_geometry.config,
        e: two_view_geometry.e,
        f: two_view_geometry.f,
        h: two_view_geometry.h,
        cam2_from_cam1: two_view_geometry.cam2_from_cam1,
        camera1: two_view_geometry.camera1.clone(),
        camera2: two_view_geometry.camera2.clone(),
        inlier_matches: two_view_geometry.inlier_matches.clone(),
        ..TwoViewGeometry::default()
    }
}

/// Port of `MaybeThrowDeprecatedPosePriorError` (`std::runtime_error`).
pub(super) fn maybe_throw_deprecated_pose_prior_error(
    is_deprecated_image_prior: bool,
) -> Result<()> {
    if is_deprecated_image_prior {
        return Err(ColmapError::new(
            ErrorKind::RuntimeError,
            "PosePrior API has changed: pose priors are now associated with frames, not \
             images. Please update your code to use frames instead of image IDs. Data is \
             automatically migrated upon opening a database. Update your API usage \
             accordingly and add pose priors to frame data.",
        ));
    }
    Ok(())
}

/// Port of `colmap::Database` over in-memory tables (see the module docs). For image pairs,
/// the order of `image_id1` and `image_id2` does not matter. COLMAP's `Database` is not
/// thread-safe; this one serializes every call.
#[derive(Debug, Default)]
pub struct Database {
    pub(super) tables: Mutex<Tables>,
    /// Ensures that only one [`DatabaseTransaction`] is active at the same time.
    transaction_mutex: Mutex<()>,
}

impl Database {
    /// An empty, open database: `Database::Open(kInMemorySqliteDatabasePath)`.
    pub fn new() -> Self {
        Self::default()
    }

    /// The tables, or `Err` once the database is closed. A panic while the lock was held
    /// cannot leave a half-applied write (every write validates before it stores), so a
    /// poisoned lock is taken over.
    pub(super) fn open_tables(&self) -> Result<MutexGuard<'_, Tables>> {
        let tables = self.tables.lock().unwrap_or_else(|e| e.into_inner());
        if tables.closed {
            return Err(ColmapError::new(
                ErrorKind::RuntimeError,
                "SQLite error: The database is closed.",
            ));
        }
        Ok(tables)
    }

    /// `Close`: explicitly close the database; every later call fails. Closing twice is
    /// allowed.
    pub fn close(&self) {
        self.tables.lock().unwrap_or_else(|e| e.into_inner()).closed = true;
    }

    /// `BeginTransaction`. Use [`DatabaseTransaction`].
    pub fn begin_transaction(&self) -> Result<()> {
        self.open_tables().map(|_| ())
    }

    /// `EndTransaction`. Use [`DatabaseTransaction`].
    pub fn end_transaction(&self) -> Result<()> {
        self.open_tables().map(|_| ())
    }

    /// `ExistsRig`.
    pub fn exists_rig(&self, rig_id: RigId) -> Result<bool> {
        Ok(self.open_tables()?.rigs.contains_key(&rig_id))
    }

    /// `ExistsCamera`.
    pub fn exists_camera(&self, camera_id: CameraId) -> Result<bool> {
        Ok(self.open_tables()?.cameras.contains_key(&camera_id))
    }

    /// `ExistsFrame`.
    pub fn exists_frame(&self, frame_id: FrameId) -> Result<bool> {
        Ok(self.open_tables()?.frames.contains_key(&frame_id))
    }

    /// `ExistsImage`.
    pub fn exists_image(&self, image_id: ImageId) -> Result<bool> {
        Ok(self.open_tables()?.images.contains_key(&image_id))
    }

    /// `ExistsImageWithName`.
    pub fn exists_image_with_name(&self, name: &str) -> Result<bool> {
        Ok(self.open_tables()?.image_ids_by_name.contains_key(name))
    }

    /// `ExistsPosePrior`. `is_deprecated_image_prior` must be false: COLMAP fails for the
    /// old image-keyed pose prior API (its default, `true`).
    pub fn exists_pose_prior(
        &self,
        pose_prior_id: PosePriorId,
        is_deprecated_image_prior: bool,
    ) -> Result<bool> {
        maybe_throw_deprecated_pose_prior_error(is_deprecated_image_prior)?;
        Ok(self.open_tables()?.pose_priors.contains_key(&pose_prior_id))
    }

    /// `ExistsKeypoints`.
    pub fn exists_keypoints(&self, image_id: ImageId) -> Result<bool> {
        Ok(self.open_tables()?.keypoints.contains_key(&image_id))
    }

    /// `ExistsDescriptors`.
    pub fn exists_descriptors(&self, image_id: ImageId) -> Result<bool> {
        Ok(self.open_tables()?.descriptors.contains_key(&image_id))
    }

    /// `ExistsMatches`.
    pub fn exists_matches(&self, image_id1: ImageId, image_id2: ImageId) -> Result<bool> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        Ok(self.open_tables()?.matches.contains_key(&pair_id))
    }

    /// `ExistsTwoViewGeometry`.
    pub fn exists_two_view_geometry(&self, image_id1: ImageId, image_id2: ImageId) -> Result<bool> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        Ok(self.open_tables()?.exists_two_view_geometry(pair_id))
    }

    /// `NumRigs`: rows in the `rigs` table.
    pub fn num_rigs(&self) -> Result<usize> {
        Ok(self.open_tables()?.rigs.len())
    }

    /// `NumCameras`.
    pub fn num_cameras(&self) -> Result<usize> {
        Ok(self.open_tables()?.cameras.len())
    }

    /// `NumFrames`.
    pub fn num_frames(&self) -> Result<usize> {
        Ok(self.open_tables()?.frames.len())
    }

    /// `NumImages`.
    pub fn num_images(&self) -> Result<usize> {
        Ok(self.open_tables()?.images.len())
    }

    /// `NumPosePriors`.
    pub fn num_pose_priors(&self) -> Result<usize> {
        Ok(self.open_tables()?.pose_priors.len())
    }

    /// `NumKeypoints`: the total number of keypoints.
    pub fn num_keypoints(&self) -> Result<usize> {
        Ok(self
            .open_tables()?
            .keypoints
            .values()
            .map(|b| b.rows())
            .sum())
    }

    /// `MaxNumKeypoints`: the keypoint count of the image with the most (0 if none).
    pub fn max_num_keypoints(&self) -> Result<usize> {
        Ok(self
            .open_tables()?
            .keypoints
            .values()
            .map(|b| b.rows())
            .max()
            .unwrap_or(0))
    }

    /// `NumKeypointsForImage` (0 if none).
    pub fn num_keypoints_for_image(&self, image_id: ImageId) -> Result<usize> {
        Ok(self
            .open_tables()?
            .keypoints
            .get(&image_id)
            .map_or(0, |b| b.rows()))
    }

    /// `NumDescriptors`: the total number of descriptors.
    pub fn num_descriptors(&self) -> Result<usize> {
        Ok(self
            .open_tables()?
            .descriptors
            .values()
            .map(|d| d.data.rows())
            .sum())
    }

    /// `MaxNumDescriptors`: the descriptor count of the image with the most (0 if none).
    pub fn max_num_descriptors(&self) -> Result<usize> {
        let tables = self.open_tables()?;
        Ok(tables
            .descriptors
            .values()
            .map(|d| d.data.rows())
            .max()
            .unwrap_or(0))
    }

    /// `NumDescriptorsForImage` (0 if none).
    pub fn num_descriptors_for_image(&self, image_id: ImageId) -> Result<usize> {
        Ok(self
            .open_tables()?
            .descriptors
            .get(&image_id)
            .map_or(0, |d| d.data.rows()))
    }

    /// `NumMatches`: the total number of raw matches.
    pub fn num_matches(&self) -> Result<usize> {
        Ok(self.open_tables()?.matches.values().map(|b| b.rows()).sum())
    }

    /// `NumInlierMatches`: the total number of inlier matches of the two-view geometries.
    pub fn num_inlier_matches(&self) -> Result<usize> {
        let tables = self.open_tables()?;
        Ok(tables
            .two_view_geometries
            .values()
            .map(|g| g.inlier_matches.len())
            .sum())
    }

    /// `NumMatchedImagePairs`: rows in the `matches` table.
    pub fn num_matched_image_pairs(&self) -> Result<usize> {
        Ok(self.open_tables()?.matches.len())
    }

    /// `NumVerifiedImagePairs`: rows in the `two_view_geometries` table.
    pub fn num_verified_image_pairs(&self) -> Result<usize> {
        Ok(self.open_tables()?.two_view_geometries.len())
    }

    /// `ReadRig` (a default rig if missing).
    pub fn read_rig(&self, rig_id: RigId) -> Result<Rig> {
        Ok(self
            .open_tables()?
            .rigs
            .get(&rig_id)
            .cloned()
            .unwrap_or_default())
    }

    /// `ReadRigWithSensor`: the rig holding the sensor. COLMAP looks among the non-reference
    /// sensors first, then the reference sensors.
    pub fn read_rig_with_sensor(&self, sensor_id: SensorId) -> Result<Option<Rig>> {
        let tables = self.open_tables()?;
        let rig = tables
            .rigs
            .values()
            .find(|rig| rig.non_ref_sensors().contains_key(&sensor_id))
            .or_else(|| {
                tables
                    .rigs
                    .values()
                    .find(|rig| rig.num_sensors() > 0 && rig.ref_sensor_id() == sensor_id)
            });
        Ok(rig.cloned())
    }

    /// `ReadAllRigs`, in id order.
    pub fn read_all_rigs(&self) -> Result<Vec<Rig>> {
        Ok(self.open_tables()?.rigs.values().cloned().collect())
    }

    /// `ReadCamera` (a default camera if missing). Fails if the stored parameter count does
    /// not fit the model.
    pub fn read_camera(&self, camera_id: CameraId) -> Result<Camera> {
        match self.open_tables()?.cameras.get(&camera_id) {
            Some(camera) => read_camera_row(camera),
            None => Ok(Camera::default()),
        }
    }

    /// `ReadAllCameras`, in id order.
    pub fn read_all_cameras(&self) -> Result<Vec<Camera>> {
        self.open_tables()?
            .cameras
            .values()
            .map(read_camera_row)
            .collect()
    }

    /// `ReadFrame` (a default frame if missing).
    pub fn read_frame(&self, frame_id: FrameId) -> Result<Frame> {
        Ok(self.open_tables()?.read_frame(frame_id))
    }

    /// `ReadAllFrames`, in id order.
    pub fn read_all_frames(&self) -> Result<Vec<Frame>> {
        Ok(self.open_tables()?.frames.values().cloned().collect())
    }

    /// `ReadImage` (a default image if missing); its frame id is the frame holding it.
    pub fn read_image(&self, image_id: ImageId) -> Result<Image> {
        let tables = self.open_tables()?;
        match tables.images.get(&image_id) {
            Some(row) => tables.to_image(image_id, row),
            None => Ok(Image::default()),
        }
    }

    /// `ReadImageWithName`.
    pub fn read_image_with_name(&self, name: &str) -> Result<Option<Image>> {
        let tables = self.open_tables()?;
        match tables.image_ids_by_name.get(name) {
            Some(&image_id) => tables
                .to_image(image_id, &tables.images[&image_id])
                .map(Some),
            None => Ok(None),
        }
    }

    /// `ReadAllImages`, in id order.
    pub fn read_all_images(&self) -> Result<Vec<Image>> {
        let tables = self.open_tables()?;
        tables
            .images
            .iter()
            .map(|(&image_id, row)| tables.to_image(image_id, row))
            .collect()
    }

    /// `ReadPosePrior` (a default pose prior if missing); see [`Self::exists_pose_prior`].
    pub fn read_pose_prior(
        &self,
        pose_prior_id: PosePriorId,
        is_deprecated_image_prior: bool,
    ) -> Result<PosePrior> {
        maybe_throw_deprecated_pose_prior_error(is_deprecated_image_prior)?;
        Ok(self
            .open_tables()?
            .pose_priors
            .get(&pose_prior_id)
            .copied()
            .unwrap_or_default())
    }

    /// `ReadAllPosePriors`, in id order.
    pub fn read_all_pose_priors(&self) -> Result<Vec<PosePrior>> {
        Ok(self.open_tables()?.pose_priors.values().copied().collect())
    }

    /// `ReadKeypointsBlob`: the stored N x 2, 4 or 6 blob (0 x 0 if missing).
    pub fn read_keypoints_blob(&self, image_id: ImageId) -> Result<FeatureKeypointsBlob> {
        let tables = self.open_tables()?;
        Ok(tables
            .keypoints
            .get(&image_id)
            .cloned()
            .unwrap_or_else(|| FeatureKeypointsBlob::new(0, 0)))
    }

    /// `ReadKeypoints` (empty if missing).
    pub fn read_keypoints(&self, image_id: ImageId) -> Result<FeatureKeypoints> {
        feature_keypoints_from_blob(&self.read_keypoints_blob(image_id)?)
    }

    /// `ReadDescriptors` (empty and `UNDEFINED` if missing).
    pub fn read_descriptors(&self, image_id: ImageId) -> Result<FeatureDescriptors> {
        Ok(self
            .open_tables()?
            .descriptors
            .get(&image_id)
            .cloned()
            .unwrap_or_default())
    }

    /// `ReadMatchesBlob`: the raw matches of a pair, oriented as asked (0 x 2 if missing).
    pub fn read_matches_blob(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<FeatureMatchesBlob> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        let tables = self.open_tables()?;
        let Some(blob) = tables.matches.get(&pair_id) else {
            return Ok(FeatureMatchesBlob::new(0, 2));
        };
        let mut blob = blob.clone();
        if should_swap_image_pair(image_id1, image_id2) {
            swap_feature_matches_blob(&mut blob);
        }
        Ok(blob)
    }

    /// `ReadMatches`: the raw matches of a pair, oriented as asked (empty if missing).
    pub fn read_matches(&self, image_id1: ImageId, image_id2: ImageId) -> Result<FeatureMatches> {
        matches_from_matrix(&self.read_matches_blob(image_id1, image_id2)?)
    }

    /// `ReadAllMatchesBlob`: every pair with at least one raw match, in pair id order and
    /// the stored orientation.
    pub fn read_all_matches_blob(&self) -> Result<Vec<(ImagePairId, FeatureMatchesBlob)>> {
        let tables = self.open_tables()?;
        Ok(tables
            .matches
            .iter()
            .filter(|(_, blob)| blob.rows() > 0)
            .map(|(&pair_id, blob)| (pair_id, blob.clone()))
            .collect())
    }

    /// `ReadAllMatches`: as [`Self::read_all_matches_blob`], as matches.
    pub fn read_all_matches(&self) -> Result<Vec<(ImagePairId, FeatureMatches)>> {
        self.read_all_matches_blob()?
            .into_iter()
            .map(|(pair_id, blob)| Ok((pair_id, matches_from_matrix(&blob)?)))
            .collect()
    }

    /// `ReadNumMatches`: the raw match count of every pair with at least one.
    pub fn read_num_matches(&self) -> Result<Vec<(ImagePairId, i32)>> {
        let tables = self.open_tables()?;
        Ok(tables
            .matches
            .iter()
            .filter(|(_, blob)| blob.rows() > 0)
            // `static_cast<int>` of the int64 `rows` column.
            .map(|(&pair_id, blob)| (pair_id, blob.rows() as i32))
            .collect())
    }

    /// `ReadTwoViewGeometry`: inverted when asked in swapped order (a default geometry if
    /// missing).
    pub fn read_two_view_geometry(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<TwoViewGeometry> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        let tables = self.open_tables()?;
        let Some(stored) = tables.two_view_geometries.get(&pair_id) else {
            return Ok(TwoViewGeometry::default());
        };
        let mut two_view_geometry = stored_columns(stored);
        if should_swap_image_pair(image_id1, image_id2) {
            two_view_geometry.invert();
        }
        Ok(two_view_geometry)
    }

    /// `ReadTwoViewGeometries`: every geometry that carries anything (inliers, a matrix, a
    /// pose or a camera), in pair id order and the stored orientation.
    pub fn read_two_view_geometries(&self) -> Result<Vec<(ImagePairId, TwoViewGeometry)>> {
        let tables = self.open_tables()?;
        Ok(tables
            .two_view_geometries
            .iter()
            .filter(|(_, g)| {
                !g.inlier_matches.is_empty()
                    || g.f.is_some()
                    || g.e.is_some()
                    || g.h.is_some()
                    || g.cam2_from_cam1.is_some()
                    || g.camera1.is_some()
                    || g.camera2.is_some()
            })
            .map(|(&pair_id, g)| (pair_id, stored_columns(g)))
            .collect())
    }

    /// `ReadTwoViewGeometryNumInliers`: the inlier count of every geometry with at least one.
    pub fn read_two_view_geometry_num_inliers(&self) -> Result<Vec<(ImagePairId, i32)>> {
        let tables = self.open_tables()?;
        Ok(tables
            .two_view_geometries
            .iter()
            .filter(|(_, g)| !g.inlier_matches.is_empty())
            // `static_cast<int>` of the int64 `rows` column.
            .map(|(&pair_id, g)| (pair_id, g.inlier_matches.len() as i32))
            .collect())
    }
}

/// Port of `colmap::DatabaseTransaction`: begins a transaction on construction and ends it
/// when dropped, holding the database's transaction lock in between, so only one transaction
/// is active at a time.
#[derive(Debug)]
pub struct DatabaseTransaction<'a> {
    database: &'a Database,
    _lock: MutexGuard<'a, ()>,
}

impl<'a> DatabaseTransaction<'a> {
    /// Waits for the transaction lock and begins a transaction.
    pub fn new(database: &'a Database) -> Result<Self> {
        let lock = database
            .transaction_mutex
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        database.begin_transaction()?;
        Ok(Self {
            database,
            _lock: lock,
        })
    }
}

impl Drop for DatabaseTransaction<'_> {
    fn drop(&mut self) {
        // A destructor cannot return an error. Ending a transaction only fails on a database
        // closed meanwhile, and every call was applied when it was made, so there is nothing
        // left to commit or lose.
        let _ = self.database.end_transaction();
    }
}
