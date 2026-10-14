//! Port of COLMAP's `scene/database_cache.h` / `.cc`: the rigs, cameras, frames, images (with
//! their keypoint locations), pose priors and correspondence graph a mapper works on, loaded
//! once from a [`Database`] (optionally restricted to some images, a minimum match count and
//! without watermark pairs) or filtered from another cache. Port of colmap-sharp's
//! `Scene/DatabaseCache.cs`. Tests: `tests/scene/database_cache.rs`
//! (`database_cache_test.cc` 1:1).
//!
//! Tier A (exact): bookkeeping, plus `GPSTransform::EllipsoidToENU` for the optional ENU
//! conversion of the pose priors.
//!
//! Decisions:
//! - Rigs, cameras, frames and images live in `BTreeMap`s, so they enumerate in ascending id
//!   order where COLMAP iterates `NodeHashMap`s (entry 267). The id sets used only for
//!   lookups are std hash sets; nothing iterates them into an output.
//! - The correspondence graph is owned by value: [`DatabaseCache::correspondence_graph`] and
//!   [`DatabaseCache::correspondence_graph_mut`] stand for COLMAP's const and non-const
//!   `shared_ptr` accessors; a caller that shares the cache wraps it in an `Arc`.
//! - `Create` / `CreateFromCache` return the cache by value instead of a `shared_ptr`.
//! - COLMAP's `LOG(INFO)` progress lines are not emitted (the core has no logger).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::geometry::gps::{Ellipsoid, GpsTransform};
use crate::geometry::pose_prior::{CoordinateSystem, PosePrior};
use crate::linalg::{Vector2d, Vector3d};
use crate::scene::camera::Camera;
use crate::scene::correspondence_graph::CorrespondenceGraph;
use crate::scene::database::Database;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::two_view_geometry::ConfigurationType;
use crate::sensor::rig::Rig;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{pair_id_to_image_pair, CameraId, FrameId, ImageId, RigId};
use crate::{check, check_le, Result};

/// Port of `DatabaseCache::Options`.
#[derive(Debug, Clone, Default)]
pub struct DatabaseCacheOptions {
    /// Only load image pairs with a minimum number of matches.
    pub min_num_matches: usize,
    /// Whether to ignore watermark image pairs.
    pub ignore_watermarks: bool,
    /// Whether to use only load the data for a subset of the images. Notice that if one image
    /// of a frame is included, all other images in the same frame will also be included. All
    /// images are used if empty.
    pub image_names: HashSet<String>,
    /// Whether to load all candidate images regardless of whether they have correspondences.
    /// If false (default), only images that participate in at least one valid match pair are
    /// loaded.
    pub load_all_images: bool,
    /// Whether to convert pose priors to ENU coordinate system.
    pub convert_pose_priors_to_enu: bool,
}

/// `UseInlierMatchesCheck`.
fn use_inlier_matches_check(
    options: &DatabaseCacheOptions,
    two_view_geometry_config: ConfigurationType,
    num_matches: usize,
) -> bool {
    num_matches >= options.min_num_matches
        && (!options.ignore_watermarks || two_view_geometry_config != ConfigurationType::Watermark)
}

/// `NodeHashMap::at`: `std::out_of_range` for a missing key.
fn at<K: std::hash::Hash + Eq, V: Copy>(map: &HashMap<K, V>, key: &K) -> Result<V> {
    map.get(key)
        .copied()
        .ok_or_else(|| ColmapError::new(ErrorKind::OutOfRange, "unordered_map::at: key not found"))
}

/// The `std::out_of_range` of `NodeHashMap::at` on the cache's own maps.
fn missing(what: &str, id: u32) -> ColmapError {
    ColmapError::new(
        ErrorKind::OutOfRange,
        format!("unordered_map::at: key not found ({what} {id})"),
    )
}

/// Port of `colmap::DatabaseCache`.
#[derive(Debug, Clone, Default)]
pub struct DatabaseCache {
    rigs: BTreeMap<RigId, Rig>,
    cameras: BTreeMap<CameraId, Camera>,
    frames: BTreeMap<FrameId, Frame>,
    images: BTreeMap<ImageId, Image>,
    pose_priors: Vec<PosePrior>,
    correspondence_graph: CorrespondenceGraph,
}

impl DatabaseCache {
    /// An empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// `Create`: a cache loaded from `database`.
    pub fn create(database: &Database, options: &DatabaseCacheOptions) -> Result<Self> {
        let mut cache = Self::new();
        cache.load(database, options)?;
        Ok(cache)
    }

    /// `Load`: loads cameras, images, features and matches from `database`. Databases from
    /// before rigs and frames get one rig per camera and one frame per image.
    pub fn load(&mut self, database: &Database, options: &DatabaseCacheOptions) -> Result<()> {
        let has_rigs = database.num_rigs()? > 0;
        let has_frames = database.num_frames()? > 0;

        // Load rigs.
        for rig in database.read_all_rigs()? {
            self.rigs.entry(rig.rig_id()).or_insert(rig);
        }

        // Load cameras.
        for camera in database.read_all_cameras()? {
            if !has_rigs {
                // For backwards compatibility with old databases from before having support
                // for rigs/frames, we create a rig for each camera.
                let mut rig = Rig::new();
                rig.set_rig_id(camera.camera_id);
                rig.add_ref_sensor(camera.sensor_id())?;
                self.rigs.entry(rig.rig_id()).or_insert(rig);
            }
            self.cameras.entry(camera.camera_id).or_insert(camera);
        }

        // Load frames.
        for frame in database.read_all_frames()? {
            self.frames.entry(frame.frame_id()).or_insert(frame);
        }

        // Load matches.
        let two_view_geometries = database.read_two_view_geometries()?;

        // Load images.
        let mut images = database.read_all_images()?;
        let mut image_to_frame_id = HashMap::with_capacity(images.len());
        for image in &mut images {
            // For backwards compatibility with old databases from before having support for
            // rigs/frames, we create a frame for each image.
            if has_frames {
                check!(image.has_frame_id());
            } else {
                let mut frame = Frame::new();
                frame.set_frame_id(image.image_id());
                frame.set_rig_id(image.camera_id())?;
                frame.add_data_id(image.data_id())?;
                image.set_frame_id(frame.frame_id());
                self.frames.entry(frame.frame_id()).or_insert(frame);
            }
            image_to_frame_id.insert(image.image_id(), image.frame_id());
        }

        // Determines for which images data should be loaded.
        let frame_ids: HashSet<FrameId> = images
            .iter()
            .filter(|image| {
                options.image_names.is_empty() || options.image_names.contains(image.name())
            })
            .map(|image| image.frame_id())
            .collect();

        // Collect all images that are connected in the correspondence graph.
        let mut connected_frame_ids = HashSet::new();
        if !options.load_all_images {
            for (pair_id, two_view_geometry) in &two_view_geometries {
                if use_inlier_matches_check(
                    options,
                    two_view_geometry.config,
                    two_view_geometry.inlier_matches.len(),
                ) {
                    let (image_id1, image_id2) = pair_id_to_image_pair(*pair_id)?;
                    let frame_id1 = at(&image_to_frame_id, &image_id1)?;
                    let frame_id2 = at(&image_to_frame_id, &image_id2)?;
                    if frame_ids.contains(&frame_id1) && frame_ids.contains(&frame_id2) {
                        connected_frame_ids.insert(frame_id1);
                        connected_frame_ids.insert(frame_id2);
                    }
                }
            }
        }

        let load_frame_ids = if options.load_all_images {
            &frame_ids
        } else {
            &connected_frame_ids
        };

        // Remove frames that should not be loaded.
        self.frames
            .retain(|frame_id, _| load_frame_ids.contains(frame_id));

        // Load images and their keypoints. When load_all_images is false, only images with
        // correspondences are loaded, as images without matches are not useful for SfM. When
        // load_all_images is true, all candidate images are loaded so that their keypoints
        // are populated (e.g., for triangulation on an existing reconstruction).
        for mut image in images {
            if !load_frame_ids.contains(&image.frame_id()) {
                continue;
            }
            let image_id = image.image_id();
            let points: Vec<Vector2d> = database
                .read_keypoints(image_id)?
                .iter()
                .map(|keypoint| Vector2d::new(f64::from(keypoint.x), f64::from(keypoint.y)))
                .collect();
            image.set_points2d_from_xy(&points);
            self.images.entry(image_id).or_insert(image);
        }

        // Load pose priors.
        self.pose_priors = database.read_all_pose_priors()?;
        if options.convert_pose_priors_to_enu {
            self.convert_pose_priors_to_enu()?;
        }

        // Build correspondence graph.
        self.correspondence_graph = CorrespondenceGraph::new();
        for (&image_id, image) in &self.images {
            self.correspondence_graph
                .add_image(image_id, image.num_points2d() as usize)?;
        }
        for (pair_id, two_view_geometry) in two_view_geometries {
            if use_inlier_matches_check(
                options,
                two_view_geometry.config,
                two_view_geometry.inlier_matches.len(),
            ) {
                let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
                let frame_id1 = at(&image_to_frame_id, &image_id1)?;
                let frame_id2 = at(&image_to_frame_id, &image_id2)?;
                if frame_ids.contains(&frame_id1) && frame_ids.contains(&frame_id2) {
                    self.correspondence_graph.add_two_view_geometry(
                        image_id1,
                        image_id2,
                        two_view_geometry,
                    )?;
                }
            }
        }
        self.correspondence_graph.finalize()
    }

    /// `CreateFromCache`: a cache holding only the images of `database_cache` selected by
    /// `options` (whole frames: every image of a frame with a selected image) and their
    /// cameras, rigs and correspondences, plus all pose priors.
    pub fn create_from_cache(
        database_cache: &DatabaseCache,
        options: &DatabaseCacheOptions,
    ) -> Result<Self> {
        let mut cache = Self::new();

        // Collect candidate image ids matching the name filter. Empty image_names means use
        // all images.
        let candidate_image_ids: HashSet<ImageId> = database_cache
            .images
            .iter()
            .filter(|(_, image)| {
                options.image_names.is_empty() || options.image_names.contains(image.name())
            })
            .map(|(&image_id, _)| image_id)
            .collect();

        let source_graph = &database_cache.correspondence_graph;

        let mut connected_image_ids = HashSet::new();
        if !options.load_all_images {
            for (pair_id, num_matches) in source_graph.num_matches_between_all_images() {
                let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
                if !candidate_image_ids.contains(&image_id1)
                    || !candidate_image_ids.contains(&image_id2)
                {
                    continue;
                }
                let two_view_geometry =
                    source_graph.extract_two_view_geometry(image_id1, image_id2, false)?;
                if !use_inlier_matches_check(
                    options,
                    two_view_geometry.config,
                    num_matches as usize,
                ) {
                    continue;
                }
                connected_image_ids.insert(image_id1);
                connected_image_ids.insert(image_id2);
            }
        }

        let load_image_ids = if options.load_all_images {
            &candidate_image_ids
        } else {
            &connected_image_ids
        };

        // Collect frame ids for images to load.
        let mut filtered_frame_ids = HashSet::new();
        for &image_id in load_image_ids {
            filtered_frame_ids.insert(database_cache.image(image_id)?.frame_id());
        }

        // Copy all images of filtered frames (not just the images matching the name filter).
        // This is needed for multi-camera rigs where the generalized pose solver needs all
        // images of a frame.
        let mut filtered_camera_ids = HashSet::new();
        for (&image_id, image) in &database_cache.images {
            if filtered_frame_ids.contains(&image.frame_id()) {
                cache.images.insert(image_id, image.clone());
                filtered_camera_ids.insert(image.camera_id());
            }
        }

        // Copy filtered frames and collect rig ids.
        let mut filtered_rig_ids = HashSet::new();
        for (&frame_id, frame) in &database_cache.frames {
            if filtered_frame_ids.contains(&frame_id) {
                cache.frames.insert(frame_id, frame.clone());
                filtered_rig_ids.insert(frame.rig_id());
            }
        }

        // Copy filtered cameras.
        for (&camera_id, camera) in &database_cache.cameras {
            if filtered_camera_ids.contains(&camera_id) {
                cache.cameras.insert(camera_id, camera.clone());
            }
        }

        // Copy filtered rigs.
        for (&rig_id, rig) in &database_cache.rigs {
            if filtered_rig_ids.contains(&rig_id) {
                cache.rigs.insert(rig_id, rig.clone());
            }
        }

        // Copy pose priors.
        cache.pose_priors = database_cache.pose_priors.clone();
        if options.convert_pose_priors_to_enu {
            cache.convert_pose_priors_to_enu()?;
        }

        // Build filtered correspondence graph with all images from connected frames.
        for (&image_id, image) in &cache.images {
            cache
                .correspondence_graph
                .add_image(image_id, image.num_points2d() as usize)?;
        }

        // Copy correspondences between all image pairs in the cache.
        for pair_id in source_graph.image_pairs() {
            let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
            if cache.images.contains_key(&image_id1) && cache.images.contains_key(&image_id2) {
                cache.correspondence_graph.add_two_view_geometry(
                    image_id1,
                    image_id2,
                    source_graph.extract_two_view_geometry(image_id1, image_id2, true)?,
                )?;
            }
        }

        cache.correspondence_graph.finalize()?;
        Ok(cache)
    }

    /// `NumRigs`.
    pub fn num_rigs(&self) -> usize {
        self.rigs.len()
    }

    /// `NumCameras`.
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    /// `NumFrames`.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// `NumImages`.
    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    /// `NumPosePriors`.
    pub fn num_pose_priors(&self) -> usize {
        self.pose_priors.len()
    }

    /// `AddRig`. Fails if the id exists.
    pub fn add_rig(&mut self, rig: Rig) -> Result<()> {
        check!(!self.exists_rig(rig.rig_id()));
        self.rigs.insert(rig.rig_id(), rig);
        Ok(())
    }

    /// `AddCamera`. Fails if the id exists.
    pub fn add_camera(&mut self, camera: Camera) -> Result<()> {
        check!(!self.exists_camera(camera.camera_id));
        self.cameras.insert(camera.camera_id, camera);
        Ok(())
    }

    /// `AddFrame`. Fails if the id exists.
    pub fn add_frame(&mut self, frame: Frame) -> Result<()> {
        check!(!self.exists_frame(frame.frame_id()));
        self.frames.insert(frame.frame_id(), frame);
        Ok(())
    }

    /// `AddImage`: also adds the image to the correspondence graph. Fails if the id exists.
    pub fn add_image(&mut self, image: Image) -> Result<()> {
        let image_id = image.image_id();
        check!(!self.exists_image(image_id));
        self.correspondence_graph
            .add_image(image_id, image.num_points2d() as usize)?;
        self.images.insert(image_id, image);
        Ok(())
    }

    /// `AddPosePrior`.
    pub fn add_pose_prior(&mut self, pose_prior: PosePrior) {
        self.pose_priors.push(pose_prior);
    }

    /// `Rig(rig_id)`. Fails if missing.
    pub fn rig(&self, rig_id: RigId) -> Result<&Rig> {
        self.rigs.get(&rig_id).ok_or_else(|| missing("rig", rig_id))
    }

    /// `Rig(rig_id)` (non-const).
    pub fn rig_mut(&mut self, rig_id: RigId) -> Result<&mut Rig> {
        self.rigs
            .get_mut(&rig_id)
            .ok_or_else(|| missing("rig", rig_id))
    }

    /// `Camera(camera_id)`. Fails if missing.
    pub fn camera(&self, camera_id: CameraId) -> Result<&Camera> {
        self.cameras
            .get(&camera_id)
            .ok_or_else(|| missing("camera", camera_id))
    }

    /// `Camera(camera_id)` (non-const).
    pub fn camera_mut(&mut self, camera_id: CameraId) -> Result<&mut Camera> {
        self.cameras
            .get_mut(&camera_id)
            .ok_or_else(|| missing("camera", camera_id))
    }

    /// `Frame(frame_id)`. Fails if missing.
    pub fn frame(&self, frame_id: FrameId) -> Result<&Frame> {
        self.frames
            .get(&frame_id)
            .ok_or_else(|| missing("frame", frame_id))
    }

    /// `Frame(frame_id)` (non-const).
    pub fn frame_mut(&mut self, frame_id: FrameId) -> Result<&mut Frame> {
        self.frames
            .get_mut(&frame_id)
            .ok_or_else(|| missing("frame", frame_id))
    }

    /// `Image(image_id)`. Fails if missing.
    pub fn image(&self, image_id: ImageId) -> Result<&Image> {
        self.images
            .get(&image_id)
            .ok_or_else(|| missing("image", image_id))
    }

    /// `Image(image_id)` (non-const).
    pub fn image_mut(&mut self, image_id: ImageId) -> Result<&mut Image> {
        self.images
            .get_mut(&image_id)
            .ok_or_else(|| missing("image", image_id))
    }

    /// `Rigs()`, in ascending id order.
    pub fn rigs(&self) -> &BTreeMap<RigId, Rig> {
        &self.rigs
    }

    /// `Cameras()`, in ascending id order.
    pub fn cameras(&self) -> &BTreeMap<CameraId, Camera> {
        &self.cameras
    }

    /// `Frames()`, in ascending id order.
    pub fn frames(&self) -> &BTreeMap<FrameId, Frame> {
        &self.frames
    }

    /// `Images()`, in ascending id order.
    pub fn images(&self) -> &BTreeMap<ImageId, Image> {
        &self.images
    }

    /// `PosePriors()`.
    pub fn pose_priors(&self) -> &[PosePrior] {
        &self.pose_priors
    }

    /// `ExistsRig`.
    pub fn exists_rig(&self, rig_id: RigId) -> bool {
        self.rigs.contains_key(&rig_id)
    }

    /// `ExistsCamera`.
    pub fn exists_camera(&self, camera_id: CameraId) -> bool {
        self.cameras.contains_key(&camera_id)
    }

    /// `ExistsFrame`.
    pub fn exists_frame(&self, frame_id: FrameId) -> bool {
        self.frames.contains_key(&frame_id)
    }

    /// `ExistsImage`.
    pub fn exists_image(&self, image_id: ImageId) -> bool {
        self.images.contains_key(&image_id)
    }

    /// `CorrespondenceGraph()` (const).
    pub fn correspondence_graph(&self) -> &CorrespondenceGraph {
        &self.correspondence_graph
    }

    /// `CorrespondenceGraph()` (non-const).
    pub fn correspondence_graph_mut(&mut self) -> &mut CorrespondenceGraph {
        &mut self.correspondence_graph
    }

    /// `FindImageWithName`: the first image (in ascending id order) with this name. Linear
    /// search.
    pub fn find_image_with_name(&self, name: &str) -> Option<&Image> {
        self.images.values().find(|image| image.name() == name)
    }

    /// `ConvertPosePriorsToENU`: when every pose prior is WGS84, converts the positions to
    /// Cartesian ENU coordinates around the first prior. Mixed coordinate systems fail.
    fn convert_pose_priors_to_enu(&mut self) -> Result<()> {
        let mut prior_is_gps = true;
        let mut gps_prior_positions = Vec::new();
        // `std::set` of the systems: only its size is used.
        let mut coordinate_systems = HashSet::new();
        for pose_prior in &self.pose_priors {
            coordinate_systems.insert(pose_prior.coordinate_system);
            if pose_prior.coordinate_system != CoordinateSystem::Wgs84 {
                prior_is_gps = false;
            } else {
                gps_prior_positions.push(pose_prior.position);
            }
        }

        check_le!(
            coordinate_systems.len(),
            1,
            "Inconsistent coordinate systems defined in pose priors"
        );

        // If GPS priors are available, convert them to Cartesian ENU coordinates.
        if prior_is_gps && !gps_prior_positions.is_empty() {
            // GPS reference to be used for EllipsoidToENU conversion.
            let reference: Vector3d = gps_prior_positions[0];
            let gps_transform = GpsTransform::new(Ellipsoid::Wgs84);
            let v_xyz_prior = gps_transform.ellipsoid_to_enu(
                &gps_prior_positions,
                reference[0],
                reference[1],
                reference[2],
            );
            for (pose_prior, xyz_prior) in self.pose_priors.iter_mut().zip(v_xyz_prior) {
                pose_prior.position = xyz_prior;
                pose_prior.coordinate_system = CoordinateSystem::Cartesian;
            }
        }
        Ok(())
    }
}
