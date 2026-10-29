//! Port of `colmap/scene/synthetic.h` / `.cc` (and colmap-sharp's `Scene/Synthetic.cs`): the
//! synthetic dataset generator most of COLMAP's downstream tests build their scenes with.
//! [`synthesize_dataset`] puts 3D points on the unit sphere and rigs of cameras on a sphere of
//! radius 5 looking at the origin, and projects the points into every image;
//! [`synthesize_dataset_with_database`] also writes the rigs, cameras, frames, images, pose
//! priors, keypoints, descriptors, matches and two-view geometries into a
//! [`crate::scene::database::Database`], as COLMAP's `Database*` argument does. Neighbors:
//! [`crate::scene::reconstruction`] (the model it fills), [`super::synthetic_matches`] (the
//! match configurations and pose prior coordinates) and [`super::synthetic_noise`]
//! (`SynthesizeNoise`, `SynthesizeImages`). Tests: `tests/scene/synthetic.rs`
//! (`synthetic_test.cc` 1:1); most `tests/scene/reconstruction*` tests build their scenes with
//! it.
//!
//! Tier A for ids, counts and poses given the PRNG stream; the 3D points' order of visits
//! differs from COLMAP's hash order (the iteration-order divergence below).
//!
//! Translation notes:
//! - Randomness: every draw goes through `math::random` (COLMAP's thread-local PRNG) in
//!   COLMAP's order. Where C++ builds a value from several draws in one expression
//!   (`Eigen::Vector3d(RandomGaussian(...), RandomGaussian(...), ...)`), they are drawn left to
//!   right, as clang (the macOS pycolmap wheel) evaluates constructor arguments.
//! - Iteration order: COLMAP visits `Reconstruction::Points3D()` in hash order; here it is
//!   ascending id order (the `scene/reconstruction.rs` header and the divergence below).
//! - COLMAP's optional `Database*` is two functions: [`synthesize_dataset`] (no database, the
//!   form every other module's tests call) and [`synthesize_dataset_with_database`].
//!
//! Deliberate divergence, iteration order (here, in [`super::synthetic_matches`] and in
//! [`super::synthetic_noise`]): COLMAP's `SynthesizeDataset` projects the reconstruction's 3D
//! points into each image, and `track_length` pruning visits them, in
//! `Reconstruction::Points3D()` order, which is the order of COLMAP's hash map; its CHAINED
//! match configuration writes the pairs of a `NodeHashMap`; `SynthesizeNoise` adds 2D and 3D
//! point noise visiting `Images()` and `Points3D()`, and `SynthesizeImages` renders the images
//! in `Images()` order. The three files visit all of them in ascending id order (the
//! reconstruction's `BTreeMap`s; a `BTreeMap` of pairs for CHAINED). The PRNG draws are the
//! same and in the same sequence; what differs is the order of each image's 2D points before
//! the shuffle, which point's track receives which pruning shuffle, which pair receives which
//! outlier-match shuffle, which point receives which noise draws, and the descriptor seeds
//! `SynthesizeImages` gives points without a 3D point. So a seeded run gives the same cameras,
//! rigs, frame poses and 3D point positions as pycolmap's, but different 2D point orders,
//! pruned tracks and noise assignment. CLAUDE.md requires deterministic iteration, and
//! COLMAP's tests rely only on order-free properties (counts, track lengths, error bounds,
//! connectivity, equality of two seeded runs): `tests/scene/synthetic.rs`,
//! `synthetic_database.rs` and `synthetic_noise.rs` (`synthetic_test.cc` 1:1, all 18 cases)
//! and every test that synthesizes its scene pass. Same as colmap-sharp entry 31.

use std::collections::BTreeSet;

use crate::feature::types::{
    FeatureDescriptors, FeatureDescriptorsData, FeatureExtractorType, FeatureKeypoint,
};
use crate::geometry::pose_prior::{CoordinateSystem, PosePrior};
use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{AngleAxisd, Quaterniond, Vector2d, Vector3d, Vector3ub};
use crate::math::random::mt19937::Mt19937;
use crate::math::random::{self, libcxx};
use crate::math::random_eigen::random_eigen_vector3d;
use crate::math::utils::deg_to_rad;
use crate::scene::camera::Camera;
use crate::scene::database::Database;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::point2d::Point2D;
use crate::scene::reconstruction::Reconstruction;
use crate::scene::track::Track;
use crate::sensor::models::CameraModelId;
use crate::sensor::rig::Rig;
use crate::util::types::{CameraId, FrameId, ImageId, Point2DIdx, Point3DId, RigId, SensorId};

use super::synthetic_matches::{
    pose_prior_position_cartesian_to_wgs84, synthesize_chained_matches,
    synthesize_exhaustive_matches, synthesize_sparse_matches,
};
use crate::Result;

/// Port of `colmap::SyntheticDatasetOptions`, with COLMAP's defaults. The feature type,
/// match, two-view geometry and pose prior options are read only with a database.
#[derive(Debug, Clone, PartialEq)]
pub struct SyntheticDatasetOptions {
    pub num_rigs: i32,
    pub num_cameras_per_rig: i32,
    pub num_frames_per_rig: i32,
    pub num_points3d: i32,
    /// Target track length per 3D point. If -1 (default), all images observe all points
    /// (dense visibility). If > 0, observations are pruned to exactly this many per point.
    /// Must be -1 or >= 2.
    pub track_length: i32,

    pub sensor_from_rig_translation_stddev: f64,
    /// Random rotation in degrees around the z-axis of the sensor.
    pub sensor_from_rig_rotation_stddev: f64,

    pub camera_width: usize,
    pub camera_height: usize,
    pub camera_model_id: CameraModelId,
    pub camera_params: Vec<f64>,
    pub camera_has_prior_focal_length: bool,

    /// The type of feature descriptors to synthesize.
    pub feature_type: FeatureExtractorType,

    pub num_points2d_without_point3d: i32,

    pub inlier_match_ratio: f64,

    /// Whether to include decomposed relative poses in two-view geometries.
    pub two_view_geometry_has_relative_pose: bool,

    pub match_config: MatchConfig,

    /// Sparsity parameter for [`MatchConfig::Sparse`], in range [0, 1].
    /// 0 = fully connected view graph, equivalent to EXHAUSTIVE (all edges)
    /// 1 = empty view graph (no edges)
    pub match_sparsity: f64,

    pub prior_position: bool,
    pub prior_position_coordinate_system: CoordinateSystem,
    pub prior_gravity: bool,
    pub prior_gravity_in_world: Vector3d,

    /// The synthesized image file extension.
    pub image_extension: String,
}

/// Port of `SyntheticDatasetOptions::MatchConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MatchConfig {
    /// Exhaustive matches between all pairs of observations of a 3D point.
    #[default]
    Exhaustive = 1,
    /// Chain of matches between images with consecutive identifiers, i.e.,
    /// there are only matches between image pairs (image_id, image_id+1).
    Chained = 2,
    /// Sparse matches with controllable sparsity, removing edges randomly while
    /// maintaining view graph connectivity.
    Sparse = 3,
}

impl Default for SyntheticDatasetOptions {
    fn default() -> Self {
        Self {
            num_rigs: 2,
            num_cameras_per_rig: 1,
            num_frames_per_rig: 5,
            num_points3d: 100,
            track_length: -1,
            sensor_from_rig_translation_stddev: 0.05,
            sensor_from_rig_rotation_stddev: 5.0,
            camera_width: 1024,
            camera_height: 768,
            camera_model_id: CameraModelId::SimpleRadial,
            camera_params: vec![1280.0, 512.0, 384.0, 0.05],
            camera_has_prior_focal_length: false,
            feature_type: FeatureExtractorType::Sift,
            num_points2d_without_point3d: 10,
            inlier_match_ratio: 1.0,
            two_view_geometry_has_relative_pose: false,
            match_config: MatchConfig::Exhaustive,
            match_sparsity: 0.0,
            prior_position: false,
            prior_position_coordinate_system: CoordinateSystem::Cartesian,
            prior_gravity: false,
            prior_gravity_in_world: Vector3d::new(0.0, 1.0, 0.0),
            image_extension: ".png".to_owned(),
        }
    }
}

/// Port of `colmap::SynthesizeDataset(options, reconstruction)` (no database): adds
/// `num_rigs` rigs of `num_cameras_per_rig` cameras with `num_frames_per_rig` posed,
/// registered frames each, and `num_points3d` points on the unit sphere observed by every
/// image they project into, plus `num_points2d_without_point3d` untriangulated points per
/// image. Ids continue from 1 as COLMAP's database-free path assigns them, so a second call
/// on the same reconstruction fails on the repeated camera id, as COLMAP's does.
pub fn synthesize_dataset(
    options: &SyntheticDatasetOptions,
    reconstruction: &mut Reconstruction,
) -> Result<()> {
    synthesize(options, reconstruction, None)
}

/// Port of `colmap::SynthesizeDataset(options, reconstruction, database)`: as
/// [`synthesize_dataset`], but the ids of rigs, cameras, frames and images are the ones the
/// database assigns, so the function can be called repeatedly on the same reconstruction and
/// database. Each image's keypoints and descriptors (identical rows for the observations of
/// one 3D point, unique random ones otherwise), pose priors when asked for, and the matches
/// and two-view geometries of `options.match_config` are written as well.
pub fn synthesize_dataset_with_database(
    options: &SyntheticDatasetOptions,
    reconstruction: &mut Reconstruction,
    database: &Database,
) -> Result<()> {
    synthesize(options, reconstruction, Some(database))
}

fn synthesize(
    options: &SyntheticDatasetOptions,
    reconstruction: &mut Reconstruction,
    database: Option<&Database>,
) -> Result<()> {
    crate::check_gt!(options.num_rigs, 0);
    crate::check_gt!(options.num_cameras_per_rig, 0);
    crate::check_gt!(options.num_frames_per_rig, 0);
    crate::check_ge!(options.num_points3d, 0);
    crate::check!(options.track_length == -1 || options.track_length >= 2);
    // COLMAP warns when track_length exceeds the number of images; pruning then removes
    // nothing (no track is longer), and the core has no logger (the line is diagnostic only).
    crate::check_ne!(options.feature_type, FeatureExtractorType::Undefined);
    crate::check_ge!(options.num_points2d_without_point3d, 0);
    crate::check_ge!(options.sensor_from_rig_translation_stddev, 0.0);
    crate::check_ge!(options.sensor_from_rig_rotation_stddev, 0.0);
    crate::check_ge!(options.match_sparsity, 0.0);
    crate::check_le!(options.match_sparsity, 1.0);
    crate::check!(!options.image_extension.is_empty());
    crate::check!(options.image_extension.starts_with('.'));

    if !random::prng_is_set() {
        random::set_prng_seed_default();
    }

    // Synthesize 3D points on unit sphere centered at origin.
    let mut new_points3d_ids: BTreeSet<Point3DId> = BTreeSet::new();
    for _ in 0..options.num_points3d {
        new_points3d_ids.insert(reconstruction.add_point3d(
            random_eigen_vector3d().normalized(),
            Track::new(),
            Vector3ub::new(0, 0, 0),
        )?);
    }

    let mut counters = Counters {
        total_num_images: match database {
            None => 0,
            Some(database) => database.num_images()? as ImageId,
        },
        total_num_descriptors: match database {
            None => 0,
            // `int total_num_descriptors = database->NumDescriptors()`.
            Some(database) => database.num_descriptors()? as i32,
        },
    };

    let desc_dim = match options.feature_type {
        FeatureExtractorType::Sift => 128,
        FeatureExtractorType::Undefined => 0,
        other => {
            return Err(crate::util::check::ColmapError::new(
                crate::util::check::ErrorKind::InvalidArgument,
                format!("Invalid FeatureExtractorType specified: {}", other as i32),
            ))
        }
    };

    for rig_idx in 0..options.num_rigs {
        let mut rig = Rig::new();

        let mut camera_sensor_ids: Vec<SensorId> =
            Vec::with_capacity(options.num_cameras_per_rig as usize);
        for camera_idx in 0..options.num_cameras_per_rig {
            let mut camera = Camera {
                camera_id: (rig_idx * options.num_cameras_per_rig + camera_idx + 1) as CameraId,
                model_id: options.camera_model_id,
                width: options.camera_width,
                height: options.camera_height,
                params: options.camera_params.clone(),
                has_prior_focal_length: options.camera_has_prior_focal_length,
            };
            crate::check!(camera.verify_params()?);
            if let Some(database) = database {
                camera.camera_id = database.write_camera(&camera, false)?;
            }
            reconstruction.add_camera(camera.clone())?;

            if rig.num_sensors() == 0 {
                rig.add_ref_sensor(camera.sensor_id())?;
            } else {
                rig.add_sensor(
                    camera.sensor_id(),
                    Some(synthesize_sensor_from_rig(options)),
                )?;
            }

            camera_sensor_ids.push(camera.sensor_id());
        }

        let rig_id = match database {
            None => (rig_idx + 1) as RigId,
            Some(database) => database.write_rig(&rig, false)?,
        };
        rig.set_rig_id(rig_id);
        reconstruction.add_rig(rig.clone())?;

        for frame_idx in 0..options.num_frames_per_rig {
            synthesize_frame(
                &FrameContext {
                    options,
                    database,
                    rig: &rig,
                    camera_sensor_ids: &camera_sensor_ids,
                    new_points3d_ids: &new_points3d_ids,
                    desc_dim,
                },
                reconstruction,
                rig_idx,
                frame_idx,
                &mut counters,
            )?;
        }
    }

    if let Some(database) = database {
        match options.match_config {
            MatchConfig::Exhaustive => synthesize_exhaustive_matches(
                options.inlier_match_ratio,
                options.two_view_geometry_has_relative_pose,
                reconstruction,
                database,
            )?,
            MatchConfig::Chained => synthesize_chained_matches(
                options.inlier_match_ratio,
                options.two_view_geometry_has_relative_pose,
                reconstruction,
                database,
            )?,
            MatchConfig::Sparse => synthesize_sparse_matches(
                options.inlier_match_ratio,
                options.two_view_geometry_has_relative_pose,
                options.match_sparsity,
                reconstruction,
                database,
            )?,
        }
    }

    if options.track_length > 0 {
        prune_tracks(options.track_length, reconstruction)?;
    }

    reconstruction.update_point3d_errors()
}

/// The non-reference sensor's pose in its rig: a random rotation around the z-axis (which
/// keeps the 2D points in front of the camera) and a random offset.
fn synthesize_sensor_from_rig(options: &SyntheticDatasetOptions) -> Rigid3d {
    let mut sensor_from_rig = Rigid3d::identity();
    if options.sensor_from_rig_rotation_stddev > 0.0 {
        // Generate a random rotation around the Z-axis.
        // This is to avoid 2D points fall behind the camera.
        let angle = random::random_gaussian(0.0, options.sensor_from_rig_rotation_stddev)
            .clamp(-180.0, 180.0);
        sensor_from_rig.rotation = Quaterniond::from_angle_axis(AngleAxisd::new(
            deg_to_rad(angle),
            Vector3d::new(0.0, 0.0, 1.0),
        ));
    }
    if options.sensor_from_rig_translation_stddev > 0.0 {
        let x = random::random_gaussian(0.0, options.sensor_from_rig_translation_stddev);
        let y = random::random_gaussian(0.0, options.sensor_from_rig_translation_stddev);
        let z = random::random_gaussian(0.0, options.sensor_from_rig_translation_stddev);
        sensor_from_rig.translation = Vector3d::new(x, y, z);
    }
    sensor_from_rig
}

/// The ids `synthesize` continues from: the images and descriptors written so far.
struct Counters {
    total_num_images: ImageId,
    total_num_descriptors: i32,
}

/// What every frame of one rig is synthesized from.
struct FrameContext<'a> {
    options: &'a SyntheticDatasetOptions,
    database: Option<&'a Database>,
    rig: &'a Rig,
    camera_sensor_ids: &'a [SensorId],
    new_points3d_ids: &'a BTreeSet<Point3DId>,
    desc_dim: usize,
}

/// One posed frame of the context's rig with an image per camera sensor, added with its
/// images (and written to the database, with the images' pose priors, features and keypoints).
fn synthesize_frame(
    cx: &FrameContext<'_>,
    reconstruction: &mut Reconstruction,
    rig_idx: i32,
    frame_idx: i32,
    counters: &mut Counters,
) -> Result<()> {
    let options = cx.options;
    let rig = cx.rig;
    let mut frame = Frame::new();
    frame.set_rig_id(rig.rig_id())?;

    // Synthesize frames as sphere centered at world origin.
    let view_dir = -random_eigen_vector3d().normalized();
    let proj_center = view_dir * -5.0;
    let rotation = Quaterniond::from_two_vectors(view_dir, Vector3d::new(0.0, 0.0, 1.0));
    let rig_from_world = Rigid3d::new(rotation, rotation * -proj_center);
    frame.set_rig_from_world(rig_from_world);

    let mut images: Vec<Image> = Vec::with_capacity(cx.camera_sensor_ids.len());
    let mut cams_from_world: Vec<Rigid3d> = Vec::with_capacity(cx.camera_sensor_ids.len());
    for sensor_id in cx.camera_sensor_ids {
        counters.total_num_images += 1;

        let mut image = Image::new();
        image.set_name(&format!(
            "camera{:06}_frame{:06}{}",
            sensor_id.id, frame_idx, options.image_extension
        ));
        image.set_camera_id(sensor_id.id)?;
        let image_id = match cx.database {
            None => counters.total_num_images,
            Some(database) => database.write_image(&image, false)?,
        };
        image.set_image_id(image_id);

        frame.add_data_id(image.data_id())?;

        // Need to compose cam_from_world manually, because the frame is not yet in the
        // reconstruction.
        let camera = reconstruction.camera(image.camera_id())?;
        let sensor_from_rig = if rig.is_ref_sensor(camera.sensor_id()) {
            Rigid3d::identity()
        } else {
            rig.sensor_from_rig(camera.sensor_id())?
        };
        let cam_from_world = sensor_from_rig * rig_from_world;
        cams_from_world.push(cam_from_world);

        if options.prior_position || options.prior_gravity {
            write_pose_prior(options, cx.database, &image, &cam_from_world)?;
        }
        images.push(image);
    }

    let frame_id = match cx.database {
        None => (rig_idx * options.num_frames_per_rig + frame_idx + 1) as FrameId,
        Some(database) => database.write_frame(&frame, false)?,
    };
    frame.set_frame_id(frame_id);
    reconstruction.add_frame(frame)?;

    for (mut image, cam_from_world) in images.into_iter().zip(cams_from_world) {
        image.set_frame_id(frame_id);
        let camera = reconstruction.camera(image.camera_id())?;
        let points2d = synthesize_points2d(
            options,
            reconstruction,
            cx.new_points3d_ids,
            camera,
            &cam_from_world,
        )?;

        if let Some(database) = cx.database {
            write_features(cx, database, &image, &points2d, counters)?;
        }

        for (point2d_idx, point2d) in points2d.iter().enumerate() {
            if point2d.has_point3d() {
                reconstruction
                    .point3d_mut(point2d.point3d_id)?
                    .track
                    .add_element_ids(image.image_id(), point2d_idx as Point2DIdx);
            }
        }

        image.set_points2d(points2d)?;

        if let Some(database) = cx.database {
            database.update_image(&image)?;
        }
        reconstruction.add_image(image)?;
    }
    Ok(())
}

/// The image's pose prior: its projection center (in Cartesian or WGS84 coordinates) and the
/// world's gravity direction in the camera frame, as asked for. COLMAP writes it through the
/// database pointer unconditionally, so asking for priors without a database is an error.
fn write_pose_prior(
    options: &SyntheticDatasetOptions,
    database: Option<&Database>,
    image: &Image,
    cam_from_world: &Rigid3d,
) -> Result<()> {
    // Deliberate divergence: with `prior_position` or `prior_gravity` set, COLMAP writes the
    // prior through `database->WritePosePrior` without checking the pointer, so calling it
    // without a database dereferences null (undefined behavior, in practice a crash). A null
    // dereference is a bug, not behavior to reproduce: the port returns an `InvalidArgument`
    // error at the first image instead, and the database form behaves exactly as COLMAP's.
    // Pinned by `rust_only_synthesize_dataset_pose_priors_need_a_database`
    // (`tests/scene/synthetic_database.rs`).
    let Some(database) = database else {
        return Err(crate::util::check::ColmapError::new(
            crate::util::check::ErrorKind::InvalidArgument,
            "SynthesizeDataset: pose priors need a database",
        ));
    };
    let mut pose_prior = PosePrior::default();

    if options.prior_position {
        pose_prior.position = cam_from_world.tgt_origin_in_src();
        pose_prior.coordinate_system = CoordinateSystem::Cartesian;
        match options.prior_position_coordinate_system {
            CoordinateSystem::Cartesian => {}
            CoordinateSystem::Wgs84 => pose_prior_position_cartesian_to_wgs84(&mut pose_prior),
            CoordinateSystem::Undefined => {
                return Err(crate::util::check::ColmapError::new(
                    crate::util::check::ErrorKind::InvalidArgument,
                    "Invalid PosePrior::CoordinateSystem specified",
                ))
            }
        }
    }

    if options.prior_gravity {
        pose_prior.gravity =
            (cam_from_world.rotation * options.prior_gravity_in_world).normalized();
    }

    pose_prior.corr_data_id = image.data_id();
    database.write_pose_prior(&pose_prior, false)?;
    Ok(())
}

/// Writes the image's keypoints and a descriptor per keypoint: the rows of the observations
/// of one 3D point are identical (seeded by the point's id), the others unique random rows.
fn write_features(
    cx: &FrameContext<'_>,
    database: &Database,
    image: &Image,
    points2d: &[Point2D],
    counters: &mut Counters,
) -> Result<()> {
    let mut keypoints: Vec<FeatureKeypoint> = Vec::with_capacity(points2d.len());
    let mut data = FeatureDescriptorsData::new(points2d.len(), cx.desc_dim);
    for (point2d_idx, point2d) in points2d.iter().enumerate() {
        keypoints.push(FeatureKeypoint::new(
            point2d.xy.x as f32,
            point2d.xy.y as f32,
        ));
        // Generate a unique descriptor for each 3D point. If the 2D point does not observe a
        // 3D point, generate a random unique descriptor. `std::mt19937`'s seed is a
        // `uint_fast32_t` (32 bits on libc++), so the id is truncated as C++ converts it.
        let seed = if point2d.has_point3d() {
            point2d.point3d_id as u32
        } else {
            counters.total_num_descriptors += 1;
            (cx.options.num_points3d + counters.total_num_descriptors) as u32
        };
        let mut feature_generator = Mt19937::new(seed);
        for d in 0..cx.desc_dim {
            // `std::uniform_int_distribution<int>(0, 255)`, stored as `uint8_t`.
            data[(point2d_idx, d)] = libcxx::uniform_int(&mut feature_generator, 0i32, 255) as u8;
        }
    }
    database.write_keypoints(image.image_id(), &keypoints)?;
    database.write_descriptors(
        image.image_id(),
        &FeatureDescriptors::new(cx.options.feature_type, data),
    )
}

/// The image's 2D points: the projections of the new 3D points that land inside the image,
/// then uniform random points without a 3D point, shuffled.
fn synthesize_points2d(
    options: &SyntheticDatasetOptions,
    reconstruction: &Reconstruction,
    new_points3d_ids: &BTreeSet<Point3DId>,
    camera: &Camera,
    cam_from_world: &Rigid3d,
) -> Result<Vec<Point2D>> {
    let mut points2d: Vec<Point2D> =
        Vec::with_capacity((options.num_points3d + options.num_points2d_without_point3d) as usize);
    let width = camera.width as f64;
    let height = camera.height as f64;

    // Create 3D point observations by projecting 3D points to the image.
    for (&point3d_id, point3d) in reconstruction.points3d() {
        if !new_points3d_ids.contains(&point3d_id) {
            // If a non-empty reconstruction is given, only add tracks for newly added
            // images and 3D points.
            continue;
        }
        let proj_point2d = camera.img_from_cam(*cam_from_world * point3d.xyz, true);
        crate::check!(proj_point2d.is_some());
        let xy = proj_point2d.unwrap_or_default();
        if xy.x >= 0.0 && xy.y >= 0.0 && xy.x <= width && xy.y <= height {
            points2d.push(Point2D { xy, point3d_id });
        }
    }

    // Synthesize uniform random 2D points without 3D points.
    for _ in 0..options.num_points2d_without_point3d {
        let x = random::random_uniform_real(0.0, width);
        let y = random::random_uniform_real(0.0, height);
        points2d.push(Point2D {
            xy: Vector2d::new(x, y),
            ..Point2D::default()
        });
    }

    // Shuffle 2D points, so each image has 3D points ordered differently.
    random::with_prng(|g| libcxx::shuffle(&mut points2d, g));
    Ok(points2d)
}

/// Prunes every longer track to exactly `track_length` observations, dropping a random
/// subset. Points are visited in ascending id order (the iteration-order
/// divergence in the file header).
fn prune_tracks(track_length: i32, reconstruction: &mut Reconstruction) -> Result<()> {
    for point3d_id in reconstruction.point3d_ids() {
        let track = &reconstruction.point3d(point3d_id)?.track;
        if track.length() as i32 <= track_length {
            continue;
        }
        let mut elements = track.elements().to_vec();
        random::with_prng(|g| libcxx::shuffle(&mut elements, g));
        let num_to_delete = elements.len() - track_length as usize;
        for element in &elements[..num_to_delete] {
            reconstruction.delete_observation(element.image_id, element.point2d_idx)?;
        }
    }
    Ok(())
}
