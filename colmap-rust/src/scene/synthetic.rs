//! Port of `colmap/scene/synthetic.h` / `.cc` (and colmap-sharp's `Scene/Synthetic.cs`): the
//! synthetic dataset generator most of COLMAP's downstream tests build their scenes with.
//! [`synthesize_dataset`] puts 3D points on the unit sphere and rigs of cameras on a sphere of
//! radius 5 looking at the origin, and projects the points into every image. Neighbors:
//! [`crate::scene::reconstruction`] (the model it fills). Tests: `tests/scene/synthetic.rs`
//! (`synthetic_test.cc` 1:1 for the database-free tests); most `tests/scene/reconstruction*`
//! tests build their scenes with it.
//!
//! Tier A for ids, counts and poses given the PRNG stream; the 3D points' order of visits
//! differs from COLMAP's hash order (docs/CPP_DIVERGENCES.md entry 240).
//!
//! Translation notes:
//! - Randomness: every draw goes through `math::random` (COLMAP's thread-local PRNG) in
//!   COLMAP's order. Where C++ builds a value from several draws in one expression
//!   (`Eigen::Vector3d(RandomGaussian(...), RandomGaussian(...), ...)`), they are drawn left to
//!   right, as clang (the macOS pycolmap wheel) evaluates constructor arguments.
//! - Iteration order: COLMAP visits `Reconstruction::Points3D()` in hash order; here it is
//!   ascending id order (entries 213 and 240).
//! - The `Database*` argument and the options only it reads (feature type, matches, two-view
//!   geometries, pose priors) are not ported yet: they land with the database (Phase 4).
//!   `SynthesizeNoise` and `SynthesizeImages` follow in the same later slice.

use std::collections::BTreeSet;

use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{AngleAxisd, Quaterniond, Vector2d, Vector3d, Vector3ub};
use crate::math::random::{self, libcxx};
use crate::math::random_eigen::random_eigen_vector3d;
use crate::math::utils::deg_to_rad;
use crate::scene::camera::Camera;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::point2d::Point2D;
use crate::scene::reconstruction::Reconstruction;
use crate::scene::track::Track;
use crate::sensor::models::CameraModelId;
use crate::sensor::rig::Rig;
use crate::util::types::{CameraId, FrameId, ImageId, Point2DIdx, Point3DId, RigId, SensorId};
use crate::Result;

/// Port of `colmap::SyntheticDatasetOptions`, the fields the database-free generator reads
/// (see the header), with COLMAP's defaults.
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

    pub num_points2d_without_point3d: i32,

    /// The synthesized image file extension.
    pub image_extension: String,
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
            num_points2d_without_point3d: 10,
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
    crate::check_gt!(options.num_rigs, 0);
    crate::check_gt!(options.num_cameras_per_rig, 0);
    crate::check_gt!(options.num_frames_per_rig, 0);
    crate::check_ge!(options.num_points3d, 0);
    crate::check!(options.track_length == -1 || options.track_length >= 2);
    // COLMAP warns when track_length exceeds the number of images; pruning then removes
    // nothing (no track is longer), and the core has no logger (entry 211's reasoning).
    crate::check_ge!(options.num_points2d_without_point3d, 0);
    crate::check_ge!(options.sensor_from_rig_translation_stddev, 0.0);
    crate::check_ge!(options.sensor_from_rig_rotation_stddev, 0.0);
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

    let mut total_num_images: ImageId = 0;

    for rig_idx in 0..options.num_rigs {
        let mut rig = Rig::new();

        let mut camera_sensor_ids: Vec<SensorId> =
            Vec::with_capacity(options.num_cameras_per_rig as usize);
        for camera_idx in 0..options.num_cameras_per_rig {
            let camera = Camera {
                camera_id: (rig_idx * options.num_cameras_per_rig + camera_idx + 1) as CameraId,
                model_id: options.camera_model_id,
                width: options.camera_width,
                height: options.camera_height,
                params: options.camera_params.clone(),
                has_prior_focal_length: options.camera_has_prior_focal_length,
            };
            crate::check!(camera.verify_params()?);
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

        rig.set_rig_id((rig_idx + 1) as RigId);
        reconstruction.add_rig(rig.clone())?;

        for frame_idx in 0..options.num_frames_per_rig {
            synthesize_frame(
                options,
                reconstruction,
                &rig,
                &camera_sensor_ids,
                rig_idx,
                frame_idx,
                &new_points3d_ids,
                &mut total_num_images,
            )?;
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

/// One posed frame of `rig` with an image per camera sensor, added with its images.
#[allow(clippy::too_many_arguments)]
fn synthesize_frame(
    options: &SyntheticDatasetOptions,
    reconstruction: &mut Reconstruction,
    rig: &Rig,
    camera_sensor_ids: &[SensorId],
    rig_idx: i32,
    frame_idx: i32,
    new_points3d_ids: &BTreeSet<Point3DId>,
    total_num_images: &mut ImageId,
) -> Result<()> {
    let mut frame = Frame::new();
    frame.set_rig_id(rig.rig_id())?;

    // Synthesize frames as sphere centered at world origin.
    let view_dir = -random_eigen_vector3d().normalized();
    let proj_center = view_dir * -5.0;
    let rotation = Quaterniond::from_two_vectors(view_dir, Vector3d::new(0.0, 0.0, 1.0));
    let rig_from_world = Rigid3d::new(rotation, rotation * -proj_center);
    frame.set_rig_from_world(rig_from_world);

    let mut images: Vec<Image> = Vec::with_capacity(camera_sensor_ids.len());
    let mut cams_from_world: Vec<Rigid3d> = Vec::with_capacity(camera_sensor_ids.len());
    for sensor_id in camera_sensor_ids {
        *total_num_images += 1;

        let mut image = Image::new();
        image.set_name(&format!(
            "camera{:06}_frame{:06}{}",
            sensor_id.id, frame_idx, options.image_extension
        ));
        image.set_camera_id(sensor_id.id)?;
        image.set_image_id(*total_num_images);

        frame.add_data_id(image.data_id())?;

        // Need to compose cam_from_world manually, because the frame is not yet in the
        // reconstruction.
        let camera = reconstruction.camera(image.camera_id())?;
        let sensor_from_rig = if rig.is_ref_sensor(camera.sensor_id()) {
            Rigid3d::identity()
        } else {
            rig.sensor_from_rig(camera.sensor_id())?
        };
        cams_from_world.push(sensor_from_rig * rig_from_world);
        images.push(image);
    }

    let frame_id = (rig_idx * options.num_frames_per_rig + frame_idx + 1) as FrameId;
    frame.set_frame_id(frame_id);
    reconstruction.add_frame(frame)?;

    for (mut image, cam_from_world) in images.into_iter().zip(cams_from_world) {
        image.set_frame_id(frame_id);
        let camera = reconstruction.camera(image.camera_id())?;
        let points2d = synthesize_points2d(
            options,
            reconstruction,
            new_points3d_ids,
            camera,
            &cam_from_world,
        )?;

        for (point2d_idx, point2d) in points2d.iter().enumerate() {
            if point2d.has_point3d() {
                reconstruction
                    .point3d_mut(point2d.point3d_id)?
                    .track
                    .add_element_ids(image.image_id(), point2d_idx as Point2DIdx);
            }
        }

        image.set_points2d(points2d)?;
        reconstruction.add_image(image)?;
    }
    Ok(())
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
/// subset. Points are visited in ascending id order (entry 240).
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
