//! Port of the free functions of COLMAP's `rotation_averaging.h/.cc`
//! (`RunRotationAveraging`, `RunRotationAveragingOnComponent`,
//! `InitializeRigRotationsFromImages`, `FilterEdgesByRelativeRotation`) and the `.cc`'s
//! anonymous-namespace helpers (gravity checks, largest component, rig expansion for cameras
//! with unknown cam_from_rig). The estimator itself is [`super::estimator`]. Port of
//! colmap-sharp's `Estimators/RotationAveraging.cs`.
//!
//! Translation notes:
//! - Frames are walked in ascending id order where COLMAP walks `Reconstruction::Frames()`
//!   (a hash map): the quaternion averages of `InitializeRigRotationsFromImages` sum their
//!   samples in that order, and `CreateExpandedReconstruction` numbers the singleton rigs
//!   (ascending rig id, then sensor) and the new frames (ascending frame id, then data id) in
//!   it (the ascending id divergence in the [`super`] module header).
//! - `LOG(ERROR)` paths return `Ok(false)`; `LOG(INFO)` messages are not emitted.

use std::collections::{BTreeMap, HashSet};

use super::{
    cam_from_world_rotation, camera_sensor_id, has_nan3, is_ref_in_frame, rotation_only_pose,
    RotationEstimator, RotationEstimatorOptions,
};
use crate::estimators::alignment::image_has_pose;
use crate::geometry::pose::average_quaternions;
use crate::geometry::{PosePrior, Rigid3d};
use crate::linalg::{Quaterniond, Vector3d};
use crate::math::utils::deg_to_rad;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::pose_graph::PoseGraph;
use crate::scene::reconstruction::Reconstruction;
use crate::sensor::rig::Rig;
use crate::util::types::{
    pair_id_to_image_pair, CameraId, FrameId, ImageId, RigId, SensorId, SensorType,
};
use crate::Result;

/// `UseGravity`: gravity mode is on and some prior has a gravity.
pub(super) fn use_gravity(options: &RotationEstimatorOptions, pose_priors: &[PosePrior]) -> bool {
    options.use_gravity && pose_priors.iter().any(PosePrior::has_gravity)
}

/// `AllSensorsFromRigKnown`: false if any rig has a sensor with unknown sensor_from_rig
/// (COLMAP logs an error for each: gravity aligned rotation needs the rig calibration).
pub(super) fn all_sensors_from_rig_known(reconstruction: &Reconstruction) -> bool {
    reconstruction
        .rigs()
        .values()
        .all(|rig| rig.non_ref_sensors().values().all(Option::is_some))
}

/// `ComputeLargestConnectedComponentImageIds`: the image ids of the pose graph's largest
/// connected frame component.
pub(super) fn compute_largest_connected_component_image_ids(
    pose_graph: &PoseGraph,
    reconstruction: &Reconstruction,
    filter_unregistered: bool,
) -> Result<HashSet<ImageId>> {
    let frame_ids =
        pose_graph.largest_connected_frame_component(reconstruction, filter_unregistered)?;
    Ok(reconstruction
        .images()
        .iter()
        .filter(|(_, image)| frame_ids.contains(&image.frame_id()))
        .map(|(&image_id, _)| image_id)
        .collect())
}

/// `HasUnknownCamsFromRig`: true if any camera in the reconstruction has unknown
/// cam_from_rig.
fn has_unknown_cams_from_rig(reconstruction: &Reconstruction) -> bool {
    reconstruction.rigs().values().any(|rig| {
        rig.non_ref_sensors()
            .iter()
            .any(|(sensor_id, sensor_from_rig)| {
                sensor_id.sensor_type == SensorType::Camera && sensor_from_rig.is_none()
            })
    })
}

/// `CreateExpandedReconstruction`: a reconstruction where cameras with unknown cam_from_rig
/// are split into separate singleton rigs (each such camera becomes its own rig). This
/// allows rotation averaging to estimate their orientations independently.
fn create_expanded_reconstruction(reconstruction: &Reconstruction) -> Result<Reconstruction> {
    let mut recon_expanded = Reconstruction::new();

    // Add all cameras first (required before adding rigs).
    for camera in reconstruction.cameras().values() {
        recon_expanded.add_camera(camera.clone())?;
    }

    // Create expanded rigs with known sensors only. Cameras with unknown cam_from_rig get
    // their own singleton rigs.
    let mut singleton_rig_ids: BTreeMap<CameraId, RigId> = BTreeMap::new();

    // First, find the max rig ID to avoid conflicts when creating singleton rigs.
    let mut next_rig_id: RigId = 0;
    for &rig_id in reconstruction.rigs().keys() {
        next_rig_id = next_rig_id.max(rig_id + 1);
    }

    for (&rig_id, rig) in reconstruction.rigs() {
        let mut rig_expanded = Rig::new();
        rig_expanded.set_rig_id(rig_id);
        rig_expanded.add_ref_sensor(rig.ref_sensor_id())?;
        for (&sensor_id, sensor_from_rig) in rig.non_ref_sensors() {
            if sensor_id.sensor_type != SensorType::Camera {
                continue;
            }
            match sensor_from_rig {
                Some(_) => rig_expanded.add_sensor(sensor_id, *sensor_from_rig)?,
                None => {
                    // Create singleton rig for this camera.
                    let singleton_rig_id = next_rig_id;
                    next_rig_id += 1;
                    let mut rig_singleton = Rig::new();
                    rig_singleton.set_rig_id(singleton_rig_id);
                    rig_singleton.add_ref_sensor(sensor_id)?;
                    recon_expanded.add_rig(rig_singleton)?;
                    singleton_rig_ids.insert(sensor_id.id, singleton_rig_id);
                }
            }
        }
        recon_expanded.add_rig(rig_expanded)?;
    }

    let mut next_frame_id: FrameId = 0;
    for &frame_id in reconstruction.frames().keys() {
        next_frame_id = next_frame_id.max(frame_id + 1);
    }

    let unknown_pose = Rigid3d::new(
        Quaterniond::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        Vector3d::new(f64::NAN, f64::NAN, f64::NAN),
    );

    // First pass: build expanded frames with their data ids, and collect images to add
    // afterwards.
    let mut expanded_images: Vec<Image> = Vec::new();
    let mut expanded_frames: Vec<Frame> = Vec::new();
    for (&frame_id, frame) in reconstruction.frames() {
        let mut frame_expanded = Frame::new();
        frame_expanded.set_frame_id(frame_id);
        frame_expanded.set_rig_id(frame.rig_id())?;
        frame_expanded.set_rig_from_world(frame.maybe_rig_from_world().unwrap_or(unknown_pose));

        let original_rig = reconstruction.rig(frame.rig_id())?;
        for data_id in frame.image_ids() {
            // `static_cast<image_t>` of the 64-bit data id.
            let image = reconstruction.image(data_id.id as ImageId)?;

            let mut image_expanded = Image::new();
            image_expanded.set_image_id(image.image_id());
            image_expanded.set_camera_id(image.camera_id())?;
            image_expanded.set_name(image.name());

            // Check if camera belongs to this frame's rig (ref sensor or known cam_from_rig).
            let sensor_id = camera_sensor_id(image);
            let belongs_to_frame_rig = original_rig.ref_sensor_id() == sensor_id
                || original_rig.maybe_sensor_from_rig(sensor_id)?.is_some();

            if belongs_to_frame_rig {
                // Camera belongs to this frame's rig.
                frame_expanded.add_data_id(image_expanded.data_id())?;
                image_expanded.set_frame_id(frame_id);
            } else {
                // Camera has its own singleton rig, create a new frame for it.
                let new_frame_id = next_frame_id;
                next_frame_id += 1;
                let singleton_rig_id =
                    *singleton_rig_ids.get(&image.camera_id()).ok_or_else(|| {
                        crate::util::check::ColmapError::new(
                            crate::util::check::ErrorKind::OutOfRange,
                            format!("Camera {} has no singleton rig", image.camera_id()),
                        )
                    })?;
                let mut new_frame = Frame::new();
                new_frame.set_frame_id(new_frame_id);
                new_frame.set_rig_id(singleton_rig_id)?;
                new_frame.add_data_id(image_expanded.data_id())?;
                new_frame.set_rig_from_world(unknown_pose);
                expanded_frames.push(new_frame);
                image_expanded.set_frame_id(new_frame_id);
            }
            expanded_images.push(image_expanded);
        }
        expanded_frames.push(frame_expanded);
    }

    // Second pass: add all frames, then all images.
    for frame in expanded_frames {
        recon_expanded.add_frame(frame)?;
    }
    for image in expanded_images {
        recon_expanded.add_image(image)?;
    }
    Ok(recon_expanded)
}

/// Port of `colmap::FilterEdgesByRelativeRotation`: marks image pairs as invalid whose
/// relative rotation disagrees with the reconstructed rotations by more than
/// `max_angle_deg`. Pairs whose images do not both have a pose are left untouched.
pub fn filter_edges_by_relative_rotation(
    pose_graph: &mut PoseGraph,
    reconstruction: &Reconstruction,
    max_angle_deg: f64,
) -> Result<()> {
    let max_angle_rad = deg_to_rad(max_angle_deg);
    let mut invalid_pair_ids = Vec::new();
    for (&pair_id, edge) in pose_graph.valid_edges() {
        let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
        let image1 = reconstruction.image(image_id1)?;
        let image2 = reconstruction.image(image_id2)?;
        if !image_has_pose(reconstruction, image1) || !image_has_pose(reconstruction, image2) {
            continue;
        }
        let cam2_from_cam1 = cam_from_world_rotation(reconstruction, image2)?
            * cam_from_world_rotation(reconstruction, image1)?.inverse();
        if cam2_from_cam1.angular_distance(edge.cam2_from_cam1.rotation) > max_angle_rad {
            invalid_pair_ids.push(pair_id);
        }
    }
    for pair_id in invalid_pair_ids {
        pose_graph.set_invalid_edge(pair_id)?;
    }
    Ok(())
}

/// Port of `colmap::RunRotationAveragingOnComponent`: estimates rotations for the connected
/// component `active_image_ids` and registers its frames. Handles rigs with unknown
/// cam_from_rig by first solving on an expanded reconstruction. Does not perform outlier
/// filtering or de-registration; callers compose these primitives to implement the desired
/// post-processing.
pub fn run_rotation_averaging_on_component(
    options: &RotationEstimatorOptions,
    pose_graph: &mut PoseGraph,
    active_image_ids: &HashSet<ImageId>,
    reconstruction: &mut Reconstruction,
    pose_priors: &[PosePrior],
) -> Result<bool> {
    if active_image_ids.is_empty() {
        // COLMAP: LOG(ERROR) "No connected components found".
        return Ok(false);
    }

    if !has_unknown_cams_from_rig(reconstruction) {
        pose_graph.invalidate_pairs_outside_active_image_ids(active_image_ids)?;
        return RotationEstimator::new(options).estimate_rotations(
            pose_graph,
            pose_priors,
            active_image_ids,
            reconstruction,
        );
    }

    // First solve on an expanded reconstruction where cameras with unknown cam_from_rig are
    // treated as independent rigs.
    let mut recon_expanded = create_expanded_reconstruction(reconstruction)?;
    let expanded_active_image_ids = compute_largest_connected_component_image_ids(
        pose_graph,
        &recon_expanded,
        options.filter_unregistered,
    )?;
    if expanded_active_image_ids.is_empty() {
        return Ok(false);
    }

    pose_graph.invalidate_pairs_outside_active_image_ids(&expanded_active_image_ids)?;
    if !RotationEstimator::new(options).estimate_rotations(
        pose_graph,
        pose_priors,
        &expanded_active_image_ids,
        &mut recon_expanded,
    )? {
        return Ok(false);
    }

    let mut expanded_cams_from_world: BTreeMap<ImageId, Rigid3d> = BTreeMap::new();
    for (&image_id, image) in recon_expanded.images() {
        if image_has_pose(&recon_expanded, image) {
            let cam_from_world =
                crate::estimators::alignment::image_cam_from_world(&recon_expanded, image)?;
            expanded_cams_from_world.insert(image_id, cam_from_world);
        }
    }

    // Initializing cam_from_rig from preliminary rotation estimates.
    initialize_rig_rotations_from_images(
        &expanded_cams_from_world,
        reconstruction,
        options.refine_sensor_from_rig,
    )?;

    // Expanding rigs changes frame connectivity, so recompute the active set on the
    // original reconstruction before the final solve.
    let final_active_image_ids = compute_largest_connected_component_image_ids(
        pose_graph,
        reconstruction,
        options.filter_unregistered,
    )?;
    if final_active_image_ids.is_empty() {
        return Ok(false);
    }

    pose_graph.invalidate_pairs_outside_active_image_ids(&final_active_image_ids)?;
    let final_options = RotationEstimatorOptions {
        skip_initialization: true,
        use_stratified: false,
        ..*options
    };
    RotationEstimator::new(&final_options).estimate_rotations(
        pose_graph,
        pose_priors,
        &final_active_image_ids,
        reconstruction,
    )
}

/// Port of `colmap::InitializeRigRotationsFromImages`: initializes rig rotations by
/// averaging per-image rotations. Estimates cam_from_rig for cameras with unknown
/// calibration, then computes rig_from_world for each frame. When `refine_sensor_from_rig`
/// is false, the per-sensor cam_from_rig values are left untouched (and must all be known).
pub fn initialize_rig_rotations_from_images(
    cams_from_world: &BTreeMap<ImageId, Rigid3d>,
    reconstruction: &mut Reconstruction,
    refine_sensor_from_rig: bool,
) -> Result<bool> {
    // Step 1: Estimate cam_from_rig for cameras with unknown calibration. Collect samples
    // across frames, then average.
    let mut cam_from_rig_samples: BTreeMap<CameraId, (RigId, Vec<Quaterniond>)> = BTreeMap::new();
    for frame in reconstruction.frames().values() {
        // Find the rotation of the reference image.
        let mut ref_rotation: Option<Quaterniond> = None;
        for data_id in frame.image_ids() {
            let image = reconstruction.image(data_id.id as ImageId)?;
            if is_ref_in_frame(reconstruction, image)? {
                ref_rotation = cams_from_world
                    .get(&(data_id.id as ImageId))
                    .map(|pose| pose.rotation);
                break;
            }
        }
        let Some(ref_rotation) = ref_rotation else {
            continue;
        };

        // Collect cam_from_rig samples for non-reference cameras.
        for data_id in frame.image_ids() {
            let image = reconstruction.image(data_id.id as ImageId)?;
            if is_ref_in_frame(reconstruction, image)? {
                continue;
            }
            let Some(cam_from_world) = cams_from_world.get(&(data_id.id as ImageId)) else {
                continue;
            };
            let samples = cam_from_rig_samples
                .entry(image.camera_id())
                .or_insert_with(|| (frame.rig_id(), Vec::new()));
            samples.0 = frame.rig_id();
            samples
                .1
                .push(cam_from_world.rotation * ref_rotation.inverse());
        }
    }

    if refine_sensor_from_rig {
        for (camera_id, (rig_id, samples)) in cam_from_rig_samples {
            let sensor_id = SensorId::new(SensorType::Camera, camera_id);
            let existing = reconstruction
                .rig(rig_id)?
                .maybe_sensor_from_rig(sensor_id)?;
            // If sensor_from_rig is already fully calibrated, preserve it rather than
            // overwriting with MST-derived approximation.
            if existing.is_some_and(|pose| !has_nan3(pose.translation)) {
                continue;
            }
            let cam_from_rig = average_quaternions(&samples, &vec![1.0; samples.len()])?;
            reconstruction
                .rig_mut(rig_id)?
                .set_sensor_from_rig(sensor_id, rotation_only_pose(cam_from_rig))?;
        }
    } else {
        // Check if rotations are valid.
        for (&rig_id, rig) in reconstruction.rigs() {
            for (sensor_id, sensor_from_rig) in rig.non_ref_sensors() {
                let valid = sensor_from_rig.is_some_and(|pose| {
                    let q = pose.rotation;
                    !(q.x.is_nan() || q.y.is_nan() || q.z.is_nan() || q.w.is_nan())
                });
                crate::check!(
                    valid,
                    "sensor_from_rig has NaN rotation but refine_sensor_from_rig=false \
                     (rig_id={}, sensor_id={})",
                    rig_id,
                    sensor_id.id
                );
            }
        }
    }

    // Step 2: Compute rig_from_world for each frame by averaging across images.
    let mut rig_from_world_updates: Vec<(FrameId, Quaterniond)> = Vec::new();
    for (&frame_id, frame) in reconstruction.frames() {
        let mut rig_from_world_samples: Vec<Quaterniond> = Vec::new();
        for data_id in frame.image_ids() {
            let image_id = data_id.id as ImageId;
            if !reconstruction.exists_image(image_id) {
                continue;
            }
            let Some(cam_from_world) = cams_from_world.get(&image_id) else {
                continue;
            };
            let image = reconstruction.image(image_id)?;
            if is_ref_in_frame(reconstruction, image)? {
                rig_from_world_samples.push(cam_from_world.rotation);
            } else {
                let Some(cam_from_rig) = reconstruction
                    .rig(frame.rig_id())?
                    .maybe_sensor_from_rig(camera_sensor_id(image))?
                else {
                    continue;
                };
                rig_from_world_samples
                    .push(cam_from_rig.rotation.inverse() * cam_from_world.rotation);
            }
        }
        if !rig_from_world_samples.is_empty() {
            let weights = vec![1.0; rig_from_world_samples.len()];
            rig_from_world_updates.push((
                frame_id,
                average_quaternions(&rig_from_world_samples, &weights)?,
            ));
        }
    }
    for (frame_id, rig_from_world) in rig_from_world_updates {
        reconstruction
            .frame_mut(frame_id)?
            .set_rig_from_world(rotation_only_pose(rig_from_world));
    }
    Ok(true)
}

/// Port of `colmap::RunRotationAveraging`: the high-level rotation averaging solver that
/// handles rig expansion. Solves the largest connected component, then (if
/// `max_rotation_error_deg > 0`) invalidates pairs whose relative rotation disagrees with
/// the result and de-registers frames outside the new largest component.
pub fn run_rotation_averaging(
    options: &RotationEstimatorOptions,
    pose_graph: &mut PoseGraph,
    reconstruction: &mut Reconstruction,
    pose_priors: &[PosePrior],
) -> Result<bool> {
    // Step 1: Compute the largest connected component and solve rotation averaging on it.
    let active_image_ids = compute_largest_connected_component_image_ids(
        pose_graph,
        reconstruction,
        options.filter_unregistered,
    )?;
    if !run_rotation_averaging_on_component(
        options,
        pose_graph,
        &active_image_ids,
        reconstruction,
        pose_priors,
    )? {
        return Ok(false);
    }

    // Step 2: Filter outlier pairs by rotation error and update the active set.
    if options.max_rotation_error_deg > 0.0 {
        filter_edges_by_relative_rotation(
            pose_graph,
            reconstruction,
            options.max_rotation_error_deg,
        )?;

        // Recompute largest connected component among registered frames.
        let filtered_active_image_ids =
            compute_largest_connected_component_image_ids(pose_graph, reconstruction, true)?;
        if filtered_active_image_ids.is_empty() {
            // COLMAP: LOG(ERROR) "No connected components found after filtering".
            return Ok(false);
        }

        pose_graph.invalidate_pairs_outside_active_image_ids(&filtered_active_image_ids)?;

        // De-register frames outside the new active set. Lookup only.
        let mut active_frame_ids: HashSet<FrameId> = HashSet::new();
        for &image_id in &filtered_active_image_ids {
            active_frame_ids.insert(reconstruction.image(image_id)?.frame_id());
        }
        let reg_frame_ids_snapshot: Vec<FrameId> = reconstruction.reg_frame_ids().to_vec();
        for frame_id in reg_frame_ids_snapshot {
            crate::check!(reconstruction.frame(frame_id)?.has_pose());
            if !active_frame_ids.contains(&frame_id) {
                reconstruction.deregister_frame(frame_id)?;
            }
        }
    }
    Ok(true)
}
