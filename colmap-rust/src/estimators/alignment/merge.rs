//! Port of COLMAP's `estimators/alignment.cc`, part 2: `MergeReconstructions` (with its
//! `CopyRegisteredImage` helper) and `AlignReconstructionToOrigRigScales`. The alignment
//! they start from is in [`super`]. Port of colmap-sharp's `Estimators/Alignment.Merge.cs`.
//! Tests: `tests/estimators/alignment.rs` (`MergeReconstructions`,
//! `MergeReconstructionsInconsistentImageNames`, `AlignReconstructionToOrigRigScales`).
//!
//! Deterministic order (docs/CPP_DIVERGENCES.md entry 277): COLMAP copies the missing
//! images in the iteration order of a `FlatHashSet`, which decides the order their frames
//! are registered in the target (`RegFrameIds`); the port copies them in the source's
//! `RegImageIds` order. The source points are merged in `Points3D` order (ascending id,
//! entry 213), which decides the ids the new target points get, and the original rigs are
//! visited in ascending rig id order.

use std::collections::{BTreeMap, BTreeSet};

use crate::geometry::pose::transform_camera_world;
use crate::geometry::Sim3d;
use crate::scene::reconstruction::Reconstruction;
use crate::scene::track::Track;
use crate::sensor::rig::Rig;
use crate::util::types::{ImageId, Point3DId, RigId};
use crate::Result;

use super::{align_reconstructions_via_reprojections, image_cam_from_world};

/// `CopyRegisteredImage`: adds a source image to the target with its camera, rig and frame
/// (the frame posed in the target's world) where the target lacks them.
fn copy_registered_image(
    image_id: ImageId,
    tgt_from_src: &Sim3d,
    src_reconstruction: &Reconstruction,
    tgt_reconstruction: &mut Reconstruction,
) -> Result<()> {
    let src_image = src_reconstruction.image(image_id)?;
    if !tgt_reconstruction.exists_camera(src_image.camera_id()) {
        tgt_reconstruction.add_camera(src_reconstruction.camera(src_image.camera_id())?.clone())?;
    }
    let src_frame = src_reconstruction.frame(src_image.frame_id())?;
    let rig_id = src_frame.rig_id();
    if !tgt_reconstruction.exists_rig(rig_id) {
        tgt_reconstruction.add_rig(src_reconstruction.rig(rig_id)?.clone())?;
    }
    if !tgt_reconstruction.exists_frame(src_image.frame_id()) {
        // COLMAP's ResetRigPtr is implicit: frames hold rig ids (entry 170).
        tgt_reconstruction.add_frame(src_frame.clone())?;
        let cam_from_tgt_world = transform_camera_world(
            tgt_from_src,
            &image_cam_from_world(src_reconstruction, src_image)?,
        );
        // The rig is cloned so the frame can be borrowed mutably next to it.
        let tgt_rig = tgt_reconstruction.rig(rig_id)?.clone();
        tgt_reconstruction
            .frame_mut(src_image.frame_id())?
            .set_cam_from_world(&tgt_rig, src_image.camera_id(), cam_from_tgt_world)?;
    }

    // COLMAP's ResetCameraPtr / ResetFramePtr are implicit: images hold ids (entry 170).
    tgt_reconstruction.add_image(src_image.clone())
}

/// Aligns the source to the target reconstruction and merges cameras, images, points3D
/// into the target using the alignment. Returns `false` on failure, in which case the
/// target is unchanged. Port of `colmap::MergeReconstructions`.
pub fn merge_reconstructions(
    max_reproj_error: f64,
    src_reconstruction: &Reconstruction,
    tgt_reconstruction: &mut Reconstruction,
) -> Result<bool> {
    let mut tgt_from_src = Sim3d::default();
    if !align_reconstructions_via_reprojections(
        src_reconstruction,
        tgt_reconstruction,
        /* min_inlier_observations= */ 0.3,
        max_reproj_error,
        &mut tgt_from_src,
    )? {
        return Ok(false);
    }

    // Find common and missing images in the two reconstructions. Images are
    // matched by image id, which assumes that both reconstructions share a
    // consistent image_id<->name mapping (i.e. were derived from the same
    // database). If this assumption is violated -- e.g. the reconstructions were
    // built from independent databases that both number their images 1..N -- then
    // distinct physical images end up with colliding ids. Detect the
    // inconsistency via the image name and fail loudly instead.
    let mut common_image_ids = BTreeSet::<ImageId>::new();
    // In the source's RegImageIds order (entry 277), with a set for the lookups below.
    let mut missing_image_ids = Vec::<ImageId>::new();
    let mut missing_image_set = BTreeSet::<ImageId>::new();
    for image_id in src_reconstruction.reg_image_ids()? {
        if tgt_reconstruction.exists_image(image_id) {
            let src_name = src_reconstruction.image(image_id)?.name();
            let tgt_name = tgt_reconstruction.image(image_id)?.name();
            if src_name != tgt_name {
                // COLMAP: LOG(ERROR) << "Cannot merge reconstructions: image_id=" ...
                return Ok(false);
            }
            common_image_ids.insert(image_id);
        } else {
            missing_image_ids.push(image_id);
            missing_image_set.insert(image_id);
        }
    }

    // Register the missing images in this src_reconstruction.
    for &image_id in &missing_image_ids {
        copy_registered_image(
            image_id,
            &tgt_from_src,
            src_reconstruction,
            tgt_reconstruction,
        )?;
    }

    // Merge the two point clouds using the following two rules:
    //    - copy points to this src_reconstruction with non-conflicting tracks,
    //      i.e. points that do not have an already triangulated observation
    //      in this src_reconstruction.
    //    - merge tracks that are unambiguous, i.e. only merge points in the two
    //      reconstructions if they have a one-to-one mapping.
    // Note that in both cases no cheirality or reprojection test is performed.

    for point3d in src_reconstruction.points3d().values() {
        let mut new_track = Track::new();
        let mut old_track = Track::new();
        let mut old_point3d_ids = BTreeSet::<Point3DId>::new();
        for &track_el in point3d.track.elements() {
            if common_image_ids.contains(&track_el.image_id) {
                let point2d = tgt_reconstruction
                    .image(track_el.image_id)?
                    .point2d(track_el.point2d_idx);
                if point2d.has_point3d() {
                    old_point3d_ids.insert(point2d.point3d_id);
                    old_track.add_element(track_el);
                } else {
                    new_track.add_element(track_el);
                }
            } else if missing_image_set.contains(&track_el.image_id) {
                tgt_reconstruction
                    .image_mut(track_el.image_id)?
                    .reset_point3d_for_point2d(track_el.point2d_idx);
                new_track.add_element(track_el);
            }
        }

        let create_new_point = new_track.length() >= 2;
        let merge_new_and_old_point =
            (new_track.length() + old_track.length()) >= 2 && old_point3d_ids.len() == 1;
        if create_new_point || merge_new_and_old_point {
            let xyz = tgt_from_src * point3d.xyz;
            let point3d_id = tgt_reconstruction.add_point3d(xyz, new_track, point3d.color)?;
            if let (1, Some(&old_point3d_id)) =
                (old_point3d_ids.len(), old_point3d_ids.iter().next())
            {
                tgt_reconstruction.merge_points3d(point3d_id, old_point3d_id)?;
            }
        }
    }

    Ok(true)
}

/// Align reconstruction to the original metric scales in rig extrinsics. Returns `false`
/// if there is no available non-panoramic rig in the alignment process. Port of
/// `colmap::AlignReconstructionToOrigRigScales`.
pub fn align_reconstruction_to_orig_rig_scales(
    orig_rigs: &BTreeMap<RigId, Rig>,
    reconstruction: &mut Reconstruction,
) -> Result<bool> {
    let mut scale_sum = 0.0;
    let mut scale_count = 0i32;
    for (&rig_id, orig_rig) in orig_rigs {
        let mut scale_sum_rig = 0.0;
        let mut scale_count_rig = 0i32;
        for (&sensor_id, sensor_from_orig_rig) in orig_rig.non_ref_sensors() {
            let Some(sensor_from_orig_rig) = sensor_from_orig_rig else {
                continue;
            };

            // Here we do not include rigs that are panoramic.
            let sensor_from_orig_rig_norm = sensor_from_orig_rig.translation.norm();
            if sensor_from_orig_rig_norm < 1e-6 {
                continue;
            }
            let rig = reconstruction.rig(rig_id)?;
            crate::check!(rig.has_sensor_from_rig(sensor_id));
            let scale =
                rig.sensor_from_rig(sensor_id)?.translation.norm() / sensor_from_orig_rig_norm;
            scale_sum_rig += scale;
            scale_count_rig += 1;
        }
        if scale_count_rig > 0 {
            scale_sum += scale_sum_rig / f64::from(scale_count_rig);
            scale_count += 1;
        }
    }
    if scale_count == 0 {
        return Ok(false);
    }
    let new_from_old_world = Sim3d {
        scale: f64::from(scale_count) / scale_sum,
        ..Sim3d::default()
    };
    reconstruction.transform(&new_from_old_world);
    Ok(true)
}
