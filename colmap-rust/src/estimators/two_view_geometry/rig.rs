//! `EstimateRigTwoViewGeometries` of COLMAP's `two_view_geometry.cc`: verify the matches
//! between the images of two rigs (or two frames) jointly with the generalized relative pose
//! ([`crate::estimators::generalized_pose::estimate_generalized_relative_pose`]), then split
//! the inliers back into one `CALIBRATED_RIG` (or, for two panoramic rigs, `CALIBRATED`)
//! geometry per image pair. Re-exported from `mod.rs`.

use std::collections::{BTreeMap, HashMap, HashSet};

use super::TwoViewGeometryOptions;
use crate::estimators::generalized_pose::estimate_generalized_relative_pose;
use crate::feature::types::{FeatureMatch, FeatureMatches};
use crate::geometry::essential_matrix::essential_matrix_from_pose;
use crate::geometry::rigid3::Rigid3d;
use crate::linalg::Vector2d;
use crate::scene::camera::Camera;
use crate::scene::image::Image;
use crate::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use crate::sensor::rig::Rig;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{
    image_pair_to_pair_id, CameraId, ImageId, Point2DIdx, RigId,
};
use crate::{check, check_eq, Result};

/// One image pair's matches: `std::pair<std::pair<image_t, image_t>, FeatureMatches>`.
pub type ImagePairMatches = ((ImageId, ImageId), FeatureMatches);

/// One image pair's geometry: `std::pair<std::pair<image_t, image_t>, TwoViewGeometry>`.
pub type ImagePairGeometry = ((ImageId, ImageId), TwoViewGeometry);

/// `NodeHashMap::at`: `std::out_of_range` for a missing key.
fn at<'a, K: Ord, V>(map: &'a BTreeMap<K, V>, key: &K) -> Result<&'a V> {
    map.get(key)
        .ok_or_else(|| ColmapError::new(ErrorKind::OutOfRange, "unordered_map::at: key not found"))
}

/// The pose of `camera`'s sensor in `rig`: identity for the reference sensor.
fn cam_from_rig(rig: &Rig, camera_id: CameraId) -> Result<Rigid3d> {
    let sensor_id = crate::util::types::SensorId::new(
        crate::util::types::SensorType::Camera,
        camera_id,
    );
    if rig.is_ref_sensor(sensor_id) {
        Ok(Rigid3d::identity())
    } else {
        rig.sensor_from_rig(sensor_id)
    }
}

/// Port of `colmap::EstimateRigTwoViewGeometries`: estimate the two-view geometries of all
/// image pairs between two rigs from their matches, jointly through the generalized relative
/// pose. Every image of a pair's first image must belong to `rig1`, of its second to `rig2`.
///
/// Returns one geometry per input pair: those with inliers first, in image-pair order, then
/// a default (`UNDEFINED`) geometry for every input pair without inliers, in input order. An
/// empty result when there are no matches or the pose estimation fails or has fewer than
/// `options.min_num_inliers` inliers.
pub fn estimate_rig_two_view_geometries(
    rig1: &Rig,
    rig2: &Rig,
    images: &BTreeMap<ImageId, Image>,
    cameras: &BTreeMap<CameraId, Camera>,
    matches: &[ImagePairMatches],
    options: &TwoViewGeometryOptions,
) -> Result<Vec<ImagePairGeometry>> {
    let mut points1: Vec<Vector2d> = Vec::new();
    let mut points2: Vec<Vector2d> = Vec::new();
    let mut camera_idxs1: Vec<usize> = Vec::new();
    let mut camera_idxs2: Vec<usize> = Vec::new();
    let mut cams_from_rig: Vec<Rigid3d> =
        Vec::with_capacity(rig1.num_sensors() + rig2.num_sensors());
    let mut cameras_vec: Vec<Camera> = Vec::with_capacity(cams_from_rig.capacity());
    let mut corrs: Vec<(ImageId, Point2DIdx, ImageId, Point2DIdx)> = Vec::new();

    // Only looked up, never iterated, so its order cannot reach an output.
    let mut camera_id_to_rig_and_camera_idx: HashMap<CameraId, (RigId, usize)> = HashMap::new();
    let mut maybe_add_camera = |rig: &Rig, camera: &Camera| -> Result<usize> {
        if let Some(&(rig_id, camera_idx)) = camera_id_to_rig_and_camera_idx.get(&camera.camera_id)
        {
            check_eq!(rig_id, rig.rig_id(), "The same camera is assigned to both rigs");
            return Ok(camera_idx);
        }
        let camera_idx = cameras_vec.len();
        camera_id_to_rig_and_camera_idx.insert(camera.camera_id, (rig.rig_id(), camera_idx));
        cameras_vec.push(camera.clone());
        cams_from_rig.push(cam_from_rig(rig, camera.camera_id)?);
        Ok(camera_idx)
    };

    let mut image_pairs = HashSet::with_capacity(matches.len());
    for ((image_id1, image_id2), pair_matches) in matches {
        let (image_id1, image_id2) = (*image_id1, *image_id2);
        check!(
            image_pairs.insert(image_pair_to_pair_id(image_id1, image_id2)?),
            "Duplicate image pair"
        );

        let image1 = at(images, &image_id1)?;
        let camera1 = at(cameras, &image1.camera_id())?;
        let camera_idx1 = maybe_add_camera(rig1, camera1)?;

        let image2 = at(images, &image_id2)?;
        let camera2 = at(cameras, &image2.camera_id())?;
        let camera_idx2 = maybe_add_camera(rig2, camera2)?;

        for feature_match in pair_matches {
            points1.push(image1.point2d(feature_match.point2d_idx1).xy);
            points2.push(image2.point2d(feature_match.point2d_idx2).xy);
            camera_idxs1.push(camera_idx1);
            camera_idxs2.push(camera_idx2);
            corrs.push((
                image_id1,
                feature_match.point2d_idx1,
                image_id2,
                feature_match.point2d_idx2,
            ));
        }
    }

    if corrs.is_empty() {
        return Ok(Vec::new());
    }

    let Some(estimate) = estimate_generalized_relative_pose(
        &options.ransac_options,
        &points1,
        &points2,
        &camera_idxs1,
        &camera_idxs2,
        &cams_from_rig,
        &cameras_vec,
    )?
    else {
        return Ok(Vec::new());
    };
    // `num_inliers < static_cast<size_t>(options.min_num_inliers)`.
    if estimate.num_inliers < super::min_num_inliers(options.min_num_inliers) {
        return Ok(Vec::new());
    }

    // `std::map`: iterated in image-pair order below.
    let mut inlier_matches: BTreeMap<(ImageId, ImageId), FeatureMatches> = BTreeMap::new();
    for (corr, &is_inlier) in corrs.iter().zip(&estimate.inlier_mask) {
        if !is_inlier {
            continue;
        }
        let &(image_id1, point2d_idx1, image_id2, point2d_idx2) = corr;
        inlier_matches
            .entry((image_id1, image_id2))
            .or_default()
            .push(FeatureMatch::new(point2d_idx1, point2d_idx2));
    }

    let config = if estimate.pose.rig2_from_rig1().is_some() {
        ConfigurationType::CalibratedRig
    } else {
        ConfigurationType::Calibrated
    };
    let rig2_from_rig1 = estimate.pose.pose();

    let mut two_view_geometries: Vec<ImagePairGeometry> = Vec::with_capacity(matches.len());
    for (&image_pair, pair_matches) in &inlier_matches {
        let cam1_from_rig1 = cam_from_rig(rig1, at(images, &image_pair.0)?.camera_id())?;
        let cam2_from_rig2 = cam_from_rig(rig2, at(images, &image_pair.1)?.camera_id())?;
        let cam2_from_cam1 = cam2_from_rig2 * rig2_from_rig1 * cam1_from_rig1.inverse();
        // TODO(jsch): For panoramic rigs, we could further distinguish between
        // panoramic/planar configurations by estimating a homography matrix.
        let two_view_geometry = TwoViewGeometry {
            config,
            inlier_matches: pair_matches.clone(),
            cam2_from_cam1: Some(cam2_from_cam1),
            e: Some(essential_matrix_from_pose(&cam2_from_cam1)),
            ..TwoViewGeometry::default()
        };
        two_view_geometries.push((image_pair, two_view_geometry));
    }

    // Ensure that each matched input pair has a corresponding two-view geometry, even if it
    // has no inliers.
    for (image_pair, _) in matches {
        if !inlier_matches.contains_key(image_pair) {
            two_view_geometries.push((*image_pair, TwoViewGeometry::default()));
        }
    }

    Ok(two_view_geometries)
}
