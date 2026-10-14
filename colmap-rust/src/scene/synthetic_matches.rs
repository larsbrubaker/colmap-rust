//! Port of the anonymous-namespace helpers of `colmap/scene/synthetic.cc` that write a
//! synthetic dataset's matches into a database (colmap-sharp's `Scene/Synthetic.Matches.cs`):
//! the exhaustive, chained and sparse match configurations, the two-view geometries they
//! write (essential and fundamental matrix from the true poses, inlier matches from the
//! shared 3D points, random outlier matches), and the pose priors' Cartesian <-> WGS84
//! conversion around a fixed reference point. Neighbors: [`super::synthetic`] (the generator
//! that calls them) and [`super::synthetic_noise`] (which converts priors back to add noise).
//! Tests: `tests/scene/synthetic.rs`.
//!
//! Tier A given the PRNG stream; the chained configuration writes its pairs in ascending
//! pair id order where COLMAP walks a hash map (docs/CPP_DIVERGENCES.md entry 240).

use std::collections::{BTreeMap, BTreeSet};

use crate::feature::types::{FeatureMatch, FeatureMatches};
use crate::geometry::essential_matrix::{
    essential_matrix_from_pose, fundamental_from_essential_matrix,
};
use crate::geometry::gps::GpsTransform;
use crate::geometry::pose_prior::{CoordinateSystem, PosePrior};
use crate::geometry::rigid3::Rigid3d;
use crate::math::random::{self, libcxx};
use crate::math::union_find::UnionFind;
use crate::scene::camera::Camera;
use crate::scene::database::Database;
use crate::scene::reconstruction::Reconstruction;
use crate::scene::track::TrackElement;
use crate::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use crate::util::types::{
    image_pair_to_pair_id, pair_id_to_image_pair, should_swap_image_pair, ImageId, ImagePairId,
    Point2DIdx,
};
use crate::Result;

/// `AddOutlierMatches`: appends `size * (1 - inlier_ratio)` random matches and shuffles.
fn add_outlier_matches(
    inlier_ratio: f64,
    num_points2d1: i32,
    num_points2d2: i32,
    matches: &mut FeatureMatches,
) {
    // `const int num_outliers = matches->size() * (1.0 - inlier_ratio)` truncates.
    let num_outliers = (matches.len() as f64 * (1.0 - inlier_ratio)) as i32;
    for _ in 0..num_outliers {
        // COLMAP draws the second index from [0, num_points2D2 - 2] (not - 1); kept, since
        // it decides the PRNG stream. The `int` bounds convert to `point2D_t` as in C++.
        let point2d_idx1 =
            random::random_uniform_integer::<Point2DIdx>(0, (num_points2d1 - 1) as Point2DIdx);
        let point2d_idx2 =
            random::random_uniform_integer::<Point2DIdx>(0, (num_points2d2 - 2) as Point2DIdx);
        matches.push(FeatureMatch {
            point2d_idx1,
            point2d_idx2,
        });
    }
    random::with_prng(|g| libcxx::shuffle(matches, g));
}

/// `ExtractExhaustiveImagePairs`: every pair of registered images, smaller id first.
fn extract_exhaustive_image_pairs(reconstruction: &Reconstruction) -> Result<Vec<ImagePairId>> {
    let num_reg_images = reconstruction.num_reg_images();
    let num_exhaustive_pairs = num_reg_images * num_reg_images.saturating_sub(1) / 2;
    let reg_image_ids = reconstruction.reg_image_ids()?;
    let mut image_pairs = Vec::with_capacity(num_exhaustive_pairs);
    for &image_id1 in &reg_image_ids {
        for &image_id2 in &reg_image_ids {
            if image_id1 >= image_id2 {
                continue;
            }
            image_pairs.push(image_pair_to_pair_id(image_id1, image_id2)?);
        }
    }
    crate::check_eq!(image_pairs.len(), num_exhaustive_pairs);
    Ok(image_pairs)
}

/// The camera and `cam_from_world` of a registered image.
fn image_camera_and_pose(
    reconstruction: &Reconstruction,
    image_id: ImageId,
) -> Result<(&Camera, Rigid3d)> {
    let image = reconstruction.image(image_id)?;
    let frame = reconstruction.frame(image.frame_id())?;
    let rig = reconstruction.rig(frame.rig_id())?;
    Ok((
        reconstruction.camera(image.camera_id())?,
        image.cam_from_world(frame, rig)?,
    ))
}

/// `SetTwoViewGeometryModel`: fills the configuration, essential matrix, and (for
/// perspective pairs) fundamental matrix of a synthetic two-view geometry.
fn set_two_view_geometry_model(
    camera1: &Camera,
    camera2: &Camera,
    cam2_from_cam1: &Rigid3d,
    two_view_geometry: &mut TwoViewGeometry,
) -> Result<()> {
    let e = essential_matrix_from_pose(cam2_from_cam1);
    two_view_geometry.e = Some(e);
    if camera1.is_spherical() || camera2.is_spherical() {
        // Omnidirectional cameras (e.g. EQUIRECTANGULAR) have no pinhole calibration
        // matrix, so the fundamental matrix is undefined; they are always calibrated and use
        // the bearing-based essential-matrix configuration.
        two_view_geometry.config = ConfigurationType::Calibrated;
        return Ok(());
    }
    let is_calibrated = camera1.has_prior_focal_length && camera2.has_prior_focal_length;
    two_view_geometry.config = if is_calibrated {
        ConfigurationType::Calibrated
    } else {
        ConfigurationType::Uncalibrated
    };
    two_view_geometry.f = Some(fundamental_from_essential_matrix(
        &camera2.calibration_matrix()?,
        &e,
        &camera1.calibration_matrix()?,
    ));
    Ok(())
}

/// The relative pose of a pair, `cam2_from_cam1`.
fn pair_cam2_from_cam1(
    reconstruction: &Reconstruction,
    image_id1: ImageId,
    image_id2: ImageId,
) -> Result<(&Camera, &Camera, Rigid3d)> {
    let (camera1, cam1_from_world) = image_camera_and_pose(reconstruction, image_id1)?;
    let (camera2, cam2_from_world) = image_camera_and_pose(reconstruction, image_id2)?;
    Ok((
        camera1,
        camera2,
        cam2_from_world * cam1_from_world.inverse(),
    ))
}

/// `BuildTwoViewGeometry`: the true geometry of a pair with a match for every 3D point the
/// two images share.
fn build_two_view_geometry(
    has_relative_pose: bool,
    reconstruction: &Reconstruction,
    pair_id: ImagePairId,
) -> Result<TwoViewGeometry> {
    let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
    let image1 = reconstruction.image(image_id1)?;
    let image2 = reconstruction.image(image_id2)?;

    let mut two_view_geometry = TwoViewGeometry::default();
    let (camera1, camera2, cam2_from_cam1) =
        pair_cam2_from_cam1(reconstruction, image_id1, image_id2)?;
    if has_relative_pose {
        two_view_geometry.cam2_from_cam1 = Some(cam2_from_cam1);
    }
    set_two_view_geometry_model(camera1, camera2, &cam2_from_cam1, &mut two_view_geometry)?;

    for (point2d_idx1, point2d1) in image1.points2d().iter().enumerate() {
        if !point2d1.has_point3d() {
            continue;
        }
        if let Some(point2d_idx2) = image2
            .points2d()
            .iter()
            .position(|point2d2| point2d1.point3d_id == point2d2.point3d_id)
        {
            two_view_geometry.inlier_matches.push(FeatureMatch {
                point2d_idx1: point2d_idx1 as Point2DIdx,
                point2d_idx2: point2d_idx2 as Point2DIdx,
            });
        }
    }
    Ok(two_view_geometry)
}

/// `WriteTwoViewGeometryToDatabase`: the inlier matches plus outliers as the raw matches,
/// and the geometry, each unless the pair already has one.
fn write_two_view_geometry_to_database(
    pair_id: ImagePairId,
    two_view_geometry: &TwoViewGeometry,
    inlier_match_ratio: f64,
    reconstruction: &Reconstruction,
    database: &Database,
) -> Result<()> {
    let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
    let image1 = reconstruction.image(image_id1)?;
    let image2 = reconstruction.image(image_id2)?;

    let mut matches = two_view_geometry.inlier_matches.clone();
    add_outlier_matches(
        inlier_match_ratio,
        image1.num_points2d() as i32,
        image2.num_points2d() as i32,
        &mut matches,
    );

    if !database.exists_matches(image_id1, image_id2)? {
        database.write_matches(image_id1, image_id2, &matches)?;
    }
    if !database.exists_two_view_geometry(image_id1, image_id2)? {
        database.write_two_view_geometry(image_id1, image_id2, two_view_geometry)?;
    }
    Ok(())
}

/// `SynthesizeExhaustiveMatches`.
pub(super) fn synthesize_exhaustive_matches(
    inlier_match_ratio: f64,
    has_relative_pose: bool,
    reconstruction: &Reconstruction,
    database: &Database,
) -> Result<()> {
    for pair_id in extract_exhaustive_image_pairs(reconstruction)? {
        let two_view_geometry =
            build_two_view_geometry(has_relative_pose, reconstruction, pair_id)?;
        write_two_view_geometry_to_database(
            pair_id,
            &two_view_geometry,
            inlier_match_ratio,
            reconstruction,
            database,
        )?;
    }
    Ok(())
}

/// `SynthesizeChainedMatches`: matches only between images with consecutive ids, from the
/// 3D points' tracks. The pairs are written in ascending pair id order (COLMAP: hash order of
/// a `NodeHashMap`; entry 240).
pub(super) fn synthesize_chained_matches(
    inlier_match_ratio: f64,
    has_relative_pose: bool,
    reconstruction: &Reconstruction,
    database: &Database,
) -> Result<()> {
    let mut two_view_geometries: BTreeMap<ImagePairId, TwoViewGeometry> = BTreeMap::new();
    for point3d in reconstruction.points3d().values() {
        let mut track_elements: Vec<TrackElement> = point3d.track.elements().to_vec();
        // `std::sort` by image id: a track holds each image at most once here, so there
        // are no ties.
        track_elements.sort_by_key(|element| element.image_id);
        for pair in track_elements.windows(2) {
            let (prev_track_el, curr_track_el) = (pair[0], pair[1]);
            if curr_track_el.image_id != prev_track_el.image_id + 1 {
                continue;
            }
            let pair_id = image_pair_to_pair_id(prev_track_el.image_id, curr_track_el.image_id)?;
            let inlier_matches = &mut two_view_geometries
                .entry(pair_id)
                .or_default()
                .inlier_matches;
            if should_swap_image_pair(prev_track_el.image_id, curr_track_el.image_id) {
                inlier_matches.push(FeatureMatch {
                    point2d_idx1: curr_track_el.point2d_idx,
                    point2d_idx2: prev_track_el.point2d_idx,
                });
            } else {
                inlier_matches.push(FeatureMatch {
                    point2d_idx1: prev_track_el.point2d_idx,
                    point2d_idx2: curr_track_el.point2d_idx,
                });
            }
        }
    }

    for (pair_id, mut two_view_geometry) in two_view_geometries {
        let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
        let (camera1, camera2, cam2_from_cam1) =
            pair_cam2_from_cam1(reconstruction, image_id1, image_id2)?;
        if has_relative_pose {
            two_view_geometry.cam2_from_cam1 = Some(cam2_from_cam1);
        }
        set_two_view_geometry_model(camera1, camera2, &cam2_from_cam1, &mut two_view_geometry)?;

        write_two_view_geometry_to_database(
            pair_id,
            &two_view_geometry,
            inlier_match_ratio,
            reconstruction,
            database,
        )?;
    }
    Ok(())
}

/// `IsViewGraphConnected`: whether the pairs connect all images.
fn is_view_graph_connected(
    images: &BTreeSet<ImageId>,
    image_pairs: &BTreeSet<ImagePairId>,
) -> bool {
    if images.len() <= 1 {
        return true;
    }
    if image_pairs.is_empty() {
        return false;
    }

    // Use UnionFind to check connectivity.
    let mut uf: UnionFind<ImageId> = UnionFind::new();
    uf.reserve(images.len());

    for &pair_id in image_pairs {
        // Every id in the set came from `image_pair_to_pair_id`, so the split cannot fail.
        if let Ok((image_id1, image_id2)) = pair_id_to_image_pair(pair_id) {
            uf.union(&image_id1, &image_id2);
        }
    }

    // Check that all images have the same root.
    let Some(first) = images.iter().next() else {
        return true;
    };
    let root = uf.find(first);
    images.iter().all(|node| uf.find(node) == root)
}

/// `SynthesizeSparseMatches`: removes `sparsity` of the exhaustive pairs at random, keeping
/// the view graph connected, and writes the rest.
pub(super) fn synthesize_sparse_matches(
    inlier_match_ratio: f64,
    has_relative_pose: bool,
    sparsity: f64,
    reconstruction: &Reconstruction,
    database: &Database,
) -> Result<()> {
    crate::check_ge!(sparsity, 0.0);
    crate::check_le!(sparsity, 1.0);

    if sparsity == 0.0 {
        return synthesize_exhaustive_matches(
            inlier_match_ratio,
            has_relative_pose,
            reconstruction,
            database,
        );
    }

    if sparsity == 1.0 {
        return Ok(());
    }

    let mut remaining_image_pairs = extract_exhaustive_image_pairs(reconstruction)?;

    let all_image_ids: BTreeSet<ImageId> = reconstruction.reg_image_ids()?.into_iter().collect();

    let num_edges_total = remaining_image_pairs.len();
    // `static_cast<size_t>(sparsity * num_edges_total)` truncates.
    let num_edges_to_remove = (sparsity * num_edges_total as f64) as usize;

    // Try to remove edges randomly while maintaining connectivity.
    random::with_prng(|g| libcxx::shuffle(&mut remaining_image_pairs, g));
    let mut remaining_edges_set: BTreeSet<ImagePairId> =
        remaining_image_pairs.iter().copied().collect();
    let mut edges_removed = 0usize;
    for &pair_id in &remaining_image_pairs {
        if edges_removed >= num_edges_to_remove {
            break;
        }
        remaining_edges_set.remove(&pair_id);
        if is_view_graph_connected(&all_image_ids, &remaining_edges_set) {
            edges_removed += 1;
        } else {
            remaining_edges_set.insert(pair_id);
        }
    }

    for pair_id in remaining_edges_set {
        let two_view_geometry =
            build_two_view_geometry(has_relative_pose, reconstruction, pair_id)?;
        write_two_view_geometry_to_database(
            pair_id,
            &two_view_geometry,
            inlier_match_ratio,
            reconstruction,
            database,
        )?;
    }
    Ok(())
}

/// The fixed reference point WGS84 priors are expressed around (Zurich).
const LAT0: f64 = 47.37851943807808;
const LON0: f64 = 8.549099927632087;
const ALT0: f64 = 451.5;

/// `PosePriorPositionCartesianToWGS84`: the position read as east-north-up meters around the
/// reference point, converted to latitude, longitude and altitude (`kGPSTransform` is the
/// default, GRS80, transform).
pub(super) fn pose_prior_position_cartesian_to_wgs84(pose_prior: &mut PosePrior) {
    pose_prior.position =
        GpsTransform::default().enu_to_ellipsoid(&[pose_prior.position], LAT0, LON0, ALT0)[0];
    pose_prior.coordinate_system = CoordinateSystem::Wgs84;
}

/// `PosePriorPositionWGS84ToCartesian`: the inverse of
/// [`pose_prior_position_cartesian_to_wgs84`].
pub(super) fn pose_prior_position_wgs84_to_cartesian(pose_prior: &mut PosePrior) {
    pose_prior.position =
        GpsTransform::default().ellipsoid_to_enu(&[pose_prior.position], LAT0, LON0, ALT0)[0];
    pose_prior.coordinate_system = CoordinateSystem::Cartesian;
}
