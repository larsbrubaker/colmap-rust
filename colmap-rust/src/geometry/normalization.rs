//! Port of COLMAP's `colmap/geometry/normalization.h` and `normalization.cc`: the percentile
//! bounding box and centroid of a point set, and Hartley normalization of image points. Port
//! of colmap-sharp's `Geometry/Normalization.cs`. Tests: `tests/geometry/normalization.rs`
//! (normalization_test.cc).
//!
//! Tier A for [`center_and_normalize_image_points`] and for the bounding box of
//! [`compute_bounding_box_and_centroid`]; the centroid can differ in the last bits because the
//! coordinates are fully sorted instead of partitioned with `std::nth_element`
//! (docs/CPP_DIVERGENCES.md entry 81).

use crate::linalg::{AlignedBox3d, Matrix3d, Vector2d, Vector3d};
use crate::math::fns;
use crate::{check, check_eq, check_ge, check_gt, check_le, Result};

/// Port of `colmap::ComputeBoundingBoxAndCentroid`: the axis-aligned box spanned by the
/// coordinates at the `min_percentile` and `max_percentile` positions of each axis, and the
/// mean of the coordinates between those positions (inclusive). Percentiles are in [0, 1].
pub fn compute_bounding_box_and_centroid(
    min_percentile: f64,
    max_percentile: f64,
    mut coords_x: Vec<f64>,
    mut coords_y: Vec<f64>,
    mut coords_z: Vec<f64>,
) -> Result<(AlignedBox3d, Vector3d)> {
    check!(!coords_x.is_empty());
    check_eq!(coords_x.len(), coords_y.len());
    check_eq!(coords_x.len(), coords_z.len());
    check_ge!(min_percentile, 0.0);
    check_le!(min_percentile, 1.0);
    check_ge!(max_percentile, 0.0);
    check_le!(max_percentile, 1.0);
    check_le!(min_percentile, max_percentile);

    let end_idx = coords_x.len() - 1;
    // `static_cast<size_t>(std::floor(...))` of a value in [0, end_idx]; floor and ceil are
    // exact, so they need not go through `fns`.
    let min_idx = end_idx.min((min_percentile * end_idx as f64).floor() as usize);
    let max_idx = end_idx.min((max_percentile * end_idx as f64).ceil() as usize);

    // COLMAP partitions with two std::nth_element calls per axis. A full sort satisfies every
    // nth_element postcondition, so the box is the same; only the summation order of the
    // centroid can differ (entry 81). `total_cmp` orders -0.0 before 0.0, which are equal
    // values either way.
    coords_x.sort_by(f64::total_cmp);
    coords_y.sort_by(f64::total_cmp);
    coords_z.sort_by(f64::total_cmp);

    let bbox_min = Vector3d::new(coords_x[min_idx], coords_y[min_idx], coords_z[min_idx]);
    let bbox_max = Vector3d::new(coords_x[max_idx], coords_y[max_idx], coords_z[max_idx]);

    let mut centroid = Vector3d::new(0.0, 0.0, 0.0);
    let normalization = 1.0 / (max_idx - min_idx + 1) as f64;
    for i in min_idx..=max_idx {
        centroid.x += normalization * coords_x[i];
        centroid.y += normalization * coords_y[i];
        centroid.z += normalization * coords_z[i];
    }

    Ok((AlignedBox3d::new(bbox_min, bbox_max), centroid))
}

/// Port of `colmap::CenterAndNormalizeImagePoints`: translates the points so their centroid
/// is the origin and scales them so their root-mean-square distance to it is `sqrt(2)`.
/// Returns the normalized points and the 3x3 `normed_from_orig` matrix. Needs at least one
/// point.
pub fn center_and_normalize_image_points(points: &[Vector2d]) -> Result<(Vec<Vector2d>, Matrix3d)> {
    let num_points = points.len();
    check_gt!(num_points, 0);

    // Calculate centroid.
    let mut centroid = Vector2d::new(0.0, 0.0);
    for &point in points {
        centroid += point;
    }
    centroid /= num_points as f64;

    // Root mean square distance to centroid of all points.
    let mut rms_mean_dist = 0.0;
    for &point in points {
        rms_mean_dist += (point - centroid).squared_norm();
    }
    rms_mean_dist = fns::sqrt(rms_mean_dist / num_points as f64);

    // Compose normalization matrix.
    let norm_factor = fns::sqrt(2.0) / rms_mean_dist;
    let normed_from_orig = Matrix3d::new(
        norm_factor,
        0.0,
        -norm_factor * centroid.x,
        0.0,
        norm_factor,
        -norm_factor * centroid.y,
        0.0,
        0.0,
        1.0,
    );

    // Apply normalization matrix.
    let normed_points = points
        .iter()
        .map(|&point| (normed_from_orig * point.homogeneous()).hnormalized())
        .collect();
    Ok((normed_points, normed_from_orig))
}
