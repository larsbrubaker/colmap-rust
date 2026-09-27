// Port of COLMAP's src/colmap/geometry/normalization_test.cc (all cases), plus a Rust-only
// check of the argument checks.

use colmap_rust::geometry::normalization::{
    center_and_normalize_image_points, compute_bounding_box_and_centroid,
};
use colmap_rust::linalg::{Vector2d, Vector3d};

use super::eigen_matrix_near;

#[test]
fn compute_bounding_box_and_centroid_single_coord() {
    let (bbox, centroid) =
        compute_bounding_box_and_centroid(0.0, 1.0, vec![1.0], vec![2.0], vec![3.0]).unwrap();
    assert_eq!(bbox.min, Vector3d::new(1.0, 2.0, 3.0));
    assert_eq!(bbox.max, Vector3d::new(1.0, 2.0, 3.0));
    assert_eq!(centroid, Vector3d::new(1.0, 2.0, 3.0));
}

#[test]
fn compute_bounding_box_and_centroid_two_coords() {
    let (bbox, centroid) = compute_bounding_box_and_centroid(
        0.0,
        1.0,
        vec![2.0, -1.0],
        vec![3.0, -2.0],
        vec![4.0, -3.0],
    )
    .unwrap();
    assert_eq!(bbox.min, Vector3d::new(-1.0, -2.0, -3.0));
    assert_eq!(bbox.max, Vector3d::new(2.0, 3.0, 4.0));
    assert_eq!(centroid, Vector3d::new(0.5, 0.5, 0.5));
}

#[test]
fn compute_bounding_box_and_centroid_three_coords() {
    let (bbox, centroid) = compute_bounding_box_and_centroid(
        0.0,
        1.0,
        vec![2.0, -1.0, 5.0],
        vec![3.0, -2.0, 5.0],
        vec![4.0, -3.0, 5.0],
    )
    .unwrap();
    assert_eq!(bbox.min, Vector3d::new(-1.0, -2.0, -3.0));
    assert_eq!(bbox.max, Vector3d::new(5.0, 5.0, 5.0));
    assert!(eigen_matrix_near(
        &centroid,
        &Vector3d::new(2.0, 2.0, 2.0),
        1e-6
    ));
}

#[test]
fn compute_bounding_box_and_centroid_five_coords() {
    let xs = vec![2.0, -1.0, 5.0, 100.0, -100.0];
    let ys = vec![3.0, -2.0, 5.0, 100.0, -100.0];
    let zs = vec![4.0, -3.0, 5.0, 100.0, -100.0];
    let (bbox1, centroid1) =
        compute_bounding_box_and_centroid(0.0, 1.0, xs.clone(), ys.clone(), zs.clone()).unwrap();
    assert_eq!(bbox1.min, Vector3d::new(-100.0, -100.0, -100.0));
    assert_eq!(bbox1.max, Vector3d::new(100.0, 100.0, 100.0));
    assert!(eigen_matrix_near(
        &centroid1,
        &Vector3d::new(1.2, 1.2, 1.2),
        1e-6
    ));

    let (bbox2, centroid2) = compute_bounding_box_and_centroid(0.3, 0.7, xs, ys, zs).unwrap();
    assert_eq!(bbox2.min, Vector3d::new(-1.0, -2.0, -3.0));
    assert_eq!(bbox2.max, Vector3d::new(5.0, 5.0, 5.0));
    assert!(eigen_matrix_near(
        &centroid2,
        &Vector3d::new(2.0, 2.0, 2.0),
        1e-6
    ));
}

#[test]
fn center_and_normalize_image_points_nominal() {
    const NUM_POINTS: usize = 11;
    let points: Vec<Vector2d> = (0..NUM_POINTS)
        .map(|i| Vector2d::new(i as f64, i as f64))
        .collect();

    let (normed_points, matrix) = center_and_normalize_image_points(&points).unwrap();

    assert_eq!(matrix[(0, 0)], 0.31622776601683794);
    assert_eq!(matrix[(1, 1)], 0.31622776601683794);
    assert_eq!(matrix[(0, 2)], -1.5811388300841898);
    assert_eq!(matrix[(1, 2)], -1.5811388300841898);

    let mut mean_point = Vector2d::new(0.0, 0.0);
    for &point in &normed_points {
        mean_point += point;
    }
    assert!(mean_point.x.abs() < 1e-6);
    assert!(mean_point.y.abs() < 1e-6);
}

#[test]
fn rust_only_normalization_rejects_bad_input() {
    assert!(compute_bounding_box_and_centroid(0.0, 1.0, vec![], vec![], vec![]).is_err());
    assert!(compute_bounding_box_and_centroid(0.0, 1.0, vec![1.0], vec![], vec![1.0]).is_err());
    assert!(compute_bounding_box_and_centroid(0.7, 0.3, vec![1.0], vec![1.0], vec![1.0]).is_err());
    assert!(compute_bounding_box_and_centroid(-0.1, 1.0, vec![1.0], vec![1.0], vec![1.0]).is_err());
    assert!(center_and_normalize_image_points(&[]).is_err());
}
