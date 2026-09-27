// Port of COLMAP's src/colmap/geometry/bbox_test.cc (all cases), plus a Rust-only check of
// the split-count checks.

use colmap_rust::geometry::bbox::compute_equal_parts_bboxes;
use colmap_rust::linalg::{AlignedBox3d, Vector3d};

use super::eigen_matrix_near;

fn aligned_box(min: [f64; 3], max: [f64; 3]) -> AlignedBox3d {
    AlignedBox3d::new(Vector3d::from_array(min), Vector3d::from_array(max))
}

#[test]
fn compute_equal_parts_bboxes_split1x1x2() {
    let bbox = aligned_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    let bboxes = compute_equal_parts_bboxes(&bbox, [1, 1, 1]).unwrap();
    assert_eq!(bboxes.len(), 1);
    assert!(eigen_matrix_near(
        &bboxes[0].min,
        &Vector3d::new(0.0, 0.0, 0.0),
        1e-10
    ));
    assert!(eigen_matrix_near(
        &bboxes[0].max,
        &Vector3d::new(1.0, 1.0, 1.0),
        1e-10
    ));
}

#[test]
fn compute_equal_parts_bboxes_split2x2x2() {
    let bbox = aligned_box([0.0, 0.0, 0.0], [2.0, 2.0, 2.0]);
    let bboxes = compute_equal_parts_bboxes(&bbox, [2, 2, 2]).unwrap();
    assert_eq!(bboxes.len(), 8);
    for sub_bbox in &bboxes {
        assert!(eigen_matrix_near(
            &sub_bbox.diagonal(),
            &Vector3d::new(1.0, 1.0, 1.0),
            1e-10
        ));
    }
    let covered = bboxes
        .iter()
        .fold(AlignedBox3d::empty(), |acc, b| acc.extend(*b));
    assert!(eigen_matrix_near(&covered.min, &bbox.min, 1e-10));
    assert!(eigen_matrix_near(&covered.max, &bbox.max, 1e-10));
}

#[test]
fn compute_equal_parts_bboxes_asymmetric() {
    let bbox = aligned_box([0.0, 0.0, 0.0], [6.0, 4.0, 2.0]);
    let bboxes = compute_equal_parts_bboxes(&bbox, [3, 2, 1]).unwrap();
    assert_eq!(bboxes.len(), 6);
    for sub_bbox in &bboxes {
        assert!(eigen_matrix_near(
            &sub_bbox.diagonal(),
            &Vector3d::new(2.0, 2.0, 2.0),
            1e-10
        ));
    }
}

#[test]
fn compute_equal_parts_bboxes_with_offset() {
    let bbox = aligned_box([10.0, 20.0, 30.0], [20.0, 30.0, 40.0]);
    let bboxes = compute_equal_parts_bboxes(&bbox, [2, 2, 2]).unwrap();
    assert_eq!(bboxes.len(), 8);
    // Check that sub-boxes cover the original box
    let covered = bboxes
        .iter()
        .fold(AlignedBox3d::empty(), |acc, b| acc.extend(*b));
    assert!(eigen_matrix_near(&covered.min, &bbox.min, 1e-10));
    assert!(eigen_matrix_near(&covered.max, &bbox.max, 1e-10));
}

#[test]
fn rust_only_compute_equal_parts_bboxes_rejects_non_positive_split() {
    let bbox = aligned_box([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
    assert!(compute_equal_parts_bboxes(&bbox, [0, 1, 1]).is_err());
    assert!(compute_equal_parts_bboxes(&bbox, [1, -1, 1]).is_err());
    assert!(compute_equal_parts_bboxes(&bbox, [1, 1, 0]).is_err());
}
