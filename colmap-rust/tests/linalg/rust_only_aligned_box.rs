// Rust-only (COLMAP has no test for Eigen itself): AlignedBox3d against Eigen's documented
// semantics (empty default box, extend = coefficient-wise min/max, contains includes the
// border and rejects NaN, diagonal = max - min). colmap-sharp exercises its AlignedBox3d only
// through its Bbox tests; these are written here.

use colmap_rust::linalg::{AlignedBox3d, Vector3d};

#[test]
fn rust_only_default_box_is_empty_and_extends_to_the_other_box() {
    let empty = AlignedBox3d::default();
    assert_eq!(empty, AlignedBox3d::empty());
    assert_eq!(empty.min, Vector3d::new(f64::MAX, f64::MAX, f64::MAX));
    assert_eq!(empty.max, Vector3d::new(f64::MIN, f64::MIN, f64::MIN));
    assert!(!empty.contains(Vector3d::zeros()));

    let b = AlignedBox3d::new(Vector3d::new(-1.0, 0.0, 2.0), Vector3d::new(1.0, 3.0, 4.0));
    assert_eq!(empty.extend(b), b);
    assert_eq!(b.extend(empty), b);
}

#[test]
fn rust_only_extend_contains_diagonal() {
    let a = AlignedBox3d::new(Vector3d::new(0.0, 0.0, 0.0), Vector3d::new(1.0, 1.0, 1.0));
    let b = AlignedBox3d::new(
        Vector3d::new(-1.0, 0.5, 0.25),
        Vector3d::new(0.5, 2.0, 0.75),
    );
    let union = a.extend(b);
    assert_eq!(union.min, Vector3d::new(-1.0, 0.0, 0.0));
    assert_eq!(union.max, Vector3d::new(1.0, 2.0, 1.0));
    assert_eq!(union.diagonal(), Vector3d::new(2.0, 2.0, 1.0));

    // Borders are inside; NaN coordinates never are.
    assert!(a.contains(Vector3d::new(0.0, 1.0, 0.5)));
    assert!(!a.contains(Vector3d::new(1.0 + f64::EPSILON, 0.5, 0.5)));
    assert!(!a.contains(Vector3d::new(f64::NAN, 0.5, 0.5)));
}
