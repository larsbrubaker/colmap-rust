// Port of colmap/scene/point3d_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::scene::point3d::Point3D. `operator<<` becomes `Display`.

use colmap_rust::linalg::{Vector3d, Vector3ub};
use colmap_rust::scene::point3d::Point3D;

#[test]
fn point3d_default() {
    let point3d = Point3D::default();
    assert_eq!(point3d.xyz, Vector3d::zeros());
    assert_eq!(point3d.color, Vector3ub::default());
    assert_eq!(point3d.error, -1.0);
    assert!(!point3d.has_error());
    assert_eq!(point3d.track.length(), 0);
}

#[test]
fn point3d_equals() {
    let mut point3d = Point3D::default();
    let mut other = point3d.clone();
    assert_eq!(point3d, other);
    point3d.xyz.x += 1.0;
    assert_ne!(point3d, other);
    other.xyz.x += 1.0;
    assert_eq!(point3d, other);
}

#[test]
fn point3d_print() {
    let point3d = Point3D {
        xyz: Vector3d::new(1.0, 2.0, 3.0),
        ..Point3D::default()
    };
    assert_eq!(point3d.to_string(), "Point3D(xyz=[1, 2, 3], track_len=0)");
}

#[test]
fn point3d_error() {
    let mut point3d = Point3D::default();
    assert_eq!(point3d.error, -1.0);
    assert!(!point3d.has_error());
    point3d.error = 1.0;
    assert_eq!(point3d.error, 1.0);
    assert!(point3d.has_error());
}
