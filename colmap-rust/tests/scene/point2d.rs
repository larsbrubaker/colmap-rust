// Port of colmap/scene/point2d_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::scene::point2d::Point2D. `operator<<` becomes `Display`.

use colmap_rust::linalg::Vector2d;
use colmap_rust::scene::point2d::Point2D;
use colmap_rust::util::types::INVALID_POINT3D_ID;

#[test]
fn point2d_default() {
    let point2d = Point2D::default();
    assert_eq!(point2d.xy, Vector2d::zeros());
    assert_eq!(point2d.point3d_id, INVALID_POINT3D_ID);
    assert!(!point2d.has_point3d());
}

#[test]
fn point2d_equals() {
    let mut point2d = Point2D::default();
    let mut other = point2d;
    assert_eq!(point2d, other);
    point2d.xy.x += 1.0;
    assert_ne!(point2d, other);
    other.xy.x += 1.0;
    assert_eq!(point2d, other);
}

#[test]
fn point2d_print() {
    let point2d = Point2D {
        xy: Vector2d::new(1.0, 2.0),
        ..Point2D::default()
    };
    assert_eq!(point2d.to_string(), "Point2D(xy=[1, 2], point3D_id=-1)");
}

#[test]
fn point2d_point3d_id() {
    let mut point2d = Point2D::default();
    assert_eq!(point2d.point3d_id, INVALID_POINT3D_ID);
    assert!(!point2d.has_point3d());
    point2d.point3d_id = 1;
    assert_eq!(point2d.point3d_id, 1);
    assert!(point2d.has_point3d());
    point2d.point3d_id = INVALID_POINT3D_ID;
    assert_eq!(point2d.point3d_id, INVALID_POINT3D_ID);
    assert!(!point2d.has_point3d());
}
