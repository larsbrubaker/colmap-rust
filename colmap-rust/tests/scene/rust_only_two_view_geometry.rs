// Rust-only checks for colmap_rust::scene::two_view_geometry, beyond the 1:1 port in
// `two_view_geometry.rs`: `Invert` swaps the recovered cameras and inverts a non-trivial
// homography, matrices left unset stay unset, and `ConfigurationType` keeps COLMAP's
// stored integers.

use colmap_rust::linalg::Matrix3d;
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};

#[test]
fn rust_only_two_view_geometry_default_tri_angle() {
    let two_view_geometry = TwoViewGeometry::default();
    assert_eq!(two_view_geometry.tri_angle, -1.0);
    assert!(two_view_geometry.camera1.is_none());
    assert!(two_view_geometry.camera2.is_none());
}

#[test]
fn rust_only_two_view_geometry_invert_swaps_cameras_and_inverts_h() {
    let camera = Camera {
        camera_id: 7,
        ..Camera::default()
    };
    let h = Matrix3d::new(2.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0);
    let mut two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Uncalibrated,
        h: Some(h),
        camera1: Some(camera.clone()),
        ..TwoViewGeometry::default()
    };
    two_view_geometry.invert();
    assert!(two_view_geometry.camera1.is_none());
    assert_eq!(two_view_geometry.camera2, Some(camera));
    assert!(two_view_geometry.f.is_none());
    assert!(two_view_geometry.e.is_none());
    assert!(two_view_geometry.cam2_from_cam1.is_none());
    assert_eq!(
        two_view_geometry.h,
        Some(Matrix3d::new(0.5, 0.0, -0.5, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0))
    );
}

#[test]
fn rust_only_two_view_geometry_configuration_integers() {
    for value in 0..=9 {
        let config = ConfigurationType::from_i32(value).unwrap();
        assert_eq!(config.as_i32(), value);
    }
    assert_eq!(ConfigurationType::CalibratedRig.as_i32(), 9);
    assert_eq!(ConfigurationType::Uncalibrated.as_i32(), 3);
    assert_eq!(ConfigurationType::from_i32(-1), None);
    assert_eq!(ConfigurationType::from_i32(10), None);
}
