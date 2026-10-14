// Rust-only checks for the scene data types, beyond the 1:1 ports next to this file:
// - every metadata method of `Camera` returns docs/CPP_DIVERGENCES.md entry 101's
//   `DomainError` for an unknown model, and `create_from_model_id` rejects it;
// - `Track::element` out of range panics (entry 160);
// - `VisibilityPyramid` with 8 levels keeps the full `max_score` and clamps negative and NaN
//   coordinates into the first cell (entry 161).

use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::track::Track;
use colmap_rust::scene::visibility_pyramid::VisibilityPyramid;
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::{ColmapError, ErrorKind};

fn assert_domain_error<T: std::fmt::Debug>(result: Result<T, ColmapError>) {
    let error = result.expect_err("expected the unknown-model error");
    assert_eq!(error.kind(), ErrorKind::DomainError);
    assert_eq!(error.message(), "Camera model does not exist");
}

#[test]
fn rust_only_camera_metadata_methods_reject_invalid_model() {
    let mut camera = Camera::default();
    assert_domain_error(camera.mean_focal_length());
    assert_domain_error(camera.focal_length());
    assert_domain_error(camera.focal_length_x());
    assert_domain_error(camera.focal_length_y());
    assert_domain_error(camera.set_focal_length(1.0));
    assert_domain_error(camera.set_focal_length_x(1.0));
    assert_domain_error(camera.set_focal_length_y(1.0));
    assert_domain_error(camera.principal_point());
    assert_domain_error(camera.set_principal_point_x(1.0));
    assert_domain_error(camera.set_principal_point_y(1.0));
    assert_domain_error(camera.calibration_matrix());
    assert_domain_error(camera.set_params_from_string("1, 2, 3"));
    assert_domain_error(camera.is_undistorted());
    assert_domain_error(camera.rescale(2.0));
    assert_eq!(
        camera.to_string(),
        "Camera(camera_id=Invalid, model=Invalid, width=0, height=0, params=[] (?))"
    );
}

#[test]
fn rust_only_camera_create_rejects_invalid_model() {
    let error = Camera::create_from_model_id(1, CameraModelId::Invalid, 1.0, 1, 1)
        .expect_err("Invalid must be rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert!(error
        .message()
        .ends_with("Check failed: exists_camera_model_with_id(model_id) "));
    let error = Camera::create_from_model_name(1, "NOT_A_MODEL", 1.0, 1, 1)
        .expect_err("an unknown name must be rejected");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
}

#[test]
fn rust_only_camera_checks_fail_with_errors() {
    let mut camera = Camera::create_from_model_id(1, CameraModelId::Pinhole, 1.0, 1, 1).unwrap();
    // FocalLength() needs a single focal length; PINHOLE has fx, fy.
    assert_eq!(
        camera.focal_length().unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
    assert_eq!(
        camera.rescale(0.0).unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
    camera = Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1.0, 1, 1).unwrap();
    assert!(camera.set_focal_length_x(2.0).is_err());
}

#[test]
#[should_panic]
fn rust_only_track_element_out_of_range_panics() {
    let mut track = Track::new();
    track.add_element_ids(0, 1);
    let _ = track.element(1);
}

#[test]
fn rust_only_track_delete_element_out_of_range_fails_check() {
    let mut track = Track::new();
    track.add_element_ids(0, 1);
    let error = track.delete_element(1).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    assert_eq!(track.length(), 1);
}

#[test]
fn rust_only_visibility_pyramid_eight_levels_max_score() {
    // sum over l = 1..=8 of (2^l)^4 = (16^9 - 16) / 15, beyond i32 (COLMAP's int overflows).
    let mut pyramid = VisibilityPyramid::new(8, 256, 256);
    assert_eq!(pyramid.max_score(), 4_581_298_448);
    pyramid.set_point(0.0, 0.0).unwrap();
    let full: usize = (1..=8).map(|l| (1usize << l) * (1usize << l)).sum();
    assert_eq!(pyramid.score(), full);
}

#[test]
fn rust_only_visibility_pyramid_out_of_range_points_clamp() {
    let mut pyramid = VisibilityPyramid::new(2, 4, 4);
    pyramid.set_point(-3.0, f64::NAN).unwrap();
    let score = pyramid.score();
    // The same cell as (0, 0): no new cell is populated.
    pyramid.set_point(0.0, 0.0).unwrap();
    assert_eq!(pyramid.score(), score);
    // Beyond the range clamps into the last cell.
    pyramid.set_point(100.0, 100.0).unwrap();
    let score = pyramid.score();
    pyramid.set_point(3.9, 3.9).unwrap();
    assert_eq!(pyramid.score(), score);
    assert!(VisibilityPyramid::default().set_point(0.0, 0.0).is_err());
    assert!(VisibilityPyramid::new(1, 0, 4).set_point(0.0, 0.0).is_err());
}
