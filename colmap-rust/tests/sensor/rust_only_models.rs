// Rust-only camera model tests (complement the 1:1 port in models.rs):
// - the boundary functions return COLMAP's domain_error as `Err(DomainError)` for
//   `CameraModelId::Invalid`, and the per-point functions panic with the same message
//   (docs/CPP_DIVERGENCES.md, entry 101);
// - `cam_ray_from_img_jacobian` against a hand-derived PINHOLE Jacobian. COLMAP tests it
//   through models_jacobian_test.cc, ported as the unit tests in
//   `src/sensor/models/jacobian_tests.rs` (they need the crate-private Jet).

use colmap_rust::linalg::{Matrix2x3d, Vector2d, Vector3d};
use colmap_rust::sensor::models::*;
use colmap_rust::{ColmapError, ErrorKind};

fn assert_domain_error<T: std::fmt::Debug>(result: Result<T, ColmapError>) {
    let error = result.expect_err("Invalid must be an error");
    assert_eq!(error.kind(), ErrorKind::DomainError);
    assert_eq!(error.message(), "Camera model does not exist");
}

#[test]
fn rust_only_boundary_functions_reject_invalid_model() {
    let id = CameraModelId::Invalid;
    let params = [1.0, 2.0, 3.0];
    assert_domain_error(camera_model_initialize_params(id, 100.0, 100, 100));
    assert_domain_error(camera_model_params_info(id));
    assert_domain_error(camera_model_focal_length_idxs(id));
    assert_domain_error(camera_model_principal_point_idxs(id));
    assert_domain_error(camera_model_extra_params_idxs(id));
    assert_domain_error(camera_model_meta_data_params_idxs(id));
    assert_domain_error(camera_model_num_params(id));
    assert_domain_error(camera_model_verify_params(id, &params));
    assert_domain_error(camera_model_has_bogus_params(
        id, &params, 100, 100, 0.1, 2.0, 1.0,
    ));

    // The non-throwing lookups answer for Invalid, as in COLMAP.
    assert_eq!(
        camera_model_name_to_id("NOT_A_MODEL"),
        CameraModelId::Invalid
    );
    assert_eq!(camera_model_id_to_name(id), "");
    assert!(!exists_camera_model_with_id(id));
    assert!(!camera_model_is_perspective_fisheye(id));
}

#[test]
#[should_panic(expected = "Camera model does not exist")]
fn rust_only_projection_panics_on_invalid_model() {
    camera_model_img_from_cam(
        CameraModelId::Invalid,
        &[1.0, 2.0, 3.0],
        Vector3d::new(0.0, 0.0, 1.0),
        true,
    );
}

#[test]
#[should_panic(expected = "Camera model does not exist")]
fn rust_only_unprojection_panics_on_invalid_model() {
    camera_model_cam_from_img(CameraModelId::Invalid, &[1.0], Vector2d::new(0.0, 0.0));
}

/// PINHOLE's projection Jacobian d(x, y) / d(u, v, w) at (u, v, w), derived by hand from
/// x = fx u / w + cx, y = fy v / w + cy.
fn pinhole_j_uvw(fx: f64, fy: f64, ray: Vector3d) -> Matrix2x3d {
    let (u, v, w) = (ray.x, ray.y, ray.z);
    Matrix2x3d::new(
        fx / w,
        0.0,
        -fx * u / (w * w),
        0.0,
        fy / w,
        -fy * v / (w * w),
    )
}

#[test]
fn rust_only_cam_ray_from_img_jacobian_pinhole_on_axis() {
    // On the optical axis the unit ray moves as (dx / fx, dy / fy, 0).
    let (fx, fy) = (500.0, 400.0);
    let ray = Vector3d::new(0.0, 0.0, 1.0);
    let j_ray = cam_ray_from_img_jacobian(ray, pinhole_j_uvw(fx, fy, ray)).expect("rank 2");
    let expected = [[1.0 / fx, 0.0], [0.0, 1.0 / fy], [0.0, 0.0]];
    for (r, row) in expected.iter().enumerate() {
        for (c, &e) in row.iter().enumerate() {
            assert!(
                (j_ray[(r, c)] - e).abs() <= 1e-18,
                "({r},{c}): {}",
                j_ray[(r, c)]
            );
        }
    }
}

#[test]
fn rust_only_cam_ray_from_img_jacobian_pinhole_off_axis() {
    // Off axis: J_ray is the pseudo-inverse of J_uvw at the unit ray, so J_uvw J_ray = I and
    // the ray is orthogonal to both columns (they lie in the sphere's tangent plane). It must
    // also match the finite-difference derivative of the normalized unprojection.
    let (fx, fy, cx, cy) = (500.0, 400.0, 320.0, 240.0);
    let params = [fx, fy, cx, cy];
    let pixel = Vector2d::new(470.0, 160.0);
    let ray = camera_model_cam_ray_from_img(CameraModelId::Pinhole, &params, pixel).unwrap();
    let j_uvw = pinhole_j_uvw(fx, fy, ray);
    let j_ray = cam_ray_from_img_jacobian(ray, j_uvw).expect("rank 2");

    let product = j_uvw * j_ray;
    for r in 0..2 {
        for c in 0..2 {
            let e = if r == c { 1.0 } else { 0.0 };
            assert!(
                (product[(r, c)] - e).abs() <= 1e-12,
                "J_uvw J_ray ({r},{c})"
            );
        }
        assert!(ray.dot(j_ray.col(r)).abs() <= 1e-15);
    }

    let h = 1e-4;
    for (c, step) in [Vector2d::new(h, 0.0), Vector2d::new(0.0, h)]
        .into_iter()
        .enumerate()
    {
        let plus =
            camera_model_cam_ray_from_img(CameraModelId::Pinhole, &params, pixel + step).unwrap();
        let minus =
            camera_model_cam_ray_from_img(CameraModelId::Pinhole, &params, pixel - step).unwrap();
        let numeric = (plus - minus) / (2.0 * h);
        assert!((numeric - j_ray.col(c)).norm() <= 1e-10, "column {c}");
    }
}

#[test]
fn rust_only_cam_ray_from_img_jacobian_rank_deficient() {
    // Identical rows: rank 1, so the triple product vanishes and there is no inverse.
    let ray = Vector3d::new(0.0, 0.0, 1.0);
    let j_uvw = Matrix2x3d::new(500.0, 0.0, 0.0, 500.0, 0.0, 0.0);
    assert!(cam_ray_from_img_jacobian(ray, j_uvw).is_none());
}
