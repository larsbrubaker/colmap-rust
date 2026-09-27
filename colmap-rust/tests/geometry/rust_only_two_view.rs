// Rust-only tests for two-view geometry functions that COLMAP's own tests do not reach
// (triangulate_optimal_point, compute_squared_homography_error,
// compute_squared_sampson_errors_homogeneous) or only reach with one nominal case
// (decompose_projection_matrix, compute_closest_rotation_matrix, average_directions).
// Hand-built cases with known answers: noise-free data gives zero error or the exact point,
// noisy data is checked against a direct computation.

use colmap_rust::geometry::essential_matrix::{
    compute_squared_sampson_errors, compute_squared_sampson_errors_homogeneous,
    essential_matrix_from_pose, find_optimal_image_observations,
};
use colmap_rust::geometry::homography_matrix::{
    compute_squared_homography_error, homography_matrix_from_pose,
};
use colmap_rust::geometry::pose::{
    average_directions, compute_closest_rotation_matrix, decompose_projection_matrix,
};
use colmap_rust::geometry::triangulation::{triangulate_optimal_point, triangulate_point};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Matrix3x4d, Quaterniond, Vector2d, Vector3d};

use super::eigen_matrix_near;

fn test_pose() -> Rigid3d {
    Rigid3d::new(
        Quaterniond::from_angle_axis(AngleAxisd::new(
            0.2,
            Vector3d::new(0.1, 1.0, 0.3).normalized(),
        )),
        Vector3d::new(1.0, 0.1, 0.2),
    )
}

// COLMAP's FindOptimalImageObservations evaluates the constraint as `x1^T E x2 = 0`
// (Lindstrom's convention), while EssentialMatrixFromPose, and so TriangulateOptimalPoint,
// use `x2^T E x1 = 0`. The two agree only when E is skew-symmetric (a pure translation).
// These tests pin COLMAP's behavior as it is; the port reproduces it (colmap-sharp does
// too), and fixing it would be a deliberate divergence.

#[test]
fn rust_only_triangulate_optimal_point_noise_free_pure_translation_is_exact() {
    let cam1_from_world = Rigid3d::identity();
    let cam2_from_world = Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.1, 0.2));
    for point3d in [
        Vector3d::new(0.0, 0.0, 4.0),
        Vector3d::new(0.5, -0.3, 3.0),
        Vector3d::new(-1.0, 0.7, 6.0),
    ] {
        let point1 = (cam1_from_world * point3d).hnormalized();
        let point2 = (cam2_from_world * point3d).hnormalized();
        let tri = triangulate_optimal_point(
            &cam1_from_world.to_matrix(),
            &cam2_from_world.to_matrix(),
            point1,
            point2,
        )
        .expect("triangulated");
        assert!(eigen_matrix_near(&tri, &point3d, 1e-10));
    }
}

#[test]
fn rust_only_find_optimal_image_observations_uses_x1_e_x2_convention() {
    // With a rotation, noise-free points satisfy x2^T E x1 = 0; they are fixed points of
    // the correction only when it is given E^T (i.e. its constraint is x1^T E x2 = 0).
    let pose = test_pose();
    let e = essential_matrix_from_pose(&pose);
    let x = Vector3d::new(0.5, -0.3, 3.0);
    let point1 = x.hnormalized();
    let point2 = (pose * x).hnormalized();
    let (c1, c2) = find_optimal_image_observations(&e.transpose(), point1, point2);
    assert!(eigen_matrix_near(&c1, &point1, 1e-12));
    assert!(eigen_matrix_near(&c2, &point2, 1e-12));

    // Given E itself (as TriangulateOptimalPoint passes it), the same exact observations
    // are moved, because they violate x1^T E x2 = 0.
    let (d1, d2) = find_optimal_image_observations(&e, point1, point2);
    assert!((d1 - point1).norm() + (d2 - point2).norm() > 1e-3);
}

#[test]
fn rust_only_triangulate_optimal_point_is_dlt_on_corrected_points() {
    let p1 = Rigid3d::identity().to_matrix();
    let cam2_from_world = test_pose();
    let p2 = cam2_from_world.to_matrix();
    let point3d = Vector3d::new(0.4, -0.2, 5.0);
    let noise = Vector2d::new(1e-3, -2e-3);
    let point1 = point3d.hnormalized() + noise;
    let point2 = (cam2_from_world * point3d).hnormalized() - noise;

    let tri = triangulate_optimal_point(&p1, &p2, point1, point2).expect("triangulated");

    // Direct computation: correct the observations, then DLT. triangulate_optimal_point
    // rebuilds the pose from the matrices; with an exact rotation matrix that round trip is
    // the same pose to rounding.
    let e = essential_matrix_from_pose(&cam2_from_world);
    let (c1, c2) = find_optimal_image_observations(&e, point1, point2);
    let direct = triangulate_point(&p1, &p2, c1, c2).expect("triangulated");
    assert!(eigen_matrix_near(&tri, &direct, 1e-9));
}

#[test]
fn rust_only_compute_squared_homography_error_known_values() {
    let h = Matrix3d::identity();
    assert_eq!(
        compute_squared_homography_error(Vector2d::new(1.0, 2.0), Vector2d::new(4.0, 6.0), &h),
        25.0
    );
    // A pure scale by 2 in x: (1, 2) -> (2, 2).
    let h = Matrix3d::new(2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
    assert_eq!(
        compute_squared_homography_error(Vector2d::new(1.0, 2.0), Vector2d::new(2.0, 3.0), &h),
        1.0
    );
    // A point mapped to infinity (third homogeneous coordinate 0).
    let h = Matrix3d::new(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0);
    assert_eq!(
        compute_squared_homography_error(Vector2d::new(0.0, 1.0), Vector2d::new(0.0, 0.0), &h),
        f64::MAX
    );
}

#[test]
fn rust_only_compute_squared_homography_error_zero_on_plane() {
    let k = Matrix3d::new(500.0, 0.0, 320.0, 0.0, 510.0, 240.0, 0.0, 0.0, 1.0);
    let pose = test_pose();
    let n = Vector3d::new(0.0, 0.0, -1.0);
    let d = 4.0;
    let h = homography_matrix_from_pose(
        &k,
        &k,
        &pose.rotation.to_rotation_matrix(),
        pose.translation,
        n,
        d,
    )
    .unwrap();
    // Points on the plane n . X + d = 0, i.e. z = 4, seen by both cameras.
    for xy in [(0.0, 0.0), (0.5, -0.3), (-1.0, 0.8)] {
        let x = Vector3d::new(xy.0, xy.1, 4.0);
        let p1 = (k * x).hnormalized();
        let p2 = (k * (pose * x)).hnormalized();
        assert!(compute_squared_homography_error(p1, p2, &h) < 1e-18);
    }
}

#[test]
fn rust_only_compute_squared_sampson_errors_homogeneous() {
    let e = essential_matrix_from_pose(&test_pose());
    let points1 = vec![
        Vector2d::new(0.1, 0.2),
        Vector2d::new(-0.3, 0.05),
        Vector2d::new(0.4, -0.4),
    ];
    let points2 = vec![
        Vector2d::new(0.15, 0.1),
        Vector2d::new(-0.2, 0.3),
        Vector2d::new(0.0, 0.0),
    ];
    let h1: Vec<Vector3d> = points1.iter().map(|p| p.homogeneous()).collect();
    let h2: Vec<Vector3d> = points2.iter().map(|p| p.homogeneous()).collect();

    let from_homogeneous = compute_squared_sampson_errors_homogeneous(&h1, &h2, &e).unwrap();
    let from_points = compute_squared_sampson_errors(&points1, &points2, &e).unwrap();
    assert_eq!(from_homogeneous, from_points);

    // Direct computation of the first residual: (x2^T E x1)^2 over the gradient norm.
    let ex1 = e * h1[0];
    let etx2 = e.transpose() * h2[0];
    let num = h2[0].dot(ex1);
    let denom = etx2.x * etx2.x + etx2.y * etx2.y + ex1.x * ex1.x + ex1.y * ex1.y;
    assert!(((from_homogeneous[0] - num * num / denom) / from_homogeneous[0]).abs() < 1e-14);

    // Noise-free correspondences have zero error.
    let pose = test_pose();
    let x = Vector3d::new(0.3, -0.2, 5.0);
    let exact = compute_squared_sampson_errors_homogeneous(&[x], &[pose * x], &e).unwrap();
    assert!(exact[0] < 1e-30);

    // Mismatched lengths fail the check.
    assert!(compute_squared_sampson_errors_homogeneous(&h1, &h2[..2], &e).is_err());
}

#[test]
fn rust_only_decompose_projection_matrix_general_calibration() {
    // Skewed calibration with different focal lengths and a non-trivial pose.
    let ref_k = Matrix3d::new(520.0, 3.0, 330.0, 0.0, 505.0, 250.0, 0.0, 0.0, 1.0);
    let pose = test_pose();
    let p = ref_k * pose.to_matrix();
    let (k, r, t) = decompose_projection_matrix(&p).expect("non-singular");
    assert!(eigen_matrix_near(&k, &ref_k, 1e-10));
    assert!(eigen_matrix_near(
        &r,
        &pose.rotation.to_rotation_matrix(),
        1e-10
    ));
    assert!(eigen_matrix_near(&t, &pose.translation, 1e-10));

    // A projection matrix scaled by a positive factor gives the scaled K and the same pose.
    let (k2, r2, t2) = decompose_projection_matrix(&(2.0 * p)).expect("non-singular");
    assert!(eigen_matrix_near(&k2, &(2.0 * ref_k), 1e-10));
    assert!(eigen_matrix_near(&r2, &r, 1e-10));
    assert!(eigen_matrix_near(&t2, &t, 1e-10));
}

#[test]
fn rust_only_decompose_projection_matrix_singular_is_none() {
    // A zero left 3x3 block: det(K) == 0.
    let p = Matrix3x4d::new(
        0.0, 0.0, 0.0, 1.0, //
        0.0, 0.0, 0.0, 2.0, //
        0.0, 0.0, 0.0, 3.0,
    );
    assert!(decompose_projection_matrix(&p).is_none());
}

#[test]
fn rust_only_compute_closest_rotation_matrix_perturbed_and_reflected() {
    let r = test_pose().rotation.to_rotation_matrix();

    // A reflection -R: U V^T has det -1 and is negated back to R.
    assert!(eigen_matrix_near(
        &compute_closest_rotation_matrix(&(-r)),
        &r,
        1e-12
    ));

    // A small perturbation projects back to a proper rotation close to R.
    let noise = Matrix3d::new(1e-3, -2e-3, 0.0, 5e-4, 0.0, 1e-3, -1e-3, 2e-3, 3e-4);
    let closest = compute_closest_rotation_matrix(&(r + noise));
    assert!(eigen_matrix_near(
        &(closest.transpose() * closest),
        &Matrix3d::identity(),
        1e-12
    ));
    assert!((closest.determinant() - 1.0).abs() < 1e-12);
    assert!((closest - r).norm() < 5e-3);
}

#[test]
fn rust_only_average_directions_weights_and_checks() {
    // Symmetric pair around +z, unequal lengths (each is normalized first).
    let vectors = [Vector3d::new(0.1, 0.0, 1.0), Vector3d::new(-0.2, 0.0, 2.0)];
    let avg = average_directions(&vectors, &[]).unwrap();
    assert!(eigen_matrix_near(&avg, &Vector3d::unit_z(), 1e-12));

    // Orthogonal inputs: A = (10 x x^T + z z^T) / 11, whose principal direction is exactly
    // the heavier input.
    let vectors = [Vector3d::unit_x(), Vector3d::unit_z()];
    let avg = average_directions(&vectors, &[10.0, 1.0]).unwrap();
    assert!(eigen_matrix_near(&avg, &Vector3d::unit_x(), 1e-12));

    // Non-orthogonal inputs: a heavier weight pulls the average towards that direction.
    let vectors = [Vector3d::unit_x(), Vector3d::new(1.0, 0.0, 1.0)];
    let even = average_directions(&vectors, &[1.0, 1.0]).unwrap();
    let heavy_x = average_directions(&vectors, &[5.0, 1.0]).unwrap();
    assert!(heavy_x.z < even.z && heavy_x.z > 0.0);
    assert!((heavy_x.norm() - 1.0).abs() < 1e-12);

    // Opposite-signed inputs: the result follows the weighted majority's sign.
    let vectors = [Vector3d::unit_x(), -Vector3d::unit_x(), Vector3d::unit_x()];
    let avg = average_directions(&vectors, &[]).unwrap();
    assert!(eigen_matrix_near(&avg, &Vector3d::unit_x(), 1e-12));

    // COLMAP's checks.
    assert!(average_directions(&[], &[]).is_err());
    assert!(average_directions(&vectors, &[1.0, 1.0]).is_err());
    assert!(average_directions(&vectors, &[1.0, 0.0, 1.0]).is_err());
}
