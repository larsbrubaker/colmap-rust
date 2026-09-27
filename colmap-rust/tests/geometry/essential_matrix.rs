// Port of COLMAP's src/colmap/geometry/essential_matrix_test.cc. Random values come from
// COLMAP's seeded PRNG (math::random_eigen), seeded with 0 like gtest_main.

use colmap_rust::geometry::essential_matrix::{
    compute_squared_sampson_error, compute_squared_sampson_errors,
    compute_squared_tangent_sampson_error, compute_squared_tangent_sampson_errors,
    compute_squared_tangent_sampson_errors_with_cheirality, decompose_essential_matrix,
    epipole_from_essential_matrix, essential_from_fundamental_matrix, essential_matrix_from_pose,
    find_optimal_image_observations, fundamental_from_essential_matrix, invert_essential_matrix,
    pose_from_essential_matrix,
};
use colmap_rust::geometry::pose::euler_angles_to_rotation_matrix;
use colmap_rust::geometry::{CamRayWithJac, Rigid3d};
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Matrix3x2d, Quaterniond, Vector2d, Vector3d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};

use super::{eigen_matrix_near, eigen_matrix_near_default, rigid3d_near};

#[test]
fn decompose_essential_matrix_nominal() {
    set_prng_seed(0);
    let cam2_from_cam1 = Rigid3d::new(
        random_eigen_quaterniond(),
        Vector3d::new(0.5, 1.0, 1.0).normalized(),
    );
    let cam2_from_cam1_rot_mat = cam2_from_cam1.rotation.to_rotation_matrix();
    let e = essential_matrix_from_pose(&cam2_from_cam1);

    let (r1, r2, t) = decompose_essential_matrix(&e);

    assert!(
        (r1 - cam2_from_cam1_rot_mat).norm() < 1e-10
            || (r2 - cam2_from_cam1_rot_mat).norm() < 1e-10
    );
    assert!(
        (t - cam2_from_cam1.translation).norm() < 1e-10
            || (t + cam2_from_cam1.translation).norm() < 1e-10
    );
}

#[test]
fn essential_matrix_from_pose_nominal() {
    let expected = Matrix3d::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    assert_eq!(
        essential_matrix_from_pose(&Rigid3d::new(
            Quaterniond::identity(),
            Vector3d::new(0.0, 0.0, 1.0)
        )),
        expected
    );
    assert_eq!(
        essential_matrix_from_pose(&Rigid3d::new(
            Quaterniond::identity(),
            Vector3d::new(0.0, 0.0, 2.0)
        )),
        expected
    );
}

fn square_points() -> [Vector3d; 4] {
    [
        Vector3d::new(0.0, 0.0, 1.0),
        Vector3d::new(0.0, 0.1, 1.0),
        Vector3d::new(0.1, 0.0, 1.0),
        Vector3d::new(0.1, 0.1, 1.0),
    ]
}

#[test]
fn pose_from_essential_matrix_nominal() {
    let cam1_from_world = Rigid3d::identity();
    let cam2_from_world = Rigid3d::new(
        Quaterniond::identity(),
        Vector3d::new(1.0, 0.0, 0.0).normalized(),
    );
    let cam2_from_cam1 = cam2_from_world * cam1_from_world.inverse();
    let e = essential_matrix_from_pose(&cam2_from_cam1);

    let points3d = square_points();
    let rays1: Vec<Vector3d> = points3d
        .iter()
        .map(|&p| (cam1_from_world * p).normalized())
        .collect();
    let rays2: Vec<Vector3d> = points3d
        .iter()
        .map(|&p| (cam2_from_world * p).normalized())
        .collect();

    let (cam2_from_cam1_est, valid_indices) =
        pose_from_essential_matrix(&e, &rays1, &rays2).unwrap();

    assert_eq!(valid_indices.len(), 4);
    assert!(rigid3d_near(
        &cam2_from_cam1_est,
        &cam2_from_cam1,
        1e-12,
        1e-12
    ));
}

#[test]
fn find_optimal_image_observations_nominal() {
    let cam1_from_world = Rigid3d::identity();
    let cam2_from_world = Rigid3d::new(
        Quaterniond::identity(),
        Vector3d::new(1.0, 0.0, 0.0).normalized(),
    );
    let e = essential_matrix_from_pose(&(cam2_from_world * cam1_from_world.inverse()));

    // Test if perfect projection is equivalent to optimal image observations.
    for p in square_points() {
        let point1 = (cam1_from_world * p).hnormalized();
        let point2 = (cam2_from_world * p).hnormalized();
        let (optimal_point1, optimal_point2) = find_optimal_image_observations(&e, point1, point2);
        assert!(eigen_matrix_near_default(&point1, &optimal_point1));
        assert!(eigen_matrix_near_default(&point2, &optimal_point2));
    }
}

#[test]
fn epipole_from_essential_matrix_nominal() {
    let cam2_from_cam1 = Rigid3d::new(
        Quaterniond::identity(),
        Vector3d::new(0.0, 0.0, -1.0).normalized(),
    );
    let e = essential_matrix_from_pose(&cam2_from_cam1);

    let left_epipole = epipole_from_essential_matrix(&e, true);
    let right_epipole = epipole_from_essential_matrix(&e, false);
    assert!(eigen_matrix_near_default(
        &left_epipole,
        &Vector3d::new(0.0, 0.0, 1.0)
    ));
    assert!(eigen_matrix_near_default(
        &right_epipole,
        &Vector3d::new(0.0, 0.0, 1.0)
    ));
}

#[test]
fn invert_essential_matrix_nominal() {
    for i in 1..10 {
        let cam2_from_cam1 = Rigid3d::new(
            Quaterniond::from_rotation_matrix(euler_angles_to_rotation_matrix(0.0, 0.1, 0.0)),
            Vector3d::new(0.0, 0.0, f64::from(i)).normalized(),
        );
        let e = essential_matrix_from_pose(&cam2_from_cam1);
        let inv_inv_e = invert_essential_matrix(&invert_essential_matrix(&e));
        assert!(eigen_matrix_near_default(&e, &inv_inv_e));
    }
}

fn k1() -> Matrix3d {
    Matrix3d::new(2.0, 0.0, 1.0, 0.0, 3.0, 2.0, 0.0, 0.0, 1.0)
}

fn k2() -> Matrix3d {
    Matrix3d::new(3.0, 0.0, 2.0, 0.0, 4.0, 1.0, 0.0, 0.0, 1.0)
}

#[test]
fn fundamental_from_essential_matrix_nominal() {
    set_prng_seed(0);
    let e = essential_matrix_from_pose(&Rigid3d::new(
        random_eigen_quaterniond(),
        random_eigen_vector3d(),
    ));
    let (k1, k2) = (k1(), k2());
    let f = fundamental_from_essential_matrix(&k2, &e, &k1);
    let x = Vector3d::new(3.0, 2.0, 1.0);
    assert!(eigen_matrix_near_default(
        &(k2.transpose().inverse() * e * x),
        &(f * k1 * x)
    ));
    assert!(eigen_matrix_near_default(
        &(e * k1.inverse() * x),
        &(k2.transpose() * f * x)
    ));
}

#[test]
fn essential_from_fundamental_matrix_nominal() {
    set_prng_seed(0);
    let e = essential_matrix_from_pose(&Rigid3d::new(
        random_eigen_quaterniond(),
        random_eigen_vector3d(),
    ));
    let (k1, k2) = (k1(), k2());
    let f = fundamental_from_essential_matrix(&k2, &e, &k1);
    assert!(eigen_matrix_near(
        &essential_from_fundamental_matrix(&k2, &f, &k1),
        &e,
        1e-6
    ));
}

#[test]
fn compute_squared_sampson_error_nominal() {
    let points1 = [Vector2d::new(0.0, 0.0); 3];
    let points2 = [
        Vector2d::new(2.0, 0.0),
        Vector2d::new(2.0, 1.0),
        Vector2d::new(2.0, 2.0),
    ];

    let e = essential_matrix_from_pose(&Rigid3d::new(
        Quaterniond::identity(),
        Vector3d::new(1.0, 0.0, 0.0),
    ));

    let residuals = compute_squared_sampson_errors(&points1, &points2, &e).unwrap();

    assert_eq!(residuals.len(), 3);
    assert_eq!(residuals[0], 0.0);
    assert_eq!(residuals[1], 0.5);
    assert_eq!(residuals[2], 2.0);
}

// A pinhole geometry to exercise the tangent Sampson error against a known
// closed-form answer.
const FOCAL: f64 = 650.0;
const PRINCIPAL_X: f64 = 512.0;
const PRINCIPAL_Y: f64 = 384.0;

fn pinhole_img_from_cam(cam_point: Vector3d) -> Vector2d {
    Vector2d::new(
        FOCAL * cam_point.x / cam_point.z + PRINCIPAL_X,
        FOCAL * cam_point.y / cam_point.z + PRINCIPAL_Y,
    )
}

// Normalized image plane representative (u, v, 1) of a pixel.
fn pinhole_normalized_from_img(image_point: Vector2d) -> Vector3d {
    Vector3d::new(
        (image_point.x - PRINCIPAL_X) / FOCAL,
        (image_point.y - PRINCIPAL_Y) / FOCAL,
        1.0,
    )
}

// d(u, v, 1) / d(x, y) for a pinhole: constant and diagonal.
fn pinhole_normalized_jacobian() -> Matrix3x2d {
    let mut j = Matrix3x2d::zeros();
    j[(0, 0)] = 1.0 / FOCAL;
    j[(1, 1)] = 1.0 / FOCAL;
    j
}

// d(unit ray) / d(x, y) for a pinhole, via the normalization quotient rule.
fn pinhole_unit_ray_jacobian(normalized: Vector3d) -> Matrix3x2d {
    let norm = normalized.norm();
    let outer = Matrix3d::from_columns(
        normalized * normalized.x,
        normalized * normalized.y,
        normalized * normalized.z,
    );
    let dnormalize = (Matrix3d::identity() - outer / (norm * norm)) / norm;
    dnormalize * pinhole_normalized_jacobian()
}

fn tangent_test_pose() -> Rigid3d {
    Rigid3d::new(
        Quaterniond::from_angle_axis(AngleAxisd::new(
            0.15,
            Vector3d::new(0.3, 1.0, 0.2).normalized(),
        )),
        Vector3d::new(1.0, 0.2, 0.1).normalized(),
    )
}

fn ray(ray: Vector3d, jacobian: Matrix3x2d) -> CamRayWithJac {
    CamRayWithJac { ray, jacobian }
}

// With the normalized image plane representative (u, v, 1), whose Jacobian
// w.r.t. pixels is the constant 1/f, the tangent Sampson error is *exactly*
// f^2 times the classical Sampson error. This is an algebraic identity, so it
// pins down the whole formula - numerator, both gradient chains, and the
// denominator - to machine precision.
#[test]
fn compute_squared_tangent_sampson_error_pinhole_matches_scaled_sampson_exactly() {
    let e = essential_matrix_from_pose(&tangent_test_pose());
    let j_norm = pinhole_normalized_jacobian();

    // Deliberately includes badly mismatched pairs: the identity is exact for
    // arbitrary inputs, not only for near-inliers.
    for x1 in [20.0, 512.0, 1000.0] {
        for y1 in [30.0, 384.0, 740.0] {
            for x2 in [45.0, 512.0, 980.0] {
                for y2 in [60.0, 384.0, 700.0] {
                    let m1 = pinhole_normalized_from_img(Vector2d::new(x1, y1));
                    let m2 = pinhole_normalized_from_img(Vector2d::new(x2, y2));

                    let tangent_sampson = compute_squared_tangent_sampson_error(
                        &ray(m1, j_norm),
                        &ray(m2, j_norm),
                        &e,
                    );
                    let scaled_sampson = FOCAL * FOCAL * compute_squared_sampson_error(m1, m2, &e);

                    assert!(scaled_sampson > 0.0);
                    assert!((tangent_sampson - scaled_sampson).abs() / scaled_sampson <= 1e-14);
                }
            }
        }
    }
}

// With unit bearing vectors the agreement is only first order: rescaling the
// homogeneous representative by a function of the measurements changes the
// Sampson approximation by a term proportional to the residual itself. The
// relative discrepancy is therefore expected to shrink linearly as the
// correspondence approaches the epipolar variety.
#[test]
fn compute_squared_tangent_sampson_error_unit_rays_agree_to_first_order() {
    let cam2_from_cam1 = tangent_test_pose();
    let e = essential_matrix_from_pose(&cam2_from_cam1);

    let point3d_in_cam1 = Vector3d::new(0.35, -0.2, 4.0);
    let image_point1 = pinhole_img_from_cam(point3d_in_cam1);
    let image_point2 = pinhole_img_from_cam(cam2_from_cam1 * point3d_in_cam1);

    // Displace image 2 perpendicular to the epipolar line of image_point1, so
    // the offset is entirely "epipolar error" rather than a slide along the
    // line. In normalized coordinates the line is l = E * m1 and the pixel-space
    // gradient of the constraint is proportional to (l.x, l.y).
    let m1 = pinhole_normalized_from_img(image_point1);
    let epipolar_line2 = e * m1;
    let perpendicular = epipolar_line2.head2().normalized();

    let mut prev_rel_diff = f64::MAX;
    let mut prev_ratio = 0.0;
    for offset in [1.0, 0.1, 0.01, 0.001] {
        let m2 = pinhole_normalized_from_img(image_point2 + offset * perpendicular);

        let tangent_sampson = compute_squared_tangent_sampson_error(
            &ray(m1.normalized(), pinhole_unit_ray_jacobian(m1)),
            &ray(m2.normalized(), pinhole_unit_ray_jacobian(m2)),
            &e,
        );
        let scaled_sampson = FOCAL * FOCAL * compute_squared_sampson_error(m1, m2, &e);

        // The residual is a squared distance, so it must scale quadratically with
        // the displacement. Note sqrt(residual) is strictly below `offset`: Sampson
        // measures distance to the epipolar variety in the joint 4-D measurement
        // space, which distributes the correction over both images.
        assert!(tangent_sampson.sqrt() < offset);
        let ratio = tangent_sampson / (offset * offset);
        if prev_ratio > 0.0 {
            assert!((ratio - prev_ratio).abs() / prev_ratio <= 1e-2);
        }
        prev_ratio = ratio;

        let rel_diff = (tangent_sampson - scaled_sampson).abs() / scaled_sampson;
        // Each tenfold reduction of the residual must reduce the discrepancy,
        // confirming it is a first-order effect and not a constant bias.
        assert!(rel_diff < prev_rel_diff);
        prev_rel_diff = rel_diff;
    }
    // At a milli-pixel residual the two formulations are indistinguishable.
    assert!(prev_rel_diff < 1e-5);
}

#[test]
fn compute_squared_tangent_sampson_error_degenerate_denominator_returns_max() {
    let j_norm = pinhole_normalized_jacobian();
    let z = Vector3d::new(0.0, 0.0, 1.0);
    assert_eq!(
        compute_squared_tangent_sampson_error(&ray(z, j_norm), &ray(z, j_norm), &Matrix3d::zeros()),
        f64::MAX
    );

    // The CamRayWithJac::Zero() sentinel that callers substitute for an
    // unprojectable point must be rejected for any essential matrix: its zero ray
    // and Jacobian force the denominator to zero regardless of a (nonzero) E.
    let e = essential_matrix_from_pose(&Rigid3d::new(
        Quaterniond::identity(),
        Vector3d::new(1.0, 0.1, 0.2).normalized(),
    ));
    assert_eq!(
        compute_squared_tangent_sampson_error(&CamRayWithJac::zero(), &ray(z, j_norm), &e),
        f64::MAX
    );
}

#[test]
fn compute_squared_tangent_sampson_error_vector_overload_and_cheirality() {
    let cam1_from_world = Rigid3d::identity();
    let cam2_from_world = Rigid3d::new(
        Quaterniond::identity(),
        Vector3d::new(1.0, 0.0, 0.0).normalized(),
    );
    let cam2_from_cam1 = cam2_from_world * cam1_from_world.inverse();
    let e = essential_matrix_from_pose(&cam2_from_cam1);

    let points3d = square_points();
    let mut cam_rays1_with_jac = Vec::new();
    let mut cam_rays2_with_jac = Vec::new();
    for &p in &points3d {
        let cam1_point = cam1_from_world * p;
        let cam2_point = cam2_from_world * p;
        cam_rays1_with_jac.push(ray(
            cam1_point.normalized(),
            pinhole_unit_ray_jacobian(cam1_point / cam1_point.z),
        ));
        cam_rays2_with_jac.push(ray(
            cam2_point.normalized(),
            pinhole_unit_ray_jacobian(cam2_point / cam2_point.z),
        ));
    }

    let residuals =
        compute_squared_tangent_sampson_errors(&cam_rays1_with_jac, &cam_rays2_with_jac, &e)
            .unwrap();
    assert_eq!(residuals.len(), points3d.len());
    for residual in residuals {
        assert!(residual < 1e-16);
    }

    // Flipping one correspondence behind both cameras leaves the epipolar
    // constraint satisfied but must be rejected once cheirality is enforced.
    cam_rays1_with_jac[1].ray = -cam_rays1_with_jac[1].ray;
    cam_rays2_with_jac[1].ray = -cam_rays2_with_jac[1].ray;

    let plain_residuals =
        compute_squared_tangent_sampson_errors(&cam_rays1_with_jac, &cam_rays2_with_jac, &e)
            .unwrap();
    assert!(plain_residuals[1] < 1e-16);

    let cheiral_residuals = compute_squared_tangent_sampson_errors_with_cheirality(
        &cam_rays1_with_jac,
        &cam_rays2_with_jac,
        &e,
    )
    .unwrap();
    assert_eq!(cheiral_residuals.len(), points3d.len());
    assert_eq!(cheiral_residuals[1], f64::MAX);
    assert_eq!(cheiral_residuals[0], plain_residuals[0]);
    assert_eq!(cheiral_residuals[2], plain_residuals[2]);
    assert_eq!(cheiral_residuals[3], plain_residuals[3]);
}
