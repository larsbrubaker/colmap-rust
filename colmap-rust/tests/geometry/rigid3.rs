// Port of COLMAP's src/colmap/geometry/rigid3_test.cc, plus the Rust-only default test.
// Random values come from COLMAP's seeded PRNG (math::random_eigen), which each test thread
// seeds with 0 like gtest_main; the tests set it explicitly for single-threaded targets.
// COLMAP's fixed 12x12 and 6x12 matrices are `MatrixXd` here.

use colmap_rust::geometry::{
    cross_product_matrix, get_covariance_for_composed_rigid3d, get_covariance_for_relative_rigid3d,
    get_covariance_for_rigid3d_inverse, Rigid3d,
};
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Matrix6d, MatrixXd, Quaterniond, Vector3d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{
    random_eigen_matrix6d, random_eigen_matrix_xd, random_eigen_quaterniond, random_eigen_vector3d,
};

use super::eigen_matrix_near;

fn test_rigid3d() -> Rigid3d {
    Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d())
}

#[test]
fn cross_product_matrix_nominal() {
    assert_eq!(
        cross_product_matrix(Vector3d::new(0.0, 0.0, 0.0)),
        Matrix3d::zeros()
    );
    let ref_matrix = Matrix3d::new(0.0, -3.0, 2.0, 3.0, 0.0, -1.0, -2.0, 1.0, 0.0);
    assert_eq!(
        cross_product_matrix(Vector3d::new(1.0, 2.0, 3.0)),
        ref_matrix
    );
}

#[test]
fn rigid3d_default() {
    let tform = Rigid3d::default();
    assert_eq!(tform.rotation.coeffs(), Quaterniond::identity().coeffs());
    assert_eq!(tform.translation, Vector3d::zeros());
}

#[test]
fn rigid3d_equals() {
    let mut tform = Rigid3d::default();
    let mut other = tform;
    assert_eq!(tform, other);
    tform.translation.x = 1.0;
    assert_ne!(tform, other);
    other.translation.x = 1.0;
    assert_eq!(tform, other);
}

#[test]
fn rigid3d_print() {
    let tform = Rigid3d::default();
    assert_eq!(
        tform.to_string(),
        "Rigid3d(rotation_xyzw=[0, 0, 0, 1], translation=[0, 0, 0])"
    );
}

#[test]
fn rigid3d_inverse() {
    set_prng_seed(0);
    let b_from_a = test_rigid3d();
    let a_from_b = b_from_a.inverse();
    for _ in 0..100 {
        let x_in_a = random_eigen_vector3d();
        let x_in_b = b_from_a * x_in_a;
        assert!(eigen_matrix_near(&(a_from_b * x_in_b), &x_in_a, 1e-6));
    }
}

#[test]
fn rigid3d_tgt_origin_in_src() {
    set_prng_seed(0);
    let b_from_a = test_rigid3d();
    let origin_b_in_a = b_from_a.tgt_origin_in_src();
    assert!((b_from_a * origin_b_in_a - Vector3d::zeros()).norm() < 1e-6);
}

#[test]
fn rigid3d_to_matrix() {
    set_prng_seed(0);
    let b_from_a = test_rigid3d();
    let b_from_a_mat = b_from_a.to_matrix();
    for _ in 0..100 {
        let x_in_a = random_eigen_vector3d();
        assert!((b_from_a * x_in_a - b_from_a_mat * x_in_a.homogeneous()).norm() < 1e-6);
    }
}

#[test]
fn rigid3d_from_matrix() {
    set_prng_seed(0);
    let b1_from_a = test_rigid3d();
    let b2_from_a = Rigid3d::from_matrix(&b1_from_a.to_matrix());
    for _ in 0..100 {
        let x_in_a = random_eigen_vector3d();
        assert!(eigen_matrix_near(
            &(b1_from_a * x_in_a),
            &(b2_from_a * x_in_a),
            1e-6
        ));
    }
}

#[test]
fn rigid3d_apply_no_rotation() {
    let b_from_a = Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 2.0, 3.0));
    assert!((b_from_a * Vector3d::new(1.0, 2.0, 3.0) - Vector3d::new(2.0, 4.0, 6.0)).norm() < 1e-6);
}

fn quarter_turn_x() -> Quaterniond {
    Quaterniond::from_angle_axis(AngleAxisd::new(
        std::f64::consts::PI / 2.0,
        Vector3d::unit_x(),
    ))
}

#[test]
fn rigid3d_apply_no_translation() {
    let b_from_a = Rigid3d::new(quarter_turn_x(), Vector3d::zeros());
    assert!(
        (b_from_a * Vector3d::new(1.0, 2.0, 3.0) - Vector3d::new(1.0, -3.0, 2.0)).norm() < 1e-6
    );
}

#[test]
fn rigid3d_apply_rotation_translation() {
    let b_from_a = Rigid3d::new(quarter_turn_x(), Vector3d::new(1.0, 2.0, 3.0));
    assert!(
        (b_from_a * Vector3d::new(1.0, 2.0, 3.0) - Vector3d::new(2.0, -1.0, 5.0)).norm() < 1e-6
    );
}

#[test]
fn rigid3d_apply_chain() {
    set_prng_seed(0);
    let b_from_a = test_rigid3d();
    let c_from_b = test_rigid3d();
    let d_from_c = test_rigid3d();
    let x_in_a = random_eigen_vector3d();
    let x_in_b = b_from_a * x_in_a;
    let x_in_c = c_from_b * x_in_b;
    let x_in_d = d_from_c * x_in_c;
    assert_eq!(d_from_c * (c_from_b * (b_from_a * x_in_a)), x_in_d);
}

#[test]
fn rigid3d_compose() {
    set_prng_seed(0);
    let b_from_a = test_rigid3d();
    let c_from_b = test_rigid3d();
    let d_from_c = test_rigid3d();
    let d_from_a = d_from_c * c_from_b * b_from_a;
    let x_in_a = random_eigen_vector3d();
    let x_in_b = b_from_a * x_in_a;
    let x_in_c = c_from_b * x_in_b;
    let x_in_d = d_from_c * x_in_c;
    assert!(eigen_matrix_near(&(d_from_a * x_in_a), &x_in_d, 1e-6));
}

#[test]
fn rigid3d_adjoint() {
    set_prng_seed(0);
    let b_from_a = test_rigid3d();
    let adjoint = b_from_a.adjoint();
    let adjoint_inv = b_from_a.adjoint_inverse();
    assert!(eigen_matrix_near(
        &(adjoint * adjoint_inv),
        &Matrix6d::identity(),
        1e-6
    ));
    let a_from_b = b_from_a.inverse();
    let adjoint_a_from_b = a_from_b.adjoint();
    assert!(eigen_matrix_near(&adjoint_inv, &adjoint_a_from_b, 1e-6));
}

#[test]
fn rigid3d_covariance_for_inverse() {
    set_prng_seed(0);
    let b_from_a = test_rigid3d();
    let a = random_eigen_matrix6d();
    let cov_b_from_a = a * a.transpose();
    let cov_a_from_b = get_covariance_for_rigid3d_inverse(&b_from_a, &cov_b_from_a);
    let a_from_b = b_from_a.inverse();
    let cov_b_from_a_test = get_covariance_for_rigid3d_inverse(&a_from_b, &cov_a_from_b);
    assert!(eigen_matrix_near(&cov_b_from_a_test, &cov_b_from_a, 1e-6));
}

fn x(m: Matrix6d) -> MatrixXd {
    MatrixXd::from(m)
}

// A * B * A^T with dynamic-size products, Eigen's `J * covar * J.transpose()`.
fn sandwich(j: &MatrixXd, covar: &MatrixXd) -> MatrixXd {
    &(j * covar) * &j.transpose()
}

#[test]
fn rigid3d_covariance_for_relative_rigid3d_perfect_correlation() {
    set_prng_seed(0);
    let world_from_a = test_rigid3d();
    let world_from_b = test_rigid3d();
    let a = random_eigen_matrix6d();
    let covar_subblock = x(a * a.transpose());
    // Two poses are perfectly correlated in world frame
    let mut covar_world_from_cam = MatrixXd::zeros(12, 12);
    covar_world_from_cam.set_block(0, 0, &covar_subblock);
    covar_world_from_cam.set_block(0, 6, &covar_subblock);
    covar_world_from_cam.set_block(6, 0, &covar_subblock);
    covar_world_from_cam.set_block(6, 6, &covar_subblock);
    // Invert poses
    let a_from_world = world_from_a.inverse();
    let b_from_world = world_from_b.inverse();
    let mut j0 = MatrixXd::zeros(12, 12);
    j0.set_block(0, 0, &x(-world_from_a.adjoint_inverse()));
    j0.set_block(6, 6, &x(-world_from_b.adjoint_inverse()));
    let covar_cam_from_world = sandwich(&j0, &covar_world_from_cam);
    // Calculate relative pose covariance, which should be a zero matrix.
    let b_cov_from_a =
        get_covariance_for_relative_rigid3d(&a_from_world, &b_from_world, &covar_cam_from_world);
    assert!(b_cov_from_a.norm() < 1e-6);
}

#[test]
fn rigid3d_covariance_for_relative_rigid3d() {
    set_prng_seed(0);
    let a_from_world = test_rigid3d();
    let b_from_world = test_rigid3d();
    let a = random_eigen_matrix_xd(12, 12);
    let covar = &a * &a.transpose();

    // Ours (in left convention)
    let b_cov_from_a = get_covariance_for_relative_rigid3d(&a_from_world, &b_from_world, &covar);

    // Use the equations from the right convention as a reference.
    // The covariance in left (right) equals to the covariance of pose inverse in
    // right (left).

    // Convert to right convention. To estimate covariance of T_2T_1^{-1} in left,
    // We can equivalently estimate covariance of T_1T_2^{-1} in right.
    let mut j0 = MatrixXd::zeros(12, 12);
    // the covariance of T_1^{-1} in left corresponds to the covariance of T_1 in
    // right
    j0.set_block(0, 0, &x(-a_from_world.adjoint_inverse()));
    // the covariance of T_2 in left corresponds to the covariance of T_2^{-1} in
    // right
    j0.set_block(6, 6, &MatrixXd::identity(6));
    // Get the covariance of (T_1, T_2^{-1}) in right
    let covar_in_right = sandwich(&j0, &covar);

    // Compose T_1T_2^{-1} in right
    // [Reference] Joan Solà, Jeremie Deray, Dinesh Atchuthan, A micro Lie theory
    // for state estimation in robotics, 2018.
    // Eqs. (177) and (178)
    let mut j_in_right = MatrixXd::zeros(6, 12);
    j_in_right.set_block(0, 0, &x(b_from_world.adjoint()));
    j_in_right.set_block(0, 6, &MatrixXd::identity(6));
    let a_cov_from_b_right = sandwich(&j_in_right, &covar_in_right).to_matrix6d();
    assert!(eigen_matrix_near(&b_cov_from_a, &a_cov_from_b_right, 1e-6));
}

#[test]
fn rigid3d_covariance_propagation_composed_vs_relative() {
    set_prng_seed(0);
    let a_from_b = test_rigid3d();
    let b_from_c = test_rigid3d();
    let a = random_eigen_matrix_xd(12, 12);
    let covar = &a * &a.transpose();

    // Covariance for the composed rigid3d
    let a_cov_from_c_composed = get_covariance_for_composed_rigid3d(&a_from_b, &covar);

    // Invert b_from_c and switch order
    let c_from_b = b_from_c.inverse();
    let mut j0 = MatrixXd::zeros(12, 12);
    j0.set_block(6, 0, &MatrixXd::identity(6));
    j0.set_block(0, 6, &x(-b_from_c.adjoint_inverse()));
    let covar_x_from_b = sandwich(&j0, &covar);
    let a_cov_from_c_relative =
        get_covariance_for_relative_rigid3d(&c_from_b, &a_from_b, &covar_x_from_b);

    // Check consistency
    assert!(eigen_matrix_near(
        &a_cov_from_c_composed,
        &a_cov_from_c_relative,
        1e-6
    ));
}

#[test]
fn rust_only_rigid3d_default_is_identity() {
    // COLMAP's default constructor is the identity; the hand-written Default must be too,
    // never an all-zero rotation.
    let tform = Rigid3d::default();
    assert_eq!(tform, Rigid3d::identity());
    assert_eq!(tform.rotation.w, 1.0);
    let x = Vector3d::new(1.0, -2.0, 3.0);
    assert_eq!(tform * x, x);
    assert_eq!(tform.inverse(), tform);
}
