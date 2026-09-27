// Port of COLMAP's src/colmap/geometry/rigid3_test.cc, plus the Rust-only default test.
// Random values come from COLMAP's seeded PRNG (math::random_eigen), which each test thread
// seeds with 0 like gtest_main; the tests set it explicitly for single-threaded targets.
//
// Deferred (need the 12x12 joint covariance, i.e. dynamic-size matrices):
// Rigid3d.CovarianceForRelativeRigid3d_PerfectCorrelation,
// Rigid3d.CovarianceForRelativeRigid3d, Rigid3d.CovariancePropagation_Composed_vs_Relative.

use colmap_rust::geometry::{cross_product_matrix, get_covariance_for_rigid3d_inverse, Rigid3d};
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Matrix6d, Quaterniond, Vector3d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{
    random_eigen_matrix6d, random_eigen_quaterniond, random_eigen_vector3d,
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
