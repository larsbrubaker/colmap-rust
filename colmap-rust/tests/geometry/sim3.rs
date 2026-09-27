// Port of COLMAP's src/colmap/geometry/sim3_test.cc (all cases), plus the Rust-only default
// test. sim3_test.cc names its chain test `TEST(Rigid3d, ApplyChain)`; here it is
// `sim3d_apply_chain` so it does not collide with rigid3_test.cc's in the same binary.

use colmap_rust::geometry::Sim3d;
use colmap_rust::linalg::{AngleAxisd, Quaterniond, Vector3d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};

use super::eigen_matrix_near;

fn test_sim3d() -> Sim3d {
    let scale = random_uniform_real::<f64>(0.1, 10.0);
    let rotation = random_eigen_quaterniond();
    Sim3d::new(scale, rotation, random_eigen_vector3d())
}

fn quarter_turn_x() -> Quaterniond {
    Quaterniond::from_angle_axis(AngleAxisd::new(
        std::f64::consts::PI / 2.0,
        Vector3d::unit_x(),
    ))
}

#[test]
fn sim3d_default() {
    let tform = Sim3d::default();
    assert_eq!(tform.scale, 1.0);
    assert_eq!(tform.rotation.coeffs(), Quaterniond::identity().coeffs());
    assert_eq!(tform.translation, Vector3d::zeros());
}

#[test]
fn sim3d_equals() {
    let mut tform = Sim3d::default();
    let mut other = tform;
    assert_eq!(tform, other);
    tform.translation.x = 1.0;
    assert_ne!(tform, other);
    other.translation.x = 1.0;
    assert_eq!(tform, other);
}

#[test]
fn sim3d_print() {
    assert_eq!(
        Sim3d::default().to_string(),
        "Sim3d(scale=1, rotation_xyzw=[0, 0, 0, 1], translation=[0, 0, 0])"
    );
}

#[test]
fn sim3d_inverse() {
    set_prng_seed(0);
    let b_from_a = test_sim3d();
    let a_from_b = b_from_a.inverse();
    for _ in 0..100 {
        let x_in_a = random_eigen_vector3d();
        let x_in_b = b_from_a * x_in_a;
        assert!(eigen_matrix_near(&(a_from_b * x_in_b), &x_in_a, 1e-6));
    }
}

#[test]
fn sim3d_to_matrix() {
    set_prng_seed(0);
    let b_from_a = test_sim3d();
    let b_from_a_mat = b_from_a.to_matrix();
    for _ in 0..100 {
        let x_in_a = random_eigen_vector3d();
        assert!((b_from_a * x_in_a - b_from_a_mat * x_in_a.homogeneous()).norm() < 1e-6);
    }
}

#[test]
fn sim3d_from_matrix() {
    set_prng_seed(0);
    let b1_from_a = test_sim3d();
    let b2_from_a = Sim3d::from_matrix(&b1_from_a.to_matrix());
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
fn sim3d_apply_scale_only() {
    let b_from_a = Sim3d::new(2.0, Quaterniond::identity(), Vector3d::zeros());
    assert!((b_from_a * Vector3d::new(1.0, 2.0, 3.0) - Vector3d::new(2.0, 4.0, 6.0)).norm() < 1e-6);
}

#[test]
fn sim3d_apply_translation_only() {
    let b_from_a = Sim3d::new(1.0, Quaterniond::identity(), Vector3d::new(1.0, 2.0, 3.0));
    assert!((b_from_a * Vector3d::new(1.0, 2.0, 3.0) - Vector3d::new(2.0, 4.0, 6.0)).norm() < 1e-6);
}

#[test]
fn sim3d_apply_rotation_only() {
    let b_from_a = Sim3d::new(1.0, quarter_turn_x(), Vector3d::zeros());
    assert!(
        (b_from_a * Vector3d::new(1.0, 2.0, 3.0) - Vector3d::new(1.0, -3.0, 2.0)).norm() < 1e-6
    );
}

#[test]
fn sim3d_apply_scale_rotation_translation() {
    let b_from_a = Sim3d::new(2.0, quarter_turn_x(), Vector3d::new(1.0, 2.0, 3.0));
    assert!(
        (b_from_a * Vector3d::new(1.0, 2.0, 3.0) - Vector3d::new(3.0, -4.0, 7.0)).norm() < 1e-6
    );
}

#[test]
fn sim3d_apply_chain() {
    set_prng_seed(0);
    let b_from_a = test_sim3d();
    let c_from_b = test_sim3d();
    let d_from_c = test_sim3d();
    let x_in_a = random_eigen_vector3d();
    let x_in_b = b_from_a * x_in_a;
    let x_in_c = c_from_b * x_in_b;
    let x_in_d = d_from_c * x_in_c;
    assert_eq!(d_from_c * (c_from_b * (b_from_a * x_in_a)), x_in_d);
}

#[test]
fn sim3d_compose() {
    set_prng_seed(0);
    let b_from_a = test_sim3d();
    let c_from_b = test_sim3d();
    let d_from_c = test_sim3d();
    let d_from_a = d_from_c * c_from_b * b_from_a;
    let x_in_a = random_eigen_vector3d();
    let x_in_b = b_from_a * x_in_a;
    let x_in_c = c_from_b * x_in_b;
    let x_in_d = d_from_c * x_in_c;
    assert!(eigen_matrix_near(&(d_from_a * x_in_a), &x_in_d, 1e-6));
}

#[test]
fn sim3d_to_from_file() {
    set_prng_seed(0);
    let dir = std::env::temp_dir().join(format!(
        "colmap_rust_sim3d_to_from_file_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("file.txt");
    let written = test_sim3d();
    written.to_file(&path).unwrap();
    let read = Sim3d::from_file(&path).unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    assert_eq!(written.scale, read.scale);
    assert_eq!(written.rotation.coeffs(), read.rotation.coeffs());
    assert_eq!(written.translation, read.translation);
}

#[test]
fn rust_only_sim3d_default_is_identity() {
    // COLMAP's default constructor is the identity; the hand-written Default must be too.
    let tform = Sim3d::default();
    assert_eq!(tform, Sim3d::identity());
    let x = Vector3d::new(1.0, -2.0, 3.0);
    assert_eq!(tform * x, x);
}
