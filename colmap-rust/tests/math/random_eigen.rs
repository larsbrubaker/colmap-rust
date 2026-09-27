// Port of COLMAP's src/colmap/math/random_eigen_test.cc (1:1, same test names in
// snake_case). Tests colmap_rust::math::random_eigen. `Eigen::Vector2f` has no linalg type,
// so `RandomEigenVectorf<2>()` draws a `[f32; 2]` (same draws, same order).
//
// gtest_main reseeds the PRNG with 0 before every test; tests that draw call
// `seed_like_gtest()` first (see random.rs).

use colmap_rust::linalg::{Matrix3x4d, Quaterniond, Vector3d, Vector4d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{
    random_eigen_matrix_xf, random_eigen_matrixd, random_eigen_quaterniond, random_eigen_vector_xd,
    random_eigen_vectord, random_eigen_vectorf,
};

fn seed_like_gtest() {
    set_prng_seed(0);
}

fn in_range_f64(values: &[f64]) -> bool {
    values.iter().all(|&x| (-1.0..=1.0).contains(&x))
}

fn in_range_f32(values: &[f32]) -> bool {
    values.iter().all(|&x| (-1.0..=1.0).contains(&x))
}

#[test]
fn random_eigen_vectord_range() {
    seed_like_gtest();
    for _ in 0..1000 {
        let vector: Vector3d = random_eigen_vectord();
        assert!(in_range_f64(&[vector.x, vector.y, vector.z]));
    }
}

#[test]
fn random_eigen_vectord_deterministic() {
    set_prng_seed(42);
    let vector1: Vector4d = random_eigen_vectord();
    set_prng_seed(42);
    let vector2: Vector4d = random_eigen_vectord();
    assert_eq!(vector1, vector2);
}

#[test]
fn random_eigen_vectorf_range() {
    seed_like_gtest();
    for _ in 0..1000 {
        let vector: [f32; 2] = random_eigen_vectorf();
        assert!(in_range_f32(&vector));
    }
}

#[test]
fn random_eigen_vector_xd_dynamic() {
    seed_like_gtest();
    let vector = random_eigen_vector_xd(7);
    assert_eq!(vector.len(), 7);
    assert!(in_range_f64(vector.as_slice()));
}

#[test]
fn random_eigen_matrixd_range() {
    seed_like_gtest();
    let matrix: Matrix3x4d = random_eigen_matrixd();
    assert!(in_range_f64(matrix.as_slice()));
}

#[test]
fn random_eigen_matrix_xf_dynamic() {
    seed_like_gtest();
    let matrix = random_eigen_matrix_xf(3, 5);
    assert_eq!(matrix.rows(), 3);
    assert_eq!(matrix.cols(), 5);
    assert!(in_range_f32(matrix.as_slice()));
}

#[test]
fn random_eigen_quaterniond_unit() {
    seed_like_gtest();
    for _ in 0..1000 {
        let quat: Quaterniond = random_eigen_quaterniond();
        assert!((quat.norm() - 1.0).abs() <= 1e-9);
    }
}

#[test]
fn random_eigen_quaterniond_deterministic() {
    set_prng_seed(42);
    let quat1 = random_eigen_quaterniond();
    set_prng_seed(42);
    let quat2 = random_eigen_quaterniond();
    assert_eq!(quat1.coeffs(), quat2.coeffs());
}
