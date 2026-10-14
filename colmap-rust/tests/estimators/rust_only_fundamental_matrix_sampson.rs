// Rust-only test of colmap_rust::estimators::solvers::refine_fundamental_matrix_sampson.
// COLMAP's three RefineFundamentalMatrixSampson cases (tests/estimators/fundamental_matrix.rs)
// check a fixed point at the optimum, rank 2 and a lower error than the eight-point fit on
// noisy data; none of them starts away from the optimum, so a refiner that never moved the
// model, or moved it to the wrong place, could pass. This starts from the F of a perturbed
// pose on noise-free correspondences, whose unique Sampson minimizer is the true F, and
// requires the refinement to land on it. Tier C (an iterative solve): over the 20 scenes the
// E-space gap drops from 1e-2..3e-2 to at most 1.5e-6 and the mean squared Sampson error
// from 4e1..2e3 to at most 2.3e-7 within the refiner's 25 iterations; the bounds leave about
// an order of magnitude of margin.

use colmap_rust::estimators::solvers::refine_fundamental_matrix_sampson;
use colmap_rust::geometry::essential_matrix::{
    compute_squared_sampson_errors, essential_matrix_from_pose, fundamental_from_essential_matrix,
};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Matrix3d, Quaterniond, Vector2d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::{
    random_eigen_quaterniond, random_eigen_vector2d, random_eigen_vector3d,
};

fn random_calibration_matrix() -> Matrix3d {
    let fx = random_uniform_real(800.0, 1200.0);
    let cx = random_uniform_real(400.0, 600.0);
    let fy = random_uniform_real(800.0, 1200.0);
    let cy = random_uniform_real(400.0, 600.0);
    Matrix3d::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
}

fn fundamental(k: &Matrix3d, cam2_from_cam1: &Rigid3d) -> Matrix3d {
    fundamental_from_essential_matrix(k, &essential_matrix_from_pose(cam2_from_cam1), k)
}

// Distance between two fundamental matrices up to scale and sign, measured on the essential
// matrices K^T F K normalized to unit Frobenius norm. In pixel units the entries of F span
// several orders of magnitude, so its own Frobenius distance barely sees a pose change.
fn gap(k: &Matrix3d, f: &Matrix3d, expected: &Matrix3d) -> f64 {
    let e = k.transpose() * *f * *k;
    let expected_e = k.transpose() * *expected * *k;
    let e = e / e.norm();
    let expected_e = expected_e / expected_e.norm();
    (e - expected_e).norm().min((e + expected_e).norm())
}

fn mean_squared_sampson_error(points1: &[Vector2d], points2: &[Vector2d], f: &Matrix3d) -> f64 {
    let residuals = compute_squared_sampson_errors(points1, points2, f).unwrap();
    residuals.iter().sum::<f64>() / residuals.len() as f64
}

#[test]
fn rust_only_refine_fundamental_matrix_sampson_converges_from_perturbed_model() {
    set_prng_seed(0);
    const NUM_POINTS: usize = 64;
    for _ in 0..20 {
        let k = random_calibration_matrix();
        let rotation = random_eigen_quaterniond();
        let translation = random_eigen_vector3d();
        let cam2_from_cam1 = Rigid3d::new(rotation, translation);

        let mut points1 = Vec::with_capacity(NUM_POINTS);
        let mut points2 = Vec::with_capacity(NUM_POINTS);
        for _ in 0..NUM_POINTS {
            let point1 = k * random_eigen_vector2d().homogeneous();
            let point1 = Vector2d::new(point1.x, point1.y);
            let depth = random_uniform_real(0.2, 2.0);
            let point3d_in_cam1 = (depth * k.inverse()) * point1.homogeneous();
            points1.push(point1);
            points2.push((k * (cam2_from_cam1 * point3d_in_cam1)).hnormalized());
        }

        // A nearby pose: a rotation of about 1 degree and a 2% translation offset.
        let delta = 0.01 * random_eigen_vector3d();
        let perturbed_rotation = rotation * Quaterniond::new(1.0, delta.x, delta.y, delta.z);
        let perturbed = Rigid3d::new(
            perturbed_rotation.normalized(),
            translation + 0.02 * translation.norm() * random_eigen_vector3d(),
        );

        let expected_f = fundamental(&k, &cam2_from_cam1);
        let mut f = fundamental(&k, &perturbed);
        let initial_gap = gap(&k, &f, &expected_f);
        let initial_error = mean_squared_sampson_error(&points1, &points2, &f);
        assert!(
            initial_gap > 1e-3,
            "the perturbation must move F: {initial_gap}"
        );

        assert!(refine_fundamental_matrix_sampson(&points1, &points2, &mut f).unwrap());

        let refined_gap = gap(&k, &f, &expected_f);
        let refined_error = mean_squared_sampson_error(&points1, &points2, &f);
        assert!(
            refined_gap < 1e-5 && refined_gap < 1e-3 * initial_gap,
            "gap {initial_gap} -> {refined_gap}"
        );
        assert!(
            refined_error < 1e-6 && refined_error < 1e-8 * initial_error,
            "error {initial_error} -> {refined_error}"
        );
    }
}
