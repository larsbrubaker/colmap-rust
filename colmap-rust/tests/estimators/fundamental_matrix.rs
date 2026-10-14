// Port of COLMAP's src/colmap/estimators/solvers/fundamental_matrix_test.cc (same values and
// tolerances), testing colmap_rust::estimators::solvers::fundamental_matrix. The TEST_P suite
// FundamentalMatrixEightPointEstimatorTests, instantiated over the point counts 8, 64 and
// 1024, becomes one test per instantiation named after the instantiation, the suite, the
// case and the count. Tier B (QR, SVD and a cubic).
//
// Not ported yet: the three RefineFundamentalMatrixSampson tests (IsFixedPointAtOptimum,
// PreservesRankTwo, RejectsDegenerateModels); the refiner needs the tiny manifolds and the
// Sampson cost functor of `estimators/cost_functions`.
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. Arguments COLMAP
// draws inside one expression (the comma initializer of RandomCalibrationMatrix,
// `Rigid3d(RandomEigenQuaterniond(), RandomEigenVectord<3>())`) are drawn here in clang's
// left-to-right order.

use crate::expect_near;
use colmap_rust::estimators::solvers::{
    FundamentalMatrixEightPointEstimator, FundamentalMatrixSevenPointEstimator,
};
use colmap_rust::geometry::essential_matrix::{
    essential_matrix_from_pose, fundamental_from_essential_matrix,
};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Matrix3d, Vector2d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::{
    random_eigen_quaterniond, random_eigen_vector2d, random_eigen_vector3d,
};
use colmap_rust::optim::Estimator;

fn random_calibration_matrix() -> Matrix3d {
    let fx = random_uniform_real(800.0, 1200.0);
    let cx = random_uniform_real(400.0, 600.0);
    let fy = random_uniform_real(800.0, 1200.0);
    let cy = random_uniform_real(400.0, 600.0);
    Matrix3d::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0)
}

fn random_rigid3d() -> Rigid3d {
    let rotation = random_eigen_quaterniond();
    let translation = random_eigen_vector3d();
    Rigid3d::new(rotation, translation)
}

fn random_epipolar_correspondences(
    cam2_from_cam1: &Rigid3d,
    k: &Matrix3d,
    num_points: usize,
    points1: &mut Vec<Vector2d>,
    points2: &mut Vec<Vector2d>,
) {
    for _ in 0..num_points {
        // K.topRows<2>() * RandomEigenVectord<2>().homogeneous().
        let projected = *k * random_eigen_vector2d().homogeneous();
        points1.push(Vector2d::new(projected.x, projected.y));
        let random_depth = random_uniform_real(0.2, 2.0);
        let point1 = (random_depth * k.inverse()) * points1.last().unwrap().homogeneous();
        points2.push((*k * (*cam2_from_cam1 * point1)).hnormalized());
    }
}

fn expect_at_least_one_valid_model<E: Estimator<X = Vector2d, Y = Vector2d, M = Matrix3d>>(
    estimator: &mut E,
    points1: &[Vector2d],
    points2: &[Vector2d],
    expected_f: &mut Matrix3d,
    models: &[Matrix3d],
    f_eps: f64,
    r_eps: f64,
) {
    *expected_f /= expected_f[(2, 2)];
    for model in models {
        let mut f = *model;
        f /= f[(2, 2)];
        if !f.is_approx_with(*expected_f, f_eps) {
            continue;
        }

        let mut residuals = Vec::new();
        estimator
            .residuals(points1, points2, &f, &mut residuals)
            .unwrap();
        for residual in &residuals[..points1.len()] {
            assert!(*residual < r_eps, "{residual} >= {r_eps}");
        }

        return;
    }
    panic!("No fundamental matrix is equal up to scale.");
}

fn points_from_raw(raw: &[f64]) -> Vec<Vector2d> {
    raw.chunks(2).map(|p| Vector2d::new(p[0], p[1])).collect()
}

#[test]
fn fundamental_seven_point_estimator_reference() {
    set_prng_seed(0);
    let points1_raw = [
        0.4964, 1.0577, 0.3650, -0.0919, -0.5412, 0.0159, -0.5239, 0.9467, 0.3467, 0.5301, 0.2797,
        0.0012, -0.1986, 0.0460,
    ];

    let points2_raw = [
        0.7570, 2.7340, 0.3961, 0.6981, -0.6014, 0.7110, -0.7385, 2.2712, 0.4177, 1.2132, 0.3052,
        0.4835, -0.2171, 0.5057,
    ];

    let points1 = points_from_raw(&points1_raw);
    let points2 = points_from_raw(&points2_raw);

    let mut estimator = FundamentalMatrixSevenPointEstimator;
    let mut models = Vec::new();
    estimator.estimate(&points1, &points2, &mut models).unwrap();

    assert_eq!(models.len(), 1);
    let f = models[0] / models[0][(2, 2)];

    // Reference values obtained from Matlab.
    expect_near(f[(0, 0)], 4.81441976, 1e-6);
    expect_near(f[(0, 1)], -8.16978909, 1e-6);
    expect_near(f[(0, 2)], 6.73133404, 1e-6);
    expect_near(f[(1, 0)], 5.16247992, 1e-6);
    expect_near(f[(1, 1)], 0.19325606, 1e-6);
    expect_near(f[(1, 2)], -2.87239381, 1e-6);
    expect_near(f[(2, 0)], -9.92570126, 1e-6);
    expect_near(f[(2, 1)], 3.64159554, 1e-6);
    expect_near(f[(2, 2)], 1., 1e-6);
}

#[test]
fn fundamental_seven_point_estimator_nominal() {
    set_prng_seed(0);
    const K_NUM_POINTS: usize = 7;
    for _ in 0..100 {
        let k = random_calibration_matrix();
        let cam2_from_cam1 = random_rigid3d();
        let mut expected_f =
            fundamental_from_essential_matrix(&k, &essential_matrix_from_pose(&cam2_from_cam1), &k);
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        random_epipolar_correspondences(
            &cam2_from_cam1,
            &k,
            K_NUM_POINTS,
            &mut points1,
            &mut points2,
        );

        let mut estimator = FundamentalMatrixSevenPointEstimator;
        let mut models = Vec::new();
        estimator.estimate(&points1, &points2, &mut models).unwrap();

        expect_at_least_one_valid_model(
            &mut estimator,
            &points1,
            &points2,
            &mut expected_f,
            &models,
            1e-6,
            1e-6,
        );
    }
}

#[test]
fn fundamental_matrix_eight_point_estimator_reference() {
    set_prng_seed(0);
    let points1_raw = [
        1.839035, 1.924743, 0.543582, 0.375221, 0.473240, 0.142522, 0.964910, 0.598376, 0.102388,
        0.140092, 15.994343, 9.622164, 0.285901, 0.430055, 0.091150, 0.254594,
    ];

    let points2_raw = [
        1.002114, 1.129644, 1.521742, 1.846002, 1.084332, 0.275134, 0.293328, 0.588992, 0.839509,
        0.087290, 1.779735, 1.116857, 0.878616, 0.602447, 0.642616, 1.028681,
    ];

    let points1 = points_from_raw(&points1_raw);
    let points2 = points_from_raw(&points2_raw);

    let mut estimator = FundamentalMatrixEightPointEstimator;
    let mut models = Vec::new();
    estimator.estimate(&points1, &points2, &mut models).unwrap();

    assert_eq!(models.len(), 1);
    let f = models[0] / models[0][(2, 2)];

    // Reference values obtained from Matlab.
    expect_near(f[(0, 0)], -9.85701, 1e-5);
    expect_near(f[(0, 1)], 18.97038, 1e-5);
    expect_near(f[(0, 2)], -1.55224, 1e-5);
    expect_near(f[(1, 0)], -3.24832, 1e-5);
    expect_near(f[(1, 1)], 2.04346, 1e-5);
    expect_near(f[(1, 2)], 0.977619, 1e-5);
    expect_near(f[(2, 0)], 11.22355, 1e-5);
    expect_near(f[(2, 1)], -19.43171, 1e-5);
    expect_near(f[(2, 2)], 1.0, 1e-5);
}

fn eight_point_nominal(num_points: usize) {
    set_prng_seed(0);
    for _ in 0..100 {
        let k = random_calibration_matrix();
        let cam2_from_cam1 = random_rigid3d();
        let mut expected_f =
            fundamental_from_essential_matrix(&k, &essential_matrix_from_pose(&cam2_from_cam1), &k);
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        random_epipolar_correspondences(
            &cam2_from_cam1,
            &k,
            num_points,
            &mut points1,
            &mut points2,
        );

        let mut estimator = FundamentalMatrixEightPointEstimator;
        let mut models = Vec::new();
        estimator.estimate(&points1, &points2, &mut models).unwrap();

        expect_at_least_one_valid_model(
            &mut estimator,
            &points1,
            &points2,
            &mut expected_f,
            &models,
            1e-6,
            1e-6,
        );
    }
}

fn eight_point_numerical_stability(num_points: usize) {
    set_prng_seed(0);
    const K_COORDINATE_SCALE: f64 = 1e3;
    for _ in 0..100 {
        let mut k = random_calibration_matrix();
        k[(0, 0)] *= K_COORDINATE_SCALE;
        k[(1, 1)] *= K_COORDINATE_SCALE;
        k[(0, 2)] *= K_COORDINATE_SCALE;
        k[(1, 2)] *= K_COORDINATE_SCALE;
        let cam2_from_cam1 = random_rigid3d();
        let mut expected_f =
            fundamental_from_essential_matrix(&k, &essential_matrix_from_pose(&cam2_from_cam1), &k);
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        random_epipolar_correspondences(
            &cam2_from_cam1,
            &k,
            num_points,
            &mut points1,
            &mut points2,
        );

        let mut estimator = FundamentalMatrixEightPointEstimator;
        let mut models = Vec::new();
        estimator.estimate(&points1, &points2, &mut models).unwrap();

        expect_at_least_one_valid_model(
            &mut estimator,
            &points1,
            &points2,
            &mut expected_f,
            &models,
            1e-4,
            1e-4,
        );
    }
}

fn eight_point_noise_stability(num_points: usize) {
    set_prng_seed(0);
    const K_NOISE: f64 = 1e-4;
    for _ in 0..100 {
        let k = random_calibration_matrix();
        let cam2_from_cam1 = random_rigid3d();
        let mut expected_f =
            fundamental_from_essential_matrix(&k, &essential_matrix_from_pose(&cam2_from_cam1), &k);
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        random_epipolar_correspondences(
            &cam2_from_cam1,
            &k,
            num_points,
            &mut points1,
            &mut points2,
        );
        for point in points2.iter_mut() {
            *point += random_eigen_vector2d() * K_NOISE;
        }

        let mut estimator = FundamentalMatrixEightPointEstimator;
        let mut models = Vec::new();
        estimator.estimate(&points1, &points2, &mut models).unwrap();

        expect_at_least_one_valid_model(
            &mut estimator,
            &points1,
            &points2,
            &mut expected_f,
            &models,
            1e-3,
            1e-2,
        );
    }
}

#[test]
fn fundamental_matrix_eight_point_estimator_fundamental_matrix_eight_point_estimator_tests_nominal_8(
) {
    eight_point_nominal(8);
}
#[test]
fn fundamental_matrix_eight_point_estimator_fundamental_matrix_eight_point_estimator_tests_nominal_64(
) {
    eight_point_nominal(64);
}
#[test]
fn fundamental_matrix_eight_point_estimator_fundamental_matrix_eight_point_estimator_tests_nominal_1024(
) {
    eight_point_nominal(1024);
}
#[test]
fn fundamental_matrix_eight_point_estimator_fundamental_matrix_eight_point_estimator_tests_numerical_stability_8(
) {
    eight_point_numerical_stability(8);
}
#[test]
fn fundamental_matrix_eight_point_estimator_fundamental_matrix_eight_point_estimator_tests_numerical_stability_64(
) {
    eight_point_numerical_stability(64);
}
#[test]
fn fundamental_matrix_eight_point_estimator_fundamental_matrix_eight_point_estimator_tests_numerical_stability_1024(
) {
    eight_point_numerical_stability(1024);
}
#[test]
fn fundamental_matrix_eight_point_estimator_fundamental_matrix_eight_point_estimator_tests_noise_stability_8(
) {
    eight_point_noise_stability(8);
}
#[test]
fn fundamental_matrix_eight_point_estimator_fundamental_matrix_eight_point_estimator_tests_noise_stability_64(
) {
    eight_point_noise_stability(64);
}
#[test]
fn fundamental_matrix_eight_point_estimator_fundamental_matrix_eight_point_estimator_tests_noise_stability_1024(
) {
    eight_point_noise_stability(1024);
}
