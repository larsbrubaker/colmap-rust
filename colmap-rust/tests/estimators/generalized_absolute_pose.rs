// Port of COLMAP's src/colmap/estimators/solvers/generalized_absolute_pose_test.cc (1:1, same
// values and tolerances), testing colmap_rust::estimators::solvers::generalized_absolute_pose.
// The TEST_P suite ParameterizedGP3PEstimatorTests.Nominal, instantiated over
// (num_cams, panoramic) pairs, becomes one test per instantiation named after the suite, the
// case and its parameters, in INSTANTIATE_TEST_SUITE_P's order. Tier C (RANSAC over the
// Tier B GP3P/P3P solvers).
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does; random_seed stays -1,
// so RANSAC draws from that thread PRNG. Arguments COLMAP draws inside one call are drawn
// here in clang's left-to-right order.

use std::f64::consts::PI;

use crate::transform_matchers::rigid3d_near;
use colmap_rust::estimators::solvers::{GP3PEstimator, GP3PObservation, GP3PResidualType};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Vector3d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::optim::{Estimator, Ransac, RansacOptions};

fn random_rigid3d() -> Rigid3d {
    let rotation = random_eigen_quaterniond();
    let translation = random_eigen_vector3d();
    Rigid3d::new(rotation, translation)
}

fn nominal(num_cams: usize, panoramic: bool) {
    set_prng_seed(0);

    // Note that we can estimate the minimal problem from only 3 points but we
    // need a 4th point to choose the correct solution. In theory, we don't need
    // RANSAC as we generate exact correspondences, but we use it in this test to
    // do the choosing of the best solution for us.
    const NUM_POINTS: usize = 4;
    const NUM_TRIALS: usize = 10;

    for _ in 0..NUM_TRIALS {
        let rig_from_world = random_rigid3d();
        let world_from_rig = rig_from_world.inverse();

        let mut cams_from_world = vec![Rigid3d::default(); num_cams];
        let mut cams_from_rig = vec![Rigid3d::default(); num_cams];
        for i in 0..num_cams {
            if panoramic {
                let cam_from_rig_rotation = random_eigen_quaterniond();
                cams_from_rig[i] = Rigid3d::new(
                    cam_from_rig_rotation,
                    cam_from_rig_rotation * Vector3d::new(1.0, 2.0, 3.0),
                );
                cams_from_world[i] = cams_from_rig[i] * rig_from_world;
            } else {
                cams_from_world[i] = random_rigid3d();
                cams_from_rig[i] = cams_from_world[i] * world_from_rig;
            }
        }

        let mut points2d = Vec::new();
        let mut points3d = Vec::new();
        let mut points3d_outlier = Vec::new();
        for i in 0..NUM_POINTS {
            let x = random_uniform_real(-0.5, 0.5);
            let y = random_uniform_real(-0.5, 0.5);
            let observation = GP3PObservation {
                cam_from_rig: cams_from_rig[i % num_cams].to_matrix(),
                ray_in_cam: Vector3d::new(x, y, 1.0).normalized(),
            };
            let point3d_in_cam = observation.ray_in_cam * random_uniform_real(0.1, 10.0);
            points2d.push(observation);
            let world_from_cam = cams_from_world[i % num_cams].inverse();
            points3d.push(world_from_cam * point3d_in_cam);
            points3d_outlier.push(
                world_from_cam
                    * (AngleAxisd::new(PI / 2.0, Vector3d::unit_x()).to_rotation_matrix()
                        * point3d_in_cam),
            );
        }

        for residual_type in [
            GP3PResidualType::CosineDistance,
            GP3PResidualType::ReprojectionError,
        ] {
            let options = RansacOptions {
                max_error: 1e-5,
                ..RansacOptions::default()
            };
            let mut ransac: Ransac<GP3PEstimator> =
                Ransac::new(&options, GP3PEstimator::new(residual_type)).unwrap();

            let report = ransac.estimate(&points2d, &points3d).unwrap();

            assert!(report.success);
            assert!(
                rigid3d_near(&report.model, &rig_from_world, 1e-6, 1e-6),
                "{:?} vs {:?}",
                report.model,
                rig_from_world
            );

            // Test residuals of inlier points.
            let mut residuals = Vec::new();
            ransac
                .estimator
                .residuals(&points2d, &points3d, &report.model, &mut residuals)
                .unwrap();
            assert_eq!(residuals.len(), points2d.len());
            for residual in &residuals {
                assert!(*residual < 1e-10);
            }

            // Test residuals of outlier points.
            let mut residuals_outlier = Vec::new();
            ransac
                .estimator
                .residuals(
                    &points2d,
                    &points3d_outlier,
                    &report.model,
                    &mut residuals_outlier,
                )
                .unwrap();
            assert_eq!(residuals_outlier.len(), points2d.len());
            for residual in &residuals_outlier {
                assert!(*residual > 1e-2);
            }
        }
    }
}

#[test]
fn gp3p_estimator_tests_parameterized_gp3p_estimator_tests_nominal_1_cam() {
    nominal(1, false);
}

#[test]
fn gp3p_estimator_tests_parameterized_gp3p_estimator_tests_nominal_2_cams() {
    nominal(2, false);
}

#[test]
fn gp3p_estimator_tests_parameterized_gp3p_estimator_tests_nominal_3_cams() {
    nominal(3, false);
}

#[test]
fn gp3p_estimator_tests_parameterized_gp3p_estimator_tests_nominal_4_cams() {
    nominal(4, false);
}

#[test]
fn gp3p_estimator_tests_parameterized_gp3p_estimator_tests_nominal_1_cam_panoramic() {
    nominal(1, true);
}

#[test]
fn gp3p_estimator_tests_parameterized_gp3p_estimator_tests_nominal_2_cams_panoramic() {
    nominal(2, true);
}
