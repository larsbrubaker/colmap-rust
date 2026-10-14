// Port of COLMAP's src/colmap/estimators/solvers/generalized_relative_pose_test.cc (same
// values and tolerances), testing colmap_rust::estimators::solvers::generalized_relative_pose.
// The TEST_P suite ParameterizedGRNPEstimatorTests, instantiated over
// (num_cams1, num_cams2, panoramic1, panoramic2), becomes one test per instantiation named
// after the instantiation, the suite, the case and its parameters, in
// INSTANTIATE_TEST_SUITE_P's order. Tier C (RANSAC over the Tier B gen_relpose_6pt solver).
//
// Not ported yet: the seven GR8P instantiations of the same suite; GR8PEstimator is not in
// the library yet.
//
// The test seeds with set_prng_seed(1), as COLMAP's does; random_seed stays -1, so RANSAC
// draws from that thread PRNG. Arguments COLMAP draws inside one call or braced list are
// drawn here in clang's left-to-right order.

use crate::transform_matchers::rigid3d_near;
use colmap_rust::estimators::solvers::{GR6PEstimator, GRNPObservation};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::Vector3d;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::optim::{Estimator, Ransac, RansacOptions};
use colmap_rust::scene::camera::Camera;
use colmap_rust::sensor::models::CameraModelId;

struct GeneralizedRelativePoseProblem {
    points1: Vec<GRNPObservation>,
    points2: Vec<GRNPObservation>,
    rig2_from_rig1: Rigid3d,
}

fn random_rig_from_world() -> Rigid3d {
    let rotation = random_eigen_quaterniond();
    let translation = random_eigen_vector3d().normalized();
    Rigid3d::new(rotation, translation)
}

fn random_cams_from_rig(num_cameras: usize, panoramic: bool, offset: Vector3d) -> Vec<Rigid3d> {
    (0..num_cameras)
        .map(|_| {
            let cam_from_rig_rotation = random_eigen_quaterniond();
            let translation = if panoramic {
                cam_from_rig_rotation * offset
            } else {
                random_eigen_vector3d().normalized()
            };
            Rigid3d::new(cam_from_rig_rotation, translation)
        })
        .collect()
}

fn create_generalized_relative_pose_problem(
    num_points: usize,
    num_cameras1: usize,
    num_cameras2: usize,
    panoramic1: bool,
    panoramic2: bool,
) -> GeneralizedRelativePoseProblem {
    let rig1_from_world = random_rig_from_world();
    let rig2_from_world = random_rig_from_world();
    let rigs_from_world = [rig1_from_world, rig2_from_world];

    let rig2_from_rig1 = rigs_from_world[1] * rigs_from_world[0].inverse();

    let cams_from_rig1 =
        random_cams_from_rig(num_cameras1, panoramic1, Vector3d::new(1.0, 2.0, 3.0));
    let cams_from_rig2 =
        random_cams_from_rig(num_cameras2, panoramic2, Vector3d::new(-3.0, -2.0, -1.0));

    let points3d: Vec<Vector3d> = (0..num_points).map(|_| random_eigen_vector3d()).collect();

    let mut points1 = Vec::with_capacity(num_points);
    let mut points2 = Vec::with_capacity(num_points);
    // GR6P/GR8P::Residuals score in pixel units with the tangent Sampson error,
    // so each observation carries its ray's unprojection Jacobian. A spherical
    // camera maps every bearing to a pixel, keeping the synthetic points valid in
    // all directions (a pinhole would drop the back hemisphere).
    let camera =
        Camera::create_from_model_id(1, CameraModelId::Equirectangular, 0.0, 1000, 500).unwrap();
    for (i, &point3d) in points3d.iter().enumerate() {
        let cam_idx1 = i % num_cameras1;
        let cam_idx2 = i % num_cameras2;
        let point3d_in_cam1 = cams_from_rig1[cam_idx1] * (rigs_from_world[0] * point3d);
        let point3d_in_cam2 = cams_from_rig2[cam_idx2] * (rigs_from_world[1] * point3d);
        if point3d_in_cam1.norm() < 1e-8 || point3d_in_cam2.norm() < 1e-8 {
            continue;
        }

        let ray1_with_jac = camera
            .cam_ray_from_img_with_jac(camera.img_from_cam(point3d_in_cam1, true).unwrap())
            .unwrap();
        let ray2_with_jac = camera
            .cam_ray_from_img_with_jac(camera.img_from_cam(point3d_in_cam2, true).unwrap())
            .unwrap();

        points1.push(GRNPObservation {
            cam_from_rig: cams_from_rig1[cam_idx1],
            ray_with_jac_in_cam: ray1_with_jac,
        });
        points2.push(GRNPObservation {
            cam_from_rig: cams_from_rig2[cam_idx2],
            ray_with_jac_in_cam: ray2_with_jac,
        });
    }

    GeneralizedRelativePoseProblem {
        points1,
        points2,
        rig2_from_rig1,
    }
}

fn gr6p(num_cams1: usize, num_cams2: usize, panoramic1: bool, panoramic2: bool) {
    set_prng_seed(1);

    // Note that we can estimate the minimal problem from only 6 points but we
    // use an additional points to choose the correct solution.
    const K_NUM_POINTS: usize = GR6PEstimator::MIN_NUM_SAMPLES + 1;
    const K_NUM_TRIALS: usize = 10;

    for _ in 0..K_NUM_TRIALS {
        let problem = create_generalized_relative_pose_problem(
            K_NUM_POINTS,
            num_cams1,
            num_cams2,
            panoramic1,
            panoramic2,
        );

        let options = RansacOptions {
            max_error: 1.0, // pixels
            ..RansacOptions::default()
        };
        let mut ransac: Ransac<GR6PEstimator> = Ransac::new(&options, GR6PEstimator).unwrap();
        let report = ransac.estimate(&problem.points1, &problem.points2).unwrap();

        assert!(report.success);
        assert!(
            rigid3d_near(&report.model, &problem.rig2_from_rig1, 1e-4, 1e-4),
            "{:?} vs {:?}",
            report.model,
            problem.rig2_from_rig1
        );

        let mut residuals = Vec::new();
        GR6PEstimator
            .residuals(
                &problem.points1,
                &problem.points2,
                &report.model,
                &mut residuals,
            )
            .unwrap();
        // Residuals are squared pixels. The RANSAC inlier bound is max_error^2.
        for residual in &residuals {
            assert!(
                *residual <= options.max_error * options.max_error,
                "{residual}"
            );
        }
    }
}

// Note that only one of the cameras must be panoramic, otherwise the
// generalized relative pose problem is ill-posed, as we cannot estimate the
// scale between the rigs.

#[test]
fn grnp_estimator_tests_parameterized_grnp_estimator_tests_gr6p_1_2_cams() {
    gr6p(1, 2, false, false);
}

#[test]
fn grnp_estimator_tests_parameterized_grnp_estimator_tests_gr6p_2_1_cams() {
    gr6p(2, 1, false, false);
}

#[test]
fn grnp_estimator_tests_parameterized_grnp_estimator_tests_gr6p_2_2_cams() {
    gr6p(2, 2, false, false);
}

#[test]
fn grnp_estimator_tests_parameterized_grnp_estimator_tests_gr6p_3_3_cams() {
    gr6p(3, 3, false, false);
}

#[test]
fn grnp_estimator_tests_parameterized_grnp_estimator_tests_gr6p_4_4_cams() {
    gr6p(4, 4, false, false);
}

#[test]
fn grnp_estimator_tests_parameterized_grnp_estimator_tests_gr6p_4_4_cams_panoramic2() {
    gr6p(4, 4, false, true);
}

#[test]
fn grnp_estimator_tests_parameterized_grnp_estimator_tests_gr6p_4_4_cams_panoramic1() {
    gr6p(4, 4, true, false);
}
