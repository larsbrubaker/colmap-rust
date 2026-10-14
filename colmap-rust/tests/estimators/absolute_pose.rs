// Port of COLMAP's src/colmap/estimators/solvers/absolute_pose_test.cc (1:1, same values and
// tolerances; Suite.Name -> suite_name), testing colmap_rust::estimators::solvers::
// absolute_pose (P3PEstimator, P4PFEstimator, compute_squared_reprojection_error) and
// ::epnp (EPNPEstimator). The minimal solvers are Tier B; the RANSAC runs are Tier C.
//
// Each RANSAC test starts with set_prng_seed(0), as COLMAP's gtest_main does; random_seed
// stays -1, so RANSAC draws from that thread PRNG. The C++ float loop counters
// (`for (double qx = 0; qx < 1; qx += 0.2)`) are repeated with the same accumulating adds.

use std::sync::Arc;

use colmap_rust::estimators::solvers::{
    compute_squared_reprojection_error, EPNPEstimator, ImgFromCamFunc, P3PEstimator, P4PFEstimator,
    Point2DWithRay,
};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Matrix3x4d, Quaterniond, Vector2d, Vector3d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::optim::{Estimator, Ransac, RansacOptions};
use colmap_rust::scene::camera::Camera;
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::util::types::INVALID_CAMERA_ID;

use crate::expect_near;

fn points3d() -> Vec<Vector3d> {
    vec![
        Vector3d::new(1.0, 1.0, 1.0),
        Vector3d::new(0.0, 1.0, 1.0),
        Vector3d::new(3.0, 1.0, 4.0),
        Vector3d::new(3.0, 1.1, 4.0),
        Vector3d::new(3.0, 1.2, 4.0),
        Vector3d::new(3.0, 1.3, 4.0),
        Vector3d::new(3.0, 1.4, 4.0),
        Vector3d::new(2.0, 1.0, 7.0),
    ]
}

fn faulty(points3d: &[Vector3d]) -> Vec<Vector3d> {
    let mut points3d_faulty = points3d.to_vec();
    for point in &mut points3d_faulty {
        point.x = 20.0;
    }
    points3d_faulty
}

/// `for (double v = start; v < end; v += step)`, the C++ float loop.
fn float_range(start: f64, end: f64, step: f64) -> Vec<f64> {
    let mut values = Vec::new();
    let mut v = start;
    while v < end {
        values.push(v);
        v += step;
    }
    values
}

fn expected_cam_from_world(qx: f64, tx: f64) -> Rigid3d {
    Rigid3d::new(
        Quaterniond::new(1.0, qx, 0.0, 0.0).normalized(),
        Vector3d::new(tx, 0.0, 0.0),
    )
}

fn camera_func(model_id: CameraModelId) -> (Camera, ImgFromCamFunc) {
    let camera = Camera::create_from_model_id(INVALID_CAMERA_ID, model_id, 12.0, 34, 56).unwrap();
    let projector = camera.clone();
    let func: ImgFromCamFunc = Arc::new(move |cam_point| projector.img_from_cam(cam_point, true));
    (camera, func)
}

/// The rays and pixels of `points3d` seen from `cam_from_world`.
fn project_with_rays(
    cam_from_world: &Rigid3d,
    points3d: &[Vector3d],
    img_from_cam_func: &ImgFromCamFunc,
) -> Vec<Point2DWithRay> {
    points3d
        .iter()
        .map(|point3d| {
            let camera_ray = (*cam_from_world * *point3d).normalized();
            let image_point = img_from_cam_func(camera_ray).unwrap();
            Point2DWithRay::new(image_point, camera_ray)
        })
        .collect()
}

/// The body of AbsolutePose.P3P and AbsolutePose.EPNP, which differ in the estimator, the
/// RANSAC max error and the model tolerance.
fn check_ray_estimator<E>(make: impl Fn(ImgFromCamFunc) -> E, max_error: f64, model_tol: f64)
where
    E: Estimator<X = Point2DWithRay, Y = Vector3d, M = Matrix3x4d>,
{
    set_prng_seed(0);
    let points3d = points3d();
    let points3d_faulty = faulty(&points3d);
    let (_camera, img_from_cam_func) = camera_func(CameraModelId::Pinhole);

    for qx in float_range(0.0, 1.0, 0.2) {
        for tx in float_range(0.0, 1.0, 0.1) {
            let expected = expected_cam_from_world(qx, tx);

            // Project points to camera coordinate system.
            let points2d = project_with_rays(&expected, &points3d, &img_from_cam_func);

            let options = RansacOptions {
                max_error,
                ..RansacOptions::default()
            };
            let mut ransac: Ransac<E> =
                Ransac::new(&options, make(img_from_cam_func.clone())).unwrap();
            let report = ransac.estimate(&points2d, &points3d).unwrap();

            assert!(report.success);
            assert!((expected.to_matrix() - report.model).norm() < model_tol);

            // Test residuals of exact points.
            let mut residuals = Vec::new();
            ransac
                .estimator
                .residuals(&points2d, &points3d, &report.model, &mut residuals)
                .unwrap();
            for residual in &residuals {
                assert!(*residual < 1e-3);
            }

            // Test residuals of faulty points.
            ransac
                .estimator
                .residuals(&points2d, &points3d_faulty, &report.model, &mut residuals)
                .unwrap();
            for residual in &residuals {
                assert!(*residual > 0.1);
            }
        }
    }
}

#[test]
fn absolute_pose_p3p() {
    check_ray_estimator(P3PEstimator::new, 1e-3, 1e-5);
}

/// The body of AbsolutePose.P4PFSharedFocalLength and P4PFSeparateFocalLengths: the focal
/// lengths are `(f, aspect * f)`.
fn check_p4pf(share_focal_length: bool, aspect: f64) {
    set_prng_seed(0);
    let points3d = points3d();
    let points3d_faulty = faulty(&points3d);

    for qx in float_range(0.0, 1.0, 0.2) {
        for tx in float_range(0.0, 1.0, 0.1) {
            for f in float_range(0.5, 20.0, 2.0) {
                let focal_lengths = Vector2d::new(f, aspect * f);
                let expected = expected_cam_from_world(qx, tx);

                // Project points to camera coordinate system.
                let points2d: Vec<Vector2d> = points3d
                    .iter()
                    .map(|point3d| {
                        let projected = (expected * *point3d).hnormalized();
                        if share_focal_length {
                            f * projected
                        } else {
                            focal_lengths.cwise_product(projected)
                        }
                    })
                    .collect();

                let options = RansacOptions {
                    max_error: 1e-5,
                    ..RansacOptions::default()
                };
                let mut ransac: Ransac<P4PFEstimator> =
                    Ransac::new(&options, P4PFEstimator::new(share_focal_length)).unwrap();
                let report = ransac.estimate(&points2d, &points3d).unwrap();

                assert!(report.success);
                if share_focal_length {
                    // In shared mode both focal lengths must be identical.
                    assert_eq!(report.model.focal_lengths.x, report.model.focal_lengths.y);
                    expect_near(report.model.focal_lengths.x, f, 1e-3);
                } else {
                    expect_near(report.model.focal_lengths.x, focal_lengths.x, 1e-3);
                    expect_near(report.model.focal_lengths.y, focal_lengths.y, 1e-3);
                }
                assert!((expected.to_matrix() - report.model.cam_from_world).norm() < 1e-3);

                // Test residuals of exact points.
                let mut estimator = P4PFEstimator::default();
                let mut residuals = Vec::new();
                estimator
                    .residuals(&points2d, &points3d, &report.model, &mut residuals)
                    .unwrap();
                for residual in &residuals {
                    assert!(*residual < 1e-3);
                }

                // Test residuals of faulty points.
                estimator
                    .residuals(&points2d, &points3d_faulty, &report.model, &mut residuals)
                    .unwrap();
                for residual in &residuals {
                    assert!(*residual > 0.1);
                }
            }
        }
    }
}

#[test]
fn absolute_pose_p4pf_shared_focal_length() {
    check_p4pf(true, 1.0);
}

#[test]
fn absolute_pose_p4pf_separate_focal_lengths() {
    check_p4pf(false, 1.5);
}

#[test]
fn absolute_pose_epnp() {
    check_ray_estimator(EPNPEstimator::new, 1e-5, 1e-3);
}

// COLMAP's 17-digit literals are kept digit for digit.
#[allow(clippy::excessive_precision)]
#[test]
fn absolute_pose_epnp_broken_solve_sign_case() {
    let image_points = [
        Vector2d::new(-2.6783007931074532e-01, 5.3457197430746251e-01),
        Vector2d::new(-4.2629907287470264e-01, 7.5623350319519789e-01),
        Vector2d::new(-1.6767413005963930e-01, -1.3387172544910089e-01),
        Vector2d::new(-5.6616329720373559e-02, 2.3621156497739373e-01),
        Vector2d::new(-1.7721225948969935e-01, 2.3395366792735982e-02),
        Vector2d::new(-5.1836259886632222e-02, -4.4380694271927049e-02),
        Vector2d::new(-3.5897765845560037e-01, 1.6252721078589397e-01),
        Vector2d::new(2.7057324473684058e-01, -1.4067450104631887e-01),
        Vector2d::new(-2.5811166424334520e-01, 8.0167171300227366e-02),
        Vector2d::new(2.0239567448222310e-02, -3.2845953375344145e-01),
        Vector2d::new(4.2571014715170657e-01, -2.8321173570154773e-01),
        Vector2d::new(-5.4597596412987237e-01, 9.1431935871671977e-02),
    ];

    let points2d: Vec<Point2DWithRay> = image_points
        .iter()
        .map(|&image_point| {
            Point2DWithRay::new(image_point, image_point.homogeneous().normalized())
        })
        .collect();

    let points3d = vec![
        Vector3d::new(
            4.4276865308679305e+00,
            -1.3384364366019632e+00,
            -3.5997423085253892e+00,
        ),
        Vector3d::new(
            2.7278555252512309e+00,
            -3.8152996187231392e-01,
            -2.6558518399902824e+00,
        ),
        Vector3d::new(
            4.8548566083054894e+00,
            -1.4756197433631739e+00,
            -6.8274946022490501e-01,
        ),
        Vector3d::new(
            3.1523013527998449e+00,
            -1.3377020437938025e+00,
            -1.6443269301929087e+00,
        ),
        Vector3d::new(
            3.8551679771512073e+00,
            -1.0557700545885551e+00,
            -1.1695994508851486e+00,
        ),
        Vector3d::new(
            5.9571373150353812e+00,
            -2.6120646101684555e+00,
            -1.0841441206050342e+00,
        ),
        Vector3d::new(
            6.3287088499358894e+00,
            -1.1761274755817175e+00,
            -2.5951879774151583e+00,
        ),
        Vector3d::new(
            2.3005305990121250e+00,
            -1.4019796626800123e+00,
            -4.4485464455072321e-01,
        ),
        Vector3d::new(
            5.9816859934587354e+00,
            -1.4211814511691452e+00,
            -2.0285923889293449e+00,
        ),
        Vector3d::new(
            5.2543344690665457e+00,
            -2.3389255564264144e+00,
            4.3708173185524052e-01,
        ),
        Vector3d::new(
            3.2181599245991688e+00,
            -2.8906671988445098e+00,
            2.6825718150064348e-01,
        ),
        Vector3d::new(
            4.4592895306946758e+00,
            -9.1235241641579902e-03,
            -1.6555237117970871e+00,
        ),
    ];

    let mut models = Vec::new();
    let mut estimator = EPNPEstimator::new(Arc::new(|point3d_in_cam: Vector3d| {
        Some(point3d_in_cam.hnormalized())
    }));
    estimator
        .estimate(&points2d, &points3d, &mut models)
        .unwrap();

    assert_eq!(models.len(), 1);

    let mut reproj = 0.0;
    for (point3d, point2d) in points3d.iter().zip(&points2d) {
        reproj += ((models[0] * point3d.homogeneous()).hnormalized() - point2d.image_point).norm();
    }

    assert!(reproj < 0.2);
}

#[test]
fn compute_squared_reprojection_error_nominal() {
    let (camera, img_from_cam_func) = camera_func(CameraModelId::SimplePinhole);

    let points3d = vec![
        Vector3d::new(-1.0, 0.0, 1.0),
        Vector3d::new(-1.0, 1.0, 1.0),
        Vector3d::new(0.0, 0.0, -1.0),
        Vector3d::new(0.0, 0.0, 0.0),
    ];

    let principal_point = Vector2d::new(
        camera.principal_point_x().unwrap(),
        camera.principal_point_y().unwrap(),
    );
    let points2d = vec![
        Point2DWithRay::new(principal_point, Vector3d::zeros()),
        Point2DWithRay::new(principal_point, Vector3d::zeros()),
        Point2DWithRay::new(Vector2d::zeros(), Vector3d::zeros()),
        Point2DWithRay::new(Vector2d::zeros(), Vector3d::zeros()),
    ];

    let cam_from_world = Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0));

    let mut residuals = Vec::new();
    compute_squared_reprojection_error(
        &points2d,
        &points3d,
        &cam_from_world.to_matrix(),
        &img_from_cam_func,
        &mut residuals,
    )
    .unwrap();

    let focal_length = camera.focal_length().unwrap();
    assert_eq!(
        residuals,
        vec![0.0, focal_length * focal_length, f64::MAX, f64::MAX]
    );
}
