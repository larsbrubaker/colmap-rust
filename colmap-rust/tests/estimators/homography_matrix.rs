// Port of COLMAP's src/colmap/estimators/solvers/homography_matrix_test.cc (1:1, same values
// and tolerances), testing colmap_rust::estimators::solvers::homography_matrix. The TEST_P
// suites HomographyMatrixTests and HomographyMatrixRayTests, instantiated over the point
// counts 4, 8, 64 and 1024, become one test per instantiation named after the instantiation,
// the suite, the case and the count. Tier B (an LU or SVD null-space solve).
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does.

use std::f64::consts::PI;

use crate::expect_near;
use colmap_rust::estimators::solvers::{
    CamRayWithImgPoint, HomographyMatrixEstimator, HomographyMatrixRayEstimator,
};
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Vector2d, Vector3d};
use colmap_rust::math::deg_to_rad;
use colmap_rust::math::fns;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{random_eigen_vector2d, random_eigen_vector3d};
use colmap_rust::optim::Estimator;
use colmap_rust::scene::camera::Camera;
use colmap_rust::sensor::models::CameraModelId;

/// Eigen's `Matrix3d::normalized()`: the matrix divided by its Frobenius norm.
fn normalized(m: Matrix3d) -> Matrix3d {
    m / m.norm()
}

#[test]
fn homography_matrix_estimator_collinear_minimal_sample_triplets() {
    set_prng_seed(0);
    let dst = vec![
        Vector2d::new(0.0, 0.0),
        Vector2d::new(1.0, 0.0),
        Vector2d::new(1.0, 1.0),
        Vector2d::new(0.0, 1.0),
    ];

    for outlier_idx in 0..4 {
        let mut src = vec![
            Vector2d::new(0.0, 0.0),
            Vector2d::new(1.0, 0.0),
            Vector2d::new(2.0, 0.0),
            Vector2d::new(3.0, 0.0),
        ];
        src[outlier_idx] = Vector2d::new(0.0, 1.0);

        let mut estimator = HomographyMatrixEstimator;
        let mut models = Vec::new();
        estimator.estimate(&src, &dst, &mut models).unwrap();

        assert!(models.is_empty());
    }
}

fn nominal(num_points: usize) {
    set_prng_seed(0);
    for x in 0..10 {
        let expected_h = Matrix3d::new(x as f64, 0.2, 0.3, 30.0, 0.2, 0.1, 0.3, 20.0, 1.0);

        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..num_points {
            src.push(random_eigen_vector2d());
            dst.push((expected_h * src[i].homogeneous()).hnormalized());
        }

        let mut estimator = HomographyMatrixEstimator;
        let mut models = Vec::new();
        estimator.estimate(&src, &dst, &mut models).unwrap();

        assert_eq!(models.len(), 1);

        let mut residuals = Vec::new();
        estimator
            .residuals(&src, &dst, &models[0], &mut residuals)
            .unwrap();

        for residual in &residuals[..num_points] {
            assert!(*residual < 1e-6, "{residual}");
        }
    }
}

// Test numerical stability with large coordinates. This is to ensure that the
// homography matrix estimator is numerically stable despite not using
// coordinate normalization. We can do this because of double precision.
fn numerical_stability(num_points: usize) {
    set_prng_seed(0);
    const K_COORDINATE_SCALE: f64 = 1e6;
    for x in 1..10 {
        let mut expected_h = Matrix3d::identity();
        expected_h[(0, 0)] = x as f64;

        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..num_points {
            src.push(random_eigen_vector2d() * K_COORDINATE_SCALE);
            dst.push((expected_h * src[i].homogeneous()).hnormalized());
        }

        let mut estimator = HomographyMatrixEstimator;
        let mut models = Vec::new();
        estimator.estimate(&src, &dst, &mut models).unwrap();
        assert_eq!(models.len(), 1);

        let mut residuals = Vec::new();
        estimator
            .residuals(&src, &dst, &models[0], &mut residuals)
            .unwrap();

        for residual in &residuals[..num_points] {
            assert!(*residual < 1e-6, "{residual}");
        }
    }
}

fn noise_stability(num_points: usize) {
    set_prng_seed(0);
    const K_NOISE: f64 = 1e-3;
    for x in 1..10 {
        let mut expected_h = Matrix3d::identity();
        expected_h[(0, 0)] = x as f64;

        let mut src = Vec::new();
        let mut dst = Vec::new();
        for i in 0..num_points {
            src.push(random_eigen_vector2d());
            let projected = (expected_h * src[i].homogeneous()).hnormalized();
            dst.push(projected + random_eigen_vector2d() * K_NOISE);
        }

        let mut estimator = HomographyMatrixEstimator;
        let mut models = Vec::new();
        estimator.estimate(&src, &dst, &mut models).unwrap();
        assert_eq!(models.len(), 1);

        let mut residuals = Vec::new();
        estimator
            .residuals(&src, &dst, &models[0], &mut residuals)
            .unwrap();

        for residual in &residuals[..num_points] {
            assert!(*residual < 1e-5, "{residual}");
        }
    }
}

fn degenerate(num_points: usize) {
    set_prng_seed(0);
    const K_NOISE: f64 = 1e-3;

    for x in 0..10 {
        let expected_h = Matrix3d::new(x as f64, 0.2, 0.3, 30.0, 0.2, 0.1, 0.3, 20.0, 1.0);

        let mut src = vec![
            Vector2d::new(2.0, 1.0),
            Vector2d::new(3.0, 1.0),
            Vector2d::new(10.0, 30.0),
        ];
        assert!(num_points >= 4);
        let num_redundant_points = num_points - src.len();
        for _ in 0..num_redundant_points {
            src.push(src[0]);
        }

        let mut dst = Vec::new();
        for point in &src {
            let dsth = expected_h * point.homogeneous();
            dst.push(dsth.hnormalized() + random_eigen_vector2d() * K_NOISE);
        }

        let mut estimator = HomographyMatrixEstimator;
        let mut models = Vec::new();
        estimator.estimate(&src, &dst, &mut models).unwrap();

        assert_eq!(models.len(), 0);
    }
}

#[test]
fn homography_matrix_homography_matrix_tests_nominal_4() {
    nominal(4);
}
#[test]
fn homography_matrix_homography_matrix_tests_nominal_8() {
    nominal(8);
}
#[test]
fn homography_matrix_homography_matrix_tests_nominal_64() {
    nominal(64);
}
#[test]
fn homography_matrix_homography_matrix_tests_nominal_1024() {
    nominal(1024);
}
#[test]
fn homography_matrix_homography_matrix_tests_numerical_stability_4() {
    numerical_stability(4);
}
#[test]
fn homography_matrix_homography_matrix_tests_numerical_stability_8() {
    numerical_stability(8);
}
#[test]
fn homography_matrix_homography_matrix_tests_numerical_stability_64() {
    numerical_stability(64);
}
#[test]
fn homography_matrix_homography_matrix_tests_numerical_stability_1024() {
    numerical_stability(1024);
}
#[test]
fn homography_matrix_homography_matrix_tests_noise_stability_4() {
    noise_stability(4);
}
#[test]
fn homography_matrix_homography_matrix_tests_noise_stability_8() {
    noise_stability(8);
}
#[test]
fn homography_matrix_homography_matrix_tests_noise_stability_64() {
    noise_stability(64);
}
#[test]
fn homography_matrix_homography_matrix_tests_noise_stability_1024() {
    noise_stability(1024);
}
#[test]
fn homography_matrix_homography_matrix_tests_degenerate_4() {
    degenerate(4);
}
#[test]
fn homography_matrix_homography_matrix_tests_degenerate_8() {
    degenerate(8);
}
#[test]
fn homography_matrix_homography_matrix_tests_degenerate_64() {
    degenerate(64);
}
#[test]
fn homography_matrix_homography_matrix_tests_degenerate_1024() {
    degenerate(1024);
}

/// The outputs of `SyntheticRayCorrespondences`.
#[derive(Default)]
struct RayCorrespondences {
    src: Vec<Vector2d>,
    dst: Vec<Vector2d>,
    cam_rays1: Vec<Vector3d>,
    cam_rays2: Vec<CamRayWithImgPoint>,
}

// Correspondences for a plane whose pixel-space homography is `H_pix`, with
// `dst` displaced by `offset` pixels so that residuals are non-trivial.
fn synthetic_ray_correspondences(
    camera1: &Camera,
    camera2: &Camera,
    h_pix: &Matrix3d,
    num_points: usize,
    offset: f64,
    out: &mut RayCorrespondences,
) {
    for _ in 0..num_points {
        let image_size = Vector2d::new(camera1.width as f64, camera1.height as f64);
        let point1 = image_size.cwise_product((random_eigen_vector2d() + Vector2d::ones()) * 0.5);
        let point2 =
            (*h_pix * point1.homogeneous()).hnormalized() + random_eigen_vector2d() * offset;
        out.src.push(point1);
        out.dst.push(point2);
        out.cam_rays1
            .push(camera1.cam_ray_from_img(point1).unwrap());
        out.cam_rays2.push(CamRayWithImgPoint {
            ray: camera2.cam_ray_from_img(point2).unwrap(),
            img_point: point2,
        });
    }
}

fn ray_nominal(num_points: usize) {
    set_prng_seed(0);
    let camera1 =
        Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1000.0, 1920, 1080).unwrap();
    let camera2 =
        Camera::create_from_model_id(2, CameraModelId::SimplePinhole, 1200.0, 1920, 1080).unwrap();

    for x in 1..10 {
        let h_pix = Matrix3d::new(
            1.0 + 0.1 * x as f64,
            0.02,
            30.0,
            0.03,
            1.1,
            20.0,
            1e-5,
            2e-5,
            1.0,
        );

        let mut data = RayCorrespondences::default();
        synthetic_ray_correspondences(&camera1, &camera2, &h_pix, num_points, 0.0, &mut data);

        let mut estimator = HomographyMatrixRayEstimator::new(Some(&camera2));
        let mut models = Vec::new();
        estimator
            .estimate(&data.cam_rays1, &data.cam_rays2, &mut models)
            .unwrap();

        assert_eq!(models.len(), 1);

        // Also guards the global sign: a negated H transfers every ray behind the
        // camera and scores every residual at the maximum.
        let mut residuals = Vec::new();
        estimator
            .residuals(&data.cam_rays1, &data.cam_rays2, &models[0], &mut residuals)
            .unwrap();
        for residual in &residuals[..num_points] {
            assert!(*residual < 1e-6, "{residual}");
        }

        // The estimate maps rays to rays, so it must agree with the pixel-space
        // homography only after conjugation by the calibration matrices.
        let h_from_rays = camera2.calibration_matrix().unwrap()
            * models[0]
            * camera1.calibration_matrix().unwrap().inverse();
        assert!(
            normalized(h_from_rays).is_approx_with(normalized(h_pix), 1e-6)
                || normalized(h_from_rays).is_approx_with(-normalized(h_pix), 1e-6)
        );
    }
}

// For undistorted pinhole cameras the two residuals are algebraically
// identical, since projecting K2^-1 H K1 x1 back through K2 dehomogenizes to
// exactly H p1. Large offsets are included because a first-order relationship
// would also pass near zero.
fn ray_pixel_equivalence(num_points: usize) {
    set_prng_seed(0);
    let camera1 =
        Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1000.0, 1920, 1080).unwrap();
    let camera2 =
        Camera::create_from_model_id(2, CameraModelId::Pinhole, 1200.0, 1920, 1080).unwrap();

    for offset in [0.0, 1.0, 20.0, 200.0] {
        let h_pix = Matrix3d::new(1.2, 0.02, 30.0, 0.03, 1.1, 20.0, 1e-5, 2e-5, 1.0);

        let mut data = RayCorrespondences::default();
        synthetic_ray_correspondences(&camera1, &camera2, &h_pix, num_points, offset, &mut data);

        let h_ray = camera2.calibration_matrix().unwrap().inverse()
            * h_pix
            * camera1.calibration_matrix().unwrap();

        let mut pixel_residuals = Vec::new();
        HomographyMatrixEstimator
            .residuals(&data.src, &data.dst, &h_pix, &mut pixel_residuals)
            .unwrap();

        let mut ray_residuals = Vec::new();
        HomographyMatrixRayEstimator::new(Some(&camera2))
            .residuals(&data.cam_rays1, &data.cam_rays2, &h_ray, &mut ray_residuals)
            .unwrap();

        for i in 0..num_points {
            expect_near(
                ray_residuals[i],
                pixel_residuals[i],
                1e-9 * 1.0f64.max(pixel_residuals[i]),
            );
        }
    }
}

// A hypothesis that transfers a ray out of a perspective camera's field has no
// image point to score against. Cannot happen for a true inlier, whose measured
// ray faces forward, so this only guards wrong hypotheses.
#[test]
fn homography_matrix_ray_behind_camera() {
    set_prng_seed(0);
    let camera =
        Camera::create_from_model_id(1, CameraModelId::SimplePinhole, 1000.0, 1920, 1080).unwrap();
    let h = AngleAxisd::new(deg_to_rad(100.0), Vector3d::unit_y()).to_rotation_matrix();
    // The second ray stays in front after the rotation, so the rejection must be
    // selective rather than blanket.
    let forward = AngleAxisd::new(deg_to_rad(-70.0), Vector3d::unit_y()).to_rotation_matrix()
        * Vector3d::unit_z();
    let cam_rays1 = vec![Vector3d::unit_z(), forward];
    let cam_rays2 = vec![
        CamRayWithImgPoint {
            ray: Vector3d::unit_z(),
            img_point: Vector2d::new(960.0, 540.0),
        },
        CamRayWithImgPoint {
            ray: (h * forward).normalized(),
            img_point: camera
                .img_from_cam((h * forward).normalized(), true)
                .unwrap(),
        },
    ];

    let mut residuals = Vec::new();
    HomographyMatrixRayEstimator::new(Some(&camera))
        .residuals(&cam_rays1, &cam_rays2, &h, &mut residuals)
        .unwrap();

    assert_eq!(residuals[0], f64::MAX);
    assert!(residuals[1] < 1e-6, "{}", residuals[1]);
}

// A spherical camera images every direction, so the estimator must work over
// the whole sphere and reject nothing on cheirality grounds.
fn ray_spherical(num_points: usize) {
    set_prng_seed(0);
    let camera =
        Camera::create_from_model_id(1, CameraModelId::Equirectangular, 0.0, 2048, 1024).unwrap();
    assert!(camera.is_spherical());

    for x in 1..10 {
        let expected_h = AngleAxisd::new(0.2 * x as f64, Vector3d::new(0.2, 1.0, 0.3).normalized())
            .to_rotation_matrix();

        let mut cam_rays1 = Vec::new();
        let mut cam_rays2 = Vec::new();
        for _ in 0..num_points {
            let ray1 = random_eigen_vector3d().normalized();
            let ray2 = (expected_h * ray1).normalized();
            cam_rays1.push(ray1);
            cam_rays2.push(CamRayWithImgPoint {
                ray: ray2,
                img_point: camera.img_from_cam(ray2, true).unwrap(),
            });
        }

        let mut estimator = HomographyMatrixRayEstimator::new(Some(&camera));
        let mut models = Vec::new();
        estimator
            .estimate(&cam_rays1, &cam_rays2, &mut models)
            .unwrap();

        assert_eq!(models.len(), 1);
        assert!(normalized(models[0]).is_approx_with(normalized(expected_h), 1e-6));

        let mut residuals = Vec::new();
        estimator
            .residuals(&cam_rays1, &cam_rays2, &models[0], &mut residuals)
            .unwrap();
        for residual in &residuals[..num_points] {
            assert!(*residual < 1e-6, "{residual}");
        }
    }
}

// The azimuth wraps, so two nearly identical directions can sit a full image
// width apart in pixels. Scoring that raw difference would reject every
// correspondence near the seam.
#[test]
fn homography_matrix_ray_spherical_seam() {
    set_prng_seed(0);
    let camera =
        Camera::create_from_model_id(1, CameraModelId::Equirectangular, 0.0, 2048, 1024).unwrap();
    const K_DELTA: f64 = 0.01;
    let ray1 = Vector3d::new(fns::sin(PI - K_DELTA), 0.0, fns::cos(PI - K_DELTA));
    let ray2 = Vector3d::new(fns::sin(-PI + K_DELTA), 0.0, fns::cos(-PI + K_DELTA));
    let point1 = camera.img_from_cam(ray1, true).unwrap();
    let point2 = camera.img_from_cam(ray2, true).unwrap();
    assert!((point1 - point2).norm() > 2000.0);

    let mut residuals = Vec::new();
    HomographyMatrixRayEstimator::new(Some(&camera))
        .residuals(
            &[ray1],
            &[CamRayWithImgPoint {
                ray: ray2,
                img_point: point2,
            }],
            &Matrix3d::identity(),
            &mut residuals,
        )
        .unwrap();

    let expected = 2.0 * K_DELTA * camera.width as f64 / (2.0 * PI);
    expect_near(residuals[0], expected * expected, 1e-6);
}

// Why Estimate must resolve the sign of H. A spherical camera projects -H x1 as
// happily as H x1, only to the antipode, so a negated model is not rejected but
// silently scored against the wrong half of the sphere.
#[test]
fn homography_matrix_ray_spherical_sign() {
    set_prng_seed(0);
    let camera =
        Camera::create_from_model_id(1, CameraModelId::Equirectangular, 0.0, 2048, 1024).unwrap();
    let h = AngleAxisd::new(0.4, Vector3d::new(0.2, 1.0, 0.3).normalized()).to_rotation_matrix();

    let mut cam_rays1 = Vec::new();
    let mut cam_rays2 = Vec::new();
    for _ in 0..64 {
        let ray1 = random_eigen_vector3d().normalized();
        let ray2 = (h * ray1).normalized();
        cam_rays1.push(ray1);
        cam_rays2.push(CamRayWithImgPoint {
            ray: ray2,
            img_point: camera.img_from_cam(ray2, true).unwrap(),
        });
    }

    let mut estimator = HomographyMatrixRayEstimator::new(Some(&camera));

    let mut residuals = Vec::new();
    estimator
        .residuals(&cam_rays1, &cam_rays2, &h, &mut residuals)
        .unwrap();
    for residual in &residuals {
        assert!(*residual < 1e-6, "{residual}");
    }

    let mut negated_residuals = Vec::new();
    estimator
        .residuals(&cam_rays1, &cam_rays2, &-h, &mut negated_residuals)
        .unwrap();
    for residual in &negated_residuals {
        // Finite, so nothing rejects it, but half the image away in azimuth.
        assert!(*residual < f64::MAX);
        assert!(*residual > 1e4, "{residual}");
    }
}

#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_nominal_4() {
    ray_nominal(4);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_nominal_8() {
    ray_nominal(8);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_nominal_64() {
    ray_nominal(64);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_nominal_1024() {
    ray_nominal(1024);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_pixel_equivalence_4() {
    ray_pixel_equivalence(4);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_pixel_equivalence_8() {
    ray_pixel_equivalence(8);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_pixel_equivalence_64() {
    ray_pixel_equivalence(64);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_pixel_equivalence_1024() {
    ray_pixel_equivalence(1024);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_spherical_4() {
    ray_spherical(4);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_spherical_8() {
    ray_spherical(8);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_spherical_64() {
    ray_spherical(64);
}
#[test]
fn homography_matrix_ray_homography_matrix_ray_tests_spherical_1024() {
    ray_spherical(1024);
}
