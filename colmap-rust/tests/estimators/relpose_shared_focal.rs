// Port of COLMAP's src/colmap/estimators/solvers/relpose_shared_focal_test.cc (1:1, same
// values and tolerances), testing colmap_rust::estimators::solvers::relpose_shared_focal.
// Tier B for the minimal solver and residuals, Tier C for the refiner.
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. Arguments COLMAP
// draws inside one call are drawn here in clang's left-to-right order.

use colmap_rust::estimators::solvers::{
    RelativePoseSharedFocalEstimator, RelativePoseSharedFocalModel,
};
use colmap_rust::geometry::essential_matrix::essential_matrix_from_pose;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Quaterniond, Vector2d, Vector3d};
use colmap_rust::math::deg_to_rad;
use colmap_rust::math::fns;
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::random_eigen_vector3d;

// Rejection thresholds used to condition a minimal sample: minimum depth in front of either
// camera and minimum parallax (sin^2 of the ray angle).
const K_MIN_DEPTH: f64 = 0.5;
const K_MIN_PARALLAX: f64 = 1e-2; // ~5.7 degrees.

/// Eigen's `normalized()` of a matrix: divided by its Frobenius norm.
fn normalized(m: Matrix3d) -> Matrix3d {
    m / m.norm()
}

// Random relative pose with a unit-norm baseline (away from the pure-rotation degeneracy) and
// a bounded rotation. The rotation is bounded so the two view frustums overlap: this keeps
// the sampled points cheirality-consistent (in front of both cameras) without the rejection
// sampler spinning for near-opposite orientations.
fn test_cam2_from_cam1() -> Rigid3d {
    let max_angle_deg = 60.0;
    // Resample until the shared focal is identifiable (the same precondition the estimator
    // enforces via IsFocalIdentifiable). For a singular pose the focal is unrecoverable and
    // the minimal solver returns a meaningless focal, which no downstream assertion can
    // meaningfully check.
    loop {
        let axis = random_eigen_vector3d().normalized();
        let angle = deg_to_rad(random_uniform_real(0.0, max_angle_deg));
        let rotation = Quaterniond::from_angle_axis(AngleAxisd::new(angle, axis));
        let cam2_from_cam1 = Rigid3d::new(rotation, random_eigen_vector3d().normalized());
        if RelativePoseSharedFocalEstimator::is_focal_identifiable(&cam2_from_cam1) {
            return cam2_from_cam1;
        }
    }
}

// Generates principal-point-centered image point pairs (f * X / Z) for a shared focal length
// `focal`, from random 3D points in front of both cameras. When reject_degenerate is set,
// resamples points that make a minimal 6-point solve ill-conditioned.
fn random_shared_focal_correspondences(
    cam2_from_cam1: &Rigid3d,
    focal: f64,
    num_points: usize,
    reject_degenerate: bool,
    points1: &mut Vec<Vector2d>,
    points2: &mut Vec<Vector2d>,
) {
    for _ in 0..num_points {
        let mut point_in_cam1;
        let mut point_in_cam2;
        loop {
            // Point in front of cam1 with a moderate field of view (|x/z|, |y/z| <= ~0.5):
            // wide-angle points span a large magnitude range that degrades the conditioning
            // of the minimal solve.
            let mut ray1 = random_eigen_vector3d();
            ray1.z = ray1.z.abs() + 2.0;
            let depth = random_uniform_real(1.0, 3.0);
            point_in_cam1 = depth * ray1.normalized();
            point_in_cam2 = *cam2_from_cam1 * point_in_cam1;
            let ray1_in_cam2 = cam2_from_cam1.rotation * point_in_cam1.normalized();
            let cos_parallax = ray1_in_cam2.dot(point_in_cam2.normalized());
            // The point is always in front of cam1; require it in front of cam2 too, with
            // sufficient parallax.
            let degenerate =
                point_in_cam2.z < K_MIN_DEPTH || 1.0 - cos_parallax * cos_parallax < K_MIN_PARALLAX;
            if !(reject_degenerate && degenerate) {
                break;
            }
        }
        points1.push(Vector2d::new(
            focal * point_in_cam1.x / point_in_cam1.z,
            focal * point_in_cam1.y / point_in_cam1.z,
        ));
        points2.push(Vector2d::new(
            focal * point_in_cam2.x / point_in_cam2.z,
            focal * point_in_cam2.y / point_in_cam2.z,
        ));
    }
}

// Whether at least one model recovers the essential matrix (up to scale/sign) and the focal
// length, with small residuals on the exact points. Returns a bool rather than asserting, so
// callers can tolerate the solver's intrinsic failure rate over many draws; see
// K_MAX_FAILURE_RATE.
#[allow(clippy::too_many_arguments)]
fn has_valid_model(
    points1: &[Vector2d],
    points2: &[Vector2d],
    expected_e: &Matrix3d,
    expected_focal: f64,
    models: &[RelativePoseSharedFocalModel],
    e_eps: f64,
    focal_rel_eps: f64,
    r_eps: f64,
) -> bool {
    let expected_e_n = normalized(*expected_e);
    for model in models {
        let e = normalized(model.e);
        if (e - expected_e_n).norm().min((e + expected_e_n).norm()) > e_eps {
            continue;
        }
        if (model.focal - expected_focal).abs() / expected_focal > focal_rel_eps {
            continue;
        }
        let mut residuals = Vec::new();
        RelativePoseSharedFocalEstimator::residuals(points1, points2, model, &mut residuals)
            .unwrap();
        if residuals.iter().any(|&r| r >= r_eps) {
            continue;
        }
        return true;
    }
    false
}

/// [`has_valid_model`] with C++'s default tolerances.
fn has_valid_model_default(
    points1: &[Vector2d],
    points2: &[Vector2d],
    expected_e: &Matrix3d,
    expected_focal: f64,
    models: &[RelativePoseSharedFocalModel],
) -> bool {
    has_valid_model(
        points1,
        points2,
        expected_e,
        expected_focal,
        models,
        5e-3,
        1e-2,
        1e-2,
    )
}

// Maximum fraction of samples that may fail. The minimal polynomial solve does not succeed on
// every sample: it loses the true root, or returns it imprecisely, for ~0.15% of samples. A
// 100-trial run therefore almost always observes a failure rate of 0% or 1%, and never
// exceeded 3% over 500 measured runs, while a real regression exceeds it immediately.
const K_NUM_TRIALS: usize = 100;
const K_MAX_FAILURE_RATE: f64 = 0.03;

// The minimal 6-point solver recovers the pose and focal on clean samples.
#[test]
fn relative_pose_shared_focal_estimator_nominal() {
    set_prng_seed(0);
    let k_focal = 1000.0;
    let mut num_failures = 0;
    for _ in 0..K_NUM_TRIALS {
        let cam2_from_cam1 = test_cam2_from_cam1();
        let expected_e = essential_matrix_from_pose(&cam2_from_cam1);
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        random_shared_focal_correspondences(
            &cam2_from_cam1,
            k_focal,
            RelativePoseSharedFocalEstimator::MIN_NUM_SAMPLES,
            true,
            &mut points1,
            &mut points2,
        );

        let mut models = Vec::new();
        RelativePoseSharedFocalEstimator::estimate(&points1, &points2, &mut models).unwrap();

        if !has_valid_model_default(&points1, &points2, &expected_e, k_focal, &models) {
            num_failures += 1;
        }
    }
    assert!(num_failures as f64 / K_NUM_TRIALS as f64 <= K_MAX_FAILURE_RATE);
}

// Residuals are near-zero on exact points, grow with a wrong focal, and are infinite for a
// non-positive focal.
#[test]
fn relative_pose_shared_focal_estimator_residuals() {
    set_prng_seed(0);
    let k_focal = 1000.0;
    let cam2_from_cam1 = test_cam2_from_cam1();
    let mut points1 = Vec::new();
    let mut points2 = Vec::new();
    random_shared_focal_correspondences(
        &cam2_from_cam1,
        k_focal,
        30,
        false,
        &mut points1,
        &mut points2,
    );

    let model = RelativePoseSharedFocalModel {
        e: essential_matrix_from_pose(&cam2_from_cam1),
        focal: k_focal,
    };
    let mut residuals = Vec::new();
    RelativePoseSharedFocalEstimator::residuals(&points1, &points2, &model, &mut residuals)
        .unwrap();
    let mut sum_exact = 0.0;
    for &residual in &residuals {
        assert!(residual < 1e-2, "{residual}");
        sum_exact += residual;
    }

    let mut wrong_model = model;
    wrong_model.focal = 1.5 * k_focal;
    let mut wrong_residuals = Vec::new();
    RelativePoseSharedFocalEstimator::residuals(
        &points1,
        &points2,
        &wrong_model,
        &mut wrong_residuals,
    )
    .unwrap();
    let sum_wrong: f64 = wrong_residuals.iter().sum();
    assert!(sum_wrong > sum_exact);

    let mut invalid_model = model;
    invalid_model.focal = 0.0;
    let mut invalid_residuals = Vec::new();
    RelativePoseSharedFocalEstimator::residuals(
        &points1,
        &points2,
        &invalid_model,
        &mut invalid_residuals,
    )
    .unwrap();
    for &residual in &invalid_residuals {
        assert_eq!(residual, f64::MAX);
    }
}

// Refinement pulls a perturbed pose + focal back to the ground truth.
#[test]
fn relative_pose_shared_focal_estimator_refine_from_initial_model() {
    set_prng_seed(0);
    let k_focal = 1000.0;
    for _ in 0..50 {
        let cam2_from_cam1 = test_cam2_from_cam1();
        let expected_e = essential_matrix_from_pose(&cam2_from_cam1);
        let mut points1 = Vec::new();
        let mut points2 = Vec::new();
        random_shared_focal_correspondences(
            &cam2_from_cam1,
            k_focal,
            50,
            false,
            &mut points1,
            &mut points2,
        );

        let seed_rotation = cam2_from_cam1.rotation
            * Quaterniond::from_angle_axis(AngleAxisd::new(
                0.02,
                random_eigen_vector3d().normalized(),
            ));
        let seed_translation =
            (cam2_from_cam1.translation + 0.02 * random_eigen_vector3d()).normalized();
        let mut model = RelativePoseSharedFocalModel {
            e: essential_matrix_from_pose(&Rigid3d::new(seed_rotation, seed_translation)),
            focal: 1.1 * k_focal,
        };

        assert!(RelativePoseSharedFocalEstimator::refine(&points1, &points2, &mut model).unwrap());

        // Refinement is a nonlinear least squares over 50 exact points seeded near the
        // solution, not a minimal polynomial solve, so it is expected to succeed on every
        // draw.
        let models = [model];
        assert!(has_valid_model(
            &points1,
            &points2,
            &expected_e,
            k_focal,
            &models,
            1e-3,
            1e-2,
            1e-2
        ));
    }
}

// A relative pose whose two optical axes intersect at a common fixation point on cam1's +z
// axis, so the axes are coplanar. cam1 sits at distance |fixation| from that point and cam2
// at dist2, so the configuration is isosceles iff dist2 == |fixation|.
fn fixating_pose(fixation: Vector3d, dist2: f64, angle_at_fixation_deg: f64) -> Rigid3d {
    let angle = deg_to_rad(angle_at_fixation_deg);
    let center2 = fixation + dist2 * Vector3d::new(-fns::sin(angle), 0.0, -fns::cos(angle));
    let rotation = Quaterniond::from_two_vectors(
        (fixation - center2).normalized(),
        Vector3d::new(0.0, 0.0, 1.0),
    );
    Rigid3d::new(rotation, rotation * -center2)
}

// The focal is unidentifiable for parallel axes and for coplanar axes meeting at an isosceles
// configuration, but identifiable for coplanar axes meeting asymmetrically, and for skew
// axes.
#[test]
fn relative_pose_shared_focal_estimator_is_focal_identifiable() {
    set_prng_seed(0);
    let fixation = Vector3d::new(0.0, 0.0, 2.0);

    // Both centers 2.0 from the fixation point: isosceles, unidentifiable.
    assert!(!RelativePoseSharedFocalEstimator::is_focal_identifiable(
        &fixating_pose(fixation, 2.0, 40.0)
    ));

    // Same axes, but cam2 much closer to the fixation point than cam1. Still coplanar, yet
    // far from isosceles, so the focal is constrained. This is the case a pure coplanarity
    // criterion would wrongly reject.
    assert!(RelativePoseSharedFocalEstimator::is_focal_identifiable(
        &fixating_pose(fixation, 1.0, 40.0)
    ));

    // Parallel axes with a lateral baseline: coplanar, no intersection.
    let parallel = Rigid3d::new(Quaterniond::identity(), Vector3d::new(1.0, 0.0, 0.0));
    assert!(!RelativePoseSharedFocalEstimator::is_focal_identifiable(
        &parallel
    ));

    // 90 deg about x with a lateral baseline: maximally skew axes.
    let skew = Rigid3d::new(
        Quaterniond::from_angle_axis(AngleAxisd::new(
            deg_to_rad(90.0),
            Vector3d::new(1.0, 0.0, 0.0),
        )),
        Vector3d::new(1.0, 0.0, 0.0),
    );
    assert!(RelativePoseSharedFocalEstimator::is_focal_identifiable(
        &skew
    ));

    // Pure rotation: no baseline at all.
    assert!(!RelativePoseSharedFocalEstimator::is_focal_identifiable(
        &Rigid3d::new(skew.rotation, Vector3d::zeros())
    ));

    // Scaling the translation scales the whole configuration, leaving both predicates
    // unchanged.
    for pose in [
        fixating_pose(fixation, 2.0, 40.0),
        fixating_pose(fixation, 1.0, 40.0),
    ] {
        let scaled = Rigid3d::new(pose.rotation, 1000.0 * pose.translation);
        assert_eq!(
            RelativePoseSharedFocalEstimator::is_focal_identifiable(&scaled),
            RelativePoseSharedFocalEstimator::is_focal_identifiable(&pose)
        );
    }
}
