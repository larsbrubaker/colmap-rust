// Port of COLMAP's src/colmap/estimators/cost_functions/tiny_sampson_error_test.cc
// (Suite_Name -> suite_name), testing colmap_rust::estimators::cost_functions::
// tiny_sampson_error. Residuals checked against geometry::essential_matrix's Sampson errors
// and closed-form Jacobians against central finite differences; COLMAP's tolerances are the
// bars.
//
// The two MatchesAutodiff tests pin the closed forms against Jets: the Ceres
// TangentSampsonErrorCostFunctor (cost_functions::sampson_error) and the one-sided functor's
// own generic operator(), through solver::AutoDiffCostFunction / TinySolverAutoDiffFunction.
// TinyFocalSampsonErrorCostFunctor is autodiff-only in C++; its closed-form Jacobian here is
// pinned against finite differences by a rust_only test (docs/CPP_DIVERGENCES.md, entry 252).

use crate::expect_near;
use colmap_rust::estimators::cost_functions::sampson_error::TangentSampsonErrorCostFunctor;
use colmap_rust::estimators::cost_functions::{
    TinyFocalSampsonErrorCostFunctor, TinyFundamentalSampsonErrorCostFunctor,
    TinyOneSidedFocalTangentSampsonErrorCostFunctor, TinyTangentSampsonErrorCostFunctor,
};
use colmap_rust::geometry::essential_matrix::{
    compute_squared_sampson_errors, compute_squared_tangent_sampson_error,
    compute_squared_tangent_sampson_error_rays, essential_matrix_from_pose,
};
use colmap_rust::geometry::pose::CamRayWithJac;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{
    AngleAxisd, Matrix3d, Matrix3x2d, Quaterniond, Svd3d, Vector2d, Vector3d,
};
use colmap_rust::math::fns;
use colmap_rust::optim::TinySolverFunction;
use colmap_rust::solver::CostFunction;

fn make(ray: Vector3d, jac: Matrix3x2d) -> CamRayWithJac {
    CamRayWithJac {
        ray: ray.normalized(),
        jacobian: jac,
    }
}

fn quaternion(angle: f64, axis: Vector3d) -> Quaterniond {
    Quaterniond::from_angle_axis(AngleAxisd::new(angle, axis.normalized()))
}

/// Central finite differences of `functor` at `p`, against its closed-form Jacobian.
fn check_jacobian_against_finite_differences<F: TinySolverFunction>(functor: &F, p: &[f64]) {
    let n = functor.num_residuals();
    let num_params = F::NUM_PARAMETERS;
    let mut residuals = vec![0.0; n];
    let mut jacobian = vec![0.0; n * num_params];
    assert!(functor.evaluate(p, &mut residuals, Some(&mut jacobian)));

    const K_EPS: f64 = 1e-6;
    for l in 0..num_params {
        let mut p_plus = p.to_vec();
        let mut p_minus = p.to_vec();
        p_plus[l] += K_EPS;
        p_minus[l] -= K_EPS;
        let mut res_plus = vec![0.0; n];
        let mut res_minus = vec![0.0; n];
        functor.evaluate(&p_plus, &mut res_plus, None);
        functor.evaluate(&p_minus, &mut res_minus, None);
        for i in 0..n {
            let finite_diff = (res_plus[i] - res_minus[i]) / (2.0 * K_EPS);
            expect_near(jacobian[i + l * n], finite_diff, 1e-5);
        }
    }
}

// The batched functor's squared residuals match ComputeSquaredTangentSampsonError at several
// 7-parameter poses.
#[test]
fn tiny_tangent_sampson_error_cost_functor_matches_squared_tangent_sampson_error() {
    let cam_rays1_with_jac = [
        make(
            Vector3d::new(0.1, 0.2, 1.0),
            Matrix3x2d::new(1.0, 0.1, 0.05, 1.0, 0.2, -0.1),
        ),
        make(
            Vector3d::new(-0.3, 0.1, 1.0),
            Matrix3x2d::new(0.9, -0.1, 0.15, 1.1, -0.05, 0.2),
        ),
        make(
            Vector3d::new(0.2, -0.25, 1.0),
            Matrix3x2d::new(1.05, 0.0, 0.0, 0.95, 0.1, 0.1),
        ),
    ];
    let cam_rays2_with_jac = [
        make(
            Vector3d::new(0.15, -0.1, 1.0),
            Matrix3x2d::new(1.0, 0.05, -0.1, 1.0, 0.2, 0.0),
        ),
        make(
            Vector3d::new(0.05, 0.3, 1.0),
            Matrix3x2d::new(0.8, 0.2, 0.1, 1.2, 0.0, -0.15),
        ),
        make(
            Vector3d::new(-0.2, -0.15, 1.0),
            Matrix3x2d::new(1.1, -0.05, 0.05, 0.9, -0.1, 0.1),
        ),
    ];
    let functor =
        TinyTangentSampsonErrorCostFunctor::new(&cam_rays1_with_jac, &cam_rays2_with_jac).unwrap();

    let quaternions = [
        quaternion(0.9, Vector3d::new(-1.0, 0.5, 2.0)),
        quaternion(0.3, Vector3d::new(0.2, -1.0, 0.7)),
    ];
    let translations = [
        Vector3d::new(1.0, -2.0, 0.5).normalized(),
        Vector3d::new(-0.4, 0.8, 1.2).normalized(),
    ];

    for (q, t) in quaternions.iter().zip(&translations) {
        let q = q.normalized();
        let cam2_from_cam1 = [q.x, q.y, q.z, q.w, t.x, t.y, t.z];

        let mut residuals = vec![0.0; cam_rays1_with_jac.len()];
        assert!(functor.evaluate(&cam2_from_cam1, &mut residuals, None));

        let e = essential_matrix_from_pose(&Rigid3d::new(q, *t));
        for i in 0..cam_rays1_with_jac.len() {
            let expected = compute_squared_tangent_sampson_error(
                &cam_rays1_with_jac[i],
                &cam_rays2_with_jac[i],
                &e,
            );
            expect_near(residuals[i] * residuals[i], expected, 1e-9);
        }
    }
}

// The closed-form 7-parameter Jacobian matches central finite differences.
#[test]
fn tiny_tangent_sampson_error_cost_functor_jacobian_matches_finite_difference() {
    let cam_rays1_with_jac = [
        make(
            Vector3d::new(0.1, 0.2, 1.0),
            Matrix3x2d::new(1.0, 0.1, 0.05, 1.0, 0.2, -0.1),
        ),
        make(
            Vector3d::new(-0.3, 0.1, 1.0),
            Matrix3x2d::new(0.9, -0.1, 0.15, 1.1, -0.05, 0.2),
        ),
    ];
    let cam_rays2_with_jac = [
        make(
            Vector3d::new(0.15, -0.1, 1.0),
            Matrix3x2d::new(1.0, 0.05, -0.1, 1.0, 0.2, 0.0),
        ),
        make(
            Vector3d::new(0.05, 0.3, 1.0),
            Matrix3x2d::new(0.8, 0.2, 0.1, 1.2, 0.0, -0.15),
        ),
    ];
    let functor =
        TinyTangentSampsonErrorCostFunctor::new(&cam_rays1_with_jac, &cam_rays2_with_jac).unwrap();

    let q = quaternion(0.7, Vector3d::new(0.2, -1.0, 0.5));
    let t = Vector3d::new(0.6, -0.3, 1.0).normalized();
    let p = [q.x, q.y, q.z, q.w, t.x, t.y, t.z];
    check_jacobian_against_finite_differences(&functor, &p);
}

#[test]
fn tiny_tangent_sampson_error_cost_functor_matches_autodiff_cost_functor() {
    let cam_rays1_with_jac = [
        make(
            Vector3d::new(0.1, 0.2, 1.0),
            Matrix3x2d::new(1.0, 0.1, 0.05, 1.0, 0.2, -0.1),
        ),
        make(
            Vector3d::new(-0.3, 0.1, 1.0),
            Matrix3x2d::new(0.9, -0.1, 0.15, 1.1, -0.05, 0.2),
        ),
    ];
    let cam_rays2_with_jac = [
        make(
            Vector3d::new(0.15, -0.1, 1.0),
            Matrix3x2d::new(1.0, 0.05, -0.1, 1.0, 0.2, 0.0),
        ),
        make(
            Vector3d::new(0.05, 0.3, 1.0),
            Matrix3x2d::new(0.8, 0.2, 0.1, 1.2, 0.0, -0.15),
        ),
    ];
    let functor =
        TinyTangentSampsonErrorCostFunctor::new(&cam_rays1_with_jac, &cam_rays2_with_jac).unwrap();
    let n = cam_rays1_with_jac.len();

    let q = quaternion(0.6, Vector3d::new(0.3, -0.7, 0.5));
    let t = Vector3d::new(0.4, 0.9, -0.2).normalized();
    let p = [q.x, q.y, q.z, q.w, t.x, t.y, t.z];

    let mut residuals = vec![0.0; n];
    let mut jacobian = vec![0.0; n * 7];
    assert!(functor.evaluate(&p, &mut residuals, Some(&mut jacobian)));

    for i in 0..n {
        let cost =
            TangentSampsonErrorCostFunctor::create(cam_rays1_with_jac[i], cam_rays2_with_jac[i])
                .unwrap();
        let mut residual_ad = [0.0];
        let mut jacobian_ad = [0.0; 7];
        let mut jacobian_ad_ptrs = [Some(&mut jacobian_ad[..])];
        let param_ptrs: [&[f64]; 1] = [&p];
        assert!(cost.evaluate(&param_ptrs, &mut residual_ad, Some(&mut jacobian_ad_ptrs)));
        expect_near(residuals[i], residual_ad[0], 1e-9);
        for l in 0..7 {
            expect_near(jacobian[i + l * n], jacobian_ad[l], 1e-9);
        }
    }
}

// Principal-point-centered image points (u - cx, v - cy), not calibrated rays, shared by the
// focal functor tests.
fn focal_test_points1() -> Vec<Vector2d> {
    vec![
        Vector2d::new(120.0, -45.0),
        Vector2d::new(-200.0, 80.0),
        Vector2d::new(33.0, 210.0),
    ]
}

fn focal_test_points2() -> Vec<Vector2d> {
    vec![
        Vector2d::new(95.0, -60.0),
        Vector2d::new(-180.0, 100.0),
        Vector2d::new(50.0, 190.0),
    ]
}

#[test]
fn tiny_focal_sampson_error_cost_functor_matches_squared_sampson_error() {
    let points1 = focal_test_points1();
    let points2 = focal_test_points2();
    let functor = TinyFocalSampsonErrorCostFunctor::new(&points1, &points2).unwrap();

    let q0 = quaternion(0.7, Vector3d::new(0.3, -1.0, 0.5));
    let t0 = Vector3d::new(1.0, -0.5, 2.0).normalized();
    let q1 = quaternion(0.25, Vector3d::new(-0.6, 0.4, 1.0));
    let t1 = Vector3d::new(-0.7, 1.1, 0.3).normalized();

    let quaternions = [q0, q1];
    let translations = [t0, t1];
    let focals = [900.0, 1500.0];

    for k in 0..quaternions.len() {
        let q = quaternions[k].normalized();
        let t = translations[k];
        let focal = focals[k];
        let params = [q.x, q.y, q.z, q.w, t.x, t.y, t.z, fns::ln(focal)];

        let mut residuals = vec![0.0; points1.len()];
        assert!(functor.evaluate(&params, &mut residuals, None));

        // Independent reference: build the pixel-space fundamental matrix and evaluate the
        // squared Sampson error with a matrix-based implementation.
        let e = essential_matrix_from_pose(&Rigid3d::new(q, t));
        let inv_f = 1.0 / focal;
        let k_inv = Matrix3d::from_diagonal(Vector3d::new(inv_f, inv_f, 1.0));
        let f = k_inv * e * k_inv;
        let expected_squared = compute_squared_sampson_errors(&points1, &points2, &f).unwrap();

        for i in 0..points1.len() {
            expect_near(residuals[i] * residuals[i], expected_squared[i], 1e-9);
        }
    }
}

// The closed-form 8-parameter Jacobian (C++ differentiates this functor by autodiff) matches
// central finite differences, including the log-focal column, at two poses and focals.
#[test]
fn rust_only_tiny_focal_sampson_error_cost_functor_jacobian_matches_finite_difference() {
    let points1 = focal_test_points1();
    let points2 = focal_test_points2();
    let functor = TinyFocalSampsonErrorCostFunctor::new(&points1, &points2).unwrap();

    for (q, t, focal) in [
        (
            quaternion(0.7, Vector3d::new(0.2, -1.0, 0.5)),
            Vector3d::new(0.6, -0.3, 1.0).normalized(),
            1100.0,
        ),
        (
            quaternion(0.25, Vector3d::new(-0.6, 0.4, 1.0)),
            Vector3d::new(-0.7, 1.1, 0.3).normalized(),
            450.0,
        ),
    ] {
        let p = [q.x, q.y, q.z, q.w, t.x, t.y, t.z, fns::ln(focal)];
        check_jacobian_against_finite_differences(&functor, &p);
    }
}

// Centered image points of the uncalibrated view and bearing rays with unprojection
// Jacobians for the calibrated view. The rays deliberately include one with z < 0, which no
// pinhole image plane could represent but a spherical camera observes.
fn one_sided_focal_test_points1() -> Vec<Vector2d> {
    vec![
        Vector2d::new(120.0, -45.0),
        Vector2d::new(-200.0, 80.0),
        Vector2d::new(33.0, 210.0),
    ]
}

fn one_sided_focal_test_rays2() -> Vec<CamRayWithJac> {
    vec![
        make(
            Vector3d::new(0.08, -0.05, 1.0),
            Matrix3x2d::new(1.1e-3, 5e-5, -1e-4, 1.2e-3, 2e-4, 0.0),
        ),
        make(
            Vector3d::new(-0.9, 0.4, 0.2),
            Matrix3x2d::new(9e-4, -1e-4, 1.5e-4, 1.1e-3, -5e-5, 2e-4),
        ),
        make(
            Vector3d::new(0.3, 0.7, -0.6),
            Matrix3x2d::new(8e-4, 2e-4, 1e-4, 1.3e-3, 0.0, -1.5e-4),
        ),
    ]
}

// The functor's squared residuals match ComputeSquaredTangentSampsonError under
// M = E * K1inv, with the uncalibrated view's constant measurement Jacobian
// d(x, y, 1)/d(x, y), at several poses and focal lengths.
#[test]
fn tiny_one_sided_focal_tangent_sampson_error_cost_functor_matches_squared_tangent_sampson_error() {
    let points1 = one_sided_focal_test_points1();
    let cam_rays2_with_jac = one_sided_focal_test_rays2();
    let functor =
        TinyOneSidedFocalTangentSampsonErrorCostFunctor::new(&points1, &cam_rays2_with_jac)
            .unwrap();

    let quaternions = [
        quaternion(0.7, Vector3d::new(0.3, -1.0, 0.5)),
        quaternion(0.25, Vector3d::new(-0.6, 0.4, 1.0)),
    ];
    let translations = [
        Vector3d::new(1.0, -0.5, 2.0).normalized(),
        Vector3d::new(-0.7, 1.1, 0.3).normalized(),
    ];
    let focals1 = [900.0, 1500.0];

    let j1 = Matrix3x2d::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);

    for k in 0..quaternions.len() {
        let q = quaternions[k].normalized();
        let t = translations[k];
        let focal1 = focals1[k];
        let params = [q.x, q.y, q.z, q.w, t.x, t.y, t.z, fns::ln(focal1)];

        let mut residuals = vec![0.0; points1.len()];
        assert!(functor.evaluate(&params, &mut residuals, None));

        let m = essential_matrix_from_pose(&Rigid3d::new(q, t))
            * Matrix3d::from_diagonal(Vector3d::new(1.0 / focal1, 1.0 / focal1, 1.0));
        for i in 0..points1.len() {
            let expected = compute_squared_tangent_sampson_error_rays(
                points1[i].homogeneous(),
                &j1,
                cam_rays2_with_jac[i].ray,
                &cam_rays2_with_jac[i].jacobian,
                &m,
            );
            expect_near(residuals[i] * residuals[i], expected, 1e-9);
        }
    }
}

// The closed-form 8-parameter Jacobian matches central finite differences. This covers the
// log-focal column, whose correctness rests on the unknown focal entering only through M and
// never through a measurement Jacobian.
#[test]
fn tiny_one_sided_focal_tangent_sampson_error_cost_functor_jacobian_matches_finite_difference() {
    let points1 = one_sided_focal_test_points1();
    let cam_rays2_with_jac = one_sided_focal_test_rays2();
    let functor =
        TinyOneSidedFocalTangentSampsonErrorCostFunctor::new(&points1, &cam_rays2_with_jac)
            .unwrap();

    let q = quaternion(0.7, Vector3d::new(0.2, -1.0, 0.5));
    let t = Vector3d::new(0.6, -0.3, 1.0).normalized();
    let p = [q.x, q.y, q.z, q.w, t.x, t.y, t.z, fns::ln(1100.0)];
    check_jacobian_against_finite_differences(&functor, &p);
}

#[test]
fn tiny_one_sided_focal_tangent_sampson_error_cost_functor_matches_autodiff_function() {
    let points1 = one_sided_focal_test_points1();
    let cam_rays2_with_jac = one_sided_focal_test_rays2();
    let functor =
        TinyOneSidedFocalTangentSampsonErrorCostFunctor::new(&points1, &cam_rays2_with_jac)
            .unwrap();
    let autodiff = functor.auto_diff_function();
    let n = points1.len();

    let q = quaternion(0.6, Vector3d::new(0.3, -0.7, 0.5));
    let t = Vector3d::new(0.4, 0.9, -0.2).normalized();
    let p = [q.x, q.y, q.z, q.w, t.x, t.y, t.z, fns::ln(1250.0)];

    let mut residuals = vec![0.0; n];
    let mut jacobian = vec![0.0; n * 8];
    assert!(functor.evaluate(&p, &mut residuals, Some(&mut jacobian)));

    let mut residuals_ad = vec![0.0; n];
    let mut jacobian_ad = vec![0.0; n * 8];
    assert!(autodiff.evaluate(&p, &mut residuals_ad, Some(&mut jacobian_ad)));

    for i in 0..n {
        expect_near(residuals[i], residuals_ad[i], 1e-9);
        for l in 0..8 {
            expect_near(jacobian[i + l * n], jacobian_ad[i + l * n], 1e-9);
        }
    }
}

// Image points shared by the fundamental matrix functor tests, in the normalized frame the
// refiner solves in.
fn fundamental_test_points1() -> Vec<Vector2d> {
    [
        (0.10, 0.20),
        (-0.30, 0.10),
        (0.20, -0.25),
        (-0.15, -0.35),
        (0.40, 0.05),
    ]
    .map(|(x, y)| Vector2d::new(x, y))
    .to_vec()
}

fn fundamental_test_points2() -> Vec<Vector2d> {
    [
        (0.15, -0.10),
        (0.05, 0.30),
        (-0.20, -0.15),
        (0.25, 0.35),
        (-0.05, 0.45),
    ]
    .map(|(x, y)| Vector2d::new(x, y))
    .to_vec()
}

fn fundamental_from_params(params: &[f64]) -> Matrix3d {
    TinyFundamentalSampsonErrorCostFunctor::fundamental_from_params(params)
}

// The factorized parameterization always yields a rank-2 matrix with singular values
// (1, sigma, 0), which is what makes the truncation step of the 8-point algorithm
// unnecessary.
#[test]
fn tiny_fundamental_sampson_error_cost_functor_parameterization_is_rank_two() {
    let q_u = quaternion(0.8, Vector3d::new(0.3, -1.0, 0.5));
    let q_v = quaternion(-0.4, Vector3d::new(1.0, 0.2, -0.7));
    let sigma = 0.6;
    let params = [
        q_u.x, q_u.y, q_u.z, q_u.w, q_v.x, q_v.y, q_v.z, q_v.w, sigma,
    ];

    let f = fundamental_from_params(&params);
    let singular_values = Svd3d::compute(&f).singular_values;
    expect_near(singular_values.x, 1.0, 1e-12);
    expect_near(singular_values.y, sigma, 1e-12);
    expect_near(singular_values.z, 0.0, 1e-12);
}

// The batched functor's squared residuals match ComputeSquaredSampsonError at several points
// in the 9-parameter space.
#[test]
fn tiny_fundamental_sampson_error_cost_functor_matches_squared_sampson_error() {
    let points1 = fundamental_test_points1();
    let points2 = fundamental_test_points2();
    let functor = TinyFundamentalSampsonErrorCostFunctor::new(&points1, &points2).unwrap();

    let quaternions_u = [
        quaternion(0.9, Vector3d::new(-1.0, 0.5, 2.0)),
        quaternion(0.2, Vector3d::new(0.4, 1.0, -0.3)),
    ];
    let quaternions_v = [
        quaternion(0.3, Vector3d::new(0.2, -1.0, 0.7)),
        quaternion(-0.6, Vector3d::new(1.0, 0.1, 0.2)),
    ];
    let sigmas = [0.85, 0.35];

    for k in 0..sigmas.len() {
        let q_u = quaternions_u[k].normalized();
        let q_v = quaternions_v[k].normalized();
        let params = [
            q_u.x, q_u.y, q_u.z, q_u.w, q_v.x, q_v.y, q_v.z, q_v.w, sigmas[k],
        ];

        let mut residuals = vec![0.0; points1.len()];
        assert!(functor.evaluate(&params, &mut residuals, None));

        let expected =
            compute_squared_sampson_errors(&points1, &points2, &fundamental_from_params(&params))
                .unwrap();
        for i in 0..points1.len() {
            expect_near(residuals[i] * residuals[i], expected[i], 1e-12);
        }
    }
}

// The closed-form 9-parameter Jacobian matches central finite differences.
#[test]
fn tiny_fundamental_sampson_error_cost_functor_jacobian_matches_finite_difference() {
    let points1 = fundamental_test_points1();
    let points2 = fundamental_test_points2();
    let functor = TinyFundamentalSampsonErrorCostFunctor::new(&points1, &points2).unwrap();

    let q_u = quaternion(0.7, Vector3d::new(0.2, -1.0, 0.5));
    let q_v = quaternion(0.5, Vector3d::new(-0.6, 0.3, 1.0));
    let p = [q_u.x, q_u.y, q_u.z, q_u.w, q_v.x, q_v.y, q_v.z, q_v.w, 0.7];
    check_jacobian_against_finite_differences(&functor, &p);
}
