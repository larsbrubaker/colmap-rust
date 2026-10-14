// Port of Ceres Solver 2.2.0's internal/ceres/corrector_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md), via colmap-sharp's `ColmapSharp.Tests/Solver/CorrectorTests.cs`:
// the robust-loss correction of `colmap_rust::solver::Corrector`, same names, values and
// tolerances. The death tests become "returns Err": Ceres CHECK-fails, the port returns the
// check's error. The random cases draw from mt19937 with libc++'s
// uniform_real_distribution, as Ceres' test does (on the macOS build); the assertions do not
// depend on the values drawn.

use colmap_rust::math::random::{libcxx::uniform_real, Mt19937};
use colmap_rust::solver::Corrector;

// If rho[1] is zero, the Corrector constructor should crash.
#[test]
fn corrector_zero_gradient_death_test() {
    let k_rho = [0.0, 0.0, 1.0];
    assert!(Corrector::new(1.0, &k_rho).is_err());
}

// If rho[1] is negative, the Corrector constructor should crash.
#[test]
fn corrector_negative_gradient_death_test() {
    let k_rho = [0.0, -0.1, 1.0];
    assert!(Corrector::new(1.0, &k_rho).is_err());
}

fn assert_near(actual: f64, expected: f64, tolerance: f64) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} vs {expected} (tolerance {tolerance})"
    );
}

#[test]
fn corrector_scalar_correction() {
    let mut residuals = [3.0_f64.sqrt()];
    let mut jacobian = [10.0];
    let sq_norm = residuals[0] * residuals[0];
    let k_rho = [sq_norm, 0.1, -0.01];

    // In light of the rho'' < 0 clamping now implemented in corrector.cc, alpha = 0 whenever
    // rho'' < 0.
    let k_alpha = 0.0;

    // Thus the expected value of the residual is
    // residual[i] * sqrt(kRho[1]) / (1.0 - kAlpha).
    let k_expected_residual = residuals[0] * k_rho[1].sqrt() / (1.0 - k_alpha);

    // The jacobian in this case will be sqrt(kRho[1]) * (1 - kAlpha) * jacobian.
    let k_expected_jacobian = k_rho[1].sqrt() * (1.0 - k_alpha) * jacobian[0];

    let c = Corrector::new(sq_norm, &k_rho).unwrap();
    c.correct_jacobian(1, 1, &residuals, &mut jacobian);
    c.correct_residuals(&mut residuals);

    assert_near(residuals[0], k_expected_residual, 1e-6);
    assert_near(k_expected_jacobian, jacobian[0], 1e-6);
}

#[test]
fn corrector_scalar_correction_zero_residual() {
    let mut residuals = [0.0];
    let mut jacobian = [10.0];
    let sq_norm = residuals[0] * residuals[0];
    let k_rho = [0.0, 0.1, -0.01];
    let c = Corrector::new(sq_norm, &k_rho).unwrap();

    // The alpha equation is
    // 1/2(alpha^2) - alpha + 0.0 = 0.
    // i.e. alpha = 1.0 - sqrt(1.0).
    //      alpha = 0.0.
    // i.e. alpha = 0.0

    // Thus the expected value of the residual is
    // residual[i] * sqrt(kRho[1])
    let k_expected_residual = residuals[0] * k_rho[1].sqrt();

    // The jacobian in this case will be
    // sqrt(kRho[1]) * jacobian.
    let k_expected_jacobian = k_rho[1].sqrt() * jacobian[0];

    c.correct_jacobian(1, 1, &residuals, &mut jacobian);
    c.correct_residuals(&mut residuals);

    assert_near(residuals[0], k_expected_residual, 1e-6);
    assert_near(k_expected_jacobian, jacobian[0], 1e-6);
}

// Scaling behaviour for one dimensional functions.
#[test]
fn corrector_scalar_correction_alpha_clamped() {
    let mut residuals = [3.0_f64.sqrt()];
    let mut jacobian = [10.0];
    let sq_norm = residuals[0] * residuals[0];
    let k_rho: [f64; 3] = [3.0, 0.1, -0.1];

    // rho[2] < 0 -> alpha = 0.0
    let k_alpha = 0.0;

    // Thus the expected value of the residual is
    // residual[i] * sqrt(kRho[1]) / (1.0 - kAlpha)
    let k_expected_residual = residuals[0] * k_rho[1].sqrt() / (1.0 - k_alpha);

    // The jacobian in this case will be scaled by
    // sqrt(rho[1]) * (1 - alpha) * J.
    let k_expected_jacobian = k_rho[1].sqrt() * (1.0 - k_alpha) * jacobian[0];

    let c = Corrector::new(sq_norm, &k_rho).unwrap();
    c.correct_jacobian(1, 1, &residuals, &mut jacobian);
    c.correct_residuals(&mut residuals);

    assert_near(residuals[0], k_expected_residual, 1e-6);
    assert_near(k_expected_jacobian, jacobian[0], 1e-6);
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn diff_norm(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// Test that the corrected multidimensional residual and jacobians match the expected values
// and the resulting modified normal equations match the robustified gauss newton
// approximation.
#[test]
fn corrector_multidimensional_gauss_newton_approximation() {
    let mut prng = Mt19937::default();
    let mut max_residual_error = 0.0_f64;
    let mut max_jacobian_error = 0.0_f64;
    let mut max_gradient_error = 0.0_f64;

    for _iter in 0..10000 {
        // Initialize the jacobian (3 x 2, row-major like Ceres' MatrixRef) and residual.
        let mut jac = [0.0; 6];
        let mut res = [0.0; 3];
        for value in &mut jac {
            *value = uniform_real(&mut prng, 0.0, 1.0);
        }
        for value in &mut res {
            *value = uniform_real(&mut prng, 0.0, 1.0);
        }

        let sq_norm = dot(&res, &res);
        let rho = [
            sq_norm,
            uniform_real(&mut prng, 0.0, 1.0),
            uniform_real(&mut prng, -1.0, 1.0),
        ];

        // If rho[2] > 0, then the curvature correction to the correction and the gauss
        // newton approximation will match. Otherwise, we will clamp alpha to 0.
        let k_d = 1.0 + (2.0 * rho[2] / rho[1] * sq_norm);
        let k_alpha = if rho[2] > 0.0 { 1.0 - k_d.sqrt() } else { 0.0 };

        // Ground truth values.
        let mut g_res = [0.0; 3];
        let mut g_jac = [0.0; 6];
        let mut g_grad = [0.0; 2];
        for r in 0..3 {
            g_res[r] = rho[1].sqrt() / (1.0 - k_alpha) * res[r];
        }
        for c in 0..2 {
            let mut r_tj = 0.0;
            for r in 0..3 {
                r_tj += res[r] * jac[r * 2 + c];
            }
            for r in 0..3 {
                g_jac[r * 2 + c] =
                    rho[1].sqrt() * (jac[r * 2 + c] - k_alpha / sq_norm * res[r] * r_tj);
            }
            g_grad[c] = rho[1] * r_tj;
        }

        let corrector = Corrector::new(sq_norm, &rho).unwrap();
        corrector.correct_jacobian(3, 2, &res, &mut jac);
        corrector.correct_residuals(&mut res);

        // Corrected gradient.
        let mut c_grad = [0.0; 2];
        for c in 0..2 {
            for r in 0..3 {
                c_grad[c] += jac[r * 2 + c] * res[r];
            }
        }

        max_residual_error = max_residual_error.max(diff_norm(&g_res, &res));
        max_jacobian_error = max_jacobian_error.max(diff_norm(&g_jac, &jac));
        max_gradient_error = max_gradient_error.max(diff_norm(&g_grad, &c_grad));
    }

    assert_near(max_residual_error, 0.0, 1e-10);
    assert_near(max_jacobian_error, 0.0, 1e-10);
    assert_near(max_gradient_error, 0.0, 1e-10);
}

#[test]
fn corrector_multidimensional_gauss_newton_approximation_zero_residual() {
    let mut prng = Mt19937::default();
    let mut max_residual_error = 0.0_f64;
    let mut max_jacobian_error = 0.0_f64;
    let mut max_gradient_error = 0.0_f64;
    let mut max_hessian_error = 0.0_f64;

    for _iter in 0..10000 {
        // Initialize the jacobian; zero residuals.
        let mut jac = [0.0; 6];
        let mut res = [0.0; 3];
        for value in &mut jac {
            *value = uniform_real(&mut prng, 0.0, 1.0);
        }

        let sq_norm = dot(&res, &res);
        let rho = [
            sq_norm,
            uniform_real(&mut prng, 0.0, 1.0),
            uniform_real(&mut prng, -1.0, 1.0),
        ];

        // Ground truth values: with r = 0 the r r' terms vanish.
        let g_res = [0.0; 3];
        let mut g_jac = [0.0; 6];
        let g_grad = [0.0; 2];
        let mut g_hess = [0.0; 4];
        for i in 0..6 {
            g_jac[i] = rho[1].sqrt() * jac[i];
        }
        for a in 0..2 {
            for b in 0..2 {
                let mut jtj = 0.0;
                for r in 0..3 {
                    jtj += jac[r * 2 + a] * jac[r * 2 + b];
                }
                g_hess[a * 2 + b] = rho[1] * jtj;
            }
        }

        let corrector = Corrector::new(sq_norm, &rho).unwrap();
        corrector.correct_jacobian(3, 2, &res, &mut jac);
        corrector.correct_residuals(&mut res);

        // Corrected gradient and hessian.
        let mut c_grad = [0.0; 2];
        let mut c_hess = [0.0; 4];
        for a in 0..2 {
            for r in 0..3 {
                c_grad[a] += jac[r * 2 + a] * res[r];
            }
            for b in 0..2 {
                for r in 0..3 {
                    c_hess[a * 2 + b] += jac[r * 2 + a] * jac[r * 2 + b];
                }
            }
        }

        max_residual_error = max_residual_error.max(diff_norm(&g_res, &res));
        max_jacobian_error = max_jacobian_error.max(diff_norm(&g_jac, &jac));
        max_gradient_error = max_gradient_error.max(diff_norm(&g_grad, &c_grad));
        max_hessian_error = max_hessian_error.max(diff_norm(&g_hess, &c_hess));
    }

    assert_near(max_residual_error, 0.0, 1e-10);
    assert_near(max_jacobian_error, 0.0, 1e-10);
    assert_near(max_gradient_error, 0.0, 1e-10);
    assert_near(max_hessian_error, 0.0, 1e-10);
}

// Rust-only: Ceres CHECK-fails on a negative squared norm; the port returns the check's error.
#[test]
fn rust_only_corrector_rejects_negative_squared_norm() {
    assert!(Corrector::new(-1.0, &[0.0, 1.0, 0.0]).is_err());
}
