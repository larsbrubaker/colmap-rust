// Port of Ceres Solver 2.2.0's internal/ceres/loss_function_test.cc (BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md), via colmap-sharp's `ColmapSharp.Tests/Solver/LossFunctionTests.cs`:
// same names, values and tolerances, for `colmap_rust::solver::loss_function`. COLMAP has no
// loss-function test of its own. Each loss's rho'(s) and rho''(s) are checked against
// symmetric finite differences of rho(s), and rho at s = 0 against its closed form. Ceres'
// `LossFunctionWrapper` test is not ported because the wrapper is not (COLMAP never uses it).

use colmap_rust::solver::{
    ArctanLoss, CauchyLoss, ComposedLoss, HuberLoss, LossFunction, ScaledLoss, SoftLOneLoss,
    TolerantLoss, TrivialLoss, TukeyLoss,
};

// Compares the values of rho'(s) and rho''(s) computed by the callback with estimates
// obtained by symmetric finite differencing of rho(s).
fn assert_loss_function_is_valid(loss: &dyn LossFunction, s: f64) {
    assert!(s > 0.0);

    // Evaluate rho(s), rho'(s) and rho''(s).
    let mut rho = [0.0; 3];
    loss.evaluate(s, &mut rho);

    // Use symmetric finite differencing to estimate rho'(s) and rho''(s).
    const K_H: f64 = 1e-4;
    // Values at s + kH.
    let mut fwd = [0.0; 3];
    // Values at s - kH.
    let mut bwd = [0.0; 3];
    loss.evaluate(s + K_H, &mut fwd);
    loss.evaluate(s - K_H, &mut bwd);

    // First derivative.
    let fd_1 = (fwd[0] - bwd[0]) / (2.0 * K_H);
    assert!(
        (fd_1 - rho[1]).abs() <= 1e-6,
        "rho' {} vs fd {fd_1} at s = {s}",
        rho[1]
    );

    // Second derivative.
    let fd_2 = (fwd[0] - 2.0 * rho[0] + bwd[0]) / (K_H * K_H);
    assert!(
        (fd_2 - rho[2]).abs() <= 1e-6,
        "rho'' {} vs fd {fd_2} at s = {s}",
        rho[2]
    );
}

fn assert_rho_at_zero(loss: &dyn LossFunction, rho_0: f64, rho_1: f64, rho_2: f64) {
    let mut rho = [0.0; 3];
    loss.evaluate(0.0, &mut rho);
    assert!((rho[0] - rho_0).abs() <= 1e-6);
    assert!((rho[1] - rho_1).abs() <= 1e-6);
    assert!((rho[2] - rho_2).abs() <= 1e-6);
}

fn tolerant(a: f64, b: f64) -> TolerantLoss {
    TolerantLoss::new(a, b).expect("valid tolerant loss parameters")
}

// Try two values of the scaling a = 0.7 and 1.3 (where scaling makes sense) and of the
// squared norm s = 0.357 and 1.792.
// For the Huber loss this exercises both code paths (small and large values of s).

#[test]
fn loss_function_trivial_loss() {
    assert_loss_function_is_valid(&TrivialLoss, 0.357);
    assert_loss_function_is_valid(&TrivialLoss, 1.792);
    // Check that at s = 0: rho = [0, 1, 0].
    assert_rho_at_zero(&TrivialLoss, 0.0, 1.0, 0.0);
}

#[test]
fn loss_function_huber_loss() {
    assert_loss_function_is_valid(&HuberLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&HuberLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&HuberLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&HuberLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, 0].
    assert_rho_at_zero(&HuberLoss::new(0.7), 0.0, 1.0, 0.0);
}

#[test]
fn loss_function_soft_l_one_loss() {
    assert_loss_function_is_valid(&SoftLOneLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&SoftLOneLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&SoftLOneLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&SoftLOneLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, -1 / (2 * a^2)].
    assert_rho_at_zero(&SoftLOneLoss::new(0.7), 0.0, 1.0, -0.5 / (0.7 * 0.7));
}

#[test]
fn loss_function_cauchy_loss() {
    assert_loss_function_is_valid(&CauchyLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&CauchyLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&CauchyLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&CauchyLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, -1 / a^2].
    assert_rho_at_zero(&CauchyLoss::new(0.7), 0.0, 1.0, -1.0 / (0.7 * 0.7));
}

#[test]
fn loss_function_arctan_loss() {
    assert_loss_function_is_valid(&ArctanLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&ArctanLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&ArctanLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&ArctanLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, 0].
    assert_rho_at_zero(&ArctanLoss::new(0.7), 0.0, 1.0, 0.0);
}

#[test]
fn loss_function_tolerant_loss() {
    assert_loss_function_is_valid(&tolerant(0.7, 0.4), 0.357);
    assert_loss_function_is_valid(&tolerant(0.7, 0.4), 1.792);
    assert_loss_function_is_valid(&tolerant(0.7, 0.4), 55.5);
    assert_loss_function_is_valid(&tolerant(1.3, 0.1), 0.357);
    assert_loss_function_is_valid(&tolerant(1.3, 0.1), 1.792);
    assert_loss_function_is_valid(&tolerant(1.3, 0.1), 55.5);
    // Check the value at zero is actually zero.
    let mut rho = [0.0; 3];
    tolerant(0.7, 0.4).evaluate(0.0, &mut rho);
    assert!(rho[0].abs() <= 1e-6);
    // Check that loss before and after the approximation threshold are good.
    // A threshold of 36.7 is used by the implementation.
    assert_loss_function_is_valid(&tolerant(20.0, 1.0), 20.0 + 36.6);
    assert_loss_function_is_valid(&tolerant(20.0, 1.0), 20.0 + 36.7);
    assert_loss_function_is_valid(&tolerant(20.0, 1.0), 20.0 + 36.8);
    assert_loss_function_is_valid(&tolerant(20.0, 1.0), 20.0 + 1000.0);
}

#[test]
fn loss_function_tukey_loss() {
    assert_loss_function_is_valid(&TukeyLoss::new(0.7), 0.357);
    assert_loss_function_is_valid(&TukeyLoss::new(0.7), 1.792);
    assert_loss_function_is_valid(&TukeyLoss::new(1.3), 0.357);
    assert_loss_function_is_valid(&TukeyLoss::new(1.3), 1.792);
    // Check that at s = 0: rho = [0, 1, -2 / a^2].
    assert_rho_at_zero(&TukeyLoss::new(0.7), 0.0, 1.0, -2.0 / (0.7 * 0.7));
}

#[test]
fn loss_function_composed_loss() {
    {
        let c = ComposedLoss::new(
            Box::new(HuberLoss::new(0.7)),
            Box::new(CauchyLoss::new(1.3)),
        );
        assert_loss_function_is_valid(&c, 0.357);
        assert_loss_function_is_valid(&c, 1.792);
    }
    {
        let c = ComposedLoss::new(
            Box::new(CauchyLoss::new(0.7)),
            Box::new(HuberLoss::new(1.3)),
        );
        assert_loss_function_is_valid(&c, 0.357);
        assert_loss_function_is_valid(&c, 1.792);
    }
}

#[test]
fn loss_function_scaled_loss() {
    // Wrap a few loss functions, and a few scale factors.
    assert_loss_function_is_valid(&ScaledLoss::new(None, 6.0), 0.323);
    assert_loss_function_is_valid(&ScaledLoss::new(Some(Box::new(TrivialLoss)), 10.0), 0.357);
    assert_loss_function_is_valid(
        &ScaledLoss::new(Some(Box::new(HuberLoss::new(0.7))), 0.1),
        1.792,
    );
    assert_loss_function_is_valid(
        &ScaledLoss::new(Some(Box::new(SoftLOneLoss::new(1.3))), 0.1),
        1.792,
    );
    assert_loss_function_is_valid(
        &ScaledLoss::new(Some(Box::new(CauchyLoss::new(1.3))), 10.0),
        1.792,
    );
    assert_loss_function_is_valid(
        &ScaledLoss::new(Some(Box::new(ArctanLoss::new(1.3))), 10.0),
        1.792,
    );
    assert_loss_function_is_valid(
        &ScaledLoss::new(Some(Box::new(tolerant(1.3, 0.1))), 10.0),
        1.792,
    );
    assert_loss_function_is_valid(
        &ScaledLoss::new(
            Some(Box::new(ComposedLoss::new(
                Box::new(HuberLoss::new(0.8)),
                Box::new(tolerant(1.3, 0.5)),
            ))),
            10.0,
        ),
        1.792,
    );
}

// Rust-only: Ceres CHECK-fails on TolerantLoss(a < 0) and TolerantLoss(b <= 0); the port
// returns the check's error.
#[test]
fn rust_only_tolerant_loss_rejects_bad_parameters() {
    assert!(TolerantLoss::new(-0.1, 1.0).is_err());
    assert!(TolerantLoss::new(1.0, 0.0).is_err());
    assert!(TolerantLoss::new(0.0, 1.0).is_ok());
}
