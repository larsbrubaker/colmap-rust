// COLMAP's `EigenMatrixNear(rhs, tol)` gmock matcher (util/eigen_matchers.h) as a predicate over
// coefficient slices (test support, not a test), for the `estimators` test binary. Same rule as
// `eigen_matrix_near` in `tests/geometry.rs`: when `rhs.isZero()` (Eigen's default precision,
// every `|coefficient| <= 1e-12`), `lhs.norm() <= tol`; otherwise Eigen's
// `lhs.isApprox(rhs, tol)`, `||lhs - rhs|| <= tol * min(||lhs||, ||rhs||)` with Frobenius
// norms. Pass `matrix.as_slice()` (storage order is irrelevant: both sides use the same one).
//
// Include it from a test binary with `#[path = "support/eigen_matchers.rs"] mod
// eigen_matchers;`.

#![allow(dead_code)]

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// `EXPECT_THAT(lhs, EigenMatrixNear(rhs, tol))` on the coefficients of two same-shaped
/// matrices.
pub fn eigen_matrix_near(lhs: &[f64], rhs: &[f64], tol: f64) -> bool {
    assert_eq!(lhs.len(), rhs.len(), "EigenMatrixNear: shape mismatch");
    if rhs.iter().all(|&v| v.abs() <= 1e-12) {
        return l2(lhs) <= tol;
    }
    let diff: Vec<f64> = lhs.iter().zip(rhs).map(|(x, y)| x - y).collect();
    l2(&diff) <= tol * l2(lhs).min(l2(rhs))
}
