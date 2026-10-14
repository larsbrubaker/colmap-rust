// Integration-test binary for COLMAP's `optim/` module (layout convention: see the header of
// `tests/math.rs`). `optim/test_estimators.rs` holds the test-only estimators shared by the
// RANSAC files.
//
// Run: `cargo test -p colmap-rust --test optim`.

use colmap_rust::linalg::VectorXd;

#[path = "optim/test_estimators.rs"]
mod test_estimators;

#[path = "optim/combination_sampler.rs"]
mod combination_sampler;
#[path = "optim/least_absolute_deviations.rs"]
mod least_absolute_deviations;
#[path = "optim/loransac.rs"]
mod loransac;
#[path = "optim/progressive_sampler.rs"]
mod progressive_sampler;
#[path = "optim/random_sampler.rs"]
mod random_sampler;
#[path = "optim/ransac.rs"]
mod ransac;
#[path = "optim/rust_only_ransac.rs"]
mod rust_only_ransac;
#[path = "optim/rust_only_sampler_sequence.rs"]
mod rust_only_sampler_sequence;
#[path = "optim/sparse_cholesky.rs"]
mod sparse_cholesky;
#[path = "optim/sprt.rs"]
mod sprt;
#[path = "optim/support_measurement.rs"]
mod support_measurement;
#[path = "optim/tiny_solver.rs"]
mod tiny_solver;

fn l2(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Port of COLMAP's `EigenMatrixNear(rhs, tol)` matcher for dynamic vectors: when
/// `rhs.isZero()` (every `|coefficient| <= 1e-12`), `lhs.norm() <= tol`; otherwise Eigen's
/// `lhs.isApprox(rhs, tol)`, `||lhs - rhs|| <= tol * min(||lhs||, ||rhs||)`. A length
/// mismatch fails, as the matcher does.
pub fn eigen_vector_xd_near(lhs: &VectorXd, rhs: &VectorXd, tol: f64) -> bool {
    if lhs.len() != rhs.len() {
        return false;
    }
    let (a, b) = (lhs.as_slice(), rhs.as_slice());
    if b.iter().all(|&v| v.abs() <= 1e-12) {
        return l2(a) <= tol;
    }
    let diff: Vec<f64> = a.iter().zip(b).map(|(x, y)| x - y).collect();
    l2(&diff) <= tol * l2(a).min(l2(b))
}

/// gtest's `EXPECT_NEAR(a, b, tol)`.
pub fn expect_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "|{a} - {b}| > {tol}");
}
