// Rust-only tests (PoseLib has no re3q3 unit test and COLMAP none either; ports of
// colmap-sharp's C#-only `Re3q3Tests`) of colmap_rust::estimators::solvers::poselib::re3q3.
// They pin the random change-of-variables branch, which COLMAP's absolute_pose_test.cc cases
// never reach (the `re3q3_var_change.rs` header). Tier B: solutions are checked against the
// system.

use colmap_rust::estimators::solvers::poselib::re3q3;

// Coefficient order: x^2, xy, xz, y^2, yz, z^2, x, y, z, 1. Column-major 3 x 10.
fn evaluate(coeffs: &[f64], k: usize, x: f64, y: f64, z: f64) -> f64 {
    let monomials = [x * x, x * y, x * z, y * y, y * z, z * z, x, y, z, 1.0];
    let mut sum = 0.0;
    for (j, m) in monomials.iter().enumerate() {
        sum += coeffs[k + 3 * j] * m;
    }
    sum
}

/// x^2 = 1, y^2 = 4, z^2 = 9. The only quadratic monomials are x^2, y^2 and z^2, so all
/// three elimination matrices (y^2 z^2 yz / x^2 z^2 xz / y^2 x^2 xy) have a zero column and
/// determinant 0, which sends re3q3 through the random change of variables. Every returned
/// solution must satisfy the original system, and all eight real solutions (+-1, +-2, +-3)
/// must be found.
#[test]
fn rust_only_degenerate_elimination_uses_random_var_change() {
    let mut coeffs = [0.0; 30];
    coeffs[0] = 1.0; // x^2
    coeffs[3 * 9] = -1.0; // -1
    coeffs[1 + 3 * 3] = 1.0; // y^2
    coeffs[1 + 3 * 9] = -4.0; // -4
    coeffs[2 + 3 * 5] = 1.0; // z^2
    coeffs[2 + 3 * 9] = -9.0; // -9

    let mut solutions = [0.0; 24];
    let n = re3q3::solve(&coeffs, &mut solutions, true);

    let mut failures = Vec::new();
    let found: Vec<[f64; 3]> = (0..n)
        .map(|i| [solutions[3 * i], solutions[3 * i + 1], solutions[3 * i + 2]])
        .collect();
    for (i, s) in found.iter().enumerate() {
        for k in 0..3 {
            let r = evaluate(&coeffs, k, s[0], s[1], s[2]);
            if r.is_nan() || r.abs() >= 1e-8 {
                failures.push(format!("solution {i} {s:?}: equation {k} residual {r}"));
            }
        }
    }

    for ex in [1.0, -1.0] {
        for ey in [2.0, -2.0] {
            for ez in [3.0, -3.0] {
                let hit = found.iter().any(|s| {
                    (s[0] - ex).abs() < 1e-8 && (s[1] - ey).abs() < 1e-8 && (s[2] - ez).abs() < 1e-8
                });
                if !hit {
                    failures.push(format!("missing solution ({ex}, {ey}, {ez}); found {n}"));
                }
            }
        }
    }

    assert!(failures.is_empty(), "{failures:#?}");
}

/// x^2 = 1, y^2 = 4, z = x + y has four finite solutions, but its third equation is linear,
/// so no affine change of variables makes any elimination matrix regular. PoseLib fa7280f
/// returns 0 for every std::srand seed colmap-sharp tried; this pins the same upstream
/// behavior (see the re3q3.rs header).
#[test]
fn rust_only_linear_equation_returns_no_solutions_like_poselib() {
    let mut coeffs = [0.0; 30];
    coeffs[0] = 1.0; // x^2
    coeffs[3 * 9] = -1.0; // -1
    coeffs[1 + 3 * 3] = 1.0; // y^2
    coeffs[1 + 3 * 9] = -4.0; // -4
    coeffs[2 + 3 * 8] = 1.0; // z
    coeffs[2 + 3 * 6] = -1.0; // -x
    coeffs[2 + 3 * 7] = -1.0; // -y

    let mut solutions = [0.0; 24];
    assert_eq!(re3q3::solve(&coeffs, &mut solutions, true), 0);
    assert_eq!(re3q3::solve(&coeffs, &mut solutions, false), 0);
}
