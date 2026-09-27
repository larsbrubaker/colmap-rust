// Rust-only (COLMAP has no test for Eigen itself): hand-checked conventions of
// SelfAdjointEigenSolver, EigenSolver and FullPivLu that the numpy comparison in
// rust_only_spectral_oracle.rs does not pin: eigenvalue order, complex-pair order, extreme
// scales, and that EigenSolver's eigenvalues-only path is bit-identical to the full solve.
// Port of the eigen/LU half of colmap-sharp's `ColmapSharp.Tests/LinearAlgebra/SpectralTests.cs`
// (the SVD half is rust_only_spectral_svd.rs). Tier B where a tolerance appears; exact where
// the value is exactly representable. The random inputs use a port of .NET's seeded
// `System.Random` (`NetRandom` below), so they are colmap-sharp's matrices.

use colmap_rust::linalg::{
    Complex, ComputationInfo, EigenSolver, FullPivLu, HouseholderQr, MatrixXd,
    SelfAdjointEigenSolver,
};

/// .NET's `new Random(seed)` (the seeded, Knuth subtractive generator .NET keeps for
/// compatibility) and its `NextDouble()`.
struct NetRandom {
    seed_array: [i32; 56],
    inext: usize,
    inextp: usize,
}

impl NetRandom {
    fn new(seed: i32) -> Self {
        const MSEED: i32 = 161_803_398;
        let mut seed_array = [0_i32; 56];
        let subtraction = if seed == i32::MIN {
            i32::MAX
        } else {
            seed.abs()
        };
        let mut mj = MSEED - subtraction;
        seed_array[55] = mj;
        let mut mk = 1_i32;
        let mut ii = 0;
        for _ in 1..55 {
            ii += 21;
            if ii >= 55 {
                ii -= 55;
            }
            seed_array[ii] = mk;
            mk = mj - mk;
            if mk < 0 {
                mk += i32::MAX;
            }
            mj = seed_array[ii];
        }
        for _ in 1..5 {
            for i in 1..56 {
                let mut n = i + 30;
                if n >= 55 {
                    n -= 55;
                }
                seed_array[i] = seed_array[i].wrapping_sub(seed_array[1 + n]);
                if seed_array[i] < 0 {
                    seed_array[i] += i32::MAX;
                }
            }
        }
        Self {
            seed_array,
            inext: 0,
            inextp: 21,
        }
    }

    fn next_double(&mut self) -> f64 {
        let mut next = self.inext + 1;
        if next >= 56 {
            next = 1;
        }
        let mut nextp = self.inextp + 1;
        if nextp >= 56 {
            nextp = 1;
        }
        let mut value = self.seed_array[next].wrapping_sub(self.seed_array[nextp]);
        if value == i32::MAX {
            value -= 1;
        }
        if value < 0 {
            value += i32::MAX;
        }
        self.seed_array[next] = value;
        self.inext = next;
        self.inextp = nextp;
        value as f64 * (1.0 / i32::MAX as f64)
    }
}

#[test]
fn rust_only_self_adjoint_eigen_solver_reads_lower_triangle_and_sorts_ascending() {
    // The upper triangle is garbage; only the lower one ([[2, 1], [1, 2]]) counts.
    let a = MatrixXd::from_row_major(2, 2, &[2.0, 100.0, 1.0, 2.0]);
    let solver = SelfAdjointEigenSolver::new(&a, true);
    let values = solver.eigenvalues();
    assert!((values[0] - 1.0).abs() <= 1e-15);
    assert!((values[1] - 3.0).abs() <= 1e-15);
    let vectors = solver.eigenvectors();
    assert!((vectors[(0, 0)] + vectors[(1, 0)]).abs() <= 1e-15);
}

#[test]
fn rust_only_eigen_solver_complex_pair_lists_positive_imaginary_first() {
    // Rotation by 90 degrees plus 2: eigenvalues 2 +- i.
    let solver = EigenSolver::new(
        &MatrixXd::from_row_major(2, 2, &[2.0, -1.0, 1.0, 2.0]),
        true,
    );
    let values = solver.eigenvalues();
    assert_eq!(values[0], Complex::new(2.0, 1.0));
    assert_eq!(values[1], Complex::new(2.0, -1.0));
    let vectors = solver.eigenvectors();
    assert_eq!(vectors[(0, 1)], vectors[(0, 0)].conj());
}

fn eigenvalues_only_case(case_index: i32) -> MatrixXd {
    let mut random = NetRandom::new(1234 + case_index);
    match case_index {
        0..=2 => {
            // Dense random matrices, 64 x 64 like GR6P's action matrix and smaller.
            let n = match case_index {
                0 => 64,
                1 => 17,
                _ => 8,
            };
            let mut m = MatrixXd::zeros(n, n);
            for r in 0..n {
                for c in 0..n {
                    m[(r, c)] = random.next_double() - 0.5;
                }
            }
            m
        }
        3 => {
            // Q B Q^T with B block triangular: real eigenvalues 3, -1, 0.5, 2, 2 (repeated)
            // and complex pairs 1 +- 2i, -0.5 +- 0.25i, on a random orthogonal basis.
            #[rustfmt::skip]
            let b = MatrixXd::from_row_major(9, 9, &[
                3.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 1.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.5, 0.3, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.5, 0.25, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.25, -0.5, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0,
            ]);
            let mut g = MatrixXd::zeros(9, 9);
            for r in 0..9 {
                for c in 0..9 {
                    g[(r, c)] = random.next_double() - 0.5;
                }
            }
            let q = HouseholderQr::new(&g).householder_q();
            &(&q * &b) * &q.transpose()
        }
        _ => {
            // Companion matrix of (x^2 + 1)(x - 1)(x - 2)(x + 3)(x^2 - 2x + 5)
            // = x^7 - 2x^6 - 3x^5 + 12x^4 - 21x^3 + 38x^2 - 55x + 30.
            let coefficients = [-2.0, -3.0, 12.0, -21.0, 38.0, -55.0, 30.0];
            let mut m = MatrixXd::zeros(7, 7);
            for (c, &coefficient) in coefficients.iter().enumerate() {
                m[(0, c)] = -coefficient;
            }
            for r in 1..7 {
                m[(r, r - 1)] = 1.0;
            }
            m
        }
    }
}

// The eigenvalues-only solve skips Z and confines the QR sweeps to the active window; it must
// reproduce the full solve's eigenvalues bit for bit (GR6P's 64 x 64 action matrix relies on
// this). The inputs deflate many times, with complex pairs, real eigenvalues and 2x2 blocks
// with real eigenvalues that are split by a rotation.
#[test]
fn rust_only_eigen_solver_eigenvalues_only_matches_full_solve_bitwise() {
    for case_index in 0..5 {
        let a = eigenvalues_only_case(case_index);
        let full = EigenSolver::new(&a, true);
        let values_only = EigenSolver::new(&a, false);
        assert_eq!(full.info(), ComputationInfo::Success);
        assert_eq!(values_only.info(), ComputationInfo::Success);
        let expected = full.eigenvalues();
        let actual = values_only.eigenvalues();
        let mut complex_count = 0;
        for i in 0..expected.len() {
            assert_eq!(
                actual[i].re.to_bits(),
                expected[i].re.to_bits(),
                "case {case_index} value {i} real"
            );
            assert_eq!(
                actual[i].im.to_bits(),
                expected[i].im.to_bits(),
                "case {case_index} value {i} imag"
            );
            if expected[i].im != 0.0 {
                complex_count += 1;
            }
        }

        // Every case mixes real eigenvalues and complex pairs.
        assert!(complex_count > 0, "case {case_index}");
        assert!(complex_count < expected.len(), "case {case_index}");
    }
}

// Regression: without scaling, squared norms under/overflowed, no sweep ran and the unrotated
// diagonal came back as the eigenvalues.
#[test]
fn rust_only_self_adjoint_eigen_solver_extreme_scales() {
    for scale in [1e-170, 1e160] {
        let a = MatrixXd::constant(2, 2, scale);
        let solver = SelfAdjointEigenSolver::new(&a, true);
        let values = solver.eigenvalues();
        assert_eq!(solver.info(), ComputationInfo::Success, "{scale}");
        assert!(values[0].abs() / scale <= 1e-15, "{scale}");
        assert!((values[1] / (2.0 * scale) - 1.0).abs() <= 1e-15, "{scale}");
    }
}

// Regression: products of ~1e200 entries overflowed in the Francis step.
#[test]
fn rust_only_eigen_solver_huge_entries() {
    let unit = MatrixXd::from_row_major(3, 3, &[1.0, 2.0, 0.0, 0.5, 1.0, 3.0, 1.0, 0.0, 2.0]);
    let huge = &unit * 1e200;
    let reference = EigenSolver::new(&unit, false);
    let solver = EigenSolver::new(&huge, true);
    assert_eq!(solver.info(), ComputationInfo::Success);
    let expected = reference.eigenvalues();
    let actual = solver.eigenvalues();
    for i in 0..3 {
        assert!((actual[i] / 1e200 - expected[i]).abs() <= 1e-12, "{i}");
    }
}

#[test]
fn rust_only_full_piv_lu_rank_of_collinear_columns() {
    let a = MatrixXd::from_row_major(
        3,
        4,
        &[1.0, 2.0, 3.0, 4.0, 2.0, 4.0, 6.0, 8.0, 0.0, 0.0, 0.0, 0.0],
    );
    assert_eq!(FullPivLu::new(&a).rank(), 1);
    assert_eq!(FullPivLu::new(&MatrixXd::identity_rect(3, 5)).rank(), 3);
}

// Degenerate input with eigenvectors requested: the factor is NaN-filled (as JacobiSvd's U/V
// are), so `eigenvectors()` never panics on bad data (docs/CPP_DIVERGENCES.md entry 31).
#[test]
fn rust_only_self_adjoint_eigen_solver_nan_input_gives_nan_eigenvectors() {
    let a = MatrixXd::from_row_major(1, 1, &[f64::NAN]);
    let solver = SelfAdjointEigenSolver::new(&a, true);
    assert_eq!(solver.info(), ComputationInfo::InvalidInput);
    assert!(solver.eigenvalues()[0].is_nan());
    let v = solver.eigenvectors();
    assert_eq!((v.rows(), v.cols()), (1, 1));
    assert!(v[(0, 0)].is_nan());
    let without = SelfAdjointEigenSolver::new(&a, false);
    let message = std::panic::catch_unwind(|| without.eigenvectors())
        .expect_err("unrequested eigenvectors must panic");
    let text = message
        .downcast_ref::<String>()
        .cloned()
        .or_else(|| message.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap_or_default();
    assert!(text.contains("not requested"), "{text}");
}

// Same contract for the general solver (docs/CPP_DIVERGENCES.md entry 32).
#[test]
fn rust_only_eigen_solver_nan_input_gives_nan_eigenvectors() {
    let a = MatrixXd::from_row_major(1, 1, &[f64::NAN]);
    let solver = EigenSolver::new(&a, true);
    assert_eq!(solver.info(), ComputationInfo::InvalidInput);
    let v = solver.eigenvectors();
    assert_eq!((v.rows(), v.cols()), (1, 1));
    assert!(v[(0, 0)].re.is_nan() && v[(0, 0)].im.is_nan());
    let without = EigenSolver::new(&a, false);
    assert!(std::panic::catch_unwind(|| without.eigenvectors()).is_err());
}
