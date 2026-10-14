// Port of COLMAP's `src/colmap/optim/least_absolute_deviations_test.cc` (1:1, same values
// and tolerances). The TEST_P suite ParameterizedLeastAbsoluteDeviationsTests, instantiated
// over both solver types, becomes one test per case named after the suite and case; each runs
// the case for both types in INSTANTIATE_TEST_SUITE_P's order. Tests
// `colmap_rust::optim::LeastAbsoluteDeviationSolver` (Tier C over the Tier B sparse
// Cholesky, docs/CPP_DIVERGENCES.md entry 33).

use colmap_rust::linalg::{SparseMatrixCsc, SparseTriplet, Vector3d, VectorXd};
use colmap_rust::optim::{
    LeastAbsoluteDeviationSolver, LeastAbsoluteDeviationSolverOptions,
    LeastAbsoluteDeviationSolverType,
};
use colmap_rust::ErrorKind;

use crate::eigen_vector_xd_near;
use crate::expect_near;

const SOLVER_TYPES: [LeastAbsoluteDeviationSolverType; 2] = [
    LeastAbsoluteDeviationSolverType::SimplicialLlt,
    LeastAbsoluteDeviationSolverType::SupernodalCholmodLlt,
];

fn get_options(
    solver_type: LeastAbsoluteDeviationSolverType,
) -> LeastAbsoluteDeviationSolverOptions {
    LeastAbsoluteDeviationSolverOptions {
        solver_type,
        ..Default::default()
    }
}

fn l1_norm(v: &VectorXd) -> f64 {
    v.as_slice().iter().map(|x| x.abs()).sum()
}

fn matrix(rows: usize, cols: usize, entries: &[(usize, usize, f64)]) -> SparseMatrixCsc {
    let triplets: Vec<_> = entries
        .iter()
        .map(|&(i, j, v)| SparseTriplet::new(i, j, v))
        .collect();
    SparseMatrixCsc::from_triplets(rows, cols, &triplets).unwrap()
}

/// A(i, j) = i * cols + j + 1 with A(0, 0) = 10, the matrix of the two "determined" cases.
fn counting_matrix(rows: usize, cols: usize) -> SparseMatrixCsc {
    let mut entries = Vec::new();
    for i in 0..rows {
        for j in 0..cols {
            let value = if i == 0 && j == 0 {
                10.0
            } else {
                (i * cols + j + 1) as f64
            };
            entries.push((i, j, value));
        }
    }
    matrix(rows, cols, &entries)
}

fn check_determined(rows: usize) {
    for solver_type in SOLVER_TYPES {
        let a = counting_matrix(rows, 3);
        let b = VectorXd::from_vec((0..rows).map(|i| (i + 1) as f64).collect());
        let mut x = VectorXd::zeros(3);

        let options = get_options(solver_type);
        let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
        assert!(solver.solve(&b, &mut x).unwrap(), "{solver_type:?}");

        // Reference solution obtained with Boyd's Matlab implementation.
        let expected: VectorXd = Vector3d::new(0.0, 0.0, 1.0 / 3.0).into();
        assert!(
            eigen_vector_xd_near(&x, &expected, 1e-12),
            "{solver_type:?}: {x}"
        );

        let residual = &(&a * &x) - &b;
        assert!(residual.norm() <= 1e-6, "{solver_type:?}");
    }
}

#[test]
fn parameterized_least_absolute_deviations_tests_over_determined() {
    check_determined(4);
}

#[test]
fn parameterized_least_absolute_deviations_tests_well_determined() {
    check_determined(3);
}

#[test]
fn parameterized_least_absolute_deviations_tests_under_determined() {
    for solver_type in SOLVER_TYPES {
        // In this case, the system is rank-deficient and not positive semi-definite.
        let a = SparseMatrixCsc::zero(2, 3);
        let options = get_options(solver_type);
        let error = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::RuntimeError, "{solver_type:?}");
    }
}

#[test]
fn parameterized_least_absolute_deviations_tests_simple_overdetermined_system() {
    for solver_type in SOLVER_TYPES {
        let a = matrix(
            4,
            3,
            &[
                (0, 0, 1.0),
                (0, 1, 0.0),
                (0, 2, 0.0),
                (1, 0, 0.0),
                (1, 1, 1.0),
                (1, 2, 0.0),
                (2, 0, 0.0),
                (2, 1, 0.0),
                (2, 2, 1.0),
                (3, 0, 1.0),
                (3, 1, 1.0),
                (3, 2, 1.0),
            ],
        );
        let b = VectorXd::from_slice(&[1.0, 2.0, 3.0, 6.0]);
        let options = get_options(solver_type);

        let mut x = VectorXd::zeros(3);
        assert!(l1_norm(&(&(&a * &x) - &b)) > 1e-1);

        let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
        solver.solve(&b, &mut x).unwrap();
        assert!(l1_norm(&(&(&a * &x) - &b)) <= 1e-6, "{solver_type:?}");
    }
}

#[test]
fn parameterized_least_absolute_deviations_tests_diagonal_system() {
    for solver_type in SOLVER_TYPES {
        let n = 5;
        let a = matrix(
            n,
            n,
            &(0..n).map(|i| (i, i, i as f64 + 1.0)).collect::<Vec<_>>(),
        );
        let b = VectorXd::from_vec(
            (0..n)
                .map(|i| (i as f64 + 1.0) * (i as f64 + 2.0))
                .collect(),
        );

        let mut options = get_options(solver_type);
        options.max_num_iterations = 100;

        let mut x = VectorXd::zeros(n);
        let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
        solver.solve(&b, &mut x).unwrap();

        // For diagonal systems, the L1 solution should be close to x_i = b_i / A_ii
        let expected = VectorXd::from_vec((0..n).map(|i| b[i] / a.coeff(i, i)).collect());
        assert!(l1_norm(&(&x - &expected)) <= 1e-6, "{solver_type:?}");
    }
}

#[test]
fn parameterized_least_absolute_deviations_tests_overdetermined_with_outliers() {
    for solver_type in SOLVER_TYPES {
        let mut entries = Vec::new();
        for i in 0..6 {
            entries.push((i, 0, 1.0));
            entries.push((i, 1, i as f64));
        }
        let a = matrix(6, 2, &entries);

        // Linear relationship b = 2 + 3*x, but with outliers
        let b = VectorXd::from_slice(&[2.0, 5.0, 8.0, 11.0, 1000.0, 17.0]); // b[4] is an outlier

        let mut options = get_options(solver_type);
        options.max_num_iterations = 1000;

        let mut x = VectorXd::zeros(2);
        let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
        solver.solve(&b, &mut x).unwrap();

        // The L1 solution should be more robust to the outlier than L2
        // Expected solution is approximately [2, 3]
        expect_near(x[0], 2.0, 1e-3);
        expect_near(x[1], 3.0, 1e-3);
    }
}

#[test]
fn parameterized_least_absolute_deviations_tests_identity_matrix() {
    for solver_type in SOLVER_TYPES {
        // Test with identity matrix - trivial case
        let n = 4;
        let a = matrix(n, n, &(0..n).map(|i| (i, i, 1.0)).collect::<Vec<_>>());
        let b = VectorXd::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let options = get_options(solver_type);

        let mut x = VectorXd::zeros(n);
        let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
        solver.solve(&b, &mut x).unwrap();

        // Solution should be exactly b for identity matrix
        assert!(eigen_vector_xd_near(&x, &b, 1e-3), "{solver_type:?}");
    }
}

#[test]
fn parameterized_least_absolute_deviations_tests_scaled_identity_matrix() {
    for solver_type in SOLVER_TYPES {
        // Test with scaled identity matrix
        let n = 3;
        let scale = 5.0;
        let a = matrix(n, n, &(0..n).map(|i| (i, i, scale)).collect::<Vec<_>>());
        let b = VectorXd::from_slice(&[5.0, 10.0, 15.0]);
        let options = get_options(solver_type);

        let mut x = VectorXd::zeros(n);
        let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
        solver.solve(&b, &mut x).unwrap();

        // Solution should be b / scale
        let expected = &b / scale;
        assert!(eigen_vector_xd_near(&x, &expected, 1e-3), "{solver_type:?}");
    }
}

#[test]
fn parameterized_least_absolute_deviations_tests_tolerance_settings() {
    for solver_type in SOLVER_TYPES {
        // Test that tighter tolerances produce more accurate results
        // Create a well-conditioned matrix with full column rank
        let a = matrix(
            5,
            3,
            &[
                (0, 0, 3.0),
                (0, 1, 0.5),
                (0, 2, 0.2),
                (1, 0, 0.5),
                (1, 1, 2.5),
                (1, 2, 0.3),
                (2, 0, 0.2),
                (2, 1, 0.3),
                (2, 2, 2.0),
                (3, 0, 1.0),
                (3, 1, 1.5),
                (3, 2, 0.5),
                (4, 0, 0.7),
                (4, 1, 0.6),
                (4, 2, 1.8),
            ],
        );
        let b = VectorXd::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);

        // Loose tolerance
        let mut x1 = VectorXd::zeros(3);
        {
            let mut options = get_options(solver_type);
            options.absolute_tolerance = 1e-1;
            options.relative_tolerance = 1e-1;
            let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
            solver.solve(&b, &mut x1).unwrap();
        }

        // Tight tolerance
        let mut x2 = VectorXd::zeros(3);
        {
            let mut options = get_options(solver_type);
            options.absolute_tolerance = 1e-6;
            options.relative_tolerance = 1e-4;
            options.max_num_iterations = 2000;
            let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
            solver.solve(&b, &mut x2).unwrap();
        }

        // The tighter tolerance solution should have lower or equal residual
        let residual1 = l1_norm(&(&(&a * &x1) - &b));
        let residual2 = l1_norm(&(&(&a * &x2) - &b));
        assert!(residual2 < 0.99 * residual1, "{solver_type:?}");
    }
}

#[test]
fn parameterized_least_absolute_deviations_tests_ridge_regularization() {
    for solver_type in SOLVER_TYPES {
        // Singular matrix (rank 1, two identical columns) makes A^T A not positive definite.
        // With ridge regularization, the solver still factorizes A^T A and Solve returns a
        // finite solution (no NaN). Without regularization, the solver should detect failure
        // and return false.
        let a = matrix(
            3,
            2,
            &[
                (0, 0, 1.0),
                (0, 1, 1.0),
                (1, 0, 2.0),
                (1, 1, 2.0),
                (2, 0, 3.0),
                (2, 1, 3.0),
            ],
        );
        let b = VectorXd::from_slice(&[2.0, 4.0, 6.0]);

        // Without regularization, the singular A^T A is not factorizable.
        {
            let options = get_options(solver_type);
            let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
            assert!(!solver.valid(), "{solver_type:?}");
            let mut x = VectorXd::zeros(2);
            assert!(!solver.solve(&b, &mut x).unwrap(), "{solver_type:?}");
        }

        // With regularization, factorization succeeds and Solve returns a finite (non-NaN)
        // solution.
        {
            let mut options = get_options(solver_type);
            options.ridge_regularization = 1e-9;
            let solver = LeastAbsoluteDeviationSolver::new(&options, &a).unwrap();
            assert!(solver.valid(), "{solver_type:?}");
            let mut x = VectorXd::zeros(2);
            assert!(solver.solve(&b, &mut x).unwrap(), "{solver_type:?}");
            assert!(!x.as_slice().iter().any(|v| v.is_nan()));
            // Either column achieves residual ~ 0 since b lies in span(A.col(0)).
            assert!(l1_norm(&(&(&a * &x) - &b)) <= 1e-3, "{solver_type:?}");
        }
    }
}
