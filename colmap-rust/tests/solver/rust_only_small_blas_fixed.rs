// Rust-only (no Ceres or COLMAP counterpart): the const-generic unrolled kernels of
// colmap_rust::solver::small_blas_fixed, reached through small_blas' dispatch, give the same
// bits as small_blas' naive loops for every operation, on random data with signed zeros and
// strided, offset output blocks. Port of colmap-sharp's C#-only
// `ColmapSharp.Tests/Solver/SmallBlasFixedTests.cs`, same seeds, shapes and offsets; the
// random values come from .NET's System.Random (`tests/support/net_random.rs`). Tier A.

use colmap_rust::math::fns;
use colmap_rust::solver::small_blas::{self, Operation};

use crate::net_random::NetRandom;

const OPERATIONS: [Operation; 3] = [Operation::Subtract, Operation::Assign, Operation::Add];

fn random_values(random: &mut NetRandom, count: usize) -> Vec<f64> {
    (0..count)
        .map(|i| {
            // Every seventh value is a negative zero, to pin the 0.0 + ... start of each sum.
            if i % 7 == 3 {
                -0.0
            } else {
                let mantissa = random.next_double() - 0.5;
                mantissa * fns::pow(10.0, random.next_range(-3, 4) as f64)
            }
        })
        .collect()
}

fn same_bits(x: &[f64], y: &[f64]) -> bool {
    x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
}

#[test]
fn rust_only_matrix_transpose_matrix_multiply_fixed_rows_match_naive() {
    let mut random = NetRandom::new(3);
    for rows_a in [2, 3] {
        for (cols_a, cols_b) in [(3, 3), (3, 6), (6, 6), (6, 3), (1, 4), (4, 1)] {
            for operation in OPERATIONS {
                let a = random_values(&mut random, rows_a * cols_a);
                let b = random_values(&mut random, rows_a * cols_b);
                const START_ROW: usize = 1;
                const START_COL: usize = 2;
                let stride = cols_b + 3;
                let mut fixed_c = random_values(&mut random, (cols_a + START_ROW) * stride);
                let mut naive_c = fixed_c.clone();
                small_blas::matrix_transpose_matrix_multiply(
                    &a,
                    rows_a,
                    cols_a,
                    &b,
                    cols_b,
                    &mut fixed_c,
                    START_ROW,
                    START_COL,
                    stride,
                    operation,
                );
                small_blas::matrix_transpose_matrix_multiply_naive(
                    &a,
                    rows_a,
                    cols_a,
                    &b,
                    cols_b,
                    &mut naive_c,
                    START_ROW,
                    START_COL,
                    stride,
                    operation,
                );
                assert!(same_bits(&fixed_c, &naive_c));
            }
        }
    }
}

#[test]
fn rust_only_matrix_matrix_multiply_fixed_inner_dimension_matches_naive() {
    let mut random = NetRandom::new(5);
    for (rows_a, cols_b) in [(3, 3), (6, 6), (6, 3), (1, 4), (4, 1)] {
        for operation in OPERATIONS {
            let a = random_values(&mut random, rows_a * 3);
            let b = random_values(&mut random, 3 * cols_b);
            const START_ROW: usize = 2;
            const START_COL: usize = 1;
            let stride = cols_b + 2;
            let mut fixed_c = random_values(&mut random, (rows_a + START_ROW) * stride);
            let mut naive_c = fixed_c.clone();
            small_blas::matrix_matrix_multiply(
                &a,
                rows_a,
                3,
                &b,
                cols_b,
                &mut fixed_c,
                START_ROW,
                START_COL,
                stride,
                operation,
            );
            small_blas::matrix_matrix_multiply_naive(
                &a,
                rows_a,
                3,
                &b,
                cols_b,
                &mut naive_c,
                START_ROW,
                START_COL,
                stride,
                operation,
            );
            assert!(same_bits(&fixed_c, &naive_c));
        }
    }
}
