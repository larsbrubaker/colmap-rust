// Rust-only (COLMAP has no test of Ceres' kernels): colmap_rust::solver::small_blas. Port of
// colmap-sharp's `ColmapSharp.Tests/Solver/SmallBlasTests.cs`, itself a port of Ceres Solver
// 2.2.0 `internal/ceres/small_blas_test.cc` (BSD-3-Clause, see THIRD_PARTY_NOTICES.md): same
// sizes, strides, start offsets, initial values and the 5 * epsilon tolerance, against
// products computed with MatrixXd. Tier A: the kernels' summation order is Ceres' naive
// loops'.
// - Ceres runs each product through four entry points: the dispatching kernel and the "Naive"
//   loop kernel, each with static and dynamic sizes. The port has dynamic sizes only, so the
//   `_dynamic` twins run the same kernel on the same sizes; the plain names run the
//   dispatching kernel and the `naive` names the loop kernel.
// - The C row stride of Ceres' kernels (row_stride_c) only bounds the matrix; the port's
//   kernels take the column stride alone, so row_stride_c sizes the test matrix as in Ceres.
// - MatrixTransposeVectorMultiply fills A and b with Eigen's setRandom(); the port draws the
//   same range, [-1, 1], from .NET's System.Random with seed 1 (colmap-sharp's values).

use colmap_rust::linalg::{MatrixXd, VectorXd};
use colmap_rust::solver::small_blas::{self, Operation};

use crate::net_random::NetRandom;

const TOLERANCE: f64 = 5.0 * f64::EPSILON;

const OPERATIONS: [Operation; 3] = [Operation::Add, Operation::Subtract, Operation::Assign];

type Kernel = fn(&[f64], usize, usize, &[f64], usize, &mut [f64], usize, usize, usize, Operation);

// Row-major data of the matrix with entries i + j + 1 (Ceres' initMatrix).
fn init_matrix(rows: usize, cols: usize) -> Vec<f64> {
    let mut m = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            m[i * cols + j] = (i + j + 1) as f64;
        }
    }

    m
}

// C op= product into rows [start_row, start_row + product.rows()) and the matching columns.
fn apply_reference(
    c: &mut MatrixXd,
    product: &MatrixXd,
    start_row: usize,
    start_col: usize,
    operation: Operation,
) {
    for r in 0..product.rows() {
        for k in 0..product.cols() {
            operation.store(&mut c[(start_row + r, start_col + k)], product[(r, k)]);
        }
    }
}

// TestMatrixFunctions (transpose = false) and TestMatrixTransposeFunctions (true).
fn test_matrix_functions(
    k_row_a: usize,
    k_col_a: usize,
    k_col_b: usize,
    transpose: bool,
    kernel: Kernel,
) {
    let a = init_matrix(k_row_a, k_col_a);
    let k_row_b = if transpose { k_row_a } else { k_col_a };
    let b = init_matrix(k_row_b, k_col_b);
    let a_matrix = MatrixXd::from_row_major(k_row_a, k_col_a, &a);
    let b_matrix = MatrixXd::from_row_major(k_row_b, k_col_b, &b);
    let product = if transpose {
        a_matrix.transpose_times(&b_matrix)
    } else {
        &a_matrix * &b_matrix
    };
    let k_row_c = if transpose { k_col_a } else { k_row_a };

    for row_stride_c in k_row_c..3 * k_row_c {
        for col_stride_c in k_col_b..3 * k_col_b {
            for operation in OPERATIONS {
                let mut c = vec![1.0; row_stride_c * col_stride_c];
                let mut c_ref = MatrixXd::constant(row_stride_c, col_stride_c, 1.0);
                let mut start_row_c = 0;
                while start_row_c + k_row_c < row_stride_c {
                    let mut start_col_c = 0;
                    while start_col_c + k_col_b < col_stride_c {
                        apply_reference(&mut c_ref, &product, start_row_c, start_col_c, operation);
                        kernel(
                            &a,
                            k_row_a,
                            k_col_a,
                            &b,
                            k_col_b,
                            &mut c,
                            start_row_c,
                            start_col_c,
                            col_stride_c,
                            operation,
                        );
                        let actual = MatrixXd::from_row_major(row_stride_c, col_stride_c, &c);
                        assert!((&c_ref - &actual).norm() <= TOLERANCE);
                        start_col_c += 1;
                    }
                    start_row_c += 1;
                }
            }
        }
    }
}

macro_rules! matrix_tests {
    ($($name:ident: ($rows:expr, $cols:expr, $cols_b:expr, $transpose:expr, $kernel:path);)*) => {
        $(
            #[test]
            fn $name() {
                test_matrix_functions($rows, $cols, $cols_b, $transpose, $kernel);
            }
        )*
    };
}

use small_blas::{
    matrix_matrix_multiply as mm, matrix_matrix_multiply_naive as mm_naive,
    matrix_transpose_matrix_multiply as mtm, matrix_transpose_matrix_multiply_naive as mtm_naive,
};

matrix_tests! {
    rust_only_blas_matrix_matrix_multiply_5_3_7: (5, 3, 7, false, mm);
    rust_only_blas_matrix_matrix_multiply_5_3_7_dynamic: (5, 3, 7, false, mm);
    rust_only_blas_matrix_matrix_multiply_1_1_1: (1, 1, 1, false, mm);
    rust_only_blas_matrix_matrix_multiply_1_1_1_dynamic: (1, 1, 1, false, mm);
    rust_only_blas_matrix_matrix_multiply_9_9_9: (9, 9, 9, false, mm);
    rust_only_blas_matrix_matrix_multiply_9_9_9_dynamic: (9, 9, 9, false, mm);
    rust_only_blas_matrix_matrix_multiply_naive_5_3_7: (5, 3, 7, false, mm_naive);
    rust_only_blas_matrix_matrix_multiply_naive_5_3_7_dynamic: (5, 3, 7, false, mm_naive);
    rust_only_blas_matrix_matrix_multiply_naive_1_1_1: (1, 1, 1, false, mm_naive);
    rust_only_blas_matrix_matrix_multiply_naive_1_1_1_dynamic: (1, 1, 1, false, mm_naive);
    rust_only_blas_matrix_matrix_multiply_naive_9_9_9: (9, 9, 9, false, mm_naive);
    rust_only_blas_matrix_matrix_multiply_naive_9_9_9_dynamic: (9, 9, 9, false, mm_naive);
    rust_only_blas_matrix_transpose_matrix_multiply_5_3_7: (5, 3, 7, true, mtm);
    rust_only_blas_matrix_transpose_matrix_multiply_5_3_7_dynamic: (5, 3, 7, true, mtm);
    rust_only_blas_matrix_transpose_matrix_multiply_1_1_1: (1, 1, 1, true, mtm);
    rust_only_blas_matrix_transpose_matrix_multiply_1_1_1_dynamic: (1, 1, 1, true, mtm);
    rust_only_blas_matrix_transpose_matrix_multiply_9_9_9: (9, 9, 9, true, mtm);
    rust_only_blas_matrix_transpose_matrix_multiply_9_9_9_dynamic: (9, 9, 9, true, mtm);
    rust_only_blas_matrix_transpose_matrix_multiply_naive_5_3_7: (5, 3, 7, true, mtm_naive);
    rust_only_blas_matrix_transpose_matrix_multiply_naive_5_3_7_dynamic: (5, 3, 7, true, mtm_naive);
    rust_only_blas_matrix_transpose_matrix_multiply_naive_1_1_1: (1, 1, 1, true, mtm_naive);
    rust_only_blas_matrix_transpose_matrix_multiply_naive_1_1_1_dynamic: (1, 1, 1, true, mtm_naive);
    rust_only_blas_matrix_transpose_matrix_multiply_naive_9_9_9: (9, 9, 9, true, mtm_naive);
    rust_only_blas_matrix_transpose_matrix_multiply_naive_9_9_9_dynamic: (9, 9, 9, true, mtm_naive);
}

fn test_matrix_vector(transpose: bool, mut fill: impl FnMut() -> f64) {
    for num_rows_a in 1..10 {
        for num_cols_a in 1..10 {
            let a: Vec<f64> = (0..num_rows_a * num_cols_a).map(|_| fill()).collect();
            let (b_length, c_length) = if transpose {
                (num_rows_a, num_cols_a)
            } else {
                (num_cols_a, num_rows_a)
            };
            let b: Vec<f64> = (0..b_length).map(|_| fill()).collect();
            let a_matrix = MatrixXd::from_row_major(num_rows_a, num_cols_a, &a);
            let b_vector = VectorXd::from_slice(&b);
            let product = if transpose {
                a_matrix.transpose_times_vector(&b_vector)
            } else {
                &a_matrix * &b_vector
            };
            for operation in OPERATIONS {
                let mut c = vec![1.0; c_length];
                let ones = VectorXd::ones(c_length);
                let c_ref = match operation {
                    Operation::Add => &ones + &product,
                    Operation::Subtract => &ones - &product,
                    Operation::Assign => product.clone(),
                };
                if transpose {
                    small_blas::matrix_transpose_vector_multiply(
                        &a, num_rows_a, num_cols_a, &b, &mut c, operation,
                    );
                } else {
                    small_blas::matrix_vector_multiply(
                        &a, num_rows_a, num_cols_a, &b, &mut c, operation,
                    );
                }

                assert!((&c_ref - &VectorXd::from_slice(&c)).norm() <= TOLERANCE);
            }
        }
    }
}

#[test]
fn rust_only_blas_matrix_vector_multiply() {
    test_matrix_vector(false, || 1.0);
}

#[test]
fn rust_only_blas_matrix_transpose_vector_multiply() {
    let mut random = NetRandom::new(1);
    test_matrix_vector(true, || 2.0 * random.next_double() - 1.0);
}
