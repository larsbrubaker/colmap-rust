// Rust-only (neither COLMAP nor colmap-sharp tests these directly; colmap-sharp reaches them
// through its Schur and linear solver tests, which land in later slices): the Jacobian
// matrices of colmap_rust::solver (DenseSparseMatrix, BlockSparseMatrix), the sparse reduced
// camera matrix's lower-triangle CSC export, and small_blas' PSD inversion. Every operation is
// checked against the same operation on a dense MatrixXd holding the same entries; inputs come
// from .NET's seeded System.Random (`tests/support/net_random.rs`).

use std::collections::BTreeSet;

use colmap_rust::linalg::{Llt, MatrixXd, VectorXd};
use colmap_rust::solver::small_blas;
use colmap_rust::solver::{
    Block, BlockRandomAccessMatrix, BlockRandomAccessSparseMatrix, BlockSparseMatrix, Cell,
    CompressedRow, CompressedRowBlockStructure, DenseSparseMatrix, LinearOperator, SparseMatrix,
};

use crate::net_random::NetRandom;

const TOLERANCE: f64 = 1e-14;

fn random_vector(random: &mut NetRandom, n: usize) -> Vec<f64> {
    (0..n).map(|_| 2.0 * random.next_double() - 1.0).collect()
}

// Column blocks of sizes 2, 3, 1; row blocks of sizes 2, 3, 2. The cells are laid out with
// column block 0's cells (the "E" block) first, as the Schur layout does, so cell positions
// are not in row order.
fn block_jacobian(random: &mut NetRandom) -> (BlockSparseMatrix, MatrixXd) {
    let cols = vec![Block::new(2, 0), Block::new(3, 2), Block::new(1, 5)];
    // (row block, column block) of each cell in value order: the E cells first.
    let rows = vec![
        CompressedRow::new(Block::new(2, 0), vec![Cell::new(0, 0), Cell::new(1, 10)]),
        CompressedRow::new(Block::new(3, 2), vec![Cell::new(0, 4), Cell::new(2, 16)]),
        CompressedRow::new(Block::new(2, 5), vec![Cell::new(1, 19), Cell::new(2, 25)]),
    ];
    let structure = CompressedRowBlockStructure::new(cols, rows);
    let mut matrix = BlockSparseMatrix::new(structure);
    assert_eq!(matrix.values().len(), 27);
    let values = random_vector(random, 27);
    matrix.values_mut().copy_from_slice(&values);

    let mut dense = MatrixXd::zeros(7, 6);
    for row in &matrix.structure().rows {
        for cell in &row.cells {
            let col = matrix.structure().cols[cell.block_id];
            for r in 0..row.block.size {
                for c in 0..col.size {
                    dense[(row.block.position + r, col.position + c)] =
                        values[cell.position + r * col.size + c];
                }
            }
        }
    }

    (matrix, dense)
}

#[test]
fn rust_only_block_sparse_matrix_matches_dense() {
    let mut random = NetRandom::new(7);
    let (mut matrix, dense) = block_jacobian(&mut random);
    assert_eq!((matrix.num_rows(), matrix.num_cols()), (7, 6));

    let x = random_vector(&mut random, 6);
    let mut y = random_vector(&mut random, 7);
    let expected = &VectorXd::from_slice(&y) + &(&dense * &VectorXd::from_slice(&x));
    matrix.right_multiply_and_accumulate(&x, &mut y);
    assert!((&expected - &VectorXd::from_slice(&y)).norm() <= TOLERANCE);

    let x = random_vector(&mut random, 7);
    let mut y = random_vector(&mut random, 6);
    let expected =
        &VectorXd::from_slice(&y) + &dense.transpose_times_vector(&VectorXd::from_slice(&x));
    matrix.left_multiply_and_accumulate(&x, &mut y);
    assert!((&expected - &VectorXd::from_slice(&y)).norm() <= TOLERANCE);

    let mut norms = vec![99.0; 6];
    matrix.squared_column_norm(&mut norms);
    for (c, &norm) in norms.iter().enumerate() {
        assert!((norm - dense.col(c).squared_norm()).abs() <= TOLERANCE);
    }

    let scale = random_vector(&mut random, 6);
    matrix.scale_columns(&scale);
    let scaled = &dense * &MatrixXd::from_diagonal(&VectorXd::from_slice(&scale));
    let x = VectorXd::ones(6);
    let mut y = vec![0.0; 7];
    matrix.right_multiply_and_accumulate(x.as_slice(), &mut y);
    assert!((&(&scaled * &x) - &VectorXd::from_slice(&y)).norm() <= TOLERANCE);

    matrix.set_zero();
    assert!(matrix.values().iter().all(|&v| v == 0.0));
}

#[test]
fn rust_only_dense_sparse_matrix_matches_dense() {
    let mut random = NetRandom::new(11);
    let mut matrix = DenseSparseMatrix::new(5, 4);
    let values = random_vector(&mut random, 20);
    matrix.matrix_mut().as_mut_slice().copy_from_slice(&values);
    let dense = matrix.matrix().clone();
    assert_eq!((matrix.num_rows(), matrix.num_cols()), (5, 4));

    let x = random_vector(&mut random, 4);
    let mut y = random_vector(&mut random, 5);
    let expected = &VectorXd::from_slice(&y) + &(&dense * &VectorXd::from_slice(&x));
    matrix.right_multiply_and_accumulate(&x, &mut y);
    assert!((&expected - &VectorXd::from_slice(&y)).norm() <= TOLERANCE);

    let x = random_vector(&mut random, 5);
    let mut y = random_vector(&mut random, 4);
    let expected =
        &VectorXd::from_slice(&y) + &dense.transpose_times_vector(&VectorXd::from_slice(&x));
    matrix.left_multiply_and_accumulate(&x, &mut y);
    assert!((&expected - &VectorXd::from_slice(&y)).norm() <= TOLERANCE);

    let mut norms = vec![0.0; 4];
    matrix.squared_column_norm(&mut norms);
    for (c, &norm) in norms.iter().enumerate() {
        assert!((norm - dense.col(c).squared_norm()).abs() <= TOLERANCE);
    }

    let scale = random_vector(&mut random, 4);
    matrix.scale_columns(&scale);
    let scaled = &dense * &MatrixXd::from_diagonal(&VectorXd::from_slice(&scale));
    assert!((&scaled - matrix.matrix()).norm() <= TOLERANCE);

    matrix.set_zero();
    assert_eq!(matrix.matrix().norm(), 0.0);
}

#[test]
fn rust_only_block_random_access_sparse_matrix_lower_csc_mirrors_the_upper_cells() {
    let blocks = vec![Block::new(3, 0), Block::new(4, 3), Block::new(5, 7)];
    let block_pairs: BTreeSet<(usize, usize)> = [(0, 0), (0, 2), (1, 1), (1, 2), (2, 2)]
        .into_iter()
        .collect();
    let mut m = BlockRandomAccessSparseMatrix::new(&blocks, &block_pairs);
    assert!(m.cell_location(0, 1).is_none());
    assert!(m.cell_location(2, 0).is_none());
    let mut random = NetRandom::new(13);
    let values = random_vector(&mut random, m.values().len());
    m.values_mut().copy_from_slice(&values);

    let (mut csc, sources) = m.to_lower_csc().unwrap();
    assert_eq!(sources.len(), csc.non_zeros());
    for (value, &source) in csc.values_mut().iter_mut().zip(&sources) {
        *value = m.values()[source];
    }

    // The CSC holds exactly the lower triangle of the symmetric matrix whose upper triangle
    // is stored.
    let mut symmetric = MatrixXd::zeros(12, 12);
    for (i, row_block) in blocks.iter().enumerate() {
        for (j, col_block) in blocks.iter().enumerate() {
            let Some((cell, stride)) = m.cell(i, j) else {
                continue;
            };
            for r in 0..row_block.size {
                // Only the upper triangle of a diagonal cell is meaningful (the eliminator
                // writes only that); its lower half holds random values here.
                let first = if i == j { r } else { 0 };
                for c in first..col_block.size {
                    let v = cell[r * stride + c];
                    symmetric[(row_block.position + r, col_block.position + c)] = v;
                    symmetric[(col_block.position + c, row_block.position + r)] = v;
                }
            }
        }
    }

    let lower = csc.to_dense();
    for r in 0..12 {
        for c in 0..12 {
            let expected = if r >= c { symmetric[(r, c)] } else { 0.0 };
            assert_eq!(lower[(r, c)], expected);
        }
    }
}

#[test]
fn rust_only_invert_upper_psd_matches_llt_and_rejects_indefinite() {
    let mut random = NetRandom::new(17);
    let n = 4;
    let a = MatrixXd::from_row_major(n, n, &random_vector(&mut random, n * n));
    let spd = &a.transpose_times(&a) + &MatrixXd::identity(n);
    // Row-major with garbage below the diagonal: only the upper triangle may be read.
    let mut m = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            m[i * n + j] = if i <= j { spd[(i, j)] } else { 1e300 };
        }
    }

    let mut inverse = vec![0.0; n * n];
    let mut scratch = vec![0.0; n * n];
    small_blas::invert_upper_psd(&m, n, &mut inverse, &mut scratch);
    let expected = Llt::new(&spd).solve_matrix(&MatrixXd::identity(n));
    let actual = MatrixXd::from_row_major(n, n, &inverse);
    assert!((&expected - &actual).norm() <= 1e-12);

    let mut in_place = m.clone();
    small_blas::invert_upper_psd_in_place(&mut in_place, n, &mut scratch);
    assert!(in_place
        .iter()
        .zip(&inverse)
        .all(|(p, q)| p.to_bits() == q.to_bits()));

    // Not positive definite: every entry becomes NaN.
    let indefinite = [1.0, 2.0, 2.0, 1.0];
    let mut inverse = [0.0; 4];
    small_blas::invert_upper_psd(&indefinite, 2, &mut inverse, &mut scratch);
    assert!(inverse.iter().all(|v| v.is_nan()));
}
