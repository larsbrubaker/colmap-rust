// Rust-only (COLMAP has no test of Ceres' matrices): the block random access matrices of
// colmap_rust::solver. Port of colmap-sharp's
// `ColmapSharp.Tests/Solver/BlockRandomAccessMatrixTests.cs`, itself a port of Ceres Solver
// 2.2.0 `internal/ceres/block_random_access_dense_matrix_test.cc`,
// `block_random_access_sparse_matrix_test.cc` and `block_random_access_diagonal_matrix_test.cc`
// (BSD-3-Clause, see THIRD_PARTY_NOTICES.md): same blocks, values and tolerances.
// - Ceres' GetCell returns the whole value array with the cell's (row, col) and strides; the
//   port's `cell_location` returns the cell's offset into the values and its row stride. So
//   "row == position, col == position, stride == 12" (dense) becomes "the cell starts at
//   position_i * 12 + position_j of the values, stride 12", and "row == col == 0, strides ==
//   block sizes" (sparse, diagonal) becomes "stride == the column block's size".
// - The dense copies of the sparse and diagonal matrices (Ceres' ToDenseMatrix of the
//   underlying BlockSparseMatrix / CompressedRowSparseMatrix) are assembled from the stored
//   cells.
// - Vector::Random draws from .NET's System.Random with seed 1, same range [-1, 1].
// Not ported: BlockRandomAccessSparseMatrixTest.IntPairToInt64Overflow and Int64ToIntPair.
// They test Ceres' private bit-packed (row, col) cell key (kRowShift); the port finds a cell
// by binary search over the sorted (row, col) pairs, which has no packing to overflow or
// unpack.

// The loops index the blocks by id, as Ceres' tests do, so the ids read like theirs.
#![allow(clippy::needless_range_loop)]

use std::collections::BTreeSet;

use colmap_rust::linalg::{Llt, MatrixXd, VectorXd};
use colmap_rust::solver::{
    Block, BlockRandomAccessDenseMatrix, BlockRandomAccessDiagonalMatrix, BlockRandomAccessMatrix,
    BlockRandomAccessSparseMatrix,
};

use crate::net_random::NetRandom;

fn blocks() -> Vec<Block> {
    vec![Block::new(3, 0), Block::new(4, 3), Block::new(5, 7)]
}

const NUM_ROWS: usize = 3 + 4 + 5;

// The stored cells as a dense matrix (absent cells are zero).
fn to_dense(m: &dyn BlockRandomAccessMatrix, blocks: &[Block]) -> MatrixXd {
    let mut dense = MatrixXd::zeros(m.num_rows(), m.num_rows());
    for (i, row_block) in blocks.iter().enumerate() {
        for (j, col_block) in blocks.iter().enumerate() {
            let Some((cell, stride)) = m.cell(i, j) else {
                continue;
            };
            for r in 0..row_block.size {
                for c in 0..col_block.size {
                    dense[(row_block.position + r, col_block.position + c)] = cell[r * stride + c];
                }
            }
        }
    }

    dense
}

// cell = value * ones (+ identity on the diagonal blocks if requested).
fn fill(cell: &mut [f64], stride: usize, rows: usize, cols: usize, value: f64, identity: bool) {
    for r in 0..rows {
        for c in 0..cols {
            cell[r * stride + c] = value + if identity && r == c { 1.0 } else { 0.0 };
        }
    }
}

fn norm_of_difference(a: &MatrixXd, b: &MatrixXd) -> f64 {
    (a - b).norm()
}

#[test]
fn rust_only_block_random_access_dense_matrix_get_cell() {
    let blocks = blocks();
    let m = BlockRandomAccessDenseMatrix::new(&blocks);
    assert_eq!(m.num_rows(), NUM_ROWS);

    for i in 0..blocks.len() {
        let row_idx = blocks[i].position;
        for j in 0..blocks.len() {
            let col_idx = blocks[j].position;
            let location = m.cell_location(i, j);
            assert!(location.is_some());
            let location = location.unwrap();
            assert_eq!(location.offset, row_idx * NUM_ROWS + col_idx);
            assert_eq!(location.stride, 3 + 4 + 5);
        }
    }
}

#[test]
fn rust_only_block_random_access_dense_matrix_write_cell() {
    let blocks = blocks();
    let mut m = BlockRandomAccessDenseMatrix::new(&blocks);

    // Fill the cell (i,j) with (i + 1) * (j + 1)
    for i in 0..blocks.len() {
        for j in 0..blocks.len() {
            let (cell, stride) = m.cell_mut(i, j).unwrap();
            let value = ((i + 1) * (j + 1)) as f64;
            fill(cell, stride, blocks[i].size, blocks[j].size, value, false);
        }
    }

    // Check the values in the array are correct by going over the entries of each block
    // manually. (As in Ceres, the position checked is each block's first entry.)
    for i in 0..blocks.len() {
        let row_idx = blocks[i].position;
        for j in 0..blocks.len() {
            let col_idx = blocks[j].position;
            for _r in 0..blocks[i].size {
                for _c in 0..blocks[j].size {
                    let pos = row_idx * NUM_ROWS + col_idx;
                    assert_eq!(m.values()[pos], ((i + 1) * (j + 1)) as f64);
                }
            }
        }
    }
}

#[test]
fn rust_only_block_random_access_sparse_matrix_get_cell() {
    let blocks = blocks();
    let mut block_pairs = BTreeSet::new();
    let mut num_nonzeros = 0;
    block_pairs.insert((0, 0));
    num_nonzeros += blocks[0].size * blocks[0].size;
    block_pairs.insert((1, 1));
    num_nonzeros += blocks[1].size * blocks[1].size;
    block_pairs.insert((1, 2));
    num_nonzeros += blocks[1].size * blocks[2].size;
    block_pairs.insert((0, 2));
    num_nonzeros += blocks[2].size * blocks[0].size;

    let mut m = BlockRandomAccessSparseMatrix::new(&blocks, &block_pairs);
    assert_eq!(m.num_rows(), NUM_ROWS);

    for &(row_block_id, col_block_id) in &block_pairs {
        let found = m.cell_mut(row_block_id, col_block_id);
        assert!(found.is_some());
        let (cell, stride) = found.unwrap();
        let cell_length = cell.len();
        // Write into the block
        let value = ((row_block_id + 1) * (col_block_id + 1)) as f64;
        let (rows, cols) = (blocks[row_block_id].size, blocks[col_block_id].size);
        fill(cell, stride, rows, cols, value, false);
        assert_eq!(stride, blocks[col_block_id].size);
        assert_eq!(cell_length, rows * cols);
    }

    assert_eq!(m.values().len(), num_nonzeros);
    let dense = to_dense(&m, &blocks);
    const TOLERANCE: f64 = 1e-14;

    // (0, 0)
    let expected = MatrixXd::constant(3, 3, 1.0);
    assert!(norm_of_difference(&dense.block(0, 0, 3, 3), &expected) <= TOLERANCE);

    // (1, 1)
    let expected = MatrixXd::constant(4, 4, 2.0 * 2.0);
    assert!(norm_of_difference(&dense.block(3, 3, 4, 4), &expected) <= TOLERANCE);

    // (1, 2)
    let expected = MatrixXd::constant(4, 5, 2.0 * 3.0);
    assert!(norm_of_difference(&dense.block(3, 3 + 4, 4, 5), &expected) <= TOLERANCE);

    // (0, 2)
    let expected = MatrixXd::constant(3, 5, 3.0 * 1.0);
    assert!(norm_of_difference(&dense.block(0, 3 + 4, 3, 5), &expected) <= TOLERANCE);

    // There is nothing else in the matrix besides these four blocks.
    let expected_norm = (9.0_f64 + 16.0 * 16.0 + 36.0 * 20.0 + 9.0 * 15.0).sqrt();
    assert!((dense.norm() - expected_norm).abs() <= TOLERANCE);

    // expected_y = dense.selfadjointView<Upper>() * x
    let mut symmetric = dense.clone();
    for c in 0..NUM_ROWS {
        for r in c + 1..NUM_ROWS {
            symmetric[(r, c)] = dense[(c, r)];
        }
    }

    let x = VectorXd::ones(NUM_ROWS);
    let expected_y = &symmetric * &x;
    let mut actual_y = vec![0.0; NUM_ROWS];
    m.symmetric_right_multiply_and_accumulate(x.as_slice(), &mut actual_y);
    assert!((&expected_y - &VectorXd::from_slice(&actual_y)).norm() <= TOLERANCE);
}

const NUM_DIAGONAL_NONZEROS: usize = 3 * 3 + 4 * 4 + 5 * 5;

fn diagonal_set_up() -> (BlockRandomAccessDiagonalMatrix, Vec<Block>) {
    let blocks = blocks();
    let mut m = BlockRandomAccessDiagonalMatrix::new(&blocks);
    assert_eq!(m.num_rows(), NUM_ROWS);

    for i in 0..blocks.len() {
        for j in 0..blocks.len() {
            let found = m.cell_mut(i, j);

            // Off diagonal entries are not present.
            if i != j {
                assert!(found.is_none());
                continue;
            }

            assert!(found.is_some());
            let (cell, stride) = found.unwrap();
            let value = ((i + 1) * (j + 1)) as f64;
            fill(cell, stride, blocks[i].size, blocks[j].size, value, true);
            assert_eq!(stride, blocks[j].size);
        }
    }

    (m, blocks)
}

#[test]
fn rust_only_block_random_access_diagonal_matrix_matrix_contents() {
    let (m, blocks) = diagonal_set_up();
    assert_eq!(m.values().len(), NUM_DIAGONAL_NONZEROS);
    let dense = to_dense(&m, &blocks);
    const TOLERANCE: f64 = 1e-14;

    // (0,0)
    let expected = &MatrixXd::constant(3, 3, 1.0) + &MatrixXd::identity(3);
    assert!(norm_of_difference(&dense.block(0, 0, 3, 3), &expected) <= TOLERANCE);

    // (1,1)
    let expected = &MatrixXd::constant(4, 4, 2.0 * 2.0) + &MatrixXd::identity(4);
    assert!(norm_of_difference(&dense.block(3, 3, 4, 4), &expected) <= TOLERANCE);

    // (2,2)
    let expected = &MatrixXd::constant(5, 5, 3.0 * 3.0) + &MatrixXd::identity(5);
    assert!(norm_of_difference(&dense.block(7, 7, 5, 5), &expected) <= TOLERANCE);

    // There is nothing else in the matrix besides these four blocks.
    let expected_norm =
        (6.0 * 1.0 + 3.0 * 4.0 + 12.0 * 16.0 + 4.0 * 25.0 + 20.0 * 81.0 + 5.0 * 100.0_f64).sqrt();
    assert!((dense.norm() - expected_norm).abs() <= TOLERANCE);
}

#[test]
fn rust_only_block_random_access_diagonal_matrix_right_multiply_and_accumulate() {
    let (m, blocks) = diagonal_set_up();
    const TOLERANCE: f64 = 1e-14;
    let dense = to_dense(&m, &blocks);
    let mut random = NetRandom::new(1);
    let x: Vec<f64> = (0..dense.rows())
        .map(|_| 2.0 * random.next_double() - 1.0)
        .collect();
    let x = VectorXd::from_vec(x);
    let expected_y = &dense * &x;
    let mut actual_y = vec![0.0; dense.rows()];
    m.right_multiply_and_accumulate(x.as_slice(), &mut actual_y);
    assert!((&expected_y - &VectorXd::from_slice(&actual_y)).norm() <= TOLERANCE);
}

#[test]
fn rust_only_block_random_access_diagonal_matrix_invert() {
    let (mut m, blocks) = diagonal_set_up();
    const TOLERANCE: f64 = 1e-14;
    let dense = to_dense(&m, &blocks);
    let expected_inverse = Llt::new(&dense).solve_matrix(&MatrixXd::identity(dense.rows()));

    m.invert();
    let dense = to_dense(&m, &blocks);

    assert!(norm_of_difference(&expected_inverse, &dense) <= TOLERANCE);
}
