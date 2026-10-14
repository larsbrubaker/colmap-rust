// Rust-only (COLMAP has no test for Eigen's sparse module): SparseMatrixCsc in
// colmap_rust::linalg. Port of colmap-sharp's `ColmapSharp.Tests/LinearAlgebra/
// SparseMatrixTests.cs`. Every operation is checked against the same operation on the dense
// MatrixXd built from the same entries, plus the CSC invariants (sorted, unique rows per
// column) and Eigen's setFromTriplets duplicate summing. Random inputs come from .NET's
// seeded `System.Random` (`tests/support/net_random.rs`), so they are colmap-sharp's.

use colmap_rust::linalg::{MatrixXd, SparseMatrixCsc, SparseTriplet, SymmetricPart, VectorXd};

use crate::net_random::NetRandom;

fn t(row: usize, col: usize, value: f64) -> SparseTriplet {
    SparseTriplet::new(row, col, value)
}

pub fn random_sparse(
    random: &mut NetRandom,
    rows: usize,
    cols: usize,
    density: f64,
) -> SparseMatrixCsc {
    let mut triplets = Vec::new();
    for j in 0..cols {
        for i in 0..rows {
            if random.next_double() < density {
                triplets.push(t(i, j, random.next_double() * 2.0 - 1.0));
            }
        }
    }
    SparseMatrixCsc::from_triplets(rows, cols, &triplets).unwrap()
}

fn max_abs_diff(a: &MatrixXd, b: &MatrixXd) -> f64 {
    let mut max = 0.0_f64;
    for (x, y) in a.as_slice().iter().zip(b.as_slice()) {
        max = max.max((x - y).abs());
    }
    max
}

fn assert_csc_invariants(m: &SparseMatrixCsc) {
    let (col_ptr, rows) = (m.col_ptr(), m.row_indices());
    assert_eq!(col_ptr[0], 0);
    assert_eq!(col_ptr[m.cols()], m.non_zeros());
    for j in 0..m.cols() {
        for p in col_ptr[j] + 1..col_ptr[j + 1] {
            assert!(rows[p - 1] < rows[p]);
        }
    }
}

#[test]
fn rust_only_from_triplets_sums_duplicates_and_sorts_rows() {
    let m = SparseMatrixCsc::from_triplets(
        3,
        2,
        &[
            t(2, 0, 1.0),
            t(0, 0, 2.0),
            t(2, 0, 3.0),
            t(1, 1, 4.0),
            t(0, 1, 0.0),
        ],
    )
    .unwrap();
    assert_eq!(m.non_zeros(), 4);
    assert_eq!(m.col_ptr(), &[0, 2, 4]);
    assert_eq!(m.row_indices(), &[0, 2, 0, 1]);
    assert_eq!(m.values(), &[2.0, 4.0, 0.0, 4.0]);
    assert_eq!(m.coeff(2, 0), 4.0);
    assert_eq!(m.coeff(1, 0), 0.0);
    assert!(SparseMatrixCsc::from_triplets(2, 2, &[t(2, 0, 1.0)]).is_err());
}

#[test]
fn rust_only_transpose_and_products_match_dense() {
    let mut random = NetRandom::new(7);
    let a = random_sparse(&mut random, 13, 9, 0.3);
    let b = random_sparse(&mut random, 9, 11, 0.3);
    let (da, db) = (a.to_dense(), b.to_dense());
    let x = VectorXd::from_vec((0..9).map(|_| random.next_double()).collect());
    let y = VectorXd::from_vec((0..13).map(|_| random.next_double()).collect());

    let at = a.transpose();
    let ab = &a * &b;
    let ata = a.transpose_times_self();
    assert_csc_invariants(&at);
    assert_csc_invariants(&ab);
    assert_csc_invariants(&ata);
    assert_eq!(max_abs_diff(&at.to_dense(), &da.transpose()), 0.0);
    assert!(max_abs_diff(&ab.to_dense(), &(&da * &db)) < 1e-14);
    assert!(max_abs_diff(&ata.to_dense(), &(&da.transpose() * &da)) < 1e-14);
    assert!((&(&a * &x) - &(&da * &x)).max_abs() < 1e-14);
    assert!((&a.transpose_multiply(&y) - &(&da.transpose() * &y)).max_abs() < 1e-14);
}

#[test]
fn rust_only_triangular_part_keeps_one_triangle_with_diagonal() {
    let mut random = NetRandom::new(3);
    let a = random_sparse(&mut random, 7, 5, 0.5).transpose_times_self();
    let lower = a.transpose_times_self_part(SymmetricPart::Lower).to_dense();
    let full = a.transpose_times_self().to_dense();
    let upper = a.triangular_part(SymmetricPart::Upper).to_dense();
    for j in 0..5 {
        for i in 0..5 {
            assert_eq!(lower[(i, j)], if i >= j { full[(i, j)] } else { 0.0 });
            assert_eq!(upper[(i, j)], if i <= j { a.coeff(i, j) } else { 0.0 });
        }
    }
}

#[test]
fn rust_only_coeff_ref_and_add_to_diagonal() {
    let mut m = SparseMatrixCsc::from_triplets(3, 3, &[t(0, 0, 1.0), t(2, 1, 5.0)]).unwrap();
    *m.coeff_ref(2, 1).unwrap() += 1.0;
    let shifted = m.add_to_diagonal(0.5);
    assert_eq!(m.coeff(2, 1), 6.0);
    assert!(m.coeff_ref(1, 1).is_err());
    assert_eq!(shifted.non_zeros(), 4);
    assert_eq!(shifted.coeff(0, 0), 1.5);
    assert_eq!(shifted.coeff(1, 1), 0.5);
    assert_eq!(shifted.coeff(2, 2), 0.5);
    assert_eq!(shifted.coeff(2, 1), 6.0);
}

#[test]
fn rust_only_block_keeps_entries_inside_with_row_and_column_offset() {
    // 4x4 with entries in and around the 2x2 block at (1, 2).
    let a = SparseMatrixCsc::from_triplets(
        4,
        4,
        &[
            t(0, 2, 1.0),
            t(1, 2, 2.0),
            t(2, 3, 3.0),
            t(3, 3, 4.0),
            t(1, 1, 5.0),
            t(2, 2, 6.0),
        ],
    )
    .unwrap();
    let block = a.block(1, 2, 2, 2).unwrap();
    assert_eq!(block.rows(), 2);
    assert_eq!(block.cols(), 2);
    assert_eq!(block.non_zeros(), 3);
    assert!(block
        .to_dense()
        .is_approx_with(&a.to_dense().block(1, 2, 2, 2), 0.0));
    assert_eq!(block.coeff(0, 0), 2.0);
    assert_eq!(block.coeff(1, 0), 6.0);
    assert_eq!(block.coeff(1, 1), 3.0);
}

#[test]
fn rust_only_subtract_union_of_non_overlapping_patterns() {
    let a = SparseMatrixCsc::from_triplets(3, 2, &[t(0, 0, 1.0), t(2, 1, 2.0)]).unwrap();
    let b = SparseMatrixCsc::from_triplets(3, 2, &[t(1, 0, 3.0), t(0, 1, 4.0)]).unwrap();
    let difference = &a - &b;
    assert_eq!(difference.non_zeros(), 4);
    assert_eq!(difference.row_indices(), &[0, 1, 0, 2]);
    assert!(difference
        .to_dense()
        .is_approx_with(&(&a.to_dense() - &b.to_dense()), 0.0));
    assert_eq!(difference.coeff(1, 0), -3.0);
    assert_eq!(difference.coeff(0, 1), -4.0);
}
