//! Unit tests of the crate-private SIFT matching kernels ([`super::match_kernels`]): the fused
//! best / second-best scan against COLMAP's plain scan of the full distance matrix and of its
//! transpose (`FindBestMatchesOneWay*`'s loops), with ties and a guided filter.

use super::match_kernels::{
    scan_distances, DistanceType, Top2, L2_SENTINEL, SQ_SIFT_DESCRIPTOR_NORM,
};
use crate::linalg::RowMajorMatrix;
use crate::math::random::Mt19937;
use crate::util::cancellation::CancelToken;

/// Descriptors with few distinct values, so equal distances (ties) are common.
fn descriptors(rows: usize, seed: u32) -> RowMajorMatrix<u8> {
    let mut g = Mt19937::new(seed);
    let values = (0..rows * 128)
        .map(|_| (g.next_u32() % 3) as u8 * 100)
        .collect();
    RowMajorMatrix::from_vec(rows, 128, values)
}

/// COLMAP's scan of one row of an explicit key matrix.
fn plain_scan(keys: &[i32], sentinel: i32) -> Top2 {
    let mut state = Top2::init(sentinel);
    for (i, &k) in keys.iter().enumerate() {
        state.update(k, i as i32);
    }
    state
}

#[test]
fn rust_only_fused_scan_matches_the_plain_matrix_scan() {
    let d1 = descriptors(13, 1);
    let d2 = descriptors(9, 2);
    let filter = |i1: usize, i2: usize| (i1 + 2 * i2).is_multiple_of(5);
    for distance_type in [DistanceType::DotProduct, DistanceType::L2] {
        for guided in [false, true] {
            let use_l2 = distance_type == DistanceType::L2;
            let sentinel = if use_l2 { L2_SENTINEL } else { 0 };
            // The explicit matrix of ComputeSiftDistanceMatrix, as keys.
            let mut matrix = vec![vec![0i32; d2.rows()]; d1.rows()];
            for (i1, row) in matrix.iter_mut().enumerate() {
                for (i2, key) in row.iter_mut().enumerate() {
                    let a = d1.row(i1);
                    let b = d2.row(i2);
                    *key = if guided && filter(i1, i2) {
                        if use_l2 {
                            -SQ_SIFT_DESCRIPTOR_NORM
                        } else {
                            0
                        }
                    } else if use_l2 {
                        -a.iter()
                            .zip(b)
                            .map(|(&x, &y)| (i32::from(x) - i32::from(y)).pow(2))
                            .sum::<i32>()
                    } else {
                        a.iter()
                            .zip(b)
                            .map(|(&x, &y)| i32::from(x) * i32::from(y))
                            .sum()
                    };
                }
            }

            let mut rows = vec![Top2::init(0); d1.rows()];
            let mut cols = vec![Top2::init(0); d2.rows()];
            let guided_filter: Option<&dyn Fn(usize, usize) -> bool> =
                if guided { Some(&filter) } else { None };
            scan_distances(
                &d1,
                &d2,
                distance_type,
                guided_filter,
                &mut rows,
                Some(&mut cols),
                &CancelToken::new(),
            )
            .unwrap();

            for (i1, state) in rows.iter().enumerate() {
                assert_eq!(*state, plain_scan(&matrix[i1], sentinel));
            }
            for (i2, state) in cols.iter().enumerate() {
                let column: Vec<i32> = matrix.iter().map(|row| row[i2]).collect();
                assert_eq!(*state, plain_scan(&column, sentinel));
            }
        }
    }
}
