// Fixed inner-dimension kernels for the shapes bundle adjustment feeds the Schur eliminator
// through `small_blas.rs`: residual blocks of 2 rows, E blocks (points) of size 3. Port of
// colmap-sharp's `ColmapSharp/Solver/SmallBlasFixed.cs` (our MIT code; the counterpart of
// Ceres Solver 2.2.0's fixed-size `small_blas.h` template instantiations, BSD-3-Clause, see
// THIRD_PARTY_NOTICES.md).
//
// colmap-sharp hand-wrote one kernel per size (2 and 3); here the inner dimension is a const
// generic `K`, so the compiler unrolls the k loop. Each element is still
// 0.0 + a_0 b_0 + a_1 b_1 (+ a_2 b_2) summed in that order into a temporary and then stored
// with the operation (Rust never reassociates or contracts floating-point sums), so the
// results are bit-identical to the naive loops of `small_blas.rs` (pinned by
// `tests/solver/rust_only_small_blas_fixed.rs`). Ceres' own fixed-size path uses Eigen's
// kernels, whose order may differ (docs/CPP_DIVERGENCES.md, entry 300), so matching the naive
// loops is the contract.
//
// The rows of B and of C are sliced once per call / per output row, so the inner loops index
// within slices whose lengths the compiler knows. A shorter operand panics on that slicing,
// as colmap-sharp's length check throws.

use super::small_blas::Operation;

/// `C op= A' B` for A (`K x cols_a`) and B (`K x cols_b`); C(row, col) is at
/// `index0 + row * stride + col`.
#[allow(clippy::too_many_arguments)]
pub fn matrix_transpose_matrix_multiply_rows<const K: usize>(
    a: &[f64],
    cols_a: usize,
    b: &[f64],
    cols_b: usize,
    c: &mut [f64],
    index0: usize,
    stride: usize,
    operation: Operation,
) {
    if cols_a == 0 || cols_b == 0 {
        return;
    }

    let a = &a[..K * cols_a];
    let b_rows: [&[f64]; K] = std::array::from_fn(|k| &b[k * cols_b..(k + 1) * cols_b]);
    for row in 0..cols_a {
        let a_k: [f64; K] = std::array::from_fn(|k| a[k * cols_a + row]);
        let start = index0 + row * stride;
        for (col, out) in c[start..start + cols_b].iter_mut().enumerate() {
            let mut tmp = 0.0;
            for k in 0..K {
                tmp += a_k[k] * b_rows[k][col];
            }

            operation.store(out, tmp);
        }
    }
}

/// `C op= A B` for A (`rows_a x K`) and B (`K x cols_b`); C(row, col) is at
/// `index0 + row * stride + col`.
#[allow(clippy::too_many_arguments)]
pub fn matrix_matrix_multiply_inner<const K: usize>(
    a: &[f64],
    rows_a: usize,
    b: &[f64],
    cols_b: usize,
    c: &mut [f64],
    index0: usize,
    stride: usize,
    operation: Operation,
) {
    if rows_a == 0 || cols_b == 0 {
        return;
    }

    let a = &a[..K * rows_a];
    let b_rows: [&[f64]; K] = std::array::from_fn(|k| &b[k * cols_b..(k + 1) * cols_b]);
    for row in 0..rows_a {
        let a_k: [f64; K] = std::array::from_fn(|k| a[K * row + k]);
        let start = index0 + row * stride;
        for (col, out) in c[start..start + cols_b].iter_mut().enumerate() {
            let mut tmp = 0.0;
            for k in 0..K {
                tmp += a_k[k] * b_rows[k][col];
            }

            operation.store(out, tmp);
        }
    }
}
