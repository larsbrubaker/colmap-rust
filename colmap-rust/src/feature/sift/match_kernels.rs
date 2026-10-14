//! The descriptor arithmetic behind the SIFT CPU matcher of `colmap/feature/sift.cc`:
//! `ComputeSiftDistanceMatrix` (uint8 descriptors as ints, dot product or squared L2) fused
//! with the best / second-best scan of `FindBestMatchesOneWayBruteForce` and
//! `FindBestMatchesOneWayIndex`, in both directions at once. Port of colmap-sharp's
//! `Feature/SiftMatchKernels.cs`. Neighbors: [`super::cpu_matcher`] (the ratio / distance
//! tests and cross-check on the scan results) and [`crate::feature::index`] (exact k-NN
//! search, which shares [`dot_product`]). Tests: `tests/feature/sift_matching.rs`
//! (`sift_test.cc`) and `src/feature/sift/tests.rs` (the fused scan against the plain matrix
//! scan).
//!
//! Tier A (exact). All distances are integers: a SIFT dot product is at most
//! 128 * 255 * 255 < 2^24, so COLMAP's conversion to float is exact and monotone, and
//! comparing the ints decides every comparison exactly as COLMAP's floats do.
//!
//! Why fused instead of COLMAP's full N x M matrix: an 8k x 8k float matrix is 256 MB, too much
//! for wasm. COLMAP's scan keeps, per row (and per column for the reverse direction), the best
//! value, the first index reaching it, and the second best of the multiset seen so far (a tie
//! with the best becomes the second best). Updating the column states in ascending row order
//! reproduces the scan of the transposed matrix exactly.

use super::SIFT_DESCRIPTOR_DIM;
use crate::linalg::RowMajorMatrix;
use crate::util::cancellation::CancelToken;
use crate::util::check::Result;

/// `kSiftDescriptorDim`, as the row width of the byte kernels.
pub(crate) const SIFT_DIM: usize = SIFT_DESCRIPTOR_DIM;

/// `kSqSiftDescriptorNorm`: SIFT descriptors are normalized to length 512 (w/ quantization
/// errors).
pub(crate) const SQ_SIFT_DESCRIPTOR_NORM: i32 = 512 * 512;

/// Best / second-best state of one row or column of a distance matrix, scanned with COLMAP's
/// strict "greater than" updates on a key (the dot product, or the negated squared L2
/// distance). `best_idx` is -1 when no value beat the initial sentinel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Top2 {
    pub best: i32,
    pub second: i32,
    pub best_idx: i32,
}

impl Top2 {
    /// A state with nothing seen: both values at `sentinel`.
    pub fn init(sentinel: i32) -> Self {
        Self {
            best: sentinel,
            second: sentinel,
            best_idx: -1,
        }
    }

    /// COLMAP's per-element update of the best / second-best scan.
    #[inline]
    pub fn update(&mut self, key: i32, idx: i32) {
        if key > self.best {
            self.best_idx = idx;
            self.second = self.best;
            self.best = key;
        } else if key > self.second {
            self.second = key;
        }
    }
}

/// Exact integer dot product of two 128-byte descriptors (`row(i1).dot(row(i2))` on the int
/// casts). Integer arithmetic, so the order of the sum cannot change the result.
#[inline]
pub(crate) fn dot_product(a: &[u8], b: &[u8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| i32::from(x) * i32::from(y))
        .sum()
}

/// Squared L2 norm of each 128-byte row.
pub(crate) fn squared_norms(descriptors: &RowMajorMatrix<u8>) -> Vec<i32> {
    (0..descriptors.rows())
        .map(|i| {
            let row = descriptors.row(i);
            dot_product(row, row)
        })
        .collect()
}

/// Which distance [`scan_distances`] scans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DistanceType {
    /// `DistanceType::DOT_PRODUCT`: key = dot, sentinel 0, as
    /// `FindBestMatchesOneWayBruteForce`.
    DotProduct,
    /// `DistanceType::L2`: key = -l2, sentinel standing for `FLT_MAX`, as
    /// `FindBestMatchesOneWayIndex`.
    L2,
}

/// The sentinel key of the L2 scan, standing for `FLT_MAX`: every real distance (at most
/// 128 * 255^2) is smaller.
pub(crate) const L2_SENTINEL: i32 = -i32::MAX;

/// The fused scan of the N x M distance matrix between `d1` and `d2` (`rows` gets N states,
/// `cols`, when given, M). `guided_filter(i1, i2)` true rejects a pair, which then gets the
/// worst distance (dot 0, L2 `kSqSiftDescriptorNorm`) like `ComputeSiftDistanceMatrix`.
/// `cancel` is polled once per row of `d1`; a cancelled scan returns `Err` with the states
/// partly written.
pub(crate) fn scan_distances(
    d1: &RowMajorMatrix<u8>,
    d2: &RowMajorMatrix<u8>,
    distance_type: DistanceType,
    guided_filter: Option<&dyn Fn(usize, usize) -> bool>,
    rows: &mut [Top2],
    mut cols: Option<&mut [Top2]>,
    cancel: &CancelToken,
) -> Result<()> {
    let use_l2 = distance_type == DistanceType::L2;
    let norms1 = if use_l2 {
        squared_norms(d1)
    } else {
        Vec::new()
    };
    let norms2 = if use_l2 {
        squared_norms(d2)
    } else {
        Vec::new()
    };
    let sentinel = if use_l2 { L2_SENTINEL } else { 0 };
    let rejected_key = if use_l2 { -SQ_SIFT_DESCRIPTOR_NORM } else { 0 };

    if let Some(cols) = cols.as_deref_mut() {
        cols.fill(Top2::init(sentinel));
    }
    for (i1, row_state) in rows.iter_mut().enumerate().take(d1.rows()) {
        cancel.check_cancelled()?;
        *row_state = Top2::init(sentinel);
        let a = d1.row(i1);
        for i2 in 0..d2.rows() {
            let key = if guided_filter.is_some_and(|filter| filter(i1, i2)) {
                rejected_key
            } else {
                let dot = dot_product(a, d2.row(i2));
                if use_l2 {
                    -(norms1[i1] + norms2[i2] - 2 * dot)
                } else {
                    dot
                }
            };
            // Descriptor counts are far below i32::MAX (COLMAP indexes them with int).
            row_state.update(key, i2 as i32);
            if let Some(cols) = cols.as_deref_mut() {
                cols[i2].update(key, i1 as i32);
            }
        }
    }
    Ok(())
}
