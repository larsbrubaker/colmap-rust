//! Port of COLMAP's `colmap/optim/support_measurement.h/.cc`: how RANSAC scores a model from
//! its residuals — [`InlierSupportMeasurer`] (most inliers, then smallest inlier residual
//! sum), [`UniqueInlierSupportMeasurer`] (most distinct inlier ids first) and
//! [`MEstimatorSupportMeasurer`] (MSAC's truncated residual sum). Consumers:
//! [`super::ransac`] and [`super::loransac`]. Port of colmap-sharp's
//! `Optim/SupportMeasurement.cs`. Tests: `tests/optim/support_measurement.rs`
//! (`support_measurement_test.cc` 1:1).
//!
//! Tier A (exact): sums in residual order, as COLMAP's loops.
//!
//! Translation notes:
//! - Each C++ `Measurer::Support` nested struct is its own type ([`InlierSupport`], ...),
//!   named by the measurer's associated type [`SupportMeasurer::Support`]. RANSAC reads
//!   `support.num_inliers` generically, so the supports implement [`MeasuredSupport`].
//! - `Default` gives COLMAP's member initializers (`residual_sum`/`score` start at
//!   `std::numeric_limits<double>::max()`, so any evaluated support beats a default one).
//! - `UniqueInlierSupportMeasurer` counts distinct ids in a `HashSet`; only its size is read,
//!   so the set's iteration order never matters.

use crate::{check_eq, Result};
use std::collections::HashSet;

/// What RANSAC needs from any support: the inlier count that drives the dynamic number of
/// trials and the final success test.
pub trait MeasuredSupport: Clone + Default {
    /// The number of inliers (`Support::num_inliers`).
    fn num_inliers(&self) -> usize;
}

/// COLMAP's SupportMeasurer concept (`Evaluate`, `IsLeftBetter`, nested `Support`).
pub trait SupportMeasurer: Clone {
    /// The measurer's `Support` struct.
    type Support: MeasuredSupport;

    /// Compute the support of the residuals.
    fn evaluate(&mut self, residuals: &[f64], max_residual: f64) -> Result<Self::Support>;

    /// Compare the two supports.
    fn is_left_better(&self, left: &Self::Support, right: &Self::Support) -> bool;
}

/// Port of `InlierSupportMeasurer::Support`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InlierSupport {
    /// The number of inliers.
    pub num_inliers: usize,
    /// The sum of all inlier residuals.
    pub residual_sum: f64,
}

impl Default for InlierSupport {
    fn default() -> Self {
        Self {
            num_inliers: 0,
            residual_sum: f64::MAX,
        }
    }
}

impl MeasuredSupport for InlierSupport {
    fn num_inliers(&self) -> usize {
        self.num_inliers
    }
}

/// Port of `colmap::InlierSupportMeasurer`: measure the support of a model by counting the
/// number of inliers and summing all inlier residuals. The support is better if it has more
/// inliers and a smaller residual sum.
#[derive(Clone, Copy, Debug, Default)]
pub struct InlierSupportMeasurer;

impl SupportMeasurer for InlierSupportMeasurer {
    type Support = InlierSupport;

    fn evaluate(&mut self, residuals: &[f64], max_residual: f64) -> Result<InlierSupport> {
        let mut support = InlierSupport {
            num_inliers: 0,
            residual_sum: 0.0,
        };
        for &residual in residuals {
            if residual <= max_residual {
                support.num_inliers += 1;
                support.residual_sum += residual;
            }
        }
        Ok(support)
    }

    fn is_left_better(&self, left: &InlierSupport, right: &InlierSupport) -> bool {
        if left.num_inliers > right.num_inliers {
            true
        } else {
            left.num_inliers == right.num_inliers && left.residual_sum < right.residual_sum
        }
    }
}

/// Port of `UniqueInlierSupportMeasurer::Support`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct UniqueInlierSupport {
    /// The number of unique inliers.
    pub num_unique_inliers: usize,
    /// The number of inliers. This is still needed for determining the dynamic number of
    /// iterations.
    pub num_inliers: usize,
    /// The sum of all inlier residuals.
    pub residual_sum: f64,
}

impl Default for UniqueInlierSupport {
    fn default() -> Self {
        Self {
            num_unique_inliers: 0,
            num_inliers: 0,
            residual_sum: f64::MAX,
        }
    }
}

impl MeasuredSupport for UniqueInlierSupport {
    fn num_inliers(&self) -> usize {
        self.num_inliers
    }
}

/// Port of `colmap::UniqueInlierSupportMeasurer`: measure the support of a model by counting
/// the number of unique inliers (e.g., visible 3D points), the number of inliers, and
/// summing all inlier residuals. Each sample carries a unique id (`unique_sample_ids`, one
/// per residual). The support is better if it has more unique inliers, more inliers, and a
/// smaller residual sum.
#[derive(Clone, Debug)]
pub struct UniqueInlierSupportMeasurer {
    unique_sample_ids: Vec<usize>,
}

impl UniqueInlierSupportMeasurer {
    /// Port of `UniqueInlierSupportMeasurer(std::vector<size_t> unique_sample_ids)`.
    pub fn new(unique_sample_ids: Vec<usize>) -> Self {
        Self { unique_sample_ids }
    }
}

impl SupportMeasurer for UniqueInlierSupportMeasurer {
    type Support = UniqueInlierSupport;

    fn evaluate(&mut self, residuals: &[f64], max_residual: f64) -> Result<UniqueInlierSupport> {
        check_eq!(residuals.len(), self.unique_sample_ids.len());
        let mut support = UniqueInlierSupport {
            num_unique_inliers: 0,
            num_inliers: 0,
            residual_sum: 0.0,
        };
        // Only the set's size is read (see the file header).
        let mut inlier_point_ids = HashSet::new();
        for (idx, &residual) in residuals.iter().enumerate() {
            if residual <= max_residual {
                support.num_inliers += 1;
                inlier_point_ids.insert(self.unique_sample_ids[idx]);
                support.residual_sum += residual;
            }
        }
        support.num_unique_inliers = inlier_point_ids.len();
        Ok(support)
    }

    fn is_left_better(&self, left: &UniqueInlierSupport, right: &UniqueInlierSupport) -> bool {
        if left.num_unique_inliers > right.num_unique_inliers {
            true
        } else if left.num_unique_inliers == right.num_unique_inliers {
            if left.num_inliers > right.num_inliers {
                true
            } else {
                left.num_inliers == right.num_inliers && left.residual_sum < right.residual_sum
            }
        } else {
            false
        }
    }
}

/// Port of `MEstimatorSupportMeasurer::Support`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MEstimatorSupport {
    /// The number of inliers.
    pub num_inliers: usize,
    /// The MSAC score, defined as the truncated sum of residuals.
    pub score: f64,
}

impl Default for MEstimatorSupport {
    fn default() -> Self {
        Self {
            num_inliers: 0,
            score: f64::MAX,
        }
    }
}

impl MeasuredSupport for MEstimatorSupport {
    fn num_inliers(&self) -> usize {
        self.num_inliers
    }
}

/// Port of `colmap::MEstimatorSupportMeasurer`: measure the support of a model by its
/// fitness to the data as used in MSAC. A support is better if it has a smaller MSAC score.
#[derive(Clone, Copy, Debug, Default)]
pub struct MEstimatorSupportMeasurer;

impl SupportMeasurer for MEstimatorSupportMeasurer {
    type Support = MEstimatorSupport;

    fn evaluate(&mut self, residuals: &[f64], max_residual: f64) -> Result<MEstimatorSupport> {
        let mut support = MEstimatorSupport {
            num_inliers: 0,
            score: 0.0,
        };
        for &residual in residuals {
            if residual <= max_residual {
                support.num_inliers += 1;
                support.score += residual;
            } else {
                support.score += max_residual;
            }
        }
        Ok(support)
    }

    fn is_left_better(&self, left: &MEstimatorSupport, right: &MEstimatorSupport) -> bool {
        left.score < right.score
    }
}
