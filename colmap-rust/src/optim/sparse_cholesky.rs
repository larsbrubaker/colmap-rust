//! Port of COLMAP's `colmap/optim/sparse_cholesky.h/.cc`: [`SparseCholeskyWithFallbackSolver`]
//! tries an LL^T factorization first and falls back to LDL^T once LL^T reports the matrix is
//! not positive definite. Used by [`super::least_absolute_deviations`] (and later rotation
//! averaging). Port of colmap-sharp's `Optim/SparseCholesky.cs`. Tests:
//! `tests/optim/sparse_cholesky.rs` (`sparse_cholesky_test.cc` 1:1).
//!
//! COLMAP's first stage is CHOLMOD's supernodal LLT (GPL, not ported); here it is the
//! simplicial LLT of [`SimplicialCholesky`] with AMD ordering, which accepts and rejects the
//! same matrices (both stop at the first pivot <= 0) but groups the floating-point work
//! differently. The fallback stage is `Eigen::SimplicialLDLT` in COLMAP and
//! [`SimplicialCholesky`]'s LDLT here. Tier B (docs/CPP_DIVERGENCES.md, entry 33).
//!
//! Translation notes:
//! - `Solve`'s `Eigen::VectorXd*` out-parameter becomes a returned `Option`: `None` where
//!   COLMAP returns false (after a failed factorization Eigen's `solve()` has no defined
//!   result).
//! - COLMAP's `LOG(WARNING)` on fallback is not emitted (the core has no logger); the
//!   fallback is visible through [`SparseCholeskyWithFallbackSolver::uses_ldlt`].
//! - Once fallen back, a new pattern re-analyzes the LDLT stage too (COLMAP's stale-analysis
//!   bug, docs/CPP_DIVERGENCES.md entry 33).

use crate::linalg::{
    ComputationInfo, SimplicialCholesky, SimplicialCholeskyKind, SparseMatrixCsc, VectorXd,
};
use crate::Result;

/// Sparse Cholesky solver that tries LLT first (fastest) and falls back to the numerically
/// more tolerant LDLT once LLT reports the matrix is not positive definite. Port of
/// `colmap::SparseCholeskyWithFallbackSolver`.
#[derive(Clone, Debug)]
pub struct SparseCholeskyWithFallbackSolver {
    llt: SimplicialCholesky,
    ldlt: SimplicialCholesky,
    use_ldlt: bool,
}

impl Default for SparseCholeskyWithFallbackSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseCholeskyWithFallbackSolver {
    /// A solver that has analyzed nothing yet.
    pub fn new() -> Self {
        Self {
            llt: SimplicialCholesky::with_kind(SimplicialCholeskyKind::Llt),
            ldlt: SimplicialCholesky::with_kind(SimplicialCholeskyKind::Ldlt),
            use_ldlt: false,
        }
    }

    /// True once LLT failed and the solver switched to LDLT for good.
    pub fn uses_ldlt(&self) -> bool {
        self.use_ldlt
    }

    /// One-shot factorization (analyze + factorize). `Err` only for a non-square matrix.
    pub fn compute(&mut self, a: &SparseMatrixCsc) -> Result<bool> {
        self.analyze_pattern(a)?;
        self.factorize(a)
    }

    /// For iterative reuse with matrices of identical sparsity but changing values, call
    /// `analyze_pattern` once and then `factorize` per iteration.
    pub fn analyze_pattern(&mut self, a: &SparseMatrixCsc) -> Result<()> {
        self.llt.analyze_pattern(a)?;
        // LDLT pattern is analyzed lazily on first fallback, since the common case never
        // needs it. Once the solver has fallen back it stays on LDLT, so a new pattern must
        // reach the LDLT stage here (COLMAP only re-analyzes the supernodal stage).
        if self.use_ldlt {
            self.ldlt.analyze_pattern(a)?;
        }
        Ok(())
    }

    /// Numeric factorization; `Ok(false)` when both stages fail.
    pub fn factorize(&mut self, a: &SparseMatrixCsc) -> Result<bool> {
        if !self.use_ldlt {
            if self.llt.factorize(a)? {
                return Ok(true);
            }
            self.ldlt.analyze_pattern(a)?;
            self.use_ldlt = true;
        }
        self.ldlt.factorize(a)
    }

    /// Solves A x = b with the current factorization; `None` when it failed.
    pub fn solve(&self, b: &VectorXd) -> Option<VectorXd> {
        let active = if self.use_ldlt { &self.ldlt } else { &self.llt };
        if active.info() != ComputationInfo::Success {
            return None;
        }
        active.solve(b).ok()
    }
}
