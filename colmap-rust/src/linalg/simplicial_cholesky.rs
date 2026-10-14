//! SimplicialCholesky: sparse Cholesky factorization P A P^T = L L^T or L D L^T of a
//! symmetric matrix in [`SparseMatrixCsc`] form, the replacement for `Eigen::SimplicialLLT`,
//! `Eigen::SimplicialLDLT` and CHOLMOD's supernodal LLT behind COLMAP's optim/sparse_cholesky
//! ([`crate::optim::SparseCholeskyWithFallbackSolver`]) and
//! optim/least_absolute_deviations. Neighbors: `sparse_matrix.rs` (storage) and
//! `amd_ordering.rs` (the fill-reducing permutation P). Port of colmap-sharp's
//! `ColmapSharp/LinearAlgebra/SimplicialCholesky.cs` (MIT, our own code).
//!
//! Algorithms, written from their published descriptions (no Eigen, CSparse, LDL or CHOLMOD
//! code was read or ported; those are MPL/LGPL/GPL, docs/LICENSE_AUDIT.md):
//! - Elimination tree: J. W. H. Liu, "The Role of Elimination Trees in Sparse
//!   Factorization", SIAM J. Matrix Anal. Appl. 11(1), 1990 (ancestor path compression).
//! - Row structure of L as the union of etree paths from the nonzeros of each column of the
//!   upper triangle, and the up-looking factorization that solves one sparse triangular
//!   system per row: Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006, ch. 4,
//!   used as a description only. Column counts come from walking the same row structures
//!   symbolically, which costs O(nnz(L)).
//!
//! `analyze_pattern` does the ordering and all symbolic work once; `factorize` only scatters
//! the new values through a precomputed map and runs the numeric pass, so repeated
//! factorizations of one pattern (every Levenberg-Marquardt or IRLS iteration) reuse it.
//!
//! Like Eigen's sparse Cholesky, only one triangle of the input is read (Lower by default).
//! Failure follows Eigen's documented `info()` contract: LLT reports `NumericalIssue` at the
//! first pivot <= 0; LDLT accepts negative pivots (indefinite matrices) and fails only on a
//! pivot that is exactly zero. Tier B: same math, different operation order than Eigen
//! (docs/CPP_DIVERGENCES.md, entry 33). Tests: `tests/linalg/rust_only_simplicial_cholesky.rs`.

use super::amd_ordering::amd_ordering_with_work;
use super::{ComputationInfo, SparseMatrixCsc, SymmetricPart, VectorXd};
use crate::math::fns;
use crate::util::check::{ColmapError, ErrorKind};
use crate::Result;

/// Which factorization a [`SimplicialCholesky`] computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SimplicialCholeskyKind {
    /// P A P^T = L L^T (`Eigen::SimplicialLLT`); needs a positive-definite matrix.
    Llt,
    /// P A P^T = L D L^T with unit L (`Eigen::SimplicialLDLT`); tolerates indefinite matrices.
    Ldlt,
}

/// Fill-reducing ordering applied before the factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SparseOrdering {
    /// Approximate minimum degree (Eigen's default `AMDOrdering`).
    Amd,
    /// No permutation (`Eigen::NaturalOrdering`).
    Natural,
}

/// Symbolic analysis of one pattern.
#[derive(Clone, Debug, Default)]
struct Symbolic {
    n: usize,
    pattern_col_ptr: Vec<usize>,
    pattern_row_idx: Vec<usize>,
    perm: Vec<usize>,      // new index -> original index
    c_col_ptr: Vec<usize>, // upper triangle of P A P^T
    c_row_idx: Vec<usize>,
    value_map: Vec<usize>, // input entry -> slot in c_values, or NOT_READ
    parent: Vec<usize>,    // elimination tree, NO_PARENT at a root
    l_col_ptr: Vec<usize>,
}

const NOT_READ: usize = usize::MAX;
const NO_PARENT: usize = usize::MAX;

/// Simplicial (column-by-column) sparse Cholesky with reusable symbolic analysis.
/// Replacement for `Eigen::SimplicialLLT` / `Eigen::SimplicialLDLT`.
#[derive(Clone, Debug)]
pub struct SimplicialCholesky {
    kind: SimplicialCholeskyKind,
    ordering: SparseOrdering,
    part: SymmetricPart,
    symbolic: Option<Symbolic>,
    c_values: Vec<f64>,
    // Strictly lower part of L, column by column. The row indices are written by the
    // numeric pass (each column fills in increasing row order), its size by the analysis.
    l_row_idx: Vec<usize>,
    l_values: Vec<f64>,
    diag: Vec<f64>, // L's diagonal (LLT) or D (LDLT)
    factorized: bool,
    info: ComputationInfo,
    ordering_work: u64,
    factorization_work: u64,
}

fn invalid(message: String) -> ColmapError {
    ColmapError::new(ErrorKind::InvalidArgument, message)
}

fn logic(message: &str) -> ColmapError {
    ColmapError::new(ErrorKind::LogicError, message)
}

impl SimplicialCholesky {
    /// Creates an unanalyzed solver.
    pub fn new(
        kind: SimplicialCholeskyKind,
        ordering: SparseOrdering,
        part: SymmetricPart,
    ) -> Self {
        Self {
            kind,
            ordering,
            part,
            symbolic: None,
            c_values: Vec::new(),
            l_row_idx: Vec::new(),
            l_values: Vec::new(),
            diag: Vec::new(),
            factorized: false,
            info: ComputationInfo::NumericalIssue,
            ordering_work: 0,
            factorization_work: 0,
        }
    }

    /// A solver of `kind` with AMD ordering reading the lower triangle (Eigen's defaults).
    pub fn with_kind(kind: SimplicialCholeskyKind) -> Self {
        Self::new(kind, SparseOrdering::Amd, SymmetricPart::Lower)
    }

    /// Result of the last factorization; `NumericalIssue` before any factorization.
    pub fn info(&self) -> ComputationInfo {
        self.info
    }

    /// The fill-reducing permutation: `permutation()[k]` is the original index of row/column
    /// k. Empty before `analyze_pattern`.
    pub fn permutation(&self) -> &[usize] {
        self.symbolic.as_ref().map_or(&[], |s| &s.perm)
    }

    /// Stored entries of L, diagonal included (one per column), from the symbolic analysis.
    pub fn non_zeros_l(&self) -> u64 {
        self.symbolic
            .as_ref()
            .map_or(0, |s| (s.l_col_ptr[s.n] + s.n) as u64)
    }

    /// Diagnostic for tests: list entries the AMD ordering visited in the last
    /// `analyze_pattern` (0 for the natural ordering). Load-independent, unlike wall-clock time.
    pub fn ordering_work(&self) -> u64 {
        self.ordering_work
    }

    /// Diagnostic for tests: inner-loop steps of the last numeric factorization, one per
    /// elimination-tree node visited while scattering a row and one per entry of L read while
    /// updating it. Load-independent, unlike wall-clock time.
    pub fn factorization_work(&self) -> u64 {
        self.factorization_work
    }

    /// Analyzes the pattern and factorizes (Eigen's `compute()`). Check `info()` for the
    /// outcome; `Err` only for a non-square matrix.
    pub fn compute(&mut self, a: &SparseMatrixCsc) -> Result<&mut Self> {
        self.analyze_pattern(a)?;
        self.factorize(a)?;
        Ok(self)
    }

    /// Orders the matrix and computes the structure of L. Only the pattern of the configured
    /// triangle of `a` is used; values are ignored.
    pub fn analyze_pattern(&mut self, a: &SparseMatrixCsc) -> Result<()> {
        if a.rows() != a.cols() {
            return Err(invalid(format!(
                "Sparse Cholesky needs a square matrix, got {}x{}.",
                a.rows(),
                a.cols()
            )));
        }
        let n = a.rows();
        self.factorized = false;
        self.info = ComputationInfo::NumericalIssue;

        let perm = match self.ordering {
            SparseOrdering::Amd => {
                let (perm, work) = amd_ordering_with_work(&a.triangular_part(self.part))?;
                self.ordering_work = work;
                perm
            }
            SparseOrdering::Natural => {
                self.ordering_work = 0;
                (0..n).collect()
            }
        };
        let mut inverse = vec![0usize; n];
        for (k, &p) in perm.iter().enumerate() {
            inverse[p] = k;
        }

        let mut s = Symbolic {
            n,
            pattern_col_ptr: a.col_ptr().to_vec(),
            pattern_row_idx: a.row_indices().to_vec(),
            perm,
            ..Symbolic::default()
        };
        self.build_permuted_upper(&mut s, a, &inverse);
        s.parent = elimination_tree(n, &s.c_col_ptr, &s.c_row_idx);
        build_structure_of_l(&mut s);
        let nnz_l = s.l_col_ptr[n];
        self.c_values = vec![0.0; s.c_row_idx.len()];
        self.l_row_idx = vec![0; nnz_l];
        self.l_values = vec![0.0; nnz_l];
        self.diag = vec![0.0; n];
        self.symbolic = Some(s);
        Ok(())
    }

    /// Numeric factorization of a matrix with the analyzed pattern. Returns `Ok(false)` (and
    /// sets `info()` to `NumericalIssue`) on a pivot the factorization kind cannot accept;
    /// `Err` before `analyze_pattern` or for a matrix with a different pattern.
    pub fn factorize(&mut self, a: &SparseMatrixCsc) -> Result<bool> {
        let Some(s) = self.symbolic.as_ref() else {
            return Err(logic("AnalyzePattern must run before Factorize."));
        };
        if a.rows() != s.n
            || a.cols() != s.n
            || a.col_ptr() != s.pattern_col_ptr.as_slice()
            || a.row_indices() != s.pattern_row_idx.as_slice()
        {
            return Err(invalid(
                "Factorize needs a matrix with the pattern given to AnalyzePattern.".into(),
            ));
        }

        self.c_values.fill(0.0);
        for (&slot, &value) in s.value_map.iter().zip(a.values()) {
            if slot != NOT_READ {
                self.c_values[slot] = value;
            }
        }

        self.factorization_work = 0;
        self.factorized = factorize_numeric(
            s,
            self.kind,
            &self.c_values,
            &mut self.l_row_idx,
            &mut self.l_values,
            &mut self.diag,
            &mut self.factorization_work,
        );
        self.info = if self.factorized {
            ComputationInfo::Success
        } else {
            ComputationInfo::NumericalIssue
        };
        Ok(self.factorized)
    }

    /// Solves A x = b with the current factorization. `Err` without a successful
    /// factorization or for a right-hand side of the wrong length.
    pub fn solve(&self, b: &VectorXd) -> Result<VectorXd> {
        let s = match (&self.symbolic, self.factorized) {
            (Some(s), true) => s,
            _ => return Err(logic("Solve needs a successful factorization.")),
        };
        let n = s.n;
        if b.len() != n {
            return Err(invalid(format!(
                "Right-hand side has length {}, expected {n}.",
                b.len()
            )));
        }
        let bs = b.as_slice();
        let mut y: Vec<f64> = s.perm.iter().map(|&p| bs[p]).collect();
        let llt = self.kind == SimplicialCholeskyKind::Llt;

        // Forward substitution with L (unit diagonal for LDLT), then D^-1 for LDLT.
        for j in 0..n {
            if llt {
                y[j] /= self.diag[j];
            }
            let yj = y[j];
            for p in s.l_col_ptr[j]..s.l_col_ptr[j + 1] {
                y[self.l_row_idx[p]] -= self.l_values[p] * yj;
            }
        }
        if !llt {
            for (yj, d) in y.iter_mut().zip(&self.diag) {
                *yj /= d;
            }
        }

        // Backward substitution with L^T.
        for j in (0..n).rev() {
            let mut yj = y[j];
            for p in s.l_col_ptr[j]..s.l_col_ptr[j + 1] {
                yj -= self.l_values[p] * y[self.l_row_idx[p]];
            }
            y[j] = if llt { yj / self.diag[j] } else { yj };
        }

        let mut x = vec![0.0; n];
        for (k, &p) in s.perm.iter().enumerate() {
            x[p] = y[k];
        }
        Ok(VectorXd::from_vec(x))
    }

    /// D of P A P^T = L D L^T (Eigen's `SimplicialLDLT::vectorD()`). LDLT only; needs a
    /// successful factorization.
    pub fn vector_d(&self) -> Result<VectorXd> {
        if !self.factorized || self.kind != SimplicialCholeskyKind::Ldlt {
            return Err(logic("VectorD needs a successful LDLT factorization."));
        }
        Ok(VectorXd::from_slice(&self.diag))
    }

    /// The factor L of the permuted matrix (Eigen's `matrixL()`), diagonal stored: ones for
    /// LDLT, L's diagonal for LLT. Needs a successful factorization.
    pub fn matrix_l(&self) -> Result<SparseMatrixCsc> {
        let s = match (&self.symbolic, self.factorized) {
            (Some(s), true) => s,
            _ => return Err(logic("MatrixL needs a successful factorization.")),
        };
        // Each column's strict-lower rows were appended in increasing row order by the
        // up-looking factorization, so putting the diagonal first keeps the rows sorted.
        let n = s.n;
        let mut col_ptr = vec![0usize; n + 1];
        let mut row_idx = Vec::with_capacity(self.l_row_idx.len() + n);
        let mut values = Vec::with_capacity(self.l_row_idx.len() + n);
        for j in 0..n {
            row_idx.push(j);
            values.push(if self.kind == SimplicialCholeskyKind::Llt {
                self.diag[j]
            } else {
                1.0
            });
            for p in s.l_col_ptr[j]..s.l_col_ptr[j + 1] {
                row_idx.push(self.l_row_idx[p]);
                values.push(self.l_values[p]);
            }
            col_ptr[j + 1] = row_idx.len();
        }
        SparseMatrixCsc::from_csc(n, n, &col_ptr, &row_idx, &values)
    }

    /// Builds the upper triangle of C = P A P^T in CSC form from the configured triangle of
    /// A, and the map from A's entries to C's value slots.
    fn build_permuted_upper(&self, s: &mut Symbolic, a: &SparseMatrixCsc, inverse: &[usize]) {
        let n = s.n;
        let (col_ptr, row_idx) = (a.col_ptr(), a.row_indices());
        let mut count = vec![0usize; n + 1];
        for j in 0..n {
            for &i in &row_idx[col_ptr[j]..col_ptr[j + 1]] {
                if self.part.contains(i, j) {
                    count[inverse[i].max(inverse[j]) + 1] += 1;
                }
            }
        }
        for k in 0..n {
            count[k + 1] += count[k];
        }
        s.c_col_ptr = count.clone();
        s.c_row_idx = vec![0; count[n]];
        s.value_map = vec![NOT_READ; a.non_zeros()];
        for j in 0..n {
            for (p, &i) in row_idx
                .iter()
                .enumerate()
                .take(col_ptr[j + 1])
                .skip(col_ptr[j])
            {
                if !self.part.contains(i, j) {
                    continue;
                }
                let (pi, pj) = (inverse[i], inverse[j]);
                let slot = count[pi.max(pj)];
                count[pi.max(pj)] += 1;
                s.c_row_idx[slot] = pi.min(pj);
                s.value_map[p] = slot;
            }
        }
    }
}

/// Elimination tree of a symmetric matrix given by its upper triangle (Liu 1990):
/// `parent[i]` is the row index of the first off-diagonal nonzero in column i of L, or
/// `NO_PARENT`.
fn elimination_tree(n: usize, col_ptr: &[usize], row_idx: &[usize]) -> Vec<usize> {
    let mut parent = vec![NO_PARENT; n];
    let mut ancestor = vec![NO_PARENT; n];
    for k in 0..n {
        for &row in &row_idx[col_ptr[k]..col_ptr[k + 1]] {
            // Climb from row to the root of its current subtree, pointing every node on the
            // way at k (path compression); that root's parent is k.
            let mut r = row;
            while r != NO_PARENT && r < k {
                let up = ancestor[r];
                ancestor[r] = k;
                if up == NO_PARENT {
                    parent[r] = k;
                }
                r = up;
            }
        }
    }
    parent
}

/// Column pointers of the strictly lower part of L: row k of L is the set of etree nodes
/// reachable upward from the nonzeros of column k of C, stopping at k.
fn build_structure_of_l(s: &mut Symbolic) {
    let n = s.n;
    let mut counts = vec![0usize; n + 1];
    let mut flag = vec![NO_PARENT; n];
    for k in 0..n {
        flag[k] = k;
        for &row in &s.c_row_idx[s.c_col_ptr[k]..s.c_col_ptr[k + 1]] {
            let mut i = row;
            while flag[i] != k {
                flag[i] = k;
                counts[i + 1] += 1;
                i = s.parent[i];
            }
        }
    }
    for k in 0..n {
        counts[k + 1] += counts[k];
    }
    s.l_col_ptr = counts;
}

/// Reach of column k of C in the elimination tree, in topological order (every node before
/// its ancestors), written to `pattern[top..n]`. Also scatters the column into y.
#[allow(clippy::too_many_arguments)]
fn scatter_row(
    s: &Symbolic,
    c_values: &[f64],
    k: usize,
    y: &mut [f64],
    flag: &mut [usize],
    pattern: &mut [usize],
    stack: &mut [usize],
    work: &mut u64,
) -> usize {
    let mut top = s.n;
    flag[k] = k;
    let range = s.c_col_ptr[k]..s.c_col_ptr[k + 1];
    for (&row, &value) in s.c_row_idx[range.clone()].iter().zip(&c_values[range]) {
        let mut i = row;
        y[i] += value;
        let mut len = 0;
        while flag[i] != k {
            *work += 1;
            stack[len] = i;
            len += 1;
            flag[i] = k;
            i = s.parent[i];
        }
        while len > 0 {
            len -= 1;
            top -= 1;
            pattern[top] = stack[len];
        }
    }
    top
}

/// Up-looking factorization: row k of L solves L(0:k,0:k) D y = C(0:k, k) (LDLT) or
/// L(0:k,0:k) l = C(0:k, k) (LLT). Returns false on an unacceptable pivot.
fn factorize_numeric(
    s: &Symbolic,
    kind: SimplicialCholeskyKind,
    c_values: &[f64],
    l_row_idx: &mut [usize],
    l_row_vals: &mut [f64],
    diag: &mut [f64],
    work: &mut u64,
) -> bool {
    let n = s.n;
    let mut y = vec![0.0; n];
    let mut flag = vec![NO_PARENT; n];
    let mut pattern = vec![0usize; n];
    let mut stack = vec![0usize; n];
    let mut fill = vec![0usize; n];
    for k in 0..n {
        let mut top = scatter_row(
            s,
            c_values,
            k,
            &mut y,
            &mut flag,
            &mut pattern,
            &mut stack,
            work,
        );
        let mut d = y[k];
        y[k] = 0.0;
        while top < n {
            let j = pattern[top];
            let end = s.l_col_ptr[j] + fill[j];
            *work += fill[j] as u64;
            let lkj = match kind {
                SimplicialCholeskyKind::Ldlt => {
                    let yj = y[j];
                    y[j] = 0.0;
                    for p in s.l_col_ptr[j]..end {
                        y[l_row_idx[p]] -= l_row_vals[p] * yj;
                    }
                    let lkj = yj / diag[j];
                    d -= lkj * yj;
                    lkj
                }
                SimplicialCholeskyKind::Llt => {
                    let lkj = y[j] / diag[j];
                    y[j] = 0.0;
                    for p in s.l_col_ptr[j]..end {
                        y[l_row_idx[p]] -= l_row_vals[p] * lkj;
                    }
                    d -= lkj * lkj;
                    lkj
                }
            };
            l_row_idx[end] = k;
            l_row_vals[end] = lkj;
            fill[j] += 1;
            top += 1;
        }
        match kind {
            SimplicialCholeskyKind::Ldlt => {
                if d == 0.0 {
                    return false;
                }
                diag[k] = d;
            }
            SimplicialCholeskyKind::Llt => {
                if d <= 0.0 {
                    return false;
                }
                diag[k] = fns::sqrt(d);
            }
        }
    }
    true
}
