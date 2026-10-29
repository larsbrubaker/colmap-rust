//! Port of COLMAP's `colmap/optim/least_absolute_deviations.h/.cc`: solves
//! `min ||A x - b||_1` by ADMM over the factored normal equations A^T A; rotation averaging
//! and global positioning build on it. The normal equations are factored by
//! [`SimplicialCholesky`] (`Eigen::SimplicialLLT`) or by [`SparseCholeskyWithFallbackSolver`]
//! (CHOLMOD's supernodal LLT in COLMAP). Port of colmap-sharp's
//! `Optim/LeastAbsoluteDeviations.cs`. Tests: `tests/optim/least_absolute_deviations.rs`
//! (`least_absolute_deviations_test.cc` 1:1).
//!
//! Tier C: an iterative solver over a Tier B factorization (see the divergence in
//! `optim/sparse_cholesky.rs`); COLMAP's tests check the converged solution within tolerances.
//!
//! Translation notes:
//! - C++ keeps references to the caller's options and A; this port owns a copy of both. The
//!   one option COLMAP's callers change after construction, `max_num_iterations` (rotation
//!   averaging's L1 phase doubles it), is settable through
//!   [`LeastAbsoluteDeviationSolver::set_max_num_iterations`].
//! - `std::runtime_error` (underdetermined system) is an `Err` of kind `RuntimeError`.
//! - `Solve` overwrites the caller's x (the `Eigen::VectorXd*` out-parameter), resizing it
//!   as Eigen's assignment does.
//! - `SolverType::SupernodalCholmodLLT` keeps COLMAP's name but selects
//!   [`SparseCholeskyWithFallbackSolver`] (CHOLMOD is not ported).
//! - COLMAP's `LOG(WARNING)` on a failed factorization is not emitted (the core has no
//!   logger); the failure is visible through [`LeastAbsoluteDeviationSolver::valid`].

use super::SparseCholeskyWithFallbackSolver;
use crate::linalg::{
    maxi, ComputationInfo, SimplicialCholesky, SimplicialCholeskyKind, SparseMatrixCsc, VectorXd,
};
use crate::math::fns;
use crate::util::check::{ColmapError, ErrorKind};
use crate::{check_eq, check_ge, check_gt, Result};

/// Linear solver for the normal equations, `LeastAbsoluteDeviationSolver::Options::SolverType`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LeastAbsoluteDeviationSolverType {
    /// Simplicial LLT (`Eigen::SimplicialLLT` in COLMAP).
    SimplicialLlt,
    /// LLT with LDLT fallback (CHOLMOD supernodal LLT in COLMAP).
    SupernodalCholmodLlt,
}

/// Port of `LeastAbsoluteDeviationSolver::Options`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LeastAbsoluteDeviationSolverOptions {
    /// Augmented Lagrangian parameter.
    pub rho: f64,
    /// Over-relaxation parameter, typical values are between 1.0 and 1.8.
    pub alpha: f64,
    /// Maximum solver iterations.
    pub max_num_iterations: i32,
    /// Absolute solution threshold, as suggested by Boyd et al.
    pub absolute_tolerance: f64,
    /// Relative solution threshold, as suggested by Boyd et al.
    pub relative_tolerance: f64,
    /// Tikhonov ridge added to the diagonal of the normal equations A^T A before
    /// factorization. Set to a small positive value (e.g., 1e-12) for poorly conditioned but
    /// mathematically positive definite systems. Default 0 disables regularization.
    pub ridge_regularization: f64,
    /// Which linear solver factors A^T A.
    pub solver_type: LeastAbsoluteDeviationSolverType,
}

impl Default for LeastAbsoluteDeviationSolverOptions {
    fn default() -> Self {
        Self {
            rho: 1.0,
            alpha: 1.0,
            max_num_iterations: 1000,
            absolute_tolerance: 1e-4,
            relative_tolerance: 1e-2,
            ridge_regularization: 0.0,
            solver_type: LeastAbsoluteDeviationSolverType::SimplicialLlt,
        }
    }
}

#[derive(Clone, Debug)]
enum LinearSolver {
    // Boxed: the factorizations are a few hundred bytes of headers each.
    Simplicial(Box<SimplicialCholesky>),
    Fallback(Box<SparseCholeskyWithFallbackSolver>),
}

/// Least absolute deviations (LAD) fitting via ADMM by solving the problem
///
/// ```text
///        min || A x - b ||_1
/// ```
///
/// This implementation is based on the paper "Distributed Optimization and Statistical
/// Learning via the Alternating Direction Method of Multipliers" by Boyd et al. and the
/// Matlab implementation at
/// <https://web.stanford.edu/~boyd/papers/admm/least_abs_deviations/lad.html>.
/// Port of `colmap::LeastAbsoluteDeviationSolver`.
#[derive(Clone, Debug)]
pub struct LeastAbsoluteDeviationSolver {
    options: LeastAbsoluteDeviationSolverOptions,
    a: SparseMatrixCsc,
    linear_solver: LinearSolver,
    valid: bool,
}

impl LeastAbsoluteDeviationSolver {
    /// Validates the options and factors A^T A (plus the ridge).
    pub fn new(options: &LeastAbsoluteDeviationSolverOptions, a: &SparseMatrixCsc) -> Result<Self> {
        let options = *options;
        check_ge!(options.ridge_regularization, 0.0);
        check_gt!(options.rho, 0.0);
        check_gt!(options.alpha, 0.0);
        check_gt!(options.max_num_iterations, 0);
        check_ge!(options.absolute_tolerance, 0.0);
        check_ge!(options.relative_tolerance, 0.0);
        if a.rows() < a.cols() {
            return Err(ColmapError::new(
                ErrorKind::RuntimeError,
                "Underdetermined systems not supported.",
            ));
        }

        let ata = normal_equations(a, options.ridge_regularization);
        let (linear_solver, valid) = match options.solver_type {
            LeastAbsoluteDeviationSolverType::SimplicialLlt => {
                let mut solver =
                    Box::new(SimplicialCholesky::with_kind(SimplicialCholeskyKind::Llt));
                solver.compute(&ata)?;
                let valid = solver.info() == ComputationInfo::Success;
                (LinearSolver::Simplicial(solver), valid)
            }
            LeastAbsoluteDeviationSolverType::SupernodalCholmodLlt => {
                let mut solver = Box::new(SparseCholeskyWithFallbackSolver::new());
                let valid = solver.compute(&ata)?;
                (LinearSolver::Fallback(solver), valid)
            }
        };
        Ok(Self {
            options,
            a: a.clone(),
            linear_solver,
            valid,
        })
    }

    /// False if the factorization of A^T A failed during construction (e.g., the system is
    /// rank deficient or numerically not positive definite), in which case `solve` always
    /// returns false without producing NaN output.
    pub fn valid(&self) -> bool {
        self.valid
    }

    /// The ADMM iteration cap of the next `solve`.
    pub fn max_num_iterations(&self) -> i32 {
        self.options.max_num_iterations
    }

    /// Sets the ADMM iteration cap of the next `solve` without refactoring (COLMAP's callers
    /// change it through their reference to the options).
    pub fn set_max_num_iterations(&mut self, value: i32) -> Result<()> {
        check_gt!(value, 0);
        self.options.max_num_iterations = value;
        Ok(())
    }

    /// Runs ADMM from scratch and writes the solution into `x`. Returns `Ok(false)`, leaving
    /// x unchanged, when the factorization failed.
    pub fn solve(&self, b: &VectorXd, x: &mut VectorXd) -> Result<bool> {
        check_eq!(b.len(), self.a.rows());
        if !self.valid {
            return Ok(false);
        }
        let a = &self.a;
        let rows = a.rows();
        let mut z = VectorXd::zeros(rows);
        let mut u = VectorXd::zeros(rows);
        let b_norm = b.norm();
        let eps_pri_threshold = fns::sqrt(rows as f64) * self.options.absolute_tolerance;
        let eps_dual_threshold = fns::sqrt(a.cols() as f64) * self.options.absolute_tolerance;
        let alpha = self.options.alpha;
        let rho = self.options.rho;

        for _ in 0..self.options.max_num_iterations {
            let rhs = a.transpose_multiply(&(&(b + &z) - &u));
            match self.solve_normal(&rhs) {
                Some(solution) => *x = solution,
                None => return Ok(false),
            }
            let ax = a.multiply(x);
            let ax_hat = &(&ax * alpha) + &(&(&z + b) * (1.0 - alpha));

            let z_old = z;
            z = shrinkage(&(&(&ax_hat - b) + &u), 1.0 / rho);
            u = &u + &(&(&ax_hat - &z) - b);

            let r_norm = (&(&ax - &z) - b).norm();
            let s_norm = (&a.transpose_multiply(&(&z - &z_old)) * -rho).norm();
            let eps_pri = eps_pri_threshold
                + self.options.relative_tolerance * maxi(b_norm, maxi(ax.norm(), z.norm()));
            let eps_dual = eps_dual_threshold
                + self.options.relative_tolerance * (&a.transpose_multiply(&u) * rho).norm();

            if r_norm < eps_pri && s_norm < eps_dual {
                break;
            }
        }
        Ok(true)
    }

    fn solve_normal(&self, rhs: &VectorXd) -> Option<VectorXd> {
        match &self.linear_solver {
            LinearSolver::Simplicial(solver) => solver.solve(rhs).ok(),
            LinearSolver::Fallback(solver) => solver.solve(rhs),
        }
    }
}

fn shrinkage(a: &VectorXd, kappa: f64) -> VectorXd {
    VectorXd::from_vec(
        a.as_slice()
            .iter()
            .map(|&v| {
                let plus = v + kappa;
                let minus = v - kappa;
                // Eigen's cwiseMin(0) / cwiseMax(0).
                (if plus < 0.0 { plus } else { 0.0 }) + (if minus > 0.0 { minus } else { 0.0 })
            })
            .collect(),
    )
}

fn normal_equations(a: &SparseMatrixCsc, ridge_regularization: f64) -> SparseMatrixCsc {
    let ata = a.transpose_times_self();
    if ridge_regularization > 0.0 {
        // The diagonal of A^T A is populated whenever the corresponding column of A has any
        // entry; for an all-zero column Eigen's coeffRef inserts one, as add_to_diagonal does.
        ata.add_to_diagonal(ridge_regularization)
    } else {
        ata
    }
}
