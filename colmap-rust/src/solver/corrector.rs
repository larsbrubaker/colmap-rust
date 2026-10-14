//! The robust-loss corrector: rescales a residual block's residuals and Jacobian so that the
//! Gauss-Newton model of rho(|f|^2) / 2 matches its gradient and (a curvature-correct
//! approximation of) its Hessian (Triggs et al., "Bundle Adjustment — A Modern Synthesis").
//! Port of the `Corrector` struct in colmap-sharp's `ColmapSharp/Solver/ResidualBlock.cs`,
//! which ports Ceres Solver 2.2.0's `internal/ceres/corrector.h` and `corrector.cc`
//! (BSD-3-Clause, see THIRD_PARTY_NOTICES.md). A residual block (the problem model, next
//! slice) builds one from a [`crate::solver::LossFunction`]'s rho after evaluating its cost
//! function.
//!
//! Jacobians are row-major, as Ceres' are. Tests: `tests/solver/corrector.rs` (Ceres'
//! `corrector_test.cc`).

use crate::math::fns;
use crate::{check_ge, check_gt, Result};

/// Port of `ceres::internal::Corrector`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Corrector {
    sqrt_rho1: f64,
    residual_scaling: f64,
    alpha_sq_norm: f64,
}

impl Corrector {
    /// The correction for squared norm `sq_norm` and `rho` = (rho, rho', rho''). Ceres
    /// CHECK-fails on a negative `sq_norm`, and on rho' <= 0 where the curvature correction
    /// applies; the port returns the check's error.
    pub fn new(sq_norm: f64, rho: &[f64; 3]) -> Result<Self> {
        check_ge!(sq_norm, 0.0);
        let sqrt_rho1 = fns::sqrt(rho[1]);

        // If sq_norm = 0.0, the correction becomes trivially zero. Ceres also skips the
        // correction when rho'' <= 0 (the common robust-loss outlier region), where the
        // alpha below would need the square root of something that can go negative; there
        // the Jacobian is only scaled, which is always a valid (if cruder) model.
        if sq_norm == 0.0 || rho[2] <= 0.0 {
            return Ok(Self {
                sqrt_rho1,
                residual_scaling: sqrt_rho1,
                alpha_sq_norm: 0.0,
            });
        }

        // We now require that the first derivative of the loss function be positive only if
        // the second derivative is positive. This is because the loss functions do not
        // satisfy rho' > 0 everywhere (Tukey).
        check_gt!(rho[1], 0.0);

        // Calculate the smaller of the two solutions to the equation
        //   0.5 * alpha^2 - alpha - rho'' / rho' * z'z = 0.
        // Start by calculating the discriminant D.
        let d = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];

        // Since both rho[1] and rho[2] are guaranteed to be positive at this point, we know
        // that D > 1.0.
        let alpha = 1.0 - fns::sqrt(d);

        // Calculate the constants needed by the correction routines.
        Ok(Self {
            sqrt_rho1,
            residual_scaling: sqrt_rho1 / (1.0 - alpha),
            alpha_sq_norm: alpha / sq_norm,
        })
    }

    /// residuals *= sqrt(rho') / (1 - alpha).
    pub fn correct_residuals(&self, residuals: &mut [f64]) {
        for residual in residuals.iter_mut() {
            *residual *= self.residual_scaling;
        }
    }

    /// J = sqrt(rho') (I - alpha r r' / |r|^2) J for a row-major `num_rows x num_cols` J,
    /// given the uncorrected `residuals` (call this before [`Self::correct_residuals`]).
    pub fn correct_jacobian(
        &self,
        num_rows: usize,
        num_cols: usize,
        residuals: &[f64],
        jacobian: &mut [f64],
    ) {
        // The common case (rho[2] <= 0).
        if self.alpha_sq_norm == 0.0 {
            for value in &mut jacobian[..num_rows * num_cols] {
                *value *= self.sqrt_rho1;
            }
            return;
        }

        // Equation 10 of the Ceres derivation: J = sqrt(rho) * (J - alpha^2 r * r' J).
        for c in 0..num_cols {
            let mut r_transpose_j = 0.0;
            for r in 0..num_rows {
                r_transpose_j += jacobian[r * num_cols + c] * residuals[r];
            }

            for r in 0..num_rows {
                jacobian[r * num_cols + c] = self.sqrt_rho1
                    * (jacobian[r * num_cols + c]
                        - self.alpha_sq_norm * residuals[r] * r_transpose_j);
            }
        }
    }
}
