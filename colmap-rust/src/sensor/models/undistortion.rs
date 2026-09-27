//! `BasePerspectiveCameraModel::IterativeUndistortion` (`colmap/sensor/models.h`): inverts a
//! model's additive distortion `x + d(x) = x0` by a trust-region Newton iteration. Ported
//! from colmap-sharp's `ColmapSharp/Sensor/IterativeUndistortion.cs`.
//!
//! The Jacobian of `d` comes from evaluating the model's `Distortion` on `Jet<2>` (`jet.rs`),
//! as COLMAP does with `ceres::Jet<double, 2>`, and each step solves the 2x2 system by LU
//! decomposition with partial (row) pivoting, which is what COLMAP's
//! `J.partialPivLu().solve(...)` computes. The 2x2 solve is written here from the textbook
//! algorithm (Golub and Van Loan, "Matrix Computations", 3.4.1: Gaussian elimination with
//! partial pivoting); Eigen (MPL-2.0) is not ported (docs/LICENSE_AUDIT.md).
//!
//! Tier A: the iteration count, step constants, trust-region clamp and every operation's order
//! match COLMAP; `tests/sensor/rust_only_camera_model_oracle.rs` pins the result bit for bit
//! against pycolmap.

use crate::math::fns;

use super::jet::Jet;
use super::DistortedCameraModel;

/// The most extra parameters of any model (RAD_TAN_THIN_PRISM_FISHEYE's 12).
const MAX_EXTRA_PARAMS: usize = 12;

/// Port of `BasePerspectiveCameraModel<M>::IterativeUndistortion`. On entry `(u, v)` is the
/// distorted point, on exit the undistorted one. Returns false if the iteration did not
/// converge within 100 steps (the last iterate is still written). `extra_params` holds the
/// model's extra (distortion) parameters only, COLMAP's `&params[first_extra]`.
pub fn iterative_undistortion<M: DistortedCameraModel>(
    extra_params: &[f64],
    u: &mut f64,
    v: &mut f64,
) -> bool {
    // Parameters for Newton iteration. 100 iterations should be enough for complex camera
    // models with higher order terms.
    const NUM_ITERATIONS: usize = 100;
    const MIN_STEP_SQUARED_NORM: f64 = 1e-10;
    // Trust region: step_x.norm() <= max(x.norm() * kRelStepRadius, kStepRadius)
    const REL_STEP_RADIUS: f64 = 0.1;
    const STEP_RADIUS: f64 = 0.1;

    let x00 = *u;
    let x01 = *v;
    let mut x0 = *u;
    let mut x1 = *v;

    // COLMAP copies `num_extra_params` values into a stack array of Jets; the largest model
    // (RAD_TAN_THIN_PRISM_FISHEYE) has MAX_EXTRA_PARAMS.
    let num_extra_params = M::EXTRA_PARAMS_IDXS.len();
    let mut params_jet_storage = [Jet::<2>::constant(0.0); MAX_EXTRA_PARAMS];
    for (jet, &p) in params_jet_storage
        .iter_mut()
        .zip(&extra_params[..num_extra_params])
    {
        *jet = Jet::constant(p);
    }
    let params_jet = &params_jet_storage[..num_extra_params];

    for _ in 0..NUM_ITERATIONS {
        // Get Jacobian
        let mut dx_jet0 = Jet::constant(0.0);
        let mut dx_jet1 = Jet::constant(0.0);
        M::distortion(
            params_jet,
            Jet::variable(x0, 0),
            Jet::variable(x1, 1),
            &mut dx_jet0,
            &mut dx_jet1,
        );
        let dx0 = dx_jet0.a;
        let dx1 = dx_jet1.a;
        let j00 = dx_jet0.v[0] + 1.0;
        let j01 = dx_jet0.v[1];
        let j10 = dx_jet1.v[0];
        let j11 = dx_jet1.v[1] + 1.0;

        // Update
        let (mut step0, mut step1) =
            solve_partial_piv_lu2(j00, j01, j10, j11, x0 + dx0 - x00, x1 + dx1 - x01);
        // std::max(a, b) is (a < b) ? b : a.
        let rel = (x0 * x0 + x1 * x1) * REL_STEP_RADIUS * REL_STEP_RADIUS;
        let abs = STEP_RADIUS * STEP_RADIUS;
        let radius_sqr = if rel < abs { abs } else { rel };
        let step_norm_sqr = step0 * step0 + step1 * step1;
        if step_norm_sqr > radius_sqr {
            let scale = fns::sqrt(radius_sqr / step_norm_sqr);
            step0 *= scale;
            step1 *= scale;
        }
        x0 -= step0;
        x1 -= step1;
        if step0 * step0 + step1 * step1 < MIN_STEP_SQUARED_NORM {
            *u = x0;
            *v = x1;
            return true;
        }
    }

    *u = x0;
    *v = x1;
    false
}

/// Solves `[a00 a01; a10 a11] x = b` by LU decomposition with partial pivoting: the pivot
/// row is the one with the larger `|a_i0|` (the first on a tie), L is unit lower
/// triangular, and both triangular solves divide by the pivots. A zero pivot column is left
/// unscaled, so a singular matrix yields inf/NaN rather than an error.
pub(crate) fn solve_partial_piv_lu2(
    mut a00: f64,
    mut a01: f64,
    mut a10: f64,
    mut a11: f64,
    mut b0: f64,
    mut b1: f64,
) -> (f64, f64) {
    if a10.abs() > a00.abs() {
        std::mem::swap(&mut a00, &mut a10);
        std::mem::swap(&mut a01, &mut a11);
        std::mem::swap(&mut b0, &mut b1);
    }

    let l10 = if a00 != 0.0 { a10 / a00 } else { a10 };
    let u11 = a11 - l10 * a01;

    // Forward substitution with the unit lower factor.
    let y1 = b1 - b0 * l10;

    // Back substitution with the upper factor.
    let x1 = y1 / u11;
    let x0 = (b0 - x1 * a01) / a00;
    (x0, x1)
}
