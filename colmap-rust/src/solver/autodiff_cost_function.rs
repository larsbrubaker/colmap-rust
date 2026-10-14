//! The residual-block contract bundle adjustment and every other COLMAP refinement is written
//! against: [`CostFunction`] (`ceres::CostFunction`), [`AutoDiffFunctor`] (a Ceres functor's
//! templated `operator()`), and [`AutoDiffCostFunction`] (`ceres::AutoDiffCostFunction`).
//! Port of colmap-sharp's `ColmapSharp/Solver/AutoDiffCostFunction.cs` (from Ceres Solver
//! 2.2.0 `include/ceres/cost_function.h`, `autodiff_cost_function.h` and
//! `internal/autodiff.h`, BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//!
//! Design (an API the rest of the solver builds on):
//! - [`CostFunction`] is object safe, so a problem holds `Box<dyn CostFunction>` residual
//!   blocks. Ceres' `double const* const* parameters` is `&[&[f64]]`, one slice per block;
//!   `double** jacobians` is `Option<&mut [Option<&mut [f64]>]>`: `None` asks for no
//!   Jacobians, a `None` entry skips that block. Jacobians are row-major
//!   `num_residuals x block_size`, as Ceres lays them out.
//! - [`AutoDiffFunctor::call`] is the Rust shape of `template <typename T> bool
//!   operator()(const T* p0, const T* p1, ..., T* residuals) const`: one generic method over
//!   [`Scalar`], given every parameter block concatenated in order (the functor slices them
//!   at the offsets its block sizes imply: `p0 = &p[..3]`, `p1 = &p[3..10]`, ...).
//! - [`AutoDiffCostFunction<F, R, N>`] is Ceres' fixed-size `AutoDiffCostFunction<F, R, N0,
//!   N1, ...>`: `R` residuals and `N = N0 + N1 + ...` parameters, which is also the Jet width
//!   (Ceres' `Jet<double, N>`). The block sizes are checked to sum to `N`. Every COLMAP cost
//!   function has a fixed residual count, so Ceres' `DYNAMIC` residuals are not needed.
//!   colmap-sharp's chunked evaluation (narrower Jets in several passes) existed because C#
//!   has no const generics; here every width is free, so this evaluates in one pass of
//!   `Jet<N>` exactly as Ceres does.
//!
//! Evaluation allocates nothing: the Jets live in `[Jet<N>; N]` and `[Jet<N>; R]` on the
//! stack (the widest COLMAP cost function, 33 parameters and 2 residuals, needs 35 * 34 * 8 =
//! 9520 bytes). One instance may be evaluated from several threads at once, since the
//! functor's `call` takes `&self` (C++'s `const`).

use crate::{check_eq, check_gt, Result};

use super::jet::Jet;
use super::scalar::Scalar;

/// Ceres' `kImpossibleValue`: residual Jets start as this so that a functor that forgets to
/// write a residual is visible rather than reading stale values.
pub(crate) const IMPOSSIBLE_VALUE: f64 = 1e302;

/// Port of `ceres::CostFunction`: a residual block's residuals and Jacobians.
pub trait CostFunction {
    /// Number of residuals, Ceres' `num_residuals()`.
    fn num_residuals(&self) -> usize;

    /// Size of each parameter block, Ceres' `parameter_block_sizes()`.
    fn parameter_block_sizes(&self) -> &[usize];

    /// Computes the residuals and, where asked, the Jacobians. `parameters` holds one slice
    /// per block, each at least the block's size. `jacobians` is `None` for no Jacobians, or
    /// holds one entry per block: `None` to skip that block, else room for a row-major
    /// `num_residuals x block_size` matrix. Returns false if the residuals could not be
    /// computed (the solver then rejects the step).
    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool;
}

/// A residual functor generic over the scalar type: the Rust form of a Ceres functor's
/// `template <typename T> bool operator()(...) const`.
pub trait AutoDiffFunctor {
    /// Computes the residuals from the parameter blocks, concatenated in block order.
    /// Returns false if they cannot be computed.
    fn call<T: Scalar>(&self, parameters: &[T], residuals: &mut [T]) -> bool;
}

/// Port of `ceres::AutoDiffCostFunction<F, R, N0, N1, ...>`: a [`CostFunction`] whose
/// Jacobians come from evaluating `F` on [`Jet<N>`], with `R` residuals and `N` parameters
/// in total.
#[derive(Clone, Debug)]
pub struct AutoDiffCostFunction<F, const R: usize, const N: usize> {
    functor: F,
    parameter_block_sizes: Vec<usize>,
}

impl<F: AutoDiffFunctor, const R: usize, const N: usize> AutoDiffCostFunction<F, R, N> {
    /// Wraps `functor`, whose parameter blocks have the given sizes (summing to `N`).
    pub fn new(functor: F, parameter_block_sizes: &[usize]) -> Result<Self> {
        check_gt!(R, 0);
        check_gt!(parameter_block_sizes.len(), 0);
        for &size in parameter_block_sizes {
            check_gt!(size, 0);
        }
        check_eq!(parameter_block_sizes.iter().sum::<usize>(), N);
        Ok(Self {
            functor,
            parameter_block_sizes: parameter_block_sizes.to_vec(),
        })
    }

    /// The wrapped functor.
    pub fn functor(&self) -> &F {
        &self.functor
    }
}

impl<F: AutoDiffFunctor, const R: usize, const N: usize> CostFunction
    for AutoDiffCostFunction<F, R, N>
{
    fn num_residuals(&self) -> usize {
        R
    }

    fn parameter_block_sizes(&self) -> &[usize] {
        &self.parameter_block_sizes
    }

    fn evaluate(
        &self,
        parameters: &[&[f64]],
        residuals: &mut [f64],
        jacobians: Option<&mut [Option<&mut [f64]>]>,
    ) -> bool {
        let sizes = &self.parameter_block_sizes;
        assert_eq!(
            parameters.len(),
            sizes.len(),
            "one slice per parameter block"
        );

        let Some(jacobians) = jacobians else {
            // No Jacobians requested: call the functor on doubles, skipping Jets.
            let mut x = [0.0; N];
            let mut offset = 0;
            for (block, &size) in parameters.iter().zip(sizes) {
                x[offset..offset + size].copy_from_slice(&block[..size]);
                offset += size;
            }
            return self.functor.call(&x, &mut residuals[..R]);
        };
        assert_eq!(
            jacobians.len(),
            sizes.len(),
            "one entry per parameter block"
        );

        // Ceres' Make1stOrderPerturbations: every parameter is a variable at its offset in
        // the concatenated vector, whether or not its block's Jacobian is wanted.
        let mut x = [Jet::<N>::constant(0.0); N];
        let mut offset = 0;
        for (block, &size) in parameters.iter().zip(sizes) {
            for c in 0..size {
                x[offset + c] = Jet::variable(block[c], offset + c);
            }
            offset += size;
        }

        // Invalidate the output Jets, so that a residual the functor leaves unassigned shows.
        let mut output = [Jet::<N>::new(IMPOSSIBLE_VALUE, [IMPOSSIBLE_VALUE; N]); R];
        if !self.functor.call(&x, &mut output) {
            return false;
        }

        // Ceres' Take0thOrderPart and Take1stOrderPart, the latter only for the blocks whose
        // Jacobian is wanted.
        for (residual, jet) in residuals[..R].iter_mut().zip(&output) {
            *residual = jet.a;
        }
        let mut offset = 0;
        for (jacobian, &size) in jacobians.iter_mut().zip(sizes) {
            if let Some(jacobian) = jacobian {
                for (r, jet) in output.iter().enumerate() {
                    jacobian[r * size..(r + 1) * size]
                        .copy_from_slice(&jet.v[offset..offset + size]);
                }
            }
            offset += size;
        }
        true
    }
}
