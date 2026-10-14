//! One block of parameters the solver optimizes. Port of colmap-sharp's
//! `ColmapSharp/Solver/ParameterBlock.cs` (our MIT code), from Ceres Solver 2.2.0
//! `internal/ceres/parameter_block.h` and `internal/ceres/array_utils.cc` (BSD-3-Clause, see
//! THIRD_PARTY_NOTICES.md).
//!
//! Ceres' parameter block is a view of user memory keyed by the user's `double*`. Here the
//! [`crate::solver::Problem`] owns every block's values in one arena (`problem.rs`, divergence
//! 306), and a block records its offset there ([`ParameterBlock::user_offset`]). During an
//! evaluation a variable block reads its values from the program's state vector at
//! [`ParameterBlock::state_offset`] (Ceres moves its `state_` pointer there); a constant block
//! keeps reading the arena. `program.rs` assigns the offsets; `residual_block.rs` reads the
//! values and the Plus Jacobian. Per-coordinate bounds are stored lazily, as in Ceres: a block
//! with none allocates nothing, and [`ParameterBlock::plus`] projects onto the box after the
//! manifold's Plus.

use std::sync::Arc;

use super::manifold::Manifold;
use crate::{check_lt, ColmapError, ErrorKind, Result};

/// Ceres' `kImpossibleValue`: evaluation outputs are pre-filled with it so that a cost
/// function that forgets to write a value is caught, along with NaN and infinity.
pub const IMPOSSIBLE_VALUE: f64 = 1e302;

/// Ceres' `InvalidateArray`: fills `x` with [`IMPOSSIBLE_VALUE`].
pub fn invalidate_array(x: &mut [f64]) {
    x.fill(IMPOSSIBLE_VALUE);
}

/// Ceres' `IsArrayValid`: every entry is finite and was written.
pub fn is_array_valid(x: &[f64]) -> bool {
    x.iter().all(|&v| v.is_finite() && v != IMPOSSIBLE_VALUE)
}

/// Port of `ceres::internal::ParameterBlock`.
#[derive(Clone)]
pub struct ParameterBlock {
    user_offset: usize,
    size: usize,
    manifold: Option<Arc<dyn Manifold>>,
    // Row-major size x tangent_size Jacobian of Plus at the block's state; empty without a
    // manifold.
    plus_jacobian: Vec<f64>,
    upper_bounds: Option<Vec<f64>>,
    lower_bounds: Option<Vec<f64>>,
    is_set_constant: bool,
    /// Position in the owning program's parameter block list, or a scratch marker (-1).
    pub index: isize,
    /// Offset of the block in the program's state vector.
    pub state_offset: usize,
    /// Offset of the block in the program's tangent (delta) vector.
    pub delta_offset: usize,
}

impl std::fmt::Debug for ParameterBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParameterBlock")
            .field("user_offset", &self.user_offset)
            .field("size", &self.size)
            .field("has_manifold", &self.manifold.is_some())
            .field("is_set_constant", &self.is_set_constant)
            .field("index", &self.index)
            .finish()
    }
}

impl ParameterBlock {
    /// A block of `size` values at `user_offset` in the problem's value arena, at position
    /// `index` (Ceres' `ParameterBlock(double* user_state, int size, int index)`).
    pub fn new(user_offset: usize, size: usize, index: isize) -> Self {
        Self {
            user_offset,
            size,
            manifold: None,
            plus_jacobian: Vec::new(),
            upper_bounds: None,
            lower_bounds: None,
            is_set_constant: false,
            index,
            state_offset: 0,
            delta_offset: 0,
        }
    }

    /// Offset of the block's values in the problem's arena (Ceres' `user_state()`).
    pub fn user_offset(&self) -> usize {
        self.user_offset
    }

    /// Number of ambient parameters.
    pub fn size(&self) -> usize {
        self.size
    }

    /// The manifold, or `None` for plain Euclidean space.
    pub fn manifold(&self) -> Option<&Arc<dyn Manifold>> {
        self.manifold.as_ref()
    }

    /// Set by `SetParameterBlockConstant`.
    pub fn is_set_constant(&self) -> bool {
        self.is_set_constant
    }

    /// Ceres' `SetConstant` / `SetVarying`.
    pub fn set_constant(&mut self, constant: bool) {
        self.is_set_constant = constant;
    }

    /// Constant if set so, or if the manifold leaves no tangent space.
    pub fn is_constant(&self) -> bool {
        self.is_set_constant || self.tangent_size() == 0
    }

    /// Dimension of the step space.
    pub fn tangent_size(&self) -> usize {
        self.manifold
            .as_ref()
            .map_or(self.size, |m| m.tangent_size())
    }

    /// The row-major `size x tangent_size` Jacobian of Plus at the block's state, or `None`
    /// without a manifold.
    pub fn plus_jacobian(&self) -> Option<&[f64]> {
        self.manifold
            .as_ref()
            .map(|_| self.plus_jacobian.as_slice())
    }

    /// Ceres' `SetManifold`, with the block's current values `x`: the manifold's ambient size
    /// must match the block's, and its Plus Jacobian must be computable at `x`. `None`
    /// clears the manifold.
    pub fn set_manifold(&mut self, manifold: Option<Arc<dyn Manifold>>, x: &[f64]) -> Result<()> {
        if let (Some(new), Some(old)) = (&manifold, &self.manifold) {
            if Arc::ptr_eq(new, old) {
                return Ok(());
            }
        }

        let Some(manifold) = manifold else {
            self.manifold = None;
            self.plus_jacobian = Vec::new();
            return Ok(());
        };

        if manifold.ambient_size() != self.size {
            return Err(ColmapError::new(
                ErrorKind::InvalidArgument,
                format!(
                    "The parameter block has size = {} while the manifold has ambient size = {}",
                    self.size,
                    manifold.ambient_size()
                ),
            ));
        }

        self.plus_jacobian = vec![0.0; manifold.ambient_size() * manifold.tangent_size()];
        self.manifold = Some(manifold);
        if !self.update_plus_jacobian(x) {
            return Err(ColmapError::new(
                ErrorKind::InvalidArgument,
                "Manifold::PlusJacobian computation failed for the parameter block's current values.",
            ));
        }
        Ok(())
    }

    /// Ceres' `SetState`: the block's values are now `x` (a `size`-long slice); refreshes the
    /// Plus Jacobian there. Only for variable blocks, as in Ceres (an internal invariant).
    pub fn set_state(&mut self, x: &[f64]) -> bool {
        assert!(
            !self.is_constant(),
            "Tried to set the state of a constant parameter block."
        );
        self.update_plus_jacobian(x)
    }

    /// `x_plus_delta = Plus(x, delta)`, through the manifold if there is one, then projected
    /// onto the box of the bounds.
    pub fn plus(&self, x: &[f64], delta: &[f64], x_plus_delta: &mut [f64]) -> bool {
        let size = self.size;
        match &self.manifold {
            Some(manifold) => {
                if !manifold.plus(x, delta, x_plus_delta) {
                    return false;
                }
            }
            None => {
                for i in 0..size {
                    x_plus_delta[i] = x[i] + delta[i];
                }
            }
        }

        // Project onto the box constraints (std::max / std::min: a NaN coordinate stays NaN
        // against a bound on the left, as in Ceres, so the evaluation fails rather than the
        // bound hiding it).
        if let Some(lower) = &self.lower_bounds {
            for i in 0..size {
                x_plus_delta[i] = cpp_max(x_plus_delta[i], lower[i]);
            }
        }
        if let Some(upper) = &self.upper_bounds {
            for i in 0..size {
                x_plus_delta[i] = cpp_min(x_plus_delta[i], upper[i]);
            }
        }
        true
    }

    /// Ceres' `SetUpperBound`: an infinite (`f64::MAX`) bound on an unbounded block allocates
    /// nothing.
    pub fn set_upper_bound(&mut self, index: usize, upper_bound: f64) -> Result<()> {
        check_lt!(index, self.size);
        if upper_bound >= f64::MAX && self.upper_bounds.is_none() {
            return Ok(());
        }
        let size = self.size;
        self.upper_bounds
            .get_or_insert_with(|| vec![f64::MAX; size])[index] = upper_bound;
        Ok(())
    }

    /// Ceres' `SetLowerBound`: an infinite (`-f64::MAX`) bound on an unbounded block
    /// allocates nothing.
    pub fn set_lower_bound(&mut self, index: usize, lower_bound: f64) -> Result<()> {
        check_lt!(index, self.size);
        if lower_bound <= -f64::MAX && self.lower_bounds.is_none() {
            return Ok(());
        }
        let size = self.size;
        self.lower_bounds
            .get_or_insert_with(|| vec![-f64::MAX; size])[index] = lower_bound;
        Ok(())
    }

    /// The upper bound of coordinate `index` (`f64::MAX` if none).
    pub fn upper_bound_for_parameter(&self, index: usize) -> f64 {
        self.upper_bounds.as_ref().map_or(f64::MAX, |b| b[index])
    }

    /// The lower bound of coordinate `index` (`-f64::MAX` if none).
    pub fn lower_bound_for_parameter(&self, index: usize) -> f64 {
        self.lower_bounds.as_ref().map_or(-f64::MAX, |b| b[index])
    }

    // Ceres' UpdatePlusJacobian: invalidate, evaluate at x, then require every entry written
    // and finite.
    fn update_plus_jacobian(&mut self, x: &[f64]) -> bool {
        let Some(manifold) = &self.manifold else {
            return true;
        };
        invalidate_array(&mut self.plus_jacobian);
        if !manifold.plus_jacobian(&x[..self.size], &mut self.plus_jacobian) {
            return false;
        }
        is_array_valid(&self.plus_jacobian)
    }
}

// std::max(a, b) is (a < b) ? b : a; std::min(a, b) is (b < a) ? b : a. f64::max/min treat NaN
// differently.
fn cpp_max(a: f64, b: f64) -> f64 {
    if a < b {
        b
    } else {
        a
    }
}

fn cpp_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}
