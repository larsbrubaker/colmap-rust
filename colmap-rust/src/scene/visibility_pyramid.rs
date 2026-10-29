//! Port of `colmap/scene/visibility_pyramid.h` / `visibility_pyramid.cc`:
//! [`VisibilityPyramid`], a score of how uniformly points cover a 2D grid (the incremental
//! mapper uses it to rank the next image by how evenly its visible 3D points spread over the
//! image). Tests: `tests/scene/visibility_pyramid.rs` (`visibility_pyramid_test.cc` 1:1).
//!
//! Tier A (integer bookkeeping). Translation notes:
//! - Each level is `Eigen::MatrixXi`; here a row-major `Vec<i32>` with its dimension (only
//!   per-cell counts and the cell count are used, so the storage order is invisible).
//! - `max_score_ += dim * dim * dim * dim` is `int` arithmetic in COLMAP and overflows (UB)
//!   from 8 levels on (dim 256); here it is `usize`, the value the formula means.
//! - `CellForPoint` converts `max_dim * x / width` to `size_t` before clamping, which is UB
//!   for negative or NaN coordinates in C++; Rust's `as usize` saturates them to 0, i.e. the
//!   first cell, and the clamp then keeps the result in range.
//!
//! Defining those two undefined-behavior cases is a deliberate divergence: undefined behavior
//! has no result to match, and these are the values a caller would expect (a coordinate
//! beyond the range lands in the last cell, as in COLMAP). COLMAP's only user, the
//! incremental mapper's observation manager, builds pyramids with
//! `kNumPoint3DVisibilityPyramidLevels = 6` levels and the image's size, where both versions
//! agree. `tests/scene/visibility_pyramid.rs` (1:1, 1 to 7 levels) passes unchanged;
//! `tests/scene/rust_only_scene_types.rs` pins the two cases in
//! `rust_only_visibility_pyramid_eight_levels_max_score` and
//! `rust_only_visibility_pyramid_out_of_range_points_clamp`.

use crate::math::utils::clamp;
use crate::util::check::Result;
use crate::{check_gt, check_le};

/// One pyramid level: a `dim` x `dim` grid of per-cell point counts.
#[derive(Debug, Clone, PartialEq)]
struct Level {
    dim: usize,
    counts: Vec<i32>,
}

/// Port of `colmap::VisibilityPyramid`. A class that captures the distribution of points in a
/// 2D grid, for example the distribution of visible 3D points in an image.
///
/// The class captures the distribution of points by a score. A higher score corresponds to a
/// more uniform distribution of the points in the grid. The score is computed by the number
/// of populated cells in a multi-resolution pyramid. A populated cell contributes to the
/// overall score if it is populated by at least one point and the contributed score is
/// according to its resolution in the pyramid. A cell in a higher resolution level
/// contributes a higher score to the overall score.
#[derive(Debug, Clone, PartialEq)]
pub struct VisibilityPyramid {
    /// Range of the input points.
    width: usize,
    height: usize,
    /// The overall visibility score.
    score: usize,
    /// The maximum score when all cells are populated.
    max_score: usize,
    /// The visibility pyramid with multiple levels; level `l` has `2^(l+1)` cells per side.
    pyramid: Vec<Level>,
}

impl Default for VisibilityPyramid {
    /// `VisibilityPyramid()`: no levels and an empty range.
    fn default() -> Self {
        Self::new(0, 0, 0)
    }
}

impl VisibilityPyramid {
    /// `VisibilityPyramid(num_levels, width, height)`.
    pub fn new(num_levels: usize, width: usize, height: usize) -> Self {
        let mut max_score = 0usize;
        let mut pyramid = Vec::with_capacity(num_levels);
        for level in 0..num_levels {
            let level_plus_one = level + 1;
            let dim = 1usize << level_plus_one;
            pyramid.push(Level {
                dim,
                counts: vec![0; dim * dim],
            });
            max_score += dim * dim * dim * dim;
        }
        Self {
            width,
            height,
            score: 0,
            max_score,
            pyramid,
        }
    }

    /// `SetPoint`: adds a point at `(x, y)`; every level whose cell becomes populated adds its
    /// cell count to the score.
    pub fn set_point(&mut self, x: f64, y: f64) -> Result<()> {
        check_gt!(self.pyramid.len(), 0);

        let (mut cx, mut cy) = self.cell_for_point(x, y)?;

        for level in self.pyramid.iter_mut().rev() {
            let cell = &mut level.counts[cy * level.dim + cx];
            *cell += 1;
            if *cell == 1 {
                self.score += level.counts.len();
            }

            cx >>= 1;
            cy >>= 1;
        }

        check_le!(self.score, self.max_score);
        Ok(())
    }

    /// `ResetPoint`: removes a point at `(x, y)`; every level whose cell becomes empty
    /// subtracts its cell count from the score.
    pub fn reset_point(&mut self, x: f64, y: f64) -> Result<()> {
        check_gt!(self.pyramid.len(), 0);

        let (mut cx, mut cy) = self.cell_for_point(x, y)?;

        for level in self.pyramid.iter_mut().rev() {
            let cell = &mut level.counts[cy * level.dim + cx];
            *cell -= 1;
            if *cell == 0 {
                self.score -= level.counts.len();
            }

            cx >>= 1;
            cy >>= 1;
        }

        check_le!(self.score, self.max_score);
        Ok(())
    }

    /// `NumLevels`.
    pub fn num_levels(&self) -> usize {
        self.pyramid.len()
    }

    /// `Width`.
    pub fn width(&self) -> usize {
        self.width
    }

    /// `Height`.
    pub fn height(&self) -> usize {
        self.height
    }

    /// `Score`.
    pub fn score(&self) -> usize {
        self.score
    }

    /// `MaxScore`: the score when all cells are populated.
    pub fn max_score(&self) -> usize {
        self.max_score
    }

    /// `CellForPoint`: the cell of the finest level that contains `(x, y)`.
    fn cell_for_point(&self, x: f64, y: f64) -> Result<(usize, usize)> {
        check_gt!(self.width, 0);
        check_gt!(self.height, 0);
        let max_dim = 1usize << self.pyramid.len();
        // `int * double / size_t` in double, then `static_cast<size_t>` (see the header for
        // negative and NaN values).
        let cx = (max_dim as f64 * x / self.width as f64) as usize;
        let cy = (max_dim as f64 * y / self.height as f64) as usize;
        Ok((clamp(cx, 0, max_dim - 1), clamp(cy, 0, max_dim - 1)))
    }
}
