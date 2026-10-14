//! Port of the `FundamentalMatrixDegensacEstimator` class, `FundamentalMatrixDegensacOptions`
//! and `EstimateFundamentalMatrixDegensac` of COLMAP's
//! `colmap/estimators/fundamental_matrix_degensac.h` and `.cc`: the DEGENSAC fundamental
//! matrix estimator that LO-RANSAC runs as both the hypothesis and the local-optimization
//! estimator. It fits the 7-point or 8-point model
//! ([`crate::estimators::solvers::fundamental_matrix`]) and replaces H-degenerate hypotheses
//! by the plane-and-parallax completion of [`super::fundamental_matrix_degensac`] (the free
//! functions, next to this file). Port of colmap-sharp's
//! `Estimators/FundamentalMatrixDegensacEstimator.cs` and the options and entry point of
//! `FundamentalMatrixDegensac.cs`. Consumer: the two-view geometry estimator's
//! `use_degensac` branch. Tests: `tests/estimators/fundamental_matrix_degensac.rs`.
//!
//! Tier C (outcome): the hypotheses depend on PRNG draws (Tier A) and solvers (Tier B).
//!
//! Translation notes:
//! - COLMAP holds `const std::vector<Vector2d>*` to the full correspondence set, which must
//!   outlive the estimator; here it borrows the slices, so the estimator carries their
//!   lifetime.
//! - The class has `Refine(X, Y, M_t*)`, so LO-RANSAC refines the current best model: the
//!   estimator implements both [`Estimator`] and [`LocalEstimator`], whose
//!   `estimate_local` refines a copy and pushes it on success.
//! - `RefineFundamentalMatrixSampson`'s `bool` is ignored, as COLMAP ignores it.

use super::fundamental_matrix_degensac::{
    detect_sample_h_degeneracy, fundamental_from_plane_and_parallax, is_sample_h_degenerate,
    DEGENSAC_MIN_NUM_SAMPLES,
};
use crate::estimators::solvers::fundamental_matrix::{
    refine_fundamental_matrix_sampson, FundamentalMatrixEightPointEstimator,
    FundamentalMatrixSevenPointEstimator,
};
use crate::geometry::essential_matrix::compute_squared_sampson_error;
use crate::linalg::{Matrix3d, Vector2d};
use crate::math::fns;
use crate::optim::estimator::{Estimator, LocalEstimator};
use crate::optim::loransac::LoRansac;
use crate::optim::random_sampler::RandomSampler;
use crate::optim::ransac::{RansacOptions, RansacReport};
use crate::optim::support_measurement::{MEstimatorSupport, MEstimatorSupportMeasurer};
use crate::{check_eq, check_ge, Result};

/// DEGENSAC-based fundamental matrix estimation, robust to a dominant scene plane. Based on
/// Ondrej Chum, Tomas Werner, Jiri Matas, "Two-View Geometry Estimation Unaffected by a
/// Dominant Plane", CVPR 2005.
///
/// When the scene contains a dominant plane, a random minimal sample frequently contains a
/// majority of coplanar correspondences. The fundamental matrix computed from such an
/// H-degenerate sample is compatible with the plane homography but its epipole is
/// essentially unconstrained, yet it can still accrue high inlier support (all plane points
/// fit it). The recovered epipolar geometry is then wrong even though RANSAC reports a
/// confident result.
///
/// This is implemented as a solver that detects H-degeneracy of a sample and, instead of
/// returning the plane-corrupted model, completes it via plane-and-parallax: the dominant
/// plane homography is refit on the H-consistent data and the epipole (hence the
/// fundamental matrix) is recovered from the off-plane parallax. If no usable off-plane
/// parallax exists, the sample is rejected.
///
/// The solver handles both minimal (7-point) and non-minimal (>= 8-point) samples, so it
/// can be used as BOTH the hypothesis and the local-optimization estimator in LO-RANSAC.
/// This is important: a plain non-minimal solver would re-fit a plane-corrupted model on
/// the (plane-dominated) inlier set during local optimization, undoing the completion.
///
/// Port of `colmap::FundamentalMatrixDegensacEstimator`.
#[derive(Clone, Copy, Debug)]
pub struct FundamentalMatrixDegensacEstimator<'a> {
    points1: &'a [Vector2d],
    points2: &'a [Vector2d],
    sampson_max_residual: f64,
    plane_max_residual: f64,
    off_plane_min_residual: f64,
    min_sample_h_inlier_ratio: f64,
    max_plane_parallax_trials: i32,
    use_sampson_refinement: bool,
}

impl<'a> FundamentalMatrixDegensacEstimator<'a> {
    /// - `points1`, `points2`: the full correspondences (the same data passed to RANSAC).
    /// - `sampson_max_residual`: squared max Sampson error used when scoring completion
    ///   candidates (typically the RANSAC max_error squared).
    /// - `plane_max_residual`: squared max transfer error for a correspondence to count as
    ///   lying on the dominant plane (degeneracy detection and homography refit).
    /// - `off_plane_min_residual`: squared min transfer error for a correspondence to be
    ///   used as an off-plane parallax source when recovering the epipole.
    /// - `min_sample_h_inlier_ratio`: fraction of a sample that must be consistent with a
    ///   single plane homography for the sample to be H-degenerate.
    /// - `max_plane_parallax_trials`: off-plane pairs sampled during completion.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        points1: &'a [Vector2d],
        points2: &'a [Vector2d],
        sampson_max_residual: f64,
        plane_max_residual: f64,
        off_plane_min_residual: f64,
        min_sample_h_inlier_ratio: f64,
        max_plane_parallax_trials: i32,
        use_sampson_refinement: bool,
    ) -> Self {
        Self {
            points1,
            points2,
            sampson_max_residual,
            plane_max_residual,
            off_plane_min_residual,
            min_sample_h_inlier_ratio,
            max_plane_parallax_trials,
            use_sampson_refinement,
        }
    }

    /// Port of `FundamentalMatrixDegensacEstimator::Refine`: local optimization over an
    /// inlier set, used by LO-RANSAC. Refits via `Estimate`, then polishes by minimizing the
    /// Sampson error, unless the refit is plane-degenerate: the Sampson cost is then nearly
    /// flat along the epipole, so refining it trades a large epipole excursion for a
    /// negligible cost reduction. `Ok(false)` (leaving `f` unchanged) when the refit yields
    /// no model.
    pub fn refine(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        f: &mut Matrix3d,
    ) -> Result<bool> {
        check_eq!(points1.len(), points2.len());

        // Refit the inlier set with the full degeneracy handling.
        let mut models = Vec::new();
        Estimator::estimate(self, points1, points2, &mut models)?;
        let Some(&refit) = models.first() else {
            return Ok(false);
        };
        *f = refit;

        // Test the refit rather than the incoming model, so the gate applies to the model
        // actually being polished.
        if self.use_sampson_refinement
            && !is_sample_h_degenerate(
                f,
                points1,
                points2,
                self.plane_max_residual,
                self.min_sample_h_inlier_ratio,
            )?
        {
            refine_fundamental_matrix_sampson(points1, points2, f)?;
        }
        Ok(true)
    }
}

// Squared Sampson error residuals over the given correspondences.
fn sampson_residuals(
    points1: &[Vector2d],
    points2: &[Vector2d],
    f: &Matrix3d,
    residuals: &mut Vec<f64>,
) -> Result<()> {
    check_eq!(points1.len(), points2.len());
    residuals.clear();
    residuals.extend(
        points1
            .iter()
            .zip(points2)
            .map(|(p1, p2)| compute_squared_sampson_error(p1.homogeneous(), p2.homogeneous(), f)),
    );
    Ok(())
}

impl<'a> Estimator for FundamentalMatrixDegensacEstimator<'a> {
    type X = Vector2d;
    type Y = Vector2d;
    type M = Matrix3d;

    const MIN_NUM_SAMPLES: usize = DEGENSAC_MIN_NUM_SAMPLES;

    /// Port of `FundamentalMatrixDegensacEstimator::Estimate`: estimate fundamental matrix
    /// hypotheses from a sample (minimal 7-point or non-minimal >= 8-point). For each
    /// hypothesis whose sample is H-degenerate, the plane-corrupted model is replaced by a
    /// plane-and-parallax completion, or dropped if no usable off-plane parallax exists.
    fn estimate(
        &mut self,
        sample_points1: &[Vector2d],
        sample_points2: &[Vector2d],
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        check_ge!(sample_points1.len(), DEGENSAC_MIN_NUM_SAMPLES);

        // Fit the fundamental matrix from the sample: the 7-point solver for a minimal
        // sample (yielding up to three roots), the 8-point solver otherwise (e.g. the
        // local-optimization inlier set).
        let mut sample_models = Vec::new();
        if sample_points1.len() == DEGENSAC_MIN_NUM_SAMPLES {
            FundamentalMatrixSevenPointEstimator.estimate(
                sample_points1,
                sample_points2,
                &mut sample_models,
            )?;
        } else {
            FundamentalMatrixEightPointEstimator.estimate(
                sample_points1,
                sample_points2,
                &mut sample_models,
            )?;
        }

        models.clear();
        models.reserve(sample_models.len());
        for sample_model in sample_models {
            let Some(plane_h) = detect_sample_h_degeneracy(
                &sample_model,
                sample_points1,
                sample_points2,
                self.plane_max_residual,
                self.min_sample_h_inlier_ratio,
            )?
            else {
                // Non-degenerate sample: keep the fitted hypothesis as-is.
                models.push(sample_model);
                continue;
            };
            // H-degenerate sample: replace the plane-corrupted hypothesis by a
            // plane-and-parallax completion, or drop it if there is no usable parallax.
            if let Some(completed_model) = fundamental_from_plane_and_parallax(
                &plane_h,
                self.points1,
                self.points2,
                self.sampson_max_residual,
                self.plane_max_residual,
                self.off_plane_min_residual,
                self.max_plane_parallax_trials,
            )? {
                models.push(completed_model);
            }
        }
        Ok(())
    }

    /// Port of `FundamentalMatrixDegensacEstimator::Residuals`: squared Sampson errors.
    fn residuals(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        f: &Matrix3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        sampson_residuals(points1, points2, f, residuals)
    }
}

impl<'a> LocalEstimator for FundamentalMatrixDegensacEstimator<'a> {
    type X = Vector2d;
    type Y = Vector2d;
    type M = Matrix3d;

    const MIN_NUM_SAMPLES: usize = DEGENSAC_MIN_NUM_SAMPLES;

    /// LO-RANSAC's local step: refines a copy of the current best model with
    /// [`FundamentalMatrixDegensacEstimator::refine`] and pushes it on success, as
    /// `loransac.h` does for an estimator with a `Refine` method.
    fn estimate_local(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        initial_model: &Matrix3d,
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        let mut refined_model = *initial_model;
        if self.refine(points1, points2, &mut refined_model)? {
            models.push(refined_model);
        }
        Ok(())
    }

    fn residuals(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        f: &Matrix3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        sampson_residuals(points1, points2, f, residuals)
    }
}

/// Port of `colmap::FundamentalMatrixDegensacOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FundamentalMatrixDegensacOptions {
    /// RANSAC options that control sampling, scoring, and termination. As in
    /// RANSAC/LO-RANSAC, `max_error` is a pixel error that is squared internally, since
    /// Sampson residuals are squared errors.
    pub ransac: RansacOptions,
    /// Maximum pixel error for a correspondence to count as lying on the dominant plane,
    /// used for the degeneracy test and the homography refit. Squared internally.
    /// Deliberately looser than the inlier threshold so noisy plane points are still
    /// recognized as coplanar. If <= 0, derived as sqrt(3) * ransac.max_error (i.e. 3x in
    /// squared-pixel units).
    pub plane_max_error: f64,
    /// Minimum pixel error for a correspondence to be used as an off-plane parallax source
    /// when recovering the epipole. Squared internally. Deliberately much larger than the
    /// inlier threshold so only clean, high-parallax points constrain the epipole
    /// (near-plane, low-parallax points are ignored). If <= 0, derived as
    /// 10 * ransac.max_error (100x squared).
    pub off_plane_min_error: f64,
    /// Fraction of a sample that must be consistent with a single plane homography for the
    /// sample to be considered H-degenerate. The default corresponds to the paper's "at
    /// least 5 of 7" criterion for a minimal sample.
    pub min_sample_h_inlier_ratio: f64,
    /// Maximum number of off-plane correspondence pairs sampled during the
    /// plane-and-parallax model completion to recover the epipole. Only a couple of
    /// off-plane inliers are needed, so a small budget suffices; the completed model is
    /// scored and locally optimized by the surrounding RANSAC anyway. The value 25 sits at
    /// the knee of a runtime/accuracy sweep: fewer trials erode success on the hardest >=98%-
    /// plane scenes (t=5: -3pts) with no runtime win (the inner pair loop is not the
    /// bottleneck), while more trials (or dynamic termination) yield no accuracy gain at
    /// added cost.
    pub max_plane_parallax_trials: i32,
    /// Polish the local-optimization refit by minimizing the Sampson error, on inlier sets
    /// that are not plane-degenerate.
    pub use_sampson_refinement: bool,
}

impl Default for FundamentalMatrixDegensacOptions {
    fn default() -> Self {
        Self {
            ransac: RansacOptions::default(),
            plane_max_error: -1.0,
            off_plane_min_error: -1.0,
            min_sample_h_inlier_ratio: 5.0 / 7.0,
            max_plane_parallax_trials: 25,
            use_sampson_refinement: true,
        }
    }
}

/// Port of `colmap::EstimateFundamentalMatrixDegensac`: robustly estimate the fundamental
/// matrix from corresponding image points using DEGENSAC inside LO-RANSAC (the DEGENSAC
/// estimator is used as both the hypothesis and the local-optimization solver).
pub fn estimate_fundamental_matrix_degensac(
    points1: &[Vector2d],
    points2: &[Vector2d],
    options: &FundamentalMatrixDegensacOptions,
) -> Result<RansacReport<Matrix3d, MEstimatorSupport>> {
    let sampson_max_residual = options.ransac.max_error * options.ransac.max_error;
    // A correspondence counts as on the dominant plane within a looser margin than the
    // inlier threshold; only points well off the plane serve as parallax.
    let plane_max_error = if options.plane_max_error > 0.0 {
        options.plane_max_error
    } else {
        fns::sqrt(3.0) * options.ransac.max_error
    };
    let off_plane_min_error = if options.off_plane_min_error > 0.0 {
        options.off_plane_min_error
    } else {
        10.0 * options.ransac.max_error
    };
    let plane_max_residual = plane_max_error * plane_max_error;
    let off_plane_min_residual = off_plane_min_error * off_plane_min_error;

    // The DEGENSAC estimator is used as BOTH the hypothesis and the local-optimization
    // solver, so the local optimization also applies the degeneracy handling instead of
    // re-fitting a plane-corrupted model.
    let estimator = FundamentalMatrixDegensacEstimator::new(
        points1,
        points2,
        sampson_max_residual,
        plane_max_residual,
        off_plane_min_residual,
        options.min_sample_h_inlier_ratio,
        options.max_plane_parallax_trials,
        options.use_sampson_refinement,
    );
    let mut ransac: LoRansac<_, _, MEstimatorSupportMeasurer, RandomSampler> =
        LoRansac::new(&options.ransac, estimator, estimator)?;
    ransac.estimate(points1, points2)
}
