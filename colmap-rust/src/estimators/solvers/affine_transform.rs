//! Port of COLMAP's `colmap/estimators/solvers/affine_transform.h` and `.cc`: the 2D affine
//! transform estimator (least squares from at least 3 correspondences) and
//! [`estimate_affine2d`] / [`estimate_affine2d_robust`] (its LO-RANSAC wrapper). COLMAP's
//! consumer is `retrieval/vote_and_verify.cc`; the 3D siblings live in
//! [`super::similarity_transform`]. Port of colmap-sharp's
//! `Estimators/Solvers/AffineTransform.cs`. Tests: `tests/estimators/affine_transform.rs`
//! (affine_transform_test.cc 1:1).
//!
//! Tier B (tolerance): an LU solve (minimal case) or an SVD least-squares solve; the robust
//! variant is Tier C.
//!
//! Translation notes:
//! - COLMAP's estimator has no `Refine`, so LO-RANSAC uses it as its own local estimator
//!   through [`EstimateAsLocal`].
//! - The over-determined case asks the SVD for thin U and V where COLMAP asks for full ones:
//!   `solve` reads only the first `min(rows, cols)` columns of U, which are computed the same
//!   way either way, and a full U of the 2N x 2N system would cost O(N^2) memory per local
//!   optimization.
//! - [`estimate_affine2d`] leaves `tgt_from_src` unchanged on failure and
//!   [`estimate_affine2d_robust`] assigns it only on success, as COLMAP does.

use crate::linalg::{
    JacobiSvd, Matrix2x3d, MatrixXd, PartialPivLu, SvdOptions, Vector2d, VectorXd,
};
use crate::optim::{
    EstimateAsLocal, Estimator, InlierSupport, LoRansac, RansacOptions, RansacReport,
};
use crate::{check_eq, check_ge, Result};

/// 2D affine transform estimator; the model is the 2x3 matrix `[A | t]` mapping source to
/// target. Port of `colmap::AffineTransformEstimator`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffineTransformEstimator;

/// The report of [`estimate_affine2d_robust`], `RANSAC<AffineTransformEstimator>::Report`.
pub type AffineTransformReport = RansacReport<Matrix2x3d, InlierSupport>;

impl Estimator for AffineTransformEstimator {
    type X = Vector2d;
    type Y = Vector2d;
    type M = Matrix2x3d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 3;

    /// Estimate the affine transformation from at least 3 correspondences. Pushes no model
    /// for a degenerate configuration.
    fn estimate(
        &mut self,
        src: &[Vector2d],
        tgt: &[Vector2d],
        tgt_from_src: &mut Vec<Matrix2x3d>,
    ) -> Result<()> {
        let num_points = src.len();
        check_eq!(num_points, tgt.len());
        check_ge!(num_points, 3);

        tgt_from_src.clear();

        // Sets up the linear system that we solve to obtain a least squared solution
        // for the affine transformation.
        let mut a = MatrixXd::zeros(2 * num_points, 6);
        let mut b = VectorXd::zeros(2 * num_points);
        for i in 0..num_points {
            let h = src[i].homogeneous();
            for k in 0..3 {
                a[(2 * i, k)] = h[k];
                a[(2 * i + 1, 3 + k)] = h[k];
            }
            b[2 * i] = tgt[i].x;
            b[2 * i + 1] = tgt[i].y;
        }

        let sol = if num_points == 3 {
            let sol = PartialPivLu::new(&a).solve(&b);
            if sol.as_slice().iter().any(|v| v.is_nan()) {
                return Ok(());
            }
            sol
        } else {
            let svd = JacobiSvd::new(&a, SvdOptions::THIN_UV);
            if svd.rank() < 6 {
                return Ok(());
            }
            svd.solve(&b)
        };

        // Eigen::Map<const Matrix<double, 3, 2>>(sol.data()).transpose(): the solution holds
        // the model row by row.
        let s = sol.as_slice();
        tgt_from_src.push(Matrix2x3d::new(s[0], s[1], s[2], s[3], s[4], s[5]));
        Ok(())
    }

    /// The squared transformation error of each correspondence, `|tgt - M [src; 1]|^2`.
    fn residuals(
        &mut self,
        src: &[Vector2d],
        tgt: &[Vector2d],
        tgt_from_src: &Matrix2x3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        let num_points = src.len();
        check_eq!(num_points, tgt.len());
        residuals.resize(num_points, 0.0);
        for i in 0..num_points {
            residuals[i] = (tgt[i] - *tgt_from_src * src[i].homogeneous()).squared_norm();
        }
        Ok(())
    }
}

/// Least-squares affine transform from at least 3 point pairs. Returns `false` (and leaves
/// `tgt_from_src` unchanged) for degenerate input. Port of `colmap::EstimateAffine2d`.
pub fn estimate_affine2d(
    src: &[Vector2d],
    tgt: &[Vector2d],
    tgt_from_src: &mut Matrix2x3d,
) -> Result<bool> {
    let mut models = Vec::new();
    AffineTransformEstimator.estimate(src, tgt, &mut models)?;
    if models.is_empty() {
        return Ok(false);
    }
    check_eq!(models.len(), 1);
    *tgt_from_src = models[0];
    Ok(true)
}

/// LO-RANSAC affine transform; `tgt_from_src` is assigned only on success. Port of
/// `colmap::EstimateAffine2dRobust`.
pub fn estimate_affine2d_robust(
    src: &[Vector2d],
    tgt: &[Vector2d],
    options: &RansacOptions,
    tgt_from_src: &mut Matrix2x3d,
) -> Result<AffineTransformReport> {
    let mut ransac = LoRansac::<_, EstimateAsLocal<_>>::new(
        options,
        AffineTransformEstimator,
        EstimateAsLocal(AffineTransformEstimator),
    )?;
    let report = ransac.estimate(src, tgt)?;
    if report.success {
        *tgt_from_src = report.model;
    }
    Ok(report)
}
