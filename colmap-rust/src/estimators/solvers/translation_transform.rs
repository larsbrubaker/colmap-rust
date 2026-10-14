//! Port of COLMAP's `colmap/estimators/solvers/translation_transform.h`: the N-D translation
//! estimator (the difference of the point means). COLMAP instantiates only `kDim = 2`
//! (`TranslationTransformEstimator<2>`, run by LO-RANSAC in
//! `estimators/two_view_geometry.cc`), so that is the one ported, as
//! [`TranslationTransformEstimator`]; [`super::similarity_transform`] made the same choice for
//! `kDim = 3`. Port of colmap-sharp's `Estimators/Solvers/TranslationTransform.cs`. Tests:
//! `tests/estimators/translation_transform.rs` (translation_transform_test.cc 1:1).
//!
//! Tier A (exact): sums and one division per coordinate, in COLMAP's order.

use crate::linalg::Vector2d;
use crate::optim::Estimator;
use crate::{check_eq, Result};

/// Estimate a 2D translation between point pairs; the model is the translation vector. Port
/// of `colmap::TranslationTransformEstimator<2>`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TranslationTransformEstimator;

impl Estimator for TranslationTransformEstimator {
    type X = Vector2d;
    type Y = Vector2d;
    type M = Vector2d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 1;

    /// Estimate the translation as `mean(points2) - mean(points1)`.
    fn estimate(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        models: &mut Vec<Vector2d>,
    ) -> Result<()> {
        check_eq!(points1.len(), points2.len());

        models.clear();

        let mut mean_src = Vector2d::zeros();
        let mut mean_dst = Vector2d::zeros();

        for i in 0..points1.len() {
            mean_src += points1[i];
            mean_dst += points2[i];
        }

        mean_src /= points1.len() as f64;
        mean_dst /= points2.len() as f64;

        models.push(mean_dst - mean_src);
        Ok(())
    }

    /// The squared translation error of each pair, `|points2 - points1 - translation|^2`.
    fn residuals(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        translation: &Vector2d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        check_eq!(points1.len(), points2.len());

        residuals.resize(points1.len(), 0.0);

        for i in 0..points1.len() {
            let diff = points2[i] - points1[i] - *translation;
            residuals[i] = diff.squared_norm();
        }
        Ok(())
    }
}
