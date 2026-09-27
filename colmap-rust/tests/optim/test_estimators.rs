// Test-only estimators and data for the RANSAC / LO-RANSAC tests (ransac.rs, loransac.rs,
// rust_only_ransac.rs). COLMAP's ransac_test.cc and loransac_test.cc use the production
// SimilarityTransformEstimator<3> (estimators/solvers/similarity_transform.h), which arrives
// with Phase 6; until then:
// - `MinSamples3Estimator` stands in for it where only its `kMinNumSamples = 3` matters
//   (the Report and NumTrials tests): a constant model, the mean of the sampled y.
// - `LineEstimator` (two-point line through (x, y) pairs) and `LineLsqEstimator` (least
//   squares line, the LO-RANSAC local estimator) drive the full estimation loops in
//   rust_only_ransac.rs, with data built like COLMAP's GenerateTestData: the first
//   `num_outliers` targets are replaced by uniform draws far off the line. LoRansac takes
//   `LineLsqEstimator` through `EstimateAsLocal`.
// - `LineRefiner` is a Refine-only `LocalEstimator` (no `Estimator` impl).

#![allow(dead_code)]

use colmap_rust::math::random::{random_gaussian, random_uniform_real};
use colmap_rust::optim::{Estimator, LocalEstimator};
use colmap_rust::Result;
use std::cell::RefCell;
use std::rc::Rc;

/// Stand-in for SimilarityTransformEstimator<3> where only kMinNumSamples = 3 is read.
#[derive(Clone, Debug, Default)]
pub struct MinSamples3Estimator;

impl Estimator for MinSamples3Estimator {
    type X = f64;
    type Y = f64;
    type M = f64;
    const MIN_NUM_SAMPLES: usize = 3;

    fn estimate(&mut self, _x: &[f64], y: &[f64], models: &mut Vec<f64>) -> Result<()> {
        models.push(y.iter().sum::<f64>() / y.len() as f64);
        Ok(())
    }

    fn residuals(
        &mut self,
        _x: &[f64],
        y: &[f64],
        model: &f64,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        residuals.clear();
        residuals.extend(y.iter().map(|&yi| (yi - model) * (yi - model)));
        Ok(())
    }
}

/// The line y = a * x + b.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Line {
    pub a: f64,
    pub b: f64,
}

fn line_residuals(x: &[f64], y: &[f64], model: &Line, residuals: &mut Vec<f64>) {
    residuals.clear();
    residuals.extend(x.iter().zip(y).map(|(&xi, &yi)| {
        let r = yi - (model.a * xi + model.b);
        r * r
    }));
}

/// Minimal estimator: the line through two points (no model for equal x).
#[derive(Clone, Debug, Default)]
pub struct LineEstimator;

impl Estimator for LineEstimator {
    type X = f64;
    type Y = f64;
    type M = Line;
    const MIN_NUM_SAMPLES: usize = 2;

    fn estimate(&mut self, x: &[f64], y: &[f64], models: &mut Vec<Line>) -> Result<()> {
        if x[0] != x[1] {
            let a = (y[1] - y[0]) / (x[1] - x[0]);
            models.push(Line {
                a,
                b: y[0] - a * x[0],
            });
        }
        Ok(())
    }

    fn residuals(
        &mut self,
        x: &[f64],
        y: &[f64],
        model: &Line,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        line_residuals(x, y, model, residuals);
        Ok(())
    }
}

/// Local estimator: the least-squares line through all given points.
#[derive(Clone, Debug, Default)]
pub struct LineLsqEstimator;

pub fn fit_line_lsq(x: &[f64], y: &[f64]) -> Option<Line> {
    let n = x.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (&xi, &yi) in x.iter().zip(y) {
        sx += xi;
        sy += yi;
        sxx += xi * xi;
        sxy += xi * yi;
    }
    let det = n * sxx - sx * sx;
    if det == 0.0 {
        return None;
    }
    let a = (n * sxy - sx * sy) / det;
    Some(Line {
        a,
        b: (sy - a * sx) / n,
    })
}

impl Estimator for LineLsqEstimator {
    type X = f64;
    type Y = f64;
    type M = Line;
    const MIN_NUM_SAMPLES: usize = 2;

    fn estimate(&mut self, x: &[f64], y: &[f64], models: &mut Vec<Line>) -> Result<()> {
        models.extend(fit_line_lsq(x, y));
        Ok(())
    }

    fn residuals(
        &mut self,
        x: &[f64],
        y: &[f64],
        model: &Line,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        line_residuals(x, y, model, residuals);
        Ok(())
    }
}

/// The line data of the rust-only RANSAC tests, shaped like COLMAP's
/// SimilarityTransformTestData.
pub struct LineTestData {
    pub expected: Line,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub num_samples: usize,
    pub num_outliers: usize,
}

/// Like COLMAP's GenerateTestData: inliers on y = 2x + 1 (plus Gaussian noise of standard
/// deviation `noise` when it is positive), then the first `num_outliers` targets replaced by
/// uniform draws in [-3000, -2000).
pub fn generate_line_data(num_samples: usize, num_outliers: usize, noise: f64) -> LineTestData {
    let expected = Line { a: 2.0, b: 1.0 };
    let x: Vec<f64> = (0..num_samples).map(|i| i as f64).collect();
    let mut y: Vec<f64> = x.iter().map(|&xi| expected.a * xi + expected.b).collect();
    if noise > 0.0 {
        for yi in &mut y {
            *yi += random_gaussian(0.0, noise);
        }
    }
    for yi in y.iter_mut().take(num_outliers) {
        *yi = random_uniform_real(-3000.0, -2000.0);
    }
    LineTestData {
        expected,
        x,
        y,
        num_samples,
        num_outliers,
    }
}

/// Refine-only local estimator (no `Estimator` impl), shaped like COLMAP's
/// FundamentalMatrixSampsonEstimator: `Refine` takes one Gauss-Newton step from the initial
/// line on the inliers, which for this linear model solves the least-squares problem. It
/// records every initial model it is handed (shared across clones, so the test sees the
/// worker copy's calls).
#[derive(Clone, Debug, Default)]
pub struct LineRefiner {
    pub initial_models: Rc<RefCell<Vec<Line>>>,
}

impl LineRefiner {
    // Shaped like `Refine(X, Y, M_t*)`: false when the normal equations are singular.
    fn refine(&self, x: &[f64], y: &[f64], model: &mut Line) -> bool {
        let n = x.len() as f64;
        let (mut sx, mut sxx, mut sr, mut sxr) = (0.0, 0.0, 0.0, 0.0);
        for (&xi, &yi) in x.iter().zip(y) {
            let r = yi - (model.a * xi + model.b);
            sx += xi;
            sxx += xi * xi;
            sr += r;
            sxr += xi * r;
        }
        let det = n * sxx - sx * sx;
        if det == 0.0 {
            return false;
        }
        let da = (n * sxr - sx * sr) / det;
        model.a += da;
        model.b += (sr - da * sx) / n;
        true
    }
}

impl LocalEstimator for LineRefiner {
    type X = f64;
    type Y = f64;
    type M = Line;
    const MIN_NUM_SAMPLES: usize = 2;

    fn estimate_local(
        &mut self,
        x: &[f64],
        y: &[f64],
        initial_model: &Line,
        models: &mut Vec<Line>,
    ) -> Result<()> {
        self.initial_models.borrow_mut().push(*initial_model);
        let mut refined = *initial_model;
        if self.refine(x, y, &mut refined) {
            models.push(refined);
        }
        Ok(())
    }

    fn residuals(
        &mut self,
        x: &[f64],
        y: &[f64],
        model: &Line,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        line_residuals(x, y, model, residuals);
        Ok(())
    }
}
