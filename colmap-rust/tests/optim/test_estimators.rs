// Test-only data and estimators for the RANSAC / LO-RANSAC tests (ransac.rs, loransac.rs,
// rust_only_ransac.rs):
// - `SimilarityTransformTestData`, `generate_test_data` and `validate_report`: the shared
//   helpers of COLMAP's ransac_test.cc and loransac_test.cc (identical in both files), run
//   with the production SimilarityTransformEstimator.
// - `LineEstimator` (two-point line through (x, y) pairs) and `LineLsqEstimator` (least
//   squares line, the LO-RANSAC local estimator) drive the Rust-only estimation loops in
//   rust_only_ransac.rs, with data built like COLMAP's GenerateTestData: the first
//   `num_outliers` targets are replaced by uniform draws far off the line. LoRansac takes
//   `LineLsqEstimator` through `EstimateAsLocal`.
// - `LineRefiner` is a Refine-only `LocalEstimator` (no `Estimator` impl).

#![allow(dead_code)]

use colmap_rust::geometry::Sim3d;
use colmap_rust::linalg::{Matrix3x4d, Vector3d};
use colmap_rust::math::fns;
use colmap_rust::math::random::{random_gaussian, random_uniform_real};
use colmap_rust::math::random_eigen::random_eigen_quaterniond;
use colmap_rust::optim::{Estimator, InlierSupport, LocalEstimator, RansacReport};
use colmap_rust::Result;
use std::cell::RefCell;
use std::rc::Rc;

/// COLMAP's `SimilarityTransformTestData`.
pub struct SimilarityTransformTestData {
    pub expected_tgt_from_src: Sim3d,
    pub src: Vec<Vector3d>,
    pub tgt: Vec<Vector3d>,
    pub num_samples: usize,
    pub num_outliers: usize,
}

/// COLMAP's `GenerateTestData(num_samples = 1000, num_outliers = 400)`: inliers mapped by a
/// random-rotation similarity with scale 2 and translation (100, 10, 10), then the first
/// `num_outliers` targets replaced by uniform draws far away.
pub fn generate_test_data(num_samples: usize, num_outliers: usize) -> SimilarityTransformTestData {
    let expected_tgt_from_src = Sim3d::new(
        2.0,
        random_eigen_quaterniond(),
        Vector3d::new(100.0, 10.0, 10.0),
    );

    let mut src = Vec::with_capacity(num_samples);
    let mut tgt = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let i = i as f64;
        src.push(Vector3d::new(
            i,
            fns::sqrt(i) + 2.0,
            fns::sqrt(2.0 * i + 2.0),
        ));
        tgt.push(expected_tgt_from_src * src[src.len() - 1]);
    }

    for target in tgt.iter_mut().take(num_outliers) {
        // Drawn in clang's left-to-right argument order.
        let x = random_uniform_real(-3000.0, -2000.0);
        let y = random_uniform_real(-4000.0, -3000.0);
        let z = random_uniform_real(-5000.0, -4000.0);
        *target = Vector3d::new(x, y, z);
    }

    SimilarityTransformTestData {
        expected_tgt_from_src,
        src,
        tgt,
        num_samples,
        num_outliers,
    }
}

/// COLMAP's `ValidateReport`: success, at least one trial, exactly the inliers found, and the
/// model within 1e-6 (Frobenius norm) of the expected matrix.
pub fn validate_report(
    report: &RansacReport<Matrix3x4d, InlierSupport>,
    data: &SimilarityTransformTestData,
) {
    assert!(report.success);
    assert!(report.num_trials > 0);

    assert_eq!(
        report.support.num_inliers,
        data.num_samples - data.num_outliers
    );
    for i in 0..data.num_samples {
        assert_eq!(report.inlier_mask[i], i >= data.num_outliers, "sample {i}");
    }

    let matrix_diff = (data.expected_tgt_from_src.to_matrix() - report.model).norm();
    assert!(matrix_diff < 1e-6, "matrix_diff = {matrix_diff}");
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
