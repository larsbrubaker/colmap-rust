//! Port of COLMAP's `colmap/estimators/solvers/fundamental_matrix.h` and `.cc`: the 7-point
//! and 8-point fundamental matrix estimators, whose residual is the squared Sampson error of
//! [`crate::geometry::essential_matrix`]. The 8-point estimator normalizes with
//! [`crate::geometry::normalization`]; the 7-point one finds the rank-2 members of the
//! null-space pencil with [`crate::math::polynomial`]'s cubic solver. Consumers: the two-view
//! geometry and DEGENSAC estimators. Port of colmap-sharp's
//! `Estimators/Solvers/FundamentalMatrixEstimators.cs`. Tests:
//! `tests/estimators/fundamental_matrix.rs` (fundamental_matrix_test.cc).
//!
//! Also here: [`refine_fundamental_matrix_sampson`] and [`FundamentalMatrixSampsonEstimator`]
//! (the LO-RANSAC refiner), which run [`crate::optim::TinySolver`] on
//! [`TinyFundamentalSampsonErrorCostFunctor`] over the factorized manifold built from
//! `cost_functions/tiny_manifold.rs`. Port of colmap-sharp's `FundamentalMatrix` and
//! `FundamentalMatrixSampsonEstimator` in the same C# file.
//!
//! Tier B (tolerance): QR, SVD and a cubic. The refiner is Tier C (iterative).
//!
//! Translation notes:
//! - Neither minimal estimator has `Refine`, so LO-RANSAC uses them through
//!   [`crate::optim::EstimateAsLocal`]. [`FundamentalMatrixSampsonEstimator`] has only
//!   `Refine` and `Residuals`, so it implements [`crate::optim::LocalEstimator`] alone.
//! - The 7-point estimator takes its 2D null space from an unpivoted Householder QR where
//!   COLMAP uses `fullPivHouseholderQr` (docs/CPP_DIVERGENCES.md entry 260).
//! - Signs of null vectors: every null vector here (7-point QR basis, 8-point QR or SVD
//!   vector, and the U, V of the rank-2 projection) has an arbitrary sign. The results are F
//!   up to scale: the 7-point models are normalized to unit norm with a sign that depends on
//!   the basis, the 8-point model's sign follows the null vector's. The squared Sampson error
//!   is invariant to F -> -F, so RANSAC scores are sign-independent, and COLMAP's tests
//!   compare F / F(2, 2).

use crate::estimators::cost_functions::{
    TinyEigenQuaternionManifold, TinyFundamentalSampsonErrorCostFunctor, TinyProductManifold,
};
use crate::geometry::essential_matrix::compute_squared_sampson_errors;
use crate::geometry::normalization::center_and_normalize_image_points;
use crate::linalg::householder::{apply_q, factor_in_place};
use crate::linalg::{
    JacobiSvd, Matrix3d, MatrixXd, Quaterniond, Svd3d, SvdFactor, SvdOptions, Vector2d,
};
use crate::math::fns;
use crate::math::polynomial::find_cubic_polynomial_roots;
use crate::optim::tiny_solver::{TinyEuclideanManifold, TinySolver, TinySolverOptions};
use crate::optim::{Estimator, LocalEstimator};
use crate::{check_eq, check_ge, Result};

/// Fundamental matrix estimator from corresponding point pairs. This algorithm solves the
/// 7-Point problem and is based on the following paper:
///
///    Zhengyou Zhang and T. Kanade, Determining the Epipolar Geometry and its Uncertainty:
///    A Review, International Journal of Computer Vision, 1998.
///
/// Port of `colmap::FundamentalMatrixSevenPointEstimator`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FundamentalMatrixSevenPointEstimator;

/// Fundamental matrix estimator from corresponding point pairs. This algorithm solves the
/// 8-Point problem based on Hartley and Zisserman, Multiple View Geometry, algorithm 11.1,
/// page 282. Port of `colmap::FundamentalMatrixEightPointEstimator`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FundamentalMatrixEightPointEstimator;

impl Estimator for FundamentalMatrixSevenPointEstimator {
    type X = Vector2d;
    type Y = Vector2d;
    type M = Matrix3d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 7;

    /// Estimate either 1 or 3 possible fundamental matrix solutions from exactly 7
    /// corresponding points.
    fn estimate(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        check_eq!(points1.len(), 7);
        check_eq!(points2.len(), 7);

        models.clear();

        // Setup system of equations: [points2(i,:), 1]' * F * [points1(i,:), 1]'. A is 9 x 7,
        // column-major, one correspondence per column; entry 3c + r multiplies F(r, c).
        let mut a = [0.0; 9 * 7];
        for i in 0..7 {
            let p1 = points1[i];
            let p2 = points2[i].homogeneous();
            let col = &mut a[9 * i..9 * i + 9];
            for r in 0..3 {
                col[r] = p1.x * p2[r];
                col[3 + r] = p1.y * p2[r];
                col[6 + r] = p2[r];
            }
        }

        // 9 unknowns with 7 equations, so we have 2D null space: the last two columns of the
        // full Q of A = Q R are orthogonal to A's columns.
        let mut tau = [0.0; 7];
        factor_in_place(&mut a, 9, 7, &mut tau);
        let mut f1 = [0.0; 9];
        let mut f2 = [0.0; 9];
        f1[7] = 1.0;
        f2[8] = 1.0;
        apply_q(&a, 9, &tau, &mut f1);
        apply_q(&a, 9, &tau, &mut f2);

        // Normalize, such that lambda + mu = 1
        // and add constraint det(F) = det(lambda * f1 + (1 - lambda) * f2).
        for k in 0..9 {
            f1[k] -= f2[k];
        }

        let t0 = f1[4] * f1[8] - f1[5] * f1[7];
        let t1 = f1[3] * f1[8] - f1[5] * f1[6];
        let t2 = f1[3] * f1[7] - f1[4] * f1[6];
        let t3 = f2[4] * f2[8] - f2[5] * f2[7];
        let t4 = f2[3] * f2[8] - f2[5] * f2[6];
        let t5 = f2[3] * f2[7] - f2[4] * f2[6];

        let coeffs0 = f1[0] * t0 - f1[1] * t1 + f1[2] * t2;
        if coeffs0.abs() < 1e-16 {
            return Ok(());
        }

        let mut coeffs1 = f2[0] * t0 - f2[1] * t1 + f2[2] * t2
            - f2[3] * (f1[1] * f1[8] - f1[2] * f1[7])
            + f2[4] * (f1[0] * f1[8] - f1[2] * f1[6])
            - f2[5] * (f1[0] * f1[7] - f1[1] * f1[6])
            + f2[6] * (f1[1] * f1[5] - f1[2] * f1[4])
            - f2[7] * (f1[0] * f1[5] - f1[2] * f1[3])
            + f2[8] * (f1[0] * f1[4] - f1[1] * f1[3]);
        let mut coeffs2 = f1[0] * t3 - f1[1] * t4 + f1[2] * t5
            - f1[3] * (f2[1] * f2[8] - f2[2] * f2[7])
            + f1[4] * (f2[0] * f2[8] - f2[2] * f2[6])
            - f1[5] * (f2[0] * f2[7] - f2[1] * f2[6])
            + f1[6] * (f2[1] * f2[5] - f2[2] * f2[4])
            - f1[7] * (f2[0] * f2[5] - f2[2] * f2[3])
            + f1[8] * (f2[0] * f2[4] - f2[1] * f2[3]);
        let mut coeffs3 = f2[0] * t3 - f2[1] * t4 + f2[2] * t5;

        coeffs1 /= coeffs0;
        coeffs2 /= coeffs0;
        coeffs3 /= coeffs0;

        let (num_roots, roots) = find_cubic_polynomial_roots(coeffs1, coeffs2, coeffs3);

        models.reserve(num_roots);
        for i in 0..num_roots {
            let mut f = [0.0; 9];
            for k in 0..9 {
                f[k] = f1[k] * roots[i] + f2[k];
            }
            // Eigen's normalized(): a zero vector is returned unchanged.
            let squared_norm = crate::linalg::dot(&f, &f);
            if squared_norm > 0.0 {
                let norm = fns::sqrt(squared_norm);
                for value in &mut f {
                    *value /= norm;
                }
            }
            // Eigen::Map<const Eigen::Matrix3d>(F.data()): column-major.
            models.push(Matrix3d::from_column_major(f));
        }
        Ok(())
    }

    /// The squared Sampson error of each correspondence.
    fn residuals(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        f: &Matrix3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        *residuals = compute_squared_sampson_errors(points1, points2, f)?;
        Ok(())
    }
}

/// Row `[x2 x1^T, y2 x1^T, x1^T]` (x1 homogeneous) of the 8-point system `x2' F x1 = 0`, so
/// entry `3r + c` multiplies `F(r, c)`.
fn eight_point_row(p1: Vector2d, p2: Vector2d) -> [f64; 9] {
    let h1 = p1.homogeneous();
    [
        p2.x * h1.x,
        p2.x * h1.y,
        p2.x * h1.z,
        p2.y * h1.x,
        p2.y * h1.y,
        p2.y * h1.z,
        h1.x,
        h1.y,
        h1.z,
    ]
}

impl Estimator for FundamentalMatrixEightPointEstimator {
    type X = Vector2d;
    type Y = Vector2d;
    type M = Matrix3d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 8;

    /// Estimate the fundamental matrix from at least 8 corresponding points (one model).
    fn estimate(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        check_eq!(points1.len(), points2.len());
        check_ge!(points1.len(), 8);

        models.clear();

        let num_points = points1.len();

        // Center and normalize image points for better numerical stability.
        let (normed_points1, normed_from_orig1) = center_and_normalize_image_points(points1)?;
        let (normed_points2, normed_from_orig2) = center_and_normalize_image_points(points2)?;

        // Setup homogeneous linear equation as x2' * F * x1 = 0.
        // Solve for the nullspace of the constraint matrix.
        let mut null_vector = [0.0; 9];
        if num_points == 8 {
            // A.transpose().householderQr().householderQ().col(8). A^T is 9 x 8,
            // column-major: column i is row i of A.
            let mut at = [0.0; 9 * 8];
            for i in 0..8 {
                at[9 * i..9 * i + 9]
                    .copy_from_slice(&eight_point_row(normed_points1[i], normed_points2[i]));
            }
            let mut tau = [0.0; 8];
            factor_in_place(&mut at, 9, 8, &mut tau);
            null_vector[8] = 1.0;
            apply_q(&at, 9, &tau, &mut null_vector);
        } else {
            let mut a = MatrixXd::zeros(num_points, 9);
            for i in 0..num_points {
                let row = eight_point_row(normed_points1[i], normed_points2[i]);
                for (c, value) in row.into_iter().enumerate() {
                    a[(i, c)] = value;
                }
            }
            let svd = JacobiSvd::new(
                &a,
                SvdOptions {
                    u: SvdFactor::None,
                    v: SvdFactor::Full,
                },
            );
            null_vector.copy_from_slice(svd.matrix_v().column(8));
        }

        // Eigen::Map<const Matrix<double, 3, 3, RowMajor>>: the null vector holds Q row by row.
        let n = null_vector;
        let q = Matrix3d::new(n[0], n[1], n[2], n[3], n[4], n[5], n[6], n[7], n[8]);

        // Enforcing the internal constraint that two singular values must non-zero
        // and one must be zero.
        let svd = Svd3d::compute(&q);
        let mut singular_values = svd.singular_values;
        singular_values.z = 0.0;
        let f = svd.matrix_u * Matrix3d::from_diagonal(singular_values) * svd.matrix_v.transpose();

        models.push(normed_from_orig2.transpose() * f * normed_from_orig1);
        Ok(())
    }

    /// The squared Sampson error of each correspondence.
    fn residuals(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        f: &Matrix3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        *residuals = compute_squared_sampson_errors(points1, points2, f)?;
        Ok(())
    }
}

/// The 7-DoF manifold of a factorized fundamental matrix: the two rotations of its SVD on
/// SO(3), plus the singular value ratio as a 1-D Euclidean parameter. The ambient layout
/// matches [`TinyFundamentalSampsonErrorCostFunctor`]: [qU (xyzw), qV (xyzw), sigma]. C++'s
/// `ProductManifold<EigenQuaternionManifold, EigenQuaternionManifold, EuclideanManifold<1>>`,
/// nested as its recursion unrolls.
type FactorizedFundamentalMatrixManifold = TinyProductManifold<
    TinyEigenQuaternionManifold,
    TinyProductManifold<TinyEigenQuaternionManifold, TinyEuclideanManifold>,
>;

/// Factorize F = U * diag(1, sigma, 0) * V^T into [qU, qV, sigma], dropping any rank-3
/// component as the eight-point estimator does. U and V are sign corrected to proper
/// rotations so they can be represented by quaternions, which at most negates F and leaves
/// the Sampson error unchanged. Returns `None` for a zero or numerically rank-1 matrix, or
/// non-finite parameters.
fn factorize_fundamental_matrix(f: &Matrix3d) -> Option<[f64; 9]> {
    let svd = Svd3d::compute(f);
    let singular_values = svd.singular_values;
    // A zero or numerically rank-1 matrix has no meaningful factorization, as U's second
    // column is arbitrary once the second singular value vanishes.
    const MIN_SINGULAR_VALUE_RATIO: f64 = 1e-12;
    // `!(s0 > 0)` rather than `s0 <= 0`, so a NaN singular value is rejected too.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(singular_values.x > 0.0)
        || singular_values.y <= MIN_SINGULAR_VALUE_RATIO * singular_values.x
    {
        return None;
    }

    let mut u = svd.matrix_u;
    let mut v = svd.matrix_v;
    if u.determinant() < 0.0 {
        u = -u;
    }
    if v.determinant() < 0.0 {
        v = -v;
    }

    let q_u = Quaterniond::from_rotation_matrix(u).normalized();
    let q_v = Quaterniond::from_rotation_matrix(v).normalized();
    let params = [
        q_u.x,
        q_u.y,
        q_u.z,
        q_u.w,
        q_v.x,
        q_v.y,
        q_v.z,
        q_v.w,
        singular_values.y / singular_values.x,
    ];
    params
        .iter()
        .all(|value| value.is_finite())
        .then_some(params)
}

/// Rescale a normalizing transform and its points, keeping the centroid.
fn rescale_normalized_image_points(
    factor: f64,
    normed_points: &mut [Vector2d],
    normed_from_orig: &mut Matrix3d,
) {
    for point in normed_points.iter_mut() {
        *point *= factor;
    }
    // normed_from_orig->topRows<2>() *= factor.
    for r in 0..2 {
        for c in 0..3 {
            normed_from_orig[(r, c)] *= factor;
        }
    }
}

/// Refine a fundamental matrix in place by minimizing the Sampson error over the given
/// correspondences, starting from `f`.
///
/// Optimizes the SVD factorization of Bartoli and Sturm, "Non-Linear Estimation of the
/// Fundamental Matrix With Minimal Parameters", PAMI 2004, which keeps rank 2 and the scale
/// gauge exact at every iterate, unlike the 8-point algorithm, which truncates the smallest
/// singular value after the fact.
///
/// Points are centered and normalized internally, with a scale shared by both views to keep
/// the cost in pixel space. The fit is a plain least squares, so the points are expected to
/// be an inlier set; robustness comes from the surrounding RANSAC.
///
/// Returns false and leaves `f` unchanged if it cannot be factorized (zero or numerically
/// rank 1) or if the solve leaves non-finite parameters. Port of
/// `colmap::RefineFundamentalMatrixSampson`.
pub fn refine_fundamental_matrix_sampson(
    points1: &[Vector2d],
    points2: &[Vector2d],
    f: &mut Matrix3d,
) -> Result<bool> {
    check_eq!(points1.len(), points2.len());
    check_ge!(
        points1.len(),
        <FundamentalMatrixSampsonEstimator as LocalEstimator>::MIN_NUM_SAMPLES
    );

    // Center and normalize image points for better numerical stability, as in the
    // eight-point estimator. On raw pixel coordinates the 7x7 normal equations of the
    // refinement are severely ill-conditioned.
    let (mut normed_points1, mut normed_from_orig1) = center_and_normalize_image_points(points1)?;
    let (mut normed_points2, mut normed_from_orig2) = center_and_normalize_image_points(points2)?;

    // Rescale both views by a common factor. The Sampson error is invariant to a common
    // scale, but not to two different ones, which would weight the two terms of its
    // denominator differently.
    let scale1 = normed_from_orig1[(0, 0)];
    let scale2 = normed_from_orig2[(0, 0)];
    let scale = fns::sqrt(scale1 * scale2);
    rescale_normalized_image_points(scale / scale1, &mut normed_points1, &mut normed_from_orig1);
    rescale_normalized_image_points(scale / scale2, &mut normed_points2, &mut normed_from_orig2);

    // Inverse of the map that the eight-point estimator applies to its solution.
    let Some(mut params) = factorize_fundamental_matrix(
        &(normed_from_orig2.transpose().inverse() * *f * normed_from_orig1.inverse()),
    ) else {
        return Ok(false);
    };

    // Plain least squares: the points are assumed to be the inlier set, so robustness comes
    // from the RANSAC inlier selection.
    let functor = TinyFundamentalSampsonErrorCostFunctor::new(&normed_points1, &normed_points2)?;
    let manifold: FactorizedFundamentalMatrixManifold = TinyProductManifold::new(
        TinyEigenQuaternionManifold,
        TinyProductManifold::new(TinyEigenQuaternionManifold, TinyEuclideanManifold::new(1)?),
    );
    let mut solver = TinySolver::new(manifold)?;
    let options = TinySolverOptions {
        max_num_iterations: 25,
        ..TinySolverOptions::default()
    };
    solver.solve(&functor, &mut params, &options)?;

    if !params.iter().all(|value| value.is_finite()) {
        return Ok(false);
    }

    *f = normed_from_orig2.transpose()
        * TinyFundamentalSampsonErrorCostFunctor::fundamental_from_params(&params)
        * normed_from_orig1;
    Ok(true)
}

/// Fundamental matrix refiner for use as the local estimator of LO-RANSAC.
///
/// Provides `Refine` rather than `Estimate`, so LO-RANSAC passes the current best model as
/// the initial value, which a non-minimal solver cannot use. It therefore implements
/// [`LocalEstimator`] only. Port of `colmap::FundamentalMatrixSampsonEstimator`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FundamentalMatrixSampsonEstimator;

impl FundamentalMatrixSampsonEstimator {
    /// Refine `f` in place, see [`refine_fundamental_matrix_sampson`].
    pub fn refine(points1: &[Vector2d], points2: &[Vector2d], f: &mut Matrix3d) -> Result<bool> {
        refine_fundamental_matrix_sampson(points1, points2, f)
    }
}

impl LocalEstimator for FundamentalMatrixSampsonEstimator {
    type X = Vector2d;
    type Y = Vector2d;
    type M = Matrix3d;

    /// The minimum number of samples needed to refine a model. Refining an already
    /// determined model is only meaningful on an over-determined set.
    const MIN_NUM_SAMPLES: usize = 8;

    /// Refines a copy of `initial_model` and pushes it when [`Self::refine`] succeeds.
    fn estimate_local(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        initial_model: &Matrix3d,
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        let mut f = *initial_model;
        if Self::refine(points1, points2, &mut f)? {
            models.push(f);
        }
        Ok(())
    }

    /// Squared Sampson error residuals, matching the estimators above.
    fn residuals(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        f: &Matrix3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        *residuals = compute_squared_sampson_errors(points1, points2, f)?;
        Ok(())
    }
}
