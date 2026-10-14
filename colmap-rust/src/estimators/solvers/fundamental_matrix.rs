//! Port of COLMAP's `colmap/estimators/solvers/fundamental_matrix.h` and `.cc`: the 7-point
//! and 8-point fundamental matrix estimators, whose residual is the squared Sampson error of
//! [`crate::geometry::essential_matrix`]. The 8-point estimator normalizes with
//! [`crate::geometry::normalization`]; the 7-point one finds the rank-2 members of the
//! null-space pencil with [`crate::math::polynomial`]'s cubic solver. Consumers: the two-view
//! geometry and DEGENSAC estimators. Port of colmap-sharp's
//! `Estimators/Solvers/FundamentalMatrixEstimators.cs`. Tests:
//! `tests/estimators/fundamental_matrix.rs` (fundamental_matrix_test.cc).
//!
//! Not here yet: `RefineFundamentalMatrixSampson` and `FundamentalMatrixSampsonEstimator`
//! (the LO-RANSAC refiner), which run the tiny solver on `TinyFundamentalSampsonErrorCostFunctor`
//! over the factorized manifold of `cost_functions/tiny_manifold.h`.
//!
//! Tier B (tolerance): QR, SVD and a cubic.
//!
//! Translation notes:
//! - Neither estimator has `Refine`, so LO-RANSAC uses them through
//!   [`crate::optim::EstimateAsLocal`].
//! - The 7-point estimator takes its 2D null space from an unpivoted Householder QR where
//!   COLMAP uses `fullPivHouseholderQr` (docs/CPP_DIVERGENCES.md entry 260).
//! - Signs of null vectors: every null vector here (7-point QR basis, 8-point QR or SVD
//!   vector, and the U, V of the rank-2 projection) has an arbitrary sign. The results are F
//!   up to scale: the 7-point models are normalized to unit norm with a sign that depends on
//!   the basis, the 8-point model's sign follows the null vector's. The squared Sampson error
//!   is invariant to F -> -F, so RANSAC scores are sign-independent, and COLMAP's tests
//!   compare F / F(2, 2).

use crate::geometry::essential_matrix::compute_squared_sampson_errors;
use crate::geometry::normalization::center_and_normalize_image_points;
use crate::linalg::householder::{apply_q, factor_in_place};
use crate::linalg::{JacobiSvd, Matrix3d, MatrixXd, Svd3d, SvdFactor, SvdOptions, Vector2d};
use crate::math::fns;
use crate::math::polynomial::find_cubic_polynomial_roots;
use crate::optim::Estimator;
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
