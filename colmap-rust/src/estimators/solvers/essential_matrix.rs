//! Port of COLMAP's `colmap/estimators/solvers/essential_matrix.h` / `.cc`, via colmap-sharp's
//! `Estimators/Solvers/EssentialMatrixEstimators.cs`: the five-point and eight-point essential
//! matrix estimators on camera rays, and [`EssentialMatrixTangentSampsonEstimator`], which
//! scores in pixels through the rays' unprojection Jacobians and refines with
//! [`crate::optim::TinySolver`] on
//! [`crate::estimators::cost_functions::TinyTangentSampsonErrorCostFunctor`] (the LO-RANSAC
//! local optimizer). Tests: `tests/estimators/essential_matrix.rs` (`essential_matrix_test.cc`).
//!
//! The minimal five-point case is PoseLib's `relpose_5pt` ([`super::poselib::relpose_5pt`]);
//! the over-determined case is COLMAP's own Nister solver, whose generated polynomial code
//! lives in `essential_matrix_poly{1,2,3}.rs` and `essential_matrix_coeffs.rs`
//! (`scripts/generate-essential-matrix-poly.py`).
//!
//! Tier B (QR, SVD, LU, a companion-matrix eigen solve); the refiner is Tier C. Every null
//! vector (QR, SVD) has an arbitrary sign, so E is determined up to sign, which the (tangent)
//! Sampson errors and COLMAP's tests (min of |E - E*| and |E + E*|) ignore.
//!
//! The five- and eight-point estimators have no residual in COLMAP (the former
//! Sampson-on-unit-bearings residual was retired; inlier scoring is in pixels through the
//! tangent Sampson estimator), so they are plain `estimate` functions, not
//! [`crate::optim::Estimator`]s.

use super::essential_matrix_coeffs::compute_determinant_coefficients;
use super::essential_matrix_poly1::fill_polynomial_matrix_part1;
use super::essential_matrix_poly2::fill_polynomial_matrix_part2;
use super::essential_matrix_poly3::fill_polynomial_matrix_part3;
use super::poselib::relpose_5pt::relpose_5pt;
use crate::estimators::cost_functions::{
    TinyEigenQuaternionManifold, TinyProductManifold, TinySphereManifold,
    TinyTangentSampsonErrorCostFunctor,
};
use crate::geometry::essential_matrix::{
    compute_squared_tangent_sampson_errors_with_cheirality, essential_matrix_from_pose,
    pose_from_essential_matrix,
};
use crate::geometry::pose::CamRayWithJac;
use crate::geometry::Rigid3d;
use crate::linalg::{
    HouseholderQr, JacobiSvd, Matrix3d, MatrixXd, PartialPivLu, Quaterniond, Svd3d, SvdFactor,
    SvdOptions, Vector3d, VectorXd,
};
use crate::math::fns;
use crate::math::polynomial::find_polynomial_roots_companion_matrix;
use crate::optim::{Estimator, LocalEstimator, TinySolver, TinySolverOptions};
use crate::{check_eq, check_ge, Result};

/// The 5-DoF manifold of a relative pose (rotation on SO(3), translation on the unit sphere),
/// matching the block layout of Rigid3d::params ([qx, qy, qz, qw, tx, ty, tz]).
type RelativePoseManifold = TinyProductManifold<TinyEigenQuaternionManifold, TinySphereManifold>;

/// `ComputeFullV` only.
const FULL_V: SvdOptions = SvdOptions {
    u: SvdFactor::None,
    v: SvdFactor::Full,
};

/// The N x 9 epipolar constraint system `cam_rays2(i)' * E * cam_rays1(i) = 0`: row i is
/// [x2 x1^T, y2 x1^T, z2 x1^T], so entry 3r + c multiplies E(r, c).
fn constraint_matrix(cam_rays1: &[Vector3d], cam_rays2: &[Vector3d]) -> MatrixXd {
    let mut q = MatrixXd::zeros(cam_rays1.len(), 9);
    for (i, (ray1, ray2)) in cam_rays1.iter().zip(cam_rays2).enumerate() {
        for r in 0..3 {
            for c in 0..3 {
                q[(i, 3 * r + c)] = ray2[r] * ray1[c];
            }
        }
    }
    q
}

/// `Eigen::Map<const Matrix<double, 3, 3, RowMajor>>(v)`: the 9-vector as a matrix, row by row.
fn row_major_3x3(v: &[f64]) -> Matrix3d {
    Matrix3d::new(v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[7], v[8])
}

/// Port of `colmap::EssentialMatrixFivePointEstimator`: essential matrices from corresponding
/// normalized camera ray pairs, solving the 5-Point problem of D. Nister, "An efficient
/// solution to the five-point relative pose problem", IEEE-T-PAMI, 26(6), 2004.
#[derive(Clone, Copy, Debug, Default)]
pub struct EssentialMatrixFivePointEstimator;

impl EssentialMatrixFivePointEstimator {
    /// The minimum number of samples needed to estimate a model.
    pub const MIN_NUM_SAMPLES: usize = 5;

    /// Estimates up to 10 possible essential matrix solutions from at least 5 corresponding
    /// camera rays into `models` (cleared first).
    pub fn estimate(
        cam_rays1: &[Vector3d],
        cam_rays2: &[Vector3d],
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        check_eq!(cam_rays1.len(), cam_rays2.len());
        check_ge!(cam_rays1.len(), Self::MIN_NUM_SAMPLES);
        models.clear();

        // PoseLib's 5-point solver only supports the minimal case. The non-minimal case
        // falls through to the SVD-based solver below.
        if cam_rays1.len() == Self::MIN_NUM_SAMPLES {
            let mut candidate_models = Vec::with_capacity(10);
            relpose_5pt(cam_rays1, cam_rays2, &mut candidate_models);
            // Keep only hypotheses whose minimal sample is in front of both cameras, pruning
            // geometrically invalid essential matrices before they are scored.
            for candidate_model in candidate_models {
                let (_, valid_indices) =
                    pose_from_essential_matrix(&candidate_model, cam_rays1, cam_rays2)?;
                if valid_indices.len() == Self::MIN_NUM_SAMPLES {
                    models.push(candidate_model);
                }
            }
            return Ok(());
        }

        Self::estimate_non_minimal(cam_rays1, cam_rays2, models)
    }

    fn estimate_non_minimal(
        cam_rays1: &[Vector3d],
        cam_rays2: &[Vector3d],
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        // Setup system of equations: cam_rays2(i)' * E * cam_rays1(i) = 0.
        let q = constraint_matrix(cam_rays1, cam_rays2);

        // Step 1: Extraction of the nullspace. The minimal case is handled by PoseLib above,
        // so we always reach this with an over-determined system. E (9 x 4, column-major)
        // holds the right singular vectors of the four smallest singular values.
        let v = JacobiSvd::new(&q, FULL_V).matrix_v();
        let mut e = [0.0; 36];
        for col in 0..4 {
            e[9 * col..9 * col + 9].copy_from_slice(v.column(5 + col));
        }

        // Step 2: Gauss-Jordan elimination with partial pivoting on A.
        let mut e2 = [0.0; 36];
        let mut e3 = [0.0; 36];
        for i in 0..36 {
            e2[i] = e[i] * e[i];
            e3[i] = e2[i] * e[i];
        }
        let mut a = [0.0; 200];
        fill_polynomial_matrix_part1(&mut a, &e, &e2, &e3);
        fill_polynomial_matrix_part2(&mut a, &e, &e2, &e3);
        fill_polynomial_matrix_part3(&mut a, &e, &e2, &e3);
        // AA = A.block<10, 10>(0, 0).partialPivLu().solve(A.block<10, 10>(0, 10)); both
        // blocks are contiguous column-major halves of A.
        let lu = PartialPivLu::new(&MatrixXd::from_column_major(10, 10, &a[..100]));
        let aa = lu.solve_matrix(&MatrixXd::from_column_major(10, 10, &a[100..]));

        // Step 3: Expansion of the determinant polynomial of the 3x3 polynomial matrix B to
        // obtain the tenth degree polynomial. B is 13 x 3, column-major.
        let mut b = [0.0; 39];
        for i in 0..3 {
            let b_col = &mut b[13 * i..13 * i + 13];
            b_col[0] = 0.0;
            b_col[4] = 0.0;
            b_col[8] = 0.0;
            for k in 0..3 {
                b_col[1 + k] = aa[(i * 2 + 4, k)];
                b_col[5 + k] = aa[(i * 2 + 4, 3 + k)];
            }
            for k in 0..4 {
                b_col[9 + k] = aa[(i * 2 + 4, 6 + k)];
            }
            for k in 0..3 {
                b_col[k] -= aa[(i * 2 + 5, k)];
                b_col[4 + k] -= aa[(i * 2 + 5, 3 + k)];
            }
            for k in 0..4 {
                b_col[8 + k] -= aa[(i * 2 + 5, 6 + k)];
            }
        }

        // Step 4: Extraction of roots from the degree 10 polynomial.
        let mut coeffs = [0.0; 11];
        compute_determinant_coefficients(&b, &mut coeffs);
        let Some(roots) =
            find_polynomial_roots_companion_matrix(&VectorXd::from_vec(coeffs.to_vec()))?
        else {
            return Ok(());
        };

        let num_roots = roots.real.len();
        models.reserve(num_roots);
        for i in 0..num_roots {
            const K_MAX_ROOT_IMAG: f64 = 1e-10;
            if roots.imag[i].abs() > K_MAX_ROOT_IMAG {
                continue;
            }

            let z1 = roots.real[i];
            let z2 = z1 * z1;
            let z3 = z2 * z1;
            let z4 = z3 * z1;

            // Bz(j, c) from column j of B.
            let mut bz = Matrix3d::zeros();
            for j in 0..3 {
                let bj = &b[13 * j..13 * j + 13];
                bz[(j, 0)] = bj[0] * z3 + bj[1] * z2 + bj[2] * z1 + bj[3];
                bz[(j, 1)] = bj[4] * z3 + bj[5] * z2 + bj[6] * z1 + bj[7];
                bz[(j, 2)] = bj[8] * z4 + bj[9] * z3 + bj[10] * z2 + bj[11] * z1 + bj[12];
            }

            let x = Svd3d::compute(&bz).matrix_v.col(2);

            const K_MAX_X3: f64 = 1e-10;
            if x.z.abs() < K_MAX_X3 {
                continue;
            }

            let mut model = [0.0; 9];
            let mut squared_norm = 0.0;
            for k in 0..9 {
                model[k] = e[k] * (x.x / x.z) + e[9 + k] * (x.y / x.z) + e[18 + k] * z1 + e[27 + k];
                squared_norm += model[k] * model[k];
            }
            // Eigen's normalized(): a zero vector is returned unchanged.
            if squared_norm > 0.0 {
                let norm = fns::sqrt(squared_norm);
                for value in &mut model {
                    *value /= norm;
                }
            }
            models.push(row_major_3x3(&model));
        }
        Ok(())
    }
}

/// Port of `colmap::EssentialMatrixEightPointEstimator`: the essential matrix from at least 8
/// corresponding normalized camera ray pairs, by Hartley and Zisserman, Multiple View
/// Geometry, algorithm 11.1, page 282.
#[derive(Clone, Copy, Debug, Default)]
pub struct EssentialMatrixEightPointEstimator;

impl EssentialMatrixEightPointEstimator {
    /// The minimum number of samples needed to estimate a model.
    pub const MIN_NUM_SAMPLES: usize = 8;

    /// Estimates the essential matrix (one model) into `models` (cleared first).
    pub fn estimate(
        cam_rays1: &[Vector3d],
        cam_rays2: &[Vector3d],
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        check_eq!(cam_rays1.len(), cam_rays2.len());
        check_ge!(cam_rays1.len(), 8);
        models.clear();

        // Setup homogeneous linear equation as x2' * E * x1 = 0.
        let a = constraint_matrix(cam_rays1, cam_rays2);

        // Solve for the nullspace of the constraint matrix.
        let q = if cam_rays1.len() == 8 {
            // The last column of the full Q of A^T = Q R is orthogonal to every constraint.
            let qq = HouseholderQr::new(&a.transpose()).householder_q();
            row_major_3x3(qq.column(8))
        } else {
            row_major_3x3(JacobiSvd::new(&a, FULL_V).matrix_v().column(8))
        };

        // Enforcing the internal constraint that two singular values must be non-zero and
        // one must be zero.
        let svd = Svd3d::compute(&q);
        let mut singular_values = svd.singular_values;
        singular_values.z = 0.0;
        let e = svd.matrix_u * Matrix3d::from_diagonal(singular_values) * svd.matrix_v.transpose();

        models.push(e);
        Ok(())
    }
}

/// Extract the bearings into the contiguous array the five-point solver expects. The
/// Jacobians play no part in estimation. They only affect scoring.
fn unpack_cam_rays_with_jac(cam_rays_with_jac: &[CamRayWithJac]) -> Vec<Vector3d> {
    cam_rays_with_jac.iter().map(|r| r.ray).collect()
}

/// Port of `colmap::EssentialMatrixTangentSampsonEstimator`: essential matrices scored by the
/// tangent Sampson error, i.e. in pixel units, for arbitrary central camera models. The model
/// is estimated by the same five-point solver as [`EssentialMatrixFivePointEstimator`]; only
/// the residual differs. Because the residual is measured in pixels rather than radians, the
/// RANSAC threshold is the plain pixel threshold and needs no per-camera conversion via
/// `Camera::CamFromImgThreshold`. As a [`LocalEstimator`] it refines a copy of the current
/// best model ([`Self::refine`], C++'s `Refine` hook of LO-RANSAC).
#[derive(Clone, Copy, Debug, Default)]
pub struct EssentialMatrixTangentSampsonEstimator;

impl EssentialMatrixTangentSampsonEstimator {
    /// Refines `e` in place by nonlinearly minimizing the pixel-unit tangent Sampson error
    /// over the given correspondences, starting from `e`. This is the local optimizer used by
    /// LO-RANSAC. Returns false and leaves `e` unchanged on a degenerate decomposition.
    pub fn refine(
        cam_rays1_with_jac: &[CamRayWithJac],
        cam_rays2_with_jac: &[CamRayWithJac],
        e: &mut Matrix3d,
    ) -> Result<bool> {
        check_eq!(cam_rays1_with_jac.len(), cam_rays2_with_jac.len());
        check_ge!(
            cam_rays1_with_jac.len(),
            EssentialMatrixFivePointEstimator::MIN_NUM_SAMPLES
        );

        // Decompose the initial E into a relative pose (resolving the four-fold ambiguity via
        // cheirality over the bearings).
        let rays1 = unpack_cam_rays_with_jac(cam_rays1_with_jac);
        let rays2 = unpack_cam_rays_with_jac(cam_rays2_with_jac);
        let (mut cam2_from_cam1, valid_indices) = pose_from_essential_matrix(e, &rays1, &rays2)?;
        if valid_indices.is_empty() {
            return Ok(false);
        }

        // Nonlinear pixel-space tangent Sampson refinement of the full 7-parameter pose via
        // TinySolver, applying the relative pose manifold. Plain least squares: robustness
        // comes from the RANSAC inlier selection.
        let f = TinyTangentSampsonErrorCostFunctor::new(cam_rays1_with_jac, cam_rays2_with_jac)?;
        let mut solver = TinySolver::new(RelativePoseManifold::default())?;
        let options = TinySolverOptions {
            max_num_iterations: 25,
            ..TinySolverOptions::default()
        };

        let q = cam2_from_cam1.rotation.normalized();
        let t = cam2_from_cam1.translation.normalized();
        let mut x = [q.x, q.y, q.z, q.w, t.x, t.y, t.z];
        solver.solve(&f, &mut x, &options)?;

        // Keep the refined pose only if the solve stayed finite.
        if x.iter().all(|v| v.is_finite()) {
            cam2_from_cam1 = Rigid3d::new(
                Quaterniond::new(x[3], x[0], x[1], x[2]).normalized(),
                Vector3d::new(x[4], x[5], x[6]),
            );
        }
        *e = essential_matrix_from_pose(&cam2_from_cam1);
        Ok(true)
    }
}

impl Estimator for EssentialMatrixTangentSampsonEstimator {
    type X = CamRayWithJac;
    type Y = CamRayWithJac;
    type M = Matrix3d;
    const MIN_NUM_SAMPLES: usize = 5;

    /// Up to 10 possible essential matrix solutions from a set of corresponding camera rays.
    /// The Jacobians are ignored here. They only affect scoring.
    fn estimate(
        &mut self,
        cam_rays1_with_jac: &[CamRayWithJac],
        cam_rays2_with_jac: &[CamRayWithJac],
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        let rays1 = unpack_cam_rays_with_jac(cam_rays1_with_jac);
        let rays2 = unpack_cam_rays_with_jac(cam_rays2_with_jac);
        EssentialMatrixFivePointEstimator::estimate(&rays1, &rays2, models)
    }

    /// The squared tangent Sampson error in squared pixels, additionally enforcing the
    /// cheirality constraint.
    fn residuals(
        &mut self,
        cam_rays1_with_jac: &[CamRayWithJac],
        cam_rays2_with_jac: &[CamRayWithJac],
        e: &Matrix3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        *residuals = compute_squared_tangent_sampson_errors_with_cheirality(
            cam_rays1_with_jac,
            cam_rays2_with_jac,
            e,
        )?;
        Ok(())
    }
}

impl LocalEstimator for EssentialMatrixTangentSampsonEstimator {
    type X = CamRayWithJac;
    type Y = CamRayWithJac;
    type M = Matrix3d;
    const MIN_NUM_SAMPLES: usize = 5;

    /// Refines a copy of `initial_model` and pushes it when [`Self::refine`] succeeds.
    fn estimate_local(
        &mut self,
        cam_rays1_with_jac: &[CamRayWithJac],
        cam_rays2_with_jac: &[CamRayWithJac],
        initial_model: &Matrix3d,
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        let mut e = *initial_model;
        if Self::refine(cam_rays1_with_jac, cam_rays2_with_jac, &mut e)? {
            models.push(e);
        }
        Ok(())
    }

    fn residuals(
        &mut self,
        cam_rays1_with_jac: &[CamRayWithJac],
        cam_rays2_with_jac: &[CamRayWithJac],
        e: &Matrix3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        Estimator::residuals(self, cam_rays1_with_jac, cam_rays2_with_jac, e, residuals)
    }
}
