//! Port of COLMAP's `colmap/estimators/cost_functions/tiny_sampson_error.h` (its closed-form
//! functors), via colmap-sharp's `Estimators/CostFunctions/TinyRelativePoseSampsonError.cs`
//! and `TinySampsonError.cs`: Sampson-error cost functions for [`crate::optim::TinySolver`],
//! each a [`TinySolverFunction`] with a closed-form Jacobian, refined over the manifolds of
//! [`super::tiny_manifold`]. Tests: `tests/estimators/tiny_sampson_error.rs`
//! (`tiny_sampson_error_test.cc`).
//!
//! - [`TinyTangentSampsonErrorCostFunctor`]: pixel-space tangent Sampson error of a calibrated
//!   relative pose (the refiner of `EssentialMatrixTangentSampsonEstimator`).
//! - [`TinyFocalSampsonErrorCostFunctor`]: pixel-space Sampson error of a pose plus a shared
//!   log-focal (the refiner of `relpose_shared_focal`). C++ differentiates it by autodiff;
//!   here its Jacobian is closed-form like the others (docs/CPP_DIVERGENCES.md, entry 252).
//! - [`TinyOneSidedFocalTangentSampsonErrorCostFunctor`]: a pose plus the first view's
//!   log-focal (the refiner of `relpose_one_sided_focal`).
//! - [`TinyFundamentalSampsonErrorCostFunctor`]: a fundamental matrix in Bartoli and Sturm's
//!   minimal SVD parameterization (the refiner of the fundamental matrix estimators).
//!
//! The one-sided functor also has C++'s generic (Jet) `operator()`, as an
//! [`AutoDiffFunctor`] built on [`super::sampson_error`], and its `AutoDiffFunction` wrapper
//! ([`TinyOneSidedFocalTangentSampsonErrorCostFunctor::auto_diff_function`]).
//!
//! The relative-pose functors build `E = [t]_x R` from the Rigid3d parameter layout
//! [qx, qy, qz, qw, tx, ty, tz]. Jacobians are column-major `num_residuals x NUM_PARAMETERS`.
//! The constructors check that the two correspondence lists have equal length (C++ leaves a
//! mismatch undefined).
//!
//! Tier C in use (an iterative solver's cost). The residuals are scalar formulas over 3x3
//! products; `cwiseProduct().sum()` is summed left to right in column-major order, where
//! Eigen may reduce in a different association.

use crate::geometry::pose::CamRayWithJac;
use crate::linalg::{Matrix3d, Matrix3x2d, Quaterniond, Vector2d, Vector3d};
use crate::math::fns;
use crate::optim::tiny_solver::TinySolverFunction;
use crate::solver::{AutoDiffFunctor, Scalar, TinySolverAutoDiffFunction};

use super::sampson_error::{essential_matrix_from_pose_params, tangent_sampson_error};
use crate::{check_eq, Result};

/// R of the Eigen-order quaternion `params[0..4]` (`Eigen::Map<const Quaterniond>`).
fn rotation_from_params(params: &[f64]) -> Matrix3d {
    Quaterniond::new(params[3], params[0], params[1], params[2]).to_rotation_matrix()
}

/// [t]_x of the translation `params[4..7]`.
fn cross_from_params(params: &[f64]) -> Matrix3d {
    Matrix3d::new(
        0.0, -params[6], params[5], //
        params[6], 0.0, -params[4], //
        -params[5], params[4], 0.0,
    )
}

/// dE/d[qx, qy, qz, qw, tx, ty, tz] of E = [t]_x R(q) into `d_e[0..7]`, assuming unit q.
fn essential_derivatives(params: &[f64], t_x: Matrix3d, r: Matrix3d, d_e: &mut [Matrix3d]) {
    let (x, y, z, w) = (params[0], params[1], params[2], params[3]);
    #[rustfmt::skip]
    let d_r = [
        // dR/dqx
        Matrix3d::new(0.0, 2.0 * y, 2.0 * z, 2.0 * y, -4.0 * x, -2.0 * w, 2.0 * z, 2.0 * w, -4.0 * x),
        // dR/dqy
        Matrix3d::new(-4.0 * y, 2.0 * x, 2.0 * w, 2.0 * x, 0.0, 2.0 * z, -2.0 * w, 2.0 * z, -4.0 * y),
        // dR/dqz
        Matrix3d::new(-4.0 * z, -2.0 * w, 2.0 * x, 2.0 * w, -4.0 * z, 2.0 * y, 2.0 * x, 2.0 * y, 0.0),
        // dR/dqw
        Matrix3d::new(0.0, -2.0 * z, 2.0 * y, 2.0 * z, 0.0, -2.0 * x, -2.0 * y, 2.0 * x, 0.0),
    ];
    for l in 0..4 {
        d_e[l] = t_x * d_r[l];
    }
    let ex = Matrix3d::new(0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0);
    let ey = Matrix3d::new(0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0);
    let ez = Matrix3d::new(0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    d_e[4] = ex * r; // dE/dtx
    d_e[5] = ey * r; // dE/dty
    d_e[6] = ez * r; // dE/dtz
}

/// J^T v for a 3x2 J.
fn transpose_times(j: &Matrix3x2d, v: Vector3d) -> Vector2d {
    Vector2d::new(
        j[(0, 0)] * v.x + j[(1, 0)] * v.y + j[(2, 0)] * v.z,
        j[(0, 1)] * v.x + j[(1, 1)] * v.y + j[(2, 1)] * v.z,
    )
}

/// a b^T.
fn outer(a: Vector3d, b: Vector3d) -> Matrix3d {
    Matrix3d::new(
        a.x * b.x,
        a.x * b.y,
        a.x * b.z,
        a.y * b.x,
        a.y * b.y,
        a.y * b.z,
        a.z * b.x,
        a.z * b.y,
        a.z * b.z,
    )
}

/// `a.cwiseProduct(b).sum()`, summed left to right in column-major order.
fn cwise_product_sum(a: &Matrix3d, b: &Matrix3d) -> f64 {
    let mut sum = a[(0, 0)] * b[(0, 0)];
    for k in 1..9 {
        let (row, col) = (k % 3, k / 3);
        sum += a[(row, col)] * b[(row, col)];
    }
    sum
}

/// `M.leftCols<2>() *= s`.
fn scale_left_cols(a: Matrix3d, s: f64) -> Matrix3d {
    let mut m = a;
    for row in 0..3 {
        m[(row, 0)] *= s;
        m[(row, 1)] *= s;
    }
    m
}

/// Writes residual `i` as zero and, with a Jacobian, zeroes its row: C++'s handling of a
/// vanishing denominator.
fn zero_residual(
    i: usize,
    n: usize,
    params: usize,
    residuals: &mut [f64],
    jac: Option<&mut [f64]>,
) {
    residuals[i] = 0.0;
    if let Some(jacobian) = jac {
        for l in 0..params {
            jacobian[i + l * n] = 0.0;
        }
    }
}

/// Port of `colmap::TinyTangentSampsonErrorCostFunctor`: tangent Sampson (pixel-space) cost
/// for TinySolver refinement of a two-view relative pose from calibrated rays with
/// unprojection Jacobians. `E = [t]_x R` is built from the pose `[qx, qy, qz, qw, tx, ty, tz]`
/// (Rigid3d params); the solver applies the manifold. Pixel-accurate for any central model,
/// matching EssentialMatrixTangentSampsonEstimator's score. The residual is
/// r = num / sqrt(denom); the 7-parameter Jacobian `dr/d[q,t]` is computed in closed form
/// (`dr/dE` contracted with `dE/d[q,t]`), ~3x faster than autodiff here.
#[derive(Clone, Copy, Debug)]
pub struct TinyTangentSampsonErrorCostFunctor<'a> {
    cam_rays1_with_jac: &'a [CamRayWithJac],
    cam_rays2_with_jac: &'a [CamRayWithJac],
}

impl<'a> TinyTangentSampsonErrorCostFunctor<'a> {
    /// The cost of the correspondences `cam_rays1_with_jac[i]` <-> `cam_rays2_with_jac[i]`.
    pub fn new(
        cam_rays1_with_jac: &'a [CamRayWithJac],
        cam_rays2_with_jac: &'a [CamRayWithJac],
    ) -> Result<Self> {
        check_eq!(cam_rays1_with_jac.len(), cam_rays2_with_jac.len());
        Ok(Self {
            cam_rays1_with_jac,
            cam_rays2_with_jac,
        })
    }
}

impl TinySolverFunction for TinyTangentSampsonErrorCostFunctor<'_> {
    const NUM_PARAMETERS: usize = 7;

    fn num_residuals(&self) -> usize {
        self.cam_rays1_with_jac.len()
    }

    fn evaluate(
        &self,
        params: &[f64],
        residuals: &mut [f64],
        mut jacobian: Option<&mut [f64]>,
    ) -> bool {
        let r = rotation_from_params(params);
        let t_x = cross_from_params(params);
        let e = t_x * r;

        let mut d_e = [Matrix3d::zeros(); 7];
        if jacobian.is_some() {
            essential_derivatives(params, t_x, r, &mut d_e);
        }

        let e_t = e.transpose();
        let n = self.cam_rays1_with_jac.len();
        for i in 0..n {
            let ray1 = self.cam_rays1_with_jac[i].ray;
            let ray2 = self.cam_rays2_with_jac[i].ray;
            let j1 = &self.cam_rays1_with_jac[i].jacobian;
            let j2 = &self.cam_rays2_with_jac[i].jacobian;
            let e_ray1 = e * ray1;
            let et_ray2 = e_t * ray2;
            let num = ray2.dot(e_ray1);
            let a = transpose_times(j1, et_ray2);
            let b = transpose_times(j2, e_ray1);
            let denom = a.squared_norm() + b.squared_norm();
            let sqrt_denom = fns::sqrt(denom);
            if sqrt_denom == 0.0 {
                zero_residual(i, n, 7, residuals, jacobian.as_deref_mut());
                continue;
            }
            residuals[i] = num / sqrt_denom;
            if let Some(jac) = jacobian.as_deref_mut() {
                // dr/dE = (1/sqrt_denom) ray2 ray1^T
                //         - (num/denom^1.5) (ray2 (J1 a)^T + (J2 b) ray1^T).
                let j1a = *j1 * a;
                let j2b = *j2 * b;
                let coef = num / (denom * sqrt_denom);
                let drd_e = (1.0 / sqrt_denom) * outer(ray2, ray1)
                    - coef * (outer(ray2, j1a) + outer(j2b, ray1));
                for (l, d) in d_e.iter().enumerate() {
                    jac[i + l * n] = cwise_product_sum(&drd_e, d);
                }
            }
        }
        true
    }
}

/// `F = diag(inv_f, inv_f, 1) * E * diag(inv_f, inv_f, 1)` in C++'s operation order: the
/// top-left 2x2 block times `inv_f * inv_f`, the rest of the last column and row times
/// `inv_f`.
fn shared_focal_scaled(e: Matrix3d, inv_f: f64) -> Matrix3d {
    let mut f = e;
    let inv_f_sq = inv_f * inv_f;
    for row in 0..2 {
        for col in 0..2 {
            f[(row, col)] *= inv_f_sq;
        }
    }
    for k in 0..2 {
        f[(k, 2)] *= inv_f;
        f[(2, k)] *= inv_f;
    }
    f
}

/// Port of `colmap::TinyFocalSampsonErrorCostFunctor`: Sampson-error cost for fixed-size
/// TinySolver refinement of a two-view relative pose *and* a shared, unknown focal length.
///
/// The pose is parameterized as [qx, qy, qz, qw, tx, ty, tz] and the focal is appended as an
/// eighth parameter optimized in log-space (log_f), which keeps it strictly positive and
/// gives a scale-invariant step. The essential matrix is built from the pose, then converted
/// to the fundamental matrix implied by the focal so the Sampson error is measured in *pixel*
/// space: F = diag(1/f, 1/f, 1) * E * diag(1/f, 1/f, 1), f = exp(log_f). The inputs
/// points1/points2 are therefore principal-point-centered image points (u - cx, v - cy), not
/// calibrated rays. The 6-DoF manifold (rotation on SO(3), translation on the unit sphere,
/// log-focal) is applied by the solver.
///
/// C++ differentiates this functor with Ceres autodiff
/// (`ceres::TinySolverAutoDiffFunction`). Here the 8-parameter Jacobian is closed-form,
/// dr/dF contracted with dF/d[q, t, log_f], as for the neighboring functors: the same
/// derivative up to rounding (docs/CPP_DIVERGENCES.md, entry 252). The residual follows the
/// C++ arithmetic of `SampsonError` exactly.
#[derive(Clone, Copy, Debug)]
pub struct TinyFocalSampsonErrorCostFunctor<'a> {
    points1: &'a [Vector2d],
    points2: &'a [Vector2d],
}

impl<'a> TinyFocalSampsonErrorCostFunctor<'a> {
    /// The cost of the centered image point correspondences `points1[i]` <-> `points2[i]`.
    pub fn new(points1: &'a [Vector2d], points2: &'a [Vector2d]) -> Result<Self> {
        check_eq!(points1.len(), points2.len());
        Ok(Self { points1, points2 })
    }
}

impl TinySolverFunction for TinyFocalSampsonErrorCostFunctor<'_> {
    const NUM_PARAMETERS: usize = 8;

    fn num_residuals(&self) -> usize {
        self.points1.len()
    }

    fn evaluate(
        &self,
        params: &[f64],
        residuals: &mut [f64],
        mut jacobian: Option<&mut [f64]>,
    ) -> bool {
        // Build E once from the pose, then scale it into the pixel-space fundamental matrix
        // F = diag(inv_f, inv_f, 1) * E * diag(inv_f, inv_f, 1).
        let r = rotation_from_params(params);
        let t_x = cross_from_params(params);
        let inv_f = fns::exp(-params[7]);
        let f = shared_focal_scaled(t_x * r, inv_f);

        let mut d_f = [Matrix3d::zeros(); 8];
        if jacobian.is_some() {
            essential_derivatives(params, t_x, r, &mut d_f);
            // The pose derivatives are of E, so carry them through the same scaling that
            // turns E into F.
            for d in d_f.iter_mut().take(7) {
                *d = shared_focal_scaled(*d, inv_f);
            }
            // d(inv_f^2)/d(log_f) = -2 inv_f^2 and d(inv_f)/d(log_f) = -inv_f, while F(2, 2)
            // carries no focal.
            let mut d7 = -f;
            for row in 0..2 {
                for col in 0..2 {
                    d7[(row, col)] *= 2.0;
                }
            }
            d7[(2, 2)] = 0.0;
            d_f[7] = d7;
        }

        let n = self.points1.len();
        for i in 0..n {
            // SampsonError: the constraint and the first two components of each view's
            // constraint gradient.
            let point1 = self.points1[i].homogeneous();
            let point2 = self.points2[i].homogeneous();
            let epipolar_line1 = f * point1;
            let num = point2.dot(epipolar_line1);
            let d0 = point2.dot(f.col(0));
            let d1 = point2.dot(f.col(1));
            // Eigen's unrolled (non-vectorized, as for the Jets C++ refines with) reduction
            // of the 4-vector's squared norm sums the halves first.
            let denom = (d0 * d0 + d1 * d1)
                + (epipolar_line1.x * epipolar_line1.x + epipolar_line1.y * epipolar_line1.y);
            let sqrt_denom = fns::sqrt(denom);
            if sqrt_denom == 0.0 {
                zero_residual(i, n, 8, residuals, jacobian.as_deref_mut());
                continue;
            }
            residuals[i] = num / sqrt_denom;
            if let Some(jac) = jacobian.as_deref_mut() {
                // dr/dF = (1/sqrt_denom) point2 point1^T
                //         - (num/denom^1.5) (point2 g1^T + g2 point1^T), where g1 and g2
                // are F^T point2 and F point1 with the third component dropped.
                let g1 = Vector3d::new(d0, d1, 0.0);
                let g2 = Vector3d::new(epipolar_line1.x, epipolar_line1.y, 0.0);
                let coef = num / (denom * sqrt_denom);
                let drd_f = (1.0 / sqrt_denom) * outer(point2, point1)
                    - coef * (outer(point2, g1) + outer(g2, point1));
                for (l, d) in d_f.iter().enumerate() {
                    jac[i + l * n] = cwise_product_sum(&drd_f, d);
                }
            }
        }
        true
    }
}

/// Port of `colmap::TinyOneSidedFocalTangentSampsonErrorCostFunctor` (its closed-form
/// evaluation): tangent Sampson (pixel-space) cost for TinySolver refinement of a two-view
/// relative pose and a *single* unknown focal length, where the second view is already
/// calibrated (the semi-calibrated, or "one-sided focal", configuration).
///
/// The pose is [qx, qy, qz, qw, tx, ty, tz] and the unknown focal of the *first* view is
/// appended as an eighth parameter optimized in log-space. Following ray2^T E b1 = 0 with
/// b1 = K1inv * point1, the mixed epipolar matrix is M = E * diag(1/f1, 1/f1, 1),
/// f1 = exp(log_f1), i.e. only columns 0-1 are scaled, and ray2^T M point1 = 0. The residual
/// is the tangent Sampson error of that constraint, in pixels: C = ray2^T M point1,
/// dC/dpx1 = (M^T ray2).head<2>(), dC/dpx2 = J2^T (M point1),
/// r = C / sqrt(||dC/dpx1||^2 + ||dC/dpx2||^2). Both denominator terms are gradients of the
/// same scalar with respect to pixels, so they share units and combine directly.
/// J2 = d(ray2)/d(pixel2) represents any central model exactly, including non-pinhole ones.
///
/// img_points1 are principal-point-centered points of the uncalibrated view. Keeping them as
/// raw pixels rather than bearings is deliberate: the measurement Jacobian
/// d(x, y, 1)/d(x, y) is then the constant [I2; 0] and f1 enters only through M. Unprojecting
/// the first view to a bearing instead would place f1 inside that view's Jacobian, so
/// differentiating the residual would need the mixed second derivative
/// d^2(ray1)/d(pixel1)d(f1), which no camera model exposes. For the same reason the
/// Jacobians hold only while the second view's intrinsics are fixed, as they are here. The
/// 8-parameter Jacobian is dr/dM contracted with dM/d[q, t, log_f1].
#[derive(Clone, Copy, Debug)]
pub struct TinyOneSidedFocalTangentSampsonErrorCostFunctor<'a> {
    img_points1: &'a [Vector2d],
    cam_rays2_with_jac: &'a [CamRayWithJac],
}

impl<'a> TinyOneSidedFocalTangentSampsonErrorCostFunctor<'a> {
    /// The cost of the correspondences `img_points1[i]` <-> `cam_rays2_with_jac[i]`.
    pub fn new(
        img_points1: &'a [Vector2d],
        cam_rays2_with_jac: &'a [CamRayWithJac],
    ) -> Result<Self> {
        check_eq!(img_points1.len(), cam_rays2_with_jac.len());
        Ok(Self {
            img_points1,
            cam_rays2_with_jac,
        })
    }
}

impl TinySolverFunction for TinyOneSidedFocalTangentSampsonErrorCostFunctor<'_> {
    const NUM_PARAMETERS: usize = 8;

    fn num_residuals(&self) -> usize {
        self.img_points1.len()
    }

    fn evaluate(
        &self,
        params: &[f64],
        residuals: &mut [f64],
        mut jacobian: Option<&mut [f64]>,
    ) -> bool {
        let r = rotation_from_params(params);
        let t_x = cross_from_params(params);
        let inv_f1 = fns::exp(-params[7]);
        let m = scale_left_cols(t_x * r, inv_f1);

        let mut d_m = [Matrix3d::zeros(); 8];
        if jacobian.is_some() {
            essential_derivatives(params, t_x, r, &mut d_m);
            // The pose derivatives are of E, so carry them through the same column scaling
            // that turns E into M.
            for d in d_m.iter_mut().take(7) {
                *d = scale_left_cols(*d, inv_f1);
            }
            // d(E * diag(1/f1, 1/f1, 1)) / d(log_f1) negates the scaled columns and leaves
            // the third one, which carries no focal, unchanged.
            let mut d7 = -m;
            for row in 0..3 {
                d7[(row, 2)] = 0.0;
            }
            d_m[7] = d7;
        }

        let m_t = m.transpose();
        let n = self.img_points1.len();
        for i in 0..n {
            let point1 = self.img_points1[i].homogeneous();
            let ray2 = self.cam_rays2_with_jac[i].ray;
            let j2 = &self.cam_rays2_with_jac[i].jacobian;
            let m_point1 = m * point1;
            let num = ray2.dot(m_point1);
            // Constraint gradients in view-1 and view-2 pixels. The former needs no
            // Jacobian, as d(x, y, 1)/d(x, y) merely selects the first two rows.
            let mt_ray2 = m_t * ray2;
            let g1 = Vector2d::new(mt_ray2.x, mt_ray2.y);
            let g2 = transpose_times(j2, m_point1);
            let denom = g1.squared_norm() + g2.squared_norm();
            let sqrt_denom = fns::sqrt(denom);
            if sqrt_denom == 0.0 {
                zero_residual(i, n, 8, residuals, jacobian.as_deref_mut());
                continue;
            }
            residuals[i] = num / sqrt_denom;
            if let Some(jac) = jacobian.as_deref_mut() {
                // dr/dM = (1/sqrt_denom) ray2 point1^T
                //         - (num/denom^1.5) (ray2 [g1; 0]^T + (J2 g2) point1^T).
                let j1g1 = Vector3d::new(g1.x, g1.y, 0.0);
                let j2g2 = *j2 * g2;
                let coef = num / (denom * sqrt_denom);
                let drd_m = (1.0 / sqrt_denom) * outer(ray2, point1)
                    - coef * (outer(ray2, j1g1) + outer(j2g2, point1));
                for (l, d) in d_m.iter().enumerate() {
                    jac[i + l * n] = cwise_product_sum(&drd_m, d);
                }
            }
        }
        true
    }
}

impl<'a> TinyOneSidedFocalTangentSampsonErrorCostFunctor<'a> {
    /// C++ `AutoDiffFunction`, `ceres::TinySolverAutoDiffFunction<Self, Dynamic, 8>`: the
    /// functor differentiated by Jets instead of its closed-form Jacobian. The solver uses
    /// the closed form; this only serves to pin it to a second implementation in tests.
    pub fn auto_diff_function(&self) -> TinySolverAutoDiffFunction<Self, 8> {
        TinySolverAutoDiffFunction::new(*self, self.img_points1.len())
    }
}

impl AutoDiffFunctor for TinyOneSidedFocalTangentSampsonErrorCostFunctor<'_> {
    /// The C++ `template <typename T> operator()`.
    fn call<T: Scalar>(&self, params: &[T], residuals: &mut [T]) -> bool {
        // Build E once from the pose, then apply the unknown focal to the columns to obtain
        // the matrix relating view-1 image points to view-2 rays.
        let mut m = essential_matrix_from_pose_params(params);
        let inv_f1 = (-params[7]).exp();
        for row in m.iter_mut() {
            row[0] = row[0] * inv_f1;
            row[1] = row[1] * inv_f1;
        }
        // Measurement Jacobian d(x, y, 1) / d(x, y) of the uncalibrated view; C++ builds it
        // as a Matrix<T, 3, 2>, whose entries are constants either way.
        let j1 = Matrix3x2d::new(1.0, 0.0, 0.0, 1.0, 0.0, 0.0);
        for (i, residual) in residuals
            .iter_mut()
            .enumerate()
            .take(self.img_points1.len())
        {
            let point1 = self.img_points1[i];
            let ray2 = self.cam_rays2_with_jac[i].ray;
            *residual = tangent_sampson_error(
                &m,
                [
                    T::from_f64(point1.x),
                    T::from_f64(point1.y),
                    T::from_f64(1.0),
                ],
                &j1,
                [
                    T::from_f64(ray2.x),
                    T::from_f64(ray2.y),
                    T::from_f64(ray2.z),
                ],
                &self.cam_rays2_with_jac[i].jacobian,
            );
        }
        true
    }
}

/// Port of `colmap::TinyFundamentalSampsonErrorCostFunctor`: Sampson-error cost for
/// TinySolver refinement of a fundamental matrix, parameterized by the SVD factorization of
/// Bartoli and Sturm, "Non-Linear Estimation of the Fundamental Matrix With Minimal
/// Parameters", PAMI 2004: F = U diag(1, sigma, 0) V^T = U0 V0^T + sigma U1 V1^T, where
/// U = R(qU), V = R(qV) and Uk, Vk are their k-th columns. The 9 ambient parameters are
/// [qU (xyzw), qV (xyzw), sigma]; the solver applies the 7-DoF manifold, so rank 2 holds
/// exactly at every iterate. Fixing the largest singular value to 1 is free, as the Sampson
/// error is scale invariant. The inputs are image points and the residual
/// r = num / sqrt(denom) is the signed Sampson error whose square
/// `ComputeSquaredSampsonError` scores. The 9-parameter Jacobian is dr/dF contracted with
/// dF/dparams.
#[derive(Clone, Copy, Debug)]
pub struct TinyFundamentalSampsonErrorCostFunctor<'a> {
    points1: &'a [Vector2d],
    points2: &'a [Vector2d],
}

impl<'a> TinyFundamentalSampsonErrorCostFunctor<'a> {
    /// The cost of the correspondences `points1[i]` <-> `points2[i]`.
    pub fn new(points1: &'a [Vector2d], points2: &'a [Vector2d]) -> Result<Self> {
        check_eq!(points1.len(), points2.len());
        Ok(Self { points1, points2 })
    }

    /// The fundamental matrix implied by the 9 ambient parameters (C++'s
    /// `FundamentalFromParams` without the optional rotation outputs).
    pub fn fundamental_from_params(params: &[f64]) -> Matrix3d {
        Self::fundamental_and_rotations_from_params(params).0
    }

    /// The fundamental matrix implied by the 9 ambient parameters, with the two rotations
    /// U and V, which only the Jacobian needs (C++'s `U_out` / `V_out`).
    pub fn fundamental_and_rotations_from_params(params: &[f64]) -> (Matrix3d, Matrix3d, Matrix3d) {
        let u = rotation_from_params(&params[0..4]);
        let v = rotation_from_params(&params[4..8]);
        let f = outer(u.col(0), v.col(0)) + params[8] * outer(u.col(1), v.col(1));
        (f, u, v)
    }

    /// Derivatives of the first two columns of R(q) w.r.t. the quaternion coefficients
    /// (x, y, z, w), assuming unit q. The factorization uses no other column.
    fn rotation_col01_derivatives(q: &[f64]) -> ([Vector3d; 4], [Vector3d; 4]) {
        let (x, y, z, w) = (q[0], q[1], q[2], q[3]);
        let dcol0 = [
            Vector3d::new(0.0, 2.0 * y, 2.0 * z),       // dR.col(0)/dqx
            Vector3d::new(-4.0 * y, 2.0 * x, -2.0 * w), // dR.col(0)/dqy
            Vector3d::new(-4.0 * z, 2.0 * w, 2.0 * x),  // dR.col(0)/dqz
            Vector3d::new(0.0, 2.0 * z, -2.0 * y),      // dR.col(0)/dqw
        ];
        let dcol1 = [
            Vector3d::new(2.0 * y, -4.0 * x, 2.0 * w),  // dR.col(1)/dqx
            Vector3d::new(2.0 * x, 0.0, 2.0 * z),       // dR.col(1)/dqy
            Vector3d::new(-2.0 * w, -4.0 * z, 2.0 * y), // dR.col(1)/dqz
            Vector3d::new(-2.0 * z, 0.0, 2.0 * x),      // dR.col(1)/dqw
        ];
        (dcol0, dcol1)
    }
}

impl TinySolverFunction for TinyFundamentalSampsonErrorCostFunctor<'_> {
    const NUM_PARAMETERS: usize = 9;

    fn num_residuals(&self) -> usize {
        self.points1.len()
    }

    fn evaluate(
        &self,
        params: &[f64],
        residuals: &mut [f64],
        mut jacobian: Option<&mut [f64]>,
    ) -> bool {
        let sigma = params[8];
        let (f, u, v) = Self::fundamental_and_rotations_from_params(params);

        let mut d_f = [Matrix3d::zeros(); 9];
        if jacobian.is_some() {
            let (d_u0, d_u1) = Self::rotation_col01_derivatives(&params[0..4]);
            let (d_v0, d_v1) = Self::rotation_col01_derivatives(&params[4..8]);
            for l in 0..4 {
                d_f[l] = outer(d_u0[l], v.col(0)) + sigma * outer(d_u1[l], v.col(1));
                d_f[4 + l] = outer(u.col(0), d_v0[l]) + sigma * outer(u.col(1), d_v1[l]);
            }
            d_f[8] = outer(u.col(1), v.col(1));
        }

        let f_t = f.transpose();
        let n = self.points1.len();
        for i in 0..n {
            let point1 = self.points1[i].homogeneous();
            let point2 = self.points2[i].homogeneous();
            let f_point1 = f * point1;
            let ft_point2 = f_t * point2;
            let num = point2.dot(f_point1);
            // Only the first two components of each constraint gradient enter, as the
            // homogeneous third coordinate is not a free variable.
            let denom = (f_point1.x * f_point1.x + f_point1.y * f_point1.y)
                + (ft_point2.x * ft_point2.x + ft_point2.y * ft_point2.y);
            let sqrt_denom = fns::sqrt(denom);
            if sqrt_denom == 0.0 {
                zero_residual(i, n, 9, residuals, jacobian.as_deref_mut());
                continue;
            }
            residuals[i] = num / sqrt_denom;
            if let Some(jac) = jacobian.as_deref_mut() {
                // dr/dF = (1/sqrt_denom) point2 point1^T
                //         - (num/denom^1.5) (g2 point1^T + point2 g1^T), where g1 and g2
                // are F^T point2 and F point1 with the third component dropped.
                let g1 = Vector3d::new(ft_point2.x, ft_point2.y, 0.0);
                let g2 = Vector3d::new(f_point1.x, f_point1.y, 0.0);
                let coef = num / (denom * sqrt_denom);
                let drd_f = (1.0 / sqrt_denom) * outer(point2, point1)
                    - coef * (outer(g2, point1) + outer(point2, g1));
                for (l, d) in d_f.iter().enumerate() {
                    jac[i + l * n] = cwise_product_sum(&drd_f, d);
                }
            }
        }
        true
    }
}
