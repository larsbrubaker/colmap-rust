//! Port of COLMAP's `colmap/estimators/solvers/similarity_transform.h` and `.cc` (plus the
//! `EuclideanTransformEstimator` alias of `euclidean_transform.h`): the 3D similarity and
//! rigid transform estimators (least-squares alignment of corresponding points, Umeyama 1991)
//! and [`estimate_rigid3d`] / [`estimate_sim3d`] with their LO-RANSAC robust variants. The
//! models feed [`Rigid3d`] and [`Sim3d`]; RANSAC runs them through
//! [`crate::optim::Estimator`]. Port of colmap-sharp's
//! `Estimators/Solvers/SimilarityTransform.cs`. Tests:
//! `tests/estimators/similarity_transform.rs` (similarity_transform_test.cc 1:1) and the
//! estimation cases of `tests/optim/ransac.rs` and `loransac.rs`.
//!
//! Tier B (tolerance): the solution goes through a 3x3 SVD; the robust variants are Tier C.
//!
//! The least-squares solution is written from the paper, not from `Eigen::umeyama` (Eigen is
//! MPL-2.0 and is not ported), as colmap-sharp's is: S. Umeyama, "Least-Squares Estimation of
//! Transformation Parameters Between Two Point Patterns", IEEE TPAMI 13(4), 1991, Theorem
//! and eqs. (34)-(42). With means `mu_x`, `mu_y`, source variance
//! `sigma_x^2 = 1/n sum |x - mu_x|^2` and covariance
//! `Sigma_xy = 1/n sum (y - mu_y)(x - mu_x)^T = U D V^T`: `R = U S V^T`,
//! `c = tr(D S) / sigma_x^2`, `t = mu_y - c R mu_x`, where `S = diag(1, 1, -1)` when
//! `det(U) det(V) < 0` and `I` otherwise. That sign test is the paper's rule
//! (`det(Sigma_xy) < 0`, or `det(U) det(V) = -1` when `Sigma_xy` has rank `m - 1`) stated so it
//! also covers the rank-deficient case, as COLMAP's Eigen call does.
//!
//! Translation notes:
//! - Only `kDim = 3` is ported: it is the only dimension COLMAP instantiates. The
//!   `kEstimateScale` template flag stays a const generic, so
//!   `SimilarityTransformEstimator<3>` is [`SimilarityTransformEstimator`] (scale on by
//!   default) and `SimilarityTransformEstimator<3, false>` /
//!   `EuclideanTransformEstimator<3>` is [`EuclideanTransformEstimator`].
//! - COLMAP's estimator has no `Refine`, so LO-RANSAC uses it as its own local estimator
//!   through [`EstimateAsLocal`].
//! - The free functions keep COLMAP's output parameters: [`estimate_rigid3d`] and
//!   [`estimate_sim3d`] leave `tgt_from_src` unchanged on failure,
//!   [`estimate_rigid3d_robust`] always assigns it (from a zero matrix when RANSAC fails) and
//!   [`estimate_sim3d_robust`] only on success, exactly as COLMAP does.
//! - The rank test of the 3 x N point matrices uses [`FullPivLu`], as COLMAP does.

use crate::geometry::{Rigid3d, Sim3d};
use crate::linalg::{FullPivLu, Matrix3d, Matrix3x4d, MatrixXd, Svd3d, Vector3d};
use crate::optim::{
    EstimateAsLocal, Estimator, InlierSupport, LoRansac, RansacOptions, RansacReport,
};
use crate::{check_eq, check_ge, Result};

/// 3D similarity transform estimator from corresponding point pairs in the source and
/// destination coordinate systems (Umeyama 1991). Port of
/// `colmap::SimilarityTransformEstimator<3, kEstimateScale>`. The model is the 3x4 matrix
/// `[c R | t]` mapping source to target (`c = 1` when `ESTIMATE_SCALE` is false).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SimilarityTransformEstimator<const ESTIMATE_SCALE: bool = true>;

/// 3D rigid (Euclidean) transform estimator: `colmap::EuclideanTransformEstimator<3>`, the
/// alias of `SimilarityTransformEstimator<3, false>`.
pub type EuclideanTransformEstimator = SimilarityTransformEstimator<false>;

/// The report of the robust estimators, `RANSAC<SimilarityTransformEstimator<3, ...>>::Report`.
pub type SimilarityTransformReport = RansacReport<Matrix3x4d, InlierSupport>;

impl<const ESTIMATE_SCALE: bool> Estimator for SimilarityTransformEstimator<ESTIMATE_SCALE> {
    type X = Vector3d;
    type Y = Vector3d;
    type M = Matrix3x4d;

    /// `kMinNumSamples = kDim`. Note that this only returns the true minimal sample in the
    /// two-dimensional case. For higher dimensions, the system will always be
    /// over-determined.
    const MIN_NUM_SAMPLES: usize = 3;

    /// Estimate the similarity transform: no model when either point set has rank below 3 or
    /// the solution has a NaN.
    fn estimate(
        &mut self,
        src: &[Vector3d],
        tgt: &[Vector3d],
        models: &mut Vec<Matrix3x4d>,
    ) -> Result<()> {
        check_eq!(src.len(), tgt.len());
        check_ge!(src.len(), Self::MIN_NUM_SAMPLES);

        models.clear();

        if FullPivLu::new(&point_matrix(src)).rank() < Self::MIN_NUM_SAMPLES
            || FullPivLu::new(&point_matrix(tgt)).rank() < Self::MIN_NUM_SAMPLES
        {
            return Ok(());
        }

        let sol = umeyama(src, tgt, ESTIMATE_SCALE);

        if sol.as_slice().iter().any(|v| v.is_nan()) {
            return Ok(());
        }

        models.push(sol);
        Ok(())
    }

    /// The squared transformation error of each pair when transforming the source to the
    /// destination coordinates, `|tgt - M [src; 1]|^2`.
    fn residuals(
        &mut self,
        src: &[Vector3d],
        tgt: &[Vector3d],
        tgt_from_src: &Matrix3x4d,
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

/// Least-squares rigid transform from at least 3 point pairs. Returns `false` (and leaves
/// `tgt_from_src` unchanged) for degenerate input. Port of `colmap::EstimateRigid3d`.
pub fn estimate_rigid3d(
    src: &[Vector3d],
    tgt: &[Vector3d],
    tgt_from_src: &mut Rigid3d,
) -> Result<bool> {
    let Some(matrix) = estimate_rigid_or_sim3d::<false>(src, tgt)? else {
        return Ok(false);
    };
    *tgt_from_src = Rigid3d::from_matrix(&matrix);
    Ok(true)
}

/// LO-RANSAC rigid transform. As in COLMAP, `tgt_from_src` is assigned even when the
/// estimation fails (from a zero matrix). Port of `colmap::EstimateRigid3dRobust`.
pub fn estimate_rigid3d_robust(
    src: &[Vector3d],
    tgt: &[Vector3d],
    options: &RansacOptions,
    tgt_from_src: &mut Rigid3d,
) -> Result<SimilarityTransformReport> {
    let (report, matrix) = estimate_rigid_or_sim3d_robust::<false>(src, tgt, options)?;
    *tgt_from_src = Rigid3d::from_matrix(&matrix);
    Ok(report)
}

/// Least-squares similarity transform from at least 3 point pairs. Returns `false` (and
/// leaves `tgt_from_src` unchanged) for degenerate input. Port of `colmap::EstimateSim3d`.
pub fn estimate_sim3d(
    src: &[Vector3d],
    tgt: &[Vector3d],
    tgt_from_src: &mut Sim3d,
) -> Result<bool> {
    let Some(matrix) = estimate_rigid_or_sim3d::<true>(src, tgt)? else {
        return Ok(false);
    };
    *tgt_from_src = Sim3d::from_matrix(&matrix);
    Ok(true)
}

/// LO-RANSAC similarity transform; `tgt_from_src` is assigned only on success. Port of
/// `colmap::EstimateSim3dRobust`.
pub fn estimate_sim3d_robust(
    src: &[Vector3d],
    tgt: &[Vector3d],
    options: &RansacOptions,
    tgt_from_src: &mut Sim3d,
) -> Result<SimilarityTransformReport> {
    let (report, matrix) = estimate_rigid_or_sim3d_robust::<true>(src, tgt, options)?;
    if report.success {
        *tgt_from_src = Sim3d::from_matrix(&matrix);
    }
    Ok(report)
}

/// `EstimateRigidOrSim3d<kEstimateScale>`: the single model, or `None` when there is none.
fn estimate_rigid_or_sim3d<const ESTIMATE_SCALE: bool>(
    src: &[Vector3d],
    tgt: &[Vector3d],
) -> Result<Option<Matrix3x4d>> {
    let mut models = Vec::new();
    SimilarityTransformEstimator::<ESTIMATE_SCALE>.estimate(src, tgt, &mut models)?;
    if models.is_empty() {
        return Ok(None);
    }
    check_eq!(models.len(), 1);
    Ok(Some(models[0]))
}

/// `EstimateRigidOrSim3dRobust<kEstimateScale>`: LO-RANSAC with the estimator as its own
/// local estimator; the matrix is the report's model on success and zero otherwise.
fn estimate_rigid_or_sim3d_robust<const ESTIMATE_SCALE: bool>(
    src: &[Vector3d],
    tgt: &[Vector3d],
    options: &RansacOptions,
) -> Result<(SimilarityTransformReport, Matrix3x4d)> {
    let estimator = SimilarityTransformEstimator::<ESTIMATE_SCALE>;
    let mut ransac =
        LoRansac::<_, EstimateAsLocal<_>>::new(options, estimator, EstimateAsLocal(estimator))?;
    let report = ransac.estimate(src, tgt)?;
    let tgt_from_src = if report.success {
        report.model
    } else {
        Matrix3x4d::zeros()
    };
    Ok((report, tgt_from_src))
}

/// Umeyama's least-squares `[c R | t]` with `tgt ~ c R src + t` (`c = 1` without scale); see
/// the file header for the equations.
fn umeyama(src: &[Vector3d], tgt: &[Vector3d], estimate_scale: bool) -> Matrix3x4d {
    let n = src.len();
    let one_over_n = 1.0 / n as f64;

    let mut src_sum = Vector3d::zeros();
    let mut tgt_sum = Vector3d::zeros();
    for i in 0..n {
        src_sum += src[i];
        tgt_sum += tgt[i];
    }

    let src_mean = src_sum * one_over_n;
    let tgt_mean = tgt_sum * one_over_n;

    // Sigma_xy = 1/n sum (y - mu_y)(x - mu_x)^T, accumulated column-major.
    let mut sigma = [0.0; 9];
    let mut src_variance_sum = 0.0;
    for i in 0..n {
        let xs = src[i] - src_mean;
        let ys = tgt[i] - tgt_mean;
        src_variance_sum += xs.squared_norm();
        for col in 0..3 {
            for row in 0..3 {
                sigma[col * 3 + row] += ys[row] * xs[col];
            }
        }
    }

    for value in &mut sigma {
        *value *= one_over_n;
    }

    let src_variance = src_variance_sum * one_over_n;

    let svd = Svd3d::compute(&Matrix3d::from_column_major(sigma));
    let u = svd.matrix_u;
    let v = svd.matrix_v;

    // S = diag(1, 1, -1) when U V^T would be a reflection.
    let s = Vector3d::new(
        1.0,
        1.0,
        if u.determinant() * v.determinant() < 0.0 {
            -1.0
        } else {
            1.0
        },
    );

    let rotation = u * Matrix3d::from_diagonal(s) * v.transpose();

    let scale = if estimate_scale {
        svd.singular_values.dot(s) / src_variance
    } else {
        1.0
    };

    let translation = tgt_mean - scale * (rotation * src_mean);
    Matrix3x4d::from_blocks(rotation * scale, translation)
}

/// The 3 x N matrix whose columns are the points.
fn point_matrix(points: &[Vector3d]) -> MatrixXd {
    let mut matrix = MatrixXd::zeros(3, points.len());
    for (j, p) in points.iter().enumerate() {
        matrix[(0, j)] = p.x;
        matrix[(1, j)] = p.y;
        matrix[(2, j)] = p.z;
    }
    matrix
}
