//! Port of COLMAP's `colmap/estimators/solvers/homography_matrix.h` and `.cc`: the direct
//! linear transformation (DLT) homography estimators, [`HomographyMatrixEstimator`] between
//! image points and [`HomographyMatrixRayEstimator`] between bearing rays (whose residual
//! projects through the second [`Camera`]). RANSAC runs them through
//! [`crate::optim::Estimator`]; the decomposition of an estimated homography lives in
//! [`crate::geometry::homography_matrix`]. Port of colmap-sharp's
//! `Estimators/Solvers/HomographyMatrixEstimator.cs`. Tests:
//! `tests/estimators/homography_matrix.rs` (homography_matrix_test.cc 1:1).
//!
//! Tier B (tolerance): an 8 x 8 LU solve (minimal case) or the null space of an SVD.
//!
//! Translation notes:
//! - Neither C++ class has `Refine`, so LO-RANSAC uses them through
//!   [`crate::optim::EstimateAsLocal`].
//! - The over-determined case asks the SVD for V only, as COLMAP does (`ComputeFullV`).
//! - `HomographyMatrixRayEstimator` keeps a `const Camera*` that must outlive it; here it
//!   borrows the camera (`Option<&Camera>`, `None` for COLMAP's default `nullptr`), so the
//!   borrow checker enforces that. `Residuals` without a camera is COLMAP's
//!   `THROW_CHECK_NOTNULL` failure, an `Err`.
//! - `std::round` in the seam wrap rounds halfway cases away from zero, which is
//!   [`f64::round`].

use crate::linalg::{
    JacobiSvd, Matrix3d, MatrixXd, PartialPivLu, SvdFactor, SvdOptions, Vector2d, Vector3d,
    VectorXd,
};
use crate::optim::Estimator;
use crate::scene::camera::Camera;
use crate::{check_eq, check_ge, check_notnull, Result};

/// Direct linear transformation algorithm to compute the homography between point pairs.
/// This algorithm computes the least squares estimate for the homography from at least 4
/// correspondences. Port of `colmap::HomographyMatrixEstimator`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HomographyMatrixEstimator;

/// A ray together with the image point it was unprojected from. The ray drives the
/// estimation, the image point is what the residual is measured against. Port of
/// `colmap::CamRayWithImgPoint`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct CamRayWithImgPoint {
    /// The unit bearing ray in the second camera.
    pub ray: Vector3d,
    /// The image point the ray was unprojected from.
    pub img_point: Vector2d,
}

/// Direct linear transformation algorithm to compute the homography between bearing rays. A
/// world plane induces `x2 ~ H x1` on the rays of any central camera, whereas image points
/// are related by a 3x3 matrix only under a pinhole projection, which every model but
/// SIMPLE_PINHOLE and PINHOLE breaks.
///
/// The rays must come from cameras with known intrinsics. The estimated H maps rays to rays;
/// conjugate it as `K2 H K1^-1` for the pixel-space homography, which spherical cameras have
/// no calibration matrix for. Port of `colmap::HomographyMatrixRayEstimator`.
#[derive(Clone, Copy, Debug, Default)]
pub struct HomographyMatrixRayEstimator<'a> {
    camera2: Option<&'a Camera>,
}

impl<'a> HomographyMatrixRayEstimator<'a> {
    /// The estimator scoring residuals in `camera2`'s pixels (`None` is COLMAP's default
    /// `nullptr`: it can estimate but not score).
    pub fn new(camera2: Option<&'a Camera>) -> Self {
        Self { camera2 }
    }
}

/// `HasCollinearTriplet`: whether any three of the first four points are collinear (or two
/// coincide), relative to the scale of the triplet.
fn has_collinear_triplet(points: &[Vector2d]) -> bool {
    let is_collinear = |i: usize, j: usize, k: usize| {
        const K_MIN_NORMALIZED_AREA_SQUARED: f64 = 1e-24;
        let delta1 = points[j] - points[i];
        let delta2 = points[k] - points[i];
        let scale_squared = delta1.squared_norm() * delta2.squared_norm();
        let area = delta1.x * delta2.y - delta1.y * delta2.x;
        scale_squared == 0.0 || area * area <= K_MIN_NORMALIZED_AREA_SQUARED * scale_squared
    };
    is_collinear(0, 1, 2) || is_collinear(0, 1, 3) || is_collinear(0, 2, 3) || is_collinear(1, 2, 3)
}

/// Solves the `2N x 9` DLT system `A h = 0` shared by both estimators: the minimal case
/// fixes `h_8 = 1` and solves the leading 8 x 8 block by LU, the over-determined case takes
/// the right singular vector of the smallest singular value. `None` for a degenerate system
/// or a near-singular H (`|det H| < 1e-8`).
fn solve_dlt(a: &MatrixXd, num_points: usize) -> Option<Matrix3d> {
    let h = if num_points == 4 {
        // A.block<8, 8>(0, 0).partialPivLu().solve(-A.block<8, 1>(0, 8)).homogeneous().
        let lhs = a.block(0, 0, 8, 8);
        let mut rhs = VectorXd::zeros(8);
        for r in 0..8 {
            rhs[r] = -a[(r, 8)];
        }
        let sol = PartialPivLu::new(&lhs).solve(&rhs);
        let mut h = [1.0; 9];
        h[..8].copy_from_slice(sol.as_slice());
        if h.iter().any(|v| v.is_nan()) {
            return None;
        }
        h
    } else {
        // Solve for the nullspace of the constraint matrix.
        let svd = JacobiSvd::new(
            a,
            SvdOptions {
                u: SvdFactor::None,
                v: SvdFactor::Full,
            },
        );
        if svd.rank() < 8 {
            return None;
        }
        let nullspace = svd.matrix_v().col(8);
        let mut h = [0.0; 9];
        h.copy_from_slice(nullspace.as_slice());
        h
    };

    // Eigen::Map<const Eigen::Matrix3d>(h.data()).transpose(): h holds H row by row.
    let big_h = Matrix3d::new(h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7], h[8]);

    if big_h.determinant().abs() < 1e-8 {
        return None;
    }
    Some(big_h)
}

impl Estimator for HomographyMatrixEstimator {
    type X = Vector2d;
    type Y = Vector2d;
    type M = Matrix3d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 4;

    /// Estimate the projective transformation (homography) from at least 4 corresponding
    /// points; no model for a degenerate configuration.
    fn estimate(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        check_eq!(points1.len(), points2.len());
        check_ge!(points1.len(), 4);

        models.clear();

        let num_points = points1.len();
        // A minimal homography requires four points in general position (no three
        // collinear) in both images. See Hartley and Zisserman, Multiple View
        // Geometry in Computer Vision, 2nd ed., Sec. 4.1.3, pp. 91-92.
        if num_points == 4 && (has_collinear_triplet(points1) || has_collinear_triplet(points2)) {
            return Ok(());
        }

        // Setup constraint matrix.
        let mut a = MatrixXd::zeros(2 * num_points, 9);
        for i in 0..num_points {
            let p1 = points1[i].homogeneous();
            for k in 0..3 {
                a[(2 * i, k)] = p1[k];
                a[(2 * i, 6 + k)] = -points2[i].x * p1[k];
                a[(2 * i + 1, 3 + k)] = p1[k];
                a[(2 * i + 1, 6 + k)] = -points2[i].y * p1[k];
            }
        }

        if let Some(h) = solve_dlt(&a, num_points) {
            models.push(h);
        }
        Ok(())
    }

    /// The squared transformation error of each pair when transforming the source to the
    /// destination coordinates.
    fn residuals(
        &mut self,
        points1: &[Vector2d],
        points2: &[Vector2d],
        h: &Matrix3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        check_eq!(points1.len(), points2.len());

        residuals.resize(points1.len(), 0.0);

        // Note that this code might not be as nice as Eigen expressions,
        // but it is significantly faster in various tests.

        let h_00 = h[(0, 0)];
        let h_01 = h[(0, 1)];
        let h_02 = h[(0, 2)];
        let h_10 = h[(1, 0)];
        let h_11 = h[(1, 1)];
        let h_12 = h[(1, 2)];
        let h_20 = h[(2, 0)];
        let h_21 = h[(2, 1)];
        let h_22 = h[(2, 2)];

        for i in 0..points1.len() {
            let s_0 = points1[i].x;
            let s_1 = points1[i].y;
            let d_0 = points2[i].x;
            let d_1 = points2[i].y;

            let pd_0 = h_00 * s_0 + h_01 * s_1 + h_02;
            let pd_1 = h_10 * s_0 + h_11 * s_1 + h_12;
            let pd_2 = h_20 * s_0 + h_21 * s_1 + h_22;

            let inv_pd_2 = 1.0 / pd_2;
            let dd_0 = d_0 - pd_0 * inv_pd_2;
            let dd_1 = d_1 - pd_1 * inv_pd_2;

            residuals[i] = dd_0 * dd_0 + dd_1 * dd_1;
        }
        Ok(())
    }
}

impl Estimator for HomographyMatrixRayEstimator<'_> {
    type X = Vector3d;
    type Y = CamRayWithImgPoint;
    type M = Matrix3d;

    /// The minimum number of samples needed to estimate a model.
    const MIN_NUM_SAMPLES: usize = 4;

    /// Estimate the projective transformation (homography) between at least 4 corresponding
    /// bearing rays, with its global sign resolved so the rays map forward.
    fn estimate(
        &mut self,
        cam_rays1: &[Vector3d],
        cam_rays2: &[CamRayWithImgPoint],
        models: &mut Vec<Matrix3d>,
    ) -> Result<()> {
        check_eq!(cam_rays1.len(), cam_rays2.len());
        check_ge!(cam_rays1.len(), 4);

        models.clear();

        let num_rays = cam_rays1.len();

        // Setup constraint matrix from x2 x (H x1) = 0. Of the three equations, the
        // rows of [x2]_x, only two are independent, and the weakest is always the one
        // omitting the largest component of x2. For a perspective camera z dominates
        // and this reduces to the pixel estimator's rows.
        let mut a = MatrixXd::zeros(2 * num_rays, 9);
        for i in 0..num_rays {
            let ray1 = cam_rays1[i];
            let ray2 = cam_rays2[i].ray;

            // equations[row][col], a 3 x 9 block.
            let mut equations = [[0.0; 9]; 3];
            for k in 0..3 {
                equations[0][3 + k] = -ray2.z * ray1[k];
                equations[0][6 + k] = ray2.y * ray1[k];
                equations[1][k] = ray2.z * ray1[k];
                equations[1][6 + k] = -ray2.x * ray1[k];
                equations[2][k] = -ray2.y * ray1[k];
                equations[2][3 + k] = ray2.x * ray1[k];
            }

            // Equation j omits component j of x2, so the one to drop is the argmax (the
            // first one on ties, as Eigen's maxCoeff).
            let mut dropped_equation_idx = 0;
            let mut max_abs = ray2.x.abs();
            if ray2.y.abs() > max_abs {
                dropped_equation_idx = 1;
                max_abs = ray2.y.abs();
            }
            if ray2.z.abs() > max_abs {
                dropped_equation_idx = 2;
            }
            let mut num_kept = 0;
            for (j, equation) in equations.iter().enumerate() {
                if j != dropped_equation_idx {
                    for (col, &value) in equation.iter().enumerate() {
                        a[(2 * i + num_kept, col)] = value;
                    }
                    num_kept += 1;
                }
            }
        }

        let Some(mut h) = solve_dlt(&a, num_rays) else {
            return Ok(());
        };

        // H is defined up to scale, but the residual projects H x1 back into an image
        // that does not contain both a direction and its opposite, so the sign
        // matters. It is global, not per correspondence: a visible plane point has
        // positive depth, so x2 ~ lambda H x1 holds with lambda > 0 throughout and
        // the only freedom is that the solver may return -H. Resolving it per point
        // instead would score each against its nearer antipode, letting a 180 degree
        // error pass as a perfect inlier.
        let mut sign_votes = 0i32;
        for i in 0..num_rays {
            sign_votes += if (h * cam_rays1[i]).dot(cam_rays2[i].ray) > 0.0 {
                1
            } else {
                -1
            };
        }
        if sign_votes < 0 {
            h = -h;
        }

        models.push(h);
        Ok(())
    }

    /// The squared transformation error of each ray pair in the second camera's pixels: the
    /// first ray is transferred through H and projected back into the image. Unlike the
    /// essential matrix, a homography transfers a point to a point rather than to a line, so
    /// this is the exact geometric error and needs no linearization. Rays with no image
    /// point, i.e. transferred behind a perspective camera, are scored at the maximum
    /// residual (`f64::MAX`).
    fn residuals(
        &mut self,
        cam_rays1: &[Vector3d],
        cam_rays2: &[CamRayWithImgPoint],
        h: &Matrix3d,
        residuals: &mut Vec<f64>,
    ) -> Result<()> {
        check_eq!(cam_rays1.len(), cam_rays2.len());
        let camera2 = check_notnull!(self.camera2);

        residuals.resize(cam_rays1.len(), 0.0);

        // Azimuthal models wrap at the +-pi seam, where a raw pixel difference jumps
        // by about the image width. Wrap it into [-width/2, width/2), as
        // WrapEquirectangularHorizontalSeam does for the reprojection error, which
        // spells the same rounding as a floor since it must stay autodiff-safe.
        let is_periodic = camera2.is_spherical();
        let width = camera2.width as f64;

        for i in 0..cam_rays1.len() {
            let Some(img_point) = camera2.img_from_cam(*h * cam_rays1[i], true) else {
                // Transferred out of the camera's field, so there is nothing to score.
                residuals[i] = f64::MAX;
                continue;
            };
            let mut error = img_point - cam_rays2[i].img_point;
            if is_periodic {
                error.x -= width * (error.x / width).round();
            }
            residuals[i] = error.squared_norm();
        }
        Ok(())
    }
}
