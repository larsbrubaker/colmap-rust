//! Port of COLMAP's `colmap/geometry/essential_matrix.h` and `essential_matrix.cc`: building
//! an essential matrix from a relative pose, decomposing it back into the four pose
//! candidates and picking one by cheirality, epipoles, optimal (Lindstrom) image corrections,
//! conversions to and from fundamental matrices, and the (tangent) Sampson errors the
//! relative-pose estimators score with. Port of colmap-sharp's `Geometry/EssentialMatrix.cs`.
//! Builds on [`super::pose`] (`check_cheirality`, [`CamRayWithJac`]) and [`super::rigid3`];
//! used by [`super::triangulation`] (`triangulate_optimal_point`). Tests:
//! `tests/geometry/essential_matrix.rs` (essential_matrix_test.cc) and
//! `tests/geometry/rust_only_two_view_oracle.rs`.
//!
//! Tiers: `essential_matrix_from_pose`, the Sampson errors and the fundamental conversions
//! are scalar products, Tier B because product grouping and the 3x3 inverse can differ from
//! Eigen's in the last bits. The oracle pins `essential_matrix_from_pose` and the Sampson
//! errors; the fundamental conversions are checked only by the ported tests.
//! `decompose_essential_matrix`,
//! `pose_from_essential_matrix` and `epipole_from_essential_matrix` go through the SVD and
//! are Tier B (docs/CPP_DIVERGENCES.md entry 30).
//!
//! SVD sign independence (singular vector signs are arbitrary and differ from Eigen's):
//! - `decompose_essential_matrix` makes U and V proper rotations first, as COLMAP does. For
//!   any such SVD of an essential matrix, {U W V^T, U W^T V^T} is the same pair of rotations
//!   (Hartley and Zisserman, "Multiple View Geometry", 2nd ed., Result 9.19) and t = +-u3,
//!   so the candidate *set* does not depend on the signs; only which rotation is called R1
//!   and the sign of t can differ from COLMAP. `pose_from_essential_matrix` therefore returns
//!   COLMAP's pose whenever one candidate has strictly the most points in front of both
//!   cameras; on a tie COLMAP keeps the last tied candidate in its order, which is
//!   sign-dependent (docs/CPP_DIVERGENCES.md entry 85).
//! - `epipole_from_essential_matrix` returns the null vector as the SVD gives it, so its
//!   overall sign is arbitrary, as in COLMAP (an epipole is a homogeneous point).

use super::pose::{check_cheirality, CamRayWithJac};
use super::rigid3::{cross_product_matrix, Rigid3d};
use crate::linalg::{
    Matrix2d, Matrix3d, Matrix3x2d, Quaterniond, Svd3d, Vector2d, Vector3d, Vector4d,
};
use crate::math::fns;
use crate::{check_eq, Result};

/// Port of `colmap::DecomposeEssentialMatrix`: the two possible rotations and the unit
/// translation direction (up to sign), `(R1, R2, t)`.
pub fn decompose_essential_matrix(e: &Matrix3d) -> (Matrix3d, Matrix3d, Vector3d) {
    let svd = Svd3d::compute(e);
    let mut u = svd.matrix_u;
    let mut v = svd.matrix_v.transpose();

    if u.determinant() < 0.0 {
        u *= -1.0;
    }
    if v.determinant() < 0.0 {
        v *= -1.0;
    }

    let w = Matrix3d::new(0.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 1.0);

    let r1 = u * w * v;
    let r2 = u * w.transpose() * v;
    let t = u.col(2).normalized();
    (r1, r2, t)
}

/// Port of `colmap::PoseFromEssentialMatrix`: the relative pose encoded by `e` whose
/// triangulated points lie in front of both cameras for the most ray pairs, and the indices
/// of those pairs. The ray lists must have equal length.
pub fn pose_from_essential_matrix(
    e: &Matrix3d,
    cam_rays1: &[Vector3d],
    cam_rays2: &[Vector3d],
) -> Result<(Rigid3d, Vec<usize>)> {
    check_eq!(cam_rays1.len(), cam_rays2.len());

    let (r1, r2, t) = decompose_essential_matrix(e);
    let quat1 = Quaterniond::from_rotation_matrix(r1);
    let quat2 = Quaterniond::from_rotation_matrix(r2);

    // Generate all possible pose combinations.
    let cams2_from_cams1 = [
        Rigid3d::new(quat1, t),
        Rigid3d::new(quat2, t),
        Rigid3d::new(quat1, -t),
        Rigid3d::new(quat2, -t),
    ];

    let mut cam2_from_cam1 = Rigid3d::identity();
    let mut valid_indices: Vec<usize> = Vec::new();
    for candidate in cams2_from_cams1 {
        let tentative_valid_indices = check_cheirality(&candidate, cam_rays1, cam_rays2)?;
        if tentative_valid_indices.len() >= valid_indices.len() {
            cam2_from_cam1 = candidate;
            valid_indices = tentative_valid_indices;
        }
    }
    Ok((cam2_from_cam1, valid_indices))
}

/// Port of `colmap::EssentialMatrixFromPose`: `E = [t / |t|]_x R`.
pub fn essential_matrix_from_pose(cam2_from_cam1: &Rigid3d) -> Matrix3d {
    cross_product_matrix(cam2_from_cam1.translation.normalized())
        * cam2_from_cam1.rotation.to_rotation_matrix()
}

/// Port of `colmap::FindOptimalImageObservations`: the observations closest to
/// `(point1, point2)` that satisfy the epipolar constraint exactly (Lindstrom,
/// "Triangulation made easy", CVPR 2010, one iteration). Returns
/// `(optimal_point1, optimal_point2)`.
pub fn find_optimal_image_observations(
    e: &Matrix3d,
    point1: Vector2d,
    point2: Vector2d,
) -> (Vector2d, Vector2d) {
    let point1_homogeneous = point1.homogeneous();
    let point2_homogeneous = point2.homogeneous();

    // Epipolar lines (S = [I2 | 0] keeps the first two coordinates).
    let mut n1 = (*e * point2_homogeneous).head2();
    let mut n2 = (e.transpose() * point1_homogeneous).head2();

    let e_tilde = Matrix2d::new(e[(0, 0)], e[(0, 1)], e[(1, 0)], e[(1, 1)]);

    let a = n1.dot(e_tilde * n2);
    let b = (n1.squared_norm() + n2.squared_norm()) / 2.0;
    let c = point1_homogeneous.dot(*e * point2_homogeneous);
    let d = fns::sqrt(b * b - a * c);
    let mut lambda = c / (b + d);

    let delta1 = lambda * n1;
    let delta2 = lambda * n2;

    n1 -= e_tilde * delta2;
    n2 -= e_tilde.transpose() * delta1;

    lambda *= (2.0 * d) / (n1.squared_norm() + n2.squared_norm());

    // (p_h - S^T * lambda * n).hnormalized(): S^T only touches the first two coordinates.
    let step1 = lambda * n1;
    let step2 = lambda * n2;
    let optimal_point1 = Vector3d::new(
        point1_homogeneous.x - step1.x,
        point1_homogeneous.y - step1.y,
        point1_homogeneous.z,
    )
    .hnormalized();
    let optimal_point2 = Vector3d::new(
        point2_homogeneous.x - step2.x,
        point2_homogeneous.y - step2.y,
        point2_homogeneous.z,
    )
    .hnormalized();
    (optimal_point1, optimal_point2)
}

/// Port of `colmap::EpipoleFromEssentialMatrix`: the epipole of the left image (null vector
/// of `E`) or the right image (null vector of `E^T`), a unit homogeneous vector of
/// arbitrary sign.
pub fn epipole_from_essential_matrix(e: &Matrix3d, left_image: bool) -> Vector3d {
    let svd = Svd3d::compute(&if left_image { *e } else { e.transpose() });
    svd.matrix_v.col(2)
}

/// Port of `colmap::InvertEssentialMatrix`: the essential matrix of the inverse pose, `E^T`.
pub fn invert_essential_matrix(e: &Matrix3d) -> Matrix3d {
    e.transpose()
}

/// Port of `colmap::FundamentalFromEssentialMatrix`: `F = K2^-T E K1^-1`.
pub fn fundamental_from_essential_matrix(k2: &Matrix3d, e: &Matrix3d, k1: &Matrix3d) -> Matrix3d {
    k2.transpose().inverse() * *e * k1.inverse()
}

/// Port of `colmap::EssentialFromFundamentalMatrix`: `E = K2^T F K1`.
pub fn essential_from_fundamental_matrix(k2: &Matrix3d, f: &Matrix3d, k1: &Matrix3d) -> Matrix3d {
    k2.transpose() * *f * *k1
}

/// Port of `colmap::ComputeSquaredSampsonError` (single homogeneous correspondence): the
/// squared Sampson error under `e` (or a fundamental matrix); `f64::MAX` when the gradient
/// vanishes.
pub fn compute_squared_sampson_error(point1: Vector3d, point2: Vector3d, e: &Matrix3d) -> f64 {
    let epipolar_line1 = *e * point1;
    let num = point2.dot(epipolar_line1);
    let denom = Vector4d::new(
        point2.dot(e.col(0)),
        point2.dot(e.col(1)),
        epipolar_line1.x,
        epipolar_line1.y,
    );
    let denom_sq_norm = denom.squared_norm();
    if denom_sq_norm == 0.0 {
        return f64::MAX;
    }
    num * num / denom_sq_norm
}

/// Port of `colmap::ComputeSquaredSampsonError` (`Vector2d` overload): the squared Sampson
/// errors of image-point correspondences. The lists must have equal length.
pub fn compute_squared_sampson_errors(
    points1: &[Vector2d],
    points2: &[Vector2d],
    e: &Matrix3d,
) -> Result<Vec<f64>> {
    check_eq!(points1.len(), points2.len());
    Ok(points1
        .iter()
        .zip(points2)
        .map(|(p1, p2)| compute_squared_sampson_error(p1.homogeneous(), p2.homogeneous(), e))
        .collect())
}

/// Port of `colmap::ComputeSquaredSampsonError` (`Vector3d` overload): the squared Sampson
/// errors of homogeneous correspondences. The lists must have equal length.
pub fn compute_squared_sampson_errors_homogeneous(
    points1: &[Vector3d],
    points2: &[Vector3d],
    e: &Matrix3d,
) -> Result<Vec<f64>> {
    check_eq!(points1.len(), points2.len());
    Ok(points1
        .iter()
        .zip(points2)
        .map(|(&p1, &p2)| compute_squared_sampson_error(p1, p2, e))
        .collect())
}

/// Port of `colmap::SquaredPixelGradientNorm`: `|J^T g|^2` for a 3x2 Jacobian, written out
/// as COLMAP does.
pub fn squared_pixel_gradient_norm(j: &Matrix3x2d, g: Vector3d) -> f64 {
    let gx = j[(0, 0)] * g.x + j[(1, 0)] * g.y + j[(2, 0)] * g.z;
    let gy = j[(0, 1)] * g.x + j[(1, 1)] * g.y + j[(2, 1)] * g.z;
    gx * gx + gy * gy
}

/// Port of `colmap::ComputeSquaredTangentSampsonError` (ray and Jacobian overload): the
/// Sampson error of a ray correspondence with the constraint gradients chained into pixel
/// space through the ray Jacobians; `f64::MAX` when they vanish.
pub fn compute_squared_tangent_sampson_error_rays(
    cam_ray1: Vector3d,
    j1: &Matrix3x2d,
    cam_ray2: Vector3d,
    j2: &Matrix3x2d,
    e: &Matrix3d,
) -> f64 {
    let e_ray1 = *e * cam_ray1;
    let et_ray2 = e.transpose() * cam_ray2;
    let num = cam_ray2.dot(e_ray1);
    // Chain the constraint gradients from ray space into pixel space. The
    // gradient w.r.t. ray1 is E^T ray2 and w.r.t. ray2 is E ray1.
    let denom_sq_norm =
        squared_pixel_gradient_norm(j1, et_ray2) + squared_pixel_gradient_norm(j2, e_ray1);
    if denom_sq_norm == 0.0 {
        return f64::MAX;
    }
    num * num / denom_sq_norm
}

/// Port of `colmap::ComputeSquaredTangentSampsonError` (`CamRayWithJac` overload).
pub fn compute_squared_tangent_sampson_error(
    cam_ray1_with_jac: &CamRayWithJac,
    cam_ray2_with_jac: &CamRayWithJac,
    e: &Matrix3d,
) -> f64 {
    compute_squared_tangent_sampson_error_rays(
        cam_ray1_with_jac.ray,
        &cam_ray1_with_jac.jacobian,
        cam_ray2_with_jac.ray,
        &cam_ray2_with_jac.jacobian,
        e,
    )
}

/// Port of `colmap::ComputeSquaredTangentSampsonError` (vector overload). The lists must
/// have equal length.
pub fn compute_squared_tangent_sampson_errors(
    cam_rays1_with_jac: &[CamRayWithJac],
    cam_rays2_with_jac: &[CamRayWithJac],
    e: &Matrix3d,
) -> Result<Vec<f64>> {
    check_eq!(cam_rays1_with_jac.len(), cam_rays2_with_jac.len());
    Ok(cam_rays1_with_jac
        .iter()
        .zip(cam_rays2_with_jac)
        .map(|(r1, r2)| compute_squared_tangent_sampson_error(r1, r2, e))
        .collect())
}

/// Port of `colmap::ComputeSquaredTangentSampsonErrorWithCheirality`: the tangent Sampson
/// errors, with `f64::MAX` for correspondences that the pose recovered from `e` puts behind
/// either camera. The lists must have equal length.
pub fn compute_squared_tangent_sampson_errors_with_cheirality(
    cam_rays1_with_jac: &[CamRayWithJac],
    cam_rays2_with_jac: &[CamRayWithJac],
    e: &Matrix3d,
) -> Result<Vec<f64>> {
    let num_rays = cam_rays1_with_jac.len();
    check_eq!(num_rays, cam_rays2_with_jac.len());

    // Recover the relative pose from E (resolving the four-fold decomposition
    // ambiguity by cheirality voting) and flag which correspondences triangulate
    // in front of both cameras. Only the bearings are materialized, since that is
    // all PoseFromEssentialMatrix needs; the Jacobians are read in place below.
    let rays1: Vec<Vector3d> = cam_rays1_with_jac.iter().map(|r| r.ray).collect();
    let rays2: Vec<Vector3d> = cam_rays2_with_jac.iter().map(|r| r.ray).collect();
    let (_, valid_indices) = pose_from_essential_matrix(e, &rays1, &rays2)?;

    // Correspondences behind either camera are not valid inliers for the relative
    // pose regardless of their residual, so they get an infinite residual.
    let mut residuals = vec![f64::MAX; num_rays];
    for idx in valid_indices {
        residuals[idx] = compute_squared_tangent_sampson_error(
            &cam_rays1_with_jac[idx],
            &cam_rays2_with_jac[idx],
            e,
        );
    }
    Ok(residuals)
}
