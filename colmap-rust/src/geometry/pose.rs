//! Port of COLMAP's `colmap/geometry/pose.h` and `pose.cc`: [`CamRayWithJac`], angle-axis
//! and Euler-angle conversions, the so(3) exponential and its left/right Jacobians, pose
//! interpolation, the cheirality test, re-expressing a camera pose in a
//! similarity-transformed world, the yaw-about-Y helpers, the closest rotation, projection
//! matrix decomposition, averaging of unit vectors and quaternions, and the gravity-aligned
//! frame. Port of colmap-sharp's `Geometry/Pose.cs` and `CamRayWithJac.cs`. Neighbors:
//! [`super::rigid3`], [`super::sim3`]; used by [`super::essential_matrix`] and
//! [`super::homography_matrix`]. Tests: `tests/geometry/pose.rs` (pose_test.cc) and
//! `tests/geometry/rust_only_two_view_oracle.rs`.
//!
//! Tiers: the scalar conversions (Euler angles, the Jacobians, the y-axis helpers,
//! `check_cheirality`) are Tier A (COLMAP's arithmetic order; transcendentals through
//! [`crate::math::fns`], docs/CPP_DIVERGENCES.md entry 1). Everything through an SVD or QR
//! (`compute_closest_rotation_matrix`, `decompose_projection_matrix`,
//! `average_unit_vectors`, `average_quaternions`, `gravity_aligned_rotation`) is Tier B
//! (entries 20 and 30). Sign independence: `compute_closest_rotation_matrix` uses `U V^T`,
//! in which each pair of flipped singular vectors cancels; `average_unit_vectors` fixes the
//! sign of the principal vector by COLMAP's weighted majority vote.

use super::rigid3::{cross_product_matrix, Rigid3d};
use super::sim3::Sim3d;
use crate::linalg::{
    AngleAxisd, HouseholderQr, JacobiSvd, Matrix3d, Matrix3x2d, Matrix3x4d, MatrixXd, Quaterniond,
    Svd3d, SvdOptions, Vector3d, VectorXd,
};
use crate::math::fns;
use crate::math::matrix::decompose_matrix_rq_3d;
use crate::{check, check_eq, check_gt, check_lt, Result};

/// A unit bearing and the Jacobian d(ray)/d(pixel) of its unprojection (see
/// `Camera::CamRayFromImgWithJac`), bundled so RANSAC subsampling keeps them index-aligned.
/// Port of `colmap::CamRayWithJac`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CamRayWithJac {
    /// The unit bearing.
    pub ray: Vector3d,
    /// d(ray)/d(pixel).
    pub jacobian: Matrix3x2d,
}

impl CamRayWithJac {
    /// `CamRayWithJac::Zero()`: the fallback when unprojection fails. The estimators take a
    /// dense list, not options, so a failed ray is kept as zero, which the tangent Sampson
    /// residual scores as infinite (rejected).
    pub const fn zero() -> Self {
        Self {
            ray: Vector3d::new(0.0, 0.0, 0.0),
            jacobian: Matrix3x2d::zeros(),
        }
    }
}

/// Port of `colmap::RotationMatrixToAngleAxis`: `angle * axis` of `Eigen::AngleAxisd(R)`.
pub fn rotation_matrix_to_angle_axis(r: &Matrix3d) -> Vector3d {
    let aa = AngleAxisd::from_rotation_matrix(*r);
    aa.angle * aa.axis
}

/// Port of `colmap::AngleAxisToRotationMatrix`. Below an angle of 1e-12 it uses the small
/// angle approximation `I + [w]_x`.
pub fn angle_axis_to_rotation_matrix(w: Vector3d) -> Matrix3d {
    let angle = w.norm();
    if angle > 1e-12 {
        AngleAxisd::new(angle, w / angle).to_rotation_matrix()
    } else {
        // Small angle approximation: I + [w]_x.
        Matrix3d::new(
            1.0, -w.z, w.y, //
            w.z, 1.0, -w.x, //
            -w.y, w.x, 1.0,
        )
    }
}

/// Port of `colmap::RotationMatrixToEulerAngles`: `(rx, ry, rz)` in radians for the
/// convention `R = Rz * Ry * Rx` (right-handed). A NaN angle becomes 0.
pub fn rotation_matrix_to_euler_angles(r: &Matrix3d) -> (f64, f64, f64) {
    let rx = fns::atan2(r[(2, 1)], r[(2, 2)]);
    let ry = fns::asin(-r[(2, 0)]);
    let rz = fns::atan2(r[(1, 0)], r[(0, 0)]);
    let zero_nan = |v: f64| if v.is_nan() { 0.0 } else { v };
    (zero_nan(rx), zero_nan(ry), zero_nan(rz))
}

/// Port of `colmap::EulerAnglesToRotationMatrix`: `Rz * Ry * Rx` from angles in radians.
pub fn euler_angles_to_rotation_matrix(rx: f64, ry: f64, rz: f64) -> Matrix3d {
    let r_x = AngleAxisd::new(rx, Vector3d::unit_x()).to_rotation_matrix();
    let r_y = AngleAxisd::new(ry, Vector3d::unit_y()).to_rotation_matrix();
    let r_z = AngleAxisd::new(rz, Vector3d::unit_z()).to_rotation_matrix();
    r_z * r_y * r_x
}

/// Below this rotation angle the so(3) helpers use their first-order expansions.
const SMALL_ANGLE_THRESHOLD: f64 = 1e-10;

/// Port of `colmap::QuaternionFromAngleAxis`: the quaternion of `omega` in so(3). Below
/// 1e-10 rad it uses the first-order expansion `(1, omega / 2)`, normalized, which keeps the
/// rotation direction instead of snapping to the identity.
pub fn quaternion_from_angle_axis(omega: Vector3d) -> Quaterniond {
    let theta = omega.norm();
    if theta < SMALL_ANGLE_THRESHOLD {
        // First-order Taylor expansion preserving rotation direction.
        return Quaterniond::new(1.0, 0.5 * omega.x, 0.5 * omega.y, 0.5 * omega.z).normalized();
    }
    Quaterniond::from_angle_axis(AngleAxisd::new(theta, omega / theta))
}

/// Port of `colmap::LeftJacobianFromAngleAxis`: the left Jacobian `J_l(omega)` of so(3),
/// used for bias Jacobian propagation in IMU preintegration.
pub fn left_jacobian_from_angle_axis(omega: Vector3d) -> Matrix3d {
    let theta = omega.norm();
    if theta < SMALL_ANGLE_THRESHOLD {
        return Matrix3d::identity() + 0.5 * cross_product_matrix(omega);
    }
    let a = omega / theta;
    let a_x = cross_product_matrix(a);
    let sin_theta = fns::sin(theta);
    let sinc_theta = sin_theta / theta;
    // Eigen evaluates `(1 - sinc) * a * a.transpose()` as the outer product of the scaled
    // vector `(1 - sinc) * a` with `a`.
    let scaled_a = (1.0 - sinc_theta) * a;
    let outer = Matrix3d::from_columns(scaled_a * a.x, scaled_a * a.y, scaled_a * a.z);
    sinc_theta * Matrix3d::identity() + outer + ((1.0 - fns::cos(theta)) / theta) * a_x
}

/// Port of `colmap::RightJacobianFromAngleAxis`: `J_r(omega) = J_l(-omega)`.
pub fn right_jacobian_from_angle_axis(omega: Vector3d) -> Matrix3d {
    left_jacobian_from_angle_axis(-omega)
}

/// Port of `colmap::InterpolateCameraPoses`: slerp of the rotations and linear
/// interpolation of the translations, `t` in [0, 1].
pub fn interpolate_camera_poses(
    cam1_from_world: &Rigid3d,
    cam2_from_world: &Rigid3d,
    t: f64,
) -> Rigid3d {
    let translation12 = cam2_from_world.translation - cam1_from_world.translation;
    Rigid3d::new(
        cam1_from_world.rotation.slerp(t, cam2_from_world.rotation),
        cam1_from_world.translation + translation12 * t,
    )
}

/// Port of `colmap::CheckCheirality`: the indices of the correspondences whose rays
/// triangulate to a point in front of both cameras. COLMAP's `bool` result is
/// `!valid_indices.empty()`.
///
/// The closed-form depth test assumes both rays are unit-normalized; rays of arbitrary scale
/// give incorrect results. The two lists must have equal length.
pub fn check_cheirality(
    cam2_from_cam1: &Rigid3d,
    cam_rays1: &[Vector3d],
    cam_rays2: &[Vector3d],
) -> Result<Vec<usize>> {
    check_eq!(cam_rays1.len(), cam_rays2.len());
    let cam2_from_cam1_rot = cam2_from_cam1.rotation.to_rotation_matrix();
    let t = cam2_from_cam1.translation;
    let mut valid_indices = Vec::new();
    for (i, (&ray1, &ray2)) in cam_rays1.iter().zip(cam_rays2).enumerate() {
        // Solve the 2x2 system for the depths of the point along both rays; both must be
        // positive for the point to lie in front of both cameras. This assumes unit-norm
        // rays: the common positive factor 1 / (1 - a^2) is dropped since it does not affect
        // the sign (a = cos angle between the rays, so |a| <= 1).
        let ray1_in_cam2 = cam2_from_cam1_rot * ray1;
        let a = -ray1_in_cam2.dot(ray2);
        let b1 = -ray1_in_cam2.dot(t);
        let b2 = ray2.dot(t);
        if b1 - a * b2 > 0.0 && b2 - a * b1 > 0.0 {
            valid_indices.push(i);
        }
    }
    Ok(valid_indices)
}

/// Port of `colmap::TransformCameraWorld`: `cam_from_world` re-expressed as
/// `cam_from_new_world`, given `new_from_old_world`.
pub fn transform_camera_world(new_from_old_world: &Sim3d, cam_from_world: &Rigid3d) -> Rigid3d {
    let cam_from_new_world = Sim3d::new(1.0, cam_from_world.rotation, cam_from_world.translation)
        * new_from_old_world.inverse();
    Rigid3d::new(
        cam_from_new_world.rotation,
        cam_from_new_world.translation * new_from_old_world.scale,
    )
}

/// Port of `colmap::YAxisAngleFromRotation`: the yaw (rotation about Y) of a gravity-aligned
/// rotation, the y component of its angle-axis vector.
pub fn y_axis_angle_from_rotation(rotation: &Matrix3d) -> f64 {
    rotation_matrix_to_angle_axis(rotation).y
}

/// Port of `colmap::RotationFromYAxisAngle`: the rotation by `angle` about the Y axis.
pub fn rotation_from_y_axis_angle(angle: f64) -> Matrix3d {
    angle_axis_to_rotation_matrix(Vector3d::new(0.0, angle, 0.0))
}

/// Port of `colmap::AverageUnitVectors`: the weighted average direction of the columns of
/// `vectors` (each normalized first), the principal left singular vector of
/// `A = N diag(w) N^T / sum(w)`, signed to agree with the weighted majority of the inputs.
/// An empty `weights` means uniform weights; otherwise its length must match and every
/// weight must be positive.
///
/// The SVD's singular-vector sign is arbitrary (docs/CPP_DIVERGENCES.md entry 30); the
/// majority vote fixes it as in COLMAP, so only an exact tie can leave it to the SVD.
pub fn average_unit_vectors(vectors: &MatrixXd, weights: &VectorXd) -> Result<VectorXd> {
    let count = vectors.cols();
    check_gt!(count, 0, "Cannot average empty set of vectors");
    check!(
        weights.is_empty() || weights.len() == count,
        "Weights size must match vectors size"
    );

    if count == 1 {
        return Ok(vectors.col(0).normalized());
    }

    // Determine weights: use provided weights or uniform weights.
    let w = if weights.is_empty() {
        VectorXd::ones(count)
    } else {
        weights.clone()
    };
    check!(
        w.as_slice().iter().all(|&v| v > 0.0),
        "Weights must be positive"
    );
    let weight_sum = sum_left_to_right(w.as_slice());

    // Normalize all columns and build weighted outer product sum matrix:
    // A = N * diag(w) * N^T / sum(w)
    let dim = vectors.rows();
    let mut normalized = MatrixXd::zeros(dim, count);
    let mut weighted = MatrixXd::zeros(dim, count);
    for j in 0..count {
        let column = vectors.col(j);
        let norm = column.norm();
        for i in 0..dim {
            normalized[(i, j)] = column[i] / norm;
            weighted[(i, j)] = normalized[(i, j)] * w[j];
        }
    }
    let a = &(&weighted * &normalized.transpose()) / weight_sum;

    // The first singular vector corresponds to the principal direction.
    let svd = JacobiSvd::new(
        &a,
        SvdOptions {
            full_u: true,
            ..SvdOptions::NONE
        },
    );
    let mut average = svd.matrix_u().col(0);

    // Ensure consistent sign by aligning with majority of input vectors.
    let dots = vectors.transpose_times_vector(&average);
    let mut negative_weight = 0.0;
    for i in 0..count {
        negative_weight += if dots[i] < 0.0 { 1.0 } else { 0.0 } * w[i];
    }
    if negative_weight > weight_sum - negative_weight {
        average = -&average;
    }
    Ok(average)
}

/// `values.sum()` left to right from the first term (Eigen's vectorized reduction order is
/// not reproduced, docs/CPP_DIVERGENCES.md entry 20).
fn sum_left_to_right(values: &[f64]) -> f64 {
    values.iter().skip(1).fold(values[0], |acc, &v| acc + v)
}

/// Port of `colmap::AverageDirections`: [`average_unit_vectors`] over 3D directions. An
/// empty `weights` means uniform weights.
pub fn average_directions(directions: &[Vector3d], weights: &[f64]) -> Result<Vector3d> {
    let mut mat = MatrixXd::zeros(3, directions.len());
    for (i, d) in directions.iter().enumerate() {
        mat.column_mut(i).copy_from_slice(&d.to_array());
    }
    Ok(average_unit_vectors(&mat, &VectorXd::from_slice(weights))?.to_vector3d())
}

/// Port of `colmap::ComputeClosestRotationMatrix`: the rotation closest to `matrix` in the
/// Frobenius norm, `U V^T` of its SVD, negated if that is a reflection. Sign-independent:
/// flipping a singular vector pair flips the matching columns of U and V together.
pub fn compute_closest_rotation_matrix(matrix: &Matrix3d) -> Matrix3d {
    let svd = Svd3d::compute(matrix);
    let mut r = svd.matrix_u * svd.matrix_v.transpose();
    if r.determinant() < 0.0 {
        r *= -1.0;
    }
    r
}

/// Port of `colmap::DecomposeProjectionMatrix`: `P = K [R | T]` with `K` upper triangular
/// with a positive diagonal. Returns `(K, R, T)`, or `None` if `K` is singular (COLMAP's
/// `false`).
pub fn decompose_projection_matrix(p: &Matrix3x4d) -> Option<(Matrix3d, Matrix3d, Vector3d)> {
    let (rr, qq) = decompose_matrix_rq_3d(&p.left_cols3());

    let mut r = compute_closest_rotation_matrix(&qq);

    let det_k = rr.determinant();
    if det_k == 0.0 {
        return None;
    }
    let mut k = if det_k > 0.0 { rr } else { -rr };

    for i in 0..3 {
        if k[(i, i)] < 0.0 {
            for row in 0..3 {
                k[(row, i)] = -k[(row, i)];
            }
            for col in 0..3 {
                r[(i, col)] = -r[(i, col)];
            }
        }
    }

    // K.triangularView<Eigen::Upper>().solve(P.col(3)): back substitution.
    let b = p.col(3);
    let t2 = b.z / k[(2, 2)];
    let t1 = (b.y - k[(1, 2)] * t2) / k[(1, 1)];
    let t0 = (b.x - k[(0, 1)] * t1 - k[(0, 2)] * t2) / k[(0, 0)];
    let mut t = Vector3d::new(t0, t1, t2);
    if det_k < 0.0 {
        t = -t;
    }
    Some((k, r, t))
}

/// Port of `colmap::AverageQuaternions`: [`average_unit_vectors`] on the normalized
/// quaternions' coefficients `(x, y, z, w)`. `weights` must match `quats` in length.
pub fn average_quaternions(quats: &[Quaterniond], weights: &[f64]) -> Result<Quaterniond> {
    check_eq!(quats.len(), weights.len());

    // Convert quaternions to coefficient matrix (each column is a quaternion).
    let mut qmat = MatrixXd::zeros(4, quats.len());
    for (i, q) in quats.iter().enumerate() {
        let n = q.normalized();
        qmat.column_mut(i).copy_from_slice(&[n.x, n.y, n.z, n.w]);
    }

    // Average using the unified unit vector averaging.
    let avg = average_unit_vectors(&qmat, &VectorXd::from_slice(weights))?;

    // Convert back to quaternion (Eigen order: x, y, z, w in coeffs).
    Ok(Quaterniond::new(avg[3], avg[0], avg[1], avg[2]))
}

/// Port of `colmap::GravityAlignedRotation`: a rotation whose second column is `gravity`
/// (which must be unit length), completed by an orthonormal basis of its complement from a
/// Householder QR, right-handed.
pub fn gravity_aligned_rotation(gravity: Vector3d) -> Result<Matrix3d> {
    check_lt!(
        (gravity.norm() - 1.0).abs(),
        1e-6,
        "Gravity vector must be normalized"
    );

    // Use Householder QR to find orthonormal basis vectors for the null space.
    let qr = HouseholderQr::new(&MatrixXd::from_column_major(3, 1, &gravity.to_array()));
    let q = qr.householder_q();
    let col0 = q.col(1).to_vector3d();
    let col2 = q.col(2).to_vector3d();
    let mut r = Matrix3d::from_columns(col0, gravity, col2);

    // Ensure right-handed coordinate system.
    if r.determinant() < 0.0 {
        r = Matrix3d::from_columns(col0, gravity, -col2);
    }
    Ok(r)
}
