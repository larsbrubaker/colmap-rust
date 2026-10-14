//! The `ceres/rotation.h` conversions COLMAP calls, generic over [`Scalar`] so they run on
//! doubles and on Jets. Port of colmap-sharp's `ColmapSharp/Solver/Rotation.cs` (from Ceres
//! Solver 2.2.0 `include/ceres/rotation.h`, BSD-3-Clause, see THIRD_PARTY_NOTICES.md).
//!
//! `AngleAxisToQuaternion` and `QuaternionToAngleAxis` are what
//! `colmap/estimators/cost_functions/quaternion_utils.h` wraps as
//! `EigenQuaternionFromAngleAxis` / `AngleAxisFromEigenQuaternion`
//! (`estimators::cost_functions::quaternion_utils`, generic over [`Scalar`] through these) for
//! the motion-averaging and pose-prior cost functions and the quaternion manifolds. COLMAP's
//! reprojection cost functors rotate with Eigen's quaternion product, not with
//! `ceres::QuaternionRotatePoint`, so no other `rotation.h` entry point is reached.
//! Quaternions here are in Ceres' order `[w, x, y, z]`; COLMAP's Eigen-order wrappers swap.
//!
//! Tier A: every operation keeps Ceres' grouping (`theta * T(0.5)`, `T(2.0) * atan2(...)`).

use super::scalar::Scalar;

/// Port of `ceres::AngleAxisToQuaternion`: the unit quaternion `[w, x, y, z]` of the rotation
/// by `|angle_axis|` radians about `angle_axis`. At the origin it uses the first-order Taylor
/// expansion (`w = 1`, `xyz = angle_axis / 2`), so Jets get correct derivatives there instead
/// of the NaN that `sqrt'(0)` would give.
pub fn angle_axis_to_quaternion<T: Scalar>(angle_axis: &[T], quaternion: &mut [T]) {
    let a0 = angle_axis[0];
    let a1 = angle_axis[1];
    let a2 = angle_axis[2];
    let theta = T::hypot3(a0, a1, a2);

    // For points not at the origin, the full conversion is numerically stable.
    if theta.value() != 0.0 {
        let half_theta = theta * T::from_f64(0.5);
        let k = half_theta.sin() / theta;
        quaternion[0] = half_theta.cos();
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    } else {
        // At the origin, sqrt() will produce NaN in the derivative since the argument is
        // zero. By approximating with a Taylor series, and truncating at one term, the value
        // and first derivatives will be computed correctly when Jets are used.
        let k = T::from_f64(0.5);
        quaternion[0] = T::from_f64(1.0);
        quaternion[1] = a0 * k;
        quaternion[2] = a1 * k;
        quaternion[3] = a2 * k;
    }
}

/// Port of `ceres::QuaternionToAngleAxis`: the angle-axis vector of the rotation by a unit
/// quaternion `[w, x, y, z]`, normalized to an angle of at most pi. For zero rotation it uses
/// the first-order Taylor expansion (`angle_axis = 2 xyz`).
pub fn quaternion_to_angle_axis<T: Scalar>(quaternion: &[T], angle_axis: &mut [T]) {
    let q1 = quaternion[1];
    let q2 = quaternion[2];
    let q3 = quaternion[3];
    let sin_theta = T::hypot3(q1, q2, q3);

    // For quaternions representing non-zero rotation, the conversion is numerically stable.
    if sin_theta.value() != 0.0 {
        let cos_theta = quaternion[0];

        // If cos_theta is negative, theta is greater than pi/2, which means that angle for
        // the angle_axis vector which is 2 * theta would be greater than pi. While this
        // would still be the correct rotation, it would not be a normalized angle-axis
        // vector. In that case 2 * theta ~ 2 * theta - 2 * pi, which is to say
        // theta - pi = atan(sin(theta - pi), cos(theta - pi)) = atan(-sin(theta), -cos(theta)).
        let two_theta = if cos_theta.value() < 0.0 {
            T::from_f64(2.0) * (-sin_theta).atan2(-cos_theta)
        } else {
            T::from_f64(2.0) * sin_theta.atan2(cos_theta)
        };
        let k = two_theta / sin_theta;
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    } else {
        // For zero rotation, sqrt() will produce NaN in the derivative since the argument is
        // zero. By approximating with a Taylor series, and truncating at one term, the value
        // and first derivatives will be computed correctly when Jets are used.
        let k = T::from_f64(2.0);
        angle_axis[0] = q1 * k;
        angle_axis[1] = q2 * k;
        angle_axis[2] = q3 * k;
    }
}
