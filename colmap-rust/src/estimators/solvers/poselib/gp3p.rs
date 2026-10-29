//! Port of PoseLib's `PoseLib/solvers/gp3p.cc` (BSD-3-Clause, Copyright (c) 2020 Viktor
//! Larsson; see THIRD_PARTY_NOTICES.md; commit fa7280fee27f97aff31ae7f98bab7f583fac7d08),
//! through colmap-sharp's `Estimators/Solvers/PoseLib/Gp3p.cs`: the minimal generalized
//! absolute pose solver (three rays with individual origins p and directions x, and their
//! three world points X). The six projection equations are linear in `[t; vec(R); 1]`; the
//! translation is eliminated with the first three and the remaining three constrain the
//! rotation alone, which [`super::re3q3::solve_rotation`] solves. Called by COLMAP's
//! `GP3PEstimator`.
//!
//! Translation notes:
//! - The 6 x 13 system A and the 3 x 10 rotation system AR are column-major stack arrays.
//! - `re3q3_rotation` draws its random pre-rotation from a fixed-seed mt19937 rather than
//!   `std::rand()` (the divergence in [`super::re3q3::solve_rotation`]'s doc).
//!
//! Tier B. Tests: `tests/estimators/rust_only_gp3p.rs`.

use super::camera_pose::{quat_to_rotmat, CameraPose};
use super::re3q3::solve_rotation;
use crate::linalg::{Matrix3d, Vector3d, Vector4d};

/// Solve for the poses (`x_cam = R X + t` in the generalized camera frame) with
/// `lambda_i x_i = R X_i + t - p_i`, given the ray origins `p` and directions `x` in the
/// generalized camera frame and the world points `big_x`. Clears `output`, appends up to
/// eight poses and returns their count. Port of `poselib::gp3p`.
pub fn gp3p(
    p: &[Vector3d],
    x: &[Vector3d],
    big_x: &[Vector3d],
    output: &mut Vec<CameraPose>,
) -> usize {
    // A(r, c) = a[r + 6 c].
    let mut a = [0.0; 6 * 13];

    for i in 0..3 {
        let (xi, bx, pi) = (x[i], big_x[i], p[i]);
        // xx = [x3 0 -x1; 0 x3 -x2]
        // eqs = [xx kron(X',xx), -xx*p] * [t; vec(R); 1]
        let row0 = [
            xi.z,
            0.0,
            -xi.x,
            bx.x * xi.z,
            0.0,
            -bx.x * xi.x,
            bx.y * xi.z,
            0.0,
            -bx.y * xi.x,
            bx.z * xi.z,
            0.0,
            -bx.z * xi.x,
            -pi.x * xi.z + pi.z * xi.x,
        ];
        let row1 = [
            0.0,
            xi.z,
            -xi.y,
            0.0,
            bx.x * xi.z,
            -bx.x * xi.y,
            0.0,
            bx.y * xi.z,
            -bx.y * xi.y,
            0.0,
            bx.z * xi.z,
            -bx.z * xi.y,
            -pi.y * xi.z + pi.z * xi.y,
        ];
        for c in 0..13 {
            a[2 * i + 6 * c] = row0[c];
            a[2 * i + 1 + 6 * c] = row1[c];
        }
    }

    let b = Matrix3d::new(a[0], a[6], a[12], a[1], a[7], a[13], a[2], a[8], a[14]).inverse();

    // Top = A.block<3, 3>(3, 0) * B (3 x 3), then
    // AR = A.block<3, 10>(3, 3) - Top * A.block<3, 10>(0, 3).
    let bottom_left = Matrix3d::new(a[3], a[9], a[15], a[4], a[10], a[16], a[5], a[11], a[17]);
    let top = bottom_left * b;

    // AR(r, c) = ar[r + 3 c].
    let mut ar = [0.0; 30];
    for c in 0..10 {
        let col = c + 3;
        for r in 0..3 {
            let sum = top[(r, 0)] * a[6 * col]
                + top[(r, 1)] * a[1 + 6 * col]
                + top[(r, 2)] * a[2 + 6 * col];
            ar[r + 3 * c] = a[3 + r + 6 * col] - sum;
        }
    }

    let mut solutions = [Vector4d::zeros(); 8];
    let n_sols = solve_rotation(&ar, &mut solutions, true);

    output.clear();
    for &q in &solutions[..n_sols] {
        let rot = quat_to_rotmat(q);

        // A.block<3, 9>(0, 3) * quat_to_rotmatvec(q) + A.block<3, 1>(0, 12), where
        // quat_to_rotmatvec is R stacked column-major.
        let rhs = Vector3d::new(
            rotation_row(&a, 0, rot) + a[6 * 12],
            rotation_row(&a, 1, rot) + a[1 + 6 * 12],
            rotation_row(&a, 2, rot) + a[2 + 6 * 12],
        );

        let t = -(b * rhs);
        output.push(CameraPose::new(q, t));
    }

    n_sols
}

/// Row `r` of `A.block<3, 9>(0, 3)` times `vec(R)` (column-major).
fn rotation_row(a: &[f64], r: usize, rot: Matrix3d) -> f64 {
    let mut sum = 0.0;
    for k in 0..9 {
        sum += a[r + 6 * (3 + k)] * rot[(k % 3, k / 3)];
    }

    sum
}
