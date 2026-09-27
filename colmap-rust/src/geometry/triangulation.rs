//! Port of COLMAP's `colmap/geometry/triangulation.h` and `triangulation.cc`: two-view DLT
//! triangulation from image points or bearing vectors, the mid-point method, multi-view DLT
//! through the 4x4 normal matrix, optimal two-view triangulation (Lindstrom correction
//! first), and the triangulation-angle helpers. Port of colmap-sharp's
//! `Geometry/Triangulation.cs`. Builds on [`super::essential_matrix`]
//! (`triangulate_optimal_point`), [`super::rigid3`], and `Svd3d`/`Svd4d`/`JacobiSvd`/
//! `SelfAdjointEigenSolver` in [`crate::linalg`]; used by [`super::homography_matrix`].
//! Tests: `tests/geometry/triangulation.rs` (triangulation_test.cc) and
//! `tests/geometry/rust_only_two_view_oracle.rs`.
//!
//! COLMAP returns `bool` and writes the point through a pointer; here the functions return
//! `Option<Vector3d>` (`None` for COLMAP's `false`).
//!
//! Tiers: `calculate_triangulation_angle(s)` and `calculate_angle_between_vectors` are
//! scalar code (`acos` through [`crate::math::fns`]); every triangulation goes through an SVD
//! or a symmetric eigensolver and is Tier B (docs/CPP_DIVERGENCES.md entries 30 and 31).
//!
//! Sign independence: each method reads a null vector (the last column of V, or the
//! eigenvector of the smallest eigenvalue) whose sign is arbitrary, and then only uses it
//! through `hnormalized()`, i.e. divided by its own last coordinate, which cancels the sign.
//! The `== 0` rejections test that same coordinate, so they are sign-independent too.

use super::essential_matrix::{essential_matrix_from_pose, find_optimal_image_observations};
use super::rigid3::Rigid3d;
use crate::linalg::{
    ComputationInfo, JacobiSvd, Matrix3d, Matrix3x4d, Matrix4d, MatrixXd, Quaterniond,
    SelfAdjointEigenSolver, Svd3d, Svd4d, SvdFactor, SvdOptions, Vector2d, Vector3d,
    MACHINE_EPSILON,
};
use crate::math::fns;
use crate::{check_eq, Result};

/// Port of `colmap::TriangulatePoint` (`Vector2d` overload): DLT triangulation from two
/// observations in normalized camera coordinates. `None` for a degenerate (e.g.
/// parallel-ray) geometry.
pub fn triangulate_point(
    cam1_from_world: &Matrix3x4d,
    cam2_from_world: &Matrix3x4d,
    cam_point1: Vector2d,
    cam_point2: Vector2d,
) -> Option<Vector3d> {
    let a = Matrix4d::from_rows(
        cam_point1.x * cam1_from_world.row(2) - cam1_from_world.row(0),
        cam_point1.y * cam1_from_world.row(2) - cam1_from_world.row(1),
        cam_point2.x * cam2_from_world.row(2) - cam2_from_world.row(0),
        cam_point2.y * cam2_from_world.row(2) - cam2_from_world.row(1),
    );

    let svd = Svd4d::compute(&a);
    if svd.info != ComputationInfo::Success || svd.matrix_v[(3, 3)] == 0.0 {
        return None;
    }
    Some(svd.matrix_v.col(3).hnormalized())
}

// Rows [row0, row0 + 3) of A = P - b (b^T P): the bearing b's projector applied to P.
fn fill_ray_projector_rows(a: &mut MatrixXd, row0: usize, p: &Matrix3x4d, b: Vector3d) {
    for col in 0..4 {
        let pc = p.col(col);
        let btp = b.dot(pc);
        a[(row0, col)] = pc.x - b.x * btp;
        a[(row0 + 1, col)] = pc.y - b.y * btp;
        a[(row0 + 2, col)] = pc.z - b.z * btp;
    }
}

/// Port of `colmap::TriangulatePoint` (`Vector3d` overload): DLT triangulation from two
/// bearing vectors (unit rays, which may point into the back hemisphere) on the projectors
/// `I - b b^T`. `None` for a degenerate geometry.
pub fn triangulate_point_from_rays(
    cam1_from_world: &Matrix3x4d,
    cam2_from_world: &Matrix3x4d,
    cam_ray1: Vector3d,
    cam_ray2: Vector3d,
) -> Option<Vector3d> {
    let mut a = MatrixXd::zeros(6, 4);
    fill_ray_projector_rows(&mut a, 0, cam1_from_world, cam_ray1);
    fill_ray_projector_rows(&mut a, 3, cam2_from_world, cam_ray2);

    let svd = JacobiSvd::new(
        &a,
        SvdOptions {
            u: SvdFactor::None,
            v: SvdFactor::Full,
        },
    );
    if svd.info() != ComputationInfo::Success {
        return None;
    }
    let v = svd.matrix_v();
    if v[(3, 3)] == 0.0 {
        return None;
    }
    Some(Vector3d::new(
        v[(0, 3)] / v[(3, 3)],
        v[(1, 3)] / v[(3, 3)],
        v[(2, 3)] / v[(3, 3)],
    ))
}

/// Port of `colmap::TriangulateMidPoint`: the mid-point of the shortest segment between the
/// two rays, in camera 1's frame. `None` for parallel rays or a point behind either camera.
pub fn triangulate_mid_point(
    cam2_from_cam1: &Rigid3d,
    cam_ray1: Vector3d,
    cam_ray2: Vector3d,
) -> Option<Vector3d> {
    let cam1_from_cam2_rotation: Quaterniond = cam2_from_cam1.rotation.inverse();
    let cam_ray2_in_cam1 = cam1_from_cam2_rotation * cam_ray2;
    let cam2_in_cam1 = cam1_from_cam2_rotation * -cam2_from_cam1.translation;

    let a = Matrix3d::new(
        cam_ray1.x,
        -cam_ray2_in_cam1.x,
        -cam2_in_cam1.x,
        cam_ray1.y,
        -cam_ray2_in_cam1.y,
        -cam2_in_cam1.y,
        cam_ray1.z,
        -cam_ray2_in_cam1.z,
        -cam2_in_cam1.z,
    );

    let svd = Svd3d::compute(&a);
    if svd.info != ComputationInfo::Success || svd.matrix_v[(2, 2)] == 0.0 {
        return None;
    }

    let lambda = svd.matrix_v.col(2).hnormalized();

    // Check if point is behind cameras.
    if lambda.x <= MACHINE_EPSILON || lambda.y <= MACHINE_EPSILON {
        return None;
    }

    Some(0.5 * (lambda.x * cam_ray1 + cam2_in_cam1 + lambda.y * cam_ray2_in_cam1))
}

// Contribution of a single bearing observation to the projector-based DLT
// normal-equation matrix. With unit bearing b and projection matrix
// P = cam_from_world, the residual operator is term = P - b b^T P, and the
// system accumulates term^T term.
fn triangulation_dlt_term(cam_from_world: &Matrix3x4d, cam_ray: Vector3d) -> Matrix4d {
    let bbt = Matrix3d::from_columns(
        cam_ray * cam_ray.x,
        cam_ray * cam_ray.y,
        cam_ray * cam_ray.z,
    );
    let term = *cam_from_world - bbt * *cam_from_world;
    let mut t = [0.0; 16];
    for i in 0..4 {
        let ci = term.col(i);
        for j in 0..4 {
            t[j * 4 + i] = ci.dot(term.col(j));
        }
    }
    Matrix4d::from_column_major(t)
}

// Solve the DLT system for the homogeneous point (smallest eigenvector of A).
fn solve_triangulation_dlt(a: &Matrix4d) -> Option<Vector3d> {
    let eigen_solver = SelfAdjointEigenSolver::new(&MatrixXd::from(*a), true);
    if eigen_solver.info() != ComputationInfo::Success {
        return None;
    }
    let vectors = eigen_solver.eigenvectors();
    if vectors[(3, 0)] == 0.0 {
        return None;
    }
    Some(Vector3d::new(
        vectors[(0, 0)] / vectors[(3, 0)],
        vectors[(1, 0)] / vectors[(3, 0)],
        vectors[(2, 0)] / vectors[(3, 0)],
    ))
}

/// Port of `colmap::TriangulateMultiViewPoint` (`Vector2d` overload): triangulation from
/// any number of observations in normalized camera coordinates (each lifted to the unit ray
/// of `(x, y, 1)`). `Ok(None)` for a degenerate geometry; the lists must have equal length.
pub fn triangulate_multi_view_point(
    cams_from_world: &[Matrix3x4d],
    cam_points: &[Vector2d],
) -> Result<Option<Vector3d>> {
    check_eq!(cams_from_world.len(), cam_points.len());
    let mut a = Matrix4d::zeros();
    for (cam_from_world, point) in cams_from_world.iter().zip(cam_points) {
        a += triangulation_dlt_term(cam_from_world, point.homogeneous().normalized());
    }
    Ok(solve_triangulation_dlt(&a))
}

/// Port of `colmap::TriangulateMultiViewPoint` (`Vector3d` overload): triangulation from any
/// number of bearing vectors. `Ok(None)` for a degenerate geometry; the lists must have
/// equal length.
pub fn triangulate_multi_view_point_from_rays(
    cams_from_world: &[Matrix3x4d],
    cam_rays: &[Vector3d],
) -> Result<Option<Vector3d>> {
    check_eq!(cams_from_world.len(), cam_rays.len());
    let mut a = Matrix4d::zeros();
    for (cam_from_world, &ray) in cams_from_world.iter().zip(cam_rays) {
        a += triangulation_dlt_term(cam_from_world, ray);
    }
    Ok(solve_triangulation_dlt(&a))
}

/// Port of `colmap::TriangulateOptimalPoint`: two-view triangulation after moving both
/// observations onto the epipolar constraint ([`find_optimal_image_observations`]).
pub fn triangulate_optimal_point(
    cam1_from_world_mat: &Matrix3x4d,
    cam2_from_world_mat: &Matrix3x4d,
    cam_point1: Vector2d,
    cam_point2: Vector2d,
) -> Option<Vector3d> {
    let cam1_from_world = Rigid3d::new(
        Quaterniond::from_rotation_matrix(cam1_from_world_mat.left_cols3()),
        cam1_from_world_mat.col(3),
    );
    let cam2_from_world = Rigid3d::new(
        Quaterniond::from_rotation_matrix(cam2_from_world_mat.left_cols3()),
        cam2_from_world_mat.col(3),
    );
    let cam2_from_cam1 = cam2_from_world * cam1_from_world.inverse();
    let e = essential_matrix_from_pose(&cam2_from_cam1);

    let (optimal_point1, optimal_point2) =
        find_optimal_image_observations(&e, cam_point1, cam_point2);

    triangulate_point(
        cam1_from_world_mat,
        cam2_from_world_mat,
        optimal_point1,
        optimal_point2,
    )
}

/// Port of `colmap::CalculateTriangulationAngle`: the smaller of the angle between the rays
/// from the two projection centers to the point and its supplement, in radians.
pub fn calculate_triangulation_angle(
    proj_center1: Vector3d,
    proj_center2: Vector3d,
    point3d: Vector3d,
) -> f64 {
    let angle = calculate_angle_between_vectors(point3d - proj_center1, point3d - proj_center2);
    // Triangulation is unstable for acute angles (far away points) and
    // obtuse angles (close points), so always compute the minimum angle
    // between the two intersecting rays.
    let supplement = std::f64::consts::PI - angle;
    // std::min(angle, pi - angle): (b < a) ? b : a.
    if supplement < angle {
        supplement
    } else {
        angle
    }
}

/// Port of `colmap::CalculateTriangulationAngles`: [`calculate_triangulation_angle`] for
/// each point.
pub fn calculate_triangulation_angles(
    proj_center1: Vector3d,
    proj_center2: Vector3d,
    points3d: &[Vector3d],
) -> Vec<f64> {
    points3d
        .iter()
        .map(|&p| calculate_triangulation_angle(proj_center1, proj_center2, p))
        .collect()
}

/// Port of `colmap::CalculateAngleBetweenVectors`: the angle between two vectors in
/// `[0, pi]`; 0 if either is zero.
pub fn calculate_angle_between_vectors(v1: Vector3d, v2: Vector3d) -> f64 {
    let squared_norm1 = v1.squared_norm();
    let squared_norm2 = v2.squared_norm();
    if squared_norm1 == 0.0 || squared_norm2 == 0.0 {
        return 0.0;
    }
    fns::acos((v1.dot(v2) / fns::sqrt(squared_norm1 * squared_norm2)).clamp(-1.0, 1.0))
}
