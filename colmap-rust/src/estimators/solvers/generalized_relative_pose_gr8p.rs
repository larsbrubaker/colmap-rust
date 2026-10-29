//! The solver half of COLMAP's GR8P (`colmap/estimators/solvers/generalized_relative_pose.cc`):
//! `GR8PEstimator::Estimate` and its anonymous-namespace helpers (`ComposePlueckerData`, the
//! Cayley conversions, `ComputeRotationBetweenPoints`, `ComposeG`, `ComputeEigenValue`,
//! `ComputeCost`, `ComputeJacobian`). The estimator type, [`super::GR8PEstimator`], is in
//! `generalized_relative_pose.rs`, which declares this file as its private `gr8p` module.
//! Port of colmap-sharp's `Estimators/Solvers/GeneralizedRelativePose.GR8P.cs`.
//!
//! Kneip and Li (CVPR 2014): the smallest eigenvalue of the 4x4 matrix G(R) vanishes at the
//! true rotation, so the Cayley parameters are found by a finite-difference gradient descent
//! on a closed-form root of G's characteristic quartic (with random restarts, drawn from the
//! thread PRNG like COLMAP's `RandomUniformReal`), and the translations are read from G's
//! eigenvectors at the optimum.
//!
//! Translation notes:
//! - Eigenvectors: G is symmetric (`ComposeG` mirrors every off-diagonal entry), so all of its
//!   eigenvalues are real and [`EigenSolver`] returns real eigenvectors for them. COLMAP's
//!   `V.real().colwise().hnormalized()` then divides each by its last entry, which removes
//!   the scale and sign that are the only freedom left. The order of the four candidates
//!   follows the Schur order, which may differ from Eigen's. That is a deliberate divergence:
//!   `GR8PEstimator::Estimate` reads the four translations from `Eigen::EigenSolver<Matrix4d>`
//!   and returns them in Eigen's eigenvector order, and [`EigenSolver`] (colmap-sharp's
//!   Hessenberg and Francis QR port) can order them differently and round the last bits
//!   differently. Eigen (MPL-2.0) is not ported, and its deflation order is not a documented
//!   convention. Since every eigenvalue and eigenvector is real and `hnormalized()` removes the
//!   only freedom left, the candidate set is the same up to rounding. RANSAC scores all four,
//!   so the order only matters when two candidates have exactly the same support and score,
//!   where the first one wins. The seven `..._gr8p_*` instantiations of COLMAP's
//!   `ParameterizedGRNPEstimatorTests.GR8P` (`tests/estimators/generalized_relative_pose.rs`)
//!   pass 1:1: RANSAC over 16 correspondences finds the true `rig2_from_rig1` within rtol and
//!   ttol 1e-2 in one of ten trials. Not compared against C++ Eigen. Same choice as
//!   colmap-sharp's `GeneralizedRelativePose.GR8P.cs` (which records it in its file header,
//!   not as a numbered entry).
//! - `ComposePlueckerData`: only the direction half of COLMAP's 6-vector Pluecker coordinate
//!   is ever read (`head<3>()`), so the moment half is not formed.
//! - The sums `xxF ... zxF`, `x1P ... z2P`, `m11P`, `m12P`, `m22P` are bundled in [`Terms`];
//!   the 3x9 and 9x9 matrices are stored row-major. A row vector times a matrix times a
//!   column is evaluated as (row * matrix) . column, left to right as the C++ expressions
//!   read.
//!
//! Tier B (an eigenvalue problem and a closed-form quartic); the descent is deterministic for
//! a given PRNG state.

use crate::geometry::Rigid3d;
use crate::linalg::{EigenSolver, Matrix3d, Matrix4d, MatrixXd, Quaterniond, Svd3d, Vector3d};
use crate::math::fns;
use crate::math::random::random_uniform_real;
use crate::{check_eq, check_ge, Result};

use super::GRNPObservation;

/// Port of `GR8PEstimator::Estimate`: four candidate `rig2_from_rig1` poses from at least six
/// correspondences (eight for stability).
pub(super) fn estimate(
    points1: &[GRNPObservation],
    points2: &[GRNPObservation],
    rigs2_from_rigs1: &mut Vec<Rigid3d>,
) -> Result<()> {
    check_ge!(points1.len(), 6);
    check_eq!(points1.len(), points2.len());

    rigs2_from_rigs1.clear();

    let n = points1.len();
    let mut origins_in_rig1 = Vec::with_capacity(n);
    let mut origins_in_rig2 = Vec::with_capacity(n);
    let mut rays1 = Vec::with_capacity(n);
    let mut rays2 = Vec::with_capacity(n);
    for i in 0..n {
        let (origin1, ray1) = compose_pluecker_data(
            &points1[i].cam_from_rig.inverse(),
            points1[i].ray_with_jac_in_cam.ray,
        );
        let (origin2, ray2) = compose_pluecker_data(
            &points2[i].cam_from_rig.inverse(),
            points2[i].ray_with_jac_in_cam.ray,
        );
        origins_in_rig1.push(origin1);
        rays1.push(ray1);
        origins_in_rig2.push(origin2);
        rays2.push(ray2);
    }

    let mut terms = Terms::default();
    for i in 0..n {
        terms.accumulate(rays1[i], rays2[i], origins_in_rig1[i], origins_in_rig2[i]);
    }

    let initial_rotation = compute_rotation_between_points(&rays1, &rays2)?;

    const K_MIN_LAMBDA: f64 = 0.00001;
    const K_MAX_LAMBDA: f64 = 0.08;
    const K_LAMBDA_MODIFIER: f64 = 2.0;
    const K_MAX_NUM_ITERATIONS: i32 = 50;
    const K_DISABLE_INCREMENTS: bool = true;

    let mut perturbation_amplitude = 0.3;
    let mut num_random_trials = 0;

    let mut rotation = Vector3d::zeros();
    while num_random_trials < 5 {
        if num_random_trials > 2 {
            perturbation_amplitude = 0.6;
        }

        if num_random_trials == 0 {
            rotation = initial_rotation;
        } else {
            // Separate statements keep the x, y, z draw order of the C++ constructor
            // arguments, which clang evaluates left to right.
            let px = random_uniform_real(-perturbation_amplitude, perturbation_amplitude);
            let py = random_uniform_real(-perturbation_amplitude, perturbation_amplitude);
            let pz = random_uniform_real(-perturbation_amplitude, perturbation_amplitude);
            rotation = initial_rotation + Vector3d::new(px, py, pz);
        }

        let mut lambda = 0.01;
        // Never incremented, as in COLMAP: with kDisableIncrements the step only grows on
        // the first iteration.
        let num_iterations = 0;
        let mut smallest_eigen_value = compute_cost(&terms, rotation, 1);

        for _ in 0..K_MAX_NUM_ITERATIONS {
            let jacobian = compute_jacobian(&terms, rotation, smallest_eigen_value, 1);

            let normalized_jacobian = jacobian.normalized();

            let mut sampling_point = rotation - lambda * normalized_jacobian;
            let mut sampling_eigen_value = compute_cost(&terms, sampling_point, 1);

            if num_iterations == 0 || !K_DISABLE_INCREMENTS {
                while sampling_eigen_value < smallest_eigen_value {
                    smallest_eigen_value = sampling_eigen_value;
                    if lambda * K_LAMBDA_MODIFIER > K_MAX_LAMBDA {
                        break;
                    }
                    lambda *= K_LAMBDA_MODIFIER;
                    sampling_point = rotation - lambda * normalized_jacobian;
                    sampling_eigen_value = compute_cost(&terms, sampling_point, 1);
                }
            }

            while sampling_eigen_value > smallest_eigen_value {
                lambda /= K_LAMBDA_MODIFIER;
                sampling_point = rotation - lambda * normalized_jacobian;
                sampling_eigen_value = compute_cost(&terms, sampling_point, 1);
            }

            rotation = sampling_point;
            smallest_eigen_value = sampling_eigen_value;

            if lambda < K_MIN_LAMBDA {
                break;
            }
        }

        if rotation.norm() < 0.01 {
            let eigen_value2 = compute_cost(&terms, rotation, 0);
            if eigen_value2 > 0.001 {
                num_random_trials += 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }

    let r = cayley_to_rotation_matrix(rotation).transpose();

    let g = compose_g(&terms, rotation);
    let mut g_dynamic = MatrixXd::zeros(4, 4);
    for row in 0..4 {
        for col in 0..4 {
            g_dynamic[(row, col)] = g[(row, col)];
        }
    }
    let v = EigenSolver::new(&g_dynamic, true).eigenvectors();

    // Eigen::Quaterniond(R): not normalized, as in COLMAP.
    let q = Quaterniond::from_rotation_matrix(r);
    rigs2_from_rigs1.reserve(4);
    for i in 0..4 {
        // V.real().colwise().hnormalized()
        let w = v[(3, i)].re;
        let vv = Vector3d::new(v[(0, i)].re / w, v[(1, i)].re / w, v[(2, i)].re / w);
        rigs2_from_rigs1.push(Rigid3d::new(q, -(r * vv)));
    }
    Ok(())
}

/// Port of `ComposePlueckerData`: the ray origin and unit direction of a camera ray in the rig
/// frame (the direction half of the Pluecker coordinate).
fn compose_pluecker_data(rig_from_cam: &Rigid3d, ray_in_cam: Vector3d) -> (Vector3d, Vector3d) {
    let ray_in_rig = (rig_from_cam.rotation * ray_in_cam).normalized();
    (rig_from_cam.translation, ray_in_rig)
}

/// Port of `CayleyToRotationMatrix`.
fn cayley_to_rotation_matrix(cayley: Vector3d) -> Matrix3d {
    let cayley0_sqr = cayley[0] * cayley[0];
    let cayley1_sqr = cayley[1] * cayley[1];
    let cayley2_sqr = cayley[2] * cayley[2];
    let cayley01 = cayley[0] * cayley[1];
    let cayley12 = cayley[1] * cayley[2];
    let cayley02 = cayley[0] * cayley[2];

    let scale = 1.0 + cayley0_sqr + cayley1_sqr + cayley2_sqr;
    let inv_scale = 1.0 / scale;

    Matrix3d::new(
        inv_scale * (1.0 + cayley0_sqr - cayley1_sqr - cayley2_sqr),
        inv_scale * (2.0 * (cayley01 - cayley[2])),
        inv_scale * (2.0 * (cayley02 + cayley[1])),
        inv_scale * (2.0 * (cayley01 + cayley[2])),
        inv_scale * (1.0 - cayley0_sqr + cayley1_sqr - cayley2_sqr),
        inv_scale * (2.0 * (cayley12 - cayley[0])),
        inv_scale * (2.0 * (cayley02 - cayley[1])),
        inv_scale * (2.0 * (cayley12 + cayley[0])),
        inv_scale * (1.0 - cayley0_sqr - cayley1_sqr + cayley2_sqr),
    )
}

/// Port of `RotationMatrixToCaley` (sic).
fn rotation_matrix_to_cayley(r: Matrix3d) -> Vector3d {
    let c1 = r - Matrix3d::identity();
    let c2 = r + Matrix3d::identity();
    let c = c1 * c2.inverse();
    Vector3d::new(-c[(1, 2)], c[(0, 2)], -c[(0, 1)])
}

/// Port of `ComputeRotationBetweenPoints`: the Cayley parameters of the rotation that best
/// aligns the centered ray directions (Kabsch on the directions).
fn compute_rotation_between_points(rays1: &[Vector3d], rays2: &[Vector3d]) -> Result<Vector3d> {
    check_eq!(rays1.len(), rays2.len());

    // Compute the center of all observed points.
    let mut points_center1 = Vector3d::zeros();
    let mut points_center2 = Vector3d::zeros();
    for i in 0..rays1.len() {
        points_center1 += rays1[i];
        points_center2 += rays2[i];
    }
    points_center1 /= rays1.len() as f64;
    points_center2 /= rays1.len() as f64;

    let mut hcross = Matrix3d::zeros();
    for i in 0..rays1.len() {
        let f1 = rays1[i] - points_center1;
        let f2 = rays2[i] - points_center2;
        hcross += outer(f2, f1);
    }

    let svd = Svd3d::compute(&hcross);
    let v = svd.matrix_v;
    let u = svd.matrix_u;

    let mut r = v * u.transpose();
    if r.determinant() < 0.0 {
        let v_prime = Matrix3d::from_columns(v.col(0), v.col(1), -v.col(2));
        r = v_prime * u.transpose();
    }

    Ok(rotation_matrix_to_cayley(r))
}

/// `a.transpose() * m * b` for a row vector `a`: (a^T M) . b.
fn quad(a: Vector3d, m: &Matrix3d, b: Vector3d) -> f64 {
    (m.transpose() * a).dot(b)
}

/// Port of `ComposeG`: the symmetric 4x4 matrix whose smallest eigenvalue vanishes at the
/// true rotation (given by its Cayley parameters).
fn compose_g(t: &Terms, rotation: Vector3d) -> Matrix4d {
    let r = cayley_to_rotation_matrix(rotation);
    let r0 = r.row(0);
    let r1 = r.row(1);
    let r2 = r.row(2);

    let r_rows = [
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
    ];
    let r_cols = r.to_column_major();

    let xx_fr1t = t.xx_f * r1;
    let yy_fr0t = t.yy_f * r0;
    let zz_fr0t = t.zz_f * r0;
    let yz_fr0t = t.yz_f * r0;
    let xy_fr1t = t.xy_f * r1;
    let xy_fr2t = t.xy_f * r2;
    let zx_fr1t = t.zx_f * r1;
    let zx_fr2t = t.zx_f * r2;

    let x1_pc = mul3x9(&t.x1_p, &r_cols);
    let y1_pc = mul3x9(&t.y1_p, &r_cols);
    let z1_pc = mul3x9(&t.z1_p, &r_cols);

    let x2_pr = mul3x9(&t.x2_p, &r_rows);
    let y2_pr = mul3x9(&t.y2_p, &r_rows);
    let z2_pr = mul3x9(&t.z2_p, &r_rows);

    let mut g = Matrix4d::zeros();

    g[(0, 0)] = quad(r2, &t.yy_f, r2);
    g[(0, 0)] += quad(-2.0 * r2, &t.yz_f, r1);
    g[(0, 0)] += quad(r1, &t.zz_f, r1);

    g[(0, 1)] = r2.dot(yz_fr0t);
    g[(0, 1)] += (-1.0 * r2).dot(xy_fr2t);
    g[(0, 1)] += (-1.0 * r1).dot(zz_fr0t);
    g[(0, 1)] += r1.dot(zx_fr2t);

    g[(0, 2)] = r2.dot(xy_fr1t);
    g[(0, 2)] += (-1.0 * r2).dot(yy_fr0t);
    g[(0, 2)] += (-1.0 * r1).dot(zx_fr1t);
    g[(0, 2)] += r1.dot(yz_fr0t);

    g[(1, 1)] = r0.dot(zz_fr0t);
    g[(1, 1)] += (-2.0 * r0).dot(zx_fr2t);
    g[(1, 1)] += quad(r2, &t.xx_f, r2);

    g[(1, 2)] = r0.dot(zx_fr1t);
    g[(1, 2)] += (-1.0 * r0).dot(yz_fr0t);
    g[(1, 2)] += (-1.0 * r2).dot(xx_fr1t);
    g[(1, 2)] += r0.dot(xy_fr2t);

    g[(2, 2)] = r1.dot(xx_fr1t);
    g[(2, 2)] += (-2.0 * r0).dot(xy_fr1t);
    g[(2, 2)] += r0.dot(yy_fr0t);

    g[(1, 0)] = g[(0, 1)];
    g[(2, 0)] = g[(0, 2)];
    g[(2, 1)] = g[(1, 2)];

    g[(0, 3)] = r2.dot(y1_pc);
    g[(0, 3)] += r2.dot(y2_pr);
    g[(0, 3)] += (-1.0 * r1).dot(z1_pc);
    g[(0, 3)] += (-1.0 * r1).dot(z2_pr);

    g[(1, 3)] = r0.dot(z1_pc);
    g[(1, 3)] += r0.dot(z2_pr);
    g[(1, 3)] += (-1.0 * r2).dot(x1_pc);
    g[(1, 3)] += (-1.0 * r2).dot(x2_pr);

    g[(2, 3)] = r1.dot(x1_pc);
    g[(2, 3)] += r1.dot(x2_pr);
    g[(2, 3)] += (-1.0 * r0).dot(y1_pc);
    g[(2, 3)] += (-1.0 * r0).dot(y2_pr);

    g[(3, 3)] = quad9(-1.0, &r_cols, &t.m11_p, &r_cols);
    g[(3, 3)] += quad9(-1.0, &r_rows, &t.m22_p, &r_rows);
    g[(3, 3)] += quad9(-2.0, &r_rows, &t.m12_p, &r_cols);

    g[(3, 0)] = g[(0, 3)];
    g[(3, 1)] = g[(1, 3)];
    g[(3, 2)] = g[(2, 3)];

    g
}

/// Port of `ComputeEigenValue`: the four eigenvalues of G as the closed-form roots of its
/// characteristic quartic (Ferrari, through a trigonometric resolvent cubic).
fn compute_eigen_value(t: &Terms, rotation: Vector3d) -> [f64; 4] {
    let g = compose_g(t, rotation);

    // Compute the roots in closed-form.
    let g01_2 = g[(0, 1)] * g[(0, 1)];
    let g02_2 = g[(0, 2)] * g[(0, 2)];
    let g03_2 = g[(0, 3)] * g[(0, 3)];
    let g12_2 = g[(1, 2)] * g[(1, 2)];
    let g13_2 = g[(1, 3)] * g[(1, 3)];
    let g23_2 = g[(2, 3)] * g[(2, 3)];
    let (g00, g11, g22, g33) = (g[(0, 0)], g[(1, 1)], g[(2, 2)], g[(3, 3)]);
    let (g01, g02, g03, g12, g13, g23) = (
        g[(0, 1)],
        g[(0, 2)],
        g[(0, 3)],
        g[(1, 2)],
        g[(1, 3)],
        g[(2, 3)],
    );

    let b = -g33 - g22 - g11 - g00;
    let c = -g23_2 + g22 * g33 - g13_2 - g12_2 + g11 * g33 + g11 * g22 - g03_2 - g02_2 - g01_2
        + g00 * g33
        + g00 * g22
        + g00 * g11;
    let d = g13_2 * g22 - 2.0 * g12 * g13 * g23 + g12_2 * g33 + g11 * g23_2 - g11 * g22 * g33
        + g03_2 * g22
        + g03_2 * g11
        - 2.0 * g02 * g03 * g23
        + g02_2 * g33
        + g02_2 * g11
        - 2.0 * g01 * g03 * g13
        - 2.0 * g01 * g02 * g12
        + g01_2 * g33
        + g01_2 * g22
        + g00 * g23_2
        - g00 * g22 * g33
        + g00 * g13_2
        + g00 * g12_2
        - g00 * g11 * g33
        - g00 * g11 * g22;
    let e = g03_2 * g12_2 - g03_2 * g11 * g22 - 2.0 * g02 * g03 * g12 * g13
        + 2.0 * g02 * g03 * g11 * g23
        + g02_2 * g13_2
        - g02_2 * g11 * g33
        + 2.0 * g01 * g03 * g13 * g22
        - 2.0 * g01 * g03 * g12 * g23
        - 2.0 * g01 * g02 * g13 * g23
        + 2.0 * g01 * g02 * g12 * g33
        + g01_2 * g23_2
        - g01_2 * g22 * g33
        - g00 * g13_2 * g22
        + 2.0 * g00 * g12 * g13 * g23
        - g00 * g12_2 * g33
        - g00 * g11 * g23_2
        + g00 * g11 * g22 * g33;

    let b_pw2 = b * b;
    let b_pw3 = b_pw2 * b;
    let b_pw4 = b_pw3 * b;
    let alpha = -0.375 * b_pw2 + c;
    let beta = b_pw3 / 8.0 - b * c / 2.0 + d;
    let gamma = -0.01171875 * b_pw4 + b_pw2 * c / 16.0 - b * d / 4.0 + e;
    let alpha_pw2 = alpha * alpha;
    let alpha_pw3 = alpha_pw2 * alpha;
    let p = -alpha_pw2 / 12.0 - gamma;
    let q = -alpha_pw3 / 108.0 + alpha * gamma / 3.0 - beta * beta / 8.0;
    let helper1 = -p * p * p / 27.0;
    let theta2 = fns::pow(helper1, 1.0 / 3.0);
    let theta1 =
        fns::sqrt(theta2) * fns::cos((1.0 / 3.0) * fns::acos((-q / 2.0) / fns::sqrt(helper1)));
    let y = -(5.0 / 6.0) * alpha - ((1.0 / 3.0) * p * theta1 - theta1 * theta2) / theta2;
    let w = fns::sqrt(alpha + 2.0 * y);

    [
        -b / 4.0 + 0.5 * w + 0.5 * fns::sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w),
        -b / 4.0 + 0.5 * w - 0.5 * fns::sqrt(-3.0 * alpha - 2.0 * y - 2.0 * beta / w),
        -b / 4.0 - 0.5 * w + 0.5 * fns::sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w),
        -b / 4.0 - 0.5 * w - 0.5 * fns::sqrt(-3.0 * alpha - 2.0 * y + 2.0 * beta / w),
    ]
}

/// Port of `ComputeCost`: root 2 (`step` 0) or root 3 (`step` 1) of G's quartic. `step` is
/// always a literal 0 or 1 here, which is what COLMAP's `THROW_CHECK_GE`/`LE` guard.
fn compute_cost(t: &Terms, rotation: Vector3d, step: usize) -> f64 {
    debug_assert!(step <= 1);
    let roots = compute_eigen_value(t, rotation);
    if step == 0 {
        roots[2]
    } else {
        roots[3]
    }
}

/// Port of `ComputeJacobian`: forward differences of the cost (unscaled by the step, as in
/// COLMAP; the caller normalizes the result).
fn compute_jacobian(t: &Terms, rotation: Vector3d, current_cost: f64, step: usize) -> Vector3d {
    const K_STEP_SIZE: f64 = 1e-8;
    let mut jacobian = Vector3d::zeros();
    for j in 0..3 {
        let mut cayley_j = rotation;
        cayley_j[j] += K_STEP_SIZE;
        let cost_j = compute_cost(t, cayley_j, step);
        jacobian[j] = cost_j - current_cost;
    }
    jacobian
}

/// `a * b^T`.
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

/// The row-major 3x9 matrix `m` times the 9-vector `v`.
fn mul3x9(m: &[f64; 27], v: &[f64; 9]) -> Vector3d {
    let mut result = [0.0; 3];
    for (r, out) in result.iter_mut().enumerate() {
        let mut sum = 0.0;
        for c in 0..9 {
            sum += m[r * 9 + c] * v[c];
        }
        *out = sum;
    }
    Vector3d::new(result[0], result[1], result[2])
}

/// `((scale * a)^T M) . b` for 9-vectors `a`, `b` and the row-major 9x9 matrix `m`.
fn quad9(scale: f64, a: &[f64; 9], m: &[f64; 81], b: &[f64; 9]) -> f64 {
    let mut result = 0.0;
    for c in 0..9 {
        let mut row_times_m = 0.0;
        for r in 0..9 {
            row_times_m += scale * a[r] * m[r * 9 + c];
        }
        result += row_times_m * b[c];
    }
    result
}

/// The sums `GR8PEstimator::Estimate` accumulates over the correspondences (`xxF ... zxF`,
/// `x1P ... z2P`, `m11P`, `m12P`, `m22P`) that [`compose_g`] combines with a rotation. The
/// 3x9 and 9x9 matrices are row-major.
struct Terms {
    xx_f: Matrix3d,
    yy_f: Matrix3d,
    zz_f: Matrix3d,
    xy_f: Matrix3d,
    yz_f: Matrix3d,
    zx_f: Matrix3d,
    x1_p: [f64; 27],
    y1_p: [f64; 27],
    z1_p: [f64; 27],
    x2_p: [f64; 27],
    y2_p: [f64; 27],
    z2_p: [f64; 27],
    m11_p: [f64; 81],
    m12_p: [f64; 81],
    m22_p: [f64; 81],
}

impl Default for Terms {
    fn default() -> Self {
        Self {
            xx_f: Matrix3d::zeros(),
            yy_f: Matrix3d::zeros(),
            zz_f: Matrix3d::zeros(),
            xy_f: Matrix3d::zeros(),
            yz_f: Matrix3d::zeros(),
            zx_f: Matrix3d::zeros(),
            x1_p: [0.0; 27],
            y1_p: [0.0; 27],
            z1_p: [0.0; 27],
            x2_p: [0.0; 27],
            y2_p: [0.0; 27],
            z2_p: [0.0; 27],
            m11_p: [0.0; 81],
            m12_p: [0.0; 81],
            m22_p: [0.0; 81],
        }
    }
}

impl Terms {
    /// Adds one correspondence: rig rays `f1`, `f2` and ray origins `t1`, `t2`.
    fn accumulate(&mut self, f1: Vector3d, f2: Vector3d, t1: Vector3d, t2: Vector3d) {
        let f = outer(f2, f2);
        self.xx_f += f1[0] * f1[0] * f;
        self.yy_f += f1[1] * f1[1] * f;
        self.zz_f += f1[2] * f1[2] * f;
        self.xy_f += f1[0] * f1[1] * f;
        self.yz_f += f1[1] * f1[2] * f;
        self.zx_f += f1[2] * f1[0] * f;

        let ff1 = [
            f1[0] * (f2[1] * t2[2] - f2[2] * t2[1]),
            f1[1] * (f2[1] * t2[2] - f2[2] * t2[1]),
            f1[2] * (f2[1] * t2[2] - f2[2] * t2[1]),
            f1[0] * (f2[2] * t2[0] - f2[0] * t2[2]),
            f1[1] * (f2[2] * t2[0] - f2[0] * t2[2]),
            f1[2] * (f2[2] * t2[0] - f2[0] * t2[2]),
            f1[0] * (f2[0] * t2[1] - f2[1] * t2[0]),
            f1[1] * (f2[0] * t2[1] - f2[1] * t2[0]),
            f1[2] * (f2[0] * t2[1] - f2[1] * t2[0]),
        ];

        add_scaled_outer(&mut self.x1_p, f1[0] * f2, &ff1);
        add_scaled_outer(&mut self.y1_p, f1[1] * f2, &ff1);
        add_scaled_outer(&mut self.z1_p, f1[2] * f2, &ff1);

        let ff2 = [
            f2[0] * (f1[1] * t1[2] - f1[2] * t1[1]),
            f2[1] * (f1[1] * t1[2] - f1[2] * t1[1]),
            f2[2] * (f1[1] * t1[2] - f1[2] * t1[1]),
            f2[0] * (f1[2] * t1[0] - f1[0] * t1[2]),
            f2[1] * (f1[2] * t1[0] - f1[0] * t1[2]),
            f2[2] * (f1[2] * t1[0] - f1[0] * t1[2]),
            f2[0] * (f1[0] * t1[1] - f1[1] * t1[0]),
            f2[1] * (f1[0] * t1[1] - f1[1] * t1[0]),
            f2[2] * (f1[0] * t1[1] - f1[1] * t1[0]),
        ];

        add_scaled_outer(&mut self.x2_p, f1[0] * f2, &ff2);
        add_scaled_outer(&mut self.y2_p, f1[1] * f2, &ff2);
        add_scaled_outer(&mut self.z2_p, f1[2] * f2, &ff2);

        // m11P -= ff1 ff1^T, m22P -= ff2 ff2^T, m12P -= ff2 ff1^T.
        for r in 0..9 {
            for c in 0..9 {
                self.m11_p[r * 9 + c] -= ff1[r] * ff1[c];
                self.m22_p[r * 9 + c] -= ff2[r] * ff2[c];
                self.m12_p[r * 9 + c] -= ff2[r] * ff1[c];
            }
        }
    }
}

/// `m` (row-major 3x9) `+= a * b^T`.
fn add_scaled_outer(m: &mut [f64; 27], a: Vector3d, b: &[f64; 9]) {
    for r in 0..3 {
        for c in 0..9 {
            m[r * 9 + c] += a[r] * b[c];
        }
    }
}
