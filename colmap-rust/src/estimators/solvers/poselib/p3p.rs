//! Port of PoseLib's `PoseLib/solvers/p3p.cc` (`poselib::p3p`, by Yaqing Ding and Mark
//! Shachkov), the helpers of `PoseLib/solvers/p3p_common.h` it uses (`root2real`,
//! `compute_pq`, `refine_lambda`) and `univariate::solve_cubic_single_real` from
//! `PoseLib/misc/univariate.cc` (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
//! THIRD_PARTY_NOTICES.md; commit fa7280fee27f97aff31ae7f98bab7f583fac7d08), through
//! colmap-sharp's `Estimators/Solvers/PoseLib/P3p.cs`. The minimal absolute-pose solver behind
//! COLMAP's `P3PEstimator`; returns [`CameraPose`]s.
//!
//! The method: Y. Ding, J. Yang, V. Larsson, C. Olsson, K. Astrom, "Revisiting the P3P
//! Problem", CVPR 2023.
//!
//! Tier B: goes through a cubic and two quadratics. The only allocation is the output list.
//! Tests: `tests/estimators/rust_only_p3p.rs` (COLMAP tests it through `P3PEstimator`).

use super::camera_pose::CameraPose;
use crate::linalg::{Matrix3d, Vector3d};
use crate::math::fns;

/// Estimate up to four camera poses from three bearing vectors `x` (unit length) and their 3D
/// points `big_x`. Clears `output`, appends the poses and returns their count. Port of
/// `poselib::p3p`.
pub fn p3p(x: &[Vector3d], big_x: &[Vector3d], output: &mut Vec<CameraPose>) -> usize {
    output.clear();

    let mut x01 = big_x[0] - big_x[1];
    let mut x02 = big_x[0] - big_x[2];
    let x12 = big_x[1] - big_x[2];

    let mut a01 = x01.squared_norm();
    let mut a02 = x02.squared_norm();
    let mut a12 = x12.squared_norm();

    let (mut big_x0, mut big_x1, mut big_x2) = (big_x[0], big_x[1], big_x[2]);
    let (mut x0, mut x1, mut x2) = (x[0], x[1], x[2]);

    // Switch X,x so that BC is the largest distance among {X01, X02, X12}
    if a01 > a02 {
        if a01 > a12 {
            std::mem::swap(&mut x0, &mut x2);
            std::mem::swap(&mut big_x0, &mut big_x2);
            std::mem::swap(&mut a01, &mut a12);
            x01 = -x12;
            x02 = -x02;
        }
    } else if a02 > a12 {
        std::mem::swap(&mut x0, &mut x1);
        std::mem::swap(&mut big_x0, &mut big_x1);
        std::mem::swap(&mut a02, &mut a12);
        x01 = -x01;
        x02 = x12;
    }

    let a12d = 1.0 / a12;
    let a = a01 * a12d;
    let b = a02 * a12d;

    let m01 = x0.dot(x1);
    let m02 = x0.dot(x2);
    let m12 = x1.dot(x2);

    // Ugly parameters to simplify the calculation
    let m12sq = -m12 * m12 + 1.0;
    let m02sq = -1.0 + m02 * m02;
    let m01sq = -1.0 + m01 * m01;
    let ab = a * b;
    let bsq = b * b;
    let asq = a * a;
    let m013 = -2.0 + 2.0 * m01 * m02 * m12;
    let bsqm12sq = bsq * m12sq;
    let asqm12sq = asq * m12sq;
    let abm12sq = 2.0 * ab * m12sq;

    let k3_inv = 1.0 / (bsqm12sq + b * m02sq);
    let k2 = k3_inv * ((-1.0 + a) * m02sq + abm12sq + bsqm12sq + b * m013);
    let k1 = k3_inv * (asqm12sq + abm12sq + a * m013 + (-1.0 + b) * m01sq);
    let k0 = k3_inv * (asqm12sq + a * m01sq);

    let (g, s) = solve_cubic_single_real(k2, k1, k0);

    let c = Matrix3d::new(
        -a + s * (1.0 - b),
        -m02 * s,
        a * m12 + b * m12 * s,
        -m02 * s,
        s + 1.0,
        -m01,
        a * m12 + b * m12 * s,
        -m01,
        -a - b * s + 1.0,
    );

    let (pq0, pq1) = compute_pq(c);

    let xx = Matrix3d::from_columns(x01, x02, x01.cross(x02)).inverse();

    let mut n_sols = 0;
    let x_rays = [x0, x1, x2];

    for pq in [pq0, pq1] {
        // [p0 p1 p2] * [1; x; y] = 0, or [p0 p1 p2] * [d2; d0; d1] = 0
        let (p0, p1, p2) = (pq.x, pq.y, pq.z);
        // here we run into trouble if p0 is zero,
        // so depending on which is larger, we solve for either d0 or d1
        // The case p0 = p1 = 0 is degenerate and can be ignored
        let switch12 = p0.abs() <= p1.abs();

        if switch12 {
            // eliminate d0
            let w0 = -p0 / p1;
            let w1 = -p2 / p1;
            let ca = 1.0 / (w1 * w1 - b);
            let cb = 2.0 * (b * m12 - m02 * w1 + w0 * w1) * ca;
            let cc = (w0 * w0 - 2.0 * m02 * w0 - b + 1.0) * ca;
            let Some((tau0, tau1)) = root2real(cb, cc) else {
                continue;
            };

            for tau in [tau0, tau1] {
                if tau <= 0.0 {
                    continue;
                }

                // positive only
                let mut d2 = fns::sqrt(a12 / (tau * (tau - 2.0 * m12) + 1.0));
                let mut d1 = tau * d2;
                let mut d0 = w0 * d2 + w1 * d1;
                if d0 < 0.0 {
                    continue;
                }

                refine_lambda(&mut d0, &mut d1, &mut d2, a01, a02, a12, m01, m02, m12);
                add_pose([d0, d1, d2], x_rays, big_x0, xx, output);
                n_sols += 1;
            }
        } else {
            let w0 = -p1 / p0;
            let w1 = -p2 / p0;
            let ca = 1.0 / (-a * w1 * w1 + 2.0 * a * m12 * w1 - a + 1.0);
            let cb = 2.0 * (a * m12 * w0 - m01 - a * w0 * w1) * ca;
            let cc = (1.0 - a * w0 * w0) * ca;

            let Some((tau0, tau1)) = root2real(cb, cc) else {
                continue;
            };

            for tau in [tau0, tau1] {
                if tau <= 0.0 {
                    continue;
                }

                let mut d0 = fns::sqrt(a01 / (tau * (tau - 2.0 * m01) + 1.0));
                let mut d1 = tau * d0;
                let mut d2 = w0 * d0 + w1 * d1;

                if d2 < 0.0 {
                    continue;
                }

                refine_lambda(&mut d0, &mut d1, &mut d2, a01, a02, a12, m01, m02, m12);
                add_pose([d0, d1, d2], x_rays, big_x0, xx, output);
                n_sols += 1;
            }
        }

        if n_sols > 0 && g {
            break;
        }
    }

    output.len()
}

fn add_pose(
    d: [f64; 3],
    x: [Vector3d; 3],
    big_x0: Vector3d,
    xx: Matrix3d,
    output: &mut Vec<CameraPose>,
) {
    let v1 = d[0] * x[0] - d[1] * x[1];
    let v2 = d[0] * x[0] - d[2] * x[2];
    let yy = Matrix3d::from_columns(v1, v2, v1.cross(v2));
    let r = yy * xx;
    output.push(CameraPose::from_rt(r, d[0] * x[0] - r * big_x0));
}

/// Real roots of `x^2 + b x + c`, or `None` when there are none. Port of `poselib::root2real`
/// (`p3p_common.h`): a slightly negative discriminant counts as a root (with the second root
/// set to -2, as PoseLib does).
pub fn root2real(b: f64, c: f64) -> Option<(f64, f64)> {
    const THRESHOLD: f64 = -1.0e-12;
    let v = b * b - 4.0 * c;
    if v < THRESHOLD {
        // PoseLib writes r1 = r2 = -0.5 b and returns v >= 0, which is false here.
        return None;
    }

    if v > THRESHOLD && v < 0.0 {
        return Some((-0.5 * b, -2.0));
    }

    let y = fns::sqrt(v);
    if b < 0.0 {
        Some((0.5 * (-b + y), 0.5 * (-b - y)))
    } else {
        Some((2.0 * c / (-b + y), 2.0 * c / (-b - y)))
    }
}

/// Port of `poselib::compute_pq` (`p3p_common.h`): splits the degenerate conic C into two
/// lines via its adjugate.
fn compute_pq(c_in: Matrix3d) -> (Vector3d, Vector3d) {
    // C(r, k) = c[r + 3k] (column-major).
    let mut c = c_in.to_column_major();
    let adj00 = c[7] * c[5] - c[4] * c[8];
    let adj11 = c[6] * c[2] - c[0] * c[8];
    let adj22 = c[3] * c[1] - c[0] * c[4];
    let adj01 = c[3] * c[8] - c[6] * c[5];
    let adj02 = c[6] * c[4] - c[3] * c[7];
    let adj10 = adj01;
    let adj12 = c[0] * c[7] - c[6] * c[1];
    let adj20 = adj02;
    let adj21 = adj12;

    let v = if adj00 > adj11 {
        if adj00 > adj22 {
            Vector3d::new(adj00, adj10, adj20) / fns::sqrt(adj00)
        } else {
            Vector3d::new(adj02, adj12, adj22) / fns::sqrt(adj22)
        }
    } else if adj11 > adj22 {
        Vector3d::new(adj01, adj11, adj21) / fns::sqrt(adj11)
    } else {
        Vector3d::new(adj02, adj12, adj22) / fns::sqrt(adj22)
    };

    c[3] -= v.z; // C(0, 1)
    c[6] += v.y; // C(0, 2)
    c[7] -= v.x; // C(1, 2)
    c[1] += v.z; // C(1, 0)
    c[2] -= v.y; // C(2, 0)
    c[5] += v.x; // C(2, 1)

    (
        Vector3d::new(c[0], c[1], c[2]),
        Vector3d::new(c[0], c[3], c[6]),
    )
}

/// Performs a few newton steps on the equations. Port of `poselib::refine_lambda`.
#[allow(clippy::too_many_arguments)]
fn refine_lambda(
    lambda1: &mut f64,
    lambda2: &mut f64,
    lambda3: &mut f64,
    a12: f64,
    a13: f64,
    a23: f64,
    b12: f64,
    b13: f64,
    b23: f64,
) {
    for _ in 0..5 {
        let (l1, l2, l3) = (*lambda1, *lambda2, *lambda3);
        let r1 = l1 * l1 - 2.0 * l1 * l2 * b12 + l2 * l2 - a12;
        let r2 = l1 * l1 - 2.0 * l1 * l3 * b13 + l3 * l3 - a13;
        let r3 = l2 * l2 - 2.0 * l2 * l3 * b23 + l3 * l3 - a23;
        if r1.abs() + r2.abs() + r3.abs() < 1e-10 {
            return;
        }

        let x11 = l1 - l2 * b12;
        let x12 = l2 - l1 * b12;
        let x21 = l1 - l3 * b13;
        let x23 = l3 - l1 * b13;
        let x32 = l2 - l3 * b23;
        let x33 = l3 - l2 * b23;
        let det_j = 0.5 / (x11 * x23 * x32 + x12 * x21 * x33); // half minus inverse determinant
                                                               // This uses the closed form of the inverse for the jacobean.
                                                               // Due to the zero elements this actually becomes quite nice.
        *lambda1 += (-x23 * x32 * r1 - x12 * x33 * r2 + x12 * x23 * r3) * det_j;
        *lambda2 += (-x21 * x33 * r1 + x11 * x33 * r2 - x11 * x23 * r3) * det_j;
        *lambda3 += (x21 * x32 * r1 - x11 * x32 * r2 - x12 * x21 * r3) * det_j;
    }
}

/// One real root of `x^3 + c2 x^2 + c1 x + c0`, and whether it is the only real root. Port of
/// `poselib::univariate::solve_cubic_single_real`.
pub fn solve_cubic_single_real(c2: f64, c1: f64, c0: f64) -> (bool, f64) {
    let a = c1 - c2 * c2 / 3.0;
    let mut b = (2.0 * c2 * c2 * c2 - 9.0 * c2 * c1) / 27.0 + c0;
    let mut c = b * b / 4.0 + a * a * a / 27.0;
    if c != 0.0 {
        if c > 0.0 {
            c = fns::sqrt(c);
            b *= -0.5;
            let root = fns::cbrt(b + c) + fns::cbrt(b - c) - c2 / 3.0;
            return (true, root);
        }

        c = 3.0 * b / (2.0 * a) * fns::sqrt(-3.0 / a);
        let root = 2.0 * fns::sqrt(-a / 3.0) * fns::cos(fns::acos(c) / 3.0) - c2 / 3.0;
        (false, root)
    } else {
        let root = -c2 / 3.0 + if a != 0.0 { 3.0 * b / a } else { 0.0 };
        (false, root)
    }
}
