//! Port of PoseLib's `PoseLib/solvers/gen_relpose_6pt.cc` (`poselib::gen_relpose_6pt`;
//! BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see THIRD_PARTY_NOTICES.md; commit
//! fa7280fee27f97aff31ae7f98bab7f583fac7d08), through colmap-sharp's
//! `Estimators/Solvers/PoseLib/GenRelpose6pt.cs`: the minimal generalized relative pose solver
//! from 6 correspondences between two generalized (multi-camera) cameras, generated with the
//! method of Larsson et al., "Efficient Solvers for Minimal Problems by Syzygy-based
//! Reduction", CVPR 2017. COLMAP's `GR6PEstimator` (`estimators/solvers/
//! generalized_relative_pose.cc`) is its only caller. The generated parts are split out:
//! `gen_relpose_6pt_coeff_matrix.rs` (the 15 equations), `gen_relpose_6pt_poly.rs` (their
//! polynomial products) and `gen_relpose_6pt_template0.rs` / `template1.rs` (the elimination
//! template indices).
//!
//! Pipeline, as in PoseLib: the 84 x 15 coefficients are scattered into the elimination
//! template `[C0 | C1]`, `C12 = C0^-1 C1` (partial-pivot LU), the 64 x 64 action matrix is
//! read off C12, its real eigenvalues (`|imag| < 1e-6`) give the third Cayley parameter, and
//! the other two come from the structured back-substitution of `fast_eigenvector_solver`
//! (PoseLib's `USE_FAST_EIGENVECTOR_SOLVER` path, which it defines). Each rotation's
//! translation is the least-squares solution of the epipolar constraints, poses failing
//! cheirality are dropped, and the survivors get 5 Newton steps (`root_refinement`).
//!
//! Only eigenvalues of the action matrix are used, never its eigenvectors, so the
//! eigenvector-phase freedom of [`crate::linalg::EigenSolver`] cannot reach the result. The
//! order of the real eigenvalues (Schur order) sets the order of the returned poses, which
//! may differ from Eigen's (docs/CPP_DIVERGENCES.md entry 256); RANSAC treats the output as a
//! set. Tier B. Tests:
//! `tests/estimators/rust_only_gen_relpose_6pt.rs`.

use super::camera_pose::{quat_step_pre, quat_to_rotmat, CameraPose};
use super::essential::{check_cheirality_generalized, quat_rotate};
use super::gen_relpose_6pt_coeff_matrix::setup_coeff_matrix;
use super::gen_relpose_6pt_template0::{C0_IND, COEFFS0_IND};
use super::gen_relpose_6pt_template1::{C1_IND, COEFFS1_IND};
use crate::linalg::{
    EigenSolver, HouseholderQr, Llt, Matrix3d, MatrixXd, PartialPivLu, Vector3d, Vector4d, VectorXd,
};
use crate::math::fns;

/// `(action matrix row, C12 row)` of every row PoseLib sets to `-C12.row(r)`.
const NEGATED_ROWS: [(usize, usize); 21] = [
    (5, 78),
    (6, 79),
    (7, 80),
    (9, 81),
    (10, 82),
    (12, 83),
    (15, 84),
    (16, 85),
    (18, 86),
    (22, 87),
    (26, 88),
    (27, 89),
    (29, 90),
    (33, 91),
    (38, 92),
    (43, 93),
    (44, 94),
    (47, 95),
    (51, 96),
    (56, 97),
    (63, 98),
];

/// `(row, column)` of every unit entry of the action matrix, in PoseLib's order.
const UNIT_ENTRIES: [(usize, usize); 43] = [
    (0, 57),
    (1, 34),
    (2, 19),
    (3, 11),
    (4, 7),
    (8, 10),
    (11, 12),
    (13, 17),
    (14, 16),
    (17, 18),
    (19, 20),
    (20, 21),
    (21, 22),
    (23, 30),
    (24, 28),
    (25, 27),
    (28, 29),
    (30, 31),
    (31, 32),
    (32, 33),
    (34, 35),
    (35, 36),
    (36, 37),
    (37, 38),
    (39, 52),
    (40, 48),
    (41, 45),
    (42, 44),
    (45, 46),
    (46, 47),
    (48, 49),
    (49, 50),
    (50, 51),
    (52, 53),
    (53, 54),
    (54, 55),
    (55, 56),
    (57, 58),
    (58, 59),
    (59, 60),
    (60, 61),
    (61, 62),
    (62, 63),
];

/// Solves for the generalized relative pose `(R, t)` with
/// `lambda2 * x2 + p2 = R * (lambda1 * x1 + p1) + t` from 6 correspondences, where `p` are the
/// ray origins and `x` the unit ray directions in each rig. Clears `output`, appends the
/// solutions (`rig2_from_rig1`) and returns their count. Port of `poselib::gen_relpose_6pt`.
pub fn gen_relpose_6pt(
    p1: &[Vector3d],
    x1: &[Vector3d],
    p2: &[Vector3d],
    x2: &[Vector3d],
    output: &mut Vec<CameraPose>,
) -> usize {
    let mut m = [0.0; 84 * 15];
    setup_coeff_matrix(p1, x1, p2, x2, &mut m);

    let mut c0 = MatrixXd::zeros(99, 99);
    let mut c1 = MatrixXd::zeros(99, 64);
    for (&dst, &src) in C0_IND.iter().zip(&COEFFS0_IND) {
        c0.as_mut_slice()[dst] = m[src];
    }
    for (&dst, &src) in C1_IND.iter().zip(&COEFFS1_IND) {
        c1.as_mut_slice()[dst] = m[src];
    }

    // C12 = C0.partialPivLu().solve(C1). (colmap-sharp eliminates [C0 | C1] together and
    // back-substitutes only the rows the action matrix reads; both are bit-identical to this
    // column-by-column solve.)
    let c12 = PartialPivLu::new(&c0).solve_matrix(&c1);

    // Setup action matrix
    let mut am = MatrixXd::zeros(64, 64);
    for &(r, c) in &UNIT_ENTRIES {
        am[(r, c)] = 1.0;
    }
    for &(row, c12_row) in &NEGATED_ROWS {
        for j in 0..64 {
            am[(row, j)] = -c12[(c12_row, j)];
        }
    }

    // Here we only compute eigenvalues and we use the structured backsubsitution to solve for
    // the eigenvectors
    let eigv: Vec<f64> = EigenSolver::new(&am, false)
        .eigenvalues()
        .iter()
        .filter(|d| d.im.abs() < 1e-6)
        .map(|d| d.re)
        .collect();

    let sols = fast_eigenvector_solver(&eigv, &am);

    output.clear();
    for w in sols {
        // From each solution we compute the rotation and solve for the translation
        let q_norm = fns::sqrt(1.0 * 1.0 + w.x * w.x + w.y * w.y + w.z * w.z);
        let q = Vector4d::new(1.0 / q_norm, w.x / q_norm, w.y / q_norm, w.z / q_norm);
        let r = quat_to_rotmat(q);

        // Solve for the translation
        let mut a = Matrix3d::zeros();
        let mut b = Vector3d::zeros();
        for i in 0..6 {
            let u = (r * x1[i]).cross(x2[i]);
            let v = p2[i] - r * p1[i];
            a += outer_product(u, u);
            b += u * u.dot(v);
        }

        let pose = CameraPose::new(q, solve_llt3(&a, b));

        // Filter solution using cheirality
        let cheiral_ok =
            (0..6).all(|k| check_cheirality_generalized(&pose, p1[k], x1[k], p2[k], x2[k], 0.0));
        if cheiral_ok {
            output.push(pose);
        }
    }

    root_refinement(p1, x1, p2, x2, output);

    output.len()
}

/// Solves for the eigenvector by using structured backsubstitution (i.e. substituting the
/// eigenvalue into the eigenvector using the known structure to get a reduced linear system).
/// Port of PoseLib's `fast_eigenvector_solver`; returns `(s(14), s(19), eigenvalue)` for each
/// eigenvalue.
fn fast_eigenvector_solver(eigv: &[f64], am: &MatrixXd) -> Vec<Vector3d> {
    const IND: [usize; 21] = [
        5, 6, 7, 9, 10, 12, 15, 16, 18, 22, 26, 27, 29, 33, 38, 43, 44, 47, 51, 56, 63,
    ];

    // Truncated action matrix containing non-trivial rows.
    let ams = |k: usize, j: usize| am[(IND[k], j)];

    let mut zi = [0.0; 8];
    let mut aa = MatrixXd::zeros(21, 20);
    let mut rhs = VectorXd::zeros(21);
    let mut sols = Vec::with_capacity(eigv.len());
    for &ev in eigv {
        zi[0] = ev;
        for j in 1..8 {
            zi[j] = zi[j - 1] * ev;
        }

        for k in 0..21 {
            aa[(k, 0)] = ams(k, 5);
            aa[(k, 1)] = ams(k, 6);
            aa[(k, 2)] = ams(k, 4) + zi[0] * ams(k, 7);
            aa[(k, 3)] = ams(k, 9);
            aa[(k, 4)] = ams(k, 8) + zi[0] * ams(k, 10);
            aa[(k, 5)] = ams(k, 3) + zi[0] * ams(k, 11) + zi[1] * ams(k, 12);
            aa[(k, 6)] = ams(k, 15);
            aa[(k, 7)] = ams(k, 14) + zi[0] * ams(k, 16);
            aa[(k, 8)] = ams(k, 13) + zi[0] * ams(k, 17) + zi[1] * ams(k, 18);
            aa[(k, 9)] = ams(k, 2)
                + zi[0] * ams(k, 19)
                + zi[1] * ams(k, 20)
                + zi[2] * ams(k, 21)
                + zi[3] * ams(k, 22);
            aa[(k, 10)] = ams(k, 26);
            aa[(k, 11)] = ams(k, 25) + zi[0] * ams(k, 27);
            aa[(k, 12)] = ams(k, 24) + zi[0] * ams(k, 28) + zi[1] * ams(k, 29);
            aa[(k, 13)] = ams(k, 23)
                + zi[0] * ams(k, 30)
                + zi[1] * ams(k, 31)
                + zi[2] * ams(k, 32)
                + zi[3] * ams(k, 33);
            aa[(k, 14)] = ams(k, 1)
                + zi[0] * ams(k, 34)
                + zi[1] * ams(k, 35)
                + zi[2] * ams(k, 36)
                + zi[3] * ams(k, 37)
                + zi[4] * ams(k, 38);
            aa[(k, 15)] = ams(k, 43);
            aa[(k, 16)] = ams(k, 42) + zi[0] * ams(k, 44);
            aa[(k, 17)] = ams(k, 41) + zi[0] * ams(k, 45) + zi[1] * ams(k, 46) + zi[2] * ams(k, 47);
            aa[(k, 18)] = ams(k, 40)
                + zi[0] * ams(k, 48)
                + zi[1] * ams(k, 49)
                + zi[2] * ams(k, 50)
                + zi[3] * ams(k, 51);
            aa[(k, 19)] = ams(k, 39)
                + zi[0] * ams(k, 52)
                + zi[1] * ams(k, 53)
                + zi[2] * ams(k, 54)
                + zi[3] * ams(k, 55)
                + zi[4] * ams(k, 56);
            let mut col20 = ams(k, 0)
                + zi[0] * ams(k, 57)
                + zi[1] * ams(k, 58)
                + zi[2] * ams(k, 59)
                + zi[3] * ams(k, 60)
                + zi[4] * ams(k, 61)
                + zi[5] * ams(k, 62)
                + zi[6] * ams(k, 63);

            // AA(20, 20) = AA(20, 20) - zi[7] lands in the right-hand-side column.
            if k == 20 {
                col20 -= zi[7];
            }

            rhs[k] = -col20;
        }

        const DIAGONAL_POWERS: [usize; 20] =
            [0, 0, 1, 0, 1, 2, 0, 1, 2, 4, 0, 1, 2, 4, 5, 0, 1, 3, 4, 5];
        for (i, &power) in DIAGONAL_POWERS.iter().enumerate() {
            aa[(i, i)] -= zi[power];
        }

        // AA.leftCols(20).householderQr().solve(-AA.col(20))
        let s = HouseholderQr::new(&aa).solve(&rhs);
        sols.push(Vector3d::new(s[14], s[19], zi[0]));
    }

    sols
}

/// Performs Newton iterations on the epipolar constraints. Port of PoseLib's
/// `root_refinement` (at most 5 steps, stopping once the residual norm is below 1e-12).
fn root_refinement(
    p1: &[Vector3d],
    x1: &[Vector3d],
    p2: &[Vector3d],
    x2: &[Vector3d],
    output: &mut [CameraPose],
) {
    let mut qq1 = [Vector3d::zeros(); 6];
    let mut qq2 = [Vector3d::zeros(); 6];
    for pt_k in 0..6 {
        qq1[pt_k] = x1[pt_k].cross(p1[pt_k]);
        qq2[pt_k] = x2[pt_k].cross(p2[pt_k]);
    }

    let mut jacobian = MatrixXd::zeros(6, 6);
    let mut res = VectorXd::zeros(6);
    for pose in output.iter_mut() {
        for _ in 0..5 {
            // compute residual and jacobian
            for pt_k in 0..6 {
                let x2t = x2[pt_k].cross(pose.t);
                let rx1 = quat_rotate(pose.q, x1[pt_k]);
                let rqq1 = quat_rotate(pose.q, qq1[pt_k]);

                res[pt_k] = (x2t - qq2[pt_k]).dot(rx1) - x2[pt_k].dot(rqq1);
                let j_rotation = -x2t.cross(rx1) + qq2[pt_k].cross(rx1) + x2[pt_k].cross(rqq1);
                let j_translation = -x2[pt_k].cross(rx1);
                for c in 0..3 {
                    jacobian[(pt_k, c)] = j_rotation[c];
                    jacobian[(pt_k, 3 + c)] = j_translation[c];
                }
            }

            if res.norm() < 1e-12 {
                break;
            }

            let dp = PartialPivLu::new(&jacobian).solve(&res);

            let w = Vector3d::new(-dp[0], -dp[1], -dp[2]);
            pose.q = quat_step_pre(pose.q, w);
            pose.t -= Vector3d::new(dp[3], dp[4], dp[5]);
        }
    }
}

/// `a b^T`.
fn outer_product(a: Vector3d, b: Vector3d) -> Matrix3d {
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

/// `A.llt().solve(b)` for the 3 x 3 normal equations of the translation.
fn solve_llt3(a: &Matrix3d, b: Vector3d) -> Vector3d {
    let mut m = MatrixXd::zeros(3, 3);
    for r in 0..3 {
        for c in 0..3 {
            m[(r, c)] = a[(r, c)];
        }
    }

    Llt::new(&m)
        .solve(&VectorXd::from_slice(&[b.x, b.y, b.z]))
        .to_vector3d()
}
