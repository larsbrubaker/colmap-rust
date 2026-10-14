//! `setup_coeff_matrix` and `pt_index` of PoseLib's `PoseLib/solvers/gen_relpose_6pt.cc`
//! (BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see THIRD_PARTY_NOTICES.md;
//! commit fa7280fee27f97aff31ae7f98bab7f583fac7d08), through colmap-sharp's
//! `Estimators/Solvers/PoseLib/GenRelpose6pt.CoeffMatrix.cs`. Each of the 15 equations takes 4
//! of the 6 correspondences, eliminates the translation through a 3 x 3 matrix whose entries
//! are quadratic in the Cayley parameters (F1, F2, F3 are its columns, 10 monomials each), and
//! expands its determinant into a quartic with 84 coefficients (the products are in
//! `gen_relpose_6pt_poly.rs`). The F expressions are generated code transcribed by a script
//! (operand order kept). The coefficients feed [`super::gen_relpose_6pt`].

use super::gen_relpose_6pt_poly::{mul2_2, mul2_2m, mul2_4p};
use crate::linalg::Vector3d;

/// Which combinations of 4 points are used to generate the 15 equations.
const PT_INDEX: [usize; 60] = [
    0, 1, 2, 3, 0, 1, 2, 4, 0, 1, 2, 5, 0, 1, 3, 4, 0, 1, 3, 5, 0, 1, 4, 5, 0, 2, 3, 4, 0, 2, 3, 5,
    0, 2, 4, 5, 0, 3, 4, 5, 1, 2, 3, 4, 1, 2, 3, 5, 1, 2, 4, 5, 1, 3, 4, 5, 2, 3, 4, 5,
];

/// Computes the 84 x 15 matrix of coefficients for the 15 equations (column-major, one column
/// of 84 monomial coefficients per equation). Port of PoseLib's `setup_coeff_matrix`.
pub(super) fn setup_coeff_matrix(
    pp1: &[Vector3d],
    xx1: &[Vector3d],
    pp2: &[Vector3d],
    xx2: &[Vector3d],
    m: &mut [f64; 84 * 15],
) {
    // F1, F2, F3 are Eigen 10x3 column-major matrices, so column i starts at 10 * i.
    let mut f1 = [0.0; 30];
    let mut f2 = [0.0; 30];
    let mut f3 = [0.0; 30];
    let mut p4 = [0.0; 35];

    let mut qq1 = [Vector3d::zeros(); 6];
    let mut qq2 = [Vector3d::zeros(); 6];
    for k in 0..6 {
        qq1[k] = xx1[k].cross(pp1[k]);
        qq2[k] = xx2[k].cross(pp2[k]);
    }

    m.fill(0.0);
    for eq_k in 0..15 {
        let i0 = PT_INDEX[4 * eq_k];

        let x1 = xx1[i0];
        let p1 = pp1[i0];
        let x2 = xx2[i0];
        let p2 = pp2[i0];

        // Compute 3x3 matrix where each element is quadratic in cayley parameters
        // F1 is the first column of the matrix, etc..
        // This is for eliminating the translation.
        for i in 0..3 {
            let i1 = PT_INDEX[4 * eq_k + i + 1];

            let xp1 = xx1[i1];
            let qp1 = qq1[i1];
            let xp2 = xx2[i1];
            let qp2 = qq2[i1];

            f1[10 * i] = qp1.x * xp2.x + qp2.x * xp1.x
                - qp1.y * xp2.y
                - qp2.y * xp1.y
                - qp1.z * xp2.z
                - qp2.z * xp1.z
                + xp1.x * (xp2.z * (p1.y + p2.y) - xp2.y * (p1.z + p2.z))
                + xp1.z * (xp2.x * (p1.y + p2.y) + xp2.y * (p1.x - p2.x))
                - xp1.y * (xp2.x * (p1.z + p2.z) + xp2.z * (p1.x - p2.x));
            f1[1 + 10 * i] = 2.0 * qp1.x * xp2.y
                + 2.0 * qp1.y * xp2.x
                + 2.0 * qp2.x * xp1.y
                + 2.0 * qp2.y * xp1.x
                - xp1.x * (2.0 * p2.x * xp2.z - xp2.x * (2.0 * p1.z + 2.0 * p2.z))
                + xp1.y * (2.0 * p2.y * xp2.z - xp2.y * (2.0 * p1.z + 2.0 * p2.z))
                - xp1.z * (2.0 * p1.x * xp2.x - 2.0 * p1.y * xp2.y);
            f1[2 + 10 * i] = qp1.y * xp2.y - qp2.x * xp1.x - qp1.x * xp2.x + qp2.y * xp1.y
                - qp1.z * xp2.z
                - qp2.z * xp1.z
                - xp1.y * (xp2.z * (p1.x + p2.x) - xp2.x * (p1.z + p2.z))
                - xp1.z * (xp2.y * (p1.x + p2.x) + xp2.x * (p1.y - p2.y))
                + xp1.x * (xp2.y * (p1.z + p2.z) + xp2.z * (p1.y - p2.y));
            f1[3 + 10 * i] = 2.0 * qp1.x * xp2.z
                + 2.0 * qp1.z * xp2.x
                + 2.0 * qp2.x * xp1.z
                + 2.0 * qp2.z * xp1.x
                + xp1.x * (2.0 * p2.x * xp2.y - xp2.x * (2.0 * p1.y + 2.0 * p2.y))
                - xp1.z * (2.0 * p2.z * xp2.y - xp2.z * (2.0 * p1.y + 2.0 * p2.y))
                + xp1.y * (2.0 * p1.x * xp2.x - 2.0 * p1.z * xp2.z);
            f1[4 + 10 * i] = 2.0 * qp1.y * xp2.z
                + 2.0 * qp1.z * xp2.y
                + 2.0 * qp2.y * xp1.z
                + 2.0 * qp2.z * xp1.y
                - xp1.y * (2.0 * p2.y * xp2.x - xp2.y * (2.0 * p1.x + 2.0 * p2.x))
                + xp1.z * (2.0 * p2.z * xp2.x - xp2.z * (2.0 * p1.x + 2.0 * p2.x))
                - xp1.x * (2.0 * p1.y * xp2.y - 2.0 * p1.z * xp2.z);
            f1[5 + 10 * i] =
                qp1.z * xp2.z - qp2.x * xp1.x - qp1.y * xp2.y - qp2.y * xp1.y - qp1.x * xp2.x
                    + qp2.z * xp1.z
                    + xp1.z * (xp2.y * (p1.x + p2.x) - xp2.x * (p1.y + p2.y))
                    + xp1.y * (xp2.z * (p1.x + p2.x) + xp2.x * (p1.z - p2.z))
                    - xp1.x * (xp2.z * (p1.y + p2.y) + xp2.y * (p1.z - p2.z));
            f1[6 + 10 * i] = 2.0 * qp1.y * xp2.z - 2.0 * qp1.z * xp2.y - 2.0 * qp2.y * xp1.z
                + 2.0 * qp2.z * xp1.y
                - xp1.y * (2.0 * p2.y * xp2.x + xp2.y * (2.0 * p1.x - 2.0 * p2.x))
                - xp1.z * (2.0 * p2.z * xp2.x + xp2.z * (2.0 * p1.x - 2.0 * p2.x))
                + xp1.x * (2.0 * p1.y * xp2.y + 2.0 * p1.z * xp2.z);
            f1[7 + 10 * i] = 2.0 * qp1.z * xp2.x - 2.0 * qp1.x * xp2.z + 2.0 * qp2.x * xp1.z
                - 2.0 * qp2.z * xp1.x
                - xp1.x * (2.0 * p2.x * xp2.y + xp2.x * (2.0 * p1.y - 2.0 * p2.y))
                - xp1.z * (2.0 * p2.z * xp2.y + xp2.z * (2.0 * p1.y - 2.0 * p2.y))
                + xp1.y * (2.0 * p1.x * xp2.x + 2.0 * p1.z * xp2.z);
            f1[8 + 10 * i] = 2.0 * qp1.x * xp2.y - 2.0 * qp1.y * xp2.x - 2.0 * qp2.x * xp1.y
                + 2.0 * qp2.y * xp1.x
                - xp1.x * (2.0 * p2.x * xp2.z + xp2.x * (2.0 * p1.z - 2.0 * p2.z))
                - xp1.y * (2.0 * p2.y * xp2.z + xp2.y * (2.0 * p1.z - 2.0 * p2.z))
                + xp1.z * (2.0 * p1.x * xp2.x + 2.0 * p1.y * xp2.y);
            f1[9 + 10 * i] = xp1.y * (xp2.z * (p1.x - p2.x) - xp2.x * (p1.z - p2.z))
                - xp1.z * (xp2.y * (p1.x - p2.x) - xp2.x * (p1.y - p2.y))
                - xp1.x * (xp2.z * (p1.y - p2.y) - xp2.y * (p1.z - p2.z))
                + qp1.x * xp2.x
                + qp2.x * xp1.x
                + qp1.y * xp2.y
                + qp2.y * xp1.y
                + qp1.z * xp2.z
                + qp2.z * xp1.z;
            f2[10 * i] = xp1.z * (x1.x * xp2.y + x1.y * xp2.x)
                - xp1.y * (x1.x * xp2.z + x1.z * xp2.x)
                + xp1.x * (x1.y * xp2.z - x1.z * xp2.y);
            f2[1 + 10 * i] = 2.0 * x1.z * xp1.x * xp2.x
                - xp1.z * (2.0 * x1.x * xp2.x - 2.0 * x1.y * xp2.y)
                - 2.0 * x1.z * xp1.y * xp2.y;
            f2[2 + 10 * i] = xp1.x * (x1.y * xp2.z + x1.z * xp2.y)
                - xp1.z * (x1.x * xp2.y + x1.y * xp2.x)
                - xp1.y * (x1.x * xp2.z - x1.z * xp2.x);
            f2[3 + 10 * i] = xp1.y * (2.0 * x1.x * xp2.x - 2.0 * x1.z * xp2.z)
                - 2.0 * x1.y * xp1.x * xp2.x
                + 2.0 * x1.y * xp1.z * xp2.z;
            f2[4 + 10 * i] = 2.0 * x1.x * xp1.y * xp2.y
                - xp1.x * (2.0 * x1.y * xp2.y - 2.0 * x1.z * xp2.z)
                - 2.0 * x1.x * xp1.z * xp2.z;
            f2[5 + 10 * i] = xp1.y * (x1.x * xp2.z + x1.z * xp2.x)
                + xp1.z * (x1.x * xp2.y - x1.y * xp2.x)
                - xp1.x * (x1.y * xp2.z + x1.z * xp2.y);
            f2[6 + 10 * i] = xp1.x * (2.0 * x1.y * xp2.y + 2.0 * x1.z * xp2.z)
                - 2.0 * x1.x * xp1.y * xp2.y
                - 2.0 * x1.x * xp1.z * xp2.z;
            f2[7 + 10 * i] = xp1.y * (2.0 * x1.x * xp2.x + 2.0 * x1.z * xp2.z)
                - 2.0 * x1.y * xp1.x * xp2.x
                - 2.0 * x1.y * xp1.z * xp2.z;
            f2[8 + 10 * i] = xp1.z * (2.0 * x1.x * xp2.x + 2.0 * x1.y * xp2.y)
                - 2.0 * x1.z * xp1.x * xp2.x
                - 2.0 * x1.z * xp1.y * xp2.y;
            f2[9 + 10 * i] = xp1.y * (x1.x * xp2.z - x1.z * xp2.x)
                - xp1.z * (x1.x * xp2.y - x1.y * xp2.x)
                - xp1.x * (x1.y * xp2.z - x1.z * xp2.y);
            f3[10 * i] = xp1.y * (x2.x * xp2.z - x2.z * xp2.x)
                - xp1.z * (x2.x * xp2.y - x2.y * xp2.x)
                + xp1.x * (x2.y * xp2.z - x2.z * xp2.y);
            f3[1 + 10 * i] = xp1.y * (2.0 * x2.y * xp2.z - 2.0 * x2.z * xp2.y)
                - xp1.x * (2.0 * x2.x * xp2.z - 2.0 * x2.z * xp2.x);
            f3[2 + 10 * i] = -xp1.z * (x2.x * xp2.y - x2.y * xp2.x)
                - xp1.y * (x2.x * xp2.z - x2.z * xp2.x)
                - xp1.x * (x2.y * xp2.z - x2.z * xp2.y);
            f3[3 + 10 * i] = xp1.x * (2.0 * x2.x * xp2.y - 2.0 * x2.y * xp2.x)
                + xp1.z * (2.0 * x2.y * xp2.z - 2.0 * x2.z * xp2.y);
            f3[4 + 10 * i] = xp1.y * (2.0 * x2.x * xp2.y - 2.0 * x2.y * xp2.x)
                - xp1.z * (2.0 * x2.x * xp2.z - 2.0 * x2.z * xp2.x);
            f3[5 + 10 * i] = xp1.z * (x2.x * xp2.y - x2.y * xp2.x)
                + xp1.y * (x2.x * xp2.z - x2.z * xp2.x)
                - xp1.x * (x2.y * xp2.z - x2.z * xp2.y);
            f3[6 + 10 * i] = xp1.y * (2.0 * x2.x * xp2.y - 2.0 * x2.y * xp2.x)
                + xp1.z * (2.0 * x2.x * xp2.z - 2.0 * x2.z * xp2.x);
            f3[7 + 10 * i] = xp1.z * (2.0 * x2.y * xp2.z - 2.0 * x2.z * xp2.y)
                - xp1.x * (2.0 * x2.x * xp2.y - 2.0 * x2.y * xp2.x);
            f3[8 + 10 * i] = -xp1.x * (2.0 * x2.x * xp2.z - 2.0 * x2.z * xp2.x)
                - xp1.y * (2.0 * x2.y * xp2.z - 2.0 * x2.z * xp2.y);
            f3[9 + 10 * i] = xp1.z * (x2.x * xp2.y - x2.y * xp2.x)
                - xp1.y * (x2.x * xp2.z - x2.z * xp2.x)
                + xp1.x * (x2.y * xp2.z - x2.z * xp2.y);
        }

        let c = &mut m[84 * eq_k..84 * eq_k + 84];

        // Compute the determinant by expansion along first column
        mul2_2(&f2[10..], &f3[20..], &mut p4);
        mul2_2m(&f2[20..], &f3[10..], &mut p4);
        mul2_4p(&f1, &p4, c);

        mul2_2(&f2[20..], &f3, &mut p4);
        mul2_2m(&f2, &f3[20..], &mut p4);
        mul2_4p(&f1[10..], &p4, c);

        mul2_2(&f2, &f3[10..], &mut p4);
        mul2_2m(&f2[10..], &f3, &mut p4);
        mul2_4p(&f1[20..], &p4, c);
    }
}
