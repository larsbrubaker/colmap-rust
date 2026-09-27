// Rust-only (COLMAP has no test for Eigen itself): Matrix2d, Matrix2x3d, Matrix3x2d, Matrix3d,
// Matrix3x4d, Matrix4d and Matrix6d. Port of colmap-sharp's
// `ColmapSharp.Tests/LinearAlgebra/MatrixTests.cs`, plus the 2x3 / 3x2 shapes. Covers the
// layout contract (row-major `new` like Eigen's comma initializer, column-major storage like
// Eigen's memory), products, transpose, determinant and inverse (inverse * M = I within
// isApprox), and the block helpers COLMAP's Rigid3d/Sim3d use.

use colmap_rust::linalg::{
    Matrix2d, Matrix2x3d, Matrix3d, Matrix3x2d, Matrix3x4d, Matrix4d, Matrix6d, Vector2d, Vector3d,
    Vector4d,
};

const A3: Matrix3d = Matrix3d::new(2.0, -1.0, 0.5, 3.0, 4.0, -2.0, 1.0, 0.25, 5.0);
const A4: Matrix4d = Matrix4d::new(
    4.0, 1.0, -2.0, 0.5, 1.0, 3.0, 0.0, -1.0, -2.0, 0.75, 5.0, 2.0, 0.5, -1.0, 2.0, 6.0,
);

#[test]
fn rust_only_layout_row_major_constructor_and_column_major_storage() {
    let from_columns = Matrix3d::from_column_major([1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    let m = Matrix3d::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0);
    assert_eq!(from_columns, m);
    assert_eq!(m[(0, 2)], 3.0);
    assert_eq!(m[(2, 0)], 7.0);
    assert_eq!(m.as_slice(), &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
    assert_eq!(m.row(1), Vector3d::new(4.0, 5.0, 6.0));
    assert_eq!(m.col(1), Vector3d::new(2.0, 5.0, 8.0));
    assert_eq!(Matrix3d::from_rows(m.row(0), m.row(1), m.row(2)), m);
    assert_eq!(Matrix3d::from_columns(m.col(0), m.col(1), m.col(2)), m);
    assert_eq!(
        m.transpose(),
        Matrix3d::from_column_major([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    );

    let mut n = m;
    n[(1, 2)] = -6.0;
    assert_eq!(n.as_slice()[7], -6.0);
}

#[test]
#[should_panic(expected = "out of range")]
fn rust_only_index_out_of_range_panics() {
    // (3, 0) would land inside the 9-element storage; the index check must still reject it.
    let _ = Matrix3d::identity()[(3, 0)];
}

#[test]
fn rust_only_matrix3d_products_determinant_inverse() {
    let m = Matrix3d::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0);
    assert_eq!(
        m * Vector3d::new(1.0, 0.0, -1.0),
        Vector3d::new(-2.0, -2.0, -3.0)
    );
    assert_eq!(m * Matrix3d::identity(), m);
    assert_eq!(Matrix3d::identity() * m, m);
    assert_eq!(m.determinant(), -3.0);
    assert_eq!(m.trace(), 16.0);
    assert!((m.inverse() * m).is_approx(Matrix3d::identity()));
    assert!((A3 * A3.inverse()).is_approx(Matrix3d::identity()));
    assert!((A3 * m)
        .transpose()
        .is_approx(m.transpose() * A3.transpose()));
    assert!((A3.determinant() * A3.inverse().determinant() - 1.0).abs() <= 1e-14);
    assert_eq!(
        Matrix3d::from_diagonal(Vector3d::new(1.0, 2.0, 3.0)).diagonal(),
        Vector3d::new(1.0, 2.0, 3.0)
    );
    assert_eq!(-m + m, Matrix3d::zeros());
    assert_eq!(m * 2.0 / 2.0, m);
    assert_eq!(2.0 * m, m + m);
    assert_eq!(Matrix3d::ones() - Matrix3d::identity(), {
        let mut off = Matrix3d::ones();
        for i in 0..3 {
            off[(i, i)] = 0.0;
        }
        off
    });
}

#[test]
fn rust_only_matrix2d_products_determinant_inverse() {
    let m = Matrix2d::new(4.0, 7.0, 2.0, 6.0);
    assert_eq!(m.determinant(), 10.0);
    assert_eq!(m.inverse(), Matrix2d::new(0.6, -0.7, -0.2, 0.4));
    assert!((m * m.inverse()).is_approx(Matrix2d::identity()));
    assert_eq!(m * Vector2d::new(1.0, 1.0), Vector2d::new(11.0, 8.0));
    assert_eq!(
        m.transpose(),
        Matrix2d::from_column_major([4.0, 7.0, 2.0, 6.0])
    );
    assert_eq!(Matrix2d::from_columns(m.col(0), m.col(1)), m);
    assert_eq!(Matrix2d::from_rows(m.row(0), m.row(1)), m);
    assert_eq!(m.trace(), 10.0);
}

#[test]
fn rust_only_jacobian_shapes_2x3_and_3x2() {
    let j = Matrix2x3d::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    assert_eq!(
        Matrix2x3d::from_row_major([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        j
    );
    assert_eq!(j.row(1), Vector3d::new(4.0, 5.0, 6.0));
    assert_eq!(j * Vector3d::new(1.0, 0.0, -1.0), Vector2d::new(-2.0, -2.0));
    let jt: Matrix3x2d = j.transpose();
    assert_eq!(jt, Matrix3x2d::new(1.0, 4.0, 2.0, 5.0, 3.0, 6.0));
    assert_eq!(jt.col(0), Vector3d::new(1.0, 2.0, 3.0));
    assert_eq!(Matrix3x2d::from_columns(jt.col(0), jt.col(1)), jt);
    assert_eq!(jt.transpose(), j);
    // J J^T = [14 32; 32 77].
    assert_eq!(j * jt, Matrix2d::new(14.0, 32.0, 32.0, 77.0));
    assert_eq!(Matrix3d::identity() * jt, jt);
    assert_eq!((jt / 2.0)[(2, 1)], 3.0);
    assert_eq!(j.norm(), 91f64.sqrt());
    assert!(jt.is_approx(jt * (1.0 + 1e-13)));
}

#[test]
fn rust_only_matrix4d_determinant_and_inverse() {
    // A diagonal-plus-permutation matrix with a known determinant: swapping rows of
    // diag(1, 2, 3, 4) once gives -24.
    let permuted = Matrix4d::new(
        0.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 0.0, 4.0,
    );
    assert_eq!(permuted.determinant(), -24.0);
    assert_eq!(Matrix4d::identity().determinant(), 1.0);
    assert!((A4.inverse() * A4).is_approx(Matrix4d::identity()));
    assert!((A4 * A4.inverse()).is_approx(Matrix4d::identity()));
    assert!((A4 * A4.transpose()).is_approx((A4 * A4.transpose()).transpose()));
    assert_eq!(A4 * Vector4d::new(1.0, 0.0, 0.0, 0.0), A4.col(0));
    assert_eq!(A4.transpose().row(2), A4.col(2));
    assert_eq!(
        Matrix4d::from_rows(A4.row(0), A4.row(1), A4.row(2), A4.row(3)),
        A4
    );
}

#[test]
fn rust_only_matrix4d_determinant_matches_cofactor_expansion() {
    // Independent check: expand along the first row with Matrix3d minors.
    let mut expected = 0.0;
    for j in 0..4 {
        let mut minor = [0.0; 9];
        let mut n = 0;
        for c in (0..4).filter(|&c| c != j) {
            for r in 1..4 {
                minor[n] = A4[(r, c)];
                n += 1;
            }
        }
        let sign = if j % 2 == 0 { 1.0 } else { -1.0 };
        expected += sign * A4[(0, j)] * Matrix3d::from_column_major(minor).determinant();
    }
    assert!((A4.determinant() - expected).abs() <= 1e-12);
}

#[test]
fn rust_only_matrix3x4d_blocks() {
    let t = Vector3d::new(1.0, 2.0, 3.0);
    let p = Matrix3x4d::from_blocks(A3, t);
    let x = Vector3d::new(-1.0, 0.5, 2.0);
    assert_eq!(p.left_cols3(), A3);
    assert_eq!(p.col(3), t);
    assert_eq!(p.row(0), Vector4d::new(2.0, -1.0, 0.5, 1.0));
    assert_eq!(p * x.homogeneous(), A3 * x + t);
    assert_eq!(Matrix3x4d::identity().left_cols3(), Matrix3d::identity());
    assert_eq!(Matrix3x4d::identity().col(3), Vector3d::zeros());
    assert_eq!(Matrix4d::from_top_rows(p).top_rows3(), p);
    assert_eq!(
        Matrix4d::from_top_rows(p).row(3),
        Vector4d::new(0.0, 0.0, 0.0, 1.0)
    );
    assert_eq!(Matrix4d::from_top_rows(p).top_left3x3(), A3);
    assert_eq!(p * Matrix4d::identity(), p);
    assert_eq!(Matrix3d::identity() * p, p);
    assert_eq!(
        Matrix3x4d::from_columns(p.col(0), p.col(1), p.col(2), p.col(3)),
        p
    );
    assert_eq!(p.transpose().transpose(), p);
    assert!((A3 * p).is_approx(Matrix3x4d::from_blocks(A3 * A3, A3 * t)));
    assert!(
        (p * Matrix4d::from_top_rows(p)).is_approx(Matrix3x4d::from_blocks(A3 * A3, A3 * t + t))
    );
}

#[test]
fn rust_only_is_approx_uses_frobenius_norm_relative_rule() {
    let scaled = A3 * (1.0 + 1e-13);
    assert!(A3.is_approx(scaled));
    assert!(!A3.is_approx(A3 * (1.0 + 1e-10)));
    assert!(Matrix3d::zeros().is_approx(Matrix3d::zeros()));
    assert!(!Matrix3d::zeros().is_approx(Matrix3d::identity() * 1e-100));
}

#[test]
fn rust_only_matrix6d_blocks_layout_and_product() {
    let b = A3.transpose();
    let m = Matrix6d::from_blocks(A3, Matrix3d::identity(), Matrix3d::zeros(), b);
    let column_major = m.as_slice();
    // [A, I; 0, B] * [A, I; 0, B] = [A*A, A + B; 0, B*B], all exact in these values.
    let square = m * m;
    let expected = Matrix6d::from_blocks(A3 * A3, A3 + b, Matrix3d::zeros(), b * b);
    assert_eq!(m.block3(0, 0), A3);
    assert_eq!(m.block3(0, 3), Matrix3d::identity());
    assert_eq!(m.block3(3, 0), Matrix3d::zeros());
    assert_eq!(m.block3(3, 3), b);
    assert_eq!(column_major[1], A3[(1, 0)]);
    assert_eq!(column_major[3 * 6], 1.0);
    assert_eq!(m.transpose().block3(3, 0), Matrix3d::identity());
    assert_eq!(square, expected);
    assert_eq!(Matrix6d::identity() * m, m);
    let perturbation = Matrix6d::from_blocks(
        Matrix3d::identity() * 1e-14,
        Matrix3d::zeros(),
        Matrix3d::zeros(),
        Matrix3d::zeros(),
    );
    assert!(m.is_approx(m + perturbation));
    assert_eq!(-m + m, Matrix6d::zeros());
}
