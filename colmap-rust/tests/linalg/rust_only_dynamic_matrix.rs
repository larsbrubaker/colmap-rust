// Rust-only (COLMAP has no test for Eigen itself): MatrixXd, VectorXd and RowMajorMatrix. Port
// of colmap-sharp's `ColmapSharp.Tests/LinearAlgebra/DynamicMatrixTests.cs`. Covers the layout
// contract (column-major storage, row-major `from_row_major` like Eigen's comma initializer),
// block/row/column access and the slice views, products against hand-computed values, the
// transposed products against explicit transposes (bit-identical: same terms in the same
// order), and the round trip to the fixed-size types. The RowMajorMatrix case is Rust-only
// here (colmap-sharp covers it through its feature-type tests).

use colmap_rust::linalg::{
    Matrix2d, Matrix3d, Matrix3x4d, Matrix4d, MatrixXd, RowMajorMatrix, Vector2d, Vector3d,
    Vector4d, VectorXd,
};
use std::panic::catch_unwind;

fn a23() -> MatrixXd {
    MatrixXd::from_row_major(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

#[test]
fn rust_only_layout_column_major_storage_and_row_major_initializer() {
    let a = a23();
    assert_eq!(a.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    assert_eq!(a[(1, 2)], 6.0);
    assert_eq!(a.row(1).as_slice(), &[4.0, 5.0, 6.0]);
    assert_eq!(a.col(1).as_slice(), &[2.0, 5.0]);
    assert_eq!(a.transpose().as_slice(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert!(catch_unwind(|| a23()[(2, 0)]).is_err());
    assert!(catch_unwind(|| a23()[(0, 3)]).is_err());
}

#[test]
fn rust_only_blocks_copy_and_write_back() {
    let mut m = MatrixXd::from_row_major(
        3,
        4,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
    );
    let mut block = m.block(1, 1, 2, 2);
    block[(0, 0)] = -1.0;
    let mut target = MatrixXd::zeros(3, 4);
    target.set_block(1, 2, &block);
    m.column_mut(3)[0] = 40.0;
    assert_eq!(block.as_slice(), &[-1.0, 10.0, 7.0, 11.0]);
    assert_eq!(m[(1, 1)], 6.0);
    assert_eq!(m[(0, 3)], 40.0);
    assert_eq!(target[(1, 2)], -1.0);
    assert_eq!(target[(2, 3)], 11.0);
    assert_eq!(target[(0, 0)], 0.0);
    assert_eq!(m.left_cols(1).as_slice(), &[1.0, 5.0, 9.0]);
    assert_eq!(m.bottom_rows(1).as_slice(), &[9.0, 10.0, 11.0, 12.0]);
    assert_eq!(m.reverse_rows()[(0, 0)], 9.0);
    assert_eq!(m.reverse_cols()[(0, 0)], 40.0);
    let m2 = m.clone();
    assert!(catch_unwind(move || m2.block(2, 0, 2, 1)).is_err());
}

#[test]
fn rust_only_products_match_hand_computed_values() {
    let a = a23();
    let b = MatrixXd::from_row_major(3, 2, &[1.0, 0.0, 0.0, 1.0, 2.0, -1.0]);
    let ab = &a * &b;
    let av = &a * &VectorXd::from_slice(&[1.0, 1.0, 1.0]);
    assert_eq!(ab.as_slice(), &[7.0, 16.0, -1.0, -1.0]);
    assert_eq!(av.as_slice(), &[6.0, 15.0]);
    assert_eq!((&a + &a).as_slice(), (&a * 2.0).as_slice());
    assert_eq!((&a - &a).norm(), 0.0);
    assert_eq!(a.squared_norm(), 91.0);
    assert_eq!(MatrixXd::identity(3).trace(), 3.0);
    assert!(catch_unwind(|| &a23() * &a23()).is_err());
}

#[test]
fn rust_only_transposed_products_equal_explicit_transpose() {
    let a = MatrixXd::from_row_major(
        4,
        3,
        &[
            0.3, -1.2, 2.5, 1.1, 0.7, -0.4, -2.2, 0.9, 1.6, 0.05, -0.8, 3.1,
        ],
    );
    let b = MatrixXd::from_row_major(4, 2, &[1.5, -0.3, 0.2, 2.2, -1.1, 0.6, 0.9, 0.4]);
    let v = VectorXd::from_slice(&[0.5, -1.5, 2.0, 0.25]);
    let gram = a.transpose_times_self();
    assert_eq!(
        a.transpose_times(&b).as_slice(),
        (&a.transpose() * &b).as_slice()
    );
    assert_eq!(gram.as_slice(), (&a.transpose() * &a).as_slice());
    assert_eq!(gram.as_slice(), gram.transpose().as_slice());
    assert_eq!(
        a.transpose_times_vector(&v).as_slice(),
        (&a.transpose() * &v).as_slice()
    );
}

#[test]
fn rust_only_fixed_size_conversions_round_trip() {
    let m3 = Matrix3d::new(2.0, -1.0, 0.5, 3.0, 4.0, -2.0, 1.0, 0.25, 5.0);
    let m34 = Matrix3x4d::new(
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    );
    let m4 = Matrix4d::identity();
    let m2 = Matrix2d::new(1.0, 2.0, 3.0, 4.0);
    let x3 = MatrixXd::from(m3);
    assert_eq!(x3[(0, 2)], 0.5);
    assert_eq!(x3.to_matrix3d(), m3);
    assert_eq!(MatrixXd::from(m34).to_matrix3x4d(), m34);
    assert_eq!(MatrixXd::from(m4).to_matrix4d(), m4);
    assert_eq!(MatrixXd::from(m2).to_matrix2d(), m2);
    assert_eq!(MatrixXd::from(m34)[(2, 3)], 12.0);
    let v3 = Vector3d::new(1.0, 2.0, 3.0);
    assert_eq!(VectorXd::from(v3).to_vector3d(), v3);
    let v2 = Vector2d::new(1.0, 2.0);
    assert_eq!(VectorXd::from(v2).to_vector2d(), v2);
    let v4 = Vector4d::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(VectorXd::from(v4).to_vector4d(), v4);
    let x3_copy = x3.clone();
    assert!(catch_unwind(move || x3_copy.to_matrix4d()).is_err());
    assert!((m3 * m3).is_approx((&x3 * &x3).to_matrix3d()));
}

#[test]
fn rust_only_vector_xd_norms_and_normalized() {
    let v = VectorXd::from_slice(&[3.0, 4.0]);
    assert_eq!(v.norm(), 5.0);
    assert_eq!(v.normalized().as_slice(), &[0.6, 0.8]);
    assert_eq!(VectorXd::zeros(3).normalized().norm(), 0.0);
    assert_eq!(v.dot(&VectorXd::from_slice(&[1.0, -1.0])), -1.0);
    assert_eq!(VectorXd::unit(3, 1)[1], 1.0);
    assert_eq!(
        VectorXd::from_slice(&[1.0, 2.0, 3.0, 4.0])
            .segment(1, 2)
            .as_slice(),
        &[2.0, 3.0]
    );
    assert_eq!(VectorXd::from_slice(&[-7.0, 2.0]).max_abs(), 7.0);
}

#[test]
fn rust_only_row_major_matrix_layout_and_equality() {
    let mut m: RowMajorMatrix<u8> = RowMajorMatrix::new(2, 3);
    assert_eq!(m.as_slice(), &[0; 6]);
    m[(1, 2)] = 7;
    m.row_mut(0)[1] = 3;
    assert_eq!(m.as_slice(), &[0, 3, 0, 0, 0, 7]);
    assert_eq!(m.row(1), &[0, 0, 7]);
    let n = RowMajorMatrix::from_vec(2, 3, vec![0, 3, 0, 0, 0, 7]);
    assert_eq!(m, n);
    assert_ne!(m, RowMajorMatrix::from_vec(3, 2, vec![0, 3, 0, 0, 0, 7]));
    assert!(catch_unwind(|| RowMajorMatrix::from_vec(2, 2, vec![1u32; 3])).is_err());
    assert!(catch_unwind(|| RowMajorMatrix::<f32>::new(1, 1)[(1, 0)]).is_err());
}
