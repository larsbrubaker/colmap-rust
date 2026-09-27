// Rust-only (COLMAP has no test for Eigen itself): Vector2d, Vector3d, Vector4d, Vector3f and
// Vector3ub against hand-computed values and Eigen's documented semantics (zero-vector
// normalized(), homogeneous/hnormalized, isApprox, coefficient-wise ==). Port of colmap-sharp's
// `ColmapSharp.Tests/LinearAlgebra/VectorTests.cs`. Values are exact where the arithmetic is
// exact, so they compare with `assert_eq!`.

use colmap_rust::linalg::{Vector2d, Vector3d, Vector3f, Vector3ub, Vector4d};

#[test]
fn rust_only_vector3d_dot_cross_norm() {
    let a = Vector3d::new(1.0, 2.0, 3.0);
    let b = Vector3d::new(4.0, -5.0, 6.0);
    assert_eq!(a.dot(b), 12.0);
    assert_eq!(a.cross(b), Vector3d::new(27.0, 6.0, -13.0));
    assert_eq!(
        Vector3d::unit_x().cross(Vector3d::unit_y()),
        Vector3d::unit_z()
    );
    assert_eq!(a.squared_norm(), 14.0);
    assert_eq!(Vector3d::new(2.0, 3.0, 6.0).norm(), 7.0);
    assert_eq!(a.cross(b).dot(a), 0.0);
}

#[test]
fn rust_only_normalized_zero_vector_stays_zero() {
    // Eigen's documented behavior: normalized() of a zero vector is the vector itself, not NaN.
    assert_eq!(Vector2d::zeros().normalized(), Vector2d::zeros());
    assert_eq!(Vector3d::zeros().normalized(), Vector3d::zeros());
    assert_eq!(Vector4d::zeros().normalized(), Vector4d::zeros());
    assert_eq!(
        Vector3d::new(0.0, 3.0, 4.0).normalized(),
        Vector3d::new(0.0, 0.6, 0.8)
    );
    assert_eq!(
        Vector2d::new(3.0, 4.0).normalized(),
        Vector2d::new(0.6, 0.8)
    );
}

#[test]
fn rust_only_homogeneous_and_hnormalized() {
    assert_eq!(
        Vector2d::new(3.0, 4.0).homogeneous(),
        Vector3d::new(3.0, 4.0, 1.0)
    );
    assert_eq!(
        Vector3d::new(3.0, 4.0, 5.0).homogeneous(),
        Vector4d::new(3.0, 4.0, 5.0, 1.0)
    );
    assert_eq!(
        Vector3d::new(4.0, 6.0, 2.0).hnormalized(),
        Vector2d::new(2.0, 3.0)
    );
    assert_eq!(
        Vector4d::new(4.0, 6.0, 8.0, 2.0).hnormalized(),
        Vector3d::new(2.0, 3.0, 4.0)
    );
    assert_eq!(
        Vector3d::new(4.0, 6.0, 2.0).head2(),
        Vector2d::new(4.0, 6.0)
    );
    assert_eq!(
        Vector4d::new(4.0, 6.0, 8.0, 2.0).head3(),
        Vector3d::new(4.0, 6.0, 8.0)
    );
}

#[test]
fn rust_only_operators_and_indexer() {
    let a = Vector4d::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(a + a, a * 2.0);
    assert_eq!(2.0 * a - a, a);
    assert_eq!(-a / 2.0, Vector4d::new(-0.5, -1.0, -1.5, -2.0));
    assert_eq!(a[3], 4.0);
    assert_eq!(a.dot(a), 30.0);
    assert_eq!(Vector3d::new(1.0, 2.0, 3.0)[1], 2.0);
    assert_eq!(
        Vector3d::from_array([1.0, 2.0, 3.0]).to_array(),
        [1.0, 2.0, 3.0]
    );
    assert_eq!(
        Vector3d::new(-1.0, 2.0, -3.0).cwise_abs(),
        Vector3d::new(1.0, 2.0, 3.0)
    );
    assert_eq!(
        Vector2d::new(2.0, 3.0).cwise_product(Vector2d::new(4.0, -1.0)),
        Vector2d::new(8.0, -3.0)
    );

    let mut b = Vector3d::zeros();
    b[2] = 5.0;
    b += Vector3d::ones();
    b *= 2.0;
    assert_eq!(b, Vector3d::new(2.0, 2.0, 12.0));
}

#[test]
#[should_panic(expected = "out of range")]
fn rust_only_indexer_out_of_range_panics() {
    let _ = Vector3d::new(1.0, 2.0, 3.0)[3];
}

#[test]
fn rust_only_equality_follows_eigen_operator_equals() {
    // == is Eigen's coefficient-wise ==, so NaN is never equal and -0 equals +0.
    let nan = Vector3d::new(f64::NAN, 0.0, 0.0);
    assert!(nan != nan);
    assert_eq!(Vector3d::new(0.0, 0.0, 0.0), Vector3d::new(-0.0, 0.0, 0.0));
}

#[test]
fn rust_only_is_approx_is_relative() {
    let a = Vector3d::new(1.0, 2.0, 3.0);
    assert!(a.is_approx(a * (1.0 + 1e-13)));
    assert!(!a.is_approx(a * (1.0 + 1e-10)));
    assert!(a.is_approx_with(a * (1.0 + 1e-10), 1e-9));
    // Relative: only an exact zero is approximately zero, as in Eigen.
    assert!(!Vector3d::zeros().is_approx(Vector3d::new(1e-100, 0.0, 0.0)));
    assert!(Vector3d::zeros().is_approx(Vector3d::zeros()));
}

#[test]
fn rust_only_vector4d_reduction_is_paired_like_eigen() {
    // Lanes {0, 2} and {1, 3} are summed first, as Eigen's 2-lane packet reduction does
    // (rust_only_rotation_oracle pins this bit for bit through quaternion norms). Here
    // (1 - 1) + (t^2 + t^2) = 2 t^2, while left to right 1 + t^2 rounds back to 1 and the sum
    // would come out as t^2.
    let tiny = 2f64.powi(-27);
    let a = Vector4d::new(1.0, tiny, 1.0, tiny);
    let b = Vector4d::new(1.0, tiny, -1.0, tiny);
    assert_eq!(a.dot(b), 2.0 * tiny * tiny);
}

#[test]
fn rust_only_float_and_byte_vectors() {
    assert_eq!(
        Vector3f::new(1.0, 2.0, 3.0),
        Vector3f {
            x: 1.0,
            y: 2.0,
            z: 3.0
        }
    );
    assert_eq!(Vector3ub::zeros(), Vector3ub::default());
    let color = Vector3ub::new(255, 128, 0);
    assert_eq!((color.x, color.y, color.z), (255, 128, 0));
}
