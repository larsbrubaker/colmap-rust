// Rust-only tests of colmap_rust::feature::vlfeat::linear (VLFeat mathop.c's vl_svd2,
// vl_lapack_dlasv2, vl_gaussian_elimination and vl_solve_linear_system_2/3), which have no COLMAP
// test of their own: the covariant detector's oracle fixture pins them end to end. These pin
// VLFeat's documented behaviour: the factorization reconstructs its input, singular values are
// signed with the largest modulus first, the branches (diagonal, swapped, g very large) give
// VLFeat's exact outputs, and a singular elimination is VL_ERR_OVERFLOW.

use colmap_rust::feature::vlfeat::linear::{
    vl_gaussian_elimination, vl_lapack_dlasv2, vl_solve_linear_system_2, vl_solve_linear_system_3,
    vl_svd2, SingularSystem,
};

/// Column-major 2x2 product.
fn mul2(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    [
        a[0] * b[0] + a[2] * b[1],
        a[1] * b[0] + a[3] * b[1],
        a[0] * b[2] + a[2] * b[3],
        a[1] * b[2] + a[3] * b[3],
    ]
}

fn transpose2(a: &[f64; 4]) -> [f64; 4] {
    [a[0], a[2], a[1], a[3]]
}

fn assert_near(actual: &[f64], expected: &[f64], tol: f64) {
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= tol, "{actual:?} vs {expected:?}");
    }
}

#[test]
fn rust_only_svd2_reconstructs_the_matrix() {
    for m in [
        [3.0, -1.5, 0.25, 2.0],
        [0.1, 4.0, -2.0, 0.3],
        [1.0, 2.0, 2.0, 4.0],
        [-5.0, 0.0, 1e-3, -7.0],
    ] {
        let (s, u, v) = vl_svd2(&m);
        assert_eq!(s[1], 0.0);
        assert_eq!(s[2], 0.0);
        assert!(s[0].abs() >= s[3].abs());
        assert_near(&mul2(&mul2(&u, &s), &transpose2(&v)), &m, 1e-12);
        assert_near(&mul2(&transpose2(&u), &u), &[1.0, 0.0, 0.0, 1.0], 1e-12);
        assert_near(&mul2(&transpose2(&v), &v), &[1.0, 0.0, 0.0, 1.0], 1e-12);
    }
}

#[test]
fn rust_only_dlasv2_diagonal_branch_is_exact() {
    // ga == 0: the singular values are the diagonal, signed by f and h.
    let r = vl_lapack_dlasv2(3.0, 0.0, -2.0);
    assert_eq!((r.smax, r.smin), (3.0, -2.0));
    assert_eq!((r.cu, r.su, r.cv, r.sv), (1.0, 0.0, 1.0, 0.0));

    // fa < ha swaps the roles of f and h (and of the left and right vectors).
    let r = vl_lapack_dlasv2(2.0, 0.0, 5.0);
    assert_eq!((r.smax, r.smin), (5.0, 2.0));
    assert_eq!((r.cu, r.su, r.cv, r.sv), (0.0, 1.0, 0.0, 1.0));
}

#[test]
fn rust_only_dlasv2_very_large_g_branch() {
    // fa / ga below VL_EPSILON_D: smax = |g|, smin = (fa / ga) * ha.
    let (f, g, h) = (1.0, 1e20, 0.5);
    let r = vl_lapack_dlasv2(f, g, h);
    assert_eq!(r.smax, 1e20);
    assert_eq!(r.smin, (1.0 / 1e20) * 0.5);
    assert_eq!((r.cu, r.su, r.cv, r.sv), (1.0, h / g, 1.0, f / g));
}

#[test]
fn rust_only_solve_linear_system_3() {
    // Column-major A = [4 1 0; 1 3 1; 0 1 2]; b = A * [1, -2, 3].
    let a = [4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let b = [2.0, -2.0, 4.0];
    let mut x = [0.0; 3];
    assert_eq!(vl_solve_linear_system_3(&mut x, &a, &b), Ok(()));
    assert_near(&x, &[1.0, -2.0, 3.0], 1e-14);
}

#[test]
fn rust_only_solve_linear_system_2_pivots() {
    // A zero leading entry needs the row swap. Column-major A = [0 2; 3 1], b = A * [-1, 4].
    let a = [0.0, 3.0, 2.0, 1.0];
    let b = [8.0, 1.0];
    let mut x = [0.0; 2];
    assert_eq!(vl_solve_linear_system_2(&mut x, &a, &b), Ok(()));
    assert_near(&x, &[-1.0, 4.0], 1e-15);
}

#[test]
fn rust_only_gaussian_elimination_singular_is_overflow() {
    // Column-major [A, b] with dependent rows: the second pivot vanishes.
    let mut m = [1.0, 2.0, 2.0, 4.0, 3.0, 6.0];
    assert_eq!(vl_gaussian_elimination(&mut m, 2, 3), Err(SingularSystem));

    // An all-zero first column fails at once and leaves the matrix untouched.
    let mut m = [0.0, 0.0, 1.0, 2.0, 3.0, 4.0];
    assert_eq!(vl_gaussian_elimination(&mut m, 2, 3), Err(SingularSystem));
    assert_eq!(m, [0.0, 0.0, 1.0, 2.0, 3.0, 4.0]);

    let mut x = [9.0; 2];
    let singular = [1e-11, 0.0, 0.0, 1.0];
    assert_eq!(
        vl_solve_linear_system_2(&mut x, &singular, &[1.0, 1.0]),
        Err(SingularSystem)
    );
}
