// Rust-only tests (no COLMAP counterpart: PoseLib's misc/sturm.h has no test in COLMAP or in
// colmap-sharp) of colmap_rust::estimators::solvers::poselib::sturm. Tier B: roots within
// the solver's tolerance of polynomials with known roots, in ascending order, and the
// characteristic polynomial of matrices with known eigenvalues.

use colmap_rust::estimators::solvers::poselib::sturm::{
    bisect_sturm, charpoly_danilevsky_piv, DEFAULT_TOL,
};

/// The coefficients (constant first) of `lead * prod (x - r)`.
fn poly_from_roots(roots: &[f64], lead: f64) -> Vec<f64> {
    let mut coeffs = vec![lead];
    for &r in roots {
        let mut next = vec![0.0; coeffs.len() + 1];
        for (i, &c) in coeffs.iter().enumerate() {
            next[i + 1] += c;
            next[i] -= r * c;
        }
        coeffs = next;
    }
    coeffs
}

fn assert_roots(found: &[f64], expected: &[f64], tol: f64) {
    assert_eq!(found.len(), expected.len(), "{found:?} vs {expected:?}");
    for (f, e) in found.iter().zip(expected) {
        assert!((f - e).abs() <= tol, "{found:?} vs {expected:?}");
    }
}

#[test]
fn rust_only_sturm_finds_cubic_roots_in_ascending_order() {
    // (x - 1)(x - 2)(x - 3) = x^3 - 6x^2 + 11x - 6.
    let coeffs = [-6.0, 11.0, -6.0, 1.0];
    let mut roots = [0.0; 3];
    let n = bisect_sturm(3, &coeffs, &mut roots, DEFAULT_TOL);
    assert_roots(&roots[..n], &[1.0, 2.0, 3.0], 1e-9);
}

#[test]
fn rust_only_sturm_normalizes_the_leading_coefficient() {
    let expected = [-2.5, -0.25, 0.75, 4.0];
    let coeffs = poly_from_roots(&expected, -3.5);
    let mut roots = [0.0; 4];
    let n = bisect_sturm(4, &coeffs, &mut roots, DEFAULT_TOL);
    assert_roots(&roots[..n], &expected, 1e-9);
}

#[test]
fn rust_only_sturm_degree_ten_mixed_real_and_complex_roots() {
    // Six real roots times (x^2 + 1)(x^2 + 2x + 5), which has none: only the six are found.
    let real = [-3.0, -1.5, -0.5, 0.2, 1.0, 2.5];
    let base = poly_from_roots(&real, 1.0);
    let mut coeffs = base.clone();
    for quad in [[1.0, 0.0, 1.0], [5.0, 2.0, 1.0]] {
        let mut next = vec![0.0; coeffs.len() + 2];
        for (i, &c) in coeffs.iter().enumerate() {
            for (j, &q) in quad.iter().enumerate() {
                next[i + j] += c * q;
            }
        }
        coeffs = next;
    }
    assert_eq!(coeffs.len(), 11);
    let mut roots = [0.0; 10];
    let n = bisect_sturm(10, &coeffs, &mut roots, DEFAULT_TOL);
    assert_roots(&roots[..n], &real, 1e-8);
}

#[test]
fn rust_only_sturm_no_real_roots() {
    // x^4 + 1.
    let coeffs = [1.0, 0.0, 0.0, 0.0, 1.0];
    let mut roots = [0.0; 4];
    assert_eq!(bisect_sturm(4, &coeffs, &mut roots, DEFAULT_TOL), 0);
}

#[test]
fn rust_only_sturm_more_sign_changes_at_upper_bound() {
    // A degree-8 polynomial from P4Pf (AbsolutePose.P4PFSharedFocalLength) whose Sturm
    // sequence, rounded, has fewer sign changes at -r0 than at r0. PoseLib's int count goes
    // negative and isolates nothing; the unsigned counts here must give the same 0 roots
    // instead of underflowing.
    let coeffs = [
        3.6445373821210745e-28,
        -4.849705123623497e-15,
        0.0018912384681492324,
        -0.0027707026293560272,
        -0.002333305695706817,
        0.00774795666155109,
        -0.002361603480685079,
        -0.0018956211953063441,
        0.0007039400206650948,
    ];
    let mut roots = [0.0; 8];
    assert_eq!(bisect_sturm(8, &coeffs, &mut roots, 1e-10), 0);
}

#[test]
fn rust_only_sturm_degenerate_degrees() {
    let mut roots = [0.0; 3];
    // Degree 0 has no roots.
    assert_eq!(bisect_sturm(0, &[5.0], &mut roots, DEFAULT_TOL), 0);
    // Degree 1 is solved directly, unless it is constant.
    assert_eq!(bisect_sturm(1, &[3.0, 2.0], &mut roots, DEFAULT_TOL), 1);
    assert_eq!(roots[0], -1.5);
    assert_eq!(bisect_sturm(1, &[3.0, 0.0], &mut roots, DEFAULT_TOL), 0);
    // An exactly zero leading coefficient returns no roots, even when lower terms have some.
    assert_eq!(
        bisect_sturm(3, &[-1.0, 0.0, 1.0, 0.0], &mut roots, DEFAULT_TOL),
        0
    );
}

/// The `n x n` column-major product `a * b`.
fn matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut m = vec![0.0; n * n];
    for r in 0..n {
        for c in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += a[r + n * k] * b[k + n * c];
            }
            m[r + n * c] = sum;
        }
    }
    m
}

#[test]
fn rust_only_charpoly_danilevsky_of_tridiagonal_matrix() {
    // [[2, 1, 0], [1, 2, 1], [0, 1, 2]] (symmetric, so column-major reads the same): trace 6,
    // principal 2x2 minors 3 + 4 + 3 = 10, determinant 4, so the monic characteristic
    // polynomial is x^3 - 6x^2 + 10x - 4. (Danilevsky's method needs nonzero subdiagonal
    // pivots, which PoseLib's dense action matrices have; a triangular matrix has none.)
    let mut a = [2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0];
    let mut p = [0.0; 4];
    charpoly_danilevsky_piv(&mut a, 3, &mut p);
    let expected = [-4.0, 10.0, -6.0, 1.0];
    for (got, want) in p.iter().zip(&expected) {
        assert!((got - want).abs() <= 1e-12, "{p:?}");
    }
}

#[test]
fn rust_only_charpoly_danilevsky_feeds_bisect_sturm() {
    // A dense 5x5 matrix with known eigenvalues, M = S D S^-1 with S = L U (L unit lower and
    // U unit upper triangular, all ones), whose inverse U^-1 L^-1 is bidiagonal and exact.
    // The roots of its characteristic polynomial are the eigenvalues, the path the
    // shared-focal solver takes.
    let eig = [-2.0, -0.5, 1.0, 1.5, 3.0];
    let n = 5;
    let mut l = vec![0.0; n * n];
    let mut u = vec![0.0; n * n];
    let mut l_inv = vec![0.0; n * n];
    let mut u_inv = vec![0.0; n * n];
    let mut d = vec![0.0; n * n];
    for c in 0..n {
        for r in 0..n {
            if r >= c {
                l[r + n * c] = 1.0;
            }
            if r <= c {
                u[r + n * c] = 1.0;
            }
        }
        l_inv[c + n * c] = 1.0;
        u_inv[c + n * c] = 1.0;
        if c + 1 < n {
            l_inv[(c + 1) + n * c] = -1.0;
            u_inv[c + n * (c + 1)] = -1.0;
        }
        d[c + n * c] = eig[c];
    }
    let s = matmul(&l, &u, n);
    let s_inv = matmul(&u_inv, &l_inv, n);
    let identity = matmul(&s, &s_inv, n);
    for r in 0..n {
        for c in 0..n {
            assert_eq!(identity[r + n * c], if r == c { 1.0 } else { 0.0 });
        }
    }
    let mut m = matmul(&matmul(&s, &d, n), &s_inv, n);
    let mut p = [0.0; 6];
    charpoly_danilevsky_piv(&mut m, n, &mut p);
    let mut roots = [0.0; 5];
    let found = bisect_sturm(n, &p, &mut roots, DEFAULT_TOL);
    assert_roots(&roots[..found], &eig, 1e-9);
}
