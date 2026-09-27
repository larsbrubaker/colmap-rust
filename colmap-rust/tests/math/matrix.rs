// Port of COLMAP's src/colmap/math/matrix_test.cc: the RQ decomposition of random 4x4
// matrices drawn from COLMAP's seeded PRNG.

use colmap_rust::linalg::MatrixXd;
use colmap_rust::math::matrix::decompose_matrix_rq;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::random_eigen_matrix4d;

#[test]
fn decompose_matrix_rq_nominal() {
    set_prng_seed(0);
    for _ in 0..10 {
        let a = MatrixXd::from(random_eigen_matrix4d());

        let (r, q) = decompose_matrix_rq(&a);

        assert!(r.bottom_rows(4).is_upper_triangular(1e-12));
        assert!(q.is_unitary(1e-12));
        let det = q.determinant();
        assert!((det - 1.0).abs() <= 1e-6, "det(Q) = {det}");
        // EigenMatrixNear(A, R * Q, 1e-6): Eigen's isApprox.
        assert!(a.is_approx_with(&(&r * &q), 1e-6));
    }
}
