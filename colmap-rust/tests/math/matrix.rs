// Port of COLMAP's src/colmap/math/matrix_test.cc (1:1, same test names in snake_case).
// Tests colmap_rust::math::matrix::decompose_matrix_rq_4d (DecomposeMatrixRQ with
// Eigen::Matrix4d). Tier B (goes through Householder QR): the tolerances are COLMAP's;
// isUpperTriangular/isUnitary use Eigen's default precision (linalg::DUMMY_PRECISION).

use colmap_rust::linalg::{Matrix4d, MatrixXd, DUMMY_PRECISION};
use colmap_rust::math::matrix::decompose_matrix_rq_4d;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::random_eigen_matrixd;

#[test]
fn decompose_matrix_rq_nominal() {
    // gtest_main reseeds the PRNG with 0 before every test (see random.rs).
    set_prng_seed(0);
    for _ in 0..10 {
        let a: Matrix4d = random_eigen_matrixd();

        let (r, q) = decompose_matrix_rq_4d(&a);

        assert!(MatrixXd::from(r)
            .bottom_rows(4)
            .is_upper_triangular(DUMMY_PRECISION));
        assert!(MatrixXd::from(q).is_unitary(DUMMY_PRECISION));
        assert!((q.determinant() - 1.0).abs() <= 1e-6);
        assert!(a.is_approx_with(r * q, 1e-6), "{a:?} vs {:?}", r * q);
    }
}
