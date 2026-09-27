// Rust-only (COLMAP has no test for Eigen itself): JacobiSvd, Svd3d/Svd4d,
// SelfAdjointEigenSolver, EigenSolver and FullPivLu::rank against numpy (LAPACK) on random,
// rank-deficient and repeated-value matrices, in the shapes COLMAP decomposes. Port of
// colmap-sharp's `ColmapSharp.Tests/LinearAlgebra/SpectralOracleTests.cs`, on the same fixture
// (`tests/data/oracle/linear_algebra_spectral.json`, written by
// `oracle/linear_algebra_spectral.py`).
//
// Tier B. Tolerances (entries O(1), small matrices):
// - Singular values and symmetric eigenvalues: 1e-12 absolute (times the largest value).
// - Reconstructions U S V^T and V D V^T, orthogonality of U and V: 1e-12 absolute.
// - Singular vectors / eigenvectors: compared up to sign (up to a complex phase for the
//   general solver) and only for simple values (relative gap to the neighbors > 1e-3), within
//   1e-9; a vector's error is ~eps / gap. Null-space columns (index >= rank) are checked by
//   ||A v|| <= 1e-12 instead, since any orthonormal null basis is correct.
// - Rank: exact. Minimum-norm solve: 1e-10 against numpy's pinv with the same cutoff.
// - General eigenvalues: matched as a multiset within 1e-9 (1e-7 for the defective
//   Jordan-block case, whose eigenvalue error is ~sqrt(eps)); every eigenvector satisfies
//   ||A v - lambda v|| <= 1e-9 (1e-7 defective).
// - Svd3d/Svd4d run the same kernel on the same buffers: bit-identical to JacobiSvd.

use crate::oracle_json::Json;
use colmap_rust::linalg::{
    Complex, ComputationInfo, EigenSolver, FullPivLu, JacobiSvd, MatrixXd, SelfAdjointEigenSolver,
    Svd3d, Svd4d, SvdOptions, VectorXd,
};

const FIXTURE: &str = include_str!("../data/oracle/linear_algebra_spectral.json");

fn fixture() -> Json {
    Json::parse(FIXTURE)
}

fn usize_of(c: &Json, name: &str) -> usize {
    c.get(name).as_f64() as usize
}

fn is_simple(values: &[f64], i: usize) -> bool {
    let scale = values[0].abs().max(values[values.len() - 1].abs());
    (0..values.len()).all(|j| j == i || (values[j] - values[i]).abs() > 1e-3 * scale)
}

fn max_abs(m: &MatrixXd) -> f64 {
    m.as_slice().iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

fn assert_column_up_to_sign(
    label: &str,
    actual: &MatrixXd,
    expected: &MatrixXd,
    column: usize,
    tolerance: f64,
) {
    let a = actual.col(column);
    let e = expected.col(column);
    let sign = if a.dot(&e) >= 0.0 { 1.0 } else { -1.0 };
    let error = (&(&a * sign) - &e).max_abs();
    assert!(error <= tolerance, "{label} column {column}: {error}");
}

#[test]
fn rust_only_jacobi_svd_matches_numpy() {
    let json = fixture();
    for c in json.get("svd").as_array() {
        let name = c.get("name").as_str();
        let rows = usize_of(c, "rows");
        let cols = usize_of(c, "cols");
        let a = MatrixXd::from_column_major(rows, cols, &c.get("a").as_f64s());
        let svd = JacobiSvd::new(&a, SvdOptions::FULL_UV);
        let s = c.get("s").as_f64s();
        let scale = s[0].max(1.0);
        let actual_s = svd.singular_values();
        assert_eq!(svd.info(), ComputationInfo::Success, "{name}");
        assert_eq!(actual_s.len(), s.len(), "{name}");
        for i in 0..s.len() {
            assert!((actual_s[i] - s[i]).abs() <= 1e-12 * scale, "{name} s[{i}]");
            if i > 0 {
                assert!(actual_s[i] <= actual_s[i - 1], "{name} s[{i}] order");
            }
        }

        let rank = usize_of(c, "rank");
        assert_eq!(svd.rank(), rank, "{name}");

        let u = svd.matrix_u();
        let v = svd.matrix_v();
        assert_eq!(u.rows(), rows);
        assert_eq!(u.cols(), rows);
        assert_eq!(v.rows(), cols);
        assert_eq!(v.cols(), cols);
        let u_orthogonality = max_abs(&(&(&u.transpose() * &u) - &MatrixXd::identity(rows)));
        assert!(u_orthogonality <= 1e-12, "{name} U orthogonal");
        let v_orthogonality = max_abs(&(&(&v.transpose() * &v) - &MatrixXd::identity(cols)));
        assert!(v_orthogonality <= 1e-12, "{name} V orthogonal");
        let mut sigma = MatrixXd::zeros(rows, cols);
        for i in 0..s.len() {
            sigma[(i, i)] = actual_s[i];
        }
        let reconstruction = max_abs(&(&(&(&u * &sigma) * &v.transpose()) - &a));
        assert!(reconstruction <= 1e-12 * scale, "{name} reconstruction");

        let expected_u = MatrixXd::from_column_major(rows, rows, &c.get("u").as_f64s());
        let expected_v = MatrixXd::from_column_major(cols, cols, &c.get("v").as_f64s());
        for i in 0..rank {
            if is_simple(&s, i) {
                assert_column_up_to_sign(&format!("{name} U"), &u, &expected_u, i, 1e-9);
                assert_column_up_to_sign(&format!("{name} V"), &v, &expected_v, i, 1e-9);
            }
        }

        // Null space: the columns COLMAP reads with matrixV().rightCols<k>().
        for i in rank..cols {
            let residual = (&a * &v.col(i)).norm();
            assert!(residual <= 1e-12 * scale, "{name} A v[{i}]");
        }

        let x = svd.solve(&VectorXd::from_vec(c.get("b").as_f64s()));
        let expected_x = VectorXd::from_vec(c.get("x").as_f64s());
        assert!((&x - &expected_x).max_abs() <= 1e-10, "{name} solve");

        if rows == 3 && cols == 3 {
            let fixed3 = Svd3d::compute(&a.to_matrix3d());
            assert_eq!(fixed3.matrix_u, u.to_matrix3d(), "{name}");
            assert_eq!(fixed3.matrix_v, v.to_matrix3d(), "{name}");
            assert_eq!(fixed3.singular_values, actual_s.to_vector3d(), "{name}");
            assert_eq!(fixed3.rank(), svd.rank(), "{name}");
        } else if rows == 4 && cols == 4 {
            let fixed4 = Svd4d::compute(&a.to_matrix4d());
            assert_eq!(fixed4.matrix_u, u.to_matrix4d(), "{name}");
            assert_eq!(fixed4.matrix_v, v.to_matrix4d(), "{name}");
            assert_eq!(fixed4.rank(), rank, "{name}");
        }
    }
}

#[test]
fn rust_only_self_adjoint_eigen_solver_matches_numpy() {
    let json = fixture();
    for c in json.get("symmetric").as_array() {
        let n = usize_of(c, "n");
        let a = MatrixXd::from_column_major(n, n, &c.get("a").as_f64s());
        let solver = SelfAdjointEigenSolver::new(&a, true);
        let values = c.get("values").as_f64s();
        let expected = MatrixXd::from_column_major(n, n, &c.get("vectors").as_f64s());
        let actual = solver.eigenvalues();
        let vectors = solver.eigenvectors();
        let label = format!("n={n}");
        assert_eq!(solver.info(), ComputationInfo::Success, "{label}");
        for i in 0..n {
            assert!(
                (actual[i] - values[i]).abs() <= 1e-12 * 4.0,
                "{label} lambda[{i}]"
            );
            if is_simple(&values, i) {
                assert_column_up_to_sign(&label, &vectors, &expected, i, 1e-9);
            }
        }

        let orthogonality = max_abs(&(&(&vectors.transpose() * &vectors) - &MatrixXd::identity(n)));
        assert!(orthogonality <= 1e-12, "{label}");
        let reconstruction = max_abs(
            &(&(&(&vectors * &MatrixXd::from_diagonal(&actual)) * &vectors.transpose()) - &a),
        );
        assert!(reconstruction <= 1e-12 * 4.0, "{label}");
    }
}

#[test]
fn rust_only_eigen_solver_matches_numpy() {
    let json = fixture();
    for c in json.get("general").as_array() {
        let n = usize_of(c, "n");
        let a = MatrixXd::from_column_major(n, n, &c.get("a").as_f64s());
        let re = c.get("values_re").as_f64s();
        let im = c.get("values_im").as_f64s();
        let defective = (0..n).any(|i| (0..n).any(|j| j != i && re[i] == re[j] && im[i] == im[j]));
        let tolerance = if defective { 1e-7 } else { 1e-9 };
        let solver = EigenSolver::new(&a, true);
        let values = solver.eigenvalues();
        let vectors = solver.eigenvectors();
        let label = format!("n={n} a[0]={:?}", a[(0, 0)]);
        assert_eq!(solver.info(), ComputationInfo::Success, "{label}");

        // Greedy multiset match, first-closest like colmap-sharp's stable OrderBy().First().
        let mut unused: Vec<usize> = (0..n).collect();
        for i in 0..n {
            let expected = Complex::new(re[i], im[i]);
            let mut best_slot = 0;
            for slot in 1..unused.len() {
                if (values[unused[slot]] - expected).abs()
                    < (values[unused[best_slot]] - expected).abs()
                {
                    best_slot = slot;
                }
            }
            let best = unused[best_slot];
            assert!(
                (values[best] - expected).abs() <= tolerance,
                "{label} lambda {expected}"
            );
            unused.remove(best_slot);
        }

        for k in 0..n {
            let mut residual = 0.0_f64;
            let mut norm = 0.0;
            for r in 0..n {
                let mut av = Complex::ZERO;
                for j in 0..n {
                    av += a[(r, j)] * vectors[(j, k)];
                }
                residual = residual.max((av - values[k] * vectors[(r, k)]).abs());
                norm += vectors[(r, k)].abs() * vectors[(r, k)].abs();
            }

            assert!(residual <= tolerance, "{label} residual {k}");
            assert!((norm - 1.0).abs() <= 1e-12, "{label} unit norm {k}");
            if values[k].im == 0.0 {
                for r in 0..n {
                    assert_eq!(vectors[(r, k)].im, 0.0, "{label} real vector {k}");
                }
            }
        }
    }
}

#[test]
fn rust_only_full_piv_lu_rank_matches_numpy() {
    let json = fixture();
    for c in json.get("full_piv_lu_rank").as_array() {
        let rows = usize_of(c, "rows");
        let cols = usize_of(c, "cols");
        let lu = FullPivLu::new(&MatrixXd::from_column_major(
            rows,
            cols,
            &c.get("a").as_f64s(),
        ));
        assert_eq!(lu.rank(), usize_of(c, "rank"), "{rows}x{cols}");
    }
}
