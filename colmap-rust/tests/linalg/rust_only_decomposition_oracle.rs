// Rust-only (COLMAP has no test for Eigen itself): the dense decompositions in
// colmap_rust::linalg against numpy (LAPACK) on random matrices. Port of colmap-sharp's
// `ColmapSharp.Tests/LinearAlgebra/DecompositionOracleTests.cs`, on the same fixture
// (`tests/data/oracle/linear_algebra_dense.json`, written by `oracle/linear_algebra_dense.py`,
// which says which LAPACK routine each field comes from).
//
// Tier B. Tolerances (entries are O(1), matrices are small and well conditioned):
// - HouseholderQr Q and R: 1e-12 absolute per entry, signs included (the reflector sign
//   convention is LAPACK's, which Eigen documents too).
// - Least-squares, LU and Cholesky solutions, inverse, Cholesky L: 1e-10 absolute per entry,
//   which leaves room for the condition number (the observed gaps are ~1e-15).
// - Determinant: 1e-12 relative.

use crate::oracle_json::Json;
use colmap_rust::linalg::{
    ColPivHouseholderQr, ComputationInfo, HouseholderQr, Ldlt, Llt, MatrixXd, PartialPivLu,
    VectorXd,
};

const FIXTURE: &str = include_str!("../data/oracle/linear_algebra_dense.json");

fn cases(name: &str) -> Vec<Json> {
    Json::parse(FIXTURE).get(name).as_array().to_vec()
}

fn try_get<'a>(c: &'a Json, name: &str) -> Option<&'a Json> {
    match c {
        Json::Object(map) => map.get(name),
        _ => None,
    }
}

fn int(c: &Json, name: &str) -> usize {
    c.get(name).as_f64() as usize
}

fn matrix(c: &Json, name: &str, rows: usize, cols: usize) -> MatrixXd {
    MatrixXd::from_column_major(rows, cols, &c.get(name).as_f64s())
}

fn vector(c: &Json, name: &str) -> VectorXd {
    VectorXd::from_vec(c.get(name).as_f64s())
}

fn assert_near(label: &str, actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        let difference = (a - e).abs();
        assert!(
            difference <= tolerance,
            "{label}[{i}]: {a:e} vs {e:e} (difference {difference:e})"
        );
    }
}

#[test]
fn rust_only_householder_qr_matches_numpy() {
    for c in cases("qr") {
        let rows = int(&c, "rows");
        let cols = int(&c, "cols");
        let a = matrix(&c, "a", rows, cols);
        let qr = HouseholderQr::new(&a);
        let label = format!("{rows}x{cols}");
        let q = c.get("q").as_f64s();
        let r = c.get("r").as_f64s();
        assert_near(
            &format!("{label} Q"),
            qr.householder_q().as_slice(),
            &q,
            1e-12,
        );
        assert_near(&format!("{label} R"), qr.matrix_r().as_slice(), &r, 1e-12);
        // Only the rows >= cols cases carry a least-squares solution.
        if let Some(x) = try_get(&c, "x") {
            let x = x.as_f64s();
            let b = vector(&c, "b");
            assert_near(&format!("{label} x"), qr.solve(&b).as_slice(), &x, 1e-10);
            let pivoted = ColPivHouseholderQr::new(&a).solve(&b);
            assert_near(&format!("{label} colpiv x"), pivoted.as_slice(), &x, 1e-10);
        }
    }
}

#[test]
fn rust_only_partial_piv_lu_matches_numpy() {
    for c in cases("lu") {
        let n = int(&c, "n");
        let lu = PartialPivLu::new(&matrix(&c, "a", n, n));
        let label = format!("n={n}");
        let x = c.get("x").as_f64s();
        let inverse = c.get("inverse").as_f64s();
        assert_near(
            &format!("{label} x"),
            lu.solve(&vector(&c, "b")).as_slice(),
            &x,
            1e-10,
        );
        assert_near(
            &format!("{label} inverse"),
            lu.inverse().as_slice(),
            &inverse,
            1e-10,
        );
        let det = c.get("det").as_f64();
        assert!(
            (lu.determinant() - det).abs() <= 1e-12 * det.abs(),
            "{label} det: {} vs {det}",
            lu.determinant()
        );
    }
}

#[test]
fn rust_only_cholesky_matches_numpy() {
    for c in cases("spd") {
        let n = int(&c, "n");
        let a = matrix(&c, "a", n, n);
        let llt = Llt::new(&a);
        let label = format!("n={n}");
        let x = c.get("x").as_f64s();
        let l = c.get("l").as_f64s();
        let b = vector(&c, "b");
        assert_eq!(llt.info(), ComputationInfo::Success);
        assert_near(&format!("{label} L"), llt.matrix_l().as_slice(), &l, 1e-10);
        assert_near(
            &format!("{label} llt x"),
            llt.solve(&b).as_slice(),
            &x,
            1e-10,
        );
        let ldlt_x = Ldlt::new(&a).solve(&b);
        assert_near(&format!("{label} ldlt x"), ldlt_x.as_slice(), &x, 1e-10);
    }
}
