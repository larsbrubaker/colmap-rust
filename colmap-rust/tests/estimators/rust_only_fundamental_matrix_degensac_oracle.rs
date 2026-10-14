// Rust-only (fundamental_matrix_degensac_test.cc checks outcomes on generated scenes, not
// COLMAP's actual model): estimate_fundamental_matrix_degensac against COLMAP through
// pycolmap 4.2.0. Port of colmap-sharp's
// `ColmapSharp.Tests/Estimators/FundamentalMatrixDegensacOracleTests.cs` on the same fixture,
// `tests/data/oracle/fundamental_matrix_degensac.json`, written by
// `oracle/fundamental_matrix_degensac.py`; that script explains why the F of
// estimate_two_view_geometry with use_degensac is exactly DEGENSAC's model.
//
// Tier C with a tight bound: the RANSAC seed is fixed and the draws are Tier A, so the same
// samples, degeneracy decisions and completions happen on both sides and F agrees to the
// rounding of the solvers (Tier B), not merely to the same basin. F is compared up to scale
// and sign, after normalizing both to unit Frobenius norm. The bounds are colmap-sharp's.

use colmap_rust::estimators::fundamental_matrix_degensac_estimator::{
    estimate_fundamental_matrix_degensac, FundamentalMatrixDegensacOptions,
};
use colmap_rust::linalg::{Matrix3d, Vector2d};
use colmap_rust::optim::RansacOptions;

use crate::oracle_json::Json;

const FIXTURE: &str = include_str!("../data/oracle/fundamental_matrix_degensac.json");

// Without Sampson refinement F comes straight from the 7/8-point solvers or the
// plane-and-parallax completion (colmap-sharp's largest observed gap: 7.2e-16).
const SAMPLED_BOUND: f64 = 1e-12;

// The Sampson refinement is iterative (TinySolver vs COLMAP's), so its stopping point
// differs in the last digits (colmap-sharp's largest observed gap: 8.2e-9).
const REFINED_BOUND: f64 = 1e-7;

fn vec2s(v: &[f64]) -> Vec<Vector2d> {
    v.chunks(2).map(|p| Vector2d::new(p[0], p[1])).collect()
}

fn mat3(m: &[f64]) -> Matrix3d {
    Matrix3d::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
}

#[test]
fn rust_only_estimate_fundamental_matrix_degensac_matches_pycolmap() {
    let fixture = Json::parse(FIXTURE);
    let mut failures = Vec::new();
    let mut num_cases = 0;
    for c in fixture.get("cases").as_array() {
        num_cases += 1;
        let name = c.get("name").as_str();
        let points1 = vec2s(&c.get("points1").as_f64s());
        let points2 = vec2s(&c.get("points2").as_f64s());
        let expected = mat3(&c.get("F").as_f64s());
        let use_sampson_refinement = *c.get("use_sampson_refinement") == Json::Bool(true);

        let options = FundamentalMatrixDegensacOptions {
            ransac: RansacOptions {
                max_error: 1.0,
                confidence: 0.9999,
                min_inlier_ratio: 0.1,
                min_num_trials: 0,
                max_num_trials: 10000,
                random_seed: 0,
                ..RansacOptions::default()
            },
            use_sampson_refinement,
            ..FundamentalMatrixDegensacOptions::default()
        };
        let report = estimate_fundamental_matrix_degensac(&points1, &points2, &options).unwrap();

        let actual = report.model / report.model.norm();
        let expected_unit = expected / expected.norm();
        let gap = (actual - expected_unit)
            .norm()
            .min((actual + expected_unit).norm());
        let gap = if report.success { gap } else { f64::INFINITY };
        let bound = if use_sampson_refinement {
            REFINED_BOUND
        } else {
            SAMPLED_BOUND
        };
        if gap.is_nan() || gap >= bound {
            failures.push(format!("{name}: gap {gap:e} >= {bound:e}"));
        }
    }
    assert_eq!(num_cases, 8);
    assert!(failures.is_empty(), "{failures:#?}");
}
