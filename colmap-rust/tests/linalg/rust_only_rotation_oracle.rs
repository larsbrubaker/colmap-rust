// Rust-only (no COLMAP *_test.cc covers Eigen itself): Quaterniond and AngleAxisd against
// Eigen, through the pycolmap 4.2.0 Rotation3d binding. Port of colmap-sharp's
// `ColmapSharp.Tests/LinearAlgebra/RotationOracleTests.cs`, on the same fixture
// (`tests/data/oracle/linear_algebra_rotations.json`, written by
// `oracle/linear_algebra_rotations.py`, which lists which Eigen call each field comes from).
//
// Tiers, per operation (Rigid3d/Sim3d algebra downstream is Tier A, so these are pinned as
// tightly as the C++ allows):
// - Tier A, bit-identical on every case: product, matrix, from_matrix, from_general, norm,
//   inverse (`rust_only_exact_fields`).
// - Tier A up to entry 1 (`rust_only_atan2_fields`): angle and angle_to call atan2, which is
//   the `libm` crate's, not Apple libm's (docs/CPP_DIVERGENCES.md, entry 1). With std's atan2
//   (Apple libm on macOS) both formulas reproduce all 138 cases bit for bit; with `fns::atan2`
//   14 angles and 13 angle_tos differ by 1 ulp. Pinned at 2 ulp (distance on the bits).
// - Tier B (`rust_only_tolerance_fields`), each for a reason on the C++ side that we
//   deliberately do not reproduce, see docs/CPP_DIVERGENCES.md:
//   rotated         - the macOS arm64 wheel contracts the cross products inside q * v into
//                     FMAs (entry 2). With FMA emulated, our formula is exact on all cases;
//                     the oracle script prints that evidence.
//   from_axis_angle - the wheel's sin(a/2) can round 1 ulp away from libm sin (entry 3), and
//                     our sin/cos are the `libm` crate's (entry 1).
//   Observed on the 138 cases: rotated differs on 108 coefficients (the same with std math),
//   from_axis_angle on 15 (3 with std's sin/cos, up to 2 ulp).
// Tolerance: 1e-14 relative per coefficient (to max(1, |expected|)), a few ulps.

use crate::oracle_json::Json;
use colmap_rust::linalg::{AngleAxisd, Matrix3d, Quaterniond, Vector3d, Vector4d};

const FIXTURE: &str = include_str!("../data/oracle/linear_algebra_rotations.json");

fn fixture() -> Json {
    Json::parse(FIXTURE)
}

fn quat(c: &Json, name: &str) -> Quaterniond {
    let v = c.get(name).as_f64s();
    Quaterniond::from_coeffs(Vector4d::new(v[0], v[1], v[2], v[3]))
}

fn vec3(c: &Json, name: &str) -> Vector3d {
    let v = c.get(name).as_f64s();
    Vector3d::new(v[0], v[1], v[2])
}

fn row_major(c: &Json, name: &str) -> Matrix3d {
    let m = c.get(name).as_f64s();
    Matrix3d::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
}

fn flatten_quat(q: Quaterniond) -> Vec<f64> {
    vec![q.x, q.y, q.z, q.w]
}

fn flatten_matrix(m: Matrix3d) -> Vec<f64> {
    (0..3)
        .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
        .collect()
}

// Rotation3d(axis_angle): Quaterniond(AngleAxisd(v.norm(), v.normalized())).
fn axis_angle_vector_to_quaternion(v: Vector3d) -> Quaterniond {
    AngleAxisd::new(v.norm(), v.normalized()).to_quaternion()
}

fn compute(c: &Json, field: &str) -> Vec<f64> {
    let q = quat(c, "q");
    let other = quat(c, "other");
    match field {
        "product" => flatten_quat(q * other),
        "rotated" => {
            let v = q * vec3(c, "v");
            vec![v.x, v.y, v.z]
        }
        "matrix" => flatten_matrix(q.to_rotation_matrix()),
        "from_matrix" => flatten_quat(Quaterniond::from_rotation_matrix(row_major(c, "matrix"))),
        "from_general" => flatten_quat(Quaterniond::from_rotation_matrix(row_major(c, "general"))),
        "from_axis_angle" => flatten_quat(axis_angle_vector_to_quaternion(vec3(c, "axis_angle"))),
        "norm" => vec![q.norm()],
        "inverse" => flatten_quat(q.inverse()),
        "angle" => vec![AngleAxisd::from_quaternion(q).angle],
        "angle_to" => vec![q.angular_distance(other)],
        _ => panic!("unknown field {field}"),
    }
}

fn assert_all_cases(field: &str, agrees: impl Fn(f64, f64) -> bool) {
    let fixture = fixture();
    let cases = fixture.get("cases").as_array();
    assert!(!cases.is_empty());
    let mut mismatches = Vec::new();
    for c in cases {
        let expected = c.get(field).as_f64s();
        let actual = compute(c, field);
        assert_eq!(expected.len(), actual.len(), "{field}");
        for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            if !agrees(e, a) {
                mismatches.push(format!(
                    "q={:?} [{i}] expected {e:e} got {a:e}",
                    c.get("q").as_f64s()
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{field}: {} mismatches, first: {:#?}",
        mismatches.len(),
        &mismatches[..mismatches.len().min(10)]
    );
}

#[test]
fn rust_only_exact_fields() {
    for field in [
        "product",
        "matrix",
        "from_matrix",
        "from_general",
        "norm",
        "inverse",
    ] {
        assert_all_cases(field, |e, a| e.to_bits() == a.to_bits());
    }
}

#[test]
fn rust_only_atan2_fields() {
    for field in ["angle", "angle_to"] {
        assert_all_cases(field, |e, a| ulp_distance(e, a) <= 2);
    }
}

// Number of representable doubles between a and b: the bits mapped to a monotonic integer
// line (negative values mirrored below zero, so -0.0 and +0.0 are both 0).
fn ulp_distance(a: f64, b: f64) -> u64 {
    fn ordered(x: f64) -> i64 {
        let bits = x.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    }
    assert!(!a.is_nan() && !b.is_nan(), "NaN in ulp comparison");
    ordered(a).abs_diff(ordered(b))
}

#[test]
fn rust_only_tolerance_fields() {
    for field in ["rotated", "from_axis_angle"] {
        assert_all_cases(field, |e, a| (e - a).abs() <= 1e-14 * e.abs().max(1.0));
    }
}

#[test]
fn rust_only_trace_zero_matrices_take_the_diagonal_branch() {
    // Matrices whose trace is exactly 0: the fixture shows Eigen does not take the trace
    // branch there, which is why from_rotation_matrix tests trace > 0, not >= 0.
    let fixture = fixture();
    let cases = fixture.get("trace_zero_cases").as_array();
    assert!(!cases.is_empty());
    for (n, c) in cases.iter().enumerate() {
        let expected = c.get("from_matrix").as_f64s();
        let actual = flatten_quat(Quaterniond::from_rotation_matrix(row_major(c, "matrix")));
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&expected),
            bits(&actual),
            "case {n}: {expected:?} vs {actual:?}"
        );
    }
}

#[test]
fn rust_only_fixture_is_from_pinned_pycolmap() {
    assert_eq!(fixture().get("pycolmap_version").as_str(), "4.2.0");
}
