// Rust-only: Rigid3d, Sim3d and GpsTransform against COLMAP through the pycolmap 4.2.0
// bindings (rigid3_test.cc, sim3_test.cc and gps_test.cc only check self-consistency). Port
// of colmap-sharp's `ColmapSharp.Tests/Geometry/GeometryOracleTests.cs` on the same fixture,
// `tests/data/oracle/geometry_transforms.json`, written by `oracle/geometry_transforms.py`
// (which lists the COLMAP call behind each field).
//
// Tiers, per field, as the fixture shows them:
// - Tier A, bit-identical on every case (`exact_fields`, `strings`): Rigid3d composition's
//   and inverse's rotation, to_matrix, from_matrix, adjoint; Sim3d composition's and
//   inverse's scale and rotation, to_matrix, from_matrix; the UTM zone; the Display strings
//   and Sim3d::to_file's text.
// - Tier B (`tolerance_fields`), each for an FMA contraction in the macOS arm64 wheel that
//   this port deliberately does not reproduce (docs/CPP_DIVERGENCES.md entries 2 and 80):
//   everything that rotates a vector with q * v, adjoint_inverse and
//   get_covariance_for_rigid3d_inverse (3x3 and 6x6 products), and the GPS conversions
//   (transcendentals through `fns` as well, entry 1). utm_to_ellipsoid also has a 1-ulp
//   latitude difference on 2/80 points (entry 83).
// Tolerances: |expected - actual| <= relative * max(1, |expected|), and for GPS coordinates
// in meters additionally within 1e-8 m (their last-bit differences at ~6.4e6 m ECEF carry
// through subtractions to small results).
//
// Not checked yet: cov_composed and cov_relative (the 12x12 covariance helpers arrive with
// the dynamic-size matrices).

use colmap_rust::geometry::{
    get_covariance_for_rigid3d_inverse, Ellipsoid, GpsTransform, Rigid3d, Sim3d,
};
use colmap_rust::linalg::{Matrix3x4d, Matrix6d, Quaterniond, Vector3d, Vector4d};

use crate::oracle_json::Json;

const FIXTURE: &str = include_str!("../data/oracle/geometry_transforms.json");

fn quat(c: &Json, name: &str) -> Quaterniond {
    let v = c.get(name).as_f64s();
    Quaterniond::from_coeffs(Vector4d::new(v[0], v[1], v[2], v[3]))
}

fn vec3(c: &Json, name: &str) -> Vector3d {
    let v = c.get(name).as_f64s();
    Vector3d::new(v[0], v[1], v[2])
}

fn points(c: &Json, name: &str) -> Vec<Vector3d> {
    c.get(name)
        .as_f64s()
        .chunks(3)
        .map(|p| Vector3d::new(p[0], p[1], p[2]))
        .collect()
}

fn matrix3x4_row_major(c: &Json, name: &str) -> Matrix3x4d {
    let m = c.get(name).as_f64s();
    Matrix3x4d::new(
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11],
    )
}

fn matrix6_row_major(c: &Json, name: &str) -> Matrix6d {
    // Row-major values read as column-major give the transpose; transpose back.
    let m = c.get(name).as_f64s();
    Matrix6d::from_column_major(m.try_into().unwrap()).transpose()
}

fn flat_q(q: Quaterniond) -> Vec<f64> {
    vec![q.x, q.y, q.z, q.w]
}

fn flat_v(v: Vector3d) -> Vec<f64> {
    v.to_array().to_vec()
}

fn flat_points(points: &[Vector3d]) -> Vec<f64> {
    points.iter().flat_map(|p| p.to_array()).collect()
}

fn flat_rows(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    // Column-major storage to row-major reading order, like the fixture.
    (0..rows)
        .flat_map(|i| (0..cols).map(move |j| data[j * rows + i]))
        .collect()
}

fn compute_rigid(c: &Json, field: &str) -> Vec<f64> {
    let r = Rigid3d::new(quat(c, "q"), vec3(c, "t"));
    let other = Rigid3d::new(quat(c, "other_q"), vec3(c, "other_t"));
    match field {
        "apply" => flat_v(r * vec3(c, "x")),
        "compose_q" => flat_q((r * other).rotation),
        "compose_t" => flat_v((r * other).translation),
        "inverse_q" => flat_q(r.inverse().rotation),
        "inverse_t" => flat_v(r.inverse().translation),
        "matrix" => flat_rows(r.to_matrix().as_slice(), 3, 4),
        "from_matrix_q" => {
            flat_q(Rigid3d::from_matrix(&matrix3x4_row_major(c, "general")).rotation)
        }
        "from_matrix_t" => {
            flat_v(Rigid3d::from_matrix(&matrix3x4_row_major(c, "general")).translation)
        }
        "tgt_origin_in_src" => flat_v(r.tgt_origin_in_src()),
        "adjoint" => flat_rows(r.adjoint().as_slice(), 6, 6),
        "adjoint_inverse" => flat_rows(r.adjoint_inverse().as_slice(), 6, 6),
        "cov_inverse" => flat_rows(
            get_covariance_for_rigid3d_inverse(&r, &matrix6_row_major(c, "cov")).as_slice(),
            6,
            6,
        ),
        _ => panic!("unknown rigid field {field}"),
    }
}

fn compute_sim(c: &Json, field: &str) -> Vec<f64> {
    let s = Sim3d::new(c.get("s").as_f64(), quat(c, "q"), vec3(c, "t"));
    let other = Sim3d::new(
        c.get("other_s").as_f64(),
        quat(c, "other_q"),
        vec3(c, "other_t"),
    );
    let from_matrix = || Sim3d::from_matrix(&matrix3x4_row_major(c, "general"));
    match field {
        "apply" => flat_v(s * vec3(c, "x")),
        "compose_s" => vec![(s * other).scale],
        "compose_q" => flat_q((s * other).rotation),
        "compose_t" => flat_v((s * other).translation),
        "inverse_s" => vec![s.inverse().scale],
        "inverse_q" => flat_q(s.inverse().rotation),
        "inverse_t" => flat_v(s.inverse().translation),
        "matrix" => flat_rows(s.to_matrix().as_slice(), 3, 4),
        "from_matrix_s" => vec![from_matrix().scale],
        "from_matrix_q" => flat_q(from_matrix().rotation),
        "from_matrix_t" => flat_v(from_matrix().translation),
        _ => panic!("unknown sim field {field}"),
    }
}

fn compute_gps(c: &Json, field: &str) -> Vec<f64> {
    let ellipsoid = match c.get("ellipsoid").as_str() {
        "GRS80" => Ellipsoid::Grs80,
        "WGS84" => Ellipsoid::Wgs84,
        other => panic!("unknown ellipsoid {other}"),
    };
    let gps = GpsTransform::new(ellipsoid);
    let lla = points(c, "lla");
    let r = vec3(c, "ref");
    let ecef = points(c, "ecef");
    let enu = points(c, "enu");
    match field {
        "ecef" => flat_points(&gps.ellipsoid_to_ecef(&lla)),
        "ecef_to_ellipsoid" => flat_points(&gps.ecef_to_ellipsoid(&ecef)),
        "enu" => flat_points(&gps.ellipsoid_to_enu(&lla, r.x, r.y, r.z)),
        "ecef_to_enu" => flat_points(&gps.ecef_to_enu(&ecef, ecef[0])),
        "enu_to_ellipsoid" => flat_points(&gps.enu_to_ellipsoid(&enu, r.x, r.y, r.z)),
        "enu_to_ecef" => flat_points(&gps.enu_to_ecef(&enu, r.x, r.y, r.z)),
        "utm" => flat_points(&gps.ellipsoid_to_utm(&lla).unwrap().0),
        "utm_zone" => vec![f64::from(gps.ellipsoid_to_utm(&lla).unwrap().1)],
        "utm_to_ellipsoid" => {
            let is_north = matches!(c.get("is_north"), Json::Bool(true));
            let zone = c.get("zone").as_f64() as i32;
            flat_points(
                &gps.utm_to_ellipsoid(&points(c, "utm"), zone, is_north)
                    .unwrap(),
            )
        }
        _ => panic!("unknown gps field {field}"),
    }
}

fn expected(c: &Json, field: &str) -> Vec<f64> {
    if field == "utm_zone" {
        vec![c.get("zone").as_f64()]
    } else {
        c.get(field).as_f64s()
    }
}

// Compares every case of one field; returns the mismatch descriptions.
fn mismatches(section: &str, field: &str, equal: impl Fn(f64, f64, usize) -> bool) -> Vec<String> {
    let fixture = Json::parse(FIXTURE);
    let mut out = Vec::new();
    for (index, c) in fixture.get(section).as_array().iter().enumerate() {
        let expected = expected(c, field);
        let actual = match section {
            "rigid" => compute_rigid(c, field),
            "sim" => compute_sim(c, field),
            "gps" => compute_gps(c, field),
            _ => panic!("unknown section {section}"),
        };
        assert_eq!(
            expected.len(),
            actual.len(),
            "{section}.{field} case {index}"
        );
        for (i, (&e, &a)) in expected.iter().zip(&actual).enumerate() {
            if !equal(e, a, i) {
                out.push(format!(
                    "{section}.{field} case {index} [{i}]: expected {e:?}, got {a:?}"
                ));
            }
        }
    }
    out
}

#[test]
fn rust_only_transforms_oracle_exact_fields() {
    let fields = [
        ("rigid", "compose_q"),
        ("rigid", "inverse_q"),
        ("rigid", "matrix"),
        ("rigid", "from_matrix_q"),
        ("rigid", "from_matrix_t"),
        ("rigid", "adjoint"),
        ("sim", "compose_s"),
        ("sim", "compose_q"),
        ("sim", "inverse_s"),
        ("sim", "inverse_q"),
        ("sim", "matrix"),
        ("sim", "from_matrix_s"),
        ("sim", "from_matrix_q"),
        ("sim", "from_matrix_t"),
        ("gps", "utm_zone"),
    ];
    let mut all = Vec::new();
    for (section, field) in fields {
        all.extend(mismatches(section, field, |e, a, _| {
            e.to_bits() == a.to_bits()
        }));
    }
    assert!(all.is_empty(), "{}", all.join("\n"));
}

// Whether coefficient `component` of a field is a coordinate in meters: every GPS output
// except the (lat, lon) of the ellipsoidal ones.
fn is_meters(section: &str, field: &str, component: usize) -> bool {
    section == "gps" && (!field.ends_with("ellipsoid") || component % 3 == 2)
}

#[test]
fn rust_only_transforms_oracle_tolerance_fields() {
    let fields = [
        ("rigid", "apply", 1e-14),
        ("rigid", "compose_t", 1e-14),
        ("rigid", "inverse_t", 1e-14),
        ("rigid", "tgt_origin_in_src", 1e-14),
        ("rigid", "adjoint_inverse", 1e-14),
        ("rigid", "cov_inverse", 1e-13),
        ("sim", "apply", 1e-14),
        ("sim", "compose_t", 1e-14),
        ("sim", "inverse_t", 1e-14),
        ("gps", "ecef", 1e-14),
        ("gps", "ecef_to_ellipsoid", 1e-14),
        ("gps", "enu", 1e-14),
        ("gps", "ecef_to_enu", 1e-14),
        ("gps", "enu_to_ellipsoid", 1e-14),
        ("gps", "enu_to_ecef", 1e-14),
        ("gps", "utm", 1e-14),
        ("gps", "utm_to_ellipsoid", 1e-14),
    ];
    const METERS_TOLERANCE: f64 = 1e-8;
    let mut all = Vec::new();
    for (section, field, relative) in fields {
        all.extend(mismatches(section, field, |e, a, component| {
            let difference = (e - a).abs();
            difference <= relative * e.abs().max(1.0)
                || (is_meters(section, field, component) && difference <= METERS_TOLERANCE)
        }));
    }
    assert!(all.is_empty(), "{}", all.join("\n"));
}

#[test]
fn rust_only_transforms_oracle_strings() {
    let fixture = Json::parse(FIXTURE);
    let mut bad = Vec::new();
    for c in fixture.get("rigid").as_array() {
        let actual = Rigid3d::new(quat(c, "q"), vec3(c, "t")).to_string();
        if actual != c.get("str").as_str() {
            bad.push(format!("expected {}, got {actual}", c.get("str").as_str()));
        }
    }
    let dir = std::env::temp_dir().join(format!("colmap_rust_sim3d_oracle_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for (index, c) in fixture.get("sim").as_array().iter().enumerate() {
        let sim = Sim3d::new(c.get("s").as_f64(), quat(c, "q"), vec3(c, "t"));
        if sim.to_string() != c.get("str").as_str() {
            bad.push(format!("expected {}, got {sim}", c.get("str").as_str()));
        }
        let path = dir.join(format!("sim_{index}.txt"));
        sim.to_file(&path).unwrap();
        let written = std::fs::read_to_string(&path).unwrap();
        let expected_file = format!("{}\n", c.get("to_file").as_str());
        if written != expected_file {
            bad.push(format!(
                "to_file: expected {expected_file:?}, got {written:?}"
            ));
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
}
