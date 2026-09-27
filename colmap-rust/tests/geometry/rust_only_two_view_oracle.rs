// Rust-only: pose, essential_matrix, homography_matrix, triangulation and pose_prior
// against COLMAP through the pycolmap 4.2.0 bindings (the ported tests check
// self-consistency and a few hand values). Port of colmap-sharp's
// `ColmapSharp.Tests/Geometry/GeometryTwoViewOracleTests.cs` on the same fixture,
// `tests/data/oracle/geometry_two_view.json`, written by `oracle/geometry_two_view.py`
// (which lists the COLMAP call behind each field).
//
// Tier B throughout (SVD, eigen and slerp paths, and 3x3 products whose grouping and FMA
// contraction differ from Eigen's in the macOS wheel, docs/CPP_DIVERGENCES.md entries 2, 20,
// 30 and 31). Each field is compared as |expected - actual| <= tolerance * max(1,
// |expected|); the bounds are colmap-sharp's (set from the largest gap it observed).
// ComputeRot90FromGravity is exact. The homography scenes carry a little noise: noise-free,
// a planar scene has two physically valid decompositions and the pick between them comes
// down to rounding (entry 86). Results COLMAP defines only up to sign are not in the fixture
// (the epipole); the ones compared here are sign-independent by construction (see the
// headers of the ported files), and rotations compare up to the quaternion sign.

use colmap_rust::geometry::essential_matrix::{
    compute_squared_sampson_errors, essential_matrix_from_pose,
};
use colmap_rust::geometry::homography_matrix::pose_from_homography_matrix;
use colmap_rust::geometry::pose::{average_quaternions, interpolate_camera_poses};
use colmap_rust::geometry::pose_prior::compute_rot90_from_gravity;
use colmap_rust::geometry::triangulation::{
    calculate_triangulation_angles, triangulate_mid_point, triangulate_multi_view_point_from_rays,
    triangulate_point, triangulate_point_from_rays,
};
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Matrix3d, Matrix3x4d, Quaterniond, Vector2d, Vector3d, Vector4d};

use crate::oracle_json::Json;

const FIXTURE: &str = include_str!("../data/oracle/geometry_two_view.json");

fn read(c: &Json, name: &str) -> Vec<f64> {
    c.get(name).as_f64s()
}

fn quat(xyzw: &[f64]) -> Quaterniond {
    Quaterniond::from_coeffs(Vector4d::new(xyzw[0], xyzw[1], xyzw[2], xyzw[3]))
}

fn vec3(v: &[f64]) -> Vector3d {
    Vector3d::new(v[0], v[1], v[2])
}

fn vec3s(v: &[f64]) -> Vec<Vector3d> {
    v.chunks(3).map(vec3).collect()
}

fn vec2s(v: &[f64]) -> Vec<Vector2d> {
    v.chunks(2).map(|p| Vector2d::new(p[0], p[1])).collect()
}

fn mat3(m: &[f64]) -> Matrix3d {
    Matrix3d::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
}

fn mat34(m: &[f64]) -> Matrix3x4d {
    Matrix3x4d::new(
        m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], m[9], m[10], m[11],
    )
}

fn flat_rows3(m: &Matrix3d) -> Vec<f64> {
    (0..3)
        .flat_map(|r| (0..3).map(move |c| m[(r, c)]))
        .collect()
}

fn flat(vectors: &[Vector3d]) -> Vec<f64> {
    vectors.iter().flat_map(|v| v.to_array()).collect()
}

// Largest |expected - actual| / max(1, |expected|) over a field.
fn gap(expected: &[f64], actual: &[f64]) -> f64 {
    if expected.len() != actual.len() {
        return f64::INFINITY;
    }
    expected.iter().zip(actual).fold(0.0, |g: f64, (&e, &a)| {
        g.max((e - a).abs() / e.abs().max(1.0))
    })
}

// A rotation compares up to sign (q and -q).
fn quat_gap(expected_xyzw: &[f64], actual: Quaterniond) -> f64 {
    let a = [actual.x, actual.y, actual.z, actual.w];
    let neg = a.map(|v| -v);
    gap(expected_xyzw, &a).min(gap(expected_xyzw, &neg))
}

struct Gaps(Vec<(&'static str, f64)>);

impl Gaps {
    fn record(&mut self, name: &'static str, g: f64) {
        match self.0.iter_mut().find(|(n, _)| *n == name) {
            Some(entry) => entry.1 = entry.1.max(g),
            None => self.0.push((name, g)),
        }
    }

    // !(gap <= bound) is deliberate: a NaN (or missing) gap must fail.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    fn assert_within(&self, bounds: &[(&str, f64)]) {
        let over: Vec<String> = bounds
            .iter()
            .filter_map(|&(name, bound)| {
                let g = self
                    .0
                    .iter()
                    .find(|(n, _)| *n == name)
                    .map_or(f64::NAN, |e| e.1);
                (!(g <= bound)).then(|| format!("{name}: gap {g:e} > {bound:e}"))
            })
            .collect();
        assert!(over.is_empty(), "{}", over.join("\n"));
    }
}

#[test]
fn rust_only_two_view_matches_pycolmap() {
    let fixture = Json::parse(FIXTURE);
    let mut gaps = Gaps(Vec::new());
    for c in fixture.get("two_view").as_array() {
        let rel = Rigid3d::new(quat(&read(c, "rel_q")), vec3(&read(c, "rel_t")));
        let e = essential_matrix_from_pose(&rel);
        gaps.record(
            "essential_matrix_from_pose",
            gap(&read(c, "E"), &flat_rows3(&e)),
        );

        let points1 = vec2s(&read(c, "noisy1"));
        let points2 = vec2s(&read(c, "noisy2"));
        let sampson =
            compute_squared_sampson_errors(&points1, &points2, &mat3(&read(c, "E"))).unwrap();
        gaps.record("sampson", gap(&read(c, "sampson"), &sampson));

        let p1 = mat34(&read(c, "P1"));
        let p2 = mat34(&read(c, "P2"));
        let p3 = mat34(&read(c, "P3"));
        let rays1 = vec3s(&read(c, "rays1"));
        let rays2 = vec3s(&read(c, "rays2"));
        let rays3 = vec3s(&read(c, "rays3"));

        let mut tri_points = Vec::new();
        let mut tri_bearings = Vec::new();
        let mut tri_mid = Vec::new();
        let mut tri_multi = Vec::new();
        for i in 0..points1.len() {
            tri_points.push(triangulate_point(&p1, &p2, points1[i], points2[i]).unwrap());
            tri_bearings.push(triangulate_point_from_rays(&p1, &p2, rays1[i], rays2[i]).unwrap());
            tri_mid.push(triangulate_mid_point(&rel, rays1[i], rays2[i]).unwrap());
            tri_multi.push(
                triangulate_multi_view_point_from_rays(
                    &[p1, p2, p3],
                    &[rays1[i], rays2[i], rays3[i]],
                )
                .unwrap()
                .unwrap(),
            );
        }
        gaps.record(
            "triangulate_point",
            gap(&read(c, "tri_points"), &flat(&tri_points)),
        );
        gaps.record(
            "triangulate_point_bearings",
            gap(&read(c, "tri_bearings"), &flat(&tri_bearings)),
        );
        gaps.record(
            "triangulate_mid_point",
            gap(&read(c, "tri_mid"), &flat(&tri_mid)),
        );
        gaps.record(
            "triangulate_multi_view_point",
            gap(&read(c, "tri_multi"), &flat(&tri_multi)),
        );

        let angles = calculate_triangulation_angles(
            vec3(&read(c, "c1")),
            vec3(&read(c, "c2")),
            &vec3s(&read(c, "points")),
        );
        gaps.record("triangulation_angle", gap(&read(c, "tri_angles"), &angles));
    }

    gaps.assert_within(&[
        ("essential_matrix_from_pose", 1e-15),
        ("sampson", 1e-15),
        ("triangulate_point", 1e-12),
        ("triangulate_point_bearings", 1e-12),
        ("triangulate_mid_point", 1e-12),
        ("triangulate_multi_view_point", 1e-12),
        // acos of a near-1 cosine.
        ("triangulation_angle", 1e-12),
    ]);
}

#[test]
fn rust_only_pose_from_homography_matrix_matches_pycolmap() {
    let fixture = Json::parse(FIXTURE);
    let mut gaps = Gaps(Vec::new());
    for c in fixture.get("homography").as_array() {
        let (cam2_from_cam1, normal, points3d) = pose_from_homography_matrix(
            &mat3(&read(c, "H")),
            &mat3(&read(c, "K1")),
            &mat3(&read(c, "K2")),
            &vec3s(&read(c, "rays1")),
            &vec3s(&read(c, "rays2")),
        )
        .unwrap();
        gaps.record("rotation", quat_gap(&read(c, "q"), cam2_from_cam1.rotation));
        gaps.record(
            "translation",
            gap(&read(c, "t"), &cam2_from_cam1.translation.to_array()),
        );
        gaps.record("normal", gap(&read(c, "normal"), &normal.to_array()));
        gaps.record("points3D", gap(&read(c, "points3D"), &flat(&points3d)));
    }

    gaps.assert_within(&[
        ("rotation", 1e-12),
        ("translation", 1e-12),
        ("normal", 1e-11),
        // Depths up to ~10.
        ("points3D", 1e-10),
    ]);
}

#[test]
fn rust_only_pose_matches_pycolmap() {
    let fixture = Json::parse(FIXTURE);
    let mut gaps = Gaps(Vec::new());
    let mut rot90_mismatches = Vec::new();
    for (index, c) in fixture.get("pose").as_array().iter().enumerate() {
        let quats: Vec<Quaterniond> = read(c, "quats").chunks(4).map(quat).collect();
        let average = average_quaternions(&quats, &read(c, "weights")).unwrap();
        gaps.record(
            "average_quaternions",
            quat_gap(&read(c, "average"), average),
        );

        let a = Rigid3d::new(quat(&read(c, "a_q")), vec3(&read(c, "a_t")));
        let b = Rigid3d::new(quat(&read(c, "b_q")), vec3(&read(c, "b_t")));
        let interp = interpolate_camera_poses(&a, &b, c.get("t").as_f64());
        gaps.record(
            "interpolate_rotation",
            quat_gap(&read(c, "interp_q"), interp.rotation),
        );
        gaps.record(
            "interpolate_translation",
            gap(&read(c, "interp_t"), &interp.translation.to_array()),
        );

        if f64::from(compute_rot90_from_gravity(vec3(&read(c, "gravity"))))
            != c.get("rot90").as_f64()
        {
            rot90_mismatches.push(index);
        }
    }

    assert!(rot90_mismatches.is_empty(), "{rot90_mismatches:?}");
    gaps.assert_within(&[
        ("average_quaternions", 1e-14),
        ("interpolate_rotation", 1e-15),
        ("interpolate_translation", 1e-15),
    ]);
}
