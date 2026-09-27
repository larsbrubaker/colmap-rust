#!/usr/bin/env python3
# geometry_two_view.py: two-view geometry outputs from pycolmap, for
# colmap-rust/tests/geometry/rust_only_two_view_oracle.rs (copied from colmap-sharp's
# oracle/, which checks the same fixture in its GeometryTwoViewOracleTests.cs).
#
# pycolmap's bindings (src/pycolmap/geometry/*.cc) call COLMAP directly, so each call below
# is one COLMAP function:
#   essential_matrix_from_pose(r)              EssentialMatrixFromPose
#   compute_squared_sampson_error(p1, p2, E)   ComputeSquaredSampsonError (Vector2d overload)
#   triangulate_point(P1, P2, x1, x2)          TriangulatePoint (image points)
#   triangulate_point(P1, P2, ray1, ray2)      TriangulatePoint (bearings)
#   triangulate_mid_point(r, ray1, ray2)       TriangulateMidPoint
#   triangulate_multi_view_point(Ps, rays)     TriangulateMultiViewPoint (bearings)
#   calculate_triangulation_angle(c1, c2, X)   CalculateTriangulationAngle
#   pose_from_homography_matrix(H, K1, K2, rays1, rays2)   PoseFromHomographyMatrix
#   average_quaternions(qs, ws)                AverageQuaternions
#   interpolate_camera_poses(a, b, t)          InterpolateCameraPoses
#   compute_rot90_from_gravity(g)              ComputeRot90FromGravity
#
# Scenes are random but well conditioned (points 2-10 units in front of both cameras,
# baselines of about one unit). Values are JSON floats (Python's repr round-trips every
# double), matrices row-major, quaternions (x, y, z, w). Deterministic (fixed numpy seed).
#
# Usage: oracle/.venv/bin/python oracle/geometry_two_view.py
# Writes colmap-rust/tests/data/oracle/geometry_two_view.json.

import json
import pathlib

import numpy as np
import pycolmap

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / "colmap-rust" / "tests" / "data" / "oracle" / "geometry_two_view.json")


def floats(array):
    return [float(v) for v in np.asarray(array, dtype=np.float64).ravel(order="C")]


def unit_quat(rng, spread=0.3):
    # Small rotations keep the scene in front of both cameras.
    q = np.concatenate([rng.normal(size=3) * spread, [1.0]])
    return q / np.linalg.norm(q)


def rigid(rng):
    return pycolmap.Rigid3d(pycolmap.Rotation3d(unit_quat(rng)), rng.normal(size=3))


def two_view_case(rng):
    cam1 = rigid(rng)
    cam2 = rigid(rng)
    cam3 = rigid(rng)
    rel = cam2 * cam1.inverse()
    points = rng.normal(size=(6, 3)) + np.array([0.0, 0.0, 6.0])
    rays1 = np.array([(cam1 * p) / np.linalg.norm(cam1 * p) for p in points])
    rays2 = np.array([(cam2 * p) / np.linalg.norm(cam2 * p) for p in points])
    rays3 = np.array([(cam3 * p) / np.linalg.norm(cam3 * p) for p in points])
    noisy1 = np.array([(cam1 * p)[:2] / (cam1 * p)[2] for p in points]) + rng.normal(size=(6, 2)) * 1e-3
    noisy2 = np.array([(cam2 * p)[:2] / (cam2 * p)[2] for p in points]) + rng.normal(size=(6, 2)) * 1e-3
    P1, P2, P3 = cam1.matrix(), cam2.matrix(), cam3.matrix()
    E = pycolmap.essential_matrix_from_pose(rel)
    c1 = cam1.inverse().translation
    c2 = cam2.inverse().translation
    return {
        "cam1_q": floats(cam1.rotation.quat), "cam1_t": floats(cam1.translation),
        "cam2_q": floats(cam2.rotation.quat), "cam2_t": floats(cam2.translation),
        "cam3_q": floats(cam3.rotation.quat), "cam3_t": floats(cam3.translation),
        "rel_q": floats(rel.rotation.quat), "rel_t": floats(rel.translation),
        "P1": floats(P1), "P2": floats(P2), "P3": floats(P3),
        "rays1": floats(rays1), "rays2": floats(rays2), "rays3": floats(rays3),
        "noisy1": floats(noisy1), "noisy2": floats(noisy2),
        "E": floats(E),
        "sampson": floats(pycolmap.compute_squared_sampson_error(noisy1, noisy2, E)),
        "tri_points": floats([pycolmap.triangulate_point(P1, P2, noisy1[i], noisy2[i]) for i in range(6)]),
        "tri_bearings": floats([pycolmap.triangulate_point(P1, P2, rays1[i], rays2[i]) for i in range(6)]),
        "tri_mid": floats([pycolmap.triangulate_mid_point(rel, rays1[i], rays2[i]) for i in range(6)]),
        "tri_multi": floats([pycolmap.triangulate_multi_view_point(
            [P1, P2, P3], np.array([rays1[i], rays2[i], rays3[i]])) for i in range(6)]),
        "c1": floats(c1), "c2": floats(c2), "points": floats(points),
        "tri_angles": [pycolmap.calculate_triangulation_angle(c1, c2, p) for p in points],
    }


def homography_case(rng):
    q = unit_quat(rng, 0.1)
    t = rng.normal(size=3)
    n = np.array([0.0, 0.0, -1.0]) + rng.normal(size=3) * 0.2
    n /= np.linalg.norm(n)
    d = float(rng.uniform(2, 5))
    R = pycolmap.Rotation3d(q).matrix()
    K1 = np.array([[500.0, 0, 320], [0, 510, 240], [0, 0, 1]])
    K2 = np.array([[520.0, 0, 330], [0, 505, 250], [0, 0, 1]])
    H = K2 @ (R - np.outer(t, n) / d) @ np.linalg.inv(K1)
    # Points on the plane n . X + d = 0 seen by camera 1 (n points towards camera 1).
    uv = rng.uniform(-1.0, 1.0, size=(8, 2))
    rays1 = np.column_stack([uv, np.ones(8)])
    rays1 /= np.linalg.norm(rays1, axis=1)[:, None]
    points = np.array([r * (-d / (n @ r)) for r in rays1])
    cam2 = np.array([R @ p + t for p in points])
    # Noise-free, both physically valid decompositions triangulate every point exactly
    # and COLMAP's choice between them comes down to rounding; a little noise on the
    # second view's rays (and the wide field of view) makes the choice well defined.
    cam2 = cam2 + rng.normal(size=cam2.shape) * 1e-3 * np.linalg.norm(cam2, axis=1)[:, None]
    rays2 = cam2 / np.linalg.norm(cam2, axis=1)[:, None]
    result = pycolmap.pose_from_homography_matrix(H, K1, K2, rays1, rays2)
    return {
        "H": floats(H), "K1": floats(K1), "K2": floats(K2),
        "rays1": floats(rays1), "rays2": floats(rays2),
        "q": floats(result["cam2_from_cam1"].rotation.quat),
        "t": floats(result["cam2_from_cam1"].translation),
        "normal": floats(result["normal"]),
        "points3D": floats(result["points3D"]),
    }


def pose_case(rng):
    quats = [unit_quat(rng, 0.5) for _ in range(4)]
    weights = rng.uniform(0.5, 2.0, size=4)
    a, b = rigid(rng), rigid(rng)
    t = float(rng.uniform())
    interp = pycolmap.interpolate_camera_poses(a, b, t)
    gravity = rng.normal(size=3)
    return {
        "quats": floats(quats), "weights": floats(weights),
        "average": floats(pycolmap.average_quaternions([pycolmap.Rotation3d(q) for q in quats], list(weights)).quat),
        "a_q": floats(a.rotation.quat), "a_t": floats(a.translation),
        "b_q": floats(b.rotation.quat), "b_t": floats(b.translation),
        "t": t,
        "interp_q": floats(interp.rotation.quat), "interp_t": floats(interp.translation),
        "gravity": floats(gravity),
        "rot90": int(pycolmap.compute_rot90_from_gravity(gravity)),
    }


def main():
    rng = np.random.default_rng(20260926)
    fixture = {
        "two_view": [two_view_case(rng) for _ in range(20)],
        "homography": [homography_case(rng) for _ in range(20)],
        "pose": [pose_case(rng) for _ in range(20)],
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
