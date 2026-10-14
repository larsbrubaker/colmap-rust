#!/usr/bin/env python3
# fundamental_matrix_degensac.py: DEGENSAC fundamental matrices from pycolmap, for
# colmap-rust/tests/estimators/rust_only_fundamental_matrix_degensac_oracle.rs. Copied from
# colmap-sharp's oracle/fundamental_matrix_degensac.py with the output path changed.
#
# pycolmap 4.2.0 does not bind EstimateFundamentalMatrixDegensac directly, so this goes
# through estimate_two_view_geometry with TwoViewGeometryOptions.use_degensac = True: two
# SIMPLE_PINHOLE cameras with different ids and no focal prior route to
# EstimateUncalibratedTwoViewGeometry, whose first step is
#   geometry.F = EstimateFundamentalMatrix(options, options.ransac_options, p1, p2).model
# and with use_degensac that is EstimateFundamentalMatrixDegensac(p1, p2, {ransac,
# use_sampson_refinement}) on the matched points in match order. Nothing later in that
# function touches geometry.F, so the returned F is DEGENSAC's model. The RANSAC seed is
# fixed, so the model is a deterministic function of the points and the draws.
#
# Scenes: dominant-plane scenes (most points on one plane, the rest at random depth) plus a
# few gross outliers, and one general scene, with and without Sampson refinement. Without
# refinement F is taken straight from a sample (or a plane-and-parallax completion), so any
# divergence in the PRNG draws shows up in F. Values are JSON floats, matrices row-major.
# Deterministic (fixed numpy seed).
#
# Usage: oracle/.venv/bin/python oracle/fundamental_matrix_degensac.py
# Writes colmap-rust/tests/data/oracle/fundamental_matrix_degensac.json.

import json
import pathlib

import numpy as np
import pycolmap

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / "colmap-rust" / "tests" / "data" / "oracle" / "fundamental_matrix_degensac.json")


def floats(array):
    return [float(v) for v in np.asarray(array, dtype=np.float64).ravel(order="C")]


def rotation(rng, spread=0.2):
    q = np.concatenate([[1.0], rng.normal(size=3) * spread])
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def scene(rng, num_points, num_on_plane, num_outliers, noise):
    f = rng.uniform(800, 1200)
    k = np.array([[f, 0, 500.0], [0, f, 400.0], [0, 0, 1]])
    r = rotation(rng)
    t = rng.normal(size=3)
    t /= np.linalg.norm(t)
    normal = np.array([0.2, -0.1, 1.0])
    normal /= np.linalg.norm(normal)
    distance = 4.0
    points1 = []
    points2 = []
    for i in range(num_points):
        x1 = np.array([rng.uniform(100, 900), rng.uniform(100, 700), 1.0])
        ray = np.linalg.solve(k, x1)
        if i < num_on_plane:
            depth = distance / normal.dot(ray)
        else:
            depth = rng.uniform(2.0, 8.0)
        p2 = k @ (r @ (depth * ray) + t)
        points1.append(x1[:2] + noise * rng.normal(size=2))
        points2.append(p2[:2] / p2[2] + noise * rng.normal(size=2))
    for _ in range(num_outliers):
        j = rng.integers(num_points)
        points2[j] = np.array([rng.uniform(0, 1000), rng.uniform(0, 800)])
    return np.array(points1), np.array(points2)


def degensac_f(points1, points2, use_sampson_refinement):
    cameras = []
    for camera_id in (1, 2):
        camera = pycolmap.Camera(model="SIMPLE_PINHOLE", width=1000, height=800, params=[1000, 500, 400])
        camera.camera_id = camera_id
        camera.has_prior_focal_length = False
        cameras.append(camera)
    options = pycolmap.TwoViewGeometryOptions()
    options.use_degensac = True
    options.use_sampson_refinement = use_sampson_refinement
    options.min_num_inliers = 8
    options.ransac.max_error = 1.0
    options.ransac.confidence = 0.9999
    options.ransac.min_inlier_ratio = 0.1
    options.ransac.min_num_trials = 0
    options.ransac.max_num_trials = 10000
    options.ransac.random_seed = 0
    n = len(points1)
    matches = np.stack([np.arange(n), np.arange(n)], axis=1).astype(np.uint32)
    geometry = pycolmap.estimate_two_view_geometry(cameras[0], points1, cameras[1], points2, matches, options)
    return geometry.F


def main():
    rng = np.random.default_rng(20260926)
    cases = []
    specs = [
        ("plane95", 200, 190, 10, 0.2),
        ("plane90", 150, 135, 8, 0.3),
        ("plane98", 250, 245, 5, 0.1),
        ("general", 120, 0, 10, 0.3),
    ]
    for name, num_points, num_on_plane, num_outliers, noise in specs:
        points1, points2 = scene(rng, num_points, num_on_plane, num_outliers, noise)
        for refine in (False, True):
            f = degensac_f(points1, points2, refine)
            cases.append({
                "name": f"{name}_{'refine' if refine else 'norefine'}",
                "use_sampson_refinement": refine,
                "points1": floats(points1),
                "points2": floats(points2),
                "F": floats(f),
            })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"pycolmap": pycolmap.__version__, "cases": cases}, indent=1) + "\n")
    print(f"wrote {OUT} ({len(cases)} cases)")


if __name__ == "__main__":
    main()
