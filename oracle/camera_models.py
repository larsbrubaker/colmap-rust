#!/usr/bin/env python3
# camera_models.py: camera model projection and unprojection outputs, via pycolmap's
# Camera binding, for colmap-rust/tests/sensor/rust_only_camera_model_oracle.rs (copied from
# colmap-sharp's oracle/, which checks the same fixture in its CameraModelOracleTests.cs).
#
# pycolmap.Camera's methods are thin wrappers over the dispatch functions of
# colmap/sensor/models.h (src/colmap/scene/camera.h forwards each one unchanged):
#   img_from_cam(p, check_cheirality)  CameraModelImgFromCam
#   cam_from_img(xy)                   CameraModelCamFromImg (the iterative undistortion)
#   cam_ray_from_img(xy)               CameraModelCamRayFromImg
#   cam_from_img_threshold(t)          CameraModelCamFromImgThreshold
# Every model is run with two random parameter sets (FOV with a third, small omega) on a grid of camera points (including
# points behind the camera with the cheirality check off) and a grid of pixels. Failures
# are recorded as null. Points on and next to the optical axis and pixels at and next to the
# principal point pin the models' small-radius branches. The fixture writes JSON floats (Python's repr round-trips every
# double exactly; NaN and infinities are written as strings), so the Rust test compares
# bit for bit.
#
# Usage: oracle/.venv/bin/python oracle/camera_models.py
# Writes colmap-rust/tests/data/oracle/camera_models.json.

import json
import math
import pathlib

import numpy as np
import pycolmap

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / "colmap-rust" / "tests" / "data" / "oracle" / "camera_models.json")

WIDTH = 800
HEIGHT = 600


def focal_pp(rng, num_focal):
    focal = list(rng.uniform(400, 900, size=num_focal))
    return focal + [rng.uniform(300, 500), rng.uniform(200, 400)]


def params_for(model, rng):
    u = rng.uniform
    if model == "SIMPLE_PINHOLE":
        return focal_pp(rng, 1)
    if model == "PINHOLE":
        return focal_pp(rng, 2)
    if model == "SIMPLE_RADIAL":
        return focal_pp(rng, 1) + [u(-0.3, 0.3)]
    if model == "RADIAL":
        return focal_pp(rng, 1) + [u(-0.3, 0.3), u(-0.1, 0.1)]
    if model == "OPENCV":
        return focal_pp(rng, 2) + [u(-0.3, 0.3), u(-0.1, 0.1), u(-0.005, 0.005), u(-0.005, 0.005)]
    if model == "OPENCV_FISHEYE":
        return focal_pp(rng, 2) + list(u(-0.05, 0.05, size=4))
    if model == "FULL_OPENCV":
        return (focal_pp(rng, 2) + [u(-0.3, 0.3), u(-0.1, 0.1), u(-0.005, 0.005), u(-0.005, 0.005)]
                + list(u(-0.05, 0.05, size=4)))
    if model == "FOV":
        return focal_pp(rng, 2) + [u(0.2, 1.0)]
    if model == "SIMPLE_RADIAL_FISHEYE":
        return focal_pp(rng, 1) + [u(-0.05, 0.05)]
    if model == "RADIAL_FISHEYE":
        return focal_pp(rng, 1) + list(u(-0.05, 0.05, size=2))
    if model == "THIN_PRISM_FISHEYE":
        return (focal_pp(rng, 2) + [u(-0.05, 0.05), u(-0.05, 0.05), u(-0.002, 0.002), u(-0.002, 0.002),
                                    u(-0.02, 0.02), u(-0.02, 0.02), u(-0.002, 0.002), u(-0.002, 0.002)])
    if model == "RAD_TAN_THIN_PRISM_FISHEYE":
        return focal_pp(rng, 2) + list(u(-0.05, 0.05, size=6)) + list(u(-0.001, 0.001, size=6))
    if model == "SIMPLE_DIVISION":
        return focal_pp(rng, 1) + [u(-0.2, 0.2)]
    if model == "DIVISION":
        return focal_pp(rng, 2) + [u(-0.2, 0.2)]
    if model == "SIMPLE_FISHEYE":
        return focal_pp(rng, 1)
    if model == "FISHEYE":
        return focal_pp(rng, 2)
    if model == "EUCM":
        return focal_pp(rng, 2) + [u(0.2, 0.8), u(0.5, 1.5)]
    if model == "EQUIRECTANGULAR":
        return [float(WIDTH), float(HEIGHT)]
    raise ValueError(model)


MODELS = [
    "SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "OPENCV_FISHEYE",
    "FULL_OPENCV", "FOV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "THIN_PRISM_FISHEYE",
    "RAD_TAN_THIN_PRISM_FISHEYE", "SIMPLE_DIVISION", "DIVISION", "SIMPLE_FISHEYE", "FISHEYE",
    "EUCM", "EQUIRECTANGULAR",
]


# On and next to the optical axis: the small-radius branches of the projections.
NEAR_AXIS_POINTS = [(0.0, 0.0, 1.0), (0.003, -0.002, 1.0)]


def num(v):
    # JSON has no NaN/inf; the Rust test reads these strings back (str::parse::<f64>
    # accepts "NaN", "Infinity" and "-Infinity").
    v = float(v)
    if np.isnan(v):
        return "NaN"
    if np.isinf(v):
        return "Infinity" if v > 0 else "-Infinity"
    return v


def opt(value):
    return None if value is None else [num(v) for v in np.asarray(value).ravel()]


def main():
    rng = np.random.default_rng(20260926)
    grid = np.linspace(-0.6, 0.6, 5)
    cam_points = [(x, y, w) for x in grid for y in grid for w in (0.5, 1.0, 2.0)]
    # Behind the camera, and one point on the camera plane, with the cheirality check off.
    back_points = [(x, y, -1.0) for x in grid[::2] for y in grid[::2]] + [(0.1, 0.2, 0.0)]
    pixels = [(x, y) for x in np.linspace(0, WIDTH, 9) for y in np.linspace(0, HEIGHT, 9)]
    pixels += [(float(rng.uniform(0, WIDTH)), float(rng.uniform(0, HEIGHT))) for _ in range(20)]

    cases = []
    for model in MODELS:
        param_sets = [[float(p) for p in params_for(model, rng)] for _ in range(2)]
        if model == "FOV":
            # omega^2 < 1e-4 selects the small-omega Taylor branch of FOV's Distortion and
            # Undistortion; the random sets above cover the general one. Its own generator,
            # so the other models' random parameters do not shift.
            small = focal_pp(np.random.default_rng(1001), 2) + [5e-4]
            param_sets.append([float(p) for p in small])
        for params in param_sets:
            camera = pycolmap.Camera(model=model, width=WIDTH, height=HEIGHT, params=params)
            # Pixels at and next to the principal point, where the models' small-radius
            # branches run (FOV's radius^2 < 1e-4 Taylor branch, the fisheye r > eps guards).
            pp_pixels = []
            if camera.principal_point_idxs():
                cx, cy = (params[i] for i in camera.principal_point_idxs())
                pp_pixels = [(cx, cy), (cx + 2.0, cy - 3.0)]
            img_from_cam = [opt(camera.img_from_cam(np.array(p, dtype=np.float64))) for p in cam_points]
            img_from_cam_back = [
                opt(camera.img_from_cam(np.array(p, dtype=np.float64), check_cheirality=False))
                for p in back_points]
            cam_from_img = [opt(camera.cam_from_img(np.array(p, dtype=np.float64))) for p in pixels]
            cam_ray_from_img = [opt(camera.cam_ray_from_img(np.array(p, dtype=np.float64))) for p in pixels]
            cases.append({
                "model": model,
                "params": params,
                "threshold": float(camera.cam_from_img_threshold(1.5)),
                "img_from_cam": img_from_cam,
                "img_from_cam_back": img_from_cam_back,
                "cam_from_img": cam_from_img,
                "cam_ray_from_img": cam_ray_from_img,
                "near_axis_img_from_cam": [
                    opt(camera.img_from_cam(np.array(p, dtype=np.float64))) for p in NEAR_AXIS_POINTS],
                "pp_pixels": [list(p) for p in pp_pixels],
                "pp_cam_from_img": [opt(camera.cam_from_img(np.array(p, dtype=np.float64))) for p in pp_pixels],
                "pp_cam_ray_from_img": [
                    opt(camera.cam_ray_from_img(np.array(p, dtype=np.float64))) for p in pp_pixels],
            })

    fixture = {
        "pycolmap_version": pycolmap.__version__,
        "cam_points": [list(map(float, p)) for p in cam_points],
        "back_points": [list(map(float, p)) for p in back_points],
        "pixels": [list(map(float, p)) for p in pixels],
        "near_axis_points": [list(p) for p in NEAR_AXIS_POINTS],
        "cases": cases,
    }
    OUT.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
    print(f"wrote {OUT} ({len(cases)} cases)")
    print_contraction_evidence(fixture)


def print_contraction_evidence(fixture):
    """Evidence for docs/CPP_DIVERGENCES.md entry 100: the wheel's last-ulp
    differences from colmap-rust are FMA contraction. Re-derive two outputs with
    colmap-rust's plain formula and with the multiply-adds fused, and count matches."""
    for case in fixture["cases"]:
        if case["model"] == "SIMPLE_RADIAL":
            f, c1, _, k = case["params"]
            plain = fused = 0
            for (u, v, w), expected in zip(fixture["cam_points"], case["img_from_cam"]):
                uu, vv = u / w, v / w
                radial = k * (uu * uu + vv * vv)
                plain += (f * (uu + uu * radial) + c1) == expected[0]
                # Only `f * x + c1` is one C++ statement with a multiply-add; du = u * radial
                # and x = uu + du are separate statements, which clang does not contract.
                fused += math.fma(f, uu + uu * radial, c1) == expected[0]
            print(f"SIMPLE_RADIAL img_from_cam x: plain {plain}, fused {fused} of {len(case['img_from_cam'])}")
        if case["model"] == "PINHOLE":
            f1, f2, c1, c2 = case["params"]
            plain = fused = 0
            for (x, y), expected in zip(fixture["pixels"], case["cam_ray_from_img"]):
                u, v = (x - c1) / f1, (y - c2) / f2
                plain += 1.0 / math.sqrt(u * u + v * v + 1.0) == expected[2]
                fused += 1.0 / math.sqrt(math.fma(u, u, v * v) + 1.0) == expected[2]
            print(f"PINHOLE cam_ray_from_img z: plain {plain}, fused {fused} of {len(case['cam_ray_from_img'])}")


if __name__ == "__main__":
    main()
