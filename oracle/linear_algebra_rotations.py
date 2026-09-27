#!/usr/bin/env python3
# linear_algebra_rotations.py: Eigen quaternion / angle-axis outputs, via pycolmap's
# Rotation3d binding, for colmap-rust/tests/linalg/rust_only_rotation_oracle.rs (copied from
# colmap-sharp's oracle/, which checks the same fixture in its RotationOracleTests.cs).
#
# pycolmap's Rotation3d (src/pycolmap/geometry/eigen.cc, pybind11_extension.h) is a thin
# wrapper over Eigen::Quaterniond, so each call below is one Eigen operation:
#   Rotation3d(xyzw)            coeffs() = xyzw
#   Rotation3d(matrix)          Eigen::Quaterniond(matrix)          -> FromRotationMatrix
#   Rotation3d(axis_angle)      Quaterniond(AngleAxisd(|v|, v.normalized()))
#   a * b                       Quaterniond product
#   q * v                       Quaterniond * Vector3d
#   q.matrix()                  toRotationMatrix()
#   q.norm()                    norm()
#   q.inverse()                 inverse()
#   q.angle()                   AngleAxisd(q).angle()
#   q.angle_to(o)               angularDistance(o)
# The fixture records inputs and outputs as JSON floats (Python's repr round-trips every
# double exactly), so the Rust test can compare bit for bit or within a stated tolerance.
#
# Usage: oracle/.venv/bin/python oracle/linear_algebra_rotations.py
# Writes colmap-rust/tests/data/oracle/linear_algebra_rotations.json.

import json
import pathlib

import numpy as np
import pycolmap

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / "colmap-rust" / "tests" / "data" / "oracle" / "linear_algebra_rotations.json")


def quat(xyzw):
    return pycolmap.Rotation3d(np.asarray(xyzw, dtype=np.float64))


def floats(array):
    return [float(v) for v in np.asarray(array, dtype=np.float64).ravel(order="C")]


def main():
    rng = np.random.default_rng(20260926)
    cases = []

    # Random quaternions: unit ones, plus deliberately non-unit ones (the product,
    # inverse and matrix constructor are defined for any quaternion).
    raw = [rng.normal(size=4) for _ in range(40)]
    raw += [q / np.linalg.norm(q) for q in (rng.normal(size=4) for _ in range(80))]
    # Special rotations that exercise every branch of the matrix -> quaternion
    # conversion: identity, half turns about each axis and a diagonal, w < 0.
    s = np.sqrt(0.5)
    raw += [np.array(v, dtype=np.float64) for v in (
        [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0],
        [s, s, 0, 0], [0, s, s, 0], [s, 0, s, 0], [0.5, 0.5, 0.5, -0.5],
        [0.1, 0.2, 0.3, -0.927361849549570], [1e-9, 0, 0, 1])]
    # Tiny and huge vector parts: AngleAxis(q) special-cases only an exactly zero vector
    # part, and its norm must not underflow (1e-200 squared is 0).
    raw += [np.array(v, dtype=np.float64) for v in (
        [1e-17, 0, 0, 1], [1e-200, 0, 0, 1], [1e-17, -2e-17, 0, -1], [5e-324, 0, 0, 1],
        [1e-160, 1e-160, 0, 1], [3e-162, 4e-162, 0, 0.5], [0, 0, 0, -1], [0, 0, 0, 1e-300])]

    for index, xyzw in enumerate(raw):
        q = quat(xyzw)
        other = quat(raw[(index * 7 + 3) % len(raw)])
        v = rng.normal(size=3) * 10
        matrix = q.matrix()
        # A general (not orthogonal) matrix too, for the branch logic on real input.
        general = matrix + rng.normal(size=(3, 3)) * 0.1
        axis_angle = rng.normal(size=3)
        cases.append({
            "q": floats(xyzw),
            "other": floats(other.quat),
            "v": floats(v),
            "general": floats(general),
            "axis_angle": floats(axis_angle),
            "product": floats((q * other).quat),
            "rotated": floats(q * v),
            "matrix": floats(matrix),
            "from_matrix": floats(pycolmap.Rotation3d(matrix).quat),
            "from_general": floats(pycolmap.Rotation3d(general).quat),
            "from_axis_angle": floats(pycolmap.Rotation3d(axis_angle).quat),
            "norm": float(q.norm()),
            "inverse": floats(q.inverse().quat),
            "angle": float(q.angle()),
            "angle_to": float(q.angle_to(other)),
        })

    # Matrices whose trace is exactly 0, so the positive-trace test (> 0 or >= 0) in
    # Quaterniond(matrix) decides the branch, and the two branches give different bits.
    trace_zero = [
        [[0.5, 0.1, 0.2], [0.3, -0.25, 0.4], [0.6, 0.7, -0.25]],
        [[-0.5, 0.3, -0.2], [0.1, 0.75, 0.9], [-0.4, 0.6, -0.25]],
        [[0.0, -0.8, 0.6], [0.36, 0.48, 0.8], [-0.6, 0.64, -0.48]],
    ]
    trace_cases = []
    for m in trace_zero:
        m = np.array(m, dtype=np.float64)
        assert m[0, 0] + (m[1, 1] + m[2, 2]) == 0 and m[0, 0] + m[1, 1] + m[2, 2] == 0
        trace_cases.append({"matrix": floats(m), "from_matrix": floats(pycolmap.Rotation3d(m).quat)})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "pycolmap_version": pycolmap.__version__,
        "note": "Quaternions are xyzw (Eigen coeffs order); matrices are row-major.",
        "cases": cases,
        "trace_zero_cases": trace_cases,
    }, indent=1) + "\n")
    print(f"wrote {len(cases)} cases to {OUT}")
    explain_divergences(cases)


def explain_divergences(cases):
    """Evidence for docs/CPP_DIVERGENCES.md entries 2 and 3: re-derive the two Tier B
    fields in plain Python (no FMA unless asked) with colmap-rust's formulas and show what
    the C++ must have done differently to produce the fixture's bits."""
    from math import fma, nextafter, sin, cos, sqrt, inf

    def cross(a, b, fused):
        if fused:  # a1*b2 - a2*b1 contracted to fma(a1, b2, -(a2*b1)), as clang does
            return [fma(a[1], b[2], -(a[2] * b[1])), fma(a[2], b[0], -(a[0] * b[2])),
                    fma(a[0], b[1], -(a[1] * b[0]))]
        return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]

    def rotate(c, fused):
        x, y, z, w = c["q"]
        u, v = [x, y, z], c["v"]
        uv = [t + t for t in cross(u, v, fused)]
        uu = cross(u, uv, fused)
        return [(v[i] + w * uv[i]) + uu[i] for i in range(3)]

    for fused in (False, True):
        bad = sum(rotate(c, fused) != c["rotated"] for c in cases)
        print(f"q * v, cross products {'with' if fused else 'without'} FMA: {bad} mismatches")

    def from_axis_angle(c, sine_ulps):
        v = c["axis_angle"]
        n = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
        s = sin(0.5 * n)
        for _ in range(abs(sine_ulps)):
            s = nextafter(s, inf if sine_ulps > 0 else -inf)
        return [s * (t / n) for t in v] + [cos(0.5 * n)]

    exact = [c for c in cases if from_axis_angle(c, 0) == c["from_axis_angle"]]
    off = [c for c in cases if from_axis_angle(c, 0) != c["from_axis_angle"]]
    explained = [c for c in off if any(from_axis_angle(c, k) == c["from_axis_angle"] for k in (-1, 1))]
    print(f"angle-axis -> quaternion: {len(exact)} exact with libm sin, {len(off)} not, "
          f"{len(explained)} of those fixed by moving sin(a/2) one ulp")


if __name__ == "__main__":
    main()
