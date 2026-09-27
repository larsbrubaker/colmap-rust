#!/usr/bin/env python3
# linear_algebra_dense.py: numpy (LAPACK) reference results for the dense decompositions in
# colmap-rust/src/linalg, for colmap-rust/tests/linalg/rust_only_decomposition_oracle.rs
# (copied from colmap-sharp's oracle/, which checks the same fixture in its
# DecompositionOracleTests.cs).
#
# This is a pure linear-algebra check, so numpy is the oracle rather than pycolmap (no
# pycolmap binding exposes Eigen's decompositions directly). numpy's qr is LAPACK
# dgeqrf + dorgqr, whose Householder sign convention (dlarfg) is the one Eigen documents
# and colmap-rust follows, so Q and R are compared entry by entry, signs included.
# solve/det/inv go through LAPACK's partial-pivot LU (dgesv/dgetrf) and cholesky through
# dpotrf; the Rust LU, LLT and LDLT are compared on those outputs.
# All matrices are stored column-major (Eigen's and MatrixXd's memory order).
#
# Usage: oracle/.venv/bin/python oracle/linear_algebra_dense.py
# Writes colmap-rust/tests/data/oracle/linear_algebra_dense.json.

import json
import pathlib

import numpy as np

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / "colmap-rust" / "tests" / "data" / "oracle" / "linear_algebra_dense.json")


def cm(matrix):
    """Column-major flat list of a 2-D array (round-trip float reprs)."""
    return [float(v) for v in np.asarray(matrix, dtype=np.float64).ravel(order="F")]


def floats(vector):
    return [float(v) for v in np.asarray(vector, dtype=np.float64).ravel()]


def main():
    rng = np.random.default_rng(20260926)

    qr_cases = []
    # Square, tall (the 8-point / DLT shapes), wide, and a single column (gravity basis).
    for rows, cols in [(4, 4), (6, 6), (9, 6), (12, 9), (9, 8), (3, 5), (3, 1)]:
        a = rng.uniform(-1, 1, size=(rows, cols))
        q, r = np.linalg.qr(a, mode="complete")
        case = {"rows": rows, "cols": cols, "a": cm(a), "q": cm(q), "r": cm(r)}
        if rows >= cols:
            b = rng.uniform(-1, 1, size=rows)
            case["b"] = floats(b)
            case["x"] = floats(np.linalg.lstsq(a, b, rcond=None)[0])
        qr_cases.append(case)

    lu_cases = []
    for n in [2, 3, 5, 8, 10]:
        a = rng.uniform(-1, 1, size=(n, n))
        b = rng.uniform(-1, 1, size=n)
        lu_cases.append({
            "n": n, "a": cm(a), "b": floats(b),
            "x": floats(np.linalg.solve(a, b)),
            "det": float(np.linalg.det(a)), "inverse": cm(np.linalg.inv(a)),
        })

    spd_cases = []
    for n in [3, 6, 9]:
        m = rng.uniform(-1, 1, size=(n + 2, n))
        a = m.T @ m + 0.1 * np.eye(n)
        b = rng.uniform(-1, 1, size=n)
        spd_cases.append({
            "n": n, "a": cm(a), "b": floats(b),
            "l": cm(np.linalg.cholesky(a)), "x": floats(np.linalg.solve(a, b)),
        })

    OUT.write_text(json.dumps({
        "numpy": np.__version__, "qr": qr_cases, "lu": lu_cases, "spd": spd_cases,
    }, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
