#!/usr/bin/env python3
# linear_algebra_spectral.py: numpy (LAPACK) reference results for the SVD and eigen
# solvers in colmap-rust/src/linalg (JacobiSvd, Svd3d/Svd4d, SelfAdjointEigenSolver,
# EigenSolver, FullPivLu rank), for colmap-rust/tests/linalg/rust_only_spectral_oracle.rs
# (copied from colmap-sharp's oracle/, which checks the same fixture in its
# SpectralOracleTests.cs; only the output path differs).
#
# numpy is the oracle because no pycolmap binding exposes Eigen's decompositions. Sources:
# svd = LAPACK dgesdd, eigh = dsyevd, eig = dgeev, pinv = SVD pseudo-inverse. Singular
# vectors and eigenvectors are only defined up to sign (up to a complex phase for eig), so
# the Rust test compares them that way, and only where the value is simple. Ranks use
# Eigen's documented rule (strictly greater than min(rows, cols) * eps * the largest
# singular value), not numpy's matrix_rank default. All matrices are column-major. The
# fixture records every double as a JSON float (Python's repr round-trips exactly).
#
# Usage: oracle/.venv/bin/python oracle/linear_algebra_spectral.py
# Writes colmap-rust/tests/data/oracle/linear_algebra_spectral.json.

import json
import pathlib

import numpy as np

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / "colmap-rust" / "tests" / "data" / "oracle" / "linear_algebra_spectral.json")
EPS = np.finfo(np.float64).eps


def cm(matrix):
    return [float(v) for v in np.asarray(matrix, dtype=np.float64).ravel(order="F")]


def floats(vector):
    return [float(v) for v in np.asarray(vector, dtype=np.float64).ravel()]


def random_orthogonal(rng, n):
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))


def with_singular_values(rng, rows, cols, values):
    u = random_orthogonal(rng, rows)[:, :len(values)]
    v = random_orthogonal(rng, cols)[:, :len(values)]
    return u @ np.diag(values) @ v.T


def svd_case(name, a, rng):
    rows, cols = a.shape
    u, s, vt = np.linalg.svd(a)
    rank = int(np.sum(s > max(1, min(rows, cols)) * EPS * s[0])) if s[0] > 0 else 0
    b = rng.uniform(-1, 1, size=rows)
    cutoff = max(1, min(rows, cols)) * EPS
    x = np.linalg.pinv(a, rcond=cutoff) @ b
    return {"name": name, "rows": rows, "cols": cols, "a": cm(a), "s": floats(s),
            "u": cm(u), "v": cm(vt.T), "rank": rank, "b": floats(b), "x": floats(x)}


def main():
    rng = np.random.default_rng(20260926)

    svd_cases = []
    # The shapes COLMAP decomposes: 3x3, 4x4, Nx9 (8-point / homography DLT), 8x9 (minimal
    # 8-point, rank 8), Nx6 (affine), 6x3..6x5 (EPnP), 12x12 (DLT pose), 4xN (quaternion
    # average, full U), plus square 9x9.
    for rows, cols in [(3, 3), (4, 4), (9, 9), (20, 9), (8, 9), (15, 6), (6, 3), (6, 4),
                       (6, 5), (12, 12), (4, 30), (3, 7)]:
        svd_cases.append(svd_case(f"random {rows}x{cols}", rng.uniform(-1, 1, size=(rows, cols)), rng))
    # Rank-deficient: an essential-like 3x3 (1, 1, 0), a rank-5 9x9, a rank-7 20x9.
    svd_cases.append(svd_case("essential 3x3", with_singular_values(rng, 3, 3, [1.0, 1.0]), rng))
    svd_cases.append(svd_case("rank 5 9x9", with_singular_values(rng, 9, 9, [3, 2.5, 2, 1, 0.5]), rng))
    svd_cases.append(svd_case("rank 7 20x9", with_singular_values(rng, 20, 9, [5, 4, 3, 2, 1, 0.5, 0.25]), rng))
    # Repeated singular values: (2, 2, 2, 1) in 4x4 and (3, 3, 1) in 3x3.
    svd_cases.append(svd_case("repeated 4x4", with_singular_values(rng, 4, 4, [2, 2, 2, 1]), rng))
    svd_cases.append(svd_case("repeated 3x3", with_singular_values(rng, 3, 3, [3, 3, 1]), rng))
    svd_cases.append(svd_case("zero 3x3", np.zeros((3, 3)), rng))

    sym_cases = []
    for n, values in [(4, None), (6, None), (4, [1.0, 1.0, 2.0, 5.0]), (3, [-2.0, 0.0, 3.0])]:
        if values is None:
            m = rng.uniform(-1, 1, size=(n, n))
            a = m + m.T
        else:
            q = random_orthogonal(rng, n)
            a = q @ np.diag(values) @ q.T
            a = 0.5 * (a + a.T)
        w, v = np.linalg.eigh(a)
        sym_cases.append({"n": n, "a": cm(a), "values": floats(w), "vectors": cm(v)})

    general_cases = []
    companions = [[10, -5, 3, -3, 1], [1, -6, 11, -6], [2, 0, 0, 0, 0, -1], [1, 2, 3, 4, 5, 6, 7]]
    matrices = [rng.uniform(-1, 1, size=(n, n)) for n in (3, 4, 4, 6, 8)]
    for coeffs in companions:
        coeffs = np.asarray(coeffs, dtype=np.float64)
        d = len(coeffs) - 1
        c = np.zeros((d, d))
        c[0, :] = -coeffs[1:] / coeffs[0]
        for i in range(1, d):
            c[i, i - 1] = 1
        matrices.append(c)
    matrices.append(np.array([[2.0, 1, 0, 0], [0, 2, 0, 0], [0, 0, 3, 1], [0, 0, -1, 3]]).T)
    for a in matrices:
        w, v = np.linalg.eig(a)
        general_cases.append({"n": a.shape[0], "a": cm(a),
                              "values_re": floats(w.real), "values_im": floats(w.imag),
                              "vectors_re": cm(v.real), "vectors_im": cm(v.imag)})

    rank_cases = []
    base = rng.uniform(-1, 1, size=(3, 10))
    rank_cases.append({"rows": 3, "cols": 10, "a": cm(base), "rank": 3})
    two = np.outer(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 6)) + np.outer(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 6))
    rank_cases.append({"rows": 3, "cols": 6, "a": cm(two), "rank": 2})
    one = np.outer(rng.uniform(-1, 1, 3), rng.uniform(-1, 1, 5))
    rank_cases.append({"rows": 3, "cols": 5, "a": cm(one), "rank": 1})
    rank_cases.append({"rows": 3, "cols": 4, "a": cm(np.zeros((3, 4))), "rank": 0})

    OUT.write_text(json.dumps({
        "numpy": np.__version__, "svd": svd_cases, "symmetric": sym_cases,
        "general": general_cases, "full_piv_lu_rank": rank_cases,
    }, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
