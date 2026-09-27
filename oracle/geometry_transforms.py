#!/usr/bin/env python3
# geometry_transforms.py: Rigid3d, Sim3d and GPSTransform outputs from pycolmap, for
# colmap-rust/tests/geometry/rust_only_transforms_oracle.rs (copied from colmap-sharp's
# oracle/, which checks the same fixture in its GeometryOracleTests.cs).
#
# pycolmap's bindings (src/pycolmap/geometry/rigid3.cc, sim3.cc, gps.cc) call COLMAP
# directly, so each call below is one COLMAP function:
#   Rigid3d(Rotation3d(q), t) / Sim3d(s, Rotation3d(q), t)   the constructors
#   a * b                    operator*(Rigid3d/Sim3d, Rigid3d/Sim3d)   composition
#   a * x (x of shape (3,))  operator*(Rigid3d/Sim3d, Vector3d)        point transform
#   a.inverse()              Inverse(...)
#   a.matrix()               ToMatrix()
#   Rigid3d(m) / Sim3d(m)    FromMatrix(m)   (m of shape (3, 4))
#   r.tgt_origin_in_src()    TgtOriginInSrc()
#   r.adjoint() / r.adjoint_inverse()
#   pycolmap.get_covariance_for_inverse(r, cov)   GetCovarianceForRigid3dInverse
#   pycolmap.get_covariance_for_composed_rigid3d(r, cov12)
#                            GetCovarianceForComposedRigid3d
#   pycolmap.get_covariance_for_relative_rigid3d(r, other, cov12)
#                            GetCovarianceForRelativeRigid3d
#   str(a)                   operator<<
#   GPSTransform(ellipsoid).<conversion>(...)     the GPSTransform member functions
# Sim3d::ToFile is not bound; its expected text is Python's '%.17g', which formats like
# C's printf (correctly rounded), i.e. like the ostream with precision 17 ToFile uses.
#
# Values are written as JSON floats (Python's repr round-trips every double), matrices
# row-major. The fixture is deterministic (numpy default_rng with a fixed seed).
#
# Usage: oracle/.venv/bin/python oracle/geometry_transforms.py
# Writes colmap-rust/tests/data/oracle/geometry_transforms.json.

import json
import math
import pathlib

import numpy as np
import pycolmap

OUT = (pathlib.Path(__file__).resolve().parent.parent
       / "colmap-rust" / "tests" / "data" / "oracle" / "geometry_transforms.json")


def floats(array):
    return [float(v) for v in np.asarray(array, dtype=np.float64).ravel(order="C")]


def scalar(value):
    # Sim3d.scale comes back as a one-element array (a view into params).
    return float(np.asarray(value, dtype=np.float64).ravel()[0])


def unit_quat(rng):
    q = rng.normal(size=4)
    return q / np.linalg.norm(q)


def rigid_case(rng):
    q, t = unit_quat(rng), rng.normal(size=3) * 5
    oq, ot = unit_quat(rng), rng.normal(size=3) * 5
    x = rng.normal(size=3) * 10
    r = pycolmap.Rigid3d(pycolmap.Rotation3d(q), t)
    other = pycolmap.Rigid3d(pycolmap.Rotation3d(oq), ot)
    composed = r * other
    inverse = r.inverse()
    # A slightly non-orthogonal matrix, as FromMatrix gets from estimators.
    general = r.matrix() + np.hstack([rng.normal(size=(3, 3)) * 1e-3, np.zeros((3, 1))])
    a = rng.normal(size=(6, 6))
    cov = a @ a.T
    b = rng.normal(size=(12, 12))
    cov12 = b @ b.T
    return {
        "q": floats(q), "t": floats(t), "other_q": floats(oq), "other_t": floats(ot),
        "x": floats(x), "cov": floats(cov), "general": floats(general),
        "apply": floats(r * x),
        "compose_q": floats(composed.rotation.quat), "compose_t": floats(composed.translation),
        "inverse_q": floats(inverse.rotation.quat), "inverse_t": floats(inverse.translation),
        "matrix": floats(r.matrix()),
        "from_matrix_q": floats(pycolmap.Rigid3d(general).rotation.quat),
        "from_matrix_t": floats(pycolmap.Rigid3d(general).translation),
        "tgt_origin_in_src": floats(r.tgt_origin_in_src()),
        "adjoint": floats(r.adjoint()),
        "adjoint_inverse": floats(r.adjoint_inverse()),
        "cov_inverse": floats(pycolmap.get_covariance_for_inverse(r, cov)),
        "cov12": floats(cov12),
        "cov_composed": floats(pycolmap.get_covariance_for_composed_rigid3d(r, cov12)),
        "cov_relative": floats(pycolmap.get_covariance_for_relative_rigid3d(r, other, cov12)),
        "str": str(r),
    }


def sim_case(rng):
    s, q, t = float(np.exp(rng.normal())), unit_quat(rng), rng.normal(size=3) * 5
    os_, oq, ot = float(np.exp(rng.normal())), unit_quat(rng), rng.normal(size=3) * 5
    x = rng.normal(size=3) * 10
    sim = pycolmap.Sim3d(s, pycolmap.Rotation3d(q), t)
    other = pycolmap.Sim3d(os_, pycolmap.Rotation3d(oq), ot)
    composed = sim * other
    inverse = sim.inverse()
    general = sim.matrix() + np.hstack([rng.normal(size=(3, 3)) * 1e-3, np.zeros((3, 1))])
    from_matrix = pycolmap.Sim3d(general)
    return {
        "s": s, "q": floats(q), "t": floats(t),
        "other_s": os_, "other_q": floats(oq), "other_t": floats(ot),
        "x": floats(x), "general": floats(general),
        "apply": floats(sim * x),
        "compose_s": scalar(composed.scale), "compose_q": floats(composed.rotation.quat),
        "compose_t": floats(composed.translation),
        "inverse_s": scalar(inverse.scale), "inverse_q": floats(inverse.rotation.quat),
        "inverse_t": floats(inverse.translation),
        "matrix": floats(sim.matrix()),
        "from_matrix_s": scalar(from_matrix.scale), "from_matrix_q": floats(from_matrix.rotation.quat),
        "from_matrix_t": floats(from_matrix.translation),
        "str": str(sim),
        "to_file": " ".join("%.17g" % v for v in (s, q[3], q[0], q[1], q[2], t[0], t[1], t[2])),
    }


def gps_case(rng, ellipsoid, center_lat, center_lon, spread):
    gps = pycolmap.GPSTransform(ellipsoid)
    lla = np.column_stack([
        np.clip(center_lat + rng.normal(size=8) * spread, -89.0, 89.0),
        center_lon + rng.normal(size=8) * spread,
        rng.uniform(-100, 5000, size=8)])
    ecef = gps.ellipsoid_to_ecef(lla)
    ref = lla[0]
    enu = gps.ellipsoid_to_enu(lla, ref[0], ref[1], ref[2])
    utm, zone = gps.ellipsoid_to_utm(lla)
    is_north = bool(center_lat > 0)
    return {
        "ellipsoid": ellipsoid.name,
        "lla": floats(lla), "ref": floats(ref), "zone": int(zone), "is_north": is_north,
        "ecef": floats(ecef),
        "ecef_to_ellipsoid": floats(gps.ecef_to_ellipsoid(ecef)),
        "enu": floats(enu),
        "ecef_to_enu": floats(gps.ecef_to_enu(ecef, ecef[0])),
        "enu_to_ellipsoid": floats(gps.enu_to_ellipsoid(enu, ref[0], ref[1], ref[2])),
        "enu_to_ecef": floats(gps.enu_to_ecef(enu, ref[0], ref[1], ref[2])),
        "utm": floats(utm),
        "utm_to_ellipsoid": floats(gps.utm_to_ellipsoid(utm, zone, is_north)),
    }


def print_fma_evidence(gps_cases):
    # docs/CPP_DIVERGENCES.md, entry 6: EllipsoidToECEF's z = (N * (1 - e2) + alt) * sin_lat
    # re-derived without and with the multiply-add fused (math.fma, Python 3.13+), against
    # the wheel. x and y have no multiply-add and match without fusing.
    deg_to_rad = 0.0174532925199432954743716805978692718781530857086181640625
    plain = fused = total = 0
    for case in gps_cases:
        f = 1.0 / (298.257222100882711243162837 if case["ellipsoid"] == "GRS80" else 298.257223563)
        e2 = f * (2.0 - f)
        lla = np.asarray(case["lla"]).reshape(-1, 3)
        ecef = np.asarray(case["ecef"]).reshape(-1, 3)
        for (lat_deg, _, alt), expected in zip(lla, ecef):
            sin_lat = math.sin(float(lat_deg) * deg_to_rad)
            n = 6378137.0 / math.sqrt(1 - e2 * sin_lat * sin_lat)
            plain += (n * (1 - e2) + float(alt)) * sin_lat != expected[2]
            fused += math.fma(n, 1 - e2, float(alt)) * sin_lat != expected[2]
            total += 1
    print(f"EllipsoidToECEF z mismatches vs the wheel: plain {plain}/{total}, fused {fused}/{total}")


def main():
    rng = np.random.default_rng(20260926)
    rigid = [rigid_case(rng) for _ in range(40)]
    sim = [sim_case(rng) for _ in range(40)]
    gps = []
    for ellipsoid in (pycolmap.GPSTransformEllipsoid.GRS80, pycolmap.GPSTransformEllipsoid.WGS84):
        # Munich (gps_test.cc's area), the southern hemisphere, a zone boundary, the
        # far west and near-polar latitudes.
        for lat, lon, spread in ((48.15, 11.57, 0.01), (-33.87, 151.21, 0.05),
                                 (10.0, 6.0, 0.5), (37.77, -122.42, 1.0), (75.0, 20.0, 2.0)):
            gps.append(gps_case(rng, ellipsoid, lat, lon, spread))
    print_fma_evidence(gps)
    OUT.write_text(json.dumps({"rigid": rigid, "sim": sim, "gps": gps}, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
