#!/usr/bin/env python3
# fixture_sift.py: writes colmap-rust/tests/data/oracle/sift.json, pycolmap's CPU SIFT
# (COLMAP's SiftCPUFeatureExtractor over VLFeat) on small deterministic grey images: the
# white-square image of sift_test.cc and a seeded, smoothed-noise texture, under several
# SiftExtractionOptions (normalization, upright, max_num_features truncation, first_octave
# -2/0/1 for the upsample, copy and downsample paths). Keypoints are written as their six
# float32 shape values (exact as doubles), descriptors as base64 uint8 rows.
# Its images are read by colmap-rust/tests/feature/rust_only_vl_sift_filter.rs. Ported from
# colmap-sharp's oracle/fixture_sift.py (same harness, same output), where SiftOracleTests.cs
# also compares the extractor's keypoints and descriptors (colmap-sharp divergence 41 has the
# arm64 FMA-contraction caveat).
#
# It also compiles oracle/sift_harness.c against cpp-reference's VLFeat with
# -ffp-contract=off and writes sift_vlfeat.json: VLFeat's raw keypoints, orientations and
# float descriptors for the same images, unfused (see sift_harness.c for why), read by
# colmap-rust/tests/feature/rust_only_vl_sift_filter.rs, which requires them bit for bit.
#
# Must run on macOS arm64 (the wheel the fixtures describe), after scripts/fetch-reference.sh.
# Usage: oracle/.venv/bin/python oracle/fixture_sift.py

import base64
import json
import os
import pathlib
import subprocess
import tempfile

import numpy as np
import pycolmap

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "colmap-rust" / "tests" / "data" / "oracle" / "sift.json"
OUTPUT_VLFEAT = ROOT / "colmap-rust" / "tests" / "data" / "oracle" / "sift_vlfeat.json"
# COLMAP_REFERENCE overrides where cpp-reference/ lives (a git worktree has none of its own).
VLFEAT = pathlib.Path(os.environ.get("COLMAP_REFERENCE", ROOT / "cpp-reference")) / "src" / "thirdparty" / "VLFeat"

# (name, image, o_min, noctaves, upright) for the raw VLFeat harness.
VLFEAT_CASES = [
    ("Square256", "square", -1, 4, 0),
    ("Texture", "texture", -1, 4, 0),
    ("TextureUpright", "texture", -1, 4, 1),
    ("TextureFirstOctave0", "texture", 0, 4, 0),
    ("TextureFirstOctave1", "texture", 1, 2, 0),
    ("TextureFirstOctaveMinus2", "texture", -2, 3, 0),
]


def build_harness(workdir):
    exe = workdir / "sift_harness"
    sources = [str(VLFEAT / f) for f in ("generic.c", "host.c", "mathop.c", "imopv.c", "sift.c", "random.c",
                                                   "scalespace.c", "covdet.c", "stringop.c")]
    subprocess.run(["clang", "-O2", "-w", "-ffp-contract=off", "-DVL_DISABLE_SSE2", "-DVL_DISABLE_AVX",
                    "-DVL_DISABLE_OPENMP", "-I", str(VLFEAT), "-o", str(exe),
                    str(ROOT / "oracle" / "sift_harness.c"), *sources], check=True)
    return exe


def run_harness(exe, workdir, img, o_min, noctaves, upright):
    raw = workdir / "image.raw"
    img.tofile(raw)
    out = subprocess.run([str(exe), str(raw), str(img.shape[1]), str(img.shape[0]), str(o_min),
                          str(noctaves), str(upright)], check=True, capture_output=True, text=True).stdout
    keypoints = []
    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "K":
            ints = [int(v) for v in parts[1:5]]
            floats = [float.fromhex(v) for v in parts[5:9]]
            keypoints.append({"k": ints + floats, "a": []})
        else:
            # Angle as an exact double, the descriptor as base64 little-endian float32 bytes.
            descriptor = np.array([float.fromhex(v) for v in parts[2:]], dtype="<f4")
            keypoints[-1]["a"].append({"angle": float.fromhex(parts[1]),
                                       "descriptor": base64.b64encode(descriptor.tobytes()).decode()})
    return keypoints


def square_image(size):
    img = np.zeros((size, size), np.uint8)
    lo, hi = size // 2 - size // 8, size // 2 + size // 8
    img[lo:hi, lo:hi] = 255
    return img


def texture_image(width, height, seed):
    rng = np.random.default_rng(seed)
    noise = rng.random((height // 8 + 2, width // 8 + 2))
    # Bilinear upsampling of coarse noise gives blob-like structure SIFT responds to.
    ys = np.linspace(0, noise.shape[0] - 1.001, height)
    xs = np.linspace(0, noise.shape[1] - 1.001, width)
    y0, x0 = ys.astype(int), xs.astype(int)
    fy, fx = (ys - y0)[:, None], (xs - x0)[None, :]
    a = noise[y0][:, x0]
    b = noise[y0][:, x0 + 1]
    c = noise[y0 + 1][:, x0]
    d = noise[y0 + 1][:, x0 + 1]
    img = a * (1 - fx) * (1 - fy) + b * fx * (1 - fy) + c * (1 - fx) * fy + d * fx * fy
    img = img * 200 + rng.random((height, width)) * 40
    return np.clip(img, 0, 255).astype(np.uint8)


CASES = [
    ("Square256", "square", {}),
    ("Texture", "texture", {}),
    ("TextureL2", "texture", {"normalization": pycolmap.Normalization.L2}),
    ("TextureUpright", "texture", {"upright": True}),
    ("TextureMaxFeatures", "texture", {"max_num_features": 40}),
    ("TextureFirstOctave0", "texture", {"first_octave": 0}),
    ("TextureFirstOctave1", "texture", {"first_octave": 1, "num_octaves": 2}),
    ("TextureFirstOctaveMinus2", "texture", {"first_octave": -2, "num_octaves": 3}),
    ("TextureOrientations4", "texture", {"max_num_orientations": 4, "peak_threshold": 0.01}),
]


def main():
    images = {"square": square_image(256), "texture": texture_image(157, 118, 7)}
    cases = []
    for name, image_name, overrides in CASES:
        options = pycolmap.FeatureExtractionOptions()
        options.use_gpu = False
        for key, value in overrides.items():
            setattr(options.sift, key, value)
        extractor = pycolmap.FeatureExtractor.create(options)
        keypoints, descriptors = extractor.extract_from_uint8_array(images[image_name])
        cases.append({
            "name": name,
            "image": image_name,
            "options": {k: (int(v.value) if hasattr(v, "value") else v) for k, v in overrides.items()},
            "keypoints": [[float(k.x), float(k.y), float(k.a11), float(k.a12), float(k.a21), float(k.a22)]
                          for k in keypoints],
            "descriptors": base64.b64encode(np.ascontiguousarray(descriptors.data).tobytes()).decode(),
        })
        print(name, len(keypoints))
    out = {
        "images": {name: {"width": int(img.shape[1]), "height": int(img.shape[0]),
                          "pixels": base64.b64encode(img.tobytes()).decode()}
                   for name, img in images.items()},
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(out) + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        exe = build_harness(workdir)
        vlfeat_cases = []
        for name, image_name, o_min, noctaves, upright in VLFEAT_CASES:
            keypoints = run_harness(exe, workdir, images[image_name], o_min, noctaves, upright)
            vlfeat_cases.append({"name": name, "image": image_name, "o_min": o_min, "noctaves": noctaves,
                                 "upright": bool(upright), "keypoints": keypoints})
            print("vlfeat", name, len(keypoints))
    OUTPUT_VLFEAT.write_text(json.dumps({"cases": vlfeat_cases}) + "\n")


if __name__ == "__main__":
    main()
