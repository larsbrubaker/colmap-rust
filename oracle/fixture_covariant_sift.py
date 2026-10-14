#!/usr/bin/env python3
# fixture_covariant_sift.py: fixtures for COLMAP's CovariantSiftCPUFeatureExtractor and VLFeat's
# covdet (colmap-rust/src/feature/vlfeat/covdet*.rs), on the images of fixture_sift.py (the
# sift_test.cc white square and the seeded texture). Ported from colmap-sharp's
# oracle/fixture_covariant_sift.py (same harness, same output).
#
# covariant_sift_vlfeat.json: oracle/sift_harness.c in covdet mode, compiled against
# cpp-reference's VLFeat with -ffp-contract=off (see sift_harness.c for why): every covdet
# feature (frame, octave, level, scores) in covdet's order, after affine adaptation and
# orientation assignment as configured, with the raw descriptor of each DSP scale. Read by
# colmap-rust/tests/feature/rust_only_vl_covdet.rs, which requires every value bit for bit.
#
# covariant_sift.json: pycolmap's covariant extractor (the macOS arm64 wheel, whose VLFeat
# fuses multiply-adds, colmap-sharp divergence 41) on the same images; keypoints as their six
# float32 shape values, descriptors as base64 uint8 rows. pycolmap does not bind
# force_covariant_extractor, so the plain covariant case is only in the harness fixture. For
# the covariant extractor's oracle test (colmap-sharp's CovariantSiftOracleTests.cs).
#
# Must run on macOS arm64, after scripts/fetch-reference.sh.
# Usage: oracle/.venv/bin/python oracle/fixture_covariant_sift.py

import base64
import json
import pathlib
import subprocess
import tempfile

import numpy as np
import pycolmap

from fixture_sift import ROOT, build_harness, square_image, texture_image

OUTPUT = ROOT / "colmap-rust" / "tests" / "data" / "oracle" / "covariant_sift.json"
OUTPUT_VLFEAT = ROOT / "colmap-rust" / "tests" / "data" / "oracle" / "covariant_sift_vlfeat.json"

# (name, image, first_octave, affine, upright, dsp) for the harness.
VLFEAT_CASES = [
    ("Square256", "square", -1, 0, 0, 0),
    ("Square256Affine", "square", -1, 1, 0, 0),
    ("Square256AffineUpright", "square", -1, 1, 1, 0),
    ("Square256Dsp", "square", -1, 0, 0, 1),
    ("Texture", "texture", -1, 0, 0, 0),
    ("TextureAffine", "texture", -1, 1, 0, 0),
    ("TextureUpright", "texture", -1, 0, 1, 0),
    ("TextureFirstOctave0", "texture", 0, 1, 0, 0),
]

# (name, image, sift option overrides) for pycolmap.
CASES = [
    ("Square256Affine", "square", {"estimate_affine_shape": True}),
    ("Square256AffineUpright", "square", {"estimate_affine_shape": True, "upright": True}),
    ("Square256Dsp", "square", {"domain_size_pooling": True}),
    ("Square256AffineDsp", "square", {"estimate_affine_shape": True, "domain_size_pooling": True}),
    ("TextureAffine", "texture", {"estimate_affine_shape": True}),
    ("TextureAffineL2", "texture", {"estimate_affine_shape": True, "normalization": pycolmap.Normalization.L2}),
    ("TextureDsp", "texture", {"domain_size_pooling": True}),
    ("TextureAffineMaxFeatures", "texture", {"estimate_affine_shape": True, "max_num_features": 20}),
]


def run_harness(exe, workdir, img, first_octave, affine, upright, dsp):
    raw = workdir / "image.raw"
    img.tofile(raw)
    out = subprocess.run([str(exe), "covdet", str(raw), str(img.shape[1]), str(img.shape[0]), str(first_octave),
                          str(affine), str(upright), str(dsp)], check=True, capture_output=True, text=True).stdout
    features = []
    for line in out.splitlines():
        parts = line.split()
        if parts[0] == "F":
            features.append({"os": [int(parts[1]), int(parts[2])],
                             "values": [float.fromhex(v) for v in parts[3:]], "d": []})
        else:
            descriptor = np.array([float.fromhex(v) for v in parts[1:]], dtype="<f4")
            features[-1]["d"].append(base64.b64encode(descriptor.tobytes()).decode())
    return features


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
    OUTPUT.write_text(json.dumps({"cases": cases}) + "\n")

    with tempfile.TemporaryDirectory() as tmp:
        workdir = pathlib.Path(tmp)
        exe = build_harness(workdir)
        vlfeat_cases = []
        for name, image_name, first_octave, affine, upright, dsp in VLFEAT_CASES:
            features = run_harness(exe, workdir, images[image_name], first_octave, affine, upright, dsp)
            vlfeat_cases.append({"name": name, "image": image_name, "first_octave": first_octave,
                                 "affine": bool(affine), "upright": bool(upright), "dsp": bool(dsp),
                                 "features": features})
            print("vlfeat", name, len(features))
    OUTPUT_VLFEAT.write_text(json.dumps({"cases": vlfeat_cases}) + "\n")


if __name__ == "__main__":
    main()
