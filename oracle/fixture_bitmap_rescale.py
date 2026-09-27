#!/usr/bin/env python3
# fixture_bitmap_rescale.py: writes colmap-rust/tests/data/oracle/bitmap_rescale.json (copied
# from colmap-sharp's oracle/, whose ColmapSharp.Tests checks the same fixture),
# pycolmap's Bitmap.rescale (OpenImageIO resize) with the default BILINEAR filter on small
# seeded images, grey and RGB, up and down. BOX is left out: it differs from OIIO where a
# source pixel center falls exactly on the box edge (docs/CPP_DIVERGENCES.md, entry 121).
# Read by colmap-rust/tests/sensor/rust_only_bitmap_rescale_oracle.rs, which checks the
# resampler of colmap-rust/src/sensor/bitmap/resize.rs against it within one gray level
# (Tier B; docs/CPP_DIVERGENCES.md, entry 121).
#
# Usage: oracle/.venv/bin/python oracle/fixture_bitmap_rescale.py

import json
import pathlib

import numpy as np
import pycolmap

OUTPUT = (
    pathlib.Path(__file__).resolve().parent.parent
    / "colmap-rust" / "tests" / "data" / "oracle" / "bitmap_rescale.json"
)

# (width, height, channels, new_width, new_height)
SHAPES = [
    (4, 4, 1, 1, 1),
    (23, 17, 1, 10, 7),
    (23, 17, 3, 10, 7),
    (17, 11, 1, 40, 29),
    (17, 11, 3, 40, 29),
    (30, 20, 1, 13, 20),
    (9, 7, 3, 4, 11),
]


def main():
    rng = np.random.default_rng(42)
    cases = []
    for width, height, channels, new_width, new_height in SHAPES:
        shape = (height, width) if channels == 1 else (height, width, channels)
        pixels = rng.integers(0, 256, size=shape, dtype=np.uint8)
        for name in ("BILINEAR",):
            bitmap = pycolmap.Bitmap.from_array(pixels)
            bitmap.rescale(new_width, new_height, getattr(pycolmap.BitmapRescaleFilter, name))
            cases.append({
                "width": width, "height": height, "channels": channels,
                "new_width": new_width, "new_height": new_height,
                "filter": name.lower(),
                "input": pixels.ravel().tolist(),
                "output": bitmap.to_array().ravel().tolist(),
            })
    OUTPUT.write_text(json.dumps({"pycolmap": pycolmap.__version__, "cases": cases}) + "\n")
    print(f"wrote {len(cases)} cases to {OUTPUT}")


if __name__ == "__main__":
    main()
