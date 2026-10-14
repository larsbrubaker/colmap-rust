#!/usr/bin/env python3
# reconstruction_io.py: writes colmap-rust/tests/data/oracle/reconstruction_io/, the
# oracle for colmap_rust::scene's reconstruction readers and writers (Tier A: the Rust port
# must read these files back to the same model and re-write them byte for byte). Read by
# colmap-rust/tests/scene/rust_only_reconstruction_io_oracle.rs. Ported from colmap-sharp's
# oracle/reconstruction_io.py (same models, same output).
#
# Two models, each written with pycolmap's write_binary and write_text into <name>/binary and
# <name>/text (not bin/, which .gitignore excludes):
# - rig2cam: synthesize_dataset with 2 rigs of 2 cameras (so rigs carry sensor_from_rig
#   poses and frames carry two data ids), 3 frames per rig, 2D points without a 3D point
#   (written as -1 / kInvalidPoint3DId), plus one extra 3D point with an empty track and a
#   color, which exercises the trailing-space edge of points3D.txt.
# - rig1cam: 3 rigs of 1 camera; its cameras/images/points3D files double as a legacy
#   (pre-rig) model, which the readers must expand to the same rigs and frames.
# manifest.json records summary numbers and a few values read through pycolmap's own
# accessors, so the Rust test checks what it reads against pycolmap, not only against itself.
#
# Usage: oracle/.venv/bin/python oracle/reconstruction_io.py

import json
import pathlib
import shutil

import numpy as np
import pycolmap

HERE = pathlib.Path(__file__).resolve().parent
OUTPUT = HERE.parent / "colmap-rust" / "tests" / "data" / "oracle" / "reconstruction_io"


def build(num_rigs, num_cameras_per_rig, num_frames_per_rig, num_points3D, extra_point):
    pycolmap.set_random_seed(7)
    options = pycolmap.SyntheticDatasetOptions(
        num_rigs=num_rigs,
        num_cameras_per_rig=num_cameras_per_rig,
        num_frames_per_rig=num_frames_per_rig,
        num_points3D=num_points3D,
        num_points2D_without_point3D=3,
    )
    reconstruction = pycolmap.synthesize_dataset(options)
    if extra_point:
        reconstruction.add_point3D(
            np.array([0.25, -1.5, 3.0e-7]), pycolmap.Track(), np.array([10, 200, 255], dtype=np.uint8))
    return reconstruction


def describe(reconstruction):
    first_image = reconstruction.image(min(reconstruction.images.keys()))
    cam_from_world = first_image.cam_from_world()
    point_ids = sorted(reconstruction.points3D.keys())
    first_point = reconstruction.point3D(point_ids[0])
    last_point = reconstruction.point3D(point_ids[-1])
    first_camera = reconstruction.camera(min(reconstruction.cameras.keys()))

    def point(p):
        return {
            "xyz": [float(v) for v in p.xyz],
            "color": [int(v) for v in p.color],
            "error": float(p.error),
            "track": [[int(e.image_id), int(e.point2D_idx)] for e in p.track.elements],
        }

    return {
        "num_rigs": reconstruction.num_rigs(),
        "num_cameras": reconstruction.num_cameras(),
        "num_frames": reconstruction.num_frames(),
        "num_reg_frames": reconstruction.num_reg_frames(),
        "num_images": reconstruction.num_images(),
        "num_reg_images": reconstruction.num_reg_images(),
        "num_points3D": reconstruction.num_points3D(),
        "first_camera": {
            "camera_id": int(first_camera.camera_id),
            "model": first_camera.model.name,
            "width": int(first_camera.width),
            "height": int(first_camera.height),
            "params": [float(v) for v in first_camera.params],
        },
        "first_image": {
            "image_id": int(first_image.image_id),
            "name": first_image.name,
            "camera_id": int(first_image.camera_id),
            "frame_id": int(first_image.frame_id),
            "qwxyz": [float(cam_from_world.rotation.quat[3])] + [float(v) for v in cam_from_world.rotation.quat[:3]],
            "t": [float(v) for v in cam_from_world.translation],
            "num_points2D": int(first_image.num_points2D()),
            "num_points3D": int(first_image.num_points3D),
        },
        "first_point3D": {"id": int(point_ids[0]), **point(first_point)},
        "last_point3D": {"id": int(point_ids[-1]), **point(last_point)},
    }


def write(name, reconstruction):
    root = OUTPUT / name
    for fmt, writer in (("bin", reconstruction.write_binary), ("txt", reconstruction.write_text)):
        out = root / {"bin": "binary", "txt": "text"}[fmt]
        out.mkdir(parents=True, exist_ok=True)
        writer(str(out))
    return describe(reconstruction)


def main():
    if OUTPUT.exists():
        shutil.rmtree(OUTPUT)
    manifest = {
        "pycolmap_version": pycolmap.__version__,
        "rig2cam": write("rig2cam", build(2, 2, 3, 12, extra_point=True)),
        "rig1cam": write("rig1cam", build(3, 1, 2, 8, extra_point=False)),
    }
    # The tests compare bytes, so git must never convert the text files' line endings.
    (OUTPUT / ".gitattributes").write_text("# Byte-exact fixtures: never convert line endings.\n* -text\n")
    with open(OUTPUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
        f.write("\n")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
