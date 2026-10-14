// Rust-only tests of COLMAP's model files (colmap_rust::scene::reconstruction_io_binary,
// reconstruction_io_text, reconstruction_io_text_write and reconstruction_files), not ports
// of a *_test.cc: byte-exact agreement with the files pycolmap 4.2.0 wrote
// (tests/data/oracle/reconstruction_io, oracle/reconstruction_io.py; Tier A), the legacy
// (pre-rig) models those files double as, and docs/CPP_DIVERGENCES.md entries 241 and 242.

use std::path::{Path, PathBuf};

use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::reconstruction_io_binary::read_rigs_binary;
use colmap_rust::scene::reconstruction_io_text::{read_cameras_text, read_points3d_text};
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};

use crate::reconstruction_io::create_test_dir;
use crate::reconstruction_synthetic::expect_reconstruction_eq;

const STEMS: [&str; 5] = ["rigs", "cameras", "frames", "images", "points3D"];
const MODELS: [&str; 2] = ["rig2cam", "rig1cam"];

fn model_dir(model: &str, format: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data/oracle/reconstruction_io")
        .join(model)
        .join(format)
}

fn expect_same_files(expected_dir: &Path, actual_dir: &Path, ext: &str) {
    for stem in STEMS {
        let expected = std::fs::read(expected_dir.join(format!("{stem}.{ext}"))).unwrap();
        let actual = std::fs::read(actual_dir.join(format!("{stem}.{ext}"))).unwrap();
        assert!(
            expected == actual,
            "{stem}.{ext} differs from pycolmap's in {}",
            expected_dir.display()
        );
    }
}

#[test]
fn rust_only_reconstruction_io_rewrites_pycolmap_files_byte_for_byte() {
    for model in MODELS {
        for format in ["binary", "text"] {
            let mut reconstruction = Reconstruction::new();
            reconstruction.read(&model_dir(model, format)).unwrap();
            assert!(reconstruction.is_valid());

            let out_dir = create_test_dir();
            reconstruction.write_binary(&out_dir).unwrap();
            reconstruction.write_text(&out_dir).unwrap();
            expect_same_files(&model_dir(model, "binary"), &out_dir, "bin");
            expect_same_files(&model_dir(model, "text"), &out_dir, "txt");
        }
    }
}

#[test]
fn rust_only_reconstruction_io_binary_and_text_read_the_same_model() {
    for model in MODELS {
        let mut from_binary = Reconstruction::new();
        from_binary
            .read_binary(&model_dir(model, "binary"))
            .unwrap();
        let mut from_text = Reconstruction::new();
        from_text.read_text(&model_dir(model, "text")).unwrap();
        expect_reconstruction_eq(&from_binary, &from_text);
        assert_eq!(from_binary.reg_frame_ids(), from_text.reg_frame_ids());
    }
}

#[test]
fn rust_only_reconstruction_io_pycolmap_model_counts() {
    // manifest.json's summary numbers, as pycolmap reported them.
    let mut rig2cam = Reconstruction::new();
    rig2cam.read(&model_dir("rig2cam", "binary")).unwrap();
    assert_eq!(
        [
            rig2cam.num_rigs(),
            rig2cam.num_cameras(),
            rig2cam.num_frames(),
            rig2cam.num_reg_frames(),
            rig2cam.num_images(),
            rig2cam.num_reg_images(),
            rig2cam.num_points3d()
        ],
        [2, 4, 6, 6, 12, 12, 13]
    );
    let last = rig2cam.point3d(13).unwrap();
    assert_eq!(last.error, -1.0);
    assert_eq!(last.track.length(), 0);
    assert_eq!(
        rig2cam.image(1).unwrap().name(),
        "camera000001_frame000000.png"
    );
}

#[test]
fn rust_only_reconstruction_io_legacy_model_gets_trivial_rigs_and_frames() {
    // rig1cam has one camera per rig, so its files without rigs and frames are a legacy
    // model that must expand to the same rigs and frames.
    for (format, ext) in [("binary", "bin"), ("text", "txt")] {
        let legacy_dir = create_test_dir();
        for stem in ["cameras", "images", "points3D"] {
            let name = format!("{stem}.{ext}");
            std::fs::copy(
                model_dir("rig1cam", format).join(&name),
                legacy_dir.join(&name),
            )
            .unwrap();
        }
        let mut full = Reconstruction::new();
        full.read(&model_dir("rig1cam", format)).unwrap();
        let mut legacy = Reconstruction::new();
        legacy.read(&legacy_dir).unwrap();
        expect_reconstruction_eq(&full, &legacy);
    }
}

#[test]
fn rust_only_reconstruction_io_rejects_what_cpp_misreads() {
    // Entry 241: a token is parsed whole ("12abc" is no camera id), a sensor type number
    // that names no enumerator is an error, and a stream that ends early is an error.
    let mut reconstruction = Reconstruction::new();
    let text = b"12abc SIMPLE_PINHOLE 10 10 1 5 5\n";
    assert!(read_cameras_text(&mut reconstruction, &mut text.as_slice()).is_err());

    let mut rigs = Vec::new();
    rigs.extend_from_slice(&1u64.to_le_bytes());
    rigs.extend_from_slice(&1u32.to_le_bytes());
    rigs.extend_from_slice(&1u32.to_le_bytes());
    rigs.extend_from_slice(&7i32.to_le_bytes());
    rigs.extend_from_slice(&1u32.to_le_bytes());
    assert!(read_rigs_binary(&mut Reconstruction::new(), &mut rigs.as_slice()).is_err());
    assert!(read_rigs_binary(&mut Reconstruction::new(), &mut &rigs[..10]).is_err());

    // A point3D_id read through `int64_t` keeps "-1" meaning none, and a point with an
    // untrackable second field stops the track as COLMAP's `while (>> a >> b)` does.
    let mut points = Reconstruction::new();
    read_points3d_text(&mut points, &mut b"5 1 2 3 4 5 6 0.5\n".as_slice()).unwrap();
    assert_eq!(points.point3d(5).unwrap().track.length(), 0);
}

#[test]
fn rust_only_reconstruction_read_resets_registration() {
    // Entry 242: reading into a used reconstruction starts from an empty one; COLMAP keeps
    // the old registered frames and counts.
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_frames_per_rig: 3,
        num_points3d: 5,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    let dir = create_test_dir();
    reconstruction.write_binary(&dir).unwrap();

    // A used reconstruction with five registered frames, two more than the files hold.
    let mut used = Reconstruction::new();
    let five_frames = SyntheticDatasetOptions {
        num_frames_per_rig: 5,
        ..options
    };
    synthesize_dataset(&five_frames, &mut used).unwrap();
    assert_eq!(used.num_reg_frames(), 5);
    used.read(&dir).unwrap();
    assert_eq!(used.num_reg_frames(), 3);
    assert_eq!(used.num_reg_images(), 3);
    assert!(used.is_valid());
}
