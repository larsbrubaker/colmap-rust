// Rust-only tests for colmap_rust::scene::reconstruction_ply: `ExtractColorsForImage`, which
// reconstruction_test.cc does not cover (only black points are colored; a missing image
// returns false), the stream form of `ImportPLY` round-tripping `ConvertToPLY`, and the line
// layout of `ExportNVM` (scene::reconstruction_export).

use std::io::Cursor;

use colmap_rust::linalg::Vector3ub;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::sensor::bitmap::{Bitmap, BitmapColor};
use colmap_rust::util::ply::write_binary_ply_points;

fn synthetic() -> Reconstruction {
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 2,
        num_points3d: 10,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    reconstruction
}

#[test]
fn rust_only_extract_colors_for_image_colors_black_points_only() {
    let mut reconstruction = synthetic();
    let ids = reconstruction.point3d_ids();
    for &id in &ids {
        reconstruction.point3d_mut(id).unwrap().color = Vector3ub::zeros();
    }
    let red = Vector3ub::new(255, 0, 0);
    reconstruction.point3d_mut(ids[0]).unwrap().color = red;

    let image_id = reconstruction.reg_image_ids().unwrap()[0];
    let camera_id = reconstruction.image(image_id).unwrap().camera_id();
    let camera = reconstruction.camera(camera_id).unwrap();
    let mut bitmap = Bitmap::new(camera.width as i32, camera.height as i32, true);
    bitmap.fill(BitmapColor::new(20, 40, 220)).unwrap();

    assert!(!reconstruction
        .extract_colors_for_image(image_id, |_| None)
        .unwrap());
    assert!(reconstruction
        .extract_colors_for_image(image_id, |_| Some(bitmap))
        .unwrap());

    let observed: Vec<_> = reconstruction
        .image(image_id)
        .unwrap()
        .points2d()
        .iter()
        .filter(|p| p.has_point3d())
        .map(|p| p.point3d_id)
        .collect();
    assert!(!observed.is_empty());
    for id in observed {
        let color = reconstruction.point3d(id).unwrap().color;
        if id == ids[0] {
            assert_eq!(color, red);
        } else {
            assert_eq!(color, Vector3ub::new(20, 40, 220));
        }
    }
}

#[test]
fn rust_only_import_ply_stream_round_trips_convert_to_ply() {
    let reconstruction = synthetic();
    let points = reconstruction.convert_to_ply();
    let mut bytes = Vec::new();
    write_binary_ply_points(&mut bytes, &points, false, true).unwrap();

    let mut imported = Reconstruction::new();
    imported.import_ply_stream(&mut Cursor::new(bytes)).unwrap();
    assert_eq!(imported.convert_to_ply(), points);
}

#[test]
fn rust_only_export_nvm_layout() {
    let reconstruction = synthetic();
    let mut out = Vec::new();
    assert!(colmap_rust::scene::reconstruction_export::export_nvm(
        &reconstruction,
        &mut out,
        false
    )
    .unwrap());
    let text = String::from_utf8(out).unwrap();
    let lines: Vec<&str> = text.split('\n').collect();
    // Header, one line per registered image, a blank line, the point count, the points.
    assert_eq!(&lines[..3], &["NVM_V3 ", " ", "2  "]);
    assert_eq!(lines[3].split(' ').count(), 11);
    assert!(lines[3].ends_with(" 0"));
    assert_eq!(lines[5], "");
    assert_eq!(lines[6], "10");
    for (line, point3d) in lines[7..17].iter().zip(reconstruction.points3d().values()) {
        let tokens: Vec<&str> = line.split(' ').collect();
        let num_obs: usize = tokens[6].parse().unwrap();
        assert_eq!(num_obs, point3d.track.length());
        assert_eq!(tokens.len(), 7 + 4 * num_obs);
    }
    assert_eq!(lines[17], "");
}
