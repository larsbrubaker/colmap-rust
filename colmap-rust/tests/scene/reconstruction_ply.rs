// Port of colmap/scene/reconstruction_test.cc, part 5, 1:1 (same test names, values and
// checks): the PLY, image-directory and color-extraction tests of
// colmap_rust::scene::reconstruction_ply. `find_if` over the points is `Iterator::find`.
// `ExtractColorsForAllImages` takes its images from the host, so the test's written image
// files are an in-memory name -> bitmap map, and deleting one file is leaving it out.

use std::collections::HashMap;

use colmap_rust::linalg::{Vector3d, Vector3ub};
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::scene::track::Track;
use colmap_rust::sensor::bitmap::{Bitmap, BitmapColor};
use colmap_rust::util::ply::PlyPoint;

use crate::reconstruction::generate_reconstruction;
use crate::reconstruction_io::create_test_dir;

fn assert_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() <= tol, "{a} vs {b}");
}

#[test]
fn reconstruction_convert_to_ply() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    reconstruction
        .add_point3d(
            Vector3d::new(1.0, 2.0, 3.0),
            Track::default(),
            Vector3ub::new(255, 0, 0),
        )
        .unwrap();
    reconstruction
        .add_point3d(
            Vector3d::new(4.0, 5.0, 6.0),
            Track::default(),
            Vector3ub::new(0, 255, 0),
        )
        .unwrap();

    let ply_points = reconstruction.convert_to_ply();
    assert_eq!(ply_points.len(), 2);

    // Decompose into individual EXPECT calls by locating both expected points.
    let find = |x: f64| -> Option<&PlyPoint> {
        ply_points
            .iter()
            .find(|p| (f64::from(p.x) - x).abs() < 1e-6)
    };
    let p1 = find(1.0).expect("point 1");
    let p2 = find(4.0).expect("point 2");

    assert_near(f64::from(p1.y), 2.0, 1e-6);
    assert_near(f64::from(p1.z), 3.0, 1e-6);
    assert_eq!(p1.r, 255);
    assert_eq!(p1.g, 0);
    assert_eq!(p1.b, 0);

    assert_near(f64::from(p2.y), 5.0, 1e-6);
    assert_near(f64::from(p2.z), 6.0, 1e-6);
    assert_eq!(p2.r, 0);
    assert_eq!(p2.g, 255);
    assert_eq!(p2.b, 0);
}

#[test]
fn reconstruction_import_ply_from_vector() {
    let mut reconstruction = Reconstruction::new();
    let mut ply_points = vec![PlyPoint::default(); 2];
    ply_points[0].x = 1.0;
    ply_points[0].y = 2.0;
    ply_points[0].z = 3.0;
    ply_points[0].r = 100;
    ply_points[0].g = 150;
    ply_points[0].b = 200;
    ply_points[1].x = 4.0;
    ply_points[1].y = 5.0;
    ply_points[1].z = 6.0;
    ply_points[1].r = 50;
    ply_points[1].g = 60;
    ply_points[1].b = 70;

    reconstruction.import_ply(&ply_points).unwrap();
    assert_eq!(reconstruction.num_points3d(), 2);

    // Verify the points were imported correctly by locating both expected points.
    let points3d = reconstruction.points3d();
    let find = |x: f64| points3d.values().find(|p| (p.xyz.x - x).abs() < 1e-6);
    let p1 = find(1.0).expect("point 1");
    let p2 = find(4.0).expect("point 2");

    assert_near(p1.xyz.y, 2.0, 1e-6);
    assert_near(p1.xyz.z, 3.0, 1e-6);
    assert_eq!(p1.color.x, 100);
    assert_eq!(p1.color.y, 150);
    assert_eq!(p1.color.z, 200);

    assert_near(p2.xyz.y, 5.0, 1e-6);
    assert_near(p2.xyz.z, 6.0, 1e-6);
    assert_eq!(p2.color.x, 50);
    assert_eq!(p2.color.y, 60);
    assert_eq!(p2.color.z, 70);
}

#[test]
fn reconstruction_create_image_dirs() {
    let mut reconstruction = Reconstruction::new();
    generate_reconstruction(2, &mut reconstruction);
    *reconstruction.image_mut(1).unwrap().name_mut() = "subdir1/image1.jpg".to_string();
    *reconstruction.image_mut(2).unwrap().name_mut() = "subdir2/subdir3/image2.jpg".to_string();

    let test_dir = create_test_dir();
    reconstruction.create_image_dirs(&test_dir).unwrap();

    assert!(test_dir.join("subdir1").exists());
    assert!(test_dir.join("subdir2").join("subdir3").exists());
}

/// The test's `WriteSolidColorImages`: one solid RGB bitmap per registered image, by name.
fn write_solid_color_images(
    reconstruction: &Reconstruction,
    color: BitmapColor<u8>,
) -> HashMap<String, Bitmap> {
    let mut images = HashMap::new();
    for image_id in reconstruction.reg_image_ids().unwrap() {
        let image = reconstruction.image(image_id).unwrap();
        let camera = reconstruction.camera(image.camera_id()).unwrap();
        let mut bitmap = Bitmap::new(camera.width as i32, camera.height as i32, true);
        bitmap.fill(color).unwrap();
        images.insert(image.name().to_string(), bitmap);
    }
    images
}

#[test]
fn reconstruction_extract_colors_for_all_images() {
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 3,
        num_points3d: 10,
        num_points2d_without_point3d: 2,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut reconstruction).unwrap();

    let color = BitmapColor::new(20u8, 40, 220);
    let mut images = write_solid_color_images(&reconstruction, color);

    // Delete one image file so extraction must handle the missing file.
    let first_image_id = reconstruction.reg_image_ids().unwrap()[0];
    let first_name = reconstruction
        .image(first_image_id)
        .unwrap()
        .name()
        .to_string();
    images.remove(&first_name);

    reconstruction
        .extract_colors_for_all_images(|name| images.get(name).cloned())
        .unwrap();

    for point3d_id in reconstruction.point3d_ids() {
        let point_color = reconstruction.point3d(point3d_id).unwrap().color;
        assert_eq!(point_color, Vector3ub::new(color.r, color.g, color.b));
    }
}
