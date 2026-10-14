// Port of colmap/scene/reconstruction_pruning_test.cc, 1:1 (same test names, values and
// checks), for colmap_rust::scene::reconstruction_pruning::find_redundant_points3d. Tier A
// (exact). `UnorderedElementsAre` compares sets. COLMAP's gtest main seeds the PRNG with 0
// before every test, so the cases that synthesize call `set_prng_seed(0)` first.

use std::collections::HashSet;

use colmap_rust::linalg::Vector2d;
use colmap_rust::math::fns;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::scene::point2d::Point2D;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::reconstruction_pruning::find_redundant_points3d;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::util::types::Point3DId;

fn as_set(ids: Vec<Point3DId>) -> HashSet<Point3DId> {
    ids.into_iter().collect()
}

#[test]
fn find_redundant_points3d_empty() {
    let reconstruction = Reconstruction::new();
    assert!(find_redundant_points3d(0.0, &reconstruction)
        .unwrap()
        .is_empty());
}

#[test]
fn find_redundant_points3d_varying_coverage_gain() {
    set_prng_seed(0);
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 5,
        num_points3d: 100,
        track_length: 5,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();

    // Give every point the same coverage so the number selected at each
    // threshold is independent of random image-tile occupancy.
    let image_ids: Vec<_> = reconstruction.images().keys().copied().collect();
    for image_id in image_ids {
        let camera_id = reconstruction.image(image_id).unwrap().camera_id();
        let camera = reconstruction.camera(camera_id).unwrap();
        let center = Vector2d::new(camera.width as f64 / 2., camera.height as f64 / 2.);
        let image = reconstruction.image_mut(image_id).unwrap();
        for point2d_idx in 0..image.num_points2d() {
            image.point2d_mut(point2d_idx).xy = center;
        }
    }

    assert!(find_redundant_points3d(0.0, &reconstruction)
        .unwrap()
        .is_empty());
    for (min_coverage_gain, expected_num_redundant_points3d) in
        [(0.1, 92), (0.4, 98), (0.7, 99), (10.0, 100)]
    {
        let redundant_point3d_ids =
            find_redundant_points3d(min_coverage_gain, &reconstruction).unwrap();
        assert_eq!(
            redundant_point3d_ids.len(),
            expected_num_redundant_points3d,
            "min_coverage_gain {min_coverage_gain}"
        );
    }
}

#[test]
fn find_redundant_points3d_varying_track_length() {
    set_prng_seed(0);
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d: 5,
        num_points2d_without_point3d: 0,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();

    let mut expected_redundant_point3d_ids = HashSet::new();
    let point3d_ids: Vec<_> = reconstruction.points3d().keys().copied().collect();
    for point3d_id in point3d_ids {
        let track_el = *reconstruction.point3d(point3d_id).unwrap().track.element(0);
        reconstruction
            .image_mut(track_el.image_id)
            .unwrap()
            .reset_point3d_for_point2d(track_el.point2d_idx);
        reconstruction
            .point3d_mut(point3d_id)
            .unwrap()
            .track
            .delete_element(0)
            .unwrap();
        expected_redundant_point3d_ids.insert(point3d_id);
        assert_eq!(
            as_set(find_redundant_points3d(0.1, &reconstruction).unwrap()),
            expected_redundant_point3d_ids
        );
    }
}

#[test]
fn find_redundant_points3d_varying_spatial_distribution() {
    set_prng_seed(0);
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 4,
        num_frames_per_rig: 1,
        num_points3d: 4,
        num_points2d_without_point3d: 0,
        // Ensure all cameras in the rig have identical poses.
        sensor_from_rig_translation_stddev: 0.0,
        sensor_from_rig_rotation_stddev: 0.0,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();

    // Generate a synthetic dataset where all points are identically distributed
    // in a circle around the camera center, i.e., each of them has the same
    // coverage.
    let width = synthetic_dataset_options.camera_width as f64;
    let height = synthetic_dataset_options.camera_height as f64;
    let point3d_ids: Vec<_> = reconstruction.points3d().keys().copied().collect();
    for point3d_id in point3d_ids {
        let track = reconstruction.point3d(point3d_id).unwrap().track.clone();
        assert_eq!(
            track.length(),
            synthetic_dataset_options.num_cameras_per_rig as usize
        );
        for i in 0..synthetic_dataset_options.num_cameras_per_rig {
            let track_el = track.element(i as usize);
            let angle = std::f64::consts::PI / 4. + f64::from(i) * std::f64::consts::PI / 2.;
            let radius = 0.25 * fns::sqrt(width * width + height * height);
            reconstruction
                .image_mut(track_el.image_id)
                .unwrap()
                .point2d_mut(track_el.point2d_idx)
                .xy = Vector2d::new(
                fns::sin(angle) * radius + width / 2.,
                fns::cos(angle) * radius + height / 2.,
            );
        }
    }

    let point2d_of = |reconstruction: &mut Reconstruction, point3d_id: Point3DId, idx: usize| {
        let track_el = *reconstruction
            .point3d(point3d_id)
            .unwrap()
            .track
            .element(idx);
        let image = reconstruction.image_mut(track_el.image_id).unwrap();
        let point2d = image.point2d_mut(track_el.point2d_idx);
        point2d.xy = point2d.xy * 2.0;
    };

    // Make sure point 3 has the largest coverage and selected first.
    reconstruction
        .point3d_mut(3)
        .unwrap()
        .track
        .add_element_ids(1, 4);
    let point2d = Point2D {
        xy: Vector2d::new(0.0, 0.0),
        point3d_id: 3,
    };
    reconstruction
        .image_mut(1)
        .unwrap()
        .points2d_mut()
        .push(point2d);

    // Make sure point 1 has larger coverage gain by moving it to another
    // tile, while point 2 and 4 have redundant and thus smaller coverage.
    point2d_of(&mut reconstruction, 1, 0);
    assert_eq!(
        as_set(find_redundant_points3d(0.6, &reconstruction).unwrap()),
        HashSet::from([2, 4])
    );

    // Now change point 2 to have redundant coverage with point 1 and point 4 to
    // have unique coverage.
    point2d_of(&mut reconstruction, 2, 0);
    point2d_of(&mut reconstruction, 4, 2);
    assert_eq!(
        as_set(find_redundant_points3d(0.6, &reconstruction).unwrap()),
        HashSet::from([1, 2])
    );
}
