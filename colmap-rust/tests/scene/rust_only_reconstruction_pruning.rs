// Rust-only checks of colmap_rust::scene::reconstruction_pruning beyond
// reconstruction_pruning_test.cc: two single-observation points in one image. In the same tile
// the second point's gain drops once the first is selected; in different tiles both keep the
// full gain. With equal gains the larger id is selected first, as in COLMAP's
// (gain, point3D_id) max-heap.

use colmap_rust::linalg::Vector2d;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::reconstruction_pruning::find_redundant_points3d;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::scene::track::TrackElement;

fn build(xy1: Vector2d, xy2: Vector2d) -> Reconstruction {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d: 2,
        num_points2d_without_point3d: 0,
        camera_width: 80,
        camera_height: 80,
        ..SyntheticDatasetOptions::default()
    };
    let mut reconstruction = Reconstruction::new();
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    // FindRedundantPoints3D reads only the tracks and the 2D coordinates: give the image
    // exactly the two points and each 3D point a one-observation track.
    let image_id = *reconstruction.images().keys().next().unwrap();
    reconstruction
        .image_mut(image_id)
        .unwrap()
        .set_points2d_from_xy(&[xy1, xy2]);
    for (point3d_id, point2d_idx) in [(1, 0), (2, 1)] {
        reconstruction
            .point3d_mut(point3d_id)
            .unwrap()
            .track
            .set_elements(vec![TrackElement {
                image_id,
                point2d_idx,
            }]);
    }
    reconstruction
}

#[test]
fn rust_only_same_tile_point_is_redundant() {
    let same_tile = build(Vector2d::new(5.0, 5.0), Vector2d::new(6.0, 6.0));
    let different_tiles = build(Vector2d::new(5.0, 5.0), Vector2d::new(75.0, 5.0));
    assert_eq!(find_redundant_points3d(0.2, &same_tile).unwrap(), vec![1]);
    assert!(find_redundant_points3d(0.1, &same_tile).unwrap().is_empty());
    assert!(find_redundant_points3d(0.2, &different_tiles)
        .unwrap()
        .is_empty());
    assert_eq!(
        find_redundant_points3d(0.3, &different_tiles).unwrap(),
        vec![1, 2]
    );
}
