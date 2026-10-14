// Rust-only checks of colmap_rust::scene::pose_graph::PoseGraph beyond pose_graph_test.cc:
// the order of equally large frame components (docs/CPP_DIVERGENCES.md entry 282).

use std::collections::HashSet;

use colmap_rust::math::random::set_prng_seed;
use colmap_rust::scene::pose_graph::PoseGraph;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};

use crate::pose_graph::synthesize_edge;

#[test]
fn rust_only_equal_size_components_ordered_by_smallest_frame_id() {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        num_rigs: 4,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d: 10,
        ..SyntheticDatasetOptions::default()
    };
    let mut reconstruction = Reconstruction::new();
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    let image_ids = reconstruction.reg_image_ids().unwrap();
    let frame = |i: usize| reconstruction.image(image_ids[i]).unwrap().frame_id();

    // Two components of two frames each; the one with the larger frame ids is added first.
    let mut pose_graph = PoseGraph::new();
    pose_graph
        .add_edge(image_ids[3], image_ids[2], synthesize_edge(50))
        .unwrap();
    pose_graph
        .add_edge(image_ids[1], image_ids[0], synthesize_edge(50))
        .unwrap();

    let (low, high) = if frame(0) < frame(2) {
        (
            HashSet::from([frame(0), frame(1)]),
            HashSet::from([frame(2), frame(3)]),
        )
    } else {
        (
            HashSet::from([frame(2), frame(3)]),
            HashSet::from([frame(0), frame(1)]),
        )
    };
    let components = pose_graph
        .connected_frame_components(&reconstruction, true)
        .unwrap();
    assert_eq!(components, vec![low.clone(), high]);
    assert_eq!(
        pose_graph
            .largest_connected_frame_component(&reconstruction, true)
            .unwrap(),
        low
    );
    let images = pose_graph
        .connected_image_ids_for_frame_components(&reconstruction, true)
        .unwrap();
    assert_eq!(images.len(), 2);
    assert!(images[0]
        .iter()
        .all(|&image_id| low.contains(&reconstruction.image(image_id).unwrap().frame_id())));
}
