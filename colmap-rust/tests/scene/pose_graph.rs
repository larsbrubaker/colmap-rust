// Port of colmap/scene/pose_graph_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::scene::pose_graph::PoseGraph. Tier A (exact). `EXPECT_THROW(...,
// std::out_of_range)` / `std::runtime_error` check the error kind. `UnorderedElementsAre` on
// pair ids and frame sets compares sets. COLMAP's gtest main seeds the PRNG with 0 before
// every test; the cases that draw random numbers call `set_prng_seed(0)` first.

use std::collections::HashSet;

use colmap_rust::feature::types::FeatureMatch;
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Quaterniond, Vector3d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::database_cache::{DatabaseCache, DatabaseCacheOptions};
use colmap_rust::scene::image::Image;
use colmap_rust::scene::pose_graph::{PoseGraph, PoseGraphEdge};
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::util::types::{image_pair_to_pair_id, ImagePairId, INVALID_CAMERA_ID};
use colmap_rust::ErrorKind;

pub(crate) fn synthesize_edge(num_matches: i32) -> PoseGraphEdge {
    let mut edge = PoseGraphEdge::new(Rigid3d::identity());
    edge.num_matches = num_matches;
    edge
}

fn edge() -> PoseGraphEdge {
    synthesize_edge(50)
}

fn edge_tx(x: f64) -> PoseGraphEdge {
    let mut edge = edge();
    edge.cam2_from_cam1 = Rigid3d::new(Quaterniond::identity(), Vector3d::new(x, 0.0, 0.0));
    edge
}

fn pair(image_id1: u32, image_id2: u32) -> ImagePairId {
    image_pair_to_pair_id(image_id1, image_id2).unwrap()
}

fn tx(pose_graph: &PoseGraph, image_id1: u32, image_id2: u32) -> f64 {
    pose_graph
        .edge_ref(image_id1, image_id2)
        .unwrap()
        .0
        .cam2_from_cam1
        .translation
        .x
}

fn synthetic(num_rigs: i32) -> Reconstruction {
    let options = SyntheticDatasetOptions {
        num_rigs,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d: 10,
        ..SyntheticDatasetOptions::default()
    };
    let mut reconstruction = Reconstruction::new();
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    reconstruction
}

#[test]
fn pose_graph_nominal() {
    let mut pose_graph = PoseGraph::new();

    // Empty view graph.
    assert!(pose_graph.is_empty());
    assert_eq!(pose_graph.num_edges(), 0);

    // Add some pairs.
    pose_graph.add_edge(1, 2, edge()).unwrap();
    pose_graph.add_edge(1, 3, edge()).unwrap();
    pose_graph.add_edge(2, 3, edge()).unwrap();

    assert!(!pose_graph.is_empty());
    assert_eq!(pose_graph.num_edges(), 3);

    // Invalidate one pair.
    pose_graph.set_invalid_edge(pair(1, 2)).unwrap();
    assert_eq!(pose_graph.num_edges(), 3);
    assert!(!pose_graph.is_valid(pair(1, 2)));

    // Clear the view graph.
    pose_graph.clear();
    assert!(pose_graph.is_empty());
    assert_eq!(pose_graph.num_edges(), 0);
}

#[test]
fn pose_graph_add_edge() {
    let mut pose_graph = PoseGraph::new();

    // Normal add.
    pose_graph.add_edge(1, 2, edge_tx(1.0)).unwrap();

    assert_eq!(pose_graph.num_edges(), 1);
    let (stored, swapped) = pose_graph.edge_ref(1, 2).unwrap();
    assert!(!swapped);
    assert_eq!(stored.cam2_from_cam1.translation.x, 1.0);

    // Add with swapped IDs should invert the pair.
    pose_graph.add_edge(4, 3, edge_tx(2.0)).unwrap(); // 4 > 3, should swap and invert

    assert_eq!(pose_graph.num_edges(), 2);
    let (stored2, swapped2) = pose_graph.edge_ref(3, 4).unwrap();
    assert!(!swapped2);
    assert_eq!(stored2.cam2_from_cam1.translation.x, -2.0);

    // Duplicate should throw.
    for (image_id1, image_id2) in [(1, 2), (2, 1)] {
        let error = pose_graph
            .add_edge(image_id1, image_id2, edge())
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::RuntimeError);
    }
}

#[test]
fn pose_graph_has_edge() {
    let mut pose_graph = PoseGraph::new();
    pose_graph.add_edge(1, 2, edge()).unwrap();

    assert!(pose_graph.has_edge(1, 2).unwrap());
    assert!(pose_graph.has_edge(2, 1).unwrap()); // Order doesn't matter
    assert!(!pose_graph.has_edge(1, 3).unwrap());
}

#[test]
fn pose_graph_edge_ref() {
    let mut pose_graph = PoseGraph::new();
    pose_graph.add_edge(1, 2, edge_tx(1.0)).unwrap();

    // Normal order: swapped = false.
    let (ref1, swapped1) = pose_graph.edge_ref_mut(1, 2).unwrap();
    assert!(!swapped1);
    assert_eq!(ref1.cam2_from_cam1.translation.x, 1.0);

    // Reversed order: swapped = true.
    let (ref2, swapped2) = pose_graph.edge_ref_mut(2, 1).unwrap();
    assert!(swapped2);
    assert_eq!(ref2.cam2_from_cam1.translation.x, 1.0); // Same reference

    // Modify validity through PoseGraph.
    pose_graph.set_invalid_edge(pair(1, 2)).unwrap();
    assert!(!pose_graph.is_valid(pair(1, 2)));

    // Non-existent pair should throw.
    let error = pose_graph.edge_ref_mut(1, 3).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::OutOfRange);
}

#[test]
fn pose_graph_get_edge() {
    let mut pose_graph = PoseGraph::new();
    pose_graph.add_edge(1, 2, edge_tx(1.0)).unwrap();

    // Normal order: returns as-is.
    let copy1 = pose_graph.get_edge(1, 2).unwrap();
    assert_eq!(copy1.cam2_from_cam1.translation.x, 1.0);

    // Reversed order: returns inverted copy.
    let copy2 = pose_graph.get_edge(2, 1).unwrap();
    assert_eq!(copy2.cam2_from_cam1.translation.x, -1.0);

    // Original unchanged.
    assert_eq!(tx(&pose_graph, 1, 2), 1.0);

    // Non-existent pair should throw.
    let error = pose_graph.get_edge(1, 3).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::OutOfRange);
}

#[test]
fn pose_graph_delete_edge() {
    let mut pose_graph = PoseGraph::new();
    pose_graph.add_edge(1, 2, edge()).unwrap();
    pose_graph.add_edge(1, 3, edge()).unwrap();

    assert!(pose_graph.delete_edge(1, 2).unwrap());
    assert!(!pose_graph.has_edge(1, 2).unwrap());
    assert_eq!(pose_graph.num_edges(), 1);

    // Delete with reversed order.
    assert!(pose_graph.delete_edge(3, 1).unwrap());
    assert_eq!(pose_graph.num_edges(), 0);

    // Delete non-existent returns false.
    assert!(!pose_graph.delete_edge(1, 2).unwrap());
}

#[test]
fn pose_graph_update_edge() {
    let mut pose_graph = PoseGraph::new();
    pose_graph.add_edge(1, 2, edge_tx(1.0)).unwrap();

    // Update with normal order.
    pose_graph.update_edge(1, 2, edge_tx(5.0)).unwrap();

    assert_eq!(tx(&pose_graph, 1, 2), 5.0);

    // Update with reversed order should invert.
    pose_graph.update_edge(2, 1, edge_tx(3.0)).unwrap();

    assert_eq!(tx(&pose_graph, 1, 2), -3.0);

    // Update non-existent should throw.
    let error = pose_graph.update_edge(1, 3, edge()).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::RuntimeError);
}

#[test]
fn pose_graph_valid_edges() {
    let mut pose_graph = PoseGraph::new();

    let pair_id1 = pair(1, 2);
    let pair_id2 = pair(1, 3);
    let pair_id3 = pair(2, 3);
    pose_graph.add_edge(1, 2, edge()).unwrap();
    pose_graph.add_edge(1, 3, edge()).unwrap();
    pose_graph.add_edge(2, 3, edge()).unwrap();

    let get_valid_pair_ids = |pose_graph: &PoseGraph| -> HashSet<ImagePairId> {
        pose_graph
            .valid_edges()
            .map(|(&pair_id, _)| pair_id)
            .collect()
    };

    // All pairs start valid.
    assert_eq!(
        get_valid_pair_ids(&pose_graph),
        HashSet::from([pair_id1, pair_id2, pair_id3])
    );

    // Invalidate one pair.
    pose_graph.set_invalid_edge(pair_id2).unwrap();
    assert_eq!(
        get_valid_pair_ids(&pose_graph),
        HashSet::from([pair_id1, pair_id3])
    );

    // Re-validate the pair.
    pose_graph.set_valid_edge(pair_id2).unwrap();
    assert_eq!(
        get_valid_pair_ids(&pose_graph),
        HashSet::from([pair_id1, pair_id2, pair_id3])
    );
}

#[test]
fn pose_graph_load() {
    set_prng_seed(0);
    let database = Database::new();

    let camera =
        Camera::create_from_model_id(INVALID_CAMERA_ID, CameraModelId::SimplePinhole, 1.0, 1, 1)
            .unwrap();
    let camera_id = database.write_camera(&camera, false).unwrap();

    // Create images.
    for i in 1..=3 {
        let mut image = Image::new();
        image.set_name(&format!("image{i}"));
        image.set_camera_id(camera_id).unwrap();
        database.write_image(&image, false).unwrap();
    }

    let two_view = TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        inlier_matches: vec![FeatureMatch::new(0, 0), FeatureMatch::new(1, 1)],
        cam2_from_cam1: Some(Rigid3d::new(
            random_eigen_quaterniond(),
            random_eigen_vector3d().normalized(),
        )),
        ..TwoViewGeometry::default()
    };

    // Create pairs (1,2) and (2,3)
    database
        .write_matches(1, 2, &[FeatureMatch::default(); 10])
        .unwrap();
    database
        .write_matches(2, 3, &[FeatureMatch::default(); 10])
        .unwrap();
    database.write_two_view_geometry(1, 2, &two_view).unwrap();
    database.write_two_view_geometry(2, 3, &two_view).unwrap();

    // Load into DatabaseCache with relative poses.
    let mut cache = DatabaseCache::new();
    cache
        .load(&database, &DatabaseCacheOptions::default())
        .unwrap();

    let mut pose_graph = PoseGraph::new();
    pose_graph.load(cache.correspondence_graph()).unwrap();

    assert_eq!(pose_graph.num_edges(), 2);
    assert!(pose_graph.has_edge(1, 2).unwrap());
    assert!(pose_graph.has_edge(2, 3).unwrap());
}

#[test]
fn pose_graph_largest_connected_frame_component_empty() {
    set_prng_seed(0);
    let reconstruction = synthetic(3);

    // Pose graph with no edges
    let pose_graph = PoseGraph::new();
    let result = pose_graph
        .largest_connected_frame_component(&reconstruction, false)
        .unwrap();
    assert!(result.is_empty());
}

#[test]
fn pose_graph_connected_frame_components() {
    set_prng_seed(0);
    // Five single-frame rigs.
    let reconstruction = synthetic(5);

    let reg_image_ids = reconstruction.reg_image_ids().unwrap();
    assert_eq!(reg_image_ids.len(), 5);

    // Component A: images 0, 1, 2 ; Component B: images 3, 4.
    let mut pose_graph = PoseGraph::new();
    pose_graph
        .add_edge(reg_image_ids[0], reg_image_ids[1], edge())
        .unwrap();
    pose_graph
        .add_edge(reg_image_ids[1], reg_image_ids[2], edge())
        .unwrap();
    pose_graph
        .add_edge(reg_image_ids[3], reg_image_ids[4], edge())
        .unwrap();

    let frame = |i: usize| reconstruction.image(reg_image_ids[i]).unwrap().frame_id();

    let components = pose_graph
        .connected_frame_components(&reconstruction, true)
        .unwrap();

    assert_eq!(components.len(), 2);
    // Components are sorted by descending size.
    assert_eq!(components[0].len(), 3);
    assert_eq!(components[1].len(), 2);
    assert_eq!(components[0], HashSet::from([frame(0), frame(1), frame(2)]));
    assert_eq!(components[1], HashSet::from([frame(3), frame(4)]));

    // The largest component must equal the first (largest) entry.
    assert_eq!(
        pose_graph
            .largest_connected_frame_component(&reconstruction, true)
            .unwrap(),
        components[0]
    );
}

#[test]
fn pose_graph_connected_frame_components_empty() {
    set_prng_seed(0);
    let reconstruction = synthetic(3);

    // Pose graph with no edges yields no components.
    let pose_graph = PoseGraph::new();
    assert!(pose_graph
        .connected_frame_components(&reconstruction, false)
        .unwrap()
        .is_empty());
}

#[test]
fn pose_graph_invalidate_pairs_outside_active_image_ids() {
    let mut pose_graph = PoseGraph::new();
    pose_graph.add_edge(1, 2, edge()).unwrap();
    pose_graph.add_edge(1, 3, edge()).unwrap();
    pose_graph.add_edge(2, 3, edge()).unwrap();

    // Only images 1 and 2 are active; image 3 is excluded
    pose_graph
        .invalidate_pairs_outside_active_image_ids(&HashSet::from([1, 2]))
        .unwrap();

    // Edge (1,2) should remain valid
    assert!(pose_graph.is_valid(pair(1, 2)));
    // Edges involving image 3 should be invalidated
    assert!(!pose_graph.is_valid(pair(1, 3)));
    assert!(!pose_graph.is_valid(pair(2, 3)));
}
