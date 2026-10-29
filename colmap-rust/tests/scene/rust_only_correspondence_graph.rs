// Rust-only tests for colmap_rust::scene::correspondence_graph: the error paths COLMAP
// throws on (std::out_of_range from map/vector `at`, THROW_CHECK), the rejected-add
// atomicity and the pair order of the `scene/correspondence_graph.rs` header.

use colmap_rust::feature::types::FeatureMatch;
use colmap_rust::scene::correspondence_graph::{Correspondence, CorrespondenceGraph};
use colmap_rust::scene::two_view_geometry::TwoViewGeometry;
use colmap_rust::util::types::{image_pair_to_pair_id, INVALID_IMAGE_ID, INVALID_POINT2D_IDX};
use colmap_rust::ErrorKind;

fn geometry(pairs: &[(u32, u32)]) -> TwoViewGeometry {
    TwoViewGeometry {
        inlier_matches: pairs
            .iter()
            .map(|&(a, b)| FeatureMatch::new(a, b))
            .collect(),
        ..TwoViewGeometry::default()
    }
}

#[test]
fn rust_only_correspondence_default_is_invalid() {
    let corr = Correspondence::default();
    assert_eq!(corr.image_id, INVALID_IMAGE_ID);
    assert_eq!(corr.point2d_idx, INVALID_POINT2D_IDX);
}

#[test]
fn rust_only_correspondence_graph_missing_image_is_out_of_range() {
    let mut graph = CorrespondenceGraph::new();
    graph.add_image(0, 2).unwrap();
    let err = graph.num_observations_for_image(7).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::OutOfRange);
    assert_eq!(err.message(), "Image with ID 7 does not exist");
    assert_eq!(
        graph.num_correspondences_for_image(7).unwrap_err().kind(),
        ErrorKind::OutOfRange
    );
    assert_eq!(
        graph.find_correspondences(7, 0).unwrap_err().kind(),
        ErrorKind::OutOfRange
    );
    assert_eq!(
        graph
            .add_two_view_geometry(0, 7, geometry(&[(0, 0)]))
            .unwrap_err()
            .kind(),
        ErrorKind::OutOfRange
    );
    // The failed add left image 0 in place and unchanged.
    assert!(graph.exists_image(0));
    assert_eq!(graph.num_correspondences_for_image(0).unwrap(), 0);
}

#[test]
fn rust_only_correspondence_graph_point_out_of_range() {
    for finalize in [false, true] {
        let mut graph = CorrespondenceGraph::new();
        graph.add_image(0, 2).unwrap();
        if finalize {
            graph.finalize().unwrap();
        }
        assert!(graph.find_correspondences(0, 1).unwrap().is_empty());
        let err = graph.find_correspondences(0, 2).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::OutOfRange);
    }
}

#[test]
fn rust_only_correspondence_graph_check_failures() {
    let mut graph = CorrespondenceGraph::new();
    graph.add_image(0, 4).unwrap();
    graph.add_image(1, 4).unwrap();
    assert_eq!(
        graph.add_image(0, 4).unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
    assert!(graph.extract_two_view_geometry(0, 1, false).is_err());
    assert!(graph
        .update_two_view_geometry(0, 1, TwoViewGeometry::default())
        .is_err());
    graph.finalize().unwrap();
    assert_eq!(
        graph.finalize().unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
}

#[test]
fn rust_only_correspondence_graph_rejected_duplicate_pair_leaves_graph_unchanged() {
    let mut graph = CorrespondenceGraph::new();
    graph.add_image(0, 4).unwrap();
    graph.add_image(1, 4).unwrap();
    graph
        .add_two_view_geometry(0, 1, geometry(&[(0, 0), (1, 1)]))
        .unwrap();
    let err = graph
        .add_two_view_geometry(1, 0, geometry(&[(2, 2)]))
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::InvalidArgument);
    assert!(err
        .message()
        .contains("Two view geometry for image pair was already added: image_id1=1, image_id2=0"));
    assert_eq!(graph.num_correspondences_for_image(0).unwrap(), 2);
    assert_eq!(graph.num_correspondences_for_image(1).unwrap(), 2);
    assert_eq!(graph.num_matches_between_images(0, 1).unwrap(), 2);
    assert!(graph.find_correspondences(0, 2).unwrap().is_empty());
}

#[test]
fn rust_only_correspondence_graph_self_match_is_ignored() {
    let mut graph = CorrespondenceGraph::new();
    graph.add_image(3, 4).unwrap();
    graph
        .add_two_view_geometry(3, 3, geometry(&[(0, 1)]))
        .unwrap();
    assert_eq!(graph.num_image_pairs(), 0);
    assert_eq!(graph.num_correspondences_for_image(3).unwrap(), 0);
}

#[test]
fn rust_only_correspondence_graph_pairs_in_insertion_order() {
    let mut graph = CorrespondenceGraph::new();
    for image_id in 0..4 {
        graph.add_image(image_id, 2).unwrap();
    }
    // Added in an order that is neither ascending nor descending.
    graph
        .add_two_view_geometry(2, 3, geometry(&[(0, 0)]))
        .unwrap();
    graph
        .add_two_view_geometry(1, 0, geometry(&[(0, 0)]))
        .unwrap();
    graph
        .add_two_view_geometry(3, 1, geometry(&[(1, 1)]))
        .unwrap();
    let expected = vec![
        image_pair_to_pair_id(2, 3).unwrap(),
        image_pair_to_pair_id(0, 1).unwrap(),
        image_pair_to_pair_id(1, 3).unwrap(),
    ];
    assert_eq!(graph.image_pairs(), expected);
    let keys: Vec<u64> = graph
        .num_matches_between_all_images()
        .into_iter()
        .map(|(pair_id, _)| pair_id)
        .collect();
    assert_eq!(keys, expected);
}

#[test]
fn rust_only_correspondence_graph_transitive_correspondences_chain() {
    // 0:0 - 1:0 - 2:0 - 3:0, a chain: transitivity bounds how far the walk reaches, and the
    // queried observation is replaced by the last collected one.
    let mut graph = CorrespondenceGraph::new();
    for image_id in 0..4 {
        graph.add_image(image_id, 1).unwrap();
    }
    graph
        .add_two_view_geometry(0, 1, geometry(&[(0, 0)]))
        .unwrap();
    graph
        .add_two_view_geometry(1, 2, geometry(&[(0, 0)]))
        .unwrap();
    graph
        .add_two_view_geometry(2, 3, geometry(&[(0, 0)]))
        .unwrap();
    graph.finalize().unwrap();
    let mut corrs = Vec::new();
    graph
        .extract_transitive_correspondences(0, 0, 2, &mut corrs)
        .unwrap();
    assert_eq!(
        corrs,
        vec![Correspondence::new(2, 0), Correspondence::new(1, 0)]
    );
    graph
        .extract_transitive_correspondences(0, 0, 3, &mut corrs)
        .unwrap();
    assert_eq!(
        corrs,
        vec![
            Correspondence::new(3, 0),
            Correspondence::new(1, 0),
            Correspondence::new(2, 0)
        ]
    );
    assert!(!graph.is_two_view_observation(1, 0).unwrap());
    assert!(!graph.is_two_view_observation(0, 0).unwrap());
}
