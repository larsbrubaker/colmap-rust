// Port of colmap/scene/correspondence_graph_test.cc, 1:1 (same test names, values and checks),
// for colmap_rust::scene::correspondence_graph::CorrespondenceGraph. The TEST_P suite
// CorrespondenceGraphFinalizeTest, instantiated over testing::Bool() as "NotFinalized" /
// "Finalized", becomes one test per instantiation calling a shared body. gmock's
// `UnorderedElementsAre` compares sorted ids and `NumMatchesBetweenAllImages().at(id)` is
// `num_matches_at`; COLMAP's `Rigid3dNear` matcher is
// `rigid3d_near` below (the same as tests/geometry.rs's).

use colmap_rust::feature::types::{FeatureMatch, FeatureMatches};
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::scene::correspondence_graph::{Correspondence, CorrespondenceGraph};
use colmap_rust::scene::two_view_geometry::{ConfigurationType, TwoViewGeometry};
use colmap_rust::util::types::{image_pair_to_pair_id, ImageId, Point2DIdx};

/// Port of COLMAP's `Rigid3dNear(rhs, rtol, ttol)` matcher (geometry/rigid3_matchers.h).
#[allow(clippy::neg_cmp_op_on_partial_ord)] // !(a <= b) is deliberate: NaN must fail.
fn rigid3d_near(lhs: &Rigid3d, rhs: &Rigid3d, rtol: f64, ttol: f64) -> bool {
    if !(lhs.rotation.angular_distance(rhs.rotation) <= rtol) {
        return false;
    }
    if rhs.translation.to_array().iter().all(|c| c.abs() <= 1e-12) {
        lhs.translation.norm() <= ttol
    } else {
        lhs.translation.is_approx_with(rhs.translation, ttol)
    }
}

fn random_rigid3d() -> Rigid3d {
    Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d())
}

fn matches(pairs: &[(Point2DIdx, Point2DIdx)]) -> FeatureMatches {
    pairs
        .iter()
        .map(|&(idx1, idx2)| FeatureMatch::new(idx1, idx2))
        .collect()
}

fn count_num_transitive_correspondences(
    graph: &CorrespondenceGraph,
    image_id: ImageId,
    point2d_idx: Point2DIdx,
    transitivity: usize,
) -> usize {
    let mut corrs = Vec::new();
    graph
        .extract_transitive_correspondences(image_id, point2d_idx, transitivity, &mut corrs)
        .unwrap();
    corrs.len()
}

/// `NumMatchesBetweenAllImages().at(pair_id)`: the entry for `pair_id`, which must exist.
fn num_matches_at(graph: &CorrespondenceGraph, pair_id: u64) -> Point2DIdx {
    graph
        .num_matches_between_all_images()
        .into_iter()
        .find(|&(id, _)| id == pair_id)
        .map(|(_, num_matches)| num_matches)
        .expect("pair exists")
}

fn sorted(mut ids: Vec<u64>) -> Vec<u64> {
    ids.sort_unstable();
    ids
}

#[test]
fn correspondence_print() {
    let correspondence = Correspondence::new(1, 2);
    assert_eq!(
        correspondence.to_string(),
        "Correspondence(image_id=1, point2D_idx=2)"
    );
}

#[test]
fn correspondence_graph_empty() {
    let correspondence_graph = CorrespondenceGraph::new();
    assert_eq!(correspondence_graph.num_images(), 0);
    assert_eq!(correspondence_graph.num_image_pairs(), 0);
    assert_eq!(
        correspondence_graph.num_matches_between_all_images().len(),
        0
    );
    assert_eq!(correspondence_graph.image_pairs().len(), 0);
}

#[test]
fn correspondence_graph_print() {
    let mut correspondence_graph = CorrespondenceGraph::new();
    correspondence_graph.add_image(0, 10).unwrap();
    correspondence_graph.add_image(1, 10).unwrap();
    correspondence_graph
        .add_two_view_geometry(0, 1, TwoViewGeometry::default())
        .unwrap();
    assert_eq!(
        correspondence_graph.to_string(),
        "CorrespondenceGraph(num_images=2, num_image_pairs=1)"
    );
}

/// Asserts `ExtractCorrespondences(image_id, point2d_idx)` is exactly `expected`.
fn expect_corrs(
    graph: &CorrespondenceGraph,
    corrs: &mut Vec<Correspondence>,
    image_id: ImageId,
    point2d_idx: Point2DIdx,
    expected: &[(ImageId, Point2DIdx)],
) {
    graph
        .extract_correspondences(image_id, point2d_idx, corrs)
        .unwrap();
    assert_eq!(corrs.len(), expected.len());
    for (corr, &(expected_image_id, expected_point2d_idx)) in corrs.iter().zip(expected) {
        assert_eq!(corr.image_id, expected_image_id);
        assert_eq!(corr.point2d_idx, expected_point2d_idx);
    }
}

fn two_view(finalize: bool) {
    let mut correspondence_graph = CorrespondenceGraph::new();
    correspondence_graph.add_image(0, 10).unwrap();
    correspondence_graph.add_image(1, 10).unwrap();
    assert!(correspondence_graph.exists_image(0));
    assert!(correspondence_graph.exists_image(1));
    assert!(!correspondence_graph.exists_image(2));
    assert_eq!(correspondence_graph.num_images(), 2);
    let mut two_view_geometry01 = TwoViewGeometry {
        cam2_from_cam1: Some(random_rigid3d()),
        ..TwoViewGeometry::default()
    };
    two_view_geometry01.inlier_matches = matches(&[(0, 0), (1, 2), (3, 7), (4, 8)]);
    correspondence_graph
        .add_two_view_geometry(0, 1, two_view_geometry01.clone())
        .unwrap();
    if finalize {
        correspondence_graph.finalize().unwrap();
    }
    let g = &correspondence_graph;
    assert_eq!(g.num_correspondences_for_image(0).unwrap(), 4);
    assert_eq!(g.num_correspondences_for_image(1).unwrap(), 4);
    let pair_id = image_pair_to_pair_id(0, 1).unwrap();
    assert_eq!(g.num_matches_between_all_images().len(), 1);
    assert_eq!(num_matches_at(g, pair_id), 4);
    assert_eq!(sorted(g.image_pairs()), vec![pair_id]);
    let two_view_geometry01_stored = g.extract_two_view_geometry(0, 1, true).unwrap();
    assert!(rigid3d_near(
        &two_view_geometry01_stored.cam2_from_cam1.unwrap(),
        &two_view_geometry01.cam2_from_cam1.unwrap(),
        1e-6,
        1e-6
    ));
    assert_eq!(
        two_view_geometry01_stored.inlier_matches,
        two_view_geometry01.inlier_matches
    );
    let mut matches01_stored = FeatureMatches::new();
    g.extract_matches_between_images(0, 1, &mut matches01_stored)
        .unwrap();
    assert_eq!(matches01_stored, two_view_geometry01.inlier_matches);
    let mut two_view_geometry10 = two_view_geometry01.clone();
    two_view_geometry10.invert();
    let two_view_geometry10_stored = g.extract_two_view_geometry(1, 0, true).unwrap();
    assert!(rigid3d_near(
        &two_view_geometry10_stored.cam2_from_cam1.unwrap(),
        &two_view_geometry10.cam2_from_cam1.unwrap(),
        1e-6,
        1e-6
    ));
    assert_eq!(
        two_view_geometry10_stored.inlier_matches,
        two_view_geometry10.inlier_matches
    );
    let mut matches10_stored = FeatureMatches::new();
    g.extract_matches_between_images(1, 0, &mut matches10_stored)
        .unwrap();
    assert_eq!(matches10_stored, two_view_geometry10.inlier_matches);
    let two_view_geometry01_without_matches_stored =
        g.extract_two_view_geometry(0, 1, false).unwrap();
    assert!(two_view_geometry01_without_matches_stored
        .inlier_matches
        .is_empty());

    let mut corrs = Vec::new();

    expect_corrs(g, &mut corrs, 0, 0, &[(1, 0)]);
    assert!(g.has_correspondences(0, 0).unwrap());
    assert!(g.is_two_view_observation(0, 0).unwrap());

    expect_corrs(g, &mut corrs, 1, 0, &[(0, 0)]);
    assert!(g.has_correspondences(1, 0).unwrap());
    assert!(g.is_two_view_observation(1, 0).unwrap());

    expect_corrs(g, &mut corrs, 0, 1, &[(1, 2)]);
    assert!(g.has_correspondences(0, 1).unwrap());
    assert!(g.is_two_view_observation(0, 1).unwrap());

    expect_corrs(g, &mut corrs, 1, 2, &[(0, 1)]);
    assert!(g.has_correspondences(1, 2).unwrap());
    assert!(g.is_two_view_observation(1, 2).unwrap());

    expect_corrs(g, &mut corrs, 0, 4, &[(1, 8)]);
    assert!(g.has_correspondences(0, 4).unwrap());
    assert!(g.is_two_view_observation(0, 4).unwrap());

    expect_corrs(g, &mut corrs, 0, 3, &[(1, 7)]);
    assert!(g.has_correspondences(0, 3).unwrap());

    expect_corrs(g, &mut corrs, 1, 7, &[(0, 3)]);
    assert!(g.has_correspondences(1, 7).unwrap());
    assert!(g.is_two_view_observation(1, 7).unwrap());

    expect_corrs(g, &mut corrs, 1, 8, &[(0, 4)]);
    assert!(g.has_correspondences(1, 8).unwrap());
    assert!(g.is_two_view_observation(1, 8).unwrap());

    for i in 0..10 {
        assert_eq!(count_num_transitive_correspondences(g, 0, i, 0), 0);
        g.extract_correspondences(0, i, &mut corrs).unwrap();
        assert_eq!(
            corrs.len(),
            count_num_transitive_correspondences(g, 0, i, 2)
        );
        assert_eq!(count_num_transitive_correspondences(g, 1, i, 0), 0);
        g.extract_correspondences(1, i, &mut corrs).unwrap();
        assert_eq!(
            corrs.len(),
            count_num_transitive_correspondences(g, 1, i, 2)
        );
    }
    let mut matches01 = FeatureMatches::new();
    g.extract_matches_between_images(0, 1, &mut matches01)
        .unwrap();
    assert_eq!(matches01, two_view_geometry01.inlier_matches);
    let mut matches10 = FeatureMatches::new();
    g.extract_matches_between_images(1, 0, &mut matches10)
        .unwrap();
    assert_eq!(matches10, two_view_geometry10.inlier_matches);
    assert_eq!(g.num_observations_for_image(0).unwrap(), 4);
    assert_eq!(g.num_observations_for_image(1).unwrap(), 4);
    assert_eq!(g.num_correspondences_for_image(0).unwrap(), 4);
    assert_eq!(g.num_correspondences_for_image(1).unwrap(), 4);
}

fn three_view(finalize: bool) {
    let mut correspondence_graph = CorrespondenceGraph::new();
    correspondence_graph.add_image(0, 10).unwrap();
    correspondence_graph.add_image(1, 10).unwrap();
    correspondence_graph.add_image(2, 10).unwrap();
    let two_view_geometry01 = TwoViewGeometry {
        inlier_matches: matches(&[(0, 0)]),
        ..TwoViewGeometry::default()
    };
    correspondence_graph
        .add_two_view_geometry(0, 1, two_view_geometry01)
        .unwrap();
    let two_view_geometry02 = TwoViewGeometry {
        inlier_matches: matches(&[(0, 0)]),
        ..TwoViewGeometry::default()
    };
    correspondence_graph
        .add_two_view_geometry(0, 2, two_view_geometry02)
        .unwrap();
    let two_view_geometry12 = TwoViewGeometry {
        inlier_matches: matches(&[(0, 0), (5, 5)]),
        ..TwoViewGeometry::default()
    };
    correspondence_graph
        .add_two_view_geometry(1, 2, two_view_geometry12)
        .unwrap();
    if finalize {
        correspondence_graph.finalize().unwrap();
    }
    let g = &correspondence_graph;
    assert_eq!(g.num_observations_for_image(0).unwrap(), 1);
    assert_eq!(g.num_observations_for_image(1).unwrap(), 2);
    assert_eq!(g.num_observations_for_image(2).unwrap(), 2);
    assert_eq!(g.num_correspondences_for_image(0).unwrap(), 2);
    assert_eq!(g.num_correspondences_for_image(1).unwrap(), 3);
    assert_eq!(g.num_correspondences_for_image(2).unwrap(), 3);
    let pair_id01 = image_pair_to_pair_id(0, 1).unwrap();
    let pair_id02 = image_pair_to_pair_id(0, 2).unwrap();
    let pair_id12 = image_pair_to_pair_id(1, 2).unwrap();
    assert_eq!(g.num_matches_between_all_images().len(), 3);
    assert_eq!(num_matches_at(g, pair_id01), 1);
    assert_eq!(num_matches_at(g, pair_id02), 1);
    assert_eq!(num_matches_at(g, pair_id12), 2);
    assert_eq!(
        sorted(g.image_pairs()),
        sorted(vec![pair_id01, pair_id02, pair_id12])
    );

    let mut corrs = Vec::new();
    expect_corrs(g, &mut corrs, 0, 0, &[(1, 0), (2, 0)]);
    expect_corrs(g, &mut corrs, 1, 0, &[(0, 0), (2, 0)]);
    expect_corrs(g, &mut corrs, 2, 0, &[(0, 0), (1, 0)]);
    expect_corrs(g, &mut corrs, 1, 5, &[(2, 5)]);
    expect_corrs(g, &mut corrs, 2, 5, &[(1, 5)]);

    assert_eq!(count_num_transitive_correspondences(g, 0, 0, 2), 2);
    assert_eq!(count_num_transitive_correspondences(g, 1, 0, 2), 2);
    assert_eq!(count_num_transitive_correspondences(g, 2, 0, 2), 2);
    assert_eq!(count_num_transitive_correspondences(g, 0, 0, 3), 2);
    assert_eq!(count_num_transitive_correspondences(g, 1, 0, 3), 2);
    assert_eq!(count_num_transitive_correspondences(g, 2, 0, 3), 2);
}

#[test]
fn correspondence_graph_finalize_test_two_view_not_finalized() {
    two_view(false);
}

#[test]
fn correspondence_graph_finalize_test_two_view_finalized() {
    two_view(true);
}

#[test]
fn correspondence_graph_finalize_test_three_view_not_finalized() {
    three_view(false);
}

#[test]
fn correspondence_graph_finalize_test_three_view_finalized() {
    three_view(true);
}

#[test]
fn correspondence_graph_out_of_bounds() {
    let mut correspondence_graph = CorrespondenceGraph::new();
    correspondence_graph.add_image(0, 10).unwrap();
    correspondence_graph.add_image(1, 4).unwrap();
    let two_view_geometry = TwoViewGeometry {
        inlier_matches: matches(&[(9, 3), (10, 3), (9, 4)]),
        ..TwoViewGeometry::default()
    };
    correspondence_graph
        .add_two_view_geometry(0, 1, two_view_geometry)
        .unwrap();
    assert_eq!(
        correspondence_graph
            .num_correspondences_for_image(0)
            .unwrap(),
        1
    );
    assert_eq!(
        correspondence_graph
            .num_correspondences_for_image(1)
            .unwrap(),
        1
    );
    let pair_id = image_pair_to_pair_id(0, 1).unwrap();
    assert_eq!(num_matches_at(&correspondence_graph, pair_id), 1);
}

#[test]
fn correspondence_graph_duplicate() {
    let mut correspondence_graph = CorrespondenceGraph::new();
    correspondence_graph.add_image(0, 10).unwrap();
    correspondence_graph.add_image(1, 10).unwrap();
    let two_view_geometry = TwoViewGeometry {
        inlier_matches: matches(&[(0, 0), (1, 1), (1, 1), (3, 3), (3, 4)]),
        ..TwoViewGeometry::default()
    };
    correspondence_graph
        .add_two_view_geometry(0, 1, two_view_geometry)
        .unwrap();
    assert_eq!(
        correspondence_graph
            .num_correspondences_for_image(0)
            .unwrap(),
        4
    );
    assert_eq!(
        correspondence_graph
            .num_correspondences_for_image(1)
            .unwrap(),
        4
    );
    let pair_id = image_pair_to_pair_id(0, 1).unwrap();
    assert_eq!(num_matches_at(&correspondence_graph, pair_id), 4);
}

#[test]
fn correspondence_graph_update_two_view_geometry() {
    let mut correspondence_graph = CorrespondenceGraph::new();
    correspondence_graph.add_image(0, 10).unwrap();
    correspondence_graph.add_image(1, 10).unwrap();
    let two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        inlier_matches: matches(&[(0, 0), (1, 2), (3, 7)]),
        ..TwoViewGeometry::default()
    };
    correspondence_graph
        .add_two_view_geometry(0, 1, two_view_geometry.clone())
        .unwrap();
    correspondence_graph.finalize().unwrap();

    // Verify initial state has no relative pose.
    let extracted = correspondence_graph
        .extract_two_view_geometry(0, 1, false)
        .unwrap();
    assert!(extracted.cam2_from_cam1.is_none());
    assert_eq!(extracted.config, ConfigurationType::Calibrated);

    // Update with a decomposed relative pose.
    let mut updated = extracted;
    updated.cam2_from_cam1 = Some(random_rigid3d());
    correspondence_graph
        .update_two_view_geometry(0, 1, updated.clone())
        .unwrap();

    // Verify the updated geometry is returned.
    let after_update = correspondence_graph
        .extract_two_view_geometry(0, 1, false)
        .unwrap();
    assert!(after_update.cam2_from_cam1.is_some());
    assert!(rigid3d_near(
        &after_update.cam2_from_cam1.unwrap(),
        &updated.cam2_from_cam1.unwrap(),
        1e-6,
        1e-6
    ));
    assert_eq!(after_update.config, ConfigurationType::Calibrated);

    // Verify matches are preserved (stored separately from geometry).
    let mut matches = FeatureMatches::new();
    correspondence_graph
        .extract_matches_between_images(0, 1, &mut matches)
        .unwrap();
    assert_eq!(matches, two_view_geometry.inlier_matches);
}

#[test]
fn correspondence_graph_update_two_view_geometry_swapped() {
    let mut correspondence_graph = CorrespondenceGraph::new();
    // Use image IDs where ShouldSwapImagePair(1, 0) is true, i.e. id1 > id2.
    correspondence_graph.add_image(0, 10).unwrap();
    correspondence_graph.add_image(1, 10).unwrap();
    let two_view_geometry = TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        inlier_matches: matches(&[(0, 0), (1, 2)]),
        ..TwoViewGeometry::default()
    };
    correspondence_graph
        .add_two_view_geometry(0, 1, two_view_geometry)
        .unwrap();
    correspondence_graph.finalize().unwrap();

    // Update using the swapped order (1, 0).
    let cam1_from_cam0 = random_rigid3d();
    let updated = TwoViewGeometry {
        config: ConfigurationType::Calibrated,
        cam2_from_cam1: Some(cam1_from_cam0),
        ..TwoViewGeometry::default()
    };
    correspondence_graph
        .update_two_view_geometry(1, 0, updated)
        .unwrap();

    // Extract in the same swapped order and verify it matches.
    let extracted_10 = correspondence_graph
        .extract_two_view_geometry(1, 0, false)
        .unwrap();
    assert!(rigid3d_near(
        &extracted_10.cam2_from_cam1.unwrap(),
        &cam1_from_cam0,
        1e-6,
        1e-6
    ));

    // Extract in canonical order and verify it's the inverse.
    let extracted_01 = correspondence_graph
        .extract_two_view_geometry(0, 1, false)
        .unwrap();
    assert!(rigid3d_near(
        &extracted_01.cam2_from_cam1.unwrap(),
        &cam1_from_cam0.inverse(),
        1e-6,
        1e-6
    ));
}
