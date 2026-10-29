//! Port of `colmap/scene/pose_graph.h` / `.cc` (and colmap-sharp's `Scene/PoseGraph.cs`): the
//! graph of relative poses between image pairs that global SfM builds from the
//! correspondence graph (one edge per pair, keyed by the pair id with the smaller image id
//! first), plus its connected components in frame terms, which take a [`Reconstruction`] for
//! the image -> frame mapping. Built on [`crate::math::connected_components`].
//! Tests: `tests/scene/pose_graph.rs` (`pose_graph_test.cc` 1:1) and
//! `tests/scene/rust_only_pose_graph.rs`.
//!
//! Tier A. Translation notes:
//! - `EdgeRef` hands out the stored edge by reference ([`PoseGraph::edge_ref`] and
//!   [`PoseGraph::edge_ref_mut`]); `AddEdge`, `UpdateEdge` and `GetEdge` take and return
//!   edges by value, as COLMAP's do.
//! - COLMAP's missing-pair `std::out_of_range` (`map::at`) is an [`ErrorKind::OutOfRange`]
//!   error; its `std::runtime_error` for a duplicate or missing pair is
//!   [`ErrorKind::RuntimeError`].
//! - The edges live in a `BTreeMap`, so they iterate in ascending pair id where COLMAP's
//!   `NodeHashMap` has its hash order; no output of the pose graph itself depends on it.
//! - Equally large frame components are ordered by their smallest frame id, a deliberate
//!   divergence. `ConnectedFrameComponents` collects the frames of the valid edges in a
//!   `FlatHashSet`, finds the components (whose order follows that set, see the
//!   `math/connected_components.rs` header) and orders them with an unstable `std::sort` by
//!   descending size, so the order of equally large components is unspecified;
//!   `LargestConnectedFrameComponent` keeps the first strictly largest one in the same hash
//!   order. Here `find_connected_components` gets the frames in ascending id order and the sort
//!   is stable, so [`PoseGraph::largest_connected_frame_component`] returns the first of the
//!   ordered components, and `connected_image_ids_for_frame_components` follows the same order.
//!   The tie order decides which component global SfM treats as the largest, so it must be
//!   deterministic (CLAUDE.md's stable-sort and hash-order rules); COLMAP relies on no
//!   particular order (`pose_graph_test.cc` uses components of different sizes and compares
//!   sets). Evidence: `tests/scene/pose_graph.rs` passes with COLMAP's expectations;
//!   `tests/scene/rust_only_pose_graph.rs`
//!   `rust_only_equal_size_components_ordered_by_smallest_frame_id` pins the tie order and that
//!   the largest component equals the first one. Same as colmap-sharp entry 38.
//! - `Load`'s `LOG(INFO)` edge count is dropped, like `LOG(INFO)` elsewhere in the library.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use crate::geometry::rigid3::Rigid3d;
use crate::math::connected_components::{
    find_connected_components, find_largest_connected_component,
};
use crate::scene::correspondence_graph::CorrespondenceGraph;
use crate::scene::reconstruction::Reconstruction;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{
    image_pair_to_pair_id, pair_id_to_image_pair, should_swap_image_pair, FrameId, ImageId,
    ImagePairId,
};
use crate::Result;

/// `FrameGraph` (pose_graph.cc's anonymous namespace): the nodes and the edges.
type FrameGraph = (Vec<FrameId>, Vec<(FrameId, FrameId)>);

/// Port of `PoseGraph::Edge`: minimal relative pose data for a pose graph edge.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoseGraphEdge {
    /// Relative pose from image 1 to image 2.
    pub cam2_from_cam1: Rigid3d,
    /// Number of two-view matches used to compute the relative pose.
    pub num_matches: i32,
    /// Whether this edge is valid for reconstruction.
    pub valid: bool,
}

impl Default for PoseGraphEdge {
    /// `Edge()`: the identity relative pose, no matches, valid.
    fn default() -> Self {
        Self::new(Rigid3d::identity())
    }
}

impl PoseGraphEdge {
    /// `Edge(cam2_from_cam1)`: an edge with the given relative pose, no matches, valid.
    pub fn new(cam2_from_cam1: Rigid3d) -> Self {
        Self {
            cam2_from_cam1,
            num_matches: 0,
            valid: true,
        }
    }

    /// `Invert()`: inverts the geometry to match swapped image order.
    pub fn invert(&mut self) {
        self.cam2_from_cam1 = self.cam2_from_cam1.inverse();
    }
}

/// Port of `colmap::PoseGraph`: relative poses between image pairs.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PoseGraph {
    // Map from pair ID to edge data. The pair ID is computed from the two image IDs using
    // ImagePairToPairId, with the smaller ID first.
    edges: BTreeMap<ImagePairId, PoseGraphEdge>,
}

fn pair_error(kind: ErrorKind, what: &str, image_id1: ImageId, image_id2: ImageId) -> ColmapError {
    ColmapError::new(kind, format!("{what}: {image_id1}, {image_id2}"))
}

impl PoseGraph {
    /// An empty pose graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// `Edges()`: all edges by pair id (smaller image id first), valid or not.
    pub fn edges(&self) -> &BTreeMap<ImagePairId, PoseGraphEdge> {
        &self.edges
    }

    /// The non-const `Edges()`.
    pub fn edges_mut(&mut self) -> &mut BTreeMap<ImagePairId, PoseGraphEdge> {
        &mut self.edges
    }

    /// `NumEdges()`: the number of edges, valid or not.
    pub fn num_edges(&self) -> usize {
        self.edges.len()
    }

    /// `Empty()`: whether there are no edges.
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }

    /// `Clear()`: removes every edge.
    pub fn clear(&mut self) {
        self.edges.clear();
    }

    /// `Load(corr_graph)`: one edge per image pair with matches whose two-view geometry has a
    /// relative pose, with its number of matches.
    pub fn load(&mut self, corr_graph: &CorrespondenceGraph) -> Result<()> {
        for (pair_id, num_matches) in corr_graph.num_matches_between_all_images() {
            let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
            let two_view_geometry =
                corr_graph.extract_two_view_geometry(image_id1, image_id2, false)?;
            if let Some(cam2_from_cam1) = two_view_geometry.cam2_from_cam1 {
                let mut edge = PoseGraphEdge::new(cam2_from_cam1);
                // point2D_t -> int, as COLMAP's assignment narrows it.
                edge.num_matches = num_matches as i32;
                self.add_edge(image_id1, image_id2, edge)?;
            }
        }
        Ok(())
    }

    /// `AddEdge`: stores `edge` for the pair, inverted if the ids are given in swapped order,
    /// and returns the stored edge. Fails if the pair already has an edge.
    pub fn add_edge(
        &mut self,
        image_id1: ImageId,
        image_id2: ImageId,
        mut edge: PoseGraphEdge,
    ) -> Result<&mut PoseGraphEdge> {
        if should_swap_image_pair(image_id1, image_id2) {
            edge.invert();
        }
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        match self.edges.entry(pair_id) {
            std::collections::btree_map::Entry::Occupied(_) => Err(pair_error(
                ErrorKind::RuntimeError,
                "Image pair already exists",
                image_id1,
                image_id2,
            )),
            std::collections::btree_map::Entry::Vacant(entry) => Ok(entry.insert(edge)),
        }
    }

    /// `HasEdge`: whether the pair has an edge, in either order.
    pub fn has_edge(&self, image_id1: ImageId, image_id2: ImageId) -> Result<bool> {
        Ok(self
            .edges
            .contains_key(&image_pair_to_pair_id(image_id1, image_id2)?))
    }

    fn missing(image_id1: ImageId, image_id2: ImageId) -> ColmapError {
        pair_error(
            ErrorKind::OutOfRange,
            "Image pair does not exist",
            image_id1,
            image_id2,
        )
    }

    /// `EdgeRef` (const): the stored edge (not inverted) and whether the ids were given in
    /// swapped order. Fails for a missing pair.
    pub fn edge_ref(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<(&PoseGraphEdge, bool)> {
        let swapped = should_swap_image_pair(image_id1, image_id2);
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        self.edges
            .get(&pair_id)
            .map(|edge| (edge, swapped))
            .ok_or_else(|| Self::missing(image_id1, image_id2))
    }

    /// `EdgeRef` (non-const): as [`Self::edge_ref`], with the stored edge mutable.
    pub fn edge_ref_mut(
        &mut self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<(&mut PoseGraphEdge, bool)> {
        let swapped = should_swap_image_pair(image_id1, image_id2);
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        self.edges
            .get_mut(&pair_id)
            .map(|edge| (edge, swapped))
            .ok_or_else(|| Self::missing(image_id1, image_id2))
    }

    /// `GetEdge`: a copy of the edge oriented from `image_id1` to `image_id2`. Fails for a
    /// missing pair.
    pub fn get_edge(&self, image_id1: ImageId, image_id2: ImageId) -> Result<PoseGraphEdge> {
        let (edge, swapped) = self.edge_ref(image_id1, image_id2)?;
        let mut result = *edge;
        if swapped {
            result.invert();
        }
        Ok(result)
    }

    /// `DeleteEdge`: removes the pair's edge; false if there was none.
    pub fn delete_edge(&mut self, image_id1: ImageId, image_id2: ImageId) -> Result<bool> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        Ok(self.edges.remove(&pair_id).is_some())
    }

    /// `UpdateEdge`: replaces the pair's edge with `edge` (inverted if the ids are swapped).
    /// Fails if the pair has no edge.
    pub fn update_edge(
        &mut self,
        image_id1: ImageId,
        image_id2: ImageId,
        mut edge: PoseGraphEdge,
    ) -> Result<()> {
        if should_swap_image_pair(image_id1, image_id2) {
            edge.invert();
        }
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        match self.edges.get_mut(&pair_id) {
            Some(stored) => {
                *stored = edge;
                Ok(())
            }
            None => Err(pair_error(
                ErrorKind::RuntimeError,
                "Image pair does not exist",
                image_id1,
                image_id2,
            )),
        }
    }

    /// `IsValid`: whether the pair has an edge and it is valid.
    pub fn is_valid(&self, pair_id: ImagePairId) -> bool {
        self.edges.get(&pair_id).is_some_and(|edge| edge.valid)
    }

    /// `SetValidEdge`: marks an existing edge valid.
    pub fn set_valid_edge(&mut self, pair_id: ImagePairId) -> Result<()> {
        self.set_edge_validity(pair_id, true)
    }

    /// `SetInvalidEdge`: marks an existing edge invalid.
    pub fn set_invalid_edge(&mut self, pair_id: ImagePairId) -> Result<()> {
        self.set_edge_validity(pair_id, false)
    }

    fn set_edge_validity(&mut self, pair_id: ImagePairId, valid: bool) -> Result<()> {
        let edge = self.edges.get_mut(&pair_id);
        crate::check!(edge.is_some(), "Edge does not exist");
        if let Some(edge) = edge {
            edge.valid = valid;
        }
        Ok(())
    }

    /// `ValidEdges()`: the valid edges only, in ascending pair id order.
    pub fn valid_edges(&self) -> impl Iterator<Item = (&ImagePairId, &PoseGraphEdge)> + '_ {
        self.edges.iter().filter(|(_, edge)| edge.valid)
    }

    /// `ConnectedFrameComponents`: the connected components as sets of frame ids, largest
    /// first, equally large ones by smallest frame id (see the module header). With
    /// `filter_unregistered`, only edges between frames with a pose count.
    pub fn connected_frame_components(
        &self,
        reconstruction: &Reconstruction,
        filter_unregistered: bool,
    ) -> Result<Vec<HashSet<FrameId>>> {
        let (nodes, edges) = self.build_frame_graph(reconstruction, filter_unregistered)?;
        if nodes.is_empty() {
            return Ok(Vec::new());
        }
        let mut components = find_connected_components(&nodes, &edges);
        // COLMAP's std::sort by size is unstable; the stable sort keeps equally large
        // components in the order find_connected_components gives them, which is by
        // smallest frame id since the nodes are sorted.
        components.sort_by_key(|component| std::cmp::Reverse(component.len()));
        Ok(components
            .into_iter()
            .map(|component| component.into_iter().collect())
            .collect())
    }

    /// `ConnectedImageIdsForFrameComponents`: the image ids of each connected frame
    /// component, in the order of [`Self::connected_frame_components`].
    pub fn connected_image_ids_for_frame_components(
        &self,
        reconstruction: &Reconstruction,
        filter_unregistered: bool,
    ) -> Result<Vec<HashSet<ImageId>>> {
        let frame_components =
            self.connected_frame_components(reconstruction, filter_unregistered)?;
        // Lookup only, never iterated.
        let mut frame_to_component: HashMap<FrameId, usize> = HashMap::new();
        for (comp, component) in frame_components.iter().enumerate() {
            for &frame_id in component {
                frame_to_component.insert(frame_id, comp);
            }
        }
        let mut image_ids = vec![HashSet::new(); frame_components.len()];
        for (&image_id, image) in reconstruction.images() {
            if let Some(&comp) = frame_to_component.get(&image.frame_id()) {
                image_ids[comp].insert(image_id);
            }
        }
        Ok(image_ids)
    }

    /// `LargestConnectedFrameComponent`: the frame ids of the largest connected component
    /// (of equally large ones, the one with the smallest frame id). With
    /// `filter_unregistered`, only edges between frames with a pose count.
    pub fn largest_connected_frame_component(
        &self,
        reconstruction: &Reconstruction,
        filter_unregistered: bool,
    ) -> Result<HashSet<FrameId>> {
        let (nodes, edges) = self.build_frame_graph(reconstruction, filter_unregistered)?;
        if nodes.is_empty() {
            return Ok(HashSet::new());
        }
        Ok(find_largest_connected_component(&nodes, &edges)
            .into_iter()
            .collect())
    }

    /// `InvalidatePairsOutsideActiveImageIds`: marks image pairs invalid if either image is
    /// not in the active set.
    pub fn invalidate_pairs_outside_active_image_ids(
        &mut self,
        active_image_ids: &HashSet<ImageId>,
    ) -> Result<()> {
        for (&pair_id, edge) in self.edges.iter_mut() {
            let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
            if !active_image_ids.contains(&image_id1) || !active_image_ids.contains(&image_id2) {
                edge.valid = false;
            }
        }
        Ok(())
    }

    // BuildFrameGraph (pose_graph.cc's anonymous namespace): the frames of the valid edges'
    // images, skipping edges with an unposed frame when filter_unregistered, and one frame
    // edge per image edge. COLMAP's node set is a FlatHashSet; here the nodes come in
    // ascending id order so that the component order is reproducible (see the module header).
    fn build_frame_graph(
        &self,
        reconstruction: &Reconstruction,
        filter_unregistered: bool,
    ) -> Result<FrameGraph> {
        let mut nodes = BTreeSet::new();
        let mut edges = Vec::new();
        for (&pair_id, _) in self.valid_edges() {
            let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
            let frame_id1 = reconstruction.image(image_id1)?.frame_id();
            let frame_id2 = reconstruction.image(image_id2)?.frame_id();
            if filter_unregistered
                && (!reconstruction.frame(frame_id1)?.has_pose()
                    || !reconstruction.frame(frame_id2)?.has_pose())
            {
                continue;
            }
            nodes.insert(frame_id1);
            nodes.insert(frame_id2);
            edges.push((frame_id1, frame_id2));
        }
        Ok((nodes.into_iter().collect(), edges))
    }
}
