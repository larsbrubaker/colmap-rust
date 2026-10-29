//! Port of COLMAP's `RotationEstimator` (`rotation_averaging.h/.cc`): `EstimateRotations`,
//! the stratified gravity-aligned pre-solve, the solve itself and the maximum spanning tree
//! initialization. It builds a [`RotationAveragingProblem`] and solves it with
//! [`RotationAveragingSolver`]; the free functions are in [`super::run`]. Port of
//! colmap-sharp's `Estimators/RotationEstimator.cs`.
//!
//! Translation notes:
//! - The spanning tree numbers the active images in ascending id order (COLMAP: `FlatHashSet`
//!   order, which picks the tree's root, node 0) and adds the valid edges in ascending pair
//!   id order; the tree is walked with children in ascending id order; frames are registered
//!   in ascending image id order (the ascending id divergence in the [`super`] module header).
//! - COLMAP's `cams_from_world[parent]` default-constructs the root's entry (identity) the
//!   first time a child of the root is visited; `entry().or_default()` does the same.

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use super::run::{
    all_sensors_from_rig_known, compute_largest_connected_component_image_ids,
    initialize_rig_rotations_from_images, use_gravity,
};
use super::{RotationAveragingProblem, RotationAveragingSolver, RotationEstimatorOptions};
use crate::geometry::{PosePrior, Rigid3d};
use crate::math::spanning_tree::compute_maximum_spanning_tree;
use crate::scene::pose_graph::{PoseGraph, PoseGraphEdge};
use crate::scene::reconstruction::Reconstruction;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{pair_id_to_image_pair, ImageId, SensorType};
use crate::Result;

/// High-level interface for rotation averaging: combines problem setup and solving into a
/// single call. Port of `colmap::RotationEstimator`.
#[derive(Clone, Debug)]
pub struct RotationEstimator {
    options: RotationEstimatorOptions,
}

impl RotationEstimator {
    /// An estimator with a copy of `options`.
    pub fn new(options: &RotationEstimatorOptions) -> Self {
        Self { options: *options }
    }

    /// `EstimateRotations`: estimates the global orientations of all views, solving rotation
    /// averaging over `active_image_ids` and registering their frames with the computed
    /// poses (unknown translation). Returns true on successful estimation.
    pub fn estimate_rotations(
        &self,
        pose_graph: &PoseGraph,
        pose_priors: &[PosePrior],
        active_image_ids: &HashSet<ImageId>,
        reconstruction: &mut Reconstruction,
    ) -> Result<bool> {
        let use_gravity = use_gravity(&self.options, pose_priors);
        if use_gravity && !all_sensors_from_rig_known(reconstruction) {
            return Ok(false);
        }

        // Handle stratified solving for mixed gravity systems.
        if use_gravity
            && self.options.use_stratified
            && !self.maybe_solve_gravity_aligned_subset(
                pose_graph,
                pose_priors,
                active_image_ids,
                reconstruction,
            )?
        {
            return Ok(false);
        }

        // Solve the full system.
        if !self.solve_rotation_averaging(
            pose_graph,
            pose_priors,
            active_image_ids,
            reconstruction,
        )? {
            return Ok(false);
        }

        // Register frames with computed poses.
        let mut image_ids: Vec<ImageId> = active_image_ids.iter().copied().collect();
        image_ids.sort_unstable();
        for image_id in image_ids {
            let frame_id = reconstruction.image(image_id)?.frame_id();
            crate::check!(reconstruction.frame(frame_id)?.has_pose());
            reconstruction.register_frame(frame_id)?;
        }
        Ok(true)
    }

    /// `MaybeSolveGravityAlignedSubset`: solves 1-DOF rotation averaging on the
    /// gravity-aligned subset, the first phase of stratified solving for mixed gravity
    /// systems, when that subset is a meaningful part of the problem.
    fn maybe_solve_gravity_aligned_subset(
        &self,
        pose_graph: &PoseGraph,
        pose_priors: &[PosePrior],
        active_image_ids: &HashSet<ImageId>,
        reconstruction: &mut Reconstruction,
    ) -> Result<bool> {
        // Build map from image to pose prior (a later prior of one image replaces an earlier
        // one, as COLMAP's operator[] assignment does). Lookup only.
        let mut image_to_pose_prior: HashMap<ImageId, &PosePrior> = HashMap::new();
        for pose_prior in pose_priors {
            if pose_prior.corr_data_id.sensor_id.sensor_type == SensorType::Camera {
                // `image_t` key: the 64-bit data id truncated, as the implicit conversion does.
                image_to_pose_prior.insert(pose_prior.corr_data_id.id as ImageId, pose_prior);
            }
        }
        let has_gravity = |image_id: ImageId| {
            image_to_pose_prior
                .get(&image_id)
                .is_some_and(|prior| prior.has_gravity())
        };

        // Separate pairs into gravity-aligned subset.
        let mut gravity_pose_graph = PoseGraph::new();
        let mut num_total_pairs = 0usize;
        for (&pair_id, edge) in pose_graph.valid_edges() {
            let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
            if !reconstruction.exists_image(image_id1) || !reconstruction.exists_image(image_id2) {
                continue;
            }
            if !active_image_ids.contains(&image_id1) || !active_image_ids.contains(&image_id2) {
                continue;
            }
            num_total_pairs += 1;
            if has_gravity(image_id1) && has_gravity(image_id2) {
                // A fresh edge: only the relative pose carries over (no matches, valid).
                gravity_pose_graph
                    .edges_mut()
                    .insert(pair_id, PoseGraphEdge::new(edge.cam2_from_cam1));
            }
        }

        let num_gravity_pairs = gravity_pose_graph.num_edges();

        // Only solve if we have a meaningful subset. Skip if no gravity pairs, or if most
        // pairs (>95%) have gravity since solving the subset separately provides little
        // benefit over the full system.
        let should_solve =
            num_gravity_pairs > 0 && num_gravity_pairs as f64 <= num_total_pairs as f64 * 0.95;
        if !should_solve {
            return Ok(true);
        }

        let mut gravity_reconstruction = reconstruction.clone();

        // Compute largest connected component for gravity subset.
        let gravity_image_ids = compute_largest_connected_component_image_ids(
            &gravity_pose_graph,
            &gravity_reconstruction,
            false,
        )?;
        gravity_pose_graph.invalidate_pairs_outside_active_image_ids(&gravity_image_ids)?;

        if !self.solve_rotation_averaging(
            &gravity_pose_graph,
            pose_priors,
            &gravity_image_ids,
            &mut gravity_reconstruction,
        )? {
            return Ok(false);
        }

        for (&gravity_frame_id, gravity_frame) in gravity_reconstruction.frames() {
            let Some(rig_from_world) = gravity_frame.maybe_rig_from_world() else {
                continue;
            };
            reconstruction
                .frame_mut(gravity_frame_id)?
                .set_rig_from_world(rig_from_world);
        }

        if self.options.refine_sensor_from_rig {
            for (&gravity_rig_id, gravity_rig) in gravity_reconstruction.rigs() {
                for (&sensor_id, sensor_from_rig) in gravity_rig.non_ref_sensors() {
                    let Some(sensor_from_rig) = sensor_from_rig else {
                        continue;
                    };
                    reconstruction
                        .rig_mut(gravity_rig_id)?
                        .set_sensor_from_rig(sensor_id, *sensor_from_rig)?;
                }
            }
        }
        Ok(true)
    }

    /// `SolveRotationAveraging`: the core rotation averaging solve.
    fn solve_rotation_averaging(
        &self,
        pose_graph: &PoseGraph,
        pose_priors: &[PosePrior],
        active_image_ids: &HashSet<ImageId>,
        reconstruction: &mut Reconstruction,
    ) -> Result<bool> {
        // Initialize rotations from maximum spanning tree. Note that without initialization,
        // the gravity-aligned rotation averaging is prone to random flips by 180deg.
        if !self.options.skip_initialization {
            self.initialize_from_maximum_spanning_tree(
                pose_graph,
                active_image_ids,
                reconstruction,
            )?;
        }

        // Build the optimization problem.
        let mut problem = RotationAveragingProblem::new(
            pose_graph,
            pose_priors,
            &self.options,
            active_image_ids,
            reconstruction,
        )?;

        // Solve and apply results.
        if !RotationAveragingSolver::new(&self.options).solve(&mut problem)? {
            return Ok(false);
        }
        problem.apply_results_to_reconstruction(reconstruction)?;
        Ok(true)
    }

    /// `InitializeFromMaximumSpanningTree`: initializes the rotations by chaining the relative
    /// rotations along the maximum spanning tree (weighted by inlier count) from its root.
    fn initialize_from_maximum_spanning_tree(
        &self,
        pose_graph: &PoseGraph,
        active_image_ids: &HashSet<ImageId>,
        reconstruction: &mut Reconstruction,
    ) -> Result<()> {
        // Compute maximum spanning tree over active images.
        let (root, parents) =
            compute_maximum_pose_graph_spanning_tree(pose_graph, active_image_ids)?;
        crate::check!(active_image_ids.contains(&root));

        // Iterate through the tree to initialize the rotation. Establish child info.
        let mut children: BTreeMap<ImageId, Vec<ImageId>> = BTreeMap::new();
        for &image_id in reconstruction.images().keys() {
            if active_image_ids.contains(&image_id) {
                children.insert(image_id, Vec::new());
            }
        }
        for (&child, &parent) in &parents {
            if root == child {
                continue;
            }
            children.entry(parent).or_default().push(child);
        }

        let mut indexes: VecDeque<ImageId> = VecDeque::new();
        indexes.push_back(root);

        let mut cams_from_world: BTreeMap<ImageId, Rigid3d> = BTreeMap::new();
        while let Some(curr) = indexes.pop_front() {
            // Add all children into the tree.
            if let Some(curr_children) = children.get(&curr) {
                indexes.extend(curr_children.iter().copied());
            }
            // If it is root, then fix it to be the original estimation.
            if curr == root {
                continue;
            }

            // Directly use the relative pose for estimation rotation.
            // GetEdge(parent, curr) returns curr_from_parent.
            let parent = *parents.get(&curr).ok_or_else(|| {
                ColmapError::new(ErrorKind::OutOfRange, format!("Image {curr} has no parent"))
            })?;
            let edge = pose_graph.get_edge(parent, curr)?;
            let parent_from_world = *cams_from_world.entry(parent).or_default();
            let rotation = (edge.cam2_from_cam1 * parent_from_world).rotation;
            cams_from_world.entry(curr).or_default().rotation = rotation;
        }

        initialize_rig_rotations_from_images(
            &cams_from_world,
            reconstruction,
            self.options.refine_sensor_from_rig,
        )?;
        Ok(())
    }
}

/// `ComputeMaximumPoseGraphSpanningTree`: the maximum spanning tree of the pose graph's
/// valid edges between `image_ids`, weighted by inlier count. Returns the root image id and
/// each other tree image's parent.
fn compute_maximum_pose_graph_spanning_tree(
    pose_graph: &PoseGraph,
    image_ids: &HashSet<ImageId>,
) -> Result<(ImageId, BTreeMap<ImageId, ImageId>)> {
    // Build mapping between image_id and contiguous indices.
    let mut idx_to_image_id: Vec<ImageId> = image_ids.iter().copied().collect();
    idx_to_image_id.sort_unstable();
    // Lookup only, never iterated.
    let image_id_to_idx: HashMap<ImageId, i32> = idx_to_image_id
        .iter()
        .enumerate()
        .map(|(idx, &image_id)| (image_id, idx as i32))
        .collect();

    // Build edges and weights from the pose graph.
    let mut edges: Vec<(i32, i32)> = Vec::with_capacity(pose_graph.num_edges());
    let mut weights: Vec<f32> = Vec::with_capacity(pose_graph.num_edges());
    for (&pair_id, edge) in pose_graph.valid_edges() {
        let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
        let (Some(&idx1), Some(&idx2)) = (
            image_id_to_idx.get(&image_id1),
            image_id_to_idx.get(&image_id2),
        ) else {
            continue;
        };
        edges.push((idx1, idx2));
        weights.push(edge.num_matches as f32);
    }

    // Compute spanning tree using generic algorithm.
    let tree = compute_maximum_spanning_tree(idx_to_image_id.len() as i32, &edges, &weights, 0)?;

    // Convert back to image_id based parent map.
    let mut parents = BTreeMap::new();
    for (i, &parent) in tree.parents.iter().enumerate() {
        if parent >= 0 {
            parents.insert(idx_to_image_id[i], idx_to_image_id[parent as usize]);
        }
    }
    let root = usize::try_from(tree.root)
        .ok()
        .and_then(|root| idx_to_image_id.get(root).copied())
        .ok_or_else(|| ColmapError::new(ErrorKind::OutOfRange, "Spanning tree has no root"))?;
    Ok((root, parents))
}
