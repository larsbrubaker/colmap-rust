//! Port of COLMAP's `RotationAveragingProblem` (`rotation_averaging_impl.h/.cc`), setup half:
//! the linear system A x = b of global rotation averaging (x: rig_from_world rotations of the
//! active frames plus unknown cam_from_rig rotations, in tangent space; b: residuals of the
//! relative rotation constraints of the pose graph's valid edges; A: the sparse constraint
//! matrix with one gauge-fixing block), built once from the pose graph and the
//! reconstruction. `problem_state.rs` holds the per-iteration half (residuals, state update,
//! step size, write-back); [`super::solver`] solves it. Port of colmap-sharp's
//! `Estimators/RotationAveragingProblem.cs`.
//!
//! Translation notes:
//! - Active frames and estimated cameras are laid out in ascending id order, pair
//!   constraints (rows) in ascending pair id order, and the gauge is fixed at the smallest-id
//!   (gravity-aligned, if any) active frame (the ascending id divergence in the [`super`]
//!   module header).
//! - COLMAP looks up a pair's frame parameter with `operator[]`, which silently inserts
//!   index 0 for a frame outside the active set (and `ComputeResiduals` later throws from
//!   `.at()`). Callers invalidate such pairs first
//!   (`PoseGraph::invalidate_pairs_outside_active_image_ids`); here the constructor fails for
//!   them instead of corrupting the layout (the same module header note).
//! - `std::variant<GravityAligned1DOF, Full3DOF>` is [`RotationConstraint`]; the `-1`
//!   "known cam_from_rig" parameter index is `None`.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use super::RotationAveragingReweighting;
use super::{camera_sensor_id, has_nan3, is_ref_in_frame, RotationEstimatorOptions};
use crate::geometry::pose::{
    gravity_aligned_rotation, rotation_matrix_to_angle_axis, y_axis_angle_from_rotation,
};
use crate::geometry::{PosePrior, Rigid3d};
use crate::linalg::VectorXd;
use crate::linalg::{AngleAxisd, Matrix3d, Quaterniond, SparseMatrixCsc, SparseTriplet, Vector3d};
use crate::scene::pose_graph::PoseGraph;
use crate::scene::reconstruction::Reconstruction;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{
    pair_id_to_image_pair, CameraId, FrameId, ImageId, ImagePairId, RigId, SensorId, SensorType,
    INVALID_FRAME_ID,
};
use crate::Result;

/// The relative rotation constraint of one image pair
/// (`std::variant<GravityAligned1DOF, Full3DOF>`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RotationConstraint {
    /// `GravityAligned1DOF`: 1-DOF constraint when both frames have gravity priors.
    GravityAligned1Dof {
        /// Relative Y-axis rotation.
        angle_cam2_from_cam1: f64,
        /// Squared error in x,z axes for IRLS.
        xz_error: f64,
    },
    /// `Full3DOF`: 3-DOF constraint for the general case (no gravity or partial gravity).
    Full3Dof {
        /// The relative rotation between the two rigs (gravity-aligned where known).
        r_cam2_from_cam1: Matrix3d,
    },
}

impl RotationConstraint {
    /// Rows of A the constraint owns: 1 for 1-DOF, 3 for 3-DOF.
    pub fn num_rows(&self) -> usize {
        match self {
            RotationConstraint::GravityAligned1Dof { .. } => 1,
            RotationConstraint::Full3Dof { .. } => 3,
        }
    }
}

/// Port of `RotationAveragingProblem::PairConstraint`: the preprocessed constraint of an
/// image pair, built once during setup.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairConstraint {
    /// First image of the pair.
    pub image_id1: ImageId,
    /// Second image of the pair.
    pub image_id2: ImageId,
    /// Starting row in matrix A (1 row for 1-DOF, 3 rows for 3-DOF).
    pub row_index: usize,
    /// Column of the unknown cam1_from_rig rotation (`None` if known).
    pub cam1_from_rig_param_idx: Option<usize>,
    /// Column of the unknown cam2_from_rig rotation (`None` if known).
    pub cam2_from_rig_param_idx: Option<usize>,
    /// The 1-DOF or 3-DOF constraint.
    pub constraint: RotationConstraint,
}

/// Rotation averaging problem formulated as linear system A*x = b where
/// x = [rig_from_world rotations, unknown cam_from_rig rotations],
/// b = residuals from relative rotation constraints and
/// A = sparse matrix encoding constraint equations.
/// Port of `colmap::RotationAveragingProblem`.
#[derive(Clone, Debug)]
pub struct RotationAveragingProblem {
    pub(super) options: RotationEstimatorOptions,
    // Pose priors indexed by frame ID.
    pub(super) frame_to_pose_prior: BTreeMap<FrameId, PosePrior>,
    // Linear system components: matrix A and vector b.
    pub(super) constraint_matrix: SparseMatrixCsc,
    pub(super) residuals: VectorXd,
    // Optional diagonal of the reweighting operator W applied to the residual space (rows of
    // A and b); the solver works on the reweighted system min ||W (A x - b)||. Populated when
    // the reweighting scheme is not Uniform.
    pub(super) residual_reweighting: Option<VectorXd>,
    // Current rotation estimates in tangent space (angle-axis).
    pub(super) estimated_rotations: VectorXd,
    // Parameter index mappings.
    pub(super) frame_id_to_param_idx: BTreeMap<FrameId, usize>,
    pub(super) camera_id_to_param_idx: BTreeMap<CameraId, usize>,
    // Preprocessed constraints for each image pair.
    pub(super) pair_constraints: BTreeMap<ImagePairId, PairConstraint>,
    // Gauge fixing (removes rotational ambiguity).
    pub(super) fixed_frame_id: FrameId,
    pub(super) fixed_frame_rotation: Vector3d,
    pub(super) num_gauge_fixing_residuals: usize, // 1 for gravity-aligned, 3 otherwise.
    // Cached lookups for ComputeResiduals and UpdateState.
    pub(super) image_id_to_frame_id: BTreeMap<ImageId, FrameId>,
    camera_id_to_rig_id: BTreeMap<CameraId, RigId>,
    pub(super) camera_to_frame_ids: BTreeMap<CameraId, Vec<FrameId>>,
    // Active frames for the current solve.
    active_frame_ids: BTreeSet<FrameId>,
}

/// `angle_axis.angle() * angle_axis.axis()` of `Eigen::AngleAxisd(rotation)`.
pub(super) fn angle_axis_vector(rotation: Quaterniond) -> Vector3d {
    let angle_axis = AngleAxisd::from_quaternion(rotation);
    angle_axis.axis * angle_axis.angle
}

/// `v.segment<3>(start)`.
pub(super) fn segment3(v: &VectorXd, start: usize) -> Vector3d {
    Vector3d::new(v[start], v[start + 1], v[start + 2])
}

/// `v.segment<3>(start) = value`.
pub(super) fn set_segment3(v: &mut VectorXd, start: usize, value: Vector3d) {
    v[start] = value.x;
    v[start + 1] = value.y;
    v[start + 2] = value.z;
}

fn missing(what: &str, id: impl std::fmt::Display) -> ColmapError {
    ColmapError::new(
        ErrorKind::OutOfRange,
        format!("{what} {id} is not in the problem"),
    )
}

impl RotationAveragingProblem {
    /// Builds the problem over the valid edges of `pose_graph` between `active_image_ids`,
    /// initializing the unknowns from the reconstruction's current rig_from_world and
    /// cam_from_rig rotations.
    pub fn new(
        pose_graph: &PoseGraph,
        pose_priors: &[PosePrior],
        options: &RotationEstimatorOptions,
        active_image_ids: &HashSet<ImageId>,
        reconstruction: &Reconstruction,
    ) -> Result<Self> {
        let mut problem = Self {
            options: *options,
            frame_to_pose_prior: BTreeMap::new(),
            constraint_matrix: SparseMatrixCsc::zero(0, 0),
            residuals: VectorXd::zeros(0),
            residual_reweighting: None,
            estimated_rotations: VectorXd::zeros(0),
            frame_id_to_param_idx: BTreeMap::new(),
            camera_id_to_param_idx: BTreeMap::new(),
            pair_constraints: BTreeMap::new(),
            fixed_frame_id: INVALID_FRAME_ID,
            fixed_frame_rotation: Vector3d::new(0.0, 0.0, 0.0),
            num_gauge_fixing_residuals: 3,
            image_id_to_frame_id: BTreeMap::new(),
            camera_id_to_rig_id: BTreeMap::new(),
            camera_to_frame_ids: BTreeMap::new(),
            active_frame_ids: BTreeSet::new(),
        };

        // Derive active_frame_ids from active_image_ids, and cache mappings.
        let mut sorted_image_ids: Vec<ImageId> = active_image_ids.iter().copied().collect();
        sorted_image_ids.sort_unstable();
        for image_id in sorted_image_ids {
            let image = reconstruction.image(image_id)?;
            let frame_id = image.frame_id();
            problem.active_frame_ids.insert(frame_id);
            problem.image_id_to_frame_id.insert(image_id, frame_id);
            let rig_id = reconstruction.frame(frame_id)?.rig_id();
            problem
                .camera_id_to_rig_id
                .insert(image.camera_id(), rig_id);
        }

        problem.frame_to_pose_prior = extract_frame_to_pose_prior(reconstruction, pose_priors)?;

        let num_params = problem.allocate_parameters(reconstruction)?;
        problem.build_pair_constraints(pose_graph, reconstruction)?;
        problem.build_constraint_matrix(num_params, pose_graph, reconstruction)?;
        Ok(problem)
    }

    /// `ConstraintMatrix()`: the constraint matrix A.
    pub fn constraint_matrix(&self) -> &SparseMatrixCsc {
        &self.constraint_matrix
    }

    /// `Residuals()`: the residual vector b.
    pub fn residuals(&self) -> &VectorXd {
        &self.residuals
    }

    /// `ResidualReweighting()`: the diagonal of the residual-space reweighting operator W
    /// (one weight per residual row), or `None` when no reweighting is configured.
    pub fn residual_reweighting(&self) -> Option<&VectorXd> {
        self.residual_reweighting.as_ref()
    }

    /// `WeightedConstraintMatrix()`: A with the reweighting operator applied to its rows
    /// (W * A), or the plain constraint matrix when no reweighting is configured.
    pub fn weighted_constraint_matrix(&self) -> SparseMatrixCsc {
        let Some(reweighting) = &self.residual_reweighting else {
            return self.constraint_matrix.clone();
        };
        let mut weighted = self.constraint_matrix.clone();
        let rows = weighted.row_indices().to_vec();
        for (value, row) in weighted.values_mut().iter_mut().zip(rows) {
            *value *= reweighting[row];
        }
        weighted
    }

    /// `WeightedResiduals()`: b with the reweighting operator applied (W * b), or the plain
    /// residuals when no reweighting is configured.
    pub fn weighted_residuals(&self) -> VectorXd {
        let Some(reweighting) = &self.residual_reweighting else {
            return self.residuals.clone();
        };
        let weighted: Vec<f64> = reweighting
            .as_slice()
            .iter()
            .zip(self.residuals.as_slice())
            .map(|(w, r)| w * r)
            .collect();
        VectorXd::from_vec(weighted)
    }

    /// `NumParameters()`: the number of unknowns (columns of A).
    pub fn num_parameters(&self) -> usize {
        self.constraint_matrix.cols()
    }

    /// `NumResiduals()`: the number of residual rows of A.
    pub fn num_residuals(&self) -> usize {
        self.constraint_matrix.rows()
    }

    /// `NumGaugeFixingResiduals()`: rows of the gauge-fixing block at the end of A (1 or 3).
    pub fn num_gauge_fixing_residuals(&self) -> usize {
        self.num_gauge_fixing_residuals
    }

    /// `PairConstraints()`: the pair constraints by pair id, in ascending pair id order.
    pub fn pair_constraints(&self) -> &BTreeMap<ImagePairId, PairConstraint> {
        &self.pair_constraints
    }

    /// `GetFrameGravityOrNull`: the frame's gravity prior, if it has a valid one.
    pub(super) fn frame_gravity(&self, frame_id: FrameId) -> Option<Vector3d> {
        self.frame_to_pose_prior
            .get(&frame_id)
            .filter(|prior| prior.has_gravity())
            .map(|prior| prior.gravity)
    }

    /// `HasFrameGravity`: the frame has a gravity prior and gravity mode is enabled.
    pub(super) fn has_frame_gravity(&self, frame_id: FrameId) -> bool {
        self.options.use_gravity && self.frame_gravity(frame_id).is_some()
    }

    /// `frame_id_to_param_idx_.at(frame_id)`.
    pub(super) fn frame_param_idx(&self, frame_id: FrameId) -> Result<usize> {
        self.frame_id_to_param_idx
            .get(&frame_id)
            .copied()
            .ok_or_else(|| missing("Frame", frame_id))
    }

    /// `image_id_to_frame_id_.at(image_id)`.
    pub(super) fn image_frame_id(&self, image_id: ImageId) -> Result<FrameId> {
        self.image_id_to_frame_id
            .get(&image_id)
            .copied()
            .ok_or_else(|| missing("Image", image_id))
    }

    /// `AllocateParameters`: allocates parameter indices for frames and cameras and
    /// initializes the rotations. Returns the number of parameters.
    fn allocate_parameters(&mut self, reconstruction: &Reconstruction) -> Result<usize> {
        // Identify cameras that need cam_from_rig estimation (non-reference cameras without
        // calibrated extrinsics).
        let mut cam_from_rig_rotations: BTreeMap<CameraId, Vector3d> = BTreeMap::new();
        let mut estimated_camera_ids: BTreeSet<CameraId> = BTreeSet::new();
        if self.options.refine_sensor_from_rig {
            for (&camera_id, &rig_id) in &self.camera_id_to_rig_id {
                let sensor_id = SensorId::new(SensorType::Camera, camera_id);
                let rig = reconstruction.rig(rig_id)?;
                if rig.is_ref_sensor(sensor_id) {
                    continue;
                }
                let cam_from_rig = rig.maybe_sensor_from_rig(sensor_id)?;
                let unknown = cam_from_rig.is_none_or(|pose| has_nan3(pose.translation));
                if unknown && estimated_camera_ids.insert(camera_id) {
                    // Marked for allocation (actual index set below).
                    if let Some(pose) = cam_from_rig {
                        cam_from_rig_rotations.insert(camera_id, angle_axis_vector(pose.rotation));
                    }
                }
            }
        }

        // Cache camera_id -> frame_id mapping for UpdateState cam_from_rig averaging.
        for &frame_id in &self.active_frame_ids {
            for data_id in reconstruction.frame(frame_id)?.image_ids() {
                // `static_cast<image_t>` of the 64-bit data id.
                let camera_id = reconstruction.image(data_id.id as ImageId)?.camera_id();
                if estimated_camera_ids.contains(&camera_id) {
                    self.camera_to_frame_ids
                        .entry(camera_id)
                        .or_default()
                        .push(frame_id);
                }
            }
        }

        // Allocate frame parameters and cache frame info.
        let mut rotations: Vec<f64> =
            Vec::with_capacity(3 * (self.active_frame_ids.len() + estimated_camera_ids.len()));
        let active_frame_ids: Vec<FrameId> = self.active_frame_ids.iter().copied().collect();
        for &frame_id in &active_frame_ids {
            let frame = reconstruction.frame(frame_id)?;
            self.frame_id_to_param_idx.insert(frame_id, rotations.len());

            match self
                .frame_gravity(frame_id)
                .filter(|_| self.options.use_gravity)
            {
                Some(gravity) => {
                    // Gravity-aligned frame: 1-DOF (Y-axis rotation only).
                    let rig_from_world_rotation = frame
                        .maybe_rig_from_world()
                        .map_or(Matrix3d::identity(), |pose| {
                            pose.rotation.to_rotation_matrix()
                        });
                    let angle = y_axis_angle_from_rotation(
                        &(gravity_aligned_rotation(gravity)?.transpose() * rig_from_world_rotation),
                    );
                    rotations.push(angle);

                    // Use first gravity-aligned frame as fixed frame.
                    if self.fixed_frame_id == INVALID_FRAME_ID {
                        self.fixed_frame_rotation = Vector3d::new(0.0, angle, 0.0);
                        self.fixed_frame_id = frame_id;
                        self.num_gauge_fixing_residuals = 1;
                    }
                }
                None => {
                    // General frame: 3-DOF.
                    let rig_from_world = frame
                        .maybe_rig_from_world()
                        .map_or(Vector3d::new(0.0, 0.0, 0.0), |pose| {
                            angle_axis_vector(pose.rotation)
                        });
                    rotations.extend_from_slice(&[
                        rig_from_world.x,
                        rig_from_world.y,
                        rig_from_world.z,
                    ]);
                }
            }
        }

        // Allocate camera parameters (for unknown cam_from_rig rotations).
        for &camera_id in &estimated_camera_ids {
            self.camera_id_to_param_idx
                .insert(camera_id, rotations.len());
            let cam_from_rig = cam_from_rig_rotations
                .get(&camera_id)
                .copied()
                .unwrap_or(Vector3d::new(0.0, 0.0, 0.0));
            rotations.extend_from_slice(&[cam_from_rig.x, cam_from_rig.y, cam_from_rig.z]);
        }

        // If no gravity-aligned frame found, use first active frame as fixed.
        if self.fixed_frame_id == INVALID_FRAME_ID {
            if let Some(&frame_id) = active_frame_ids.first() {
                let frame = reconstruction.frame(frame_id)?;
                self.fixed_frame_id = frame_id;
                // Use identity rotation if frame doesn't have a pose yet.
                self.fixed_frame_rotation = match frame.maybe_rig_from_world() {
                    Some(pose) => angle_axis_vector(pose.rotation),
                    None => Vector3d::new(0.0, 0.0, 0.0),
                };
                self.num_gauge_fixing_residuals = 3;
            }
        }

        let num_params = rotations.len();
        self.estimated_rotations = VectorXd::from_vec(rotations);
        Ok(num_params)
    }

    /// `BuildPairConstraints`: the [`PairConstraint`] of each valid image pair.
    fn build_pair_constraints(
        &mut self,
        pose_graph: &PoseGraph,
        reconstruction: &Reconstruction,
    ) -> Result<()> {
        for (&pair_id, edge) in pose_graph.valid_edges() {
            let (image_id1, image_id2) = pair_id_to_image_pair(pair_id)?;
            let image1 = reconstruction.image(image_id1)?;
            let image2 = reconstruction.image(image_id2)?;
            let frame1 = reconstruction.frame(image1.frame_id())?;
            let frame2 = reconstruction.frame(image2.frame_id())?;

            let frame_param_idx1 = self.frame_param_idx(frame1.frame_id())?;
            let frame_param_idx2 = self.frame_param_idx(frame2.frame_id())?;

            // Get known cam_from_rig transforms (None for reference cameras or cameras with
            // unknown cam_from_rig that need to be estimated).
            let known_cam_from_rig =
                |image: &crate::scene::image::Image, rig_id: RigId| -> Result<Option<Rigid3d>> {
                    if is_ref_in_frame(reconstruction, image)?
                        || self.camera_id_to_param_idx.contains_key(&image.camera_id())
                    {
                        return Ok(None);
                    }
                    Ok(Some(
                        reconstruction
                            .rig(rig_id)?
                            .sensor_from_rig(camera_sensor_id(image))?,
                    ))
                };
            let cam1_from_rig1 = known_cam_from_rig(image1, frame1.rig_id())?;
            let cam2_from_rig2 = known_cam_from_rig(image2, frame2.rig_id())?;

            // Skip self-loops within the same frame when both cam_from_rig are known.
            if cam1_from_rig1.is_some()
                && cam2_from_rig2.is_some()
                && frame_param_idx1 == frame_param_idx2
            {
                continue;
            }

            // Compute relative rotation between rigs.
            let mut r_cam2_from_cam1 = (cam2_from_rig2.unwrap_or_default().rotation.inverse()
                * edge.cam2_from_cam1.rotation
                * cam1_from_rig1.unwrap_or_default().rotation)
                .to_rotation_matrix();

            let frame_gravity1 = self.frame_gravity(frame1.frame_id());
            let frame_gravity2 = self.frame_gravity(frame2.frame_id());

            // Apply gravity alignment transformations if available.
            if self.options.use_gravity {
                if let Some(gravity1) = frame_gravity1 {
                    r_cam2_from_cam1 = r_cam2_from_cam1 * gravity_aligned_rotation(gravity1)?;
                }
                if let Some(gravity2) = frame_gravity2 {
                    r_cam2_from_cam1 =
                        gravity_aligned_rotation(gravity2)?.transpose() * r_cam2_from_cam1;
                }
            }

            // Create constraint based on gravity availability.
            let constraint =
                if self.options.use_gravity && frame_gravity1.is_some() && frame_gravity2.is_some()
                {
                    // Both frames have gravity: use 1-DOF constraint.
                    let aa = rotation_matrix_to_angle_axis(&r_cam2_from_cam1);
                    RotationConstraint::GravityAligned1Dof {
                        angle_cam2_from_cam1: aa.y,
                        xz_error: aa.x * aa.x + aa.z * aa.z,
                    }
                } else {
                    // General case: use 3-DOF constraint.
                    RotationConstraint::Full3Dof { r_cam2_from_cam1 }
                };
            self.pair_constraints.insert(
                pair_id,
                PairConstraint {
                    image_id1,
                    image_id2,
                    row_index: 0,
                    cam1_from_rig_param_idx: None,
                    cam2_from_rig_param_idx: None,
                    constraint,
                },
            );
        }
        Ok(())
    }

    /// `BuildConstraintMatrix`: the sparse matrix A and the residual-space reweighting
    /// operator W.
    fn build_constraint_matrix(
        &mut self,
        num_params: usize,
        pose_graph: &PoseGraph,
        reconstruction: &Reconstruction,
    ) -> Result<()> {
        if num_params == 0 {
            return Ok(());
        }

        let mut coeffs: Vec<SparseTriplet> = Vec::new();
        let mut curr_row = 0usize;
        let use_gravity = self.options.use_gravity;
        let pair_ids: Vec<ImagePairId> = self.pair_constraints.keys().copied().collect();
        for pair_id in pair_ids {
            let (image_id1, image_id2) = (
                self.pair_constraints[&pair_id].image_id1,
                self.pair_constraints[&pair_id].image_id2,
            );
            let image1 = reconstruction.image(image_id1)?;
            let image2 = reconstruction.image(image_id2)?;
            // Every constraint's frames are active (build_pair_constraints checked them).
            let frame_param_idx1 = self.frame_param_idx(image1.frame_id())?;
            let frame_param_idx2 = self.frame_param_idx(image2.frame_id())?;

            // Look up camera parameter indices (None if cam_from_rig is known).
            let cam1_param_idx = self
                .camera_id_to_param_idx
                .get(&image1.camera_id())
                .copied();
            let cam2_param_idx = self
                .camera_id_to_param_idx
                .get(&image2.camera_id())
                .copied();
            let gravity1 = use_gravity && self.frame_gravity(image1.frame_id()).is_some();
            let gravity2 = use_gravity && self.frame_gravity(image2.frame_id()).is_some();

            let constraint = self
                .pair_constraints
                .get_mut(&pair_id)
                .ok_or_else(|| missing("Pair", pair_id))?;
            constraint.row_index = curr_row;
            constraint.cam1_from_rig_param_idx = cam1_param_idx;
            constraint.cam2_from_rig_param_idx = cam2_param_idx;

            if let RotationConstraint::GravityAligned1Dof { .. } = constraint.constraint {
                // 1-DOF constraint: single row.
                coeffs.push(SparseTriplet::new(curr_row, frame_param_idx1, -1.0));
                coeffs.push(SparseTriplet::new(curr_row, frame_param_idx2, 1.0));
                curr_row += 1;
                continue;
            }

            // 3-DOF constraint: three rows.
            if gravity1 {
                // Gravity-aligned frame1: only Y-axis contributes.
                coeffs.push(SparseTriplet::new(curr_row + 1, frame_param_idx1, -1.0));
            } else {
                for i in 0..3 {
                    coeffs.push(SparseTriplet::new(curr_row + i, frame_param_idx1 + i, -1.0));
                }
            }
            if gravity2 {
                // Gravity-aligned frame2: only Y-axis contributes.
                coeffs.push(SparseTriplet::new(curr_row + 1, frame_param_idx2, 1.0));
            } else {
                for i in 0..3 {
                    coeffs.push(SparseTriplet::new(curr_row + i, frame_param_idx2 + i, 1.0));
                }
            }

            // Add cam_from_rig terms if being estimated.
            if let Some(cam1_param_idx) = cam1_param_idx {
                for i in 0..3 {
                    coeffs.push(SparseTriplet::new(curr_row + i, cam1_param_idx + i, -1.0));
                }
            }
            if let Some(cam2_param_idx) = cam2_param_idx {
                for i in 0..3 {
                    coeffs.push(SparseTriplet::new(curr_row + i, cam2_param_idx + i, 1.0));
                }
            }
            curr_row += 3;
        }

        // Add gauge-fixing constraint for the fixed frame.
        let fixed_frame_param_idx = self.frame_param_idx(self.fixed_frame_id)?;
        for i in 0..self.num_gauge_fixing_residuals {
            coeffs.push(SparseTriplet::new(
                curr_row + i,
                fixed_frame_param_idx + i,
                1.0,
            ));
        }
        curr_row += self.num_gauge_fixing_residuals;

        // Build sparse matrix.
        self.constraint_matrix = SparseMatrixCsc::from_triplets(curr_row, num_params, &coeffs)?;

        // Initialize residual vector.
        self.residuals = VectorXd::zeros(curr_row);

        // Optionally build the residual-space reweighting operator W. Weights are normalized
        // to (0, 1] for numerical stability (a global scale leaves the minimizer unchanged).
        // Only the rows owned by a pair constraint are reweighted below; the gauge-fixing rows
        // (which are not part of the pair constraints) retain their default weight of 1.
        if self.options.reweighting == RotationAveragingReweighting::InlierMatchCount {
            let num_matches = |pair_id: ImagePairId| -> Result<f64> {
                Ok(pose_graph
                    .edges()
                    .get(&pair_id)
                    .ok_or_else(|| missing("Pose graph edge", pair_id))?
                    .num_matches as f64)
            };
            let mut max_num_matches = 0.0f64;
            for &pair_id in self.pair_constraints.keys() {
                max_num_matches = max_num_matches.max(num_matches(pair_id)?);
            }
            // The inlier match count (normalized to (0, 1]) is used directly as the diagonal
            // reweighting entry.
            let mut reweighting_diagonal = VectorXd::ones(curr_row);
            if max_num_matches > 0.0 {
                for (&pair_id, constraint) in &self.pair_constraints {
                    let reweighting = num_matches(pair_id)? / max_num_matches;
                    for i in 0..constraint.constraint.num_rows() {
                        reweighting_diagonal[constraint.row_index + i] = reweighting;
                    }
                }
            }
            self.residual_reweighting = Some(reweighting_diagonal);
        }
        Ok(())
    }
}

/// `ExtractFrameToPosePrior`: the pose prior of each frame, taken from the camera priors of
/// the frames' reference images. A second prior for one frame is an error.
fn extract_frame_to_pose_prior(
    reconstruction: &Reconstruction,
    pose_priors: &[PosePrior],
) -> Result<BTreeMap<FrameId, PosePrior>> {
    let mut frame_to_pose_prior = BTreeMap::new();
    for pose_prior in pose_priors {
        if pose_prior.corr_data_id.sensor_id.sensor_type != SensorType::Camera {
            continue;
        }
        // `static_cast<image_t>` of the 64-bit data id.
        let image_id = pose_prior.corr_data_id.id as ImageId;
        let Some(image) = reconstruction.images().get(&image_id) else {
            continue;
        };
        if is_ref_in_frame(reconstruction, image)? {
            let frame_id = image.frame_id();
            crate::check!(
                !frame_to_pose_prior.contains_key(&frame_id),
                "Duplicate pose prior for frame{}",
                frame_id
            );
            frame_to_pose_prior.insert(frame_id, *pose_prior);
        }
    }
    Ok(frame_to_pose_prior)
}
