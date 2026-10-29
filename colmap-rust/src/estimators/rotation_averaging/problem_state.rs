//! Port of COLMAP's `RotationAveragingProblem` (`rotation_averaging_impl.cc`), state half:
//! `ComputeResiduals`, `UpdateState`, `AverageStepSize` and `ApplyResultsToReconstruction`,
//! which [`super::solver`] calls each iteration. Setup and the constraint matrix are in
//! `problem.rs`. Port of colmap-sharp's `Estimators/RotationAveragingProblem.State.cs`.
//!
//! Translation notes:
//! - `std::remainder` is `libm::remainder` (both exact IEEE remainders: the quotient rounds
//!   to nearest, ties to even; no transcendental, so no `math::fns` wrapper is needed).
//! - Pair constraints (and so the jitter draws of the 1-DOF residual) are visited in
//!   ascending pair id order, frames and cameras in ascending id order
//!   (the ascending id divergence in the [`super`] module header).

use std::collections::BTreeMap;

use super::problem::{segment3, set_segment3};
use super::{rotation_only_pose, RotationAveragingProblem, RotationConstraint};
use crate::geometry::pose::{
    angle_axis_to_rotation_matrix, average_quaternions, gravity_aligned_rotation,
    rotation_from_y_axis_angle, rotation_matrix_to_angle_axis,
};
use crate::geometry::Rigid3d;
use crate::linalg::{Matrix3d, Quaterniond, VectorXd};
use crate::math::random::{random_uniform_real, set_prng_seed};
use crate::scene::reconstruction::Reconstruction;
use crate::util::types::{FrameId, RigId, SensorId};
use crate::Result;

/// `ComputeGravityAligned1DOFResidual`: `(angle_2 - angle_1) - angle_12`, wrapped to
/// [-pi, pi] with jitter near the boundaries to avoid local minima.
fn compute_gravity_aligned_1dof_residual(angle_12: f64, angle_1: f64, angle_2: f64) -> f64 {
    let mut residual = libm::remainder((angle_2 - angle_1) - angle_12, 2.0 * std::f64::consts::PI);

    // Inject random noise if the angle is too close to the boundary to break the possible
    // balance at the local minima.
    const EPS: f64 = 0.01;
    if residual.abs() > std::f64::consts::PI - EPS {
        let jitter = random_uniform_real(0.0, EPS);
        if residual < 0.0 {
            residual += jitter;
        } else {
            residual -= jitter;
        }
    }
    residual
}

impl RotationAveragingProblem {
    /// The estimated rig_from_world rotation of a frame as used by the 3-DOF residual
    /// (gravity frames through their Y angle only when gravity mode is on).
    fn estimated_frame_rotation(&self, frame_id: FrameId, frame_param_idx: usize) -> Matrix3d {
        if self.has_frame_gravity(frame_id) {
            rotation_from_y_axis_angle(self.estimated_rotations[frame_param_idx])
        } else {
            angle_axis_to_rotation_matrix(segment3(&self.estimated_rotations, frame_param_idx))
        }
    }

    /// `ComputeResiduals`: the residual vector b from the current rotation estimates.
    pub fn compute_residuals(&mut self) -> Result<()> {
        // Set PRNG seed for deterministic jitter injection.
        if self.options.random_seed >= 0 {
            set_prng_seed(self.options.random_seed as u32);
        }

        for constraint in self.pair_constraints.values() {
            let frame_id1 = self.image_frame_id(constraint.image_id1)?;
            let frame_id2 = self.image_frame_id(constraint.image_id2)?;
            let frame_param_idx1 = self.frame_param_idx(frame_id1)?;
            let frame_param_idx2 = self.frame_param_idx(frame_id2)?;

            match constraint.constraint {
                RotationConstraint::GravityAligned1Dof {
                    angle_cam2_from_cam1,
                    ..
                } => {
                    // 1-DOF case: compute Y-axis angle residual.
                    self.residuals[constraint.row_index] = compute_gravity_aligned_1dof_residual(
                        angle_cam2_from_cam1,
                        self.estimated_rotations[frame_param_idx1],
                        self.estimated_rotations[frame_param_idx2],
                    );
                }
                RotationConstraint::Full3Dof { r_cam2_from_cam1 } => {
                    // 3-DOF case: compute full rotation error.
                    let mut estimated_cam1_from_world =
                        self.estimated_frame_rotation(frame_id1, frame_param_idx1);
                    let mut estimated_cam2_from_world =
                        self.estimated_frame_rotation(frame_id2, frame_param_idx2);
                    if let Some(idx) = constraint.cam1_from_rig_param_idx {
                        estimated_cam1_from_world =
                            angle_axis_to_rotation_matrix(segment3(&self.estimated_rotations, idx))
                                * estimated_cam1_from_world;
                    }
                    if let Some(idx) = constraint.cam2_from_rig_param_idx {
                        estimated_cam2_from_world =
                            angle_axis_to_rotation_matrix(segment3(&self.estimated_rotations, idx))
                                * estimated_cam2_from_world;
                    }
                    let residual = -rotation_matrix_to_angle_axis(
                        &(estimated_cam2_from_world.transpose()
                            * r_cam2_from_cam1
                            * estimated_cam1_from_world),
                    );
                    set_segment3(&mut self.residuals, constraint.row_index, residual);
                }
            }
        }

        // Fixed frame residual.
        let fixed_frame_param_idx = self.frame_param_idx(self.fixed_frame_id)?;
        let len = self.residuals.len();
        if self.num_gauge_fixing_residuals == 1 {
            self.residuals[len - 1] =
                self.estimated_rotations[fixed_frame_param_idx] - self.fixed_frame_rotation.y;
        } else {
            let residual = rotation_matrix_to_angle_axis(
                &(angle_axis_to_rotation_matrix(self.fixed_frame_rotation).transpose()
                    * angle_axis_to_rotation_matrix(segment3(
                        &self.estimated_rotations,
                        fixed_frame_param_idx,
                    ))),
            );
            set_segment3(&mut self.residuals, len - 3, residual);
        }
        Ok(())
    }

    /// `UpdateState`: updates the rotation estimates by applying the solution step.
    pub fn update_state(&mut self, step: &VectorXd) -> Result<()> {
        // Update frame rotations.
        let frames: Vec<(FrameId, usize)> = self
            .frame_id_to_param_idx
            .iter()
            .map(|(&f, &i)| (f, i))
            .collect();
        for &(frame_id, frame_param_idx) in &frames {
            if !self.has_frame_gravity(frame_id) {
                let estimated_rig_from_world = angle_axis_to_rotation_matrix(segment3(
                    &self.estimated_rotations,
                    frame_param_idx,
                ));
                let updated = rotation_matrix_to_angle_axis(
                    &(estimated_rig_from_world
                        * angle_axis_to_rotation_matrix(-segment3(step, frame_param_idx))),
                );
                set_segment3(&mut self.estimated_rotations, frame_param_idx, updated);
            } else {
                self.estimated_rotations[frame_param_idx] -= step[frame_param_idx];
            }
        }

        // Compute current frame rotations for cam_from_rig averaging.
        let mut frame_rotations: BTreeMap<FrameId, Matrix3d> = BTreeMap::new();
        for &(frame_id, frame_param_idx) in &frames {
            let rotation = if !self.has_frame_gravity(frame_id) {
                angle_axis_to_rotation_matrix(segment3(&self.estimated_rotations, frame_param_idx))
            } else {
                rotation_from_y_axis_angle(self.estimated_rotations[frame_param_idx])
            };
            frame_rotations.insert(frame_id, rotation);
        }

        // Update the global rotations for cam_from_rig cameras.
        // Note: the update is non-trivial, and we need to average the rotations from all the
        // frames.
        for (&camera_id, &camera_param_idx) in &self.camera_id_to_param_idx {
            let estimated_cam_from_rig = angle_axis_to_rotation_matrix(segment3(
                &self.estimated_rotations,
                camera_param_idx,
            ));
            let r_update = angle_axis_to_rotation_matrix(-segment3(step, camera_param_idx));

            // COLMAP's camera_to_frame_ids_[camera_id] default-constructs an empty list for a
            // camera without active frames; AverageQuaternions then fails on no samples.
            let frame_ids = self
                .camera_to_frame_ids
                .get(&camera_id)
                .map(Vec::as_slice)
                .unwrap_or(&[]);
            let mut rig_rotations: Vec<Quaterniond> = Vec::with_capacity(frame_ids.len());
            for frame_id in frame_ids {
                let r = frame_rotations[frame_id];
                rig_rotations.push(Quaterniond::from_rotation_matrix(
                    estimated_cam_from_rig * r * r_update * r.transpose(),
                ));
            }

            // Average the rotations for the rig.
            let r_ave = average_quaternions(&rig_rotations, &vec![1.0; rig_rotations.len()])?;
            let updated = rotation_matrix_to_angle_axis(&r_ave.to_rotation_matrix());
            set_segment3(&mut self.estimated_rotations, camera_param_idx, updated);
        }
        Ok(())
    }

    /// `AverageStepSize`: the average rotation step size, for convergence checking.
    pub fn average_step_size(&self, step: &VectorXd) -> f64 {
        let mut total_update = 0.0;
        for (&frame_id, &frame_param_idx) in &self.frame_id_to_param_idx {
            if self.has_frame_gravity(frame_id) {
                total_update += step[frame_param_idx].abs();
            } else {
                total_update += segment3(step, frame_param_idx).norm();
            }
        }
        total_update / self.frame_id_to_param_idx.len() as f64
    }

    /// `ApplyResultsToReconstruction`: writes the optimized rotations back. Every problem
    /// frame gets a rig_from_world with unknown (NaN) translation, and, with
    /// `refine_sensor_from_rig`, every estimated camera a cam_from_rig with unknown
    /// translation.
    pub fn apply_results_to_reconstruction(
        &self,
        reconstruction: &mut Reconstruction,
    ) -> Result<()> {
        for (&frame_id, &frame_param_idx) in &self.frame_id_to_param_idx {
            let rotation = match self
                .frame_gravity(frame_id)
                .filter(|_| self.options.use_gravity)
            {
                Some(gravity) => {
                    gravity_aligned_rotation(gravity)?
                        * rotation_from_y_axis_angle(self.estimated_rotations[frame_param_idx])
                }
                None => angle_axis_to_rotation_matrix(segment3(
                    &self.estimated_rotations,
                    frame_param_idx,
                )),
            };
            reconstruction
                .frame_mut(frame_id)?
                .set_rig_from_world(rotation_only_pose(Quaterniond::from_rotation_matrix(
                    rotation,
                )));
        }

        if self.options.refine_sensor_from_rig {
            let mut updates: Vec<(RigId, SensorId, Rigid3d)> = Vec::new();
            for (&rig_id, rig) in reconstruction.rigs() {
                for sensor_id in rig.non_ref_sensors().keys() {
                    // Skip cameras that are not estimated.
                    let Some(&param_idx) = self.camera_id_to_param_idx.get(&sensor_id.id) else {
                        continue;
                    };
                    let rotation =
                        Quaterniond::from_rotation_matrix(angle_axis_to_rotation_matrix(segment3(
                            &self.estimated_rotations,
                            param_idx,
                        )));
                    // No translation yet.
                    updates.push((rig_id, *sensor_id, rotation_only_pose(rotation)));
                }
            }
            for (rig_id, sensor_id, cam_from_rig) in updates {
                reconstruction
                    .rig_mut(rig_id)?
                    .set_sensor_from_rig(sensor_id, cam_from_rig)?;
            }
        }
        Ok(())
    }
}
