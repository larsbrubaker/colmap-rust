//! Port of COLMAP's `colmap/estimators/rotation_averaging.h/.cc` and
//! `rotation_averaging_impl.h/.cc`: global rotation averaging over a [`PoseGraph`]'s relative
//! rotations, adapted by COLMAP from Theia's `RobustRotationEstimator`
//! (<http://www.theia-sfm.org/>); for gravity aligned rotation averaging see the paper
//! "Gravity Aligned Rotation Averaging". Port of colmap-sharp's
//! `Estimators/RotationAveraging*.cs`, `RotationEstimator*.cs`.
//!
//! - this file: the options ([`RotationEstimatorOptions`], [`RotationAveragingReweighting`],
//!   [`RotationWeightType`]) and the image helpers the submodules share;
//! - `problem.rs`: [`RotationAveragingProblem`] setup, the linear system A x = b
//!   (`problem_state.rs` holds its residuals, state update and write-back);
//! - `solver.rs`: [`RotationAveragingSolver`], L1 regression then IRLS;
//! - `estimator.rs`: [`RotationEstimator`], the stratified gravity pre-solve and the maximum
//!   spanning tree initialization;
//! - `run.rs`: the free functions ([`run_rotation_averaging`],
//!   [`run_rotation_averaging_on_component`], [`initialize_rig_rotations_from_images`],
//!   [`filter_edges_by_relative_rotation`]) and the rig expansion for unknown cam_from_rig.
//!
//! Tests: `colmap-rust/tests/estimators/rotation_averaging.rs` (`rotation_averaging_test.cc`
//! 1:1). Tier C (iterative, with a random jitter near the +-pi boundary of the 1-DOF
//! residual).
//!
//! Translation notes:
//! - COLMAP's hash containers decide the parameter layout, the row order, the jitter draw
//!   order, the gauge frame, the spanning tree's numbering and the order quaternion samples
//!   are averaged in; here every one of them is in ascending id order
//!   (docs/CPP_DIVERGENCES.md entry 278).
//! - `THROW_CHECK`s and lookups of missing objects are `Err`; `LOG(ERROR)` paths return
//!   `Ok(false)` (the core has no logger, so `LOG`/`VLOG` messages are not emitted).
//! - `FlatHashSet<image_t>` arguments are `HashSet<ImageId>` (only probed, or sorted before
//!   iterating); `NodeHashMap<image_t, Rigid3d>` is a `BTreeMap`.

mod estimator;
mod problem;
mod problem_state;
mod run;
mod solver;

pub use estimator::RotationEstimator;
pub use problem::{PairConstraint, RotationAveragingProblem, RotationConstraint};
pub use run::{
    filter_edges_by_relative_rotation, initialize_rig_rotations_from_images,
    run_rotation_averaging, run_rotation_averaging_on_component,
};
pub use solver::RotationAveragingSolver;

use crate::estimators::alignment::image_cam_from_world;
use crate::geometry::Rigid3d;
use crate::linalg::{Quaterniond, Vector3d};
use crate::scene::image::Image;
use crate::scene::reconstruction::Reconstruction;
use crate::util::types::{SensorId, SensorType};
use crate::Result;

#[cfg(doc)]
use crate::scene::pose_graph::PoseGraph;

/// Reweighting scheme applied to the relative-rotation constraints. Port of
/// `colmap::RotationAveragingReweighting`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RotationAveragingReweighting {
    /// All constraints are weighted equally.
    #[default]
    Uniform,
    /// Weight each constraint by the number of inlier two-view matches
    /// (`PoseGraphEdge::num_matches`) of the corresponding edge, normalized to (0, 1].
    InlierMatchCount,
}

/// Robust IRLS weight function. Port of `RotationEstimatorOptions::WeightType`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum RotationWeightType {
    /// Geman-McClure weight from "Efficient and robust large-scale rotation averaging"
    /// (Chatterjee et al., 2013).
    #[default]
    GemanMcClure,
    /// Half norm from "Robust Relative Rotation Averaging" (Chatterjee et al., 2017).
    HalfNorm,
}

/// Port of `colmap::RotationEstimatorOptions`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotationEstimatorOptions {
    /// PRNG seed for stochastic methods during rotation averaging. If -1 (default), the
    /// global PRNG is not reseeded (non-deterministic across runs). If >= 0, the rotation
    /// averaging is deterministic with the given seed.
    pub random_seed: i32,
    /// Maximum number of times to run L1 minimization.
    pub max_num_l1_iterations: i32,
    /// Average step size threshold to terminate the L1 minimization.
    pub l1_step_convergence_threshold: f64,
    /// The number of iterative reweighted least squares iterations to perform.
    pub max_num_irls_iterations: i32,
    /// Average step size threshold to terminate the IRLS minimization.
    pub irls_step_convergence_threshold: f64,
    /// Gravity direction.
    pub gravity_dir: Vector3d,
    /// The point where the Huber-like cost function switches from L1 to L2, in degrees.
    pub irls_loss_parameter_sigma: f64,
    /// Tikhonov ridge added to the diagonal of the normal equations A^T (W) A before each
    /// Cholesky factorization in the L1 and IRLS phases. The theoretical normal equations of
    /// a connected pose graph plus gauge fix are positive definite, but Cholesky may still
    /// report "matrix not positive definite" on poorly conditioned graphs (e.g., long
    /// sequential video chains). A small positive value (e.g., 1e-9) stabilizes such systems;
    /// zero disables regularization.
    pub ridge_regularization: f64,
    /// Robust weight function of the IRLS phase.
    pub weight_type: RotationWeightType,
    /// Flag to skip maximum spanning tree initialization.
    pub skip_initialization: bool,
    /// Flag to use gravity priors for rotation averaging.
    pub use_gravity: bool,
    /// Flag to use stratified solving for mixed gravity systems. If true and `use_gravity`
    /// is true, first solves the 1-DOF system with gravity-only pairs, then solves the full
    /// 3-DOF system.
    pub use_stratified: bool,
    /// If true, only consider frames with existing poses when computing connected
    /// components. Set to true for refinement passes.
    pub filter_unregistered: bool,
    /// If > 0, filter image pairs with rotation error exceeding this threshold after
    /// solving, then recompute active set.
    pub max_rotation_error_deg: f64,
    /// When false, treat each non-ref sensor's cam_from_rig rotation as a pre-calibrated
    /// constant.
    pub refine_sensor_from_rig: bool,
    /// Reweighting scheme for the relative-rotation constraints. Weights are applied as a
    /// block-diagonal scaling of the linear system, so a noise-free (consistent) system
    /// yields an identical solution regardless of reweighting.
    pub reweighting: RotationAveragingReweighting,
}

impl Default for RotationEstimatorOptions {
    /// COLMAP's member initializers.
    fn default() -> Self {
        Self {
            random_seed: -1,
            max_num_l1_iterations: 5,
            l1_step_convergence_threshold: 0.001,
            max_num_irls_iterations: 100,
            irls_step_convergence_threshold: 0.001,
            gravity_dir: Vector3d::new(0.0, 1.0, 0.0),
            irls_loss_parameter_sigma: 5.0,
            ridge_regularization: 1e-9,
            weight_type: RotationWeightType::GemanMcClure,
            skip_initialization: false,
            use_gravity: false,
            use_stratified: true,
            filter_unregistered: false,
            max_rotation_error_deg: 10.0,
            refine_sensor_from_rig: true,
            reweighting: RotationAveragingReweighting::Uniform,
        }
    }
}

/// `image.CameraPtr()->SensorId()`: the camera sensor of an image.
fn camera_sensor_id(image: &Image) -> SensorId {
    SensorId::new(SensorType::Camera, image.camera_id())
}

/// `image.IsRefInFrame()` for an image of `reconstruction`, through its frame and rig.
fn is_ref_in_frame(reconstruction: &Reconstruction, image: &Image) -> Result<bool> {
    let frame = reconstruction.frame(image.frame_id())?;
    let rig = reconstruction.rig(frame.rig_id())?;
    image.is_ref_in_frame(frame, rig)
}

/// `image.CamFromWorld().rotation()` for an image of `reconstruction`.
fn cam_from_world_rotation(reconstruction: &Reconstruction, image: &Image) -> Result<Quaterniond> {
    Ok(image_cam_from_world(reconstruction, image)?.rotation)
}

/// `Eigen::Vector3d::Constant(NaN)`: the translation of a pose known only up to rotation.
const UNKNOWN_TRANSLATION: Vector3d = Vector3d::new(f64::NAN, f64::NAN, f64::NAN);

/// `Rigid3d(rotation, kUnknownTranslation)`.
fn rotation_only_pose(rotation: Quaterniond) -> Rigid3d {
    Rigid3d::new(rotation, UNKNOWN_TRANSLATION)
}

/// `v.hasNaN()` of a 3-vector.
fn has_nan3(v: Vector3d) -> bool {
    v.x.is_nan() || v.y.is_nan() || v.z.is_nan()
}
