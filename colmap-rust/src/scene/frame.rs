//! Port of `colmap/scene/frame.h` / `frame.cc`: [`Frame`], a posed instance of a rig
//! ([`crate::sensor::rig::Rig`]) with the ids of the measurements its sensors took. Images
//! ([`super::image`]) point back at their frame by id. Tests: `tests/scene/frame.rs`
//! (`frame_test.cc` 1:1).
//!
//! Tier A (plain data and pose composition in COLMAP's order).
//!
//! **Ids into owning maps** (the ownership divergence in the [`super`] module header). COLMAP's
//! `Frame` holds a raw `Rig* rig_ptr_` that the reconstruction sets (`SetRigPtr`) and clears
//! (`ResetRigPtr`). Here a frame stores only its rig's id; whoever owns the rigs (the
//! reconstruction) passes the rig to the methods that need it:
//! - `SetRigPtr(&rig)` is [`Frame::verify_rig`]: the same checks (valid rig id, the frame's rig
//!   id, every camera measurement a sensor of the rig), with nothing stored;
//! - `SensorFromWorld` and `SetCamFromWorld` take `&Rig` and fail unless it is the frame's rig,
//!   which replaces COLMAP's `THROW_CHECK_NOTNULL(rig_ptr_)`;
//! - `AddDataId` on a frame bound to a rig is [`Frame::add_data_id_in_rig`]; plain
//!   [`Frame::add_data_id`] is the unbound form;
//! - `HasRigPtr`, `RigPtr` and `ResetRigPtr` have no counterpart, and `SetRigId` no longer
//!   refuses to change the id of a bound frame: a frame inside a reconstruction is only reached
//!   through it, and the reconstruction's own checks keep its rig ids consistent.

use std::collections::BTreeSet;
use std::fmt;

use crate::geometry::rigid3::Rigid3d;
use crate::sensor::rig::Rig;
use crate::util::check::Result;
use crate::util::types::{
    CameraId, DataId, FrameId, RigId, SensorId, SensorType, INVALID_FRAME_ID, INVALID_RIG_ID,
};
use crate::{check, check_eq, check_ne, fatal_throw};

/// Port of `colmap::Frame`. Frames represent (posed) instantiations of rigs with associated
/// measurements for the different sensors. The captured sensor measurements are defined by the
/// list of data ids.
///
/// `Clone` is COLMAP's copy constructor: the copy is never finalized (see
/// [`Frame::finalize_data_ids`]); a move keeps the flag, as COLMAP's defaulted move does.
#[derive(Debug)]
pub struct Frame {
    frame_id: FrameId,
    rig_id: RigId,
    /// COLMAP's `std::set<data_t>`, ordered by `(sensor type, sensor id, data id)`.
    data_ids: BTreeSet<DataId>,
    has_final_data_ids: bool,
    /// The rig_from_world transformation, if the frame is posed.
    rig_from_world: Option<Rigid3d>,
}

impl Default for Frame {
    fn default() -> Self {
        Frame {
            frame_id: INVALID_FRAME_ID,
            rig_id: INVALID_RIG_ID,
            data_ids: BTreeSet::new(),
            has_final_data_ids: false,
            rig_from_world: None,
        }
    }
}

impl Clone for Frame {
    /// COLMAP's copy constructor and copy assignment: everything but the finalized flag.
    fn clone(&self) -> Self {
        Frame {
            frame_id: self.frame_id,
            rig_id: self.rig_id,
            data_ids: self.data_ids.clone(),
            has_final_data_ids: false,
            rig_from_world: self.rig_from_world,
        }
    }
}

impl PartialEq for Frame {
    /// COLMAP's `operator==`: ids, data ids and pose; not the finalized flag.
    fn eq(&self, other: &Self) -> bool {
        self.frame_id == other.frame_id
            && self.rig_id == other.rig_id
            && self.data_ids == other.data_ids
            && self.rig_from_world == other.rig_from_world
    }
}

impl Frame {
    /// A frame with invalid ids, no data and no pose.
    pub fn new() -> Self {
        Self::default()
    }

    /// `FrameId`: unique identifier of the frame.
    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    /// `SetFrameId`.
    pub fn set_frame_id(&mut self, frame_id: FrameId) {
        self.frame_id = frame_id;
    }

    /// `DataIds()`: the frame's measurements, in `(sensor type, sensor id, data id)` order.
    pub fn data_ids(&self) -> &BTreeSet<DataId> {
        &self.data_ids
    }

    /// `AddDataId` on a frame not bound to a rig. Fails once the data ids are finalized.
    pub fn add_data_id(&mut self, data_id: DataId) -> Result<()> {
        check!(
            !self.has_final_data_ids,
            "Cannot add data id to a finalized frame. Data ids must be added before the frame \
             is added to a reconstruction."
        );
        self.data_ids.insert(data_id);
        Ok(())
    }

    /// `AddDataId` on a frame bound to `rig` (COLMAP's `HasRigPtr()` branch): `rig` must be
    /// the frame's rig and have the measurement's sensor.
    pub fn add_data_id_in_rig(&mut self, rig: &Rig, data_id: DataId) -> Result<()> {
        check!(
            !self.has_final_data_ids,
            "Cannot add data id to a finalized frame. Data ids must be added before the frame \
             is added to a reconstruction."
        );
        self.check_is_frame_rig(rig)?;
        check!(rig.has_sensor(data_id.sensor_id));
        self.data_ids.insert(data_id);
        Ok(())
    }

    /// `NumDataIds`.
    pub fn num_data_ids(&self) -> usize {
        self.data_ids.len()
    }

    /// `HasDataId`: whether the measurement belongs to the frame.
    pub fn has_data_id(&self, data_id: DataId) -> bool {
        self.data_ids.contains(&data_id)
    }

    /// `FinalizeDataIds`: prevent further modification of the data ids. Called when the frame
    /// is added to a reconstruction, to keep cached counters like the number of registered
    /// images consistent.
    pub fn finalize_data_ids(&mut self) {
        self.has_final_data_ids = true;
    }

    /// `HasFinalDataIds`.
    pub fn has_final_data_ids(&self) -> bool {
        self.has_final_data_ids
    }

    /// `ClearDataIds`. Fails once the data ids are finalized.
    pub fn clear_data_ids(&mut self) -> Result<()> {
        check!(
            !self.has_final_data_ids,
            "Cannot clear data ids of a finalized frame."
        );
        self.data_ids.clear();
        Ok(())
    }

    /// `RigId`: identifier of the rig. Multiple frames may share a rig.
    pub fn rig_id(&self) -> RigId {
        self.rig_id
    }

    /// `SetRigId`. The id must be valid.
    pub fn set_rig_id(&mut self, rig_id: RigId) -> Result<()> {
        check_ne!(rig_id, INVALID_RIG_ID);
        self.rig_id = rig_id;
        Ok(())
    }

    /// `HasRigId`.
    pub fn has_rig_id(&self) -> bool {
        self.rig_id != INVALID_RIG_ID
    }

    /// Port of `SetRigPtr`'s checks: `rig` has a valid id equal to the frame's rig id, and
    /// every camera measurement of the frame is a sensor of `rig` (IMU data is not checked:
    /// COLMAP does not support IMU measurements yet). To bind the frame to another rig, set
    /// its rig id first.
    pub fn verify_rig(&self, rig: &Rig) -> Result<()> {
        check_ne!(rig.rig_id(), INVALID_RIG_ID);
        for data_id in &self.data_ids {
            match data_id.sensor_id.sensor_type {
                SensorType::Camera => check!(rig.has_sensor(data_id.sensor_id)),
                SensorType::Imu => {}
                SensorType::Invalid => {
                    fatal_throw!("Invalid sensor type: {}", data_id.sensor_id.sensor_type)
                }
            }
        }
        check_eq!(rig.rig_id(), self.rig_id);
        Ok(())
    }

    /// `RigFromWorld`. Fails when the frame has no pose.
    pub fn rig_from_world(&self) -> Result<Rigid3d> {
        self.check_has_pose()?;
        Ok(self.rig_from_world.unwrap_or_default())
    }

    /// `RigFromWorld()` for in-place edits. Fails when the frame has no pose.
    pub fn rig_from_world_mut(&mut self) -> Result<&mut Rigid3d> {
        self.check_has_pose()?;
        Ok(self
            .rig_from_world
            .as_mut()
            .expect("pose presence checked above"))
    }

    /// `MaybeRigFromWorld`.
    pub fn maybe_rig_from_world(&self) -> Option<Rigid3d> {
        self.rig_from_world
    }

    /// `MaybeRigFromWorld()` for in-place edits.
    pub fn maybe_rig_from_world_mut(&mut self) -> &mut Option<Rigid3d> {
        &mut self.rig_from_world
    }

    /// `SetRigFromWorld(const Rigid3d&)`.
    pub fn set_rig_from_world(&mut self, rig_from_world: Rigid3d) {
        self.rig_from_world = Some(rig_from_world);
    }

    /// `SetRigFromWorld(const std::optional<Rigid3d>&)`.
    pub fn set_maybe_rig_from_world(&mut self, rig_from_world: Option<Rigid3d>) {
        self.rig_from_world = rig_from_world;
    }

    /// `HasPose`.
    pub fn has_pose(&self) -> bool {
        self.rig_from_world.is_some()
    }

    /// `ResetPose`.
    pub fn reset_pose(&mut self) {
        self.rig_from_world = None;
    }

    /// `SensorFromWorld`: `rig_from_world` for the reference sensor, else
    /// `sensor_from_rig * rig_from_world`. `rig` must be the frame's rig.
    pub fn sensor_from_world(&self, rig: &Rig, sensor_id: SensorId) -> Result<Rigid3d> {
        self.check_is_frame_rig(rig)?;
        if rig.is_ref_sensor(sensor_id) {
            self.rig_from_world()
        } else {
            Ok(rig.sensor_from_rig(sensor_id)? * self.rig_from_world()?)
        }
    }

    /// `SetCamFromWorld`: set the frame's pose from one of its cameras' pose,
    /// `rig_from_world = Inverse(cam_from_rig) * cam_from_world`. `rig` must be the frame's
    /// rig.
    pub fn set_cam_from_world(
        &mut self,
        rig: &Rig,
        camera_id: CameraId,
        cam_from_world: Rigid3d,
    ) -> Result<()> {
        self.check_is_frame_rig(rig)?;
        let sensor_id = SensorId::new(SensorType::Camera, camera_id);
        if rig.is_ref_sensor(sensor_id) {
            self.set_rig_from_world(cam_from_world);
        } else {
            let cam_from_rig = rig.sensor_from_rig(sensor_id)?;
            self.set_rig_from_world(cam_from_rig.inverse() * cam_from_world);
        }
        Ok(())
    }

    /// `DataIds(SensorType)`: the measurements of one sensor type, in order.
    pub fn data_ids_of_type(&self, sensor_type: SensorType) -> impl Iterator<Item = DataId> + '_ {
        self.data_ids
            .iter()
            .copied()
            .filter(move |data_id| data_id.sensor_id.sensor_type == sensor_type)
    }

    /// `ImageIds`: the camera measurements (images), in order.
    pub fn image_ids(&self) -> impl Iterator<Item = DataId> + '_ {
        self.data_ids_of_type(SensorType::Camera)
    }

    /// `THROW_CHECK(rig_from_world_) << "Frame does not have a valid pose."`.
    fn check_has_pose(&self) -> Result<()> {
        check!(
            self.rig_from_world.is_some(),
            "Frame does not have a valid pose."
        );
        Ok(())
    }

    /// The id-based form of COLMAP's `THROW_CHECK_NOTNULL(rig_ptr_)`: the rig passed in must
    /// be the frame's.
    fn check_is_frame_rig(&self, rig: &Rig) -> Result<()> {
        check_ne!(self.rig_id, INVALID_RIG_ID);
        check_eq!(rig.rig_id(), self.rig_id);
        Ok(())
    }
}

impl fmt::Display for Frame {
    /// Port of `operator<<(std::ostream&, const Frame&)`:
    /// `Frame(frame_id=1, rig_id=2, has_pose=0, data_ids=[(CAMERA, 1, 3), (IMU, 0, 2)])`.
    /// COLMAP's `"Invalid"` branch for the rig id sits inside `HasRigId()`, which is exactly
    /// `rig_id != kInvalidRigId`, so it can never print; an unset rig id prints `"Unknown"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Frame(frame_id={}, rig_id=", self.frame_id)?;
        if self.has_rig_id() {
            write!(f, "{}", self.rig_id)?;
        } else {
            f.write_str("Unknown")?;
        }
        write!(
            f,
            ", has_pose={}, data_ids=[",
            if self.has_pose() { 1 } else { 0 }
        )?;
        for (i, data_id) in self.data_ids.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(
                f,
                "({}, {}, {})",
                data_id.sensor_id.sensor_type, data_id.sensor_id.id, data_id.id
            )?;
        }
        f.write_str("])")
    }
}
