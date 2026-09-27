//! Port of `colmap/sensor/rig.h` / `rig.cc`: [`Rig`], a set of rigidly mounted sensors with
//! their poses relative to a reference sensor. Port of colmap-sharp's
//! `ColmapSharp/Sensor/Rig.cs`; the scene-level rig helpers (`colmap/scene/rig.h`) are a
//! separate port. Tests: `tests/sensor/rig.rs` (`rig_test.cc` 1:1).
//!
//! Tier A. Translation notes:
//! - `std::map<sensor_t, std::optional<Rigid3d>>` is a `BTreeMap`, ordered by
//!   [`SensorId`]'s `(type, id)` order like the C++ map, so [`Rig::non_ref_sensors`] and the
//!   printed form iterate as COLMAP's do.
//! - C++ hands out `Rigid3d&` / `std::optional<Rigid3d>&`; here the getters return copies
//!   and the `_mut` variants return `&mut` for in-place edits.
//! - `SensorFromRig` on a sensor without a pose is `std::optional::value()` on `nullopt`
//!   (`std::bad_optional_access`); here it is an `Err`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::geometry::rigid3::Rigid3d;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{RigId, SensorId, INVALID_RIG_ID, INVALID_SENSOR_ID};
use crate::{check, check_ge, Result};

/// Port of `colmap::Rig`. Rigs represent a collection of rigidly mounted sensors. The
/// reference sensor defines the rig's origin (its `sensor_from_rig` is fixed to identity);
/// every other sensor has an optional `sensor_from_rig` pose.
#[derive(Debug, Clone, PartialEq)]
pub struct Rig {
    rig_id: RigId,
    ref_sensor_id: SensorId,
    sensors_from_rig: BTreeMap<SensorId, Option<Rigid3d>>,
}

impl Default for Rig {
    fn default() -> Self {
        Rig {
            rig_id: INVALID_RIG_ID,
            ref_sensor_id: INVALID_SENSOR_ID,
            sensors_from_rig: BTreeMap::new(),
        }
    }
}

impl Rig {
    /// An empty rig with an invalid id and no sensors.
    pub fn new() -> Self {
        Self::default()
    }

    /// `RigId`: unique identifier of the rig.
    pub fn rig_id(&self) -> RigId {
        self.rig_id
    }

    /// `SetRigId`.
    pub fn set_rig_id(&mut self, rig_id: RigId) {
        self.rig_id = rig_id;
    }

    /// Port of `Rig::AddRefSensor`: sets the reference sensor, which must be added first and
    /// only once.
    pub fn add_ref_sensor(&mut self, ref_sensor_id: SensorId) -> Result<()> {
        check!(
            self.ref_sensor_id == INVALID_SENSOR_ID,
            "Reference sensor already set"
        );
        self.ref_sensor_id = ref_sensor_id;
        Ok(())
    }

    /// Port of `Rig::AddSensor`: adds a non-reference sensor with an optional pose.
    pub fn add_sensor(
        &mut self,
        sensor_id: SensorId,
        sensor_from_rig: Option<Rigid3d>,
    ) -> Result<()> {
        check_ge!(
            self.num_sensors(),
            1,
            "The reference sensor needs to be added first before other sensors."
        );
        check!(
            !self.has_sensor(sensor_id),
            "Sensor ({}, {}) is inserted twice into the rig",
            sensor_id.sensor_type,
            sensor_id.id
        );
        self.sensors_from_rig.insert(sensor_id, sensor_from_rig);
        Ok(())
    }

    /// `HasSensor`: whether the sensor is the reference sensor or one of the others.
    pub fn has_sensor(&self, sensor_id: SensorId) -> bool {
        sensor_id == self.ref_sensor_id || self.sensors_from_rig.contains_key(&sensor_id)
    }

    /// `NumSensors`: the reference sensor (once set) plus the others.
    pub fn num_sensors(&self) -> usize {
        let mut num_sensors = self.sensors_from_rig.len();
        if self.ref_sensor_id != INVALID_SENSOR_ID {
            num_sensors += 1;
        }
        num_sensors
    }

    /// `RefSensorId`.
    pub fn ref_sensor_id(&self) -> SensorId {
        self.ref_sensor_id
    }

    /// `IsRefSensor`.
    pub fn is_ref_sensor(&self, sensor_id: SensorId) -> bool {
        sensor_id == self.ref_sensor_id
    }

    /// `HasSensorFromRig`: whether a non-reference sensor has a known pose.
    pub fn has_sensor_from_rig(&self, sensor_id: SensorId) -> bool {
        sensor_id != self.ref_sensor_id
            && self.has_sensor(sensor_id)
            && matches!(self.sensors_from_rig.get(&sensor_id), Some(Some(_)))
    }

    /// `SensorIds`: the reference sensor (even when invalid, as in COLMAP) and the others.
    pub fn sensor_ids(&self) -> BTreeSet<SensorId> {
        let mut sensor_ids = BTreeSet::new();
        sensor_ids.insert(self.ref_sensor_id);
        sensor_ids.extend(self.sensors_from_rig.keys().copied());
        sensor_ids
    }

    /// `NonRefSensors() const`: every non-reference sensor with its optional pose.
    pub fn non_ref_sensors(&self) -> &BTreeMap<SensorId, Option<Rigid3d>> {
        &self.sensors_from_rig
    }

    /// `NonRefSensors()`: mutable access to the non-reference sensors.
    pub fn non_ref_sensors_mut(&mut self) -> &mut BTreeMap<SensorId, Option<Rigid3d>> {
        &mut self.sensors_from_rig
    }

    /// `SensorFromRig`: the pose of a non-reference sensor. Errors for the reference sensor,
    /// an unknown sensor, or a sensor without a pose.
    pub fn sensor_from_rig(&self, sensor_id: SensorId) -> Result<Rigid3d> {
        let maybe = self.find_sensor_from_rig_or_throw(sensor_id)?;
        maybe.ok_or_else(bad_optional_access)
    }

    /// `SensorFromRig()` (non-const): the pose for in-place edits.
    pub fn sensor_from_rig_mut(&mut self, sensor_id: SensorId) -> Result<&mut Rigid3d> {
        self.find_sensor_from_rig_or_throw_mut(sensor_id)?
            .as_mut()
            .ok_or_else(bad_optional_access)
    }

    /// `MaybeSensorFromRig`: the optional pose of a non-reference sensor.
    pub fn maybe_sensor_from_rig(&self, sensor_id: SensorId) -> Result<Option<Rigid3d>> {
        self.find_sensor_from_rig_or_throw(sensor_id)
    }

    /// `MaybeSensorFromRig()` (non-const).
    pub fn maybe_sensor_from_rig_mut(
        &mut self,
        sensor_id: SensorId,
    ) -> Result<&mut Option<Rigid3d>> {
        self.find_sensor_from_rig_or_throw_mut(sensor_id)
    }

    /// `SetSensorFromRig(sensor_id, const Rigid3d&)`.
    pub fn set_sensor_from_rig(
        &mut self,
        sensor_id: SensorId,
        sensor_from_rig: Rigid3d,
    ) -> Result<()> {
        *self.find_sensor_from_rig_or_throw_mut(sensor_id)? = Some(sensor_from_rig);
        Ok(())
    }

    /// `SetSensorFromRig(sensor_id, const std::optional<Rigid3d>&)`.
    pub fn set_maybe_sensor_from_rig(
        &mut self,
        sensor_id: SensorId,
        sensor_from_rig: Option<Rigid3d>,
    ) -> Result<()> {
        *self.find_sensor_from_rig_or_throw_mut(sensor_id)? = sensor_from_rig;
        Ok(())
    }

    /// `ResetSensorFromRig`: forget the pose of a non-reference sensor.
    pub fn reset_sensor_from_rig(&mut self, sensor_id: SensorId) -> Result<()> {
        *self.find_sensor_from_rig_or_throw_mut(sensor_id)? = None;
        Ok(())
    }

    fn find_sensor_from_rig_or_throw(&self, sensor_id: SensorId) -> Result<Option<Rigid3d>> {
        self.check_not_ref(sensor_id)?;
        let found = self.sensors_from_rig.get(&sensor_id);
        check!(
            found.is_some(),
            "Sensor ({}, {}) not found in the rig",
            sensor_id.sensor_type,
            sensor_id.id
        );
        Ok(found.copied().flatten())
    }

    fn find_sensor_from_rig_or_throw_mut(
        &mut self,
        sensor_id: SensorId,
    ) -> Result<&mut Option<Rigid3d>> {
        self.check_not_ref(sensor_id)?;
        check!(
            self.sensors_from_rig.contains_key(&sensor_id),
            "Sensor ({}, {}) not found in the rig",
            sensor_id.sensor_type,
            sensor_id.id
        );
        Ok(self
            .sensors_from_rig
            .get_mut(&sensor_id)
            .expect("presence checked above"))
    }

    fn check_not_ref(&self, sensor_id: SensorId) -> Result<()> {
        check!(
            sensor_id != self.ref_sensor_id,
            "The reference sensor does not have a SensorFromRig transformation, which is \
             fixed to identity"
        );
        Ok(())
    }
}

/// `std::bad_optional_access` from `std::optional::value()` on `nullopt`.
fn bad_optional_access() -> ColmapError {
    ColmapError::new(ErrorKind::LogicError, "bad optional access")
}

impl fmt::Display for Rig {
    /// Port of `operator<<(std::ostream&, const Rig&)`:
    /// `Rig(rig_id=<id|Invalid>, ref_sensor_id=(TYPE, id), sensors=[(TYPE, id), ...])`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let rig_id = if self.rig_id != INVALID_RIG_ID {
            self.rig_id.to_string()
        } else {
            "Invalid".to_string()
        };
        write!(
            f,
            "Rig(rig_id={rig_id}, ref_sensor_id=({}, {}), sensors=[",
            self.ref_sensor_id.sensor_type, self.ref_sensor_id.id
        )?;
        for (i, sensor_id) in self.sensors_from_rig.keys().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "({}, {})", sensor_id.sensor_type, sensor_id.id)?;
        }
        f.write_str("])")
    }
}
