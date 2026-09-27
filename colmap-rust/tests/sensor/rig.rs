// Port of colmap/sensor/rig_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::sensor::rig::Rig. EXPECT_ANY_THROW becomes `is_err()`.

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{SensorId, SensorType, INVALID_RIG_ID, INVALID_SENSOR_ID};

fn test_rigid3d() -> Rigid3d {
    Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d())
}

#[test]
fn rig_default() {
    let rig = Rig::new();
    assert_eq!(rig.rig_id(), INVALID_RIG_ID);
    assert_eq!(rig.ref_sensor_id(), INVALID_SENSOR_ID);
    assert_eq!(rig.num_sensors(), 0);
    assert_eq!(rig.non_ref_sensors().len(), 0);
}

#[test]
fn rig_set_up() {
    let mut rig = Rig::new();
    let sensor_id0 = SensorId::new(SensorType::Imu, 0);
    rig.add_ref_sensor(sensor_id0).unwrap();
    let sensor_id1 = SensorId::new(SensorType::Imu, 1);
    let sensor1_from_rig = test_rigid3d();
    rig.add_sensor(sensor_id1, Some(sensor1_from_rig)).unwrap();
    let sensor_id2 = SensorId::new(SensorType::Camera, 0);
    let sensor2_from_rig = test_rigid3d();
    rig.add_sensor(sensor_id2, Some(sensor2_from_rig)).unwrap();
    let sensor_id3 = SensorId::new(SensorType::Camera, 1);
    rig.add_sensor(sensor_id3, None).unwrap(); // no input sensor_from_rig

    assert_eq!(rig.num_sensors(), 4);
    assert_eq!(rig.non_ref_sensors().len(), 3);
    let mut sensor_ids: Vec<SensorId> = rig.sensor_ids().into_iter().collect();
    let mut expected = vec![sensor_id0, sensor_id1, sensor_id2, sensor_id3];
    sensor_ids.sort();
    expected.sort();
    assert_eq!(sensor_ids, expected);

    assert_eq!(rig.ref_sensor_id().sensor_type, SensorType::Imu);
    assert_eq!(rig.ref_sensor_id().id, 0);

    assert!(rig.is_ref_sensor(sensor_id0));
    assert!(!rig.is_ref_sensor(sensor_id1));
    assert!(!rig.is_ref_sensor(sensor_id2));
    assert!(!rig.is_ref_sensor(sensor_id3));

    assert!(!rig.has_sensor_from_rig(sensor_id0));
    assert!(rig.has_sensor_from_rig(sensor_id1));
    assert!(rig.has_sensor_from_rig(sensor_id2));
    assert!(!rig.has_sensor_from_rig(sensor_id3)); // no sensor_from_rig

    assert_eq!(rig.sensor_from_rig(sensor_id1).unwrap(), sensor1_from_rig);
    assert_eq!(
        rig.maybe_sensor_from_rig(sensor_id1).unwrap().unwrap(),
        sensor1_from_rig
    );

    assert_eq!(rig.sensor_from_rig(sensor_id2).unwrap(), sensor2_from_rig);
    assert_eq!(
        rig.maybe_sensor_from_rig(sensor_id2).unwrap().unwrap(),
        sensor2_from_rig
    );

    assert!(rig.has_sensor(sensor_id3));
    assert!(rig.sensor_from_rig(sensor_id3).is_err());
    assert_eq!(rig.maybe_sensor_from_rig(sensor_id3).unwrap(), None);
    let sensor3_from_rig = test_rigid3d();
    rig.set_sensor_from_rig(sensor_id3, sensor3_from_rig)
        .unwrap();
    assert_eq!(rig.sensor_from_rig(sensor_id3).unwrap(), sensor3_from_rig);
    assert_eq!(
        rig.maybe_sensor_from_rig(sensor_id3).unwrap().unwrap(),
        sensor3_from_rig
    );
}

#[test]
fn rig_print() {
    let mut rig = Rig::new();
    rig.set_rig_id(0);
    rig.add_ref_sensor(SensorId::new(SensorType::Imu, 0))
        .unwrap();
    rig.add_sensor(
        SensorId::new(SensorType::Camera, 1),
        Some(Rigid3d::default()),
    )
    .unwrap();
    rig.add_sensor(
        SensorId::new(SensorType::Camera, 2),
        Some(Rigid3d::default()),
    )
    .unwrap();
    assert_eq!(
        rig.to_string(),
        "Rig(rig_id=0, ref_sensor_id=(IMU, 0), sensors=[(CAMERA, 1), (CAMERA, 2)])"
    );
}
