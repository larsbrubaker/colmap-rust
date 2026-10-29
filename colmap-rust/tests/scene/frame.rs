// Port of colmap/scene/frame_test.cc, 1:1 (same test names, values and checks), for
// colmap_rust::scene::frame::Frame. EXPECT_ANY_THROW becomes `is_err()`.
//
// COLMAP's frame holds a `Rig*`; the Rust frame holds the rig id and takes the rig as a
// parameter (the `scene` module header). The pointer checks become checks of the
// id-based API: `SetRigPtr(&rig)` succeeding is `verify_rig(&rig)` being `Ok`, `HasRigPtr()`
// false on a fresh frame is no rig verifying against it, and `AddDataId` on a frame with a
// rig is `add_data_id_in_rig`.

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::Vector3d;
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::scene::frame::Frame;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{DataId, SensorId, SensorType, INVALID_FRAME_ID, INVALID_RIG_ID};

fn test_rigid3d() -> Rigid3d {
    Rigid3d::new(random_eigen_quaterniond(), random_eigen_vector3d())
}

fn rig_with_id(rig_id: u32) -> Rig {
    let mut rig = Rig::new();
    rig.set_rig_id(rig_id);
    rig
}

fn assert_near(a: &[f64], b: &[f64], tol: f64) {
    assert_eq!(a.len(), b.len());
    for (x, y) in a.iter().zip(b) {
        assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
    }
}

fn vec3(v: Vector3d) -> [f64; 3] {
    [v.x, v.y, v.z]
}

#[test]
fn frame_default() {
    let frame = Frame::new();
    assert_eq!(frame.frame_id(), INVALID_FRAME_ID);
    assert!(!frame.has_pose());
    assert_eq!(frame.rig_id(), INVALID_RIG_ID);
    assert!(!frame.has_rig_id());
    // HasRigPtr() is false: no rig is bound, so not even a rig with the (invalid) frame rig
    // id verifies against the frame.
    assert!(frame.verify_rig(&Rig::new()).is_err());
    assert_eq!(frame.num_data_ids(), 0);
}

#[test]
fn frame_copy() {
    let mut frame = Frame::new();
    frame.set_frame_id(1);
    frame.set_rig_id(2).unwrap();
    let data_id1 = DataId::new(SensorId::new(SensorType::Camera, 1), 1);
    let data_id2 = DataId::new(SensorId::new(SensorType::Camera, 1), 2);
    frame.add_data_id(data_id1).unwrap();
    frame.finalize_data_ids();

    // Copy constructor: copy should not be finalized.
    let mut copy = frame.clone();
    assert!(!copy.has_final_data_ids());
    assert_eq!(copy.frame_id(), 1);
    assert_eq!(copy.rig_id(), 2);
    assert!(copy.has_data_id(data_id1));
    assert!(copy.add_data_id(data_id2).is_ok());
    assert!(copy.has_data_id(data_id2));
    assert!(copy.clear_data_ids().is_ok());
    assert_eq!(copy.num_data_ids(), 0);

    // Copy assignment: target should not be finalized.
    let mut assigned = Frame::new();
    assigned.clone_from(&frame);
    assert!(!assigned.has_final_data_ids());
    assert_eq!(assigned.frame_id(), 1);
    assert!(assigned.has_data_id(data_id1));
    assert!(assigned.add_data_id(data_id2).is_ok());

    // Original remains finalized.
    assert!(frame.has_final_data_ids());
    assert!(frame.add_data_id(data_id2).is_err());
}

#[test]
fn frame_set_up() {
    let mut frame = Frame::new();
    let mut rig = rig_with_id(1);

    assert!(!frame.has_rig_id());
    assert!(frame.verify_rig(&rig).is_err());

    frame.set_rig_id(1).unwrap();
    assert!(frame.has_rig_id());

    let sensor_id1 = SensorId::new(SensorType::Imu, 0);
    rig.add_ref_sensor(sensor_id1).unwrap();
    let sensor_id2 = SensorId::new(SensorType::Camera, 0);
    rig.add_sensor(sensor_id2, Some(test_rigid3d())).unwrap();
    assert!(frame.verify_rig(&rig).is_ok());

    assert!(frame.data_ids().is_empty());
    let data_id1 = DataId::new(sensor_id1, 2);
    assert!(!frame.has_data_id(data_id1));
    frame.add_data_id_in_rig(&rig, data_id1).unwrap();
    assert_eq!(frame.num_data_ids(), 1);
    assert!(frame.has_data_id(data_id1));
    assert_eq!(
        frame.data_ids().iter().copied().collect::<Vec<_>>(),
        vec![data_id1]
    );
    let data_id2 = DataId::new(sensor_id2, 5);
    assert!(!frame.has_data_id(data_id2));
    frame.add_data_id_in_rig(&rig, data_id2).unwrap();
    assert_eq!(frame.num_data_ids(), 2);
    assert!(frame.has_data_id(data_id2));
    let mut data_ids: Vec<DataId> = frame.data_ids().iter().copied().collect();
    let mut expected = vec![data_id1, data_id2];
    data_ids.sort();
    expected.sort();
    assert_eq!(data_ids, expected);
    frame.clear_data_ids().unwrap();
    assert_eq!(frame.num_data_ids(), 0);
    assert!(frame.data_ids().is_empty());
    assert!(!frame.has_pose());
}

#[test]
fn frame_set_reset_rig_ptr() {
    let mut frame = Frame::new();
    let mut rig = Rig::new();

    assert!(!frame.has_rig_id());
    assert!(frame.verify_rig(&rig).is_err());

    frame.set_rig_id(1).unwrap();
    assert!(frame.has_rig_id());

    // The rig still has an invalid id.
    assert!(frame.verify_rig(&rig).is_err());
    rig.set_rig_id(1);
    assert!(frame.verify_rig(&rig).is_ok());
    // ResetRigPtr: nothing is stored, so there is nothing to reset; the frame keeps its id.
    assert_eq!(frame.rig_id(), 1);

    let sensor_id1 = SensorId::new(SensorType::Imu, 0);
    let sensor_id2 = SensorId::new(SensorType::Camera, 0);
    frame.add_data_id(DataId::new(sensor_id1, 1)).unwrap();
    frame.add_data_id(DataId::new(sensor_id2, 1)).unwrap();
    assert!(frame.verify_rig(&rig).is_err());

    rig.add_ref_sensor(sensor_id1).unwrap();
    rig.add_sensor(sensor_id2, Some(test_rigid3d())).unwrap();
    assert!(frame.verify_rig(&rig).is_ok());
}

#[test]
fn frame_add_data_id() {
    let mut rig = rig_with_id(1);
    let sensor_id1 = SensorId::new(SensorType::Imu, 0);
    let sensor_id2 = SensorId::new(SensorType::Camera, 0);
    let sensor_id3 = SensorId::new(SensorType::Camera, 1);
    rig.add_ref_sensor(sensor_id1).unwrap();
    rig.add_sensor(sensor_id2, Some(test_rigid3d())).unwrap();

    let mut frame = Frame::new();
    frame.set_rig_id(1).unwrap();
    frame.verify_rig(&rig).unwrap();
    frame
        .add_data_id_in_rig(&rig, DataId::new(sensor_id1, 1))
        .unwrap();
    frame
        .add_data_id_in_rig(&rig, DataId::new(sensor_id2, 1))
        .unwrap();
    assert!(frame
        .add_data_id_in_rig(&rig, DataId::new(sensor_id3, 2))
        .is_err());
}

#[test]
fn frame_filtered_data_ids() {
    let mut frame = Frame::new();
    let data_id1 = DataId::new(SensorId::new(SensorType::Imu, 0), 2);
    frame.add_data_id(data_id1).unwrap();
    let data_id2 = DataId::new(SensorId::new(SensorType::Camera, 0), 2);
    frame.add_data_id(data_id2).unwrap();
    let data_id3 = DataId::new(SensorId::new(SensorType::Camera, 1), 1);
    frame.add_data_id(data_id3).unwrap();
    assert_eq!(
        frame.data_ids_of_type(SensorType::Imu).collect::<Vec<_>>(),
        vec![data_id1]
    );
    let mut cameras: Vec<DataId> = frame.data_ids_of_type(SensorType::Camera).collect();
    cameras.sort();
    assert_eq!(cameras, vec![data_id2, data_id3]);
    assert_eq!(frame.data_ids_of_type(SensorType::Invalid).count(), 0);
}

#[test]
fn frame_image_ids() {
    let mut frame = Frame::new();
    let data_id1 = DataId::new(SensorId::new(SensorType::Imu, 0), 2);
    frame.add_data_id(data_id1).unwrap();
    let data_id2 = DataId::new(SensorId::new(SensorType::Camera, 0), 2);
    frame.add_data_id(data_id2).unwrap();
    let data_id3 = DataId::new(SensorId::new(SensorType::Camera, 1), 1);
    frame.add_data_id(data_id3).unwrap();
    let mut image_ids: Vec<DataId> = frame.image_ids().collect();
    image_ids.sort();
    assert_eq!(image_ids, vec![data_id2, data_id3]);
}

#[test]
fn frame_set_reset_pose() {
    let mut frame = Frame::new();
    assert!(!frame.has_pose());
    assert!(frame.rig_from_world().is_err());
    assert_eq!(frame.maybe_rig_from_world(), None);
    frame.set_rig_from_world(Rigid3d::default());
    assert!(frame.has_pose());
    assert_eq!(frame.rig_from_world().unwrap(), Rigid3d::default());
    assert_eq!(frame.maybe_rig_from_world().unwrap(), Rigid3d::default());
    frame.reset_pose();
    assert!(!frame.has_pose());
    assert!(frame.rig_from_world().is_err());
    assert_eq!(frame.maybe_rig_from_world(), None);
}

#[test]
fn frame_set_cam_from_world() {
    let mut frame = Frame::new();
    let mut rig = rig_with_id(1);
    frame.set_rig_id(1).unwrap();
    let sensor_id1 = SensorId::new(SensorType::Camera, 0);
    rig.add_ref_sensor(sensor_id1).unwrap();
    let sensor_id2 = SensorId::new(SensorType::Camera, 1);
    rig.add_sensor(sensor_id2, Some(test_rigid3d())).unwrap();
    frame.verify_rig(&rig).unwrap();

    let cam1_from_world = test_rigid3d();
    frame
        .set_cam_from_world(&rig, sensor_id1.id, cam1_from_world)
        .unwrap();
    assert_eq!(frame.rig_from_world().unwrap(), cam1_from_world);
    assert_eq!(
        frame.sensor_from_world(&rig, sensor_id1).unwrap(),
        cam1_from_world
    );

    let cam2_from_world = test_rigid3d();
    frame
        .set_cam_from_world(&rig, sensor_id2.id, cam2_from_world)
        .unwrap();
    let sensor2_from_world = frame.sensor_from_world(&rig, sensor_id2).unwrap();
    assert_near(
        &vec3(cam2_from_world.translation),
        &vec3(sensor2_from_world.translation),
        1e-6,
    );
    let c2 = cam2_from_world.rotation.coeffs();
    let s2 = sensor2_from_world.rotation.coeffs();
    assert_near(&[c2.x, c2.y, c2.z, c2.w], &[s2.x, s2.y, s2.z, s2.w], 1e-6);
}

#[test]
fn image_equals() {
    let mut frame = Frame::new();
    let mut other = frame.clone();
    assert_eq!(frame, other);
    frame.set_frame_id(2);
    assert_ne!(frame, other);
    other.set_frame_id(2);
    assert_eq!(frame, other);
}

#[test]
fn frame_print() {
    let mut frame = Frame::new();
    frame.set_frame_id(1);
    frame.set_rig_id(2).unwrap();
    frame
        .add_data_id(DataId::new(SensorId::new(SensorType::Imu, 0), 2))
        .unwrap();
    frame
        .add_data_id(DataId::new(SensorId::new(SensorType::Camera, 1), 3))
        .unwrap();
    assert_eq!(
        frame.to_string(),
        "Frame(frame_id=1, rig_id=2, has_pose=0, data_ids=[(CAMERA, 1, 3), (IMU, 0, 2)])"
    );
}

#[test]
fn frame_finalize_data_ids() {
    let mut frame = Frame::new();
    frame.set_frame_id(1);
    frame.set_rig_id(2).unwrap();
    let data_id1 = DataId::new(SensorId::new(SensorType::Camera, 1), 1);
    let data_id2 = DataId::new(SensorId::new(SensorType::Camera, 1), 2);
    frame.add_data_id(data_id1).unwrap();
    assert!(!frame.has_final_data_ids());
    frame.finalize_data_ids();
    assert!(frame.has_final_data_ids());
    assert!(frame.add_data_id(data_id2).is_err());
    assert!(frame.clear_data_ids().is_err());
    assert_eq!(frame.num_data_ids(), 1);
    assert!(frame.has_data_id(data_id1));
}
