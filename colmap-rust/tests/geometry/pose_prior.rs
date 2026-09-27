// Port of COLMAP's src/colmap/geometry/pose_prior_test.cc (all cases), plus a Rust-only
// test of the default values and the Has* predicates.

use colmap_rust::geometry::pose_prior::{
    compute_rot90_from_gravity, gravity_from_exif_orientation,
};
use colmap_rust::geometry::{CoordinateSystem, PosePrior};
use colmap_rust::linalg::{Matrix3d, Vector3d};
use colmap_rust::util::types::{DataId, SensorId, SensorType};

fn nominal_prior() -> PosePrior {
    PosePrior {
        pose_prior_id: 0,
        corr_data_id: DataId::new(SensorId::new(SensorType::Camera, 1), 2),
        position: Vector3d::zeros(),
        position_covariance: Matrix3d::identity(),
        coordinate_system: CoordinateSystem::Cartesian,
        gravity: Vector3d::unit_z(),
    }
}

#[test]
fn pose_prior_equals() {
    let mut prior = nominal_prior();
    let mut other = prior;
    assert_eq!(prior, other);
    prior.position.x = 1.0;
    assert_ne!(prior, other);
    other.position.x = 1.0;
    assert_eq!(prior, other);
}

#[test]
fn pose_prior_nan_equals() {
    let mut prior = PosePrior::default();
    let mut other = prior;
    assert_eq!(prior, other);
    prior.position = Vector3d::new(1.0, 2.0, f64::NAN);
    other.position = Vector3d::new(1.0, 2.0, f64::NAN);
    assert_eq!(prior, other);
    prior.position_covariance = Matrix3d::identity();
    prior.position_covariance[(0, 0)] = f64::NAN;
    other.position_covariance = Matrix3d::identity();
    other.position_covariance[(0, 0)] = f64::NAN;
    assert_eq!(prior, other);
    other.position.z = 3.0;
    assert_ne!(prior, other);
}

#[test]
fn pose_prior_print() {
    assert_eq!(
        nominal_prior().to_string(),
        "PosePrior(pose_prior_id=0, corr_data_id=(CAMERA, 1, 2), \
         position=[0, 0, 0], \
         position_covariance=[1, 0, 0, 0, 1, 0, 0, 0, 1], \
         coordinate_system=CARTESIAN, gravity=[0, 0, 1])"
    );
}

#[test]
fn pose_prior_gravity_from_exif_orientation() {
    assert_eq!(
        gravity_from_exif_orientation(1).unwrap(),
        Vector3d::new(0.0, 1.0, 0.0)
    );
    assert_eq!(
        gravity_from_exif_orientation(3).unwrap(),
        Vector3d::new(0.0, -1.0, 0.0)
    );
    assert_eq!(
        gravity_from_exif_orientation(6).unwrap(),
        Vector3d::new(1.0, 0.0, 0.0)
    );
    assert_eq!(
        gravity_from_exif_orientation(8).unwrap(),
        Vector3d::new(-1.0, 0.0, 0.0)
    );
    for orientation in [2, 4, 5, 7, 0, 42, -1] {
        assert!(gravity_from_exif_orientation(orientation).is_none());
    }
}

#[test]
fn pose_prior_compute_rot90_from_gravity() {
    // Normal
    assert_eq!(compute_rot90_from_gravity(Vector3d::new(0.0, 1.0, 0.0)), 0);
    // Gravity is +x (right). Need 90 CW (270 CCW) to make upright.
    assert_eq!(compute_rot90_from_gravity(Vector3d::new(1.0, 0.0, 0.0)), 3);
    // Gravity is -y (up). Need 180 CCW to make upright.
    assert_eq!(compute_rot90_from_gravity(Vector3d::new(0.0, -1.0, 0.0)), 2);
    // Gravity is -x (left). Need 90 CCW to make upright.
    assert_eq!(compute_rot90_from_gravity(Vector3d::new(-1.0, 0.0, 0.0)), 1);
    // Robustness to slight inaccuracies.
    assert_eq!(
        compute_rot90_from_gravity(Vector3d::new(0.01, 0.99, 0.1)),
        0
    );
    assert_eq!(
        compute_rot90_from_gravity(Vector3d::new(0.99, -0.01, 0.1)),
        3
    );
}

#[test]
fn rust_only_pose_prior_default_and_predicates() {
    let prior = PosePrior::default();
    assert_eq!(prior.coordinate_system, CoordinateSystem::Undefined);
    assert!(!prior.has_position());
    assert!(!prior.has_position_cov());
    assert!(!prior.has_gravity());
    let nominal = nominal_prior();
    assert!(nominal.has_position());
    assert!(nominal.has_position_cov());
    assert!(nominal.has_gravity());
    assert_eq!(
        PosePrior::default().to_string(),
        "PosePrior(pose_prior_id=4294967295, corr_data_id=(INVALID, 4294967295, 4294967295), \
         position=[nan, nan, nan], \
         position_covariance=[nan, nan, nan, nan, nan, nan, nan, nan, nan], \
         coordinate_system=UNDEFINED, gravity=[nan, nan, nan])"
    );
}
