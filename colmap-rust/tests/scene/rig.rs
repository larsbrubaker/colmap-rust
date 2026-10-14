// Port of colmap/scene/rig_test.cc, 1:1 (same test names, values and checks), with its test
// helper `CreateTestData`: colmap_rust::scene::rig::read_rig_config and
// colmap_rust::scene::rig_apply::apply_rig_config. `ReadRigConfig(WriteTestConfig(text))`
// reads the text the test would have written to the file (`read_rig_config` takes the file's
// text); `Database::Open(kInMemorySqliteDatabasePath)` is `Database::new()`. COLMAP's gtest
// main seeds the PRNG with 0 before every test, so each test that draws calls
// `set_prng_seed(0)` first. `ResetRigPtr`/`ResetCameraPtr`/`ResetFramePtr` have no
// counterpart: the Rust scene types hold ids, not pointers (docs/CPP_DIVERGENCES.md
// entry 170).

use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{Vector3d, Vector4d};
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::rig::{read_rig_config, RigCamera, RigConfig};
use colmap_rust::scene::rig_apply::apply_rig_config;
use colmap_rust::scene::synthetic::{synthesize_dataset_with_database, SyntheticDatasetOptions};
use colmap_rust::sensor::models::CameraModelId;
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{DataId, SensorId};

/// `WriteTestConfig` + `ReadRigConfig`: the file holds the config and a newline.
fn read_test_config(config: &str) -> colmap_rust::Result<Vec<RigConfig>> {
    read_rig_config(&format!("{config}\n"))
}

#[test]
fn read_rig_config_empty() {
    assert!(read_test_config("[]").unwrap().is_empty());
}

#[test]
fn read_rig_config_invalid_json() {
    assert!(read_test_config("").is_err());
    assert!(read_test_config("[{").is_err());
}

#[test]
fn read_rig_config_missing_image_prefix() {
    assert!(read_test_config(
        r#"
          [
            {
              "cameras": [
                {
                    "ref_sensor": true,
                }
              ]
            }
          ]
          "#
    )
    .is_err());
}

#[test]
fn read_rig_config_invalid_ref_sensor() {
    assert!(read_test_config(
        r#"
          [
            {
              "cameras": [
                {
                    "image_prefix": "rig1/camera1/",
                    "ref_sensor": true,
                },
                {
                    "image_prefix": "rig1/camera2/",
                    "ref_sensor": true,
                }
              ]
            }
          ]
          "#
    )
    .is_err());
    assert!(read_test_config(
        r#"
              [
                {
                  "cameras": [
                    {
                        "image_prefix": "rig1/camera1/",
                    },
                    {
                        "image_prefix": "rig1/camera2/",
                    }
                  ]
                }
              ]
              "#
    )
    .is_err());
}

#[test]
fn read_rig_config_nominal() {
    let configs = read_test_config(
        r#"
[
  {
    "cameras": [
      {
          "image_prefix": "rig1/camera1/",
          "ref_sensor": true,
          "camera_model_name": "OPENCV",
          "camera_params": [640, 480, 320, 240, 0.1, 0.2, 0.3, 0.4]
      },
      {
          "image_prefix": "rig1/camera2/",
          "cam_from_rig_rotation": [0, 1, 0, 0],
          "cam_from_rig_translation": [1, 2, 3]
      }
    ]
  },
  {
    "cameras": [
      {
          "image_prefix": "rig2/camera1/",
          "ref_sensor": true
      },
      {
          "image_prefix": "rig2/camera2/"
      }
    ]
  }
]
"#,
    )
    .unwrap();
    assert_eq!(configs.len(), 2);
    assert_eq!(configs[0].cameras.len(), 2);
    assert_eq!(configs[1].cameras.len(), 2);

    assert_eq!(configs[0].cameras[0].image_prefix, "rig1/camera1/");
    assert!(configs[0].cameras[0].ref_sensor);
    assert!(configs[0].cameras[0].cam_from_rig.is_none());
    let camera = configs[0].cameras[0].camera.as_ref().unwrap();
    assert_eq!(camera.model_id, CameraModelId::OpenCV);
    assert!(camera.has_prior_focal_length);
    assert_eq!(
        camera.params,
        [640.0, 480.0, 320.0, 240.0, 0.1, 0.2, 0.3, 0.4]
    );

    assert_eq!(configs[0].cameras[1].image_prefix, "rig1/camera2/");
    assert!(!configs[0].cameras[1].ref_sensor);
    let cam_from_rig = configs[0].cameras[1].cam_from_rig.unwrap();
    assert_eq!(
        cam_from_rig.rotation.coeffs(),
        Vector4d::new(1.0, 0.0, 0.0, 0.0)
    );
    assert_eq!(cam_from_rig.translation, Vector3d::new(1.0, 2.0, 3.0));
    assert!(configs[0].cameras[1].camera.is_none());

    assert_eq!(configs[1].cameras[0].image_prefix, "rig2/camera1/");
    assert!(configs[1].cameras[0].ref_sensor);
    assert!(configs[1].cameras[0].cam_from_rig.is_none());
    assert!(configs[1].cameras[0].camera.is_none());

    assert_eq!(configs[1].cameras[1].image_prefix, "rig2/camera2/");
    assert!(!configs[1].cameras[1].ref_sensor);
    assert!(configs[1].cameras[1].cam_from_rig.is_none());
    assert!(configs[1].cameras[1].camera.is_none());
}

/// Port of rig_test.cc's `CreateTestData`: a synthetic dataset with a single rig and trivial
/// frames, whose images are then renamed `camera<i>_frame<j>` to match a rig config.
pub(crate) fn create_test_data(
    num_frames: i32,
    num_cameras_per_rig: i32,
    database: &Database,
    reconstruction: &mut Reconstruction,
) {
    // Create a synthetic dataset with a single rig and trivial frames.
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: num_frames * num_cameras_per_rig,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, reconstruction, database).unwrap();

    // Manually overwrite the image names to match the expected rig config.
    let mut image_idx = 0;
    let mut frame_idx = 0;
    for mut image in database.read_all_images().unwrap() {
        let is_first_of_frame = image_idx % num_cameras_per_rig == 0;
        image_idx += 1;
        if is_first_of_frame {
            frame_idx += 1;
        }
        image.set_name(&format!(
            "camera{}_frame{}",
            image_idx % num_cameras_per_rig,
            frame_idx
        ));
        reconstruction
            .image_mut(image.image_id())
            .unwrap()
            .set_name(image.name());
        database.update_image(&image).unwrap();
    }
}

/// A config with one rig of two cameras: "camera0_" (reference) and "camera1_".
fn two_camera_config() -> Vec<RigConfig> {
    vec![RigConfig {
        cameras: vec![
            RigCamera {
                image_prefix: "camera0_".to_string(),
                ref_sensor: true,
                ..RigCamera::default()
            },
            RigCamera {
                image_prefix: "camera1_".to_string(),
                ..RigCamera::default()
            },
        ],
    }]
}

#[test]
fn apply_rig_config_with_reconstruction() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    create_test_data(
        /*num_frames=*/ 5,
        /*num_cameras_per_rig=*/ 2,
        &database,
        &mut reconstruction,
    );

    let configs = two_camera_config();

    apply_rig_config(&configs, &database, Some(&mut reconstruction)).unwrap();
    assert_eq!(database.num_rigs().unwrap(), 1);
    assert_eq!(database.num_frames().unwrap(), 5);
    assert_eq!(reconstruction.num_rigs(), 1);
    assert_eq!(reconstruction.num_frames(), 5);
    assert_eq!(reconstruction.num_reg_frames(), 5);
}

#[test]
fn apply_rig_config_with_differing_database_and_reconstruction_ids() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    create_test_data(
        /*num_frames=*/ 5,
        /*num_cameras_per_rig=*/ 2,
        &database,
        &mut reconstruction,
    );

    // Create a reconstruction with differing rig/camera/frame/image ids.
    let mut differing_reconstruction = Reconstruction::new();
    for camera in reconstruction.cameras().values() {
        let mut differing_camera = camera.clone();
        differing_camera.camera_id = camera.camera_id + 1;
        differing_reconstruction
            .add_camera(differing_camera)
            .unwrap();
    }
    for rig in reconstruction.rigs().values() {
        let mut differing_rig = Rig::new();
        differing_rig.set_rig_id(rig.rig_id() + 1);
        differing_rig
            .add_ref_sensor(SensorId::new(
                rig.ref_sensor_id().sensor_type,
                rig.ref_sensor_id().id + 1,
            ))
            .unwrap();
        for (sensor_id, sensor_from_rig) in rig.non_ref_sensors() {
            differing_rig
                .add_sensor(
                    SensorId::new(sensor_id.sensor_type, sensor_id.id + 1),
                    *sensor_from_rig,
                )
                .unwrap();
        }
        differing_reconstruction.add_rig(differing_rig).unwrap();
    }
    for frame in reconstruction.frames().values() {
        let mut differing_frame = frame.clone();
        differing_frame.set_frame_id(frame.frame_id() + 1);
        differing_frame.set_rig_id(frame.rig_id() + 1).unwrap();
        differing_frame.clear_data_ids().unwrap();
        for data_id in frame.data_ids() {
            differing_frame
                .add_data_id(DataId::new(
                    SensorId::new(data_id.sensor_id.sensor_type, data_id.sensor_id.id + 1),
                    data_id.id as u32 + 1,
                ))
                .unwrap();
        }
        differing_reconstruction.add_frame(differing_frame).unwrap();
    }
    for image in reconstruction.images().values() {
        let mut differing_image = image.clone();
        differing_image.set_image_id(image.image_id() + 1);
        differing_image
            .set_camera_id(image.camera_id() + 1)
            .unwrap();
        differing_image.set_frame_id(image.frame_id() + 1);
        differing_reconstruction.add_image(differing_image).unwrap();
    }

    let configs = two_camera_config();

    apply_rig_config(&configs, &database, Some(&mut differing_reconstruction)).unwrap();
    assert_eq!(database.num_rigs().unwrap(), 1);
    assert_eq!(database.num_frames().unwrap(), 5);
    assert_eq!(differing_reconstruction.num_rigs(), 1);
    assert_eq!(differing_reconstruction.num_frames(), 5);
    assert_eq!(differing_reconstruction.num_reg_frames(), 5);
}

#[test]
fn apply_rig_config_with_partial_reconstruction() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    create_test_data(
        /*num_frames=*/ 5,
        /*num_cameras_per_rig=*/ 2,
        &database,
        &mut reconstruction,
    );

    reconstruction.deregister_frame(1).unwrap();
    reconstruction.deregister_frame(2).unwrap();
    // De-register only one image in the frame.
    // The other image allows us to register the frame.
    reconstruction.deregister_frame(3).unwrap();

    let configs = two_camera_config();

    apply_rig_config(&configs, &database, Some(&mut reconstruction)).unwrap();
    assert_eq!(database.num_rigs().unwrap(), 1);
    assert_eq!(database.num_frames().unwrap(), 5);
    assert_eq!(reconstruction.num_rigs(), 1);
    assert_eq!(reconstruction.num_frames(), 5);
    assert_eq!(reconstruction.num_reg_frames(), 4);
}

#[test]
fn apply_rig_config_without_reconstruction() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    create_test_data(
        /*num_frames=*/ 5,
        /*num_cameras_per_rig=*/ 2,
        &database,
        &mut reconstruction,
    );

    let mut configs = two_camera_config();
    let camera2 = &mut configs[0].cameras[1];
    camera2.camera =
        Some(Camera::create_from_model_id(2, CameraModelId::OpenCV, 2.0, 1024, 768).unwrap());
    camera2.cam_from_rig = Some(Rigid3d::new(
        random_eigen_quaterniond(),
        random_eigen_vector3d(),
    ));
    let camera2 = configs[0].cameras[1].clone();

    apply_rig_config(&configs, &database, None).unwrap();
    assert_eq!(database.num_rigs().unwrap(), 1);
    assert_eq!(database.num_frames().unwrap(), 5);
    let rigs = database.read_all_rigs().unwrap();
    let (sensor_id2, sensor2_from_rig) = rigs[0].non_ref_sensors().iter().next().unwrap();
    assert_eq!(
        sensor2_from_rig.unwrap().rotation.coeffs(),
        camera2.cam_from_rig.unwrap().rotation.coeffs()
    );
    assert_eq!(
        sensor2_from_rig.unwrap().translation,
        camera2.cam_from_rig.unwrap().translation
    );
    assert_eq!(
        Some(database.read_camera(sensor_id2.id).unwrap()),
        camera2.camera
    );
}

#[test]
fn apply_rig_config_with_unconfigured_single_and_configured_multi_camera_rigs() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let mut options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 3,
        num_frames_per_rig: 5,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();

    options.num_rigs = 1;
    options.num_cameras_per_rig = 1;
    options.num_frames_per_rig = 3;
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();

    let configs = vec![RigConfig {
        cameras: vec![
            RigCamera {
                image_prefix: "camera000001_".to_string(),
                ref_sensor: true,
                ..RigCamera::default()
            },
            RigCamera {
                image_prefix: "camera000002_".to_string(),
                ..RigCamera::default()
            },
        ],
    }];

    apply_rig_config(&configs, &database, Some(&mut reconstruction)).unwrap();
    assert_eq!(database.num_rigs().unwrap(), 3);
    assert_eq!(database.num_frames().unwrap(), 13);
    assert_eq!(reconstruction.num_rigs(), 3);
    assert_eq!(reconstruction.num_frames(), 13);
    assert_eq!(reconstruction.num_reg_frames(), 13);

    let mut num_non_trivial_rigs = 0;
    for rig in database.read_all_rigs().unwrap() {
        num_non_trivial_rigs += i32::from(rig.num_sensors() > 1);
    }
    assert_eq!(num_non_trivial_rigs, 1);

    let mut num_non_trivial_frames = 0;
    for frame in database.read_all_frames().unwrap() {
        num_non_trivial_frames += i32::from(frame.num_data_ids() > 1);
    }
    assert_eq!(num_non_trivial_frames, 5);
}
