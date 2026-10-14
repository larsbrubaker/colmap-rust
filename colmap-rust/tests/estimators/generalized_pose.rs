// Port of COLMAP's src/colmap/estimators/generalized_pose_test.cc (same values and
// tolerances), testing colmap_rust::estimators::generalized_pose: the
// EstimateGeneralizedAbsolutePose, EstimateGeneralizedRelativePose and
// EstimateStructureLessAbsolutePose cases. The three RefineGeneralizedAbsolutePose cases wait
// for its Ceres-based function. Tier C (RANSAC).
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. Arguments COLMAP draws
// inside one call are drawn here in clang's left-to-right order, and `std::shuffle(..., *PRNG)`
// is libc++'s algorithm (`math::random::libcxx::shuffle`). The tests' hash maps are only
// looked up, never iterated, so std's HashMap stands in for FlatHashMap / NodeHashMap.

use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

use crate::transform_matchers::rigid3d_near;
use colmap_rust::estimators::generalized_pose::{
    estimate_generalized_absolute_pose, estimate_generalized_relative_pose,
    estimate_structure_less_absolute_pose, StructureLessAbsolutePoseEstimationOptions,
};
use colmap_rust::estimators::solvers::GP3PEstimator;
use colmap_rust::geometry::Rigid3d;
use colmap_rust::linalg::{Vector2d, Vector3d};
use colmap_rust::math::random::libcxx;
use colmap_rust::math::random::{set_prng_seed, with_prng};
use colmap_rust::math::random_eigen::{
    random_eigen_quaterniond, random_eigen_vector2d, random_eigen_vector3d,
};
use colmap_rust::optim::{Estimator, RansacOptions};
use colmap_rust::scene::camera::Camera;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::util::types::{CameraId, ImageId, Point2DIdx, Point3DId};

fn image_cam_from_world(reconstruction: &Reconstruction, image: &Image) -> Rigid3d {
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    image.cam_from_world(frame, rig).unwrap()
}

fn image_camera(reconstruction: &Reconstruction, image: &Image) -> Camera {
    reconstruction.camera(image.camera_id()).unwrap().clone()
}

// `std::shuffle(v.begin(), v.end(), *PRNG)` over `std::iota` indices.
fn shuffled_indices(len: usize) -> Vec<usize> {
    let mut shuffled_idxs: Vec<usize> = (0..len).collect();
    with_prng(|g| libcxx::shuffle(&mut shuffled_idxs, g));
    shuffled_idxs
}

struct GeneralizedAbsolutePoseProblem {
    gt_rig_from_world: Rigid3d,
    points2d: Vec<Vector2d>,
    points3d: Vec<Vector3d>,
    point3d_ids: Vec<Point3DId>,
    camera_idxs: Vec<usize>,
    cams_from_rig: Vec<Rigid3d>,
    cameras: Vec<Camera>,
}

fn build_generalized_absolute_pose_problem() -> GeneralizedAbsolutePoseProblem {
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 2,
        num_frames_per_rig: 1,
        num_points3d: 50,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();

    let rotation = random_eigen_quaterniond();
    let translation = random_eigen_vector3d();
    let mut problem = GeneralizedAbsolutePoseProblem {
        gt_rig_from_world: Rigid3d::new(rotation, translation),
        points2d: Vec::new(),
        points3d: Vec::new(),
        point3d_ids: Vec::new(),
        camera_idxs: Vec::new(),
        cams_from_rig: Vec::new(),
        cameras: Vec::new(),
    };
    for image_id in reconstruction.reg_image_ids().unwrap() {
        let image = reconstruction.image(image_id).unwrap();
        for point2d in image.points2d() {
            if point2d.has_point3d() {
                problem.points2d.push(point2d.xy);
                problem
                    .points3d
                    .push(reconstruction.point3d(point2d.point3d_id).unwrap().xyz);
                problem.point3d_ids.push(point2d.point3d_id);
                problem.camera_idxs.push(problem.cameras.len());
            }
        }
        problem.cameras.push(image_camera(&reconstruction, image));
        problem.cams_from_rig.push(
            image_cam_from_world(&reconstruction, image) * problem.gt_rig_from_world.inverse(),
        );
    }
    problem
}

#[test]
fn estimate_generalized_absolute_pose_nominal() {
    set_prng_seed(0);
    let mut problem = build_generalized_absolute_pose_problem();
    let num_points = problem.points2d.len();

    let gt_inlier_ratio = 0.8;
    let outlier_distance = 50.0;
    let gt_num_inliers =
        ((gt_inlier_ratio * num_points as f64) as usize).max(GP3PEstimator::MIN_NUM_SAMPLES);
    let shuffled_idxs = shuffled_indices(num_points);

    let mut unique_inlier_ids = HashSet::with_capacity(gt_num_inliers);
    for &shuffled_idx in &shuffled_idxs[..gt_num_inliers] {
        unique_inlier_ids.insert(problem.point3d_ids[shuffled_idx]);
    }

    let mut gt_inlier_mask = vec![true; num_points];
    for &shuffled_idx in &shuffled_idxs[gt_num_inliers..] {
        problem.points2d[shuffled_idx] += random_eigen_vector2d().normalized() * outlier_distance;
        gt_inlier_mask[shuffled_idx] = false;
    }

    let ransac_options = RansacOptions {
        max_error: 2.0,
        min_inlier_ratio: gt_inlier_ratio / 2.0,
        confidence: 0.99999,
        ..RansacOptions::default()
    };

    let estimate = estimate_generalized_absolute_pose(
        &ransac_options,
        &problem.points2d,
        &problem.points3d,
        &problem.camera_idxs,
        &problem.cams_from_rig,
        &problem.cameras,
    )
    .unwrap()
    .expect("EstimateGeneralizedAbsolutePose succeeds");
    assert_eq!(estimate.num_inliers, unique_inlier_ids.len());
    assert_eq!(estimate.inlier_mask, gt_inlier_mask);
    assert!(
        rigid3d_near(&estimate.pose, &problem.gt_rig_from_world, 1e-6, 1e-6),
        "{:?} vs {:?}",
        estimate.pose,
        problem.gt_rig_from_world
    );
}

struct GeneralizedRelativePoseProblem {
    gt_rig2_from_rig1: Rigid3d,
    points2d1: Vec<Vector2d>,
    points2d2: Vec<Vector2d>,
    camera_idxs1: Vec<usize>,
    camera_idxs2: Vec<usize>,
    cams_from_rig: Vec<Rigid3d>,
    cameras: Vec<Camera>,
}

impl GeneralizedRelativePoseProblem {
    // The `maybe_add_and_get_camera` lambda: the problem's index of the image's camera, added
    // with its sensor_from_rig on first use.
    fn maybe_add_and_get_camera(
        &mut self,
        camera_id_to_idx: &mut HashMap<CameraId, usize>,
        reconstruction: &Reconstruction,
        image: &Image,
    ) -> usize {
        if let Some(&idx) = camera_id_to_idx.get(&image.camera_id()) {
            return idx;
        }
        let idx = self.cameras.len();
        camera_id_to_idx.insert(image.camera_id(), idx);
        let camera = image_camera(reconstruction, image);
        let frame = reconstruction.frame(image.frame_id()).unwrap();
        let rig = reconstruction.rig(frame.rig_id()).unwrap();
        if rig.is_ref_sensor(camera.sensor_id()) {
            self.cams_from_rig.push(Rigid3d::default());
        } else {
            self.cams_from_rig
                .push(rig.sensor_from_rig(camera.sensor_id()).unwrap());
        }
        self.cameras.push(camera);
        idx
    }
}

fn build_generalized_relative_pose_problem(
    mut num_cameras_per_rig1: i32,
    mut num_cameras_per_rig2: i32,
    sensor_from_rig_translation_stddev: f64,
) -> GeneralizedRelativePoseProblem {
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: num_cameras_per_rig1.max(num_cameras_per_rig2),
        num_frames_per_rig: 1,
        num_points3d: 100,
        sensor_from_rig_translation_stddev,
        sensor_from_rig_rotation_stddev: 10.0,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();

    let frame1 = reconstruction.frame(1).unwrap();
    let frame2 = reconstruction.frame(2).unwrap();
    assert_ne!(frame1.rig_id(), frame2.rig_id());

    let mut problem = GeneralizedRelativePoseProblem {
        gt_rig2_from_rig1: frame2.rig_from_world().unwrap()
            * frame1.rig_from_world().unwrap().inverse(),
        points2d1: Vec::new(),
        points2d2: Vec::new(),
        camera_idxs1: Vec::new(),
        camera_idxs2: Vec::new(),
        cams_from_rig: Vec::new(),
        cameras: Vec::new(),
    };

    let mut observations2: HashMap<Point3DId, Vec<(ImageId, Point2DIdx)>> = HashMap::new();
    for data_id in frame2.image_ids() {
        let image = reconstruction.image(data_id.id as ImageId).unwrap();
        for point2d_idx in 0..image.num_points2d() {
            let point2d = image.point2d(point2d_idx);
            if point2d.has_point3d() {
                observations2
                    .entry(point2d.point3d_id)
                    .or_default()
                    .push((image.image_id(), point2d_idx));
            }
        }
        num_cameras_per_rig2 -= 1;
        if num_cameras_per_rig2 == 0 {
            break;
        }
    }

    let mut camera_id_to_idx: HashMap<CameraId, usize> = HashMap::new();
    for data_id in frame1.image_ids() {
        let image1 = reconstruction.image(data_id.id as ImageId).unwrap();
        for point2d_idx1 in 0..image1.num_points2d() {
            let point2d1 = image1.point2d(point2d_idx1);
            let Some(observations) = observations2.get(&point2d1.point3d_id) else {
                continue;
            };
            for &(image2_id, point2d_idx2) in observations {
                let image2 = reconstruction.image(image2_id).unwrap();
                problem.points2d1.push(point2d1.xy);
                problem.points2d2.push(image2.point2d(point2d_idx2).xy);
                let camera_idx1 = problem.maybe_add_and_get_camera(
                    &mut camera_id_to_idx,
                    &reconstruction,
                    image1,
                );
                problem.camera_idxs1.push(camera_idx1);
                let camera_idx2 = problem.maybe_add_and_get_camera(
                    &mut camera_id_to_idx,
                    &reconstruction,
                    image2,
                );
                problem.camera_idxs2.push(camera_idx2);
            }
        }

        num_cameras_per_rig1 -= 1;
        if num_cameras_per_rig1 == 0 {
            break;
        }
    }

    problem
}

#[test]
fn estimate_generalized_relative_pose_nominal() {
    set_prng_seed(0);
    for num_cameras_per_rig1 in [1, 2, 3] {
        for num_cameras_per_rig2 in [1, 2, 3] {
            // A meaningful inter-camera baseline is needed to recover metric scale
            // reliably in non-panoramic configurations.
            for sensor_from_rig_translation_stddev in [0.0, 0.2] {
                let case = format!(
                    "rig1 {num_cameras_per_rig1}, rig2 {num_cameras_per_rig2}, \
                     stddev {sensor_from_rig_translation_stddev}"
                );
                let problem = build_generalized_relative_pose_problem(
                    num_cameras_per_rig1,
                    num_cameras_per_rig2,
                    sensor_from_rig_translation_stddev,
                );

                let ransac_options = RansacOptions {
                    max_error: 1.0,
                    ..RansacOptions::default()
                };

                let estimate = estimate_generalized_relative_pose(
                    &ransac_options,
                    &problem.points2d1,
                    &problem.points2d2,
                    &problem.camera_idxs1,
                    &problem.camera_idxs2,
                    &problem.cams_from_rig,
                    &problem.cameras,
                )
                .unwrap()
                .unwrap_or_else(|| panic!("{case}: EstimateGeneralizedRelativePose succeeds"));
                assert_eq!(estimate.num_inliers, problem.points2d1.len(), "{case}");
                assert!(estimate.inlier_mask.iter().all(|&inlier| inlier), "{case}");
                if (num_cameras_per_rig1 == 1 && num_cameras_per_rig2 == 1)
                    || sensor_from_rig_translation_stddev == 0.0
                {
                    // Panoramic pairs do not allow for recovery of translation scale.
                    assert!(estimate.pose.rig2_from_rig1().is_none(), "{case}");
                    let pano2_from_pano1 = estimate.pose.pano2_from_pano1().expect(&case);
                    let expected = Rigid3d::new(
                        problem.gt_rig2_from_rig1.rotation,
                        problem.gt_rig2_from_rig1.translation.normalized(),
                    );
                    assert!(
                        rigid3d_near(&pano2_from_pano1, &expected, 1e-6, 1e-6),
                        "{case}: {pano2_from_pano1:?} vs {expected:?}"
                    );
                } else {
                    let rig2_from_rig1 = estimate.pose.rig2_from_rig1().expect(&case);
                    assert!(estimate.pose.pano2_from_pano1().is_none(), "{case}");
                    assert!(
                        rigid3d_near(&rig2_from_rig1, &problem.gt_rig2_from_rig1, 1e-3, 2e-3),
                        "{case}: {rig2_from_rig1:?} vs {:?}",
                        problem.gt_rig2_from_rig1
                    );
                }
            }
        }
    }
}

struct StructureLessAbsolutePoseProblem {
    gt_cam_from_world: Rigid3d,
    world_points2d: Vec<Vector2d>,
    query_points2d: Vec<Vector2d>,
    world_camera_idxs: Vec<usize>,
    world_cams_from_world: Vec<Rigid3d>,
    world_cameras: Vec<Camera>,
    query_camera: Camera,
}

fn build_structure_less_absolute_pose_problem(
    num_world_cams: i32,
) -> StructureLessAbsolutePoseProblem {
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: num_world_cams + 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 1,
        num_points3d: 100,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();

    let reg_image_ids = reconstruction.reg_image_ids().unwrap();
    let query_image_id = reg_image_ids[0];
    let query_image = reconstruction.image(query_image_id).unwrap();

    let mut problem = StructureLessAbsolutePoseProblem {
        gt_cam_from_world: image_cam_from_world(&reconstruction, query_image),
        world_points2d: Vec::new(),
        query_points2d: Vec::new(),
        world_camera_idxs: Vec::new(),
        world_cams_from_world: Vec::new(),
        world_cameras: Vec::new(),
        query_camera: image_camera(&reconstruction, query_image),
    };

    // Build mapping of world cameras
    let mut world_image_id_to_camera_idx: HashMap<ImageId, usize> = HashMap::new();
    let mut world_obs: HashMap<Point3DId, Vec<(ImageId, Point2DIdx)>> = HashMap::new();

    for &world_image_id in &reg_image_ids {
        if world_image_id == query_image_id {
            continue;
        }

        let world_image = reconstruction.image(world_image_id).unwrap();

        if let Entry::Vacant(entry) = world_image_id_to_camera_idx.entry(world_image.image_id()) {
            entry.insert(problem.world_cameras.len());
            problem
                .world_cameras
                .push(image_camera(&reconstruction, world_image));
            problem
                .world_cams_from_world
                .push(image_cam_from_world(&reconstruction, world_image));
        }

        for point2d_idx in 0..world_image.num_points2d() {
            let point2d = world_image.point2d(point2d_idx);
            if point2d.has_point3d() {
                world_obs
                    .entry(point2d.point3d_id)
                    .or_default()
                    .push((world_image.image_id(), point2d_idx));
            }
        }
    }

    for point2d_idx in 0..query_image.num_points2d() {
        let query_point2d = query_image.point2d(point2d_idx);
        if !query_point2d.has_point3d() {
            continue;
        }

        let Some(observations) = world_obs.get(&query_point2d.point3d_id) else {
            continue;
        };

        for &(world_image_id, world_point2d_idx) in observations {
            let world_image = reconstruction.image(world_image_id).unwrap();
            let world_point2d = world_image.point2d(world_point2d_idx);
            problem.world_points2d.push(world_point2d.xy);
            problem.query_points2d.push(query_point2d.xy);
            problem
                .world_camera_idxs
                .push(world_image_id_to_camera_idx[&world_image.image_id()]);
        }
    }

    problem
}

#[test]
fn estimate_structure_less_absolute_pose_nominal() {
    set_prng_seed(0);
    let problem = build_structure_less_absolute_pose_problem(5);

    let options = StructureLessAbsolutePoseEstimationOptions::default();
    let estimate = estimate_structure_less_absolute_pose(
        &options,
        &problem.query_points2d,
        &problem.world_points2d,
        &problem.world_camera_idxs,
        &problem.world_cams_from_world,
        &problem.world_cameras,
        &problem.query_camera,
    )
    .unwrap()
    .expect("EstimateStructureLessAbsolutePose succeeds");
    assert_eq!(estimate.num_inliers, problem.world_points2d.len());
    assert_eq!(estimate.inlier_mask.len(), problem.world_points2d.len());
    assert!(
        rigid3d_near(&estimate.pose, &problem.gt_cam_from_world, 1e-6, 1e-6),
        "{:?} vs {:?}",
        estimate.pose,
        problem.gt_cam_from_world
    );
}

#[test]
fn estimate_structure_less_absolute_pose_with_outliers() {
    set_prng_seed(0);
    let mut problem = build_structure_less_absolute_pose_problem(10);

    // Add outliers by perturbing some query observations.
    const OUTLIER_RATIO: f64 = 0.3;
    let num_outliers = (OUTLIER_RATIO * problem.query_points2d.len() as f64) as usize;
    let shuffled_idxs = shuffled_indices(problem.query_points2d.len());
    for &shuffled_idx in &shuffled_idxs[..num_outliers] {
        problem.query_points2d[shuffled_idx] += Vector2d::new(1000.0, 1000.0);
    }

    let mut options = StructureLessAbsolutePoseEstimationOptions::default();
    options.ransac_options.max_error = 1.0; // pixels
    let estimate = estimate_structure_less_absolute_pose(
        &options,
        &problem.query_points2d,
        &problem.world_points2d,
        &problem.world_camera_idxs,
        &problem.world_cams_from_world,
        &problem.world_cameras,
        &problem.query_camera,
    )
    .unwrap()
    .expect("EstimateStructureLessAbsolutePose succeeds");
    assert!(
        estimate.num_inliers as f64
            > problem.world_points2d.len() as f64 * (1.0 - OUTLIER_RATIO) * 0.9,
        "{} inliers of {}",
        estimate.num_inliers,
        problem.world_points2d.len()
    );
    assert!(
        rigid3d_near(&estimate.pose, &problem.gt_cam_from_world, 5e-3, 5e-3),
        "{:?} vs {:?}",
        estimate.pose,
        problem.gt_cam_from_world
    );
}

#[test]
fn estimate_structure_less_absolute_pose_panoramic_world_cameras() {
    set_prng_seed(0);
    let problem = build_structure_less_absolute_pose_problem(1);

    let options = StructureLessAbsolutePoseEstimationOptions::default();
    assert!(estimate_structure_less_absolute_pose(
        &options,
        &problem.query_points2d,
        &problem.world_points2d,
        &problem.world_camera_idxs,
        &problem.world_cams_from_world,
        &problem.world_cameras,
        &problem.query_camera,
    )
    .unwrap()
    .is_none());
}
