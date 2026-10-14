// Port of COLMAP's src/colmap/estimators/rotation_averaging_test.cc (1:1, all 15 tests) over
// colmap_rust::estimators::rotation_averaging. Tier C: the expectations compare
// gauge-invariant relative rotations with COLMAP's tolerances. Each test seeds the PRNG with
// 0 first (COLMAP's tests share one lazily seeded thread PRNG across a test binary).

use std::collections::{BTreeMap, HashSet};

use colmap_rust::estimators::rotation_averaging::{
    initialize_rig_rotations_from_images, run_rotation_averaging, RotationAveragingReweighting,
    RotationEstimator, RotationEstimatorOptions,
};
use colmap_rust::geometry::pose_prior::PosePrior;
use colmap_rust::geometry::rigid3::Rigid3d;
use colmap_rust::linalg::{AngleAxisd, Quaterniond, Vector3d};
use colmap_rust::math::random::{random_gaussian, random_uniform_real, set_prng_seed};
use colmap_rust::math::utils::{deg_to_rad, rad_to_deg};
use colmap_rust::scene::database::Database;
use colmap_rust::scene::database_cache::{DatabaseCache, DatabaseCacheOptions};
use colmap_rust::scene::image::Image;
use colmap_rust::scene::pose_graph::PoseGraph;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset_with_database, SyntheticDatasetOptions};
use colmap_rust::scene::synthetic_noise::{synthesize_noise_with_database, SyntheticNoiseOptions};
use colmap_rust::util::types::{pair_id_to_image_pair, FrameId, ImageId, RigId, SensorId};

struct TestData {
    gt_reconstruction: Reconstruction,
    reconstruction: Reconstruction,
    pose_graph: PoseGraph,
    pose_priors: Vec<PosePrior>,
}

fn load_reconstruction_and_pose_graph(database: &Database) -> (Reconstruction, PoseGraph) {
    let mut database_cache = DatabaseCache::new();
    database_cache
        .load(database, &DatabaseCacheOptions::default())
        .unwrap();
    let mut reconstruction = Reconstruction::new();
    reconstruction.load(&database_cache).unwrap();
    let mut pose_graph = PoseGraph::new();
    pose_graph
        .load(database_cache.correspondence_graph())
        .unwrap();
    (reconstruction, pose_graph)
}

fn create_test_data(
    dataset_options: &SyntheticDatasetOptions,
    noise_options: Option<&SyntheticNoiseOptions>,
) -> TestData {
    let database = Database::new();
    let mut gt_reconstruction = Reconstruction::new();
    synthesize_dataset_with_database(dataset_options, &mut gt_reconstruction, &database).unwrap();
    if let Some(noise_options) = noise_options {
        synthesize_noise_with_database(noise_options, &mut gt_reconstruction, &database).unwrap();
    }
    let (reconstruction, pose_graph) = load_reconstruction_and_pose_graph(&database);
    let pose_priors = database.read_all_pose_priors().unwrap();
    TestData {
        gt_reconstruction,
        reconstruction,
        pose_graph,
        pose_priors,
    }
}

fn create_ra_test_options(use_gravity: bool) -> RotationEstimatorOptions {
    RotationEstimatorOptions {
        skip_initialization: false,
        use_gravity,
        use_stratified: true,
        ..RotationEstimatorOptions::default()
    }
}

fn cam_from_world_rotation(reconstruction: &Reconstruction, image_id: ImageId) -> Quaterniond {
    let image = reconstruction.image(image_id).unwrap();
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    image.cam_from_world(frame, rig).unwrap().rotation
}

fn image_has_pose(reconstruction: &Reconstruction, image: &Image) -> bool {
    image.has_pose(reconstruction.frames().get(&image.frame_id()))
}

fn expect_equal_rotations(
    gt: &Reconstruction,
    computed: &Reconstruction,
    max_rotation_error_deg: f64,
) {
    let max_rotation_error_rad = deg_to_rad(max_rotation_error_deg);
    let reg_image_ids = gt.reg_image_ids().unwrap();
    for i in 0..reg_image_ids.len() {
        let image_id1 = reg_image_ids[i];
        for &image_id2 in &reg_image_ids[..i] {
            let cam2_from_cam1 = cam_from_world_rotation(computed, image_id2)
                * cam_from_world_rotation(computed, image_id1).inverse();
            let cam2_from_cam1_gt = cam_from_world_rotation(gt, image_id2)
                * cam_from_world_rotation(gt, image_id1).inverse();
            let distance = cam2_from_cam1.angular_distance(cam2_from_cam1_gt);
            assert!(
                distance <= max_rotation_error_rad,
                "images ({image_id1}, {image_id2}): {distance} > {max_rotation_error_rad}"
            );
        }
    }
}

// Gauge-invariant mean rotation error (in degrees) over all registered image pairs,
// comparing the computed relative rotations against the ground truth.
fn mean_relative_rotation_error_deg(gt: &Reconstruction, computed: &Reconstruction) -> f64 {
    let reg_image_ids = gt.reg_image_ids().unwrap();
    let mut total_error_rad = 0.0;
    let mut count = 0;
    for i in 0..reg_image_ids.len() {
        for j in 0..i {
            let rel = cam_from_world_rotation(computed, reg_image_ids[j])
                * cam_from_world_rotation(computed, reg_image_ids[i]).inverse();
            let rel_gt = cam_from_world_rotation(gt, reg_image_ids[j])
                * cam_from_world_rotation(gt, reg_image_ids[i]).inverse();
            total_error_rad += rel.angular_distance(rel_gt);
            count += 1;
        }
    }
    rad_to_deg(total_error_rad / count as f64)
}

fn reset_sensors_from_rig(reconstruction: &mut Reconstruction) {
    let mut known: Vec<(RigId, SensorId)> = Vec::new();
    for (&rig_id, rig) in reconstruction.rigs() {
        for (&sensor_id, sensor) in rig.non_ref_sensors() {
            if sensor.is_some() {
                known.push((rig_id, sensor_id));
            }
        }
    }
    for (rig_id, sensor_id) in known {
        reconstruction
            .rig_mut(rig_id)
            .unwrap()
            .reset_sensor_from_rig(sensor_id)
            .unwrap();
    }
}

fn run_and_verify_rotation_averaging(
    data: &TestData,
    use_gravity_values: &[bool],
    max_rotation_error_deg: f64,
) {
    for &use_gravity in use_gravity_values {
        let mut reconstruction_copy = data.reconstruction.clone();
        let mut pose_graph_copy = data.pose_graph.clone();
        run_rotation_averaging(
            &create_ra_test_options(use_gravity),
            &mut pose_graph_copy,
            &mut reconstruction_copy,
            &data.pose_priors,
        )
        .unwrap();
        expect_equal_rotations(
            &data.gt_reconstruction,
            &reconstruction_copy,
            max_rotation_error_deg,
        );
    }
}

fn gt_cams_from_world(data: &TestData) -> BTreeMap<ImageId, Rigid3d> {
    let gt = &data.gt_reconstruction;
    let mut cams_from_world = BTreeMap::new();
    for (&image_id, image) in gt.images() {
        if image_has_pose(gt, image) {
            let frame = gt.frame(image.frame_id()).unwrap();
            let rig = gt.rig(frame.rig_id()).unwrap();
            cams_from_world.insert(image_id, image.cam_from_world(frame, rig).unwrap());
        }
    }
    cams_from_world
}

fn sensor_from_rig_snapshot(
    reconstruction: &Reconstruction,
) -> BTreeMap<(RigId, SensorId), Rigid3d> {
    let mut snapshot = BTreeMap::new();
    for (&rig_id, rig) in reconstruction.rigs() {
        for (&sensor_id, sensor_from_rig) in rig.non_ref_sensors() {
            snapshot.insert(
                (rig_id, sensor_id),
                sensor_from_rig.expect("calibrated rig"),
            );
        }
    }
    snapshot
}

fn expect_rig_matches_snapshot(
    reconstruction: &Reconstruction,
    snapshot: &BTreeMap<(RigId, SensorId), Rigid3d>,
) {
    for (&rig_id, rig) in reconstruction.rigs() {
        for (&sensor_id, sensor_from_rig_after) in rig.non_ref_sensors() {
            let after = sensor_from_rig_after
                .unwrap_or_else(|| panic!("rig_id={rig_id}, sensor_id={}", sensor_id.id));
            assert_eq!(
                after,
                snapshot[&(rig_id, sensor_id)],
                "rig_id={rig_id}, sensor_id={}",
                sensor_id.id
            );
        }
    }
}

fn single_rig_options(
    num_cameras_per_rig: i32,
    num_frames_per_rig: i32,
) -> SyntheticDatasetOptions {
    SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig,
        num_frames_per_rig,
        num_points3d: 50,
        sensor_from_rig_rotation_stddev: 20.0,
        two_view_geometry_has_relative_pose: true,
        ..SyntheticDatasetOptions::default()
    }
}

fn noisy_options(num_cameras_per_rig: i32) -> (SyntheticDatasetOptions, SyntheticNoiseOptions) {
    let dataset_options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig,
        num_frames_per_rig: 7,
        num_points3d: 100,
        inlier_match_ratio: 0.6,
        prior_gravity: true,
        two_view_geometry_has_relative_pose: true,
        ..SyntheticDatasetOptions::default()
    };
    let noise_options = SyntheticNoiseOptions {
        point2d_stddev: 1.0,
        prior_gravity_stddev: 3e-1,
        ..SyntheticNoiseOptions::default()
    };
    (dataset_options, noise_options)
}

#[test]
fn rotation_averaging_without_noise() {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        prior_gravity: true,
        ..single_rig_options(1, 5)
    };
    let data = create_test_data(&options, None);
    run_and_verify_rotation_averaging(&data, &[true, false], 1e-2);
}

#[test]
fn rotation_averaging_weighted_noise_free_matches_invariant() {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        prior_gravity: true,
        ..single_rig_options(1, 5)
    };
    let mut data = create_test_data(&options, None);

    // Assign varying positive match counts so the edge weighting is non-trivial.
    // (C++: a counter from 1, post-incremented per edge.)
    for (counter, edge) in (1..).zip(data.pose_graph.edges_mut().values_mut()) {
        edge.num_matches = 10 * (counter % 7) + 1;
    }

    for use_gravity in [true, false] {
        // Unweighted baseline.
        let mut recon_unweighted = data.reconstruction.clone();
        let mut pose_graph_unweighted = data.pose_graph.clone();
        let options_unweighted = RotationEstimatorOptions {
            reweighting: RotationAveragingReweighting::Uniform,
            ..create_ra_test_options(use_gravity)
        };
        run_rotation_averaging(
            &options_unweighted,
            &mut pose_graph_unweighted,
            &mut recon_unweighted,
            &data.pose_priors,
        )
        .unwrap();

        // Weighted.
        let mut recon_weighted = data.reconstruction.clone();
        let mut pose_graph_weighted = data.pose_graph.clone();
        let options_weighted = RotationEstimatorOptions {
            reweighting: RotationAveragingReweighting::InlierMatchCount,
            ..create_ra_test_options(use_gravity)
        };
        run_rotation_averaging(
            &options_weighted,
            &mut pose_graph_weighted,
            &mut recon_weighted,
            &data.pose_priors,
        )
        .unwrap();

        // The weighted solution recovers the ground truth and, for a noise-free system, is
        // identical to the unweighted solution.
        expect_equal_rotations(&data.gt_reconstruction, &recon_weighted, 1e-2);
        expect_equal_rotations(&recon_unweighted, &recon_weighted, 1e-2);
    }
}

#[test]
fn rotation_averaging_weighted_reduces_error_with_noisy_low_match_edges() {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 15,
        num_points3d: 150,
        prior_gravity: false,
        two_view_geometry_has_relative_pose: true,
        ..SyntheticDatasetOptions::default()
    };
    let mut data = create_test_data(&options, None);

    // Inject controlled rotation noise into each relative pose and halve the match count of
    // the noisy edges. A uniform baseline match count ensures the only weight difference
    // between runs is the halving of the noisy edges.
    const NOISE_THRESHOLD_DEG: f64 = 5.0;
    const BASELINE_MATCHES: i32 = 100;
    for edge in data.pose_graph.edges_mut().values_mut() {
        // Range kept just above the 5 deg threshold: noisy edges (5-8 deg) still carry
        // meaningful IRLS weight (sigma = 5 deg), so the 2x down-weighting has real
        // leverage; sub-5 deg edges anchor the solution.
        let noise_deg = random_uniform_real(0.0, 8.0);
        // Seeded isotropic axis (RandomEigenVectord<3>() is NOT seeded by SetPRNGSeed).
        let x = random_gaussian(0.0, 1.0);
        let y = random_gaussian(0.0, 1.0);
        let z = random_gaussian(0.0, 1.0);
        let axis = Vector3d::new(x, y, z).normalized();
        let perturb = Quaterniond::from_angle_axis(AngleAxisd {
            angle: deg_to_rad(noise_deg),
            axis,
        });
        edge.cam2_from_cam1.rotation = perturb * edge.cam2_from_cam1.rotation;
        edge.num_matches = BASELINE_MATCHES;
        if noise_deg > NOISE_THRESHOLD_DEG {
            edge.num_matches /= 2; // Down-weight noisy edges.
        }
    }

    let run = |reweighting: RotationAveragingReweighting| {
        let mut reconstruction = data.reconstruction.clone();
        let mut pose_graph = data.pose_graph.clone();
        let options = RotationEstimatorOptions {
            reweighting,
            random_seed: 0,              // Deterministic solve.
            max_rotation_error_deg: 0.0, // Disable post-solve edge filtering so only the
            // solver reweighting differs.
            ..create_ra_test_options(false)
        };
        run_rotation_averaging(
            &options,
            &mut pose_graph,
            &mut reconstruction,
            &data.pose_priors,
        )
        .unwrap();
        mean_relative_rotation_error_deg(&data.gt_reconstruction, &reconstruction)
    };

    let error_unweighted = run(RotationAveragingReweighting::Uniform);
    let error_weighted = run(RotationAveragingReweighting::InlierMatchCount);

    assert!(
        error_weighted < error_unweighted,
        "{error_weighted} >= {error_unweighted}"
    );
}

#[test]
fn rotation_averaging_without_noise_with_non_trivial_known_rig() {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        prior_gravity: true,
        ..single_rig_options(2, 4)
    };
    let data = create_test_data(&options, None);
    run_and_verify_rotation_averaging(&data, &[true, false], 1e-2);
}

#[test]
fn rotation_averaging_without_noise_with_non_trivial_unknown_rig() {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        prior_gravity: true,
        ..single_rig_options(2, 4)
    };
    let mut data = create_test_data(&options, None);

    reset_sensors_from_rig(&mut data.reconstruction);

    // For unknown rigs, it is not supported to use gravity.
    run_and_verify_rotation_averaging(&data, &[false], 1e-2);
}

#[test]
fn rotation_averaging_with_noise_and_outliers() {
    set_prng_seed(0);
    let (dataset_options, noise_options) = noisy_options(1);
    let data = create_test_data(&dataset_options, Some(&noise_options));
    run_and_verify_rotation_averaging(&data, &[true, false], 3.0);
}

#[test]
fn rotation_averaging_with_noise_and_outliers_with_non_trivial_known_rigs() {
    set_prng_seed(0);
    let (dataset_options, noise_options) = noisy_options(2);
    let data = create_test_data(&dataset_options, Some(&noise_options));
    run_and_verify_rotation_averaging(&data, &[true, false], 2.0);
}

#[test]
fn rotation_averaging_deterministic_random_seed() {
    set_prng_seed(0);
    let data = create_test_data(&single_rig_options(1, 5), None);

    let options = RotationEstimatorOptions {
        random_seed: 42,
        ..create_ra_test_options(false)
    };

    // Run twice with the same seed and verify identical results.
    let mut reconstruction1 = data.reconstruction.clone();
    let mut pose_graph1 = data.pose_graph.clone();
    assert!(run_rotation_averaging(
        &options,
        &mut pose_graph1,
        &mut reconstruction1,
        &data.pose_priors
    )
    .unwrap());

    let mut reconstruction2 = data.reconstruction.clone();
    let mut pose_graph2 = data.pose_graph.clone();
    assert!(run_rotation_averaging(
        &options,
        &mut pose_graph2,
        &mut reconstruction2,
        &data.pose_priors
    )
    .unwrap());

    for image_id in reconstruction1.reg_image_ids().unwrap() {
        let q1 = cam_from_world_rotation(&reconstruction1, image_id);
        let q2 = cam_from_world_rotation(&reconstruction2, image_id);
        // In the presence of optimizations like FMA, q.angularDistance(q) can be near-zero
        // instead of zero, so check equality explicitly instead of with
        // expect_equal_rotations.
        assert_eq!([q1.x, q1.y, q1.z, q1.w], [q2.x, q2.y, q2.z, q2.w]);
    }
}

#[test]
fn rotation_averaging_ridge_regularization_does_not_bias_solution() {
    set_prng_seed(0);
    // Use a noisy multi-rig setup to make the solution non-trivial and the regularization's
    // effect non-degenerate.
    let (dataset_options, noise_options) = noisy_options(1);
    let data = create_test_data(&dataset_options, Some(&noise_options));

    let mut options = RotationEstimatorOptions {
        random_seed: 42,
        ..create_ra_test_options(true)
    };

    // Run once with no regularization.
    let mut reconstruction_no_ridge = data.reconstruction.clone();
    let mut pose_graph_no_ridge = data.pose_graph.clone();
    options.ridge_regularization = 0.0;
    assert!(run_rotation_averaging(
        &options,
        &mut pose_graph_no_ridge,
        &mut reconstruction_no_ridge,
        &data.pose_priors
    )
    .unwrap());

    // Run again with the same default ridge that the global mapper uses. The option must flow
    // through L1 and IRLS without biasing the solution.
    let mut reconstruction_ridge = data.reconstruction.clone();
    let mut pose_graph_ridge = data.pose_graph.clone();
    options.ridge_regularization = 1e-9;
    assert!(run_rotation_averaging(
        &options,
        &mut pose_graph_ridge,
        &mut reconstruction_ridge,
        &data.pose_priors
    )
    .unwrap());

    // The two solutions should be effectively identical since 1e-9 is far below any
    // meaningful residual scale in the optimization.
    expect_equal_rotations(&reconstruction_no_ridge, &reconstruction_ridge, 1e-12);
}

#[test]
fn rotation_averaging_empty_pose_graph() {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 3,
        num_points3d: 20,
        two_view_geometry_has_relative_pose: true,
        ..SyntheticDatasetOptions::default()
    };
    let mut data = create_test_data(&options, None);

    // Invalidate all edges so connected components are empty.
    for edge in data.pose_graph.edges_mut().values_mut() {
        edge.valid = false;
    }

    assert!(!run_rotation_averaging(
        &create_ra_test_options(false),
        &mut data.pose_graph,
        &mut data.reconstruction,
        &data.pose_priors
    )
    .unwrap());
}

#[test]
fn rotation_averaging_multi_image_rig_frame_deregister_does_not_crash_on_second_visit() {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 2,
        num_frames_per_rig: 4,
        num_points3d: 50,
        two_view_geometry_has_relative_pose: true,
        ..SyntheticDatasetOptions::default()
    };
    let mut data = create_test_data(&options, None);

    let frame_ids: Vec<FrameId> = data.reconstruction.frames().keys().copied().collect();
    assert_eq!(frame_ids.len(), 4);
    let isolated_frame_id = *frame_ids.last().unwrap();

    // 1. Collect every image_id that belongs to the isolated frame.
    let isolated_image_ids: HashSet<ImageId> = data
        .reconstruction
        .frame(isolated_frame_id)
        .unwrap()
        .image_ids()
        .map(|data_id| data_id.id as ImageId)
        .collect();

    // 2. Strip every pose-graph edge touching the isolated frame. After this the frame is
    //    unreachable from any other frame in the pose-graph CC.
    let mut edges_to_remove: Vec<(ImageId, ImageId)> = Vec::new();
    for &pair_id in data.pose_graph.edges().keys() {
        let (id1, id2) = pair_id_to_image_pair(pair_id).unwrap();
        if isolated_image_ids.contains(&id1) || isolated_image_ids.contains(&id2) {
            edges_to_remove.push((id1, id2));
        }
    }
    assert!(!edges_to_remove.is_empty());
    for (id1, id2) in edges_to_remove {
        data.pose_graph.delete_edge(id1, id2).unwrap();
    }

    // 3. Pre-register the isolated frame with its GT pose. This puts it into reg_frame_ids
    //    even though no edges touch it.
    let gt_frame = data.gt_reconstruction.frame(isolated_frame_id).unwrap();
    assert!(gt_frame.has_pose());
    let gt_rig_from_world = gt_frame.rig_from_world().unwrap();
    data.reconstruction
        .frame_mut(isolated_frame_id)
        .unwrap()
        .set_rig_from_world(gt_rig_from_world);
    data.reconstruction
        .register_frame(isolated_frame_id)
        .unwrap();

    let ra_options = RotationEstimatorOptions {
        max_rotation_error_deg: 1.0,
        ..create_ra_test_options(false)
    };
    assert!(run_rotation_averaging(
        &ra_options,
        &mut data.pose_graph,
        &mut data.reconstruction,
        &data.pose_priors
    )
    .unwrap());

    // Post-condition: the isolated frame was deregistered cleanly. The other frames remain
    // registered with poses recovered by RA.
    assert!(!data
        .reconstruction
        .frame(isolated_frame_id)
        .unwrap()
        .has_pose());
    for &frame_id in &frame_ids[..frame_ids.len() - 1] {
        assert!(
            data.reconstruction.frame(frame_id).unwrap().has_pose(),
            "Frame {frame_id} should remain registered after deregistration of the isolated \
             frame."
        );
    }
}

#[test]
fn rotation_averaging_gravity_with_unknown_rig_sensors_returns_false() {
    set_prng_seed(0);
    let options = SyntheticDatasetOptions {
        prior_gravity: true,
        ..single_rig_options(2, 4)
    };
    let mut data = create_test_data(&options, None);

    reset_sensors_from_rig(&mut data.reconstruction);

    // With gravity enabled and unknown rig sensors, EstimateRotations should fail inside
    // RunRotationAveraging because AllSensorsFromRigKnown returns false. However,
    // RunRotationAveraging takes the HasUnknownCamsFromRig path which creates an expanded
    // reconstruction (singleton rigs) that avoids the AllSensorsFromRigKnown check. To
    // directly hit the AllSensorsFromRigKnown check, we use RotationEstimator directly.
    let ra_options = create_ra_test_options(true);
    let active_image_ids: HashSet<ImageId> = data.reconstruction.images().keys().copied().collect();

    let estimator = RotationEstimator::new(&ra_options);
    assert!(!estimator
        .estimate_rotations(
            &data.pose_graph,
            &data.pose_priors,
            &active_image_ids,
            &mut data.reconstruction
        )
        .unwrap());
}

// Covers InitializeRigRotationsFromImages standalone with a multi-camera rig to exercise
// cam_from_rig estimation and rig_from_world averaging.
#[test]
fn rotation_averaging_initialize_sensor_from_rig_using_cams_from_world() {
    set_prng_seed(0);
    let mut data = create_test_data(&single_rig_options(2, 4), None);

    // Build cams_from_world from the ground truth.
    let cams_from_world = gt_cams_from_world(&data);

    reset_sensors_from_rig(&mut data.reconstruction);

    assert!(
        initialize_rig_rotations_from_images(&cams_from_world, &mut data.reconstruction, true)
            .unwrap()
    );

    for (&rig_id, rig) in data.reconstruction.rigs() {
        for (&sensor_id, sensor_from_rig) in rig.non_ref_sensors() {
            let gt_rotation = data
                .gt_reconstruction
                .rig(rig_id)
                .unwrap()
                .sensor_from_rig(sensor_id)
                .unwrap()
                .rotation;
            let distance = sensor_from_rig
                .unwrap()
                .rotation
                .angular_distance(gt_rotation);
            assert!(distance < 1e-6, "{distance}");
        }
    }
}

// When a sensor_from_rig is already fully calibrated (valid rotation AND translation),
// InitializeRigRotationsFromImages must preserve it rather than resetting the translation to
// NaN.
#[test]
fn rotation_averaging_initialize_sensor_from_rig_preserves_calibrated_rig() {
    set_prng_seed(0);
    let mut data = create_test_data(&single_rig_options(2, 4), None);

    let cams_from_world = gt_cams_from_world(&data);

    // Snapshot the (already-calibrated) rig BEFORE initialization.
    let snapshot = sensor_from_rig_snapshot(&data.reconstruction);
    assert!(!snapshot.is_empty());

    assert!(
        initialize_rig_rotations_from_images(&cams_from_world, &mut data.reconstruction, true)
            .unwrap()
    );

    expect_rig_matches_snapshot(&data.reconstruction, &snapshot);
}

#[test]
fn rotation_averaging_refine_sensor_from_rig_false_preserves_rig() {
    set_prng_seed(0);
    // A non-trivial multi-camera rig so both rotation AND translation are non-zero.
    let options = SyntheticDatasetOptions {
        prior_gravity: true,
        ..single_rig_options(2, 4)
    };
    let mut data = create_test_data(&options, None);

    // Snapshot the rig BEFORE RA so we can compare element-wise.
    let snapshot = sensor_from_rig_snapshot(&data.reconstruction);
    // Sanity check: at least one sensor should have a non-zero translation so the test would
    // actually catch the old "reset to zero" behaviour.
    assert!(!snapshot.is_empty());

    // Run RA with refine_sensor_from_rig=false.
    let ra_options = RotationEstimatorOptions {
        refine_sensor_from_rig: false,
        ..create_ra_test_options(true)
    };
    assert!(run_rotation_averaging(
        &ra_options,
        &mut data.pose_graph,
        &mut data.reconstruction,
        &data.pose_priors
    )
    .unwrap());

    // Every sensor_from_rig must match the snapshot exactly.
    expect_rig_matches_snapshot(&data.reconstruction, &snapshot);
}
