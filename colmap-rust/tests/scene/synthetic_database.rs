// Port of colmap/scene/synthetic_test.cc, part 2, 1:1 (same test names, values and checks):
// the SynthesizeDataset tests that write into a database
// (colmap_rust::scene::synthetic::synthesize_dataset_with_database over the in-memory
// Database). `Database::Open(kInMemorySqliteDatabasePath)` is `Database::new()`. COLMAP's
// gtest main seeds the PRNG with 0 before every test, so each test calls `set_prng_seed(0)`
// first.

use std::collections::{BTreeSet, HashMap};

use colmap_rust::geometry::triangulation::calculate_triangulation_angle;
use colmap_rust::linalg::Vector3d;
use colmap_rust::math::random::set_prng_seed;
use colmap_rust::math::random_eigen::random_eigen_vector3d;
use colmap_rust::math::union_find::UnionFind;
use colmap_rust::math::utils::deg_to_rad;
use colmap_rust::scene::database::Database;
use colmap_rust::scene::projection::calculate_squared_reprojection_error;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{
    synthesize_dataset_with_database, MatchConfig, SyntheticDatasetOptions,
};
use colmap_rust::util::types::{pair_id_to_image_pair, ImageId, SensorType};

use crate::eigen_matchers::eigen_matrix_near;
use crate::reconstruction_io::create_test_dir;

/// `EigenMatrixNear` on two vectors.
fn near(a: Vector3d, b: Vector3d, tol: f64) -> bool {
    eigen_matrix_near(&a.to_array(), &b.to_array(), tol)
}

fn image_pose(
    reconstruction: &Reconstruction,
    image_id: ImageId,
) -> colmap_rust::geometry::rigid3::Rigid3d {
    let image = reconstruction.image(image_id).unwrap();
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    image.cam_from_world(frame, rig).unwrap()
}

fn image_projection_center(reconstruction: &Reconstruction, image_id: ImageId) -> Vector3d {
    let image = reconstruction.image(image_id).unwrap();
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    image.projection_center(frame, rig).unwrap()
}

#[test]
fn synthesize_dataset_nominal() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 3,
        num_frames_per_rig: 3,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();

    let test_dir = create_test_dir();
    let sparse_path = test_dir.join("sparse");
    std::fs::create_dir_all(&sparse_path).unwrap();
    reconstruction.write(&sparse_path).unwrap();

    assert_eq!(database.num_rigs().unwrap(), options.num_rigs as usize);
    assert_eq!(reconstruction.num_rigs(), options.num_rigs as usize);
    for rig in reconstruction.rigs().values() {
        assert!(rig.num_sensors() >= options.num_cameras_per_rig as usize);
    }

    let num_cameras = (options.num_rigs * options.num_cameras_per_rig) as usize;
    assert_eq!(database.num_cameras().unwrap(), num_cameras);
    assert_eq!(reconstruction.num_cameras(), num_cameras);
    for (&camera_id, camera) in reconstruction.cameras() {
        assert_eq!(*camera, database.read_camera(camera_id).unwrap());
        assert_eq!(camera.model_id, options.camera_model_id);
    }

    let num_frames = (options.num_rigs * options.num_frames_per_rig) as usize;
    assert_eq!(database.num_frames().unwrap(), num_frames);
    assert_eq!(reconstruction.num_frames(), num_frames);
    for (&frame_id, frame) in reconstruction.frames() {
        let mut reconstruction_frame = frame.clone();
        assert!(reconstruction_frame.has_pose());
        reconstruction_frame.reset_pose();
        assert_eq!(reconstruction_frame, database.read_frame(frame_id).unwrap());
        assert_eq!(
            reconstruction_frame.num_data_ids(),
            reconstruction
                .rig(reconstruction_frame.rig_id())
                .unwrap()
                .num_sensors()
        );
    }

    let num_images =
        (options.num_rigs * options.num_cameras_per_rig * options.num_frames_per_rig) as usize;
    assert_eq!(database.num_images().unwrap(), num_images);
    assert_eq!(reconstruction.num_images(), num_images);
    assert_eq!(reconstruction.num_reg_frames(), num_frames);
    let mut image_names = BTreeSet::new();
    for (&image_id, image) in reconstruction.images() {
        assert_eq!(image.name(), database.read_image(image_id).unwrap().name());
        assert!(image.name().ends_with(&options.image_extension));
        image_names.insert(image.name().to_owned());
        let num_points2d = image.num_points2d() as usize;
        assert_eq!(
            num_points2d,
            database.read_keypoints(image_id).unwrap().len()
        );
        assert_eq!(
            num_points2d,
            database.read_descriptors(image_id).unwrap().data.rows()
        );
        assert_eq!(
            database.read_descriptors(image_id).unwrap().data.cols(),
            128
        );
        assert_eq!(
            num_points2d,
            (options.num_points3d + options.num_points2d_without_point3d) as usize
        );
        assert_eq!(image.num_points3d() as i32, options.num_points3d);
    }
    assert_eq!(image_names.len(), reconstruction.num_images());

    let num_image_pairs = reconstruction.num_images() * (reconstruction.num_images() - 1) / 2;
    assert_eq!(
        database.num_verified_image_pairs().unwrap(),
        num_image_pairs
    );
    assert_eq!(
        database.num_inlier_matches().unwrap(),
        num_image_pairs * options.num_points3d as usize
    );

    assert!(reconstruction.compute_mean_reprojection_error().abs() <= 1e-6);
    assert!(
        reconstruction
            .compute_centroid(0.0, 1.0, false)
            .unwrap()
            .norm()
            .abs()
            <= 0.2
    );
    assert!(
        (reconstruction.compute_mean_track_length().unwrap() - reconstruction.num_images() as f64)
            .abs()
            <= 0.1
    );
    assert_eq!(
        reconstruction.compute_num_observations().unwrap(),
        reconstruction.num_images() * options.num_points3d as usize
    );

    assert_eq!(reconstruction.num_points3d(), options.num_points3d as usize);

    // All observations should be perfect and have sufficient triangulation
    // angle. No points or observations should be filtered.
    let k_max_reproj_error = 1e-3;
    let k_min_tri_angle_deg = 0.4;
    let mut proj_centers: HashMap<ImageId, Vector3d> = HashMap::new();
    for point3d_id in reconstruction.point3d_ids() {
        let point3d = reconstruction.point3d(point3d_id).unwrap();

        // Make sure all descriptors of the same 3D point have identical features.
        let element0 = point3d.track.element(0);
        let desc0 = database.read_descriptors(element0.image_id).unwrap();
        let descriptors = desc0.data.row(element0.point2d_idx as usize).to_vec();

        let mut max_tri_angle: f64 = 0.0;
        for i1 in 0..point3d.track.length() {
            let track_el = point3d.track.element(i1);
            let image_id1 = track_el.image_id;
            let image1 = reconstruction.image(image_id1).unwrap();
            let camera1 = reconstruction.camera(image1.camera_id()).unwrap();
            let point2d = image1.point2d(track_el.point2d_idx);
            let squared_reproj_error = calculate_squared_reprojection_error(
                point2d.xy,
                point3d.xyz,
                &image_pose(&reconstruction, image_id1),
                camera1,
            );
            assert!(squared_reproj_error <= k_max_reproj_error * k_max_reproj_error);
            let desc1 = database.read_descriptors(track_el.image_id).unwrap();
            assert_eq!(
                descriptors.as_slice(),
                desc1.data.row(track_el.point2d_idx as usize)
            );

            let proj_center1 = *proj_centers
                .entry(image_id1)
                .or_insert_with(|| image_projection_center(&reconstruction, image_id1));

            for i2 in 0..i1 {
                let image_id2 = point3d.track.element(i2).image_id;
                let proj_center2 = proj_centers[&image_id2];
                max_tri_angle = max_tri_angle.max(calculate_triangulation_angle(
                    proj_center1,
                    proj_center2,
                    point3d.xyz,
                ));
            }
        }

        assert!(max_tri_angle >= deg_to_rad(k_min_tri_angle_deg));
    }
}

#[test]
fn synthesize_dataset_multiple_times() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 3,
        num_frames_per_rig: 3,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();

    assert_eq!(database.num_rigs().unwrap(), 2 * options.num_rigs as usize);
    assert_eq!(reconstruction.num_rigs(), database.num_rigs().unwrap());

    assert_eq!(
        database.num_cameras().unwrap(),
        (2 * options.num_rigs * options.num_cameras_per_rig) as usize
    );
    assert_eq!(
        reconstruction.num_cameras(),
        database.num_cameras().unwrap()
    );

    assert_eq!(
        database.num_frames().unwrap(),
        (2 * options.num_rigs * options.num_frames_per_rig) as usize
    );
    assert_eq!(database.num_frames().unwrap(), reconstruction.num_frames());

    assert_eq!(
        database.num_images().unwrap(),
        (2 * options.num_rigs * options.num_cameras_per_rig * options.num_frames_per_rig) as usize
    );
    assert_eq!(database.num_images().unwrap(), reconstruction.num_images());

    let num_image_pairs = reconstruction.num_images() * (reconstruction.num_images() - 1) / 2;
    assert_eq!(
        database.num_verified_image_pairs().unwrap(),
        num_image_pairs
    );

    assert_eq!(
        reconstruction.num_points3d(),
        2 * options.num_points3d as usize
    );
}

#[test]
fn synthesize_dataset_with_priors() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        prior_position: true,
        prior_gravity: true,
        prior_gravity_in_world: random_eigen_vector3d().normalized(),
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();

    let pose_priors = database.read_all_pose_priors().unwrap();
    let mut image_to_prior = HashMap::new();
    for pose_prior in &pose_priors {
        assert_eq!(
            pose_prior.corr_data_id.sensor_id.sensor_type,
            SensorType::Camera
        );
        image_to_prior.insert(pose_prior.corr_data_id.id, pose_prior);
    }

    for &image_id in reconstruction.images().keys() {
        assert!(image_to_prior.contains_key(&u64::from(image_id)));
        let pose_prior = image_to_prior[&u64::from(image_id)];
        assert!(near(
            image_projection_center(&reconstruction, image_id),
            pose_prior.position,
            1e-9
        ));
        assert!(near(
            image_pose(&reconstruction, image_id).rotation * options.prior_gravity_in_world,
            pose_prior.gravity,
            1e-9
        ));
    }
}

#[test]
fn synthesize_dataset_multi_reconstruction() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction1 = Reconstruction::new();
    let mut reconstruction2 = Reconstruction::new();
    let options = SyntheticDatasetOptions::default();
    synthesize_dataset_with_database(&options, &mut reconstruction1, &database).unwrap();
    synthesize_dataset_with_database(&options, &mut reconstruction2, &database).unwrap();

    let num_cameras = (options.num_rigs * options.num_cameras_per_rig) as usize;
    assert_eq!(database.num_cameras().unwrap(), 2 * num_cameras);
    assert_eq!(reconstruction1.num_cameras(), num_cameras);
    assert_eq!(reconstruction1.num_cameras(), num_cameras);
    let num_images = num_cameras * options.num_frames_per_rig as usize;
    assert_eq!(database.num_images().unwrap(), 2 * num_images);
    assert_eq!(reconstruction1.num_images(), num_images);
    assert_eq!(reconstruction2.num_images(), num_images);
    assert_eq!(reconstruction1.num_reg_frames(), num_images);
    assert_eq!(reconstruction2.num_reg_frames(), num_images);
    let num_image_pairs = num_images * (num_images - 1) / 2;
    assert_eq!(
        database.num_verified_image_pairs().unwrap(),
        2 * num_image_pairs
    );
    assert_eq!(
        database.num_inlier_matches().unwrap(),
        2 * num_image_pairs * options.num_points3d as usize
    );
}

fn num_images_of(options: &SyntheticDatasetOptions) -> usize {
    (options.num_rigs * options.num_cameras_per_rig * options.num_frames_per_rig) as usize
}

#[test]
fn synthesize_dataset_exhaustive_matches() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        match_config: MatchConfig::Exhaustive,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();

    let num_images = num_images_of(&options);
    let num_image_pairs = num_images * (num_images - 1) / 2;
    assert_eq!(database.num_matched_image_pairs().unwrap(), num_image_pairs);
    assert_eq!(
        database.num_verified_image_pairs().unwrap(),
        num_image_pairs
    );
    assert_eq!(
        database.num_inlier_matches().unwrap(),
        num_image_pairs * options.num_points3d as usize
    );
}

#[test]
fn synthesize_dataset_chained_matches() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        match_config: MatchConfig::Chained,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();

    let num_images = num_images_of(&options);
    let num_image_pairs = num_images - 1;
    assert_eq!(database.num_matched_image_pairs().unwrap(), num_image_pairs);
    assert_eq!(
        database.num_verified_image_pairs().unwrap(),
        num_image_pairs
    );
    assert_eq!(
        database.num_inlier_matches().unwrap(),
        num_image_pairs * options.num_points3d as usize
    );
    for (pair_id, _) in database.read_all_matches().unwrap() {
        let (image_id1, image_id2) = pair_id_to_image_pair(pair_id).unwrap();
        assert_eq!(image_id1 + 1, image_id2);
    }
    for (pair_id, _) in database.read_two_view_geometries().unwrap() {
        let (image_id1, image_id2) = pair_id_to_image_pair(pair_id).unwrap();
        assert_eq!(image_id1 + 1, image_id2);
    }
}

#[test]
fn synthesize_dataset_sparse_matches_zero_sparsity() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        match_config: MatchConfig::Sparse,
        match_sparsity: 0.0,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
    let num_images = num_images_of(&options);
    let num_image_pairs = num_images * (num_images - 1) / 2;
    assert_eq!(database.num_matched_image_pairs().unwrap(), num_image_pairs);
    assert_eq!(
        database.num_verified_image_pairs().unwrap(),
        num_image_pairs
    );
}

#[test]
fn synthesize_dataset_sparse_matches_full_sparsity() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        match_config: MatchConfig::Sparse,
        match_sparsity: 1.0,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
    assert_eq!(database.num_matched_image_pairs().unwrap(), 0);
    assert_eq!(database.num_verified_image_pairs().unwrap(), 0);
}

#[test]
fn synthesize_dataset_sparse_matches_partial_sparsity() {
    set_prng_seed(0);
    let database = Database::new();
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 1,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 8,
        match_config: MatchConfig::Sparse,
        match_sparsity: 0.5,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();

    let num_images = num_images_of(&options);
    let num_exhaustive_pairs = num_images * (num_images - 1) / 2;
    assert_eq!(
        database.num_matched_image_pairs().unwrap(),
        num_exhaustive_pairs / 2
    );

    // Verify graph connectivity using UnionFind.
    let mut uf: UnionFind<ImageId> = UnionFind::new();
    uf.reserve(num_images);
    for (pair_id, _) in database.read_all_matches().unwrap() {
        let (image_id1, image_id2) = pair_id_to_image_pair(pair_id).unwrap();
        uf.union(&image_id1, &image_id2);
    }
    let reg_image_ids = reconstruction.reg_image_ids().unwrap();
    let root = uf.find(&reg_image_ids[0]);
    for image_id in &reg_image_ids {
        assert_eq!(uf.find(image_id), root);
    }
}

// Rust-only: COLMAP writes pose priors through the `Database*` without checking it, so asking
// for priors without a database dereferences null; the port fails with an error
// (docs/CPP_DIVERGENCES.md entry 269).
#[test]
fn rust_only_synthesize_dataset_pose_priors_need_a_database() {
    set_prng_seed(0);
    for (prior_position, prior_gravity) in [(true, false), (false, true)] {
        let mut reconstruction = Reconstruction::new();
        let options = SyntheticDatasetOptions {
            prior_position,
            prior_gravity,
            ..SyntheticDatasetOptions::default()
        };
        let error =
            colmap_rust::scene::synthetic::synthesize_dataset(&options, &mut reconstruction)
                .unwrap_err();
        assert!(error.to_string().contains("pose priors need a database"));
    }
}

// Rust-only: the database path draws the same PRNG stream as the database-free one (the
// descriptors use their own generators), so with the default EXHAUSTIVE config and an inlier
// ratio of 1 (no outlier draws, but one shuffle per pair) a seeded run reproduces itself.
#[test]
fn rust_only_synthesize_dataset_with_database_is_deterministic() {
    let options = SyntheticDatasetOptions::default();
    let run = || {
        set_prng_seed(42);
        let database = Database::new();
        let mut reconstruction = Reconstruction::new();
        synthesize_dataset_with_database(&options, &mut reconstruction, &database).unwrap();
        (reconstruction, database)
    };
    let (reconstruction1, database1) = run();
    let (reconstruction2, database2) = run();
    assert_eq!(reconstruction1.images(), reconstruction2.images());
    assert_eq!(reconstruction1.points3d(), reconstruction2.points3d());
    assert_eq!(
        database1.read_all_matches().unwrap(),
        database2.read_all_matches().unwrap()
    );
    for &image_id in reconstruction1.images().keys() {
        assert_eq!(
            database1.read_descriptors(image_id).unwrap(),
            database2.read_descriptors(image_id).unwrap()
        );
    }
}
