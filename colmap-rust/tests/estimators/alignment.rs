// Port of COLMAP's src/colmap/estimators/alignment_test.cc (same values and tolerances).
// Tier C (LO-RANSAC) for the robust alignments, Tier B for AlignmentErrorSummary and
// ComputeImageAlignmentError (pinned by the Rust-only case at the end).
//
// Each test starts with set_prng_seed(0), as COLMAP's gtest_main does. The Rust
// Reconstruction iterates its objects in ascending id order where COLMAP iterates hash maps
// (the `scene/reconstruction.rs` header), so the RANSAC samples can differ from COLMAP's; the
// assertions are outcome checks.

use std::collections::BTreeMap;

use colmap_rust::estimators::alignment::{
    align_reconstruction_to_locations, align_reconstruction_to_orig_rig_scales,
    align_reconstruction_to_pose_priors, align_reconstructions_via_points,
    align_reconstructions_via_proj_centers, align_reconstructions_via_reprojections,
    merge_reconstructions, AlignmentErrorSummary, ImageAlignmentError,
};
use colmap_rust::geometry::{CoordinateSystem, PosePrior, Sim3d};
use colmap_rust::linalg::{Matrix3d, Vector3d};
use colmap_rust::math::random::{random_uniform_real, set_prng_seed};
use colmap_rust::math::random_eigen::{random_eigen_quaterniond, random_eigen_vector3d};
use colmap_rust::optim::ransac::RansacOptions;
use colmap_rust::scene::image::Image;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};
use colmap_rust::sensor::rig::Rig;
use colmap_rust::util::types::{PosePriorId, RigId};

use crate::transform_matchers::rigid3d_near;

fn test_sim3d() -> Sim3d {
    // C++ evaluates the constructor arguments left to right under clang.
    let scale = random_uniform_real(0.5, 2.0);
    let rotation = random_eigen_quaterniond();
    let translation = random_eigen_vector3d();
    Sim3d::new(scale, rotation, translation)
}

fn expect_equal_sim3d(gt_tgt_from_src: &Sim3d, tgt_from_src: &Sim3d) {
    assert!((gt_tgt_from_src.scale - tgt_from_src.scale).abs() <= 1e-6);
    assert!(
        gt_tgt_from_src
            .rotation
            .angular_distance(tgt_from_src.rotation)
            < 1e-6
    );
    assert!((gt_tgt_from_src.translation - tgt_from_src.translation).norm() < 1e-6);
}

fn generate_reconstruction_for_alignment() -> Reconstruction {
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 10,
        num_points3d: 50,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    reconstruction
}

fn projection_center(reconstruction: &Reconstruction, image: &Image) -> Vector3d {
    let frame = reconstruction.frame(image.frame_id()).unwrap();
    let rig = reconstruction.rig(frame.rig_id()).unwrap();
    image.projection_center(frame, rig).unwrap()
}

fn transformed_pair() -> (Reconstruction, Reconstruction, Sim3d) {
    let src_reconstruction = generate_reconstruction_for_alignment();
    let mut tgt_reconstruction = src_reconstruction.clone();
    let gt_tgt_from_src = test_sim3d();
    tgt_reconstruction.transform(&gt_tgt_from_src);
    (src_reconstruction, tgt_reconstruction, gt_tgt_from_src)
}

#[test]
fn alignment_align_reconstruction_to_locations() {
    set_prng_seed(0);
    let (src_reconstruction, tgt_reconstruction, gt_tgt_from_src) = transformed_pair();

    let mut tgt_image_names = Vec::new();
    let mut tgt_image_locations = Vec::new();
    for image in tgt_reconstruction.images().values() {
        tgt_image_names.push(image.name().to_string());
        tgt_image_locations.push(projection_center(&tgt_reconstruction, image));
    }

    let ransac_options = RansacOptions {
        max_error: 1e-2,
        ..RansacOptions::default()
    };

    let mut tgt_from_src = Sim3d::default();
    assert!(!align_reconstruction_to_locations(
        &src_reconstruction,
        &tgt_image_names,
        &tgt_image_locations,
        /* min_common_images= */ tgt_image_names.len() as i32 + 1,
        &ransac_options,
        &mut tgt_from_src,
    )
    .unwrap());
    assert!(align_reconstruction_to_locations(
        &src_reconstruction,
        &tgt_image_names,
        &tgt_image_locations,
        /* min_common_images= */ 3,
        &ransac_options,
        &mut tgt_from_src,
    )
    .unwrap());
    expect_equal_sim3d(&gt_tgt_from_src, &tgt_from_src);
}

fn pose_priors_of(tgt_reconstruction: &Reconstruction, covariance: f64) -> Vec<PosePrior> {
    let mut tgt_pose_priors = Vec::new();
    for image in tgt_reconstruction.images().values() {
        let pose_prior = PosePrior {
            pose_prior_id: (tgt_pose_priors.len() + 1) as PosePriorId,
            corr_data_id: image.data_id(),
            coordinate_system: CoordinateSystem::Cartesian,
            position: projection_center(tgt_reconstruction, image),
            position_covariance: Matrix3d::identity() * covariance,
            ..PosePrior::default()
        };
        tgt_pose_priors.push(pose_prior);
    }
    tgt_pose_priors
}

#[test]
fn alignment_align_reconstruction_to_pose_priors() {
    set_prng_seed(0);
    let (src_reconstruction, tgt_reconstruction, gt_tgt_from_src) = transformed_pair();

    let tgt_pose_priors = pose_priors_of(&tgt_reconstruction, 1e-2);

    let ransac_options = RansacOptions {
        max_error: 1e-2,
        ..RansacOptions::default()
    };

    let mut tgt_from_src = Sim3d::default();
    assert!(align_reconstruction_to_pose_priors(
        &src_reconstruction,
        &tgt_pose_priors,
        &ransac_options,
        /* prior_position_fallback_stddev= */ 1.0,
        &mut tgt_from_src,
    )
    .unwrap());
    expect_equal_sim3d(&gt_tgt_from_src, &tgt_from_src);
}

#[test]
fn alignment_align_reconstruction_to_pose_priors_with_automatic_max_error() {
    set_prng_seed(0);
    let (src_reconstruction, tgt_reconstruction, gt_tgt_from_src) = transformed_pair();

    let tgt_pose_priors = pose_priors_of(&tgt_reconstruction, 1e-4);

    let ransac_options = RansacOptions {
        max_error: 0.0,
        ..RansacOptions::default()
    };
    let mut tgt_from_src = Sim3d::default();
    assert!(align_reconstruction_to_pose_priors(
        &src_reconstruction,
        &tgt_pose_priors,
        &ransac_options,
        /* prior_position_fallback_stddev= */ 1.0,
        &mut tgt_from_src,
    )
    .unwrap());
    expect_equal_sim3d(&gt_tgt_from_src, &tgt_from_src);
}

#[test]
fn alignment_align_reconstructions_via_reprojections() {
    set_prng_seed(0);
    let (src_reconstruction, tgt_reconstruction, gt_tgt_from_src) = transformed_pair();

    let mut tgt_from_src = Sim3d::default();
    assert!(align_reconstructions_via_reprojections(
        &src_reconstruction,
        &tgt_reconstruction,
        /* min_inlier_observations= */ 0.9,
        /* max_reproj_error= */ 2.0,
        &mut tgt_from_src,
    )
    .unwrap());
    expect_equal_sim3d(&gt_tgt_from_src, &tgt_from_src);
}

#[test]
fn alignment_align_reconstructions_via_proj_centers() {
    set_prng_seed(0);
    let (src_reconstruction, tgt_reconstruction, gt_tgt_from_src) = transformed_pair();

    let mut tgt_from_src = Sim3d::default();
    assert!(align_reconstructions_via_proj_centers(
        &src_reconstruction,
        &tgt_reconstruction,
        /* max_proj_center_error= */ 0.1,
        &mut tgt_from_src,
    )
    .unwrap());
    expect_equal_sim3d(&gt_tgt_from_src, &tgt_from_src);
}

#[test]
fn alignment_align_reconstructions_via_points() {
    set_prng_seed(0);
    let (src_reconstruction, tgt_reconstruction, gt_tgt_from_src) = transformed_pair();

    let mut tgt_from_src = Sim3d::default();
    assert!(align_reconstructions_via_points(
        &src_reconstruction,
        &tgt_reconstruction,
        /* min_common_observations= */ 3,
        /* max_error= */ 0.01,
        /* min_inlier_ratio= */ 0.9,
        &mut tgt_from_src,
    )
    .unwrap());
    expect_equal_sim3d(&gt_tgt_from_src, &tgt_from_src);
}

fn remove_rig_frames(reconstruction: &mut Reconstruction, rig_id: RigId) {
    let frame_ids = reconstruction.reg_frame_ids().to_vec();
    for frame_id in frame_ids {
        if reconstruction.frame(frame_id).unwrap().rig_id() == rig_id {
            reconstruction.deregister_frame(frame_id).unwrap();
        }
    }
}

/// Synthesize a reconstruction which has at least two cameras.
fn generate_three_rig_reconstruction() -> Reconstruction {
    let mut src_reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 3,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 10,
        num_points3d: 50,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut src_reconstruction).unwrap();
    src_reconstruction
}

#[test]
fn alignment_merge_reconstructions() {
    set_prng_seed(0);
    let mut src_reconstruction = generate_three_rig_reconstruction();
    let orig_reconstruction = src_reconstruction.clone();
    let mut tgt_reconstruction = src_reconstruction.clone();

    remove_rig_frames(&mut src_reconstruction, 1);
    remove_rig_frames(&mut tgt_reconstruction, 2);

    // Remove all unregistered rigs/cameras/frames/images.
    src_reconstruction.tear_down();
    tgt_reconstruction.tear_down();
    assert_eq!(src_reconstruction.num_rigs(), 2);
    assert_eq!(src_reconstruction.num_cameras(), 2);
    assert_eq!(src_reconstruction.num_frames(), 20);
    assert_eq!(src_reconstruction.num_reg_frames(), 20);
    assert_eq!(src_reconstruction.num_images(), 20);
    assert_eq!(tgt_reconstruction.num_rigs(), 2);
    assert_eq!(tgt_reconstruction.num_cameras(), 2);
    assert_eq!(tgt_reconstruction.num_frames(), 20);
    assert_eq!(tgt_reconstruction.num_reg_frames(), 20);
    assert_eq!(tgt_reconstruction.num_images(), 20);

    // Merge reconstructions.
    assert!(merge_reconstructions(
        /* max_reproj_error= */ 1e-4,
        &src_reconstruction,
        &mut tgt_reconstruction,
    )
    .unwrap());
    assert_eq!(tgt_reconstruction.num_rigs(), 3);
    assert_eq!(tgt_reconstruction.num_cameras(), 3);
    assert_eq!(tgt_reconstruction.num_frames(), 30);
    assert_eq!(tgt_reconstruction.num_reg_frames(), 30);
    assert_eq!(tgt_reconstruction.num_images(), 30);
    assert_eq!(tgt_reconstruction.num_points3d(), 50);
    assert_eq!(
        tgt_reconstruction.compute_num_observations().unwrap(),
        orig_reconstruction.compute_num_observations().unwrap()
    );
}

#[test]
fn alignment_merge_reconstructions_inconsistent_image_names() {
    // Reconstructions built from independent databases can assign the same image
    // id to distinct physical images. Merging by id would then silently drop
    // images and corrupt tracks, so such an inconsistent id<->name mapping must
    // be detected and rejected rather than merged (see issue #3405).
    set_prng_seed(0);
    let mut src_reconstruction = generate_three_rig_reconstruction();
    let mut tgt_reconstruction = src_reconstruction.clone();

    remove_rig_frames(&mut src_reconstruction, 1);
    remove_rig_frames(&mut tgt_reconstruction, 2);
    src_reconstruction.tear_down();
    tgt_reconstruction.tear_down();

    // Find an image id registered in both reconstructions and give it a different
    // name in the target, simulating an id collision between two distinct images.
    // Enough images still share a consistent name for the alignment step (which
    // matches by name) to succeed, so the merge reaches -- and must fail at --
    // the id/name consistency check.
    let mut found_shared_id = false;
    for image_id in src_reconstruction.reg_image_ids().unwrap() {
        if tgt_reconstruction.exists_image(image_id) {
            tgt_reconstruction
                .image_mut(image_id)
                .unwrap()
                .set_name("colliding_name.jpg");
            found_shared_id = true;
            break;
        }
    }
    assert!(found_shared_id);

    let num_tgt_images_before_merge = tgt_reconstruction.num_images();
    assert!(!merge_reconstructions(
        /* max_reproj_error= */ 1e-4,
        &src_reconstruction,
        &mut tgt_reconstruction,
    )
    .unwrap());
    // The merge must abort before mutating the target reconstruction.
    assert_eq!(tgt_reconstruction.num_images(), num_tgt_images_before_merge);
}

#[test]
fn alignment_align_reconstruction_to_orig_rig_scales() {
    set_prng_seed(0);
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 4,
        num_frames_per_rig: 10,
        num_points3d: 50,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    let orig_rigs: BTreeMap<RigId, Rig> = reconstruction.rigs().clone();

    reconstruction.transform(&test_sim3d());
    align_reconstruction_to_orig_rig_scales(&orig_rigs, &mut reconstruction).unwrap();
    for (&rig_id, orig_rig) in &orig_rigs {
        for (&sensor_id, sensor_from_orig_rig) in orig_rig.non_ref_sensors() {
            let Some(sensor_from_orig_rig) = sensor_from_orig_rig else {
                continue;
            };
            let sensor_from_rig = reconstruction
                .rig(rig_id)
                .unwrap()
                .sensor_from_rig(sensor_id)
                .unwrap();
            assert!(rigid3d_near(
                &sensor_from_rig,
                sensor_from_orig_rig,
                /* rtol= */ 1e-6,
                /* ttol= */ 1e-6,
            ));
        }
    }
}

#[test]
fn alignment_error_summary_empty() {
    let errors: Vec<ImageAlignmentError> = Vec::new();
    let summary = AlignmentErrorSummary::compute(&errors).unwrap();
    assert_eq!(summary.rotation_errors_deg.min, 0.0);
    assert_eq!(summary.proj_center_errors.min, 0.0);
}

#[test]
fn alignment_error_summary_multiple_errors() {
    let mut errors = vec![ImageAlignmentError::default(); 5];
    for (i, error) in errors.iter_mut().enumerate() {
        error.rotation_error_deg = (i + 1) as f64;
        error.proj_center_error = (i + 1) as f64 * 0.1;
    }
    let summary = AlignmentErrorSummary::compute(&errors).unwrap();
    let near = |a: f64, b: f64| assert!((a - b).abs() <= 1e-10, "{a} vs {b}");
    near(summary.rotation_errors_deg.min, 1.0);
    near(summary.rotation_errors_deg.max, 5.0);
    near(summary.rotation_errors_deg.mean, 3.0);
    near(summary.rotation_errors_deg.median, 3.0);
    near(summary.rotation_errors_deg.p90, 4.6);
    near(summary.rotation_errors_deg.p99, 4.96);
    near(summary.proj_center_errors.min, 0.1);
    near(summary.proj_center_errors.max, 0.5);
    near(summary.proj_center_errors.mean, 0.3);
    near(summary.proj_center_errors.p90, 0.46);
    near(summary.proj_center_errors.p99, 0.496);
}

// ComputeImageAlignmentError has no case of its own in alignment_test.cc (COLMAP exercises
// it through the ReconstructionNear test matcher), so this pins it: zero error under the
// true transform, and the known rotation / projection center offsets under a perturbed one.
#[test]
fn rust_only_compute_image_alignment_error() {
    use colmap_rust::estimators::alignment::compute_image_alignment_error;
    use colmap_rust::linalg::Quaterniond;
    use colmap_rust::math::fns;
    use colmap_rust::math::utils::deg_to_rad;

    set_prng_seed(0);
    let (src_reconstruction, tgt_reconstruction, gt_tgt_from_src) = transformed_pair();

    let errors =
        compute_image_alignment_error(&src_reconstruction, &tgt_reconstruction, &gt_tgt_from_src)
            .unwrap();
    assert_eq!(errors.len(), 20);
    for (error, image_id) in errors
        .iter()
        .zip(src_reconstruction.reg_image_ids().unwrap())
    {
        assert_eq!(
            error.image_name,
            src_reconstruction.image(image_id).unwrap().name()
        );
        assert!(error.rotation_error_deg.abs() < 1e-6);
        assert!(error.proj_center_error < 1e-6);
    }

    // Shift the target world by 0.5 along x after the true transform: every projection
    // center moves by 0.5 and no orientation changes. A 3 degree turn about z on top of the
    // true transform changes every camera orientation by 3 degrees.
    let shifted =
        Sim3d::new(1.0, Quaterniond::identity(), Vector3d::new(0.5, 0.0, 0.0)) * gt_tgt_from_src;
    for error in
        compute_image_alignment_error(&src_reconstruction, &tgt_reconstruction, &shifted).unwrap()
    {
        assert!(error.rotation_error_deg.abs() < 1e-6);
        assert!((error.proj_center_error - 0.5).abs() < 1e-9);
    }
    let half_angle = deg_to_rad(3.0) / 2.0;
    let turn = Quaterniond::new(fns::cos(half_angle), 0.0, 0.0, fns::sin(half_angle));
    let turned = Sim3d::new(1.0, turn, Vector3d::new(0.0, 0.0, 0.0)) * gt_tgt_from_src;
    for error in
        compute_image_alignment_error(&src_reconstruction, &tgt_reconstruction, &turned).unwrap()
    {
        assert!((error.rotation_error_deg - 3.0).abs() < 1e-6);
    }
}
