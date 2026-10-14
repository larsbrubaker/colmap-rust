// Port of colmap/scene/synthetic_test.cc, 1:1 (same test names, values and checks) for the
// tests that need no database: colmap_rust::scene::synthetic::synthesize_dataset. The tests
// that write to or read from a `Database` (Nominal, MultipleTimes, WithPriors,
// MultiReconstruction, the *Matches tests and every SynthesizeNoise test) and
// SynthesizeImages.Nominal land with the database and `synthesize_noise` /
// `synthesize_images` (PORTING_PLAN.md, Phase 4). COLMAP's NoDatabase test opens an
// in-memory database it never uses; the port leaves it out.

use colmap_rust::math::random::set_prng_seed;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::synthetic::{synthesize_dataset, SyntheticDatasetOptions};

#[test]
fn synthesize_dataset_no_database() {
    let options = SyntheticDatasetOptions::default();
    let mut reconstruction = Reconstruction::new();
    synthesize_dataset(&options, &mut reconstruction).unwrap();
}

#[test]
fn synthesize_dataset_track_length() {
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 5,
        num_points3d: 50,
        num_points2d_without_point3d: 0,
        track_length: 3,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut reconstruction).unwrap();

    let num_images = options.num_rigs * options.num_cameras_per_rig * options.num_frames_per_rig;
    assert_eq!(reconstruction.num_reg_images(), num_images as usize);
    assert_eq!(reconstruction.num_points3d(), options.num_points3d as usize);
    assert!(
        (reconstruction.compute_mean_track_length().unwrap() - f64::from(options.track_length))
            .abs()
            <= 1e-6
    );
    assert_eq!(
        reconstruction.compute_num_observations().unwrap(),
        (options.num_points3d * options.track_length) as usize
    );
}

#[test]
fn synthesize_dataset_determinism() {
    let options = SyntheticDatasetOptions::default();

    let mut reconstruction1 = Reconstruction::new();
    set_prng_seed(42);
    synthesize_dataset(&options, &mut reconstruction1).unwrap();

    let mut reconstruction2 = Reconstruction::new();
    set_prng_seed(42);
    synthesize_dataset(&options, &mut reconstruction2).unwrap();

    assert_eq!(reconstruction1.rigs(), reconstruction2.rigs());
    assert_eq!(reconstruction1.frames(), reconstruction2.frames());
    assert_eq!(reconstruction1.cameras(), reconstruction2.cameras());
    assert_eq!(reconstruction1.images(), reconstruction2.images());
    assert_eq!(reconstruction1.points3d(), reconstruction2.points3d());
}
