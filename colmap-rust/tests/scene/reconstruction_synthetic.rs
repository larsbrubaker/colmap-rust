// Port of colmap/scene/reconstruction_test.cc, part 4, 1:1 (same test names, values and
// checks): the tests that build their scene with SynthesizeDataset — ConstructCopy,
// AssignCopy, Print, DeleteAllPoints2DAndPoints3D, TearDown, ReadWriteTextRoundtrip,
// ReadWriteBinaryRoundtrip, ReadAutoDetectFormat, SetRigsAndFrames (over the in-memory
// Database) and SetRigsAndFramesResetsNumRegImages. COLMAP's `ReconstructionEq` matcher
// (reconstruction_matchers.h) is `expect_reconstruction_eq`, and the test's
// `ExpectEqualSerialization` writes both models' text files to strings. A copy is the
// derived `Clone` (copy construction and assignment alike; the `scene` module header).
// `Frame::ResetRigPtr` has no counterpart (frames hold only their rig's id), and the
// image-to-frame pointer check is `expect_valid_ptrs`.

use colmap_rust::scene::database::Database;
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::reconstruction_io_text_write::{
    write_cameras_text, write_frames_text, write_images_text, write_points3d_text, write_rigs_text,
};
use colmap_rust::scene::synthetic::{
    synthesize_dataset, synthesize_dataset_with_database, SyntheticDatasetOptions,
};

use crate::reconstruction::expect_valid_ptrs;
use crate::reconstruction_io::create_test_dir;

/// Port of `ReconstructionEq` (exact equality of the rigs, cameras, frames, images and
/// points).
pub(crate) fn expect_reconstruction_eq(lhs: &Reconstruction, rhs: &Reconstruction) {
    assert!(lhs.rigs() == rhs.rigs(), "have different rigs");
    assert!(lhs.cameras() == rhs.cameras(), "have different cameras");
    assert!(lhs.frames() == rhs.frames(), "have different frames");
    assert!(lhs.images() == rhs.images(), "have different images");
    assert!(lhs.points3d() == rhs.points3d(), "have different points");
}

type TextWriter = fn(&Reconstruction, &mut Vec<u8>) -> colmap_rust::Result<()>;

/// Port of the test's `ExpectEqualSerialization`.
fn expect_equal_serialization(reconstruction1: &Reconstruction, reconstruction2: &Reconstruction) {
    let writers: [TextWriter; 5] = [
        |r, s| write_rigs_text(r, s),
        |r, s| write_cameras_text(r, s),
        |r, s| write_frames_text(r, s),
        |r, s| write_images_text(r, s),
        |r, s| write_points3d_text(r, s),
    ];
    for write in writers {
        let mut stream1 = Vec::new();
        let mut stream2 = Vec::new();
        write(reconstruction1, &mut stream1).unwrap();
        write(reconstruction2, &mut stream2).unwrap();
        assert_eq!(stream1, stream2);
    }
}

fn synthesize(
    num_rigs: i32,
    num_frames_per_rig: i32,
    num_points3d: i32,
) -> (Reconstruction, SyntheticDatasetOptions) {
    let mut reconstruction = Reconstruction::new();
    let options = SyntheticDatasetOptions {
        num_rigs,
        num_cameras_per_rig: 1,
        num_frames_per_rig,
        num_points3d,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&options, &mut reconstruction).unwrap();
    (reconstruction, options)
}

#[test]
fn reconstruction_construct_copy() {
    // COLMAP sets num_rigs = 2, then 3.
    let (reconstruction, _) = synthesize(3, 8, 21);
    let reconstruction_copy = reconstruction.clone();
    expect_reconstruction_eq(&reconstruction, &reconstruction_copy);
    expect_equal_serialization(&reconstruction, &reconstruction_copy);
    expect_valid_ptrs(&reconstruction);
    expect_valid_ptrs(&reconstruction_copy);
}

#[test]
fn reconstruction_assign_copy() {
    // COLMAP sets num_rigs = 2, then 3.
    let (reconstruction, _) = synthesize(3, 8, 21);
    let mut reconstruction_copy = Reconstruction::new();
    assert_eq!(reconstruction_copy.num_rigs(), 0);
    reconstruction_copy.clone_from(&reconstruction);
    expect_reconstruction_eq(&reconstruction, &reconstruction_copy);
    expect_equal_serialization(&reconstruction, &reconstruction_copy);
    expect_valid_ptrs(&reconstruction);
    expect_valid_ptrs(&reconstruction_copy);
}

#[test]
fn reconstruction_print() {
    let (reconstruction, _) = synthesize(1, 2, 3);
    assert_eq!(
        reconstruction.to_string(),
        "Reconstruction(num_rigs=1, num_cameras=1, num_frames=2, num_reg_frames=2, \
         num_images=2, num_points3D=3)"
    );
}

#[test]
fn reconstruction_delete_all_points2d_and_points3d() {
    let (mut reconstruction, _) = synthesize(2, 20, 50);
    reconstruction.delete_all_points2d_and_points3d().unwrap();
    assert_eq!(reconstruction.num_points3d(), 0);
    expect_valid_ptrs(&reconstruction);
}

#[test]
fn reconstruction_tear_down() {
    let (mut reconstruction, _) = synthesize(2, 3, 10);

    // De-register one frame to create an unregistered frame
    let reg_frames_before = reconstruction.num_reg_frames();
    let frame_ids = reconstruction.reg_frame_ids().to_vec();
    reconstruction.deregister_frame(frame_ids[0]).unwrap();
    assert_eq!(reconstruction.num_reg_frames(), reg_frames_before - 1);

    // TearDown should remove the unregistered frame and its images
    let num_frames_before = reconstruction.num_frames();
    reconstruction.tear_down();
    assert!(reconstruction.num_frames() < num_frames_before);
    assert_eq!(reconstruction.num_frames(), reconstruction.num_reg_frames());
    expect_valid_ptrs(&reconstruction);
}

#[test]
fn reconstruction_read_write_text_roundtrip() {
    let (reconstruction, _) = synthesize(1, 3, 5);

    let test_dir = create_test_dir();
    reconstruction.write_text(&test_dir).unwrap();

    let mut loaded = Reconstruction::new();
    loaded.read_text(&test_dir).unwrap();

    expect_reconstruction_eq(&loaded, &reconstruction);
    expect_valid_ptrs(&loaded);
}

#[test]
fn reconstruction_read_write_binary_roundtrip() {
    let (reconstruction, _) = synthesize(1, 3, 5);

    let test_dir = create_test_dir();
    reconstruction.write_binary(&test_dir).unwrap();

    let mut loaded = Reconstruction::new();
    loaded.read_binary(&test_dir).unwrap();

    expect_reconstruction_eq(&loaded, &reconstruction);
    expect_valid_ptrs(&loaded);
}

#[test]
fn reconstruction_read_auto_detect_format() {
    let (reconstruction, _) = synthesize(1, 2, 3);

    // Write binary and verify Read auto-detects binary format
    {
        let test_dir = create_test_dir();
        reconstruction.write_binary(&test_dir).unwrap();

        let mut loaded = Reconstruction::new();
        loaded.read(&test_dir).unwrap();

        expect_reconstruction_eq(&loaded, &reconstruction);
        expect_valid_ptrs(&loaded);
    }

    // Write text and verify Read auto-detects text format
    {
        let test_dir = create_test_dir();
        reconstruction.write_text(&test_dir).unwrap();

        let mut loaded = Reconstruction::new();
        loaded.read(&test_dir).unwrap();

        expect_reconstruction_eq(&loaded, &reconstruction);
        expect_valid_ptrs(&loaded);
    }
}

#[test]
fn reconstruction_set_rigs_and_frames() {
    colmap_rust::math::random::set_prng_seed(0);
    let mut reconstruction = Reconstruction::new();
    let database = Database::new();
    // COLMAP's test sets `num_rigs` twice (2, then 3); the second assignment wins.
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 3,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 8,
        num_points3d: 21,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset_with_database(&synthetic_dataset_options, &mut reconstruction, &database)
        .unwrap();
    let frame_ids: Vec<_> = reconstruction.frames().keys().copied().collect();
    for frame_id in frame_ids {
        reconstruction.deregister_frame(frame_id).unwrap();
    }
    let orig_reconstruction = reconstruction.clone();
    reconstruction
        .set_rigs_and_frames(
            database.read_all_rigs().unwrap(),
            database.read_all_frames().unwrap(),
        )
        .unwrap();
    expect_reconstruction_eq(&reconstruction, &orig_reconstruction);
    expect_equal_serialization(&reconstruction, &orig_reconstruction);
}

#[test]
fn reconstruction_set_rigs_and_frames_resets_num_reg_images() {
    let mut reconstruction = Reconstruction::new();
    let synthetic_dataset_options = SyntheticDatasetOptions {
        num_rigs: 2,
        num_cameras_per_rig: 1,
        num_frames_per_rig: 4,
        num_points3d: 0,
        ..SyntheticDatasetOptions::default()
    };
    synthesize_dataset(&synthetic_dataset_options, &mut reconstruction).unwrap();
    let num_reg_images_before = reconstruction.num_reg_images();
    assert!(num_reg_images_before > 0);
    // Copy rigs and frames (with poses) from the reconstruction to re-apply.
    let rigs: Vec<_> = reconstruction.rigs().values().cloned().collect();
    let frames: Vec<_> = reconstruction.frames().values().cloned().collect();
    let num_rigs_before = reconstruction.num_rigs();
    let num_frames_before = reconstruction.num_frames();
    let num_reg_frames_before = reconstruction.num_reg_frames();
    // Call SetRigsAndFrames while frames are still registered. Previously this
    // would double-count num_reg_images_ because it was not reset to zero.
    reconstruction.set_rigs_and_frames(rigs, frames).unwrap();
    // Verify num_reg_images_ is not double-counted.
    assert_eq!(reconstruction.num_reg_images(), num_reg_images_before);
    // Verify rigs, frames, and registered frames are preserved.
    assert_eq!(reconstruction.num_rigs(), num_rigs_before);
    assert_eq!(reconstruction.num_frames(), num_frames_before);
    assert_eq!(reconstruction.num_reg_frames(), num_reg_frames_before);
    // Verify every registered frame still has a pose.
    for &frame_id in reconstruction.reg_frame_ids() {
        assert!(reconstruction.frame(frame_id).unwrap().has_pose());
    }
    // Verify image-to-frame pointers are correctly re-wired.
    for image in reconstruction.images().values() {
        assert!(image.has_frame_id());
    }
    expect_valid_ptrs(&reconstruction);
}
