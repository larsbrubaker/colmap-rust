// Rust-only checks of colmap_rust::scene::reconstruction_manager beyond
// reconstruction_manager_test.cc: Write/Read round trips through model folders, ordered by
// point count with ties in index order (the `scene/reconstruction_manager.rs` header), and the
// error paths of Get, Set and Delete.

use colmap_rust::linalg::{Vector3d, Vector3ub};
use colmap_rust::scene::reconstruction::Reconstruction;
use colmap_rust::scene::reconstruction_manager::ReconstructionManager;
use colmap_rust::scene::track::Track;
use colmap_rust::ErrorKind;

use crate::reconstruction_io::create_test_dir;

#[test]
fn rust_only_write_orders_by_point_count_then_index() {
    // Point counts 1, 2, 1, 2 by index: written as 1, 3, 0, 2.
    let counts = [1usize, 2, 1, 2];
    let mut manager = ReconstructionManager::new();
    for (i, &count) in counts.iter().enumerate() {
        let idx = manager.add();
        let reconstruction = manager.get_mut(idx).unwrap();
        for j in 0..count {
            // x encodes the source index so the written folder can be traced back.
            reconstruction
                .add_point3d(
                    Vector3d::new(i as f64, j as f64, 0.0),
                    Track::new(),
                    Vector3ub::default(),
                )
                .unwrap();
        }
    }

    let dir = create_test_dir();
    manager.write(&dir).unwrap();

    let expected_source = [1usize, 3, 0, 2];
    let mut read_back = ReconstructionManager::new();
    for (i, &source) in expected_source.iter().enumerate() {
        let idx = read_back.read(&dir.join(i.to_string())).unwrap();
        assert_eq!(idx, i);
        let written = read_back.get(idx).unwrap();
        assert_eq!(written.num_points3d(), counts[source]);
        assert_eq!(
            written.points3d().values().next().unwrap().xyz.x,
            source as f64
        );
    }
}

#[test]
fn rust_only_get_set_delete_out_of_range() {
    let mut manager = ReconstructionManager::new();
    assert_eq!(manager.get(0).unwrap_err().kind(), ErrorKind::OutOfRange);
    assert_eq!(
        manager.get_mut(0).unwrap_err().kind(),
        ErrorKind::OutOfRange
    );
    assert!(manager.set(0, Reconstruction::new()).is_err());
    assert!(manager.delete(0).is_err());

    let idx = manager.add();
    let mut replacement = Reconstruction::new();
    replacement
        .add_point3d(Vector3d::default(), Track::new(), Vector3ub::default())
        .unwrap();
    let old = manager.set(idx, replacement).unwrap();
    assert_eq!(old.num_points3d(), 0);
    assert_eq!(manager.get(idx).unwrap().num_points3d(), 1);
}

#[test]
fn rust_only_read_missing_folder_keeps_the_added_model() {
    // COLMAP's Read adds the reconstruction before reading into it.
    let mut manager = ReconstructionManager::new();
    let dir = create_test_dir();
    assert!(manager.read(&dir.join("missing")).is_err());
    assert_eq!(manager.size(), 1);
}
