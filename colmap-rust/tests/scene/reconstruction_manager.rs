// Port of colmap/scene/reconstruction_manager_test.cc, 1:1 (same test names, values and
// checks), for colmap_rust::scene::reconstruction_manager::ReconstructionManager.

use colmap_rust::scene::reconstruction_manager::ReconstructionManager;

#[test]
fn reconstruction_manager_empty() {
    let reconstruction_manager = ReconstructionManager::new();
    assert_eq!(reconstruction_manager.size(), 0);
}

#[test]
fn reconstruction_manager_add_get() {
    let mut reconstruction_manager = ReconstructionManager::new();
    assert_eq!(reconstruction_manager.size(), 0);
    for i in 0..10 {
        let idx = reconstruction_manager.add();
        assert_eq!(reconstruction_manager.size(), i + 1);
        assert_eq!(idx, i);
        assert_eq!(reconstruction_manager.get(idx).unwrap().num_cameras(), 0);
        assert_eq!(reconstruction_manager.get(idx).unwrap().num_images(), 0);
        assert_eq!(reconstruction_manager.get(idx).unwrap().num_points3d(), 0);
    }
}

#[test]
fn reconstruction_manager_delete() {
    let mut reconstruction_manager = ReconstructionManager::new();
    assert_eq!(reconstruction_manager.size(), 0);
    for _ in 0..10 {
        reconstruction_manager.add();
    }

    assert_eq!(reconstruction_manager.size(), 10);
    for i in 0..10 {
        reconstruction_manager.delete(0).unwrap();
        assert_eq!(reconstruction_manager.size(), 9 - i);
    }
}

#[test]
fn reconstruction_manager_clear() {
    let mut reconstruction_manager = ReconstructionManager::new();
    assert_eq!(reconstruction_manager.size(), 0);
    for _ in 0..10 {
        reconstruction_manager.add();
    }

    assert_eq!(reconstruction_manager.size(), 10);
    reconstruction_manager.clear();
    assert_eq!(reconstruction_manager.size(), 0);
}
