// Port of the GetEffectiveNumThreads case of COLMAP's src/colmap/util/threading_test.cc,
// testing colmap_rust::util::threading. The Thread, ThreadPool, JobQueue, ThreadSafeQueue
// and Barrier cases are not ported: those classes are replaced by rayon at the call sites
// (src/util/threading.rs header).

use colmap_rust::util::threading::get_effective_num_threads;

#[test]
fn get_effective_num_threads_nominal() {
    assert!(get_effective_num_threads(-2) > 0);
    assert!(get_effective_num_threads(-1) > 0);
    assert!(get_effective_num_threads(0) > 0);
    assert_eq!(get_effective_num_threads(1), 1);
    assert_eq!(get_effective_num_threads(2), 2);
    assert_eq!(get_effective_num_threads(3), 3);
}
