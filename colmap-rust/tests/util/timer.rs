// Port of COLMAP's src/colmap/util/timer_test.cc, 1:1 (Suite_Name -> suite_name), testing
// colmap_rust::util::timer. The elapsed values depend on the wall clock, so, as in COLMAP,
// only zero, monotonicity and the frozen value while paused are asserted.

use colmap_rust::util::timer::Timer;

#[test]
fn timer_default() {
    let timer = Timer::new();
    assert_eq!(timer.elapsed_micro_seconds(), 0.0);
    assert_eq!(timer.elapsed_seconds(), 0.0);
    assert_eq!(timer.elapsed_minutes(), 0.0);
    assert_eq!(timer.elapsed_hours(), 0.0);
}

#[test]
fn timer_start() {
    let mut timer = Timer::new();
    timer.start();
    assert!(timer.elapsed_micro_seconds() >= 0.0);
    assert!(timer.elapsed_seconds() >= 0.0);
    assert!(timer.elapsed_minutes() >= 0.0);
    assert!(timer.elapsed_hours() >= 0.0);
}

#[test]
fn timer_pause() {
    let mut timer = Timer::new();
    timer.start();
    timer.pause();
    let mut prev_time = timer.elapsed_micro_seconds();
    for _ in 0..1000 {
        assert_eq!(timer.elapsed_micro_seconds(), prev_time);
        prev_time = timer.elapsed_micro_seconds();
    }
    timer.resume();
    for _ in 0..1000 {
        assert!(timer.elapsed_micro_seconds() >= prev_time);
    }
    timer.reset();
    assert_eq!(timer.elapsed_micro_seconds(), 0.0);
}

// Rust-only: time advances natively, and Restart starts the count over.
#[test]
fn rust_only_timer_advances_and_restarts() {
    let mut timer = Timer::new();
    timer.start();
    std::thread::sleep(std::time::Duration::from_millis(5));
    let elapsed = timer.elapsed_micro_seconds();
    assert!(elapsed >= 5000.0, "{elapsed}");
    assert!(timer.elapsed_seconds() >= 0.005);
    timer.restart();
    assert!(timer.elapsed_micro_seconds() < elapsed);
}
