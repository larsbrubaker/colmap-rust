// Integration-test binary for allocation bounds in `colmap_rust::sensor::bitmap`. Rust-only
// (COLMAP has no memory test for Bitmap::Rescale): the web app runs Rescale/Thumbnail on the
// single wasm thread, where a full-image double-precision intermediate (height x new_width x
// channels x 8 bytes, ~232 MB for a 4032x3024 -> 3200 thumbnail) would exhaust memory. The
// resampler keeps only the source rows the vertical filter needs; this pins that.
//
// Like `tests/linalg_alloc.rs`, the binary installs a `#[global_allocator]` wrapping
// `std::alloc::System`, here tracking the thread's live bytes and their peak. It lives in its
// own test binary so the counting allocator cannot affect any other test; the counters are
// thread-local, so the harness's other threads do not pollute the measurement.
//
// Run: `cargo test -p colmap-rust --test sensor_alloc`.

use colmap_rust::sensor::bitmap::{Bitmap, RescaleFilter};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    // Bytes currently allocated on this thread (may go negative for memory freed here that
    // another thread allocated; the peak is measured relative to a baseline).
    static LIVE: Cell<i64> = const { Cell::new(0) };
    static PEAK: Cell<i64> = const { Cell::new(0) };
}

fn add(bytes: i64) {
    // `try_with` because the allocator also runs while thread-locals are being torn down.
    let _ = LIVE.try_with(|live| {
        let now = live.get() + bytes;
        live.set(now);
        let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
    });
}

struct CountingAllocator;

// SAFETY: every method forwards to `System` with the caller's arguments unchanged, so the
// `GlobalAlloc` contract is `System`'s; the counters only read the layout sizes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        add(layout.size() as i64);
        // SAFETY: forwarded unchanged (see the impl comment).
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        add(layout.size() as i64);
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        add(-(layout.size() as i64));
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        add(new_size as i64 - layout.size() as i64);
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

/// Peak bytes allocated above the live total at the start of `f`.
fn peak_extra_bytes(f: impl FnOnce()) -> i64 {
    let baseline = LIVE.with(Cell::get);
    PEAK.with(|peak| peak.set(baseline));
    f();
    PEAK.with(Cell::get) - baseline
}

#[test]
fn rust_only_rescale_peak_memory_is_bounded_by_the_output() {
    for (filter, new_width, new_height) in [
        (RescaleFilter::Bilinear, 800, 640), // Thumbnail-style downscale.
        (RescaleFilter::Box, 800, 640),
        (RescaleFilter::Bilinear, 1500, 1200), // Upscale.
    ] {
        let mut bitmap = Bitmap::new(1000, 800, true);
        for (i, v) in bitmap.row_major_data_mut().iter_mut().enumerate() {
            *v = (i * 31 % 251) as u8;
        }
        let output_bytes = (new_width * new_height * 3) as i64;
        let peak = peak_extra_bytes(|| bitmap.rescale(new_width, new_height, filter));
        // The output buffer plus a few filter rows and the per-axis weights. A full
        // intermediate would be 800 x new_width x 3 x 8 bytes (15-29 MB here).
        let budget = output_bytes + (1 << 20);
        assert!(
            peak <= budget,
            "{filter:?} -> {new_width}x{new_height}: peak {peak} bytes > budget {budget}"
        );
    }
}
