// Integration-test binary for allocation bounds in `colmap_rust::linalg`. Rust-only (COLMAP
// has no test for Eigen itself): the allocation half of colmap-sharp's
// `SpectralTests.JacobiSVD_ThinFactorsOfWideInputStaySmall`, which measures allocated bytes
// with .NET's `GC.GetAllocatedBytesForCurrentThread`.
//
// Rust has no such counter, so this binary installs a `#[global_allocator]` that wraps
// `std::alloc::System` and adds every allocation's size to a thread-local counter. It lives
// in its own test binary (not `tests/linalg.rs`) so the counting allocator cannot slow down or
// otherwise affect any other test. The counter is thread-local, so the test harness's other
// threads do not pollute the measurement.
//
// Run: `cargo test -p colmap-rust --test linalg_alloc`.

use colmap_rust::linalg::{JacobiSvd, MatrixXd, SvdOptions};
use colmap_rust::math::fns;
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

thread_local! {
    // Bytes allocated (alloc, alloc_zeroed and the new size of realloc) on this thread.
    static ALLOCATED: Cell<u64> = const { Cell::new(0) };
}

struct CountingAllocator;

fn count(bytes: usize) {
    // `try_with` because the allocator also runs while thread-locals are being torn down.
    let _ = ALLOCATED.try_with(|c| c.set(c.get() + bytes as u64));
}

// SAFETY: every method forwards to `System` with the caller's arguments unchanged, so the
// `GlobalAlloc` contract is `System`'s; the counter only reads the layout sizes.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: forwarded unchanged (see the impl comment).
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count(layout.size());
        // SAFETY: forwarded unchanged.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: forwarded unchanged.
        unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count(new_size);
        // SAFETY: forwarded unchanged.
        unsafe { System.realloc(ptr, layout, new_size) }
    }
}

#[global_allocator]
static GLOBAL: CountingAllocator = CountingAllocator;

fn allocated_bytes() -> u64 {
    ALLOCATED.with(Cell::get)
}

fn max_abs(m: &MatrixXd) -> f64 {
    m.as_slice().iter().fold(0.0_f64, |acc, v| acc.max(v.abs()))
}

// Regression: a thin U of a wide matrix went through the full N x N Householder Q
// (3 x 10000 allocated ~800 MB). Thin factors must stay O(N): the decomposition of a
// 3 x 20000 input allocates under 20 MB (an N x N Q would be 3.2 GB).
#[test]
fn rust_only_jacobi_svd_thin_factors_of_wide_input_stay_small() {
    let n = 20000;
    let mut a = MatrixXd::zeros(3, n);
    for j in 0..n {
        let jf = j as f64;
        a[(0, j)] = fns::sin(jf);
        a[(1, j)] = fns::cos(0.5 * jf);
        a[(2, j)] = fns::sin(0.25 * jf + 1.0);
    }

    let before = allocated_bytes();
    let svd = JacobiSvd::new(&a, SvdOptions::THIN_UV);
    let allocated = allocated_bytes() - before;
    // The lower bound guards the counter itself: V alone is 20000 x 3 doubles.
    assert!(allocated >= 480_000, "counter saw only {allocated} bytes");
    assert!(allocated < 20_000_000, "allocated {allocated} bytes");

    let v = svd.matrix_v();
    let u = svd.matrix_u();
    assert_eq!(v.rows(), n);
    assert_eq!(v.cols(), 3);
    let sigma = MatrixXd::from_diagonal(&svd.singular_values());
    let error = max_abs(&(&(&(&u * &sigma) * &v.transpose()) - &a));
    assert!(error <= 1e-11, "error {error}");
    let orthogonality = max_abs(&(&(&v.transpose() * &v) - &MatrixXd::identity(3)));
    assert!(orthogonality <= 1e-12, "orthogonality {orthogonality}");
}
