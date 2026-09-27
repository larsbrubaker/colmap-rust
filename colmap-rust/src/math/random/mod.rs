//! Port of COLMAP's `colmap/math/random.h` and `random.cc`: the thread-local PRNG
//! (`SetPRNGSeed`, `kDefaultPRNGSeed`) and the draws every seeded algorithm goes through
//! (`RandomUniformInteger`, `RandomUniformReal`, `RandomGaussian` and the partial
//! Fisher-Yates `Shuffle`). The engine is [`mt19937`] and the distribution algorithms are
//! libc++'s, in [`libcxx`], because the pycolmap oracle is built against libc++. Port of
//! colmap-sharp's `Mathematics/RandomUtils.cs`. Tests: `tests/math/random.rs`
//! (random_test.cc 1:1) and `tests/math/rust_only_random_oracle.rs` (bit-exact oracle).
//!
//! Tier A (exact): the same seed gives the same draws as COLMAP on macOS built without
//! floating-point contraction, bit for bit (entry 40 for the FMA caveat; `random_gaussian`
//! calls `log`, entry 1).
//!
//! Translation notes:
//! - `thread_local std::unique_ptr<std::mt19937> PRNG` becomes a `thread_local!`
//!   `RefCell<Option<Mt19937>>`, `None` until first use on each thread, exactly like COLMAP.
//!   Rust has no async thread migration to worry about (colmap-sharp's `[ThreadStatic]`
//!   caveat), and rayon workers each get their own lazily seeded engine, as COLMAP's
//!   ThreadPool workers do.
//! - Test isolation: COLMAP's gtest_main reseeds with 0 before every test. libtest runs each
//!   test on a fresh thread, whose PRNG lazily seeds with `kDefaultPRNGSeed` (0) — the same
//!   state — so no hook is needed natively. On targets without threads (wasm32-wasip1) tests
//!   share one thread, so the ported tests that depend on the state call `set_prng_seed(0)`
//!   first, which is what gtest_main does.
//! - `SetPRNGSeed` also calls `srand(seed)` in COLMAP. Nothing in COLMAP 4.2.0 calls `rand()`
//!   and Rust has no C runtime `rand()` to seed, so that line has no counterpart.
//! - `random_gaussian` builds a fresh `normal_distribution` per call as COLMAP does, so the
//!   polar method's cached second value is thrown away and every call consumes a new pair.

pub mod libcxx;
pub mod mt19937;

pub use libcxx::{CanonicalFloat, NormalDistribution, UniformInt};
pub use mt19937::Mt19937;

use crate::check_le;
use std::cell::RefCell;
use std::sync::atomic::{AtomicI32, Ordering};

thread_local! {
    // COLMAP's `thread_local std::unique_ptr<std::mt19937> PRNG`.
    static PRNG: RefCell<Option<Mt19937>> = const { RefCell::new(None) };
}

// COLMAP's `int kDefaultPRNGSeed = 0;`, a mutable global shared by all threads.
static DEFAULT_PRNG_SEED: AtomicI32 = AtomicI32::new(0);

/// `colmap::kDefaultPRNGSeed`: the seed used by [`set_prng_seed_default`] and by a thread's
/// first draw when it was never seeded.
pub fn default_prng_seed() -> i32 {
    DEFAULT_PRNG_SEED.load(Ordering::Relaxed)
}

/// Assigns `colmap::kDefaultPRNGSeed` (a global, not per thread).
pub fn set_default_prng_seed(seed: i32) {
    DEFAULT_PRNG_SEED.store(seed, Ordering::Relaxed);
}

/// `SetPRNGSeed(seed)`: initialize the calling thread's PRNG with the given seed.
pub fn set_prng_seed(seed: u32) {
    PRNG.with(|p| *p.borrow_mut() = Some(Mt19937::new(seed)));
}

/// `SetPRNGSeed()`: initialize the calling thread's PRNG with `kDefaultPRNGSeed` (the int is
/// converted to unsigned, as COLMAP's default argument does).
pub fn set_prng_seed_default() {
    set_prng_seed(default_prng_seed() as u32);
}

/// `PRNG.reset()`: drop the calling thread's PRNG; the next draw seeds it lazily.
pub fn reset_prng() {
    PRNG.with(|p| *p.borrow_mut() = None);
}

/// `PRNG != nullptr` for the calling thread.
pub fn prng_is_set() -> bool {
    PRNG.with(|p| p.borrow().is_some())
}

/// Runs `f` with the calling thread's PRNG, seeding it first if it is unset (COLMAP's
/// `if (PRNG == nullptr) SetPRNGSeed();` at the top of every draw). This is also how code
/// that draws from `*PRNG` directly (e.g. `std::shuffle(..., *PRNG)`) is ported.
pub fn with_prng<R>(f: impl FnOnce(&mut Mt19937) -> R) -> R {
    PRNG.with(|p| {
        let mut slot = p.borrow_mut();
        let engine = slot.get_or_insert_with(|| Mt19937::new(default_prng_seed() as u32));
        f(engine)
    })
}

/// `RandomUniformInteger(min, max)`: a uniformly distributed integer in `[min, max]`, both
/// inclusive.
pub fn random_uniform_integer<T: UniformInt>(min: T, max: T) -> T {
    with_prng(|g| libcxx::uniform_int(g, min, max))
}

/// `RandomUniformReal(min, max)`: a uniformly distributed real number in `[min, max)`.
pub fn random_uniform_real<T: CanonicalFloat>(min: T, max: T) -> T {
    with_prng(|g| libcxx::uniform_real(g, min, max))
}

/// `RandomGaussian(mean, stddev)`: a Gaussian distributed real number.
pub fn random_gaussian<T: CanonicalFloat>(mean: T, stddev: T) -> T {
    with_prng(|g| NormalDistribution::new(mean, stddev).sample(g))
}

/// `Shuffle(num_to_shuffle, &elems)`: Fisher-Yates shuffling of the first `num_to_shuffle`
/// elements (each swapped with a uniformly chosen element at or after it).
pub fn shuffle<T>(num_to_shuffle: u32, elems: &mut [T]) -> crate::Result<()> {
    check_le!(num_to_shuffle as usize, elems.len());
    // Wraps to u32::MAX for an empty slice, as in COLMAP; the loop never runs then.
    let last_idx = (elems.len() as u32).wrapping_sub(1);
    for i in 0..num_to_shuffle {
        let j = random_uniform_integer::<u32>(i, last_idx);
        elems.swap(i as usize, j as usize);
    }
    Ok(())
}
