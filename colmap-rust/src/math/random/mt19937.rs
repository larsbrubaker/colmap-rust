//! `std::mt19937`: the 32-bit Mersenne Twister as the C++ standard defines it
//! ([rand.eng.mers] and [rand.predef]: w=32, n=624, m=397, r=31, a=0x9908b0df, u=11,
//! d=0xffffffff, s=7, b=0x9d2c5680, t=15, c=0xefc60000, l=18, f=1812433253). It is the engine
//! behind COLMAP's thread-local PRNG (`colmap/math/random.h`, ported in [`super`]); the
//! libc++ distributions that turn its words into numbers are in [`super::libcxx`].
//!
//! Port of colmap-sharp's `Mathematics/Mt19937.cs`, which was written from the standard's
//! definition (Matsumoto & Nishimura, 1998), not from any library's source. The standard pins
//! the output sequence (the 10000th draw of a default-seeded engine is 4123659995), so every
//! conforming library produces the same words for the same seed. Tier A (exact); pinned by
//! `tests/math/rust_only_random_oracle.rs` against the libc++ oracle fixture.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// The 32-bit Mersenne Twister engine, bit-identical to C++'s `std::mt19937`.
#[derive(Clone)]
pub struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    /// Smallest value [`Mt19937::next_u32`] returns (`std::mt19937::min()`).
    pub const MIN: u32 = 0;
    /// Largest value [`Mt19937::next_u32`] returns (`std::mt19937::max()`).
    pub const MAX: u32 = u32::MAX;
    /// `std::mt19937::default_seed`.
    pub const DEFAULT_SEED: u32 = 5489;

    /// Creates an engine seeded like `std::mt19937(seed)`.
    pub fn new(seed: u32) -> Self {
        let mut engine = Self {
            state: [0; N],
            index: N,
        };
        engine.seed(seed);
        engine
    }

    /// Re-seeds the engine like `std::mt19937::seed(value)`.
    pub fn seed(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..N {
            let previous = self.state[i - 1];
            self.state[i] = 1_812_433_253u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    /// Returns the next 32-bit word (`std::mt19937::operator()`).
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    // Regenerates all N words at once. libc++ twists one word per draw instead; the words
    // come out the same either way, because word i only depends on words i+1 and i+M of the
    // previous generation, which the in-place loop has not overwritten yet when it needs them
    // (or has, for i+M >= N, exactly as the recurrence requires).
    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & UPPER_MASK) | (self.state[(i + 1) % N] & LOWER_MASK);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= MATRIX_A;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }
}

impl Default for Mt19937 {
    /// `std::mt19937()`: seeded with [`Mt19937::DEFAULT_SEED`].
    fn default() -> Self {
        Self::new(Self::DEFAULT_SEED)
    }
}
