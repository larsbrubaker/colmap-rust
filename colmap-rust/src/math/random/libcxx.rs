//! libc++'s `<random>` distributions and `std::shuffle`, driven by [`Mt19937`].
//!
//! Ported from LLVM libc++ (Apache-2.0 WITH LLVM-exception, see THIRD_PARTY_NOTICES.md) by way
//! of colmap-sharp's `Mathematics/LibcxxRandom.cs`: `<__random/uniform_int_distribution.h>`
//! (including `__independent_bits_engine`), `<__random/generate_canonical.h>`,
//! `<__random/uniform_real_distribution.h>`, `<__random/normal_distribution.h>`,
//! `<__random/log2.h>` and `<__algorithm/shuffle.h>`, as shipped in the macOS SDK that Apple
//! clang 21 uses. Modified: translated to Rust and specialized to `std::mt19937` as the engine
//! (Apache-2.0 section 4(b) notice).
//!
//! Why libc++: the C++ standard leaves the algorithms of these distributions to the
//! implementation, so the same mt19937 stream gives different numbers under libc++ and
//! libstdc++. COLMAP's oracle (the macOS pycolmap wheel) links libc++, so these are libc++'s
//! algorithms and seeded COLMAP code (RANSAC sampling, synthetic datasets) draws the same
//! numbers here. [`super`] wraps them in COLMAP's thread-local PRNG API.
//!
//! Tier A (exact), pinned by `tests/math/rust_only_random_oracle.rs` against
//! `oracle/fixture_random.py`'s libc++ harness. The integer paths are exact everywhere. The
//! real paths are plain IEEE `+ - * /` and `sqrt` (never fused: docs/CPP_DIVERGENCES.md entry
//! 40), except that [`NormalDistribution`] calls `log`, which goes through
//! [`crate::math::fns`] (docs/CPP_DIVERGENCES.md entry 1).
//!
//! Constants are libc++'s template parameters evaluated for `std::mt19937` on macOS, where
//! `mt19937::result_type` (`uint_fast32_t`) is a 32-bit unsigned int: the engine range
//! R = max - min + 1 = 2^32, so `__log2` gives 32 bits per engine call.

use super::mt19937::Mt19937;
use crate::math::fns;
use std::ops::{Add, Div, Mul, Neg, Sub};

// __log2<uint64_t, 2^32>: bits delivered by one mt19937 call.
const ENGINE_BITS: u32 = 32;

/// An integer type `std::uniform_int_distribution<T>` accepts. libc++ works in `uint32_t`
/// for types of at most 32 bits and in `uint64_t` for wider ones; the conversions are C++'s
/// modular `static_cast`s (sign-extending for signed types, truncating back).
pub trait UniformInt: Copy {
    /// True when libc++'s working type is `uint64_t` (the type is wider than 32 bits).
    const WIDE: bool;
    /// `static_cast<uint32_t>(self)`.
    fn to_u32_wrapping(self) -> u32;
    /// `static_cast<uint64_t>(self)`.
    fn to_u64_wrapping(self) -> u64;
    /// `static_cast<T>(v)` from `uint32_t`.
    fn from_u32_wrapping(v: u32) -> Self;
    /// `static_cast<T>(v)` from `uint64_t`.
    fn from_u64_wrapping(v: u64) -> Self;
}

macro_rules! impl_uniform_int {
    ($($t:ty => $wide:expr),*) => {$(
        impl UniformInt for $t {
            const WIDE: bool = $wide;
            #[inline]
            fn to_u32_wrapping(self) -> u32 { self as u32 }
            #[inline]
            fn to_u64_wrapping(self) -> u64 { self as u64 }
            #[inline]
            fn from_u32_wrapping(v: u32) -> Self { v as $t }
            #[inline]
            fn from_u64_wrapping(v: u64) -> Self { v as $t }
        }
    )*};
}

// isize/usize are 64-bit on every native target and 32-bit on wasm32, exactly like
// ptrdiff_t/size_t on those targets, so WIDE follows the pointer width.
impl_uniform_int!(
    i16 => false, u16 => false, i32 => false, u32 => false,
    i64 => true, u64 => true,
    isize => usize::BITS > 32, usize => usize::BITS > 32
);

/// `std::uniform_int_distribution<T>(a, b)(g)`: uniform in `[a, b]`, both inclusive.
pub fn uniform_int<T: UniformInt>(g: &mut Mt19937, a: T, b: T) -> T {
    if !T::WIDE {
        let rp = b
            .to_u32_wrapping()
            .wrapping_sub(a.to_u32_wrapping())
            .wrapping_add(1);
        if rp == 1 {
            return a;
        }
        const DT: u32 = 32;
        if rp == 0 {
            // The full 2^32 range: one engine word.
            return T::from_u32_wrapping(independent_bits_32(g, DT));
        }
        let mut w = DT - rp.leading_zeros() - 1;
        if rp & (u32::MAX >> (DT - w)) != 0 {
            w += 1;
        }
        let mut u;
        loop {
            u = independent_bits_32(g, w);
            if u < rp {
                break;
            }
        }
        T::from_u32_wrapping(u.wrapping_add(a.to_u32_wrapping()))
    } else {
        let rp = b
            .to_u64_wrapping()
            .wrapping_sub(a.to_u64_wrapping())
            .wrapping_add(1);
        if rp == 1 {
            return a;
        }
        const DT: u32 = 64;
        if rp == 0 {
            return T::from_u64_wrapping(independent_bits_64(g, DT));
        }
        let mut w = DT - rp.leading_zeros() - 1;
        if rp & (u64::MAX >> (DT - w)) != 0 {
            w += 1;
        }
        let mut u;
        loop {
            u = independent_bits_64(g, w);
            if u < rp {
                break;
            }
        }
        T::from_u64_wrapping(u.wrapping_add(a.to_u64_wrapping()))
    }
}

// __independent_bits_engine<mt19937, uint32_t>(g, w)(). The working type is uint32, in
// which R = 2^32 wraps to 0, so libc++ takes its __eval(false_type) branch: one engine
// word, masked to the low w bits (n = 1, w0 = w for every w <= 32).
fn independent_bits_32(g: &mut Mt19937, w: u32) -> u32 {
    let mask0 = if w > 0 {
        u32::MAX >> (ENGINE_BITS - w)
    } else {
        0
    };
    g.next_u32() & mask0
}

// __independent_bits_engine<mt19937, uint64_t>(g, w)(). The working type is uint64 and
// R = 2^32 != 0, so this is libc++'s constructor plus __eval(true_type), kept literal:
// the rejection bounds y0/y1 never reject a 32-bit word here, but the shape is the spec.
fn independent_bits_64(g: &mut Mt19937, w: u32) -> u64 {
    const R: u64 = 1u64 << ENGINE_BITS;
    const WDT: u32 = 64;
    const EDT: u32 = 32;

    let mut n = w / ENGINE_BITS + u32::from(!w.is_multiple_of(ENGINE_BITS));
    let mut w0 = w / n;
    let mut y0 = if w0 < WDT { (R >> w0) << w0 } else { 0 };
    if R - y0 > y0 / u64::from(n) {
        n += 1;
        w0 = w / n;
        y0 = if w0 < WDT { (R >> w0) << w0 } else { 0 };
    }
    let n0 = n - (w % n);
    let y1 = if w0 < WDT - 1 {
        (R >> (w0 + 1)) << (w0 + 1)
    } else {
        0
    };
    let mask0 = if w0 > 0 { u32::MAX >> (EDT - w0) } else { 0 };
    let mask1 = if w0 < EDT - 1 {
        u32::MAX >> (EDT - (w0 + 1))
    } else {
        u32::MAX
    };

    let mut sp: u64 = 0;
    for _ in 0..n0 {
        let mut u;
        loop {
            u = g.next_u32() - Mt19937::MIN;
            if u64::from(u) < y0 {
                break;
            }
        }
        sp = if w0 < WDT { sp << w0 } else { 0 };
        sp = sp.wrapping_add(u64::from(u & mask0));
    }
    for _ in n0..n {
        let mut u;
        loop {
            u = g.next_u32() - Mt19937::MIN;
            if u64::from(u) < y1 {
                break;
            }
        }
        sp = if w0 < WDT - 1 { sp << (w0 + 1) } else { 0 };
        sp = sp.wrapping_add(u64::from(u & mask1));
    }
    sp
}

/// A floating-point type libc++'s real distributions are ported for: `f32` and `f64`.
pub trait CanonicalFloat:
    Copy
    + PartialOrd
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// `std::numeric_limits<T>::digits`.
    const DIGITS: u32;
    /// 0.
    const ZERO: Self;
    /// 1.
    const ONE: Self;
    /// `static_cast<T>(v)` (round to nearest).
    fn from_u32(v: u32) -> Self;
    /// `static_cast<T>(v)` from an exact small integer.
    fn from_i32(v: i32) -> Self;
    /// `std::sqrt` (via [`crate::math::fns`]).
    fn sqrt(self) -> Self;
    /// `std::log` (via [`crate::math::fns`]).
    fn ln(self) -> Self;
}

impl CanonicalFloat for f64 {
    const DIGITS: u32 = 53;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_u32(v: u32) -> Self {
        f64::from(v)
    }
    fn from_i32(v: i32) -> Self {
        f64::from(v)
    }
    fn sqrt(self) -> Self {
        fns::sqrt(self)
    }
    fn ln(self) -> Self {
        fns::ln(self)
    }
}

impl CanonicalFloat for f32 {
    const DIGITS: u32 = 24;
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    fn from_u32(v: u32) -> Self {
        // Round to nearest, like the C++ integral-to-floating conversion.
        v as f32
    }
    fn from_i32(v: i32) -> Self {
        v as f32
    }
    fn sqrt(self) -> Self {
        fns::sqrtf(self)
    }
    fn ln(self) -> Self {
        fns::logf(self)
    }
}

/// `std::generate_canonical<T, digits>(g)`: a value in `[0, 1)` built from
/// ceil(digits / 32) engine words (1 for f32, 2 for f64). Like libc++, f32 can round up to
/// exactly 1.
pub fn generate_canonical<T: CanonicalFloat>(g: &mut Mt19937) -> T {
    let b = T::DIGITS;
    let k = b / ENGINE_BITS + u32::from(!b.is_multiple_of(ENGINE_BITS)) + u32::from(b == 0);
    let rp = T::from_u32(Mt19937::MAX - Mt19937::MIN) + T::ONE;
    let mut base = rp;
    let mut sp = T::from_u32(g.next_u32() - Mt19937::MIN);
    for _ in 1..k {
        sp = sp + T::from_u32(g.next_u32() - Mt19937::MIN) * base;
        base = base * rp;
    }
    sp / base
}

/// `std::uniform_real_distribution<T>(a, b)(g)`: uniform in `[a, b)`. The multiply-add is
/// evaluated unfused (docs/CPP_DIVERGENCES.md entry 40).
pub fn uniform_real<T: CanonicalFloat>(g: &mut Mt19937, a: T, b: T) -> T {
    (b - a) * generate_canonical::<T>(g) + a
}

/// `std::shuffle(v.begin(), v.end(), g)`: libc++'s Fisher-Yates pass, which draws from
/// `uniform_int_distribution<ptrdiff_t>` and skips self-swaps.
pub fn shuffle<T>(list: &mut [T], g: &mut Mt19937) {
    let mut d = list.len() as isize;
    if d > 1 {
        let mut first = 0usize;
        let last = list.len() - 1;
        d -= 1;
        while first < last {
            let i = uniform_int::<isize>(g, 0, d);
            if i != 0 {
                list.swap(first, first + i as usize);
            }
            first += 1;
            d -= 1;
        }
    }
}

/// `std::normal_distribution<T>` as libc++ implements it: the Marsaglia polar method, which
/// makes two normal values per accepted pair and caches the second for the next call.
#[derive(Clone, Debug)]
pub struct NormalDistribution<T: CanonicalFloat> {
    mean: T,
    stddev: T,
    cached: T,
    cached_is_hot: bool,
}

impl<T: CanonicalFloat> NormalDistribution<T> {
    /// Creates a distribution with the given mean and standard deviation.
    pub fn new(mean: T, stddev: T) -> Self {
        Self {
            mean,
            stddev,
            cached: T::ZERO,
            cached_is_hot: false,
        }
    }

    /// The distribution's mean.
    pub fn mean(&self) -> T {
        self.mean
    }

    /// The distribution's standard deviation.
    pub fn stddev(&self) -> T {
        self.stddev
    }

    /// `normal_distribution::operator()(g)`.
    pub fn sample(&mut self, g: &mut Mt19937) -> T {
        let up = if self.cached_is_hot {
            self.cached_is_hot = false;
            self.cached
        } else {
            let (mut u, mut v, mut s);
            loop {
                u = uniform_real(g, -T::ONE, T::ONE);
                v = uniform_real(g, -T::ONE, T::ONE);
                s = u * u + v * v;
                if !(s > T::ONE || s == T::ZERO) {
                    break;
                }
            }
            let fp = (T::from_i32(-2) * s.ln() / s).sqrt();
            self.cached = v * fp;
            self.cached_is_hot = true;
            u * fp
        };
        up * self.stddev + self.mean
    }
}
