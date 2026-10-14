// Port of .NET's seeded `System.Random` (test support, not a test): colmap-sharp's C#-only
// tests draw their random inputs from `new Random(seed)`, so the Rust ports of those tests use
// this to build the same matrices. Covers `NextDouble()`, `Next(maxValue)` and
// `Next(minValue, maxValue)`, the calls the ported tests make.
//
// Include it from a test binary with `#[path = "support/net_random.rs"] mod net_random;`.

#![allow(dead_code)]

/// .NET's `new Random(seed)` (the seeded, Knuth subtractive generator .NET keeps for
/// compatibility) and its `NextDouble()`.
pub struct NetRandom {
    seed_array: [i32; 56],
    inext: usize,
    inextp: usize,
}

impl NetRandom {
    pub fn new(seed: i32) -> Self {
        const MSEED: i32 = 161_803_398;
        let mut seed_array = [0_i32; 56];
        let subtraction = if seed == i32::MIN {
            i32::MAX
        } else {
            seed.abs()
        };
        let mut mj = MSEED - subtraction;
        seed_array[55] = mj;
        let mut mk = 1_i32;
        let mut ii = 0;
        for _ in 1..55 {
            ii += 21;
            if ii >= 55 {
                ii -= 55;
            }
            seed_array[ii] = mk;
            mk = mj - mk;
            if mk < 0 {
                mk += i32::MAX;
            }
            mj = seed_array[ii];
        }
        for _ in 1..5 {
            for i in 1..56 {
                let mut n = i + 30;
                if n >= 55 {
                    n -= 55;
                }
                seed_array[i] = seed_array[i].wrapping_sub(seed_array[1 + n]);
                if seed_array[i] < 0 {
                    seed_array[i] += i32::MAX;
                }
            }
        }
        Self {
            seed_array,
            inext: 0,
            inextp: 21,
        }
    }

    pub fn next_double(&mut self) -> f64 {
        let mut next = self.inext + 1;
        if next >= 56 {
            next = 1;
        }
        let mut nextp = self.inextp + 1;
        if nextp >= 56 {
            nextp = 1;
        }
        let mut value = self.seed_array[next].wrapping_sub(self.seed_array[nextp]);
        if value == i32::MAX {
            value -= 1;
        }
        if value < 0 {
            value += i32::MAX;
        }
        self.seed_array[next] = value;
        self.inext = next;
        self.inextp = nextp;
        value as f64 * (1.0 / i32::MAX as f64)
    }

    /// `Next(maxValue)`: `(int)(Sample() * maxValue)`, truncating.
    pub fn next(&mut self, max_value: i32) -> i32 {
        (self.next_double() * max_value as f64) as i32
    }

    /// `Next(minValue, maxValue)` for a range that fits an `int` (the legacy seeded
    /// generator's `(int)(Sample() * range) + minValue`).
    pub fn next_range(&mut self, min_value: i32, max_value: i32) -> i32 {
        let range = max_value as i64 - min_value as i64;
        assert!(range <= i32::MAX as i64, "only int-sized ranges are ported");
        (self.next_double() * range as f64) as i32 + min_value
    }
}
