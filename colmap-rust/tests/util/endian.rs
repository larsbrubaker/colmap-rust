// Port of COLMAP's src/colmap/util/endian_test.cc, 1:1 (Suite_Name -> suite_name), testing
// colmap_rust::util::endian. Tier A (exact). `std::stringstream` is a `Vec<u8>` written and
// then read through a `Cursor`.
//
// The random values: COLMAP draws them with std::default_random_engine (libc++'s
// minstd_rand0) through uniform_int/real_distribution over the type's full range. Every
// assertion is a round trip (x == convert(convert(x))), which holds for any value, so the
// draws here come from minstd_rand0 as raw bit patterns (all integer values, and every
// non-NaN float/double bit pattern, a superset of the distribution's range) instead of the
// libc++ distributions, which are not this module's code.

use std::io::Cursor;

use colmap_rust::util::endian::*;

#[test]
fn reverse_bytes_nominal() {
    for i in 0..256usize {
        assert_eq!(reverse_bytes(i as i8), i as i8);
        assert_eq!(reverse_bytes(i as u8), i as u8);
    }

    assert_eq!(reverse_bytes::<i16>(0), 0);
    assert_eq!(reverse_bytes::<i16>(1), 256);
    assert_eq!(reverse_bytes::<i16>(2), 512);
    assert_eq!(reverse_bytes::<i16>(3), 768);
    assert_eq!(reverse_bytes::<i16>(256), 1);
    assert_eq!(reverse_bytes::<i16>(512), 2);
    assert_eq!(reverse_bytes::<i16>(768), 3);

    assert_eq!(reverse_bytes::<u16>(0), 0);
    assert_eq!(reverse_bytes::<u16>(1), 256);
    assert_eq!(reverse_bytes::<u16>(2), 512);
    assert_eq!(reverse_bytes::<u16>(3), 768);
    assert_eq!(reverse_bytes::<u16>(256), 1);
    assert_eq!(reverse_bytes::<u16>(512), 2);
    assert_eq!(reverse_bytes::<u16>(768), 3);

    assert_eq!(reverse_bytes::<i32>(0), 0);
    assert_eq!(reverse_bytes::<i32>(1), 16777216);
    assert_eq!(reverse_bytes::<i32>(2), 33554432);
    assert_eq!(reverse_bytes::<i32>(3), 50331648);
    assert_eq!(reverse_bytes::<i32>(16777216), 1);
    assert_eq!(reverse_bytes::<i32>(33554432), 2);
    assert_eq!(reverse_bytes::<i32>(50331648), 3);

    assert_eq!(reverse_bytes::<u32>(0), 0);
    assert_eq!(reverse_bytes::<u32>(1), 16777216);
    assert_eq!(reverse_bytes::<u32>(2), 33554432);
    assert_eq!(reverse_bytes::<u32>(3), 50331648);
    assert_eq!(reverse_bytes::<u32>(16777216), 1);
    assert_eq!(reverse_bytes::<u32>(33554432), 2);
    assert_eq!(reverse_bytes::<u32>(50331648), 3);

    assert_eq!(reverse_bytes::<i64>(0), 0);
    assert_eq!(reverse_bytes::<i64>(1), 72057594037927936);
    assert_eq!(reverse_bytes::<i64>(2), 144115188075855872);
    assert_eq!(reverse_bytes::<i64>(3), 216172782113783808);
    assert_eq!(reverse_bytes::<i64>(72057594037927936), 1);
    assert_eq!(reverse_bytes::<i64>(144115188075855872), 2);
    assert_eq!(reverse_bytes::<i64>(216172782113783808), 3);

    assert_eq!(reverse_bytes::<u64>(0), 0);
    assert_eq!(reverse_bytes::<u64>(1), 72057594037927936);
    assert_eq!(reverse_bytes::<u64>(2), 144115188075855872);
    assert_eq!(reverse_bytes::<u64>(3), 216172782113783808);
    assert_eq!(reverse_bytes::<u64>(72057594037927936), 1);
    assert_eq!(reverse_bytes::<u64>(144115188075855872), 2);
    assert_eq!(reverse_bytes::<u64>(216172782113783808), 3);
}

#[test]
fn is_little_big_endian_nominal() {
    assert_ne!(is_little_endian(), is_big_endian());
}

/// libc++'s `std::default_random_engine` (minstd_rand0), default seed 1.
struct MinstdRand0(u64);

impl MinstdRand0 {
    fn new() -> Self {
        Self(1)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0 * 16807 % 2_147_483_647;
        self.0
    }

    /// 64 random bits from three 31-bit draws.
    fn bits(&mut self) -> u64 {
        (self.next() << 33) ^ (self.next() << 16) ^ self.next()
    }
}

/// A type the round-trip tests draw: every bit pattern, except NaN for the floats (NaN
/// never compares equal, and the libc++ distributions never return it).
trait Draw: EndianValue + PartialEq + std::fmt::Debug + Default {
    fn draw(prng: &mut MinstdRand0) -> Self;
}

macro_rules! impl_draw_int {
    ($($t:ty),*) => {$(
        impl Draw for $t {
            fn draw(prng: &mut MinstdRand0) -> Self {
                prng.bits() as $t
            }
        }
    )*};
}
impl_draw_int!(i8, u8, i16, u16, i32, u32, i64, u64);

impl Draw for f32 {
    fn draw(prng: &mut MinstdRand0) -> Self {
        loop {
            let x = f32::from_bits(prng.bits() as u32);
            if !x.is_nan() {
                return x;
            }
        }
    }
}

impl Draw for f64 {
    fn draw(prng: &mut MinstdRand0) -> Self {
        loop {
            let x = f64::from_bits(prng.bits());
            if !x.is_nan() {
                return x;
            }
        }
    }
}

fn test_native_to_little_big_endian<T: Draw>() {
    let mut prng = MinstdRand0::new();
    for _ in 0..100 {
        let x = T::draw(&mut prng);
        assert_eq!(little_endian_to_native(native_to_little_endian(x)), x);
        assert_eq!(big_endian_to_native(native_to_big_endian(x)), x);
        assert_eq!(native_to_little_endian(little_endian_to_native(x)), x);
        assert_eq!(native_to_big_endian(big_endian_to_native(x)), x);
    }
}

#[test]
fn native_to_litte_big_endian_nominal() {
    test_native_to_little_big_endian::<i8>();
    test_native_to_little_big_endian::<i16>();
    test_native_to_little_big_endian::<i32>();
    test_native_to_little_big_endian::<i64>();
    test_native_to_little_big_endian::<u8>();
    test_native_to_little_big_endian::<u16>();
    test_native_to_little_big_endian::<u32>();
    test_native_to_little_big_endian::<u64>();
    test_native_to_little_big_endian::<f32>();
    test_native_to_little_big_endian::<f64>();
}

fn test_read_write_binary_little_endian<T: Draw + Clone>() {
    let mut prng = MinstdRand0::new();
    for _ in 0..100 {
        let mut file = Vec::new();
        let orig_value = T::draw(&mut prng);
        write_binary_little_endian(&mut file, orig_value).unwrap();
        let read_value: T = read_binary_little_endian(&mut Cursor::new(&file)).unwrap();
        assert_eq!(orig_value, read_value);

        let mut file_vector = Vec::new();
        let orig_vector: Vec<T> = (0..100).map(|_| T::draw(&mut prng)).collect();
        write_binary_little_endian_slice(&mut file_vector, &orig_vector).unwrap();
        let mut read_vector = vec![T::default(); orig_vector.len()];
        read_binary_little_endian_into(&mut Cursor::new(&file_vector), &mut read_vector).unwrap();
        assert_eq!(orig_vector, read_vector);
    }
}

#[test]
fn read_write_binary_little_endian_nominal() {
    test_read_write_binary_little_endian::<i8>();
    test_read_write_binary_little_endian::<i16>();
    test_read_write_binary_little_endian::<i32>();
    test_read_write_binary_little_endian::<i64>();
    test_read_write_binary_little_endian::<u8>();
    test_read_write_binary_little_endian::<u16>();
    test_read_write_binary_little_endian::<u32>();
    test_read_write_binary_little_endian::<u64>();
    test_read_write_binary_little_endian::<f32>();
    test_read_write_binary_little_endian::<f64>();
}

// Rust-only: the bytes on the wire are little-endian whatever the host (COLMAP's binary
// formats depend on it), and a short stream is an error (docs/CPP_DIVERGENCES.md entry 61).
#[test]
fn rust_only_little_endian_wire_bytes_and_short_read() {
    let mut file = Vec::new();
    write_binary_little_endian(&mut file, 0x0102_0304u32).unwrap();
    write_binary_little_endian(&mut file, 1.0f64).unwrap();
    write_binary_little_endian(&mut file, -2i16).unwrap();
    assert_eq!(file, [4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0xF0, 0x3F, 0xFE, 0xFF]);
    let err = read_binary_little_endian::<u64, _>(&mut Cursor::new(&file[..7])).unwrap_err();
    assert_eq!(err.kind(), std::io::ErrorKind::UnexpectedEof);
}
