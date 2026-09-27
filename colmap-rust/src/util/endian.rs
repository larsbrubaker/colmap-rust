//! Port of COLMAP's `src/colmap/util/endian.h/.cc`: byte-order conversion and the
//! little-endian binary stream helpers that COLMAP's binary file formats (`cameras.bin`,
//! `images.bin`, `points3D.bin`, depth/normal maps) are read and written with.
//! Tests: `tests/util/endian.rs` (`endian_test.cc`).
//!
//! Rust's `to_le_bytes`/`from_le_bytes` family replaces COLMAP's `reinterpret_cast` +
//! `std::reverse`; [`EndianValue`] gives the integer and float types one generic interface
//! so the helpers keep COLMAP's template shape. `std::istream`/`std::ostream` become
//! `std::io::Read`/`Write`. Tier A (exact): pure byte shuffling.
//!
//! One behavior differs on purpose: COLMAP's `ReadBinaryLittleEndian` on a stream that ends
//! early returns an unspecified value (the stream's failbit is set and the read is not
//! checked there); here the read returns the `std::io::Error` (`UnexpectedEof`).
//! `docs/CPP_DIVERGENCES.md` entry 61.

use std::io::{self, Read, Write};

mod sealed {
    /// Seals [`super::EndianValue`]: its byte-slice methods index by `SIZE` and would panic
    /// for a foreign impl whose `SIZE` exceeds 8, so only the types below implement it.
    pub trait Sealed {}
    impl Sealed for i8 {}
    impl Sealed for u8 {}
    impl Sealed for i16 {}
    impl Sealed for u16 {}
    impl Sealed for i32 {}
    impl Sealed for u32 {}
    impl Sealed for i64 {}
    impl Sealed for u64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// A plain value COLMAP reads and writes byte for byte: the fixed-size integers and
/// floating-point types. Sealed: implemented only for i8..u64, f32 and f64.
pub trait EndianValue: Copy + sealed::Sealed {
    /// `sizeof(T)`.
    const SIZE: usize;
    /// Port of `ReverseBytes<T>`.
    fn reverse_bytes(self) -> Self;
    /// The value's bytes in little-endian order (the first `SIZE` of the array).
    fn to_le_byte_array(self) -> [u8; 8];
    /// The value from `SIZE` little-endian bytes.
    fn from_le_byte_slice(bytes: &[u8]) -> Self;
}

macro_rules! impl_endian_value {
    ($($t:ty),*) => {$(
        impl EndianValue for $t {
            const SIZE: usize = std::mem::size_of::<$t>();

            fn reverse_bytes(self) -> Self {
                let mut array = self.to_ne_bytes();
                array.reverse();
                <$t>::from_ne_bytes(array)
            }

            fn to_le_byte_array(self) -> [u8; 8] {
                let mut out = [0u8; 8];
                out[..Self::SIZE].copy_from_slice(&self.to_le_bytes());
                out
            }

            fn from_le_byte_slice(bytes: &[u8]) -> Self {
                let mut array = [0u8; std::mem::size_of::<$t>()];
                array.copy_from_slice(&bytes[..Self::SIZE]);
                <$t>::from_le_bytes(array)
            }
        }
    )*};
}

impl_endian_value!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64);

/// Port of `ReverseBytes<T>`.
pub fn reverse_bytes<T: EndianValue>(data: T) -> T {
    data.reverse_bytes()
}

/// Port of `IsLittleEndian`. COLMAP rejects word-swapped ("PDP") byte orders, which Rust
/// does not target at all.
pub fn is_little_endian() -> bool {
    cfg!(target_endian = "little")
}

/// Port of `IsBigEndian`.
pub fn is_big_endian() -> bool {
    cfg!(target_endian = "big")
}

/// Port of `LittleEndianToNative<T>`.
pub fn little_endian_to_native<T: EndianValue>(x: T) -> T {
    if is_little_endian() {
        x
    } else {
        x.reverse_bytes()
    }
}

/// Port of `BigEndianToNative<T>`.
pub fn big_endian_to_native<T: EndianValue>(x: T) -> T {
    if is_big_endian() {
        x
    } else {
        x.reverse_bytes()
    }
}

/// Port of `NativeToLittleEndian<T>`.
pub fn native_to_little_endian<T: EndianValue>(x: T) -> T {
    if is_little_endian() {
        x
    } else {
        x.reverse_bytes()
    }
}

/// Port of `NativeToBigEndian<T>`.
pub fn native_to_big_endian<T: EndianValue>(x: T) -> T {
    if is_big_endian() {
        x
    } else {
        x.reverse_bytes()
    }
}

/// Port of `ReadBinaryLittleEndian<T>(stream)`: reads `sizeof(T)` little-endian bytes.
/// Errors (`UnexpectedEof`) if the stream ends first (entry 61).
pub fn read_binary_little_endian<T: EndianValue, R: Read + ?Sized>(
    stream: &mut R,
) -> io::Result<T> {
    let mut buffer = [0u8; 8];
    stream.read_exact(&mut buffer[..T::SIZE])?;
    Ok(T::from_le_byte_slice(&buffer[..T::SIZE]))
}

/// Port of `ReadBinaryLittleEndian<T>(stream, &data)`: fills every element of `data`.
pub fn read_binary_little_endian_into<T: EndianValue, R: Read + ?Sized>(
    stream: &mut R,
    data: &mut [T],
) -> io::Result<()> {
    for value in data.iter_mut() {
        *value = read_binary_little_endian(stream)?;
    }
    Ok(())
}

/// Port of `WriteBinaryLittleEndian<T>(stream, data)`.
pub fn write_binary_little_endian<T: EndianValue, W: Write + ?Sized>(
    stream: &mut W,
    data: T,
) -> io::Result<()> {
    stream.write_all(&data.to_le_byte_array()[..T::SIZE])
}

/// Port of `WriteBinaryLittleEndian<T>(stream, span<const T>)`.
pub fn write_binary_little_endian_slice<T: EndianValue, W: Write + ?Sized>(
    stream: &mut W,
    data: &[T],
) -> io::Result<()> {
    for &value in data {
        write_binary_little_endian(stream, value)?;
    }
    Ok(())
}
