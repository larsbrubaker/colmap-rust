//! The subset of C++ `std::istream` text parsing that COLMAP's text readers use —
//! `std::getline` over a stream, `StringTrim`, and `line_stream >> value` in the classic
//! ("C") locale — so `scene::reconstruction_io_text` reads its files token by token the way
//! `reconstruction_io_text.cc` does. Port of colmap-sharp's `Util/CppLineTokens.cs` (written
//! there, not a port of libc++). Tests: `tests/scene/rust_only_reconstruction_io.rs`.
//!
//! A value is one whitespace-separated token; a failed extraction is `None`, which is where
//! C++ sets `failbit`. Spellings follow libc++'s `num_get`: an unsigned integer accepts an
//! optional '-' and negates with wrap-around ("-1" reads as the maximum), and a magnitude
//! above the type's maximum fails; a double is parsed by
//! [`crate::util::string::parse_double_token`] (decimal and exponent notation; entry 60), and a
//! float by `parse_float_token`, which accepts the same spellings and rounds once to `f32`.
//! A token is taken whole, so "12abc" fails where libc++ would read 12 into an integer and
//! leave "abc" for the next extraction (docs/CPP_DIVERGENCES.md entry 241).

use std::io::Read;

use crate::util::string::{parse_double_token, parse_float_token};

/// The characters C's `isspace` accepts in the "C" locale: what separates tokens.
const C_ISSPACE: &[char] = &[' ', '\t', '\n', '\u{b}', '\u{c}', '\r'];

/// `std::getline` over a whole stream: lines split at '\n' only (a '\r' stays in the line,
/// for `StringTrim` to remove), and no empty final line after a trailing '\n'. Each line is
/// raw bytes; the caller decodes it.
pub fn read_lines<R: Read + ?Sized>(stream: &mut R) -> std::io::Result<Vec<Vec<u8>>> {
    let mut bytes = Vec::new();
    stream.read_to_end(&mut bytes)?;
    let mut lines: Vec<Vec<u8>> = bytes.split(|&b| b == b'\n').map(<[u8]>::to_vec).collect();
    if bytes.is_empty() || bytes.last() == Some(&b'\n') {
        // `split` yields an empty piece after the final '\n' (or for no bytes at all) that
        // `std::getline` does not.
        lines.pop();
    }
    Ok(lines)
}

/// Whitespace-separated tokens of one line, read the way `istream >>` reads them.
#[derive(Debug, Clone)]
pub struct LineTokens<'a> {
    tokens: Vec<&'a str>,
    next: usize,
}

impl<'a> LineTokens<'a> {
    /// Splits `line` at C `isspace` characters.
    pub fn new(line: &'a str) -> Self {
        Self {
            tokens: line.split(C_ISSPACE).filter(|t| !t.is_empty()).collect(),
            next: 0,
        }
    }

    /// Whether a token is left.
    pub fn has_token(&self) -> bool {
        self.next < self.tokens.len()
    }

    /// Takes the next token if `parse` accepts it; a rejected token stays (the stream is in
    /// its failed state then, and every caller stops reading).
    fn take<T>(&mut self, parse: impl FnOnce(&str) -> Option<T>) -> Option<T> {
        let value = parse(self.tokens.get(self.next)?)?;
        self.next += 1;
        Some(value)
    }

    /// `stream >> std::string`: the next token.
    pub fn read_string(&mut self) -> Option<String> {
        self.take(|t| Some(t.to_owned()))
    }

    /// `stream >> double`.
    pub fn read_f64(&mut self) -> Option<f64> {
        self.take(parse_double_token)
    }

    /// `stream >> float`.
    pub fn read_f32(&mut self) -> Option<f32> {
        self.take(parse_float_token)
    }

    /// `stream >> uint32_t`.
    pub fn read_u32(&mut self) -> Option<u32> {
        // The u64 result is at most u32::MAX here, so the cast is exact.
        self.take(|t| parse_unsigned(t, u64::from(u32::MAX)).map(|v| v as u32))
    }

    /// `stream >> uint64_t`.
    pub fn read_u64(&mut self) -> Option<u64> {
        self.take(|t| parse_unsigned(t, u64::MAX))
    }

    /// `stream >> int`.
    pub fn read_i32(&mut self) -> Option<i32> {
        self.take(|t| parse_signed(t)?.try_into().ok())
    }

    /// `stream >> int64_t`.
    pub fn read_i64(&mut self) -> Option<i64> {
        self.take(parse_signed)
    }
}

/// libc++'s unsigned `num_get`: an optional sign, decimal digits, at most `max`; a '-'
/// negates with wrap-around in the type's width.
fn parse_unsigned(token: &str, max: u64) -> Option<u64> {
    let (negative, digits) = match token.as_bytes().first()? {
        b'-' => (true, &token[1..]),
        b'+' => (false, &token[1..]),
        _ => (false, token),
    };
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let magnitude: u64 = digits.parse().ok()?;
    if magnitude > max {
        return None;
    }
    if negative {
        // `-magnitude` in the unsigned type of width `max`.
        Some(magnitude.wrapping_neg() & max)
    } else {
        Some(magnitude)
    }
}

/// libc++'s signed `num_get`: an optional sign and decimal digits within `i64`.
fn parse_signed(token: &str) -> Option<i64> {
    let digits = token.strip_prefix(['-', '+']).unwrap_or(token);
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    token.parse().ok()
}
