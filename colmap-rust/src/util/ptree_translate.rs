//! boost::property_tree's value translators for `bool` and `double`, as COLMAP's rig config
//! reader (`scene/rig.cc`) uses them through `ptree::get_value<T>`: written here, not ported.
//! Port of colmap-sharp's `ColmapSharp/Scene/RigConfig.Translators.cs`, whose rules were
//! pinned against a libc++ harness (Apple clang): ptree's `stream_translator` is
//! `istringstream >> value` on the node's text, so what it accepts is libc++'s `num_get`
//! (the macOS pycolmap wheel links libc++). Used by [`crate::scene::rig::read_rig_config`]
//! on [`crate::util::json::JsonValue::ptree_data`]. Tests: `tests/scene/rust_only_rig.rs`
//! (colmap-sharp's libc++-pinned cases).
//!
//! The translator skips leading white space, reads the value, skips trailing white space, and
//! fails unless the whole text was used.
//! - bool: `num_get` reads a decimal long (optional sign, then digits). 0 is false, 1 is
//!   true, and anything else (no digits, out of range, another value) fails. On that failure
//!   ptree retries from where `num_get` stopped with `boolalpha`, which matches the words
//!   "true" or "false" exactly. So "01", "+1" and even "2true" read true, while "TRUE" and
//!   "-1" fail.
//! - double: `strtod` on the characters `num_get` collected, failing on `ERANGE`. That gives
//!   decimal and hex floats ("0x1p3"), and rejects inf, nan, overflow, and results that are
//!   below `DBL_MIN` after rounding and inexact (IEEE underflow). An exact subnormal (e.g.
//!   "0x1p-1074") reads fine.
//!
//! Tier A: the decimal path is Rust's correctly rounded `str::parse::<f64>` (as `strtod`),
//! the hex path rounds to nearest-even by hand, and the exactness test of a decimal subnormal
//! uses the small exact integer arithmetic at the end of this file.

use crate::util::check::{ColmapError, ErrorKind, Result};

/// The C locale's `isspace`, which istream's `skipws` and `std::ws` use.
fn is_c_space(c: u8) -> bool {
    matches!(c, b' ' | b'\t' | b'\n' | 0x0b | 0x0c | b'\r')
}

fn skip_c_space(text: &[u8], mut pos: usize) -> usize {
    while pos < text.len() && is_c_space(text[pos]) {
        pos += 1;
    }
    pos
}

/// Boost's `ptree_bad_data` message.
fn conversion_failed(type_name: &str, text: &str) -> ColmapError {
    ColmapError::new(
        ErrorKind::RuntimeError,
        format!("conversion of data to type \"{type_name}\" failed ({text})"),
    )
}

/// `ptree::get_value<bool>()` on the node text (see the module docs).
pub fn ptree_get_bool(text: &str) -> Result<bool> {
    let bytes = text.as_bytes();
    let mut pos = skip_c_space(bytes, 0);

    // num_get<bool> without boolalpha: a decimal long, which must be 0 or 1.
    let start = pos;
    if pos < bytes.len() && matches!(bytes[pos], b'+' | b'-') {
        pos += 1;
    }
    let digits_start = pos;
    while pos < bytes.len() && bytes[pos].is_ascii_digit() {
        pos += 1;
    }
    let mut value = None;
    if pos > digits_start {
        // A long that overflows fails, as num_get does (ERANGE).
        match text[start..pos].parse::<i64>() {
            Ok(0) => value = Some(false),
            Ok(1) => value = Some(true),
            _ => {}
        }
    }

    let value = match value {
        Some(value) => value,
        None => {
            // Retry in word form from where the number read stopped.
            pos = skip_c_space(bytes, pos);
            if bytes[pos..].starts_with(b"true") {
                pos += 4;
                true
            } else if bytes[pos..].starts_with(b"false") {
                pos += 5;
                false
            } else {
                return Err(conversion_failed("bool", text));
            }
        }
    };

    if skip_c_space(bytes, pos) == bytes.len() {
        Ok(value)
    } else {
        Err(conversion_failed("bool", text))
    }
}

/// `ptree::get_value<double>()` on the node text (see the module docs).
pub fn ptree_get_double(text: &str) -> Result<f64> {
    let bytes = text.as_bytes();
    let begin = skip_c_space(bytes, 0);
    let mut end = bytes.len();
    while end > begin && is_c_space(bytes[end - 1]) {
        end -= 1;
    }
    let token = &text[begin..end];
    let value = if is_hex_prefix(token.as_bytes()) {
        parse_hex_float(token)
    } else {
        parse_decimal_float(token)
    };
    value.ok_or_else(|| conversion_failed("double", text))
}

fn is_hex_prefix(token: &[u8]) -> bool {
    let i = usize::from(matches!(token.first(), Some(b'+' | b'-')));
    token.len() > i + 1 && token[i] == b'0' && matches!(token[i + 1], b'x' | b'X')
}

/// The pieces of `[sign] mantissa [exponent]`: the mantissa digits (at most one '.' among
/// them, at least one digit), how many came after the '.', and the exponent (saturated far
/// outside any double's range, which keeps every comparison below the same).
struct ScannedFloat {
    negative: bool,
    mantissa_digits: Vec<u8>,
    fraction_digits: i64,
    exponent: i64,
}

/// Scans the float after the sign (and `0x`) at `pos`. `None` unless the token matches in
/// full: digits with at most one '.', at least one digit, then optionally the exponent marker,
/// an optional sign and decimal digits.
fn scan_float(
    token: &str,
    mut pos: usize,
    is_digit: fn(u8) -> bool,
    exponent_marker: u8,
) -> Option<ScannedFloat> {
    let bytes = token.as_bytes();
    let negative = bytes.first() == Some(&b'-');
    let mut mantissa_digits = Vec::new();
    let mut fraction_digits = 0i64;
    let mut seen_point = false;
    while pos < bytes.len() {
        let c = bytes[pos];
        if is_digit(c) {
            mantissa_digits.push(c);
            fraction_digits += i64::from(seen_point);
        } else if c == b'.' && !seen_point {
            seen_point = true;
        } else {
            break;
        }
        pos += 1;
    }
    if mantissa_digits.is_empty() {
        return None;
    }
    let mut exponent = 0i64;
    if pos < bytes.len() {
        if bytes[pos].to_ascii_lowercase() != exponent_marker {
            return None;
        }
        pos += 1;
        let exponent_negative = bytes.get(pos) == Some(&b'-');
        if matches!(bytes.get(pos), Some(b'+' | b'-')) {
            pos += 1;
        }
        let exponent_digits = pos;
        while pos < bytes.len() && bytes[pos].is_ascii_digit() {
            // Saturate at 10^15: far beyond any double, so the result is the same.
            exponent = (exponent * 10 + i64::from(bytes[pos] - b'0')).min(1_000_000_000_000_000);
            pos += 1;
        }
        if pos == exponent_digits || pos != bytes.len() {
            return None;
        }
        if exponent_negative {
            exponent = -exponent;
        }
    }
    Some(ScannedFloat {
        negative,
        mantissa_digits,
        fraction_digits,
        exponent,
    })
}

/// `DBL_MIN`, the smallest normal double.
const DBL_MIN: f64 = 2.2250738585072014e-308;

fn parse_decimal_float(token: &str) -> Option<f64> {
    let pos = usize::from(matches!(token.as_bytes().first(), Some(b'+' | b'-')));
    let scanned = scan_float(token, pos, |c| c.is_ascii_digit(), b'e')?;
    // Rust's parse is correctly rounded (round half to even) like strtod.
    let value = token.parse::<f64>().ok()?;
    if value.is_infinite() {
        return None;
    }
    let magnitude = value.abs();
    if magnitude < DBL_MIN {
        // Below DBL_MIN: strtod reports ERANGE unless the result is exact.
        let decimal_exponent = scanned.exponent - scanned.fraction_digits;
        if !is_exact_subnormal(&scanned.mantissa_digits, decimal_exponent, magnitude) {
            return None;
        }
    }
    Some(value)
}

/// Whether `mantissa * 10^exponent` (decimal digits) equals the subnormal (or zero)
/// `magnitude` exactly.
fn is_exact_subnormal(mantissa_digits: &[u8], exponent: i64, magnitude: f64) -> bool {
    // Strip leading zeros, and trailing zeros into the exponent.
    let first = mantissa_digits.iter().position(|&c| c != b'0');
    let Some(first) = first else {
        return true; // A zero mantissa reads 0 exactly.
    };
    let last = mantissa_digits
        .iter()
        .rposition(|&c| c != b'0')
        .unwrap_or(first);
    let digits = &mantissa_digits[first..=last];
    let exponent = exponent + (mantissa_digits.len() - 1 - last) as i64;
    if magnitude == 0.0 {
        return false; // A non-zero mantissa that rounded to 0.
    }
    // magnitude = units * 2^-1074 with units < 2^52. units * 2^-1074 has at most 1074
    // decimal fraction digits, the last one non-zero, so an exact match needs
    // -1074 <= exponent < 0: mantissa * 2^1074 == units * 10^-exponent.
    if !(-1074..0).contains(&exponent) {
        return false;
    }
    let units = magnitude.to_bits();
    let mut lhs = BigUint::from_decimal(digits);
    lhs.shl(1074);
    let mut rhs = BigUint::from_u64(units);
    for _ in 0..-exponent {
        rhs.mul_small(10);
    }
    lhs == rhs
}

fn parse_hex_float(token: &str) -> Option<f64> {
    let pos = usize::from(matches!(token.as_bytes().first(), Some(b'+' | b'-'))) + 2;
    let scanned = scan_float(token, pos, |c| c.is_ascii_hexdigit(), b'p')?;
    // The mantissa's bits, most significant first, without leading zeros.
    let mut bits: Vec<bool> = Vec::with_capacity(scanned.mantissa_digits.len() * 4);
    for &c in &scanned.mantissa_digits {
        let nibble = (c as char).to_digit(16).unwrap_or(0);
        for shift in (0..4).rev() {
            let bit = (nibble >> shift) & 1 == 1;
            if bit || !bits.is_empty() {
                bits.push(bit);
            }
        }
    }
    if bits.is_empty() {
        return Some(if scanned.negative { -0.0 } else { 0.0 });
    }

    // value = mantissa * 2^binary_exponent; round it to the double grid (53 bits, or the
    // subnormal grid 2^-1074 below DBL_MIN) with round half to even.
    let bit_length = bits.len() as i64;
    let binary_exponent = scanned.exponent - 4 * scanned.fraction_digits;
    let top_exponent = binary_exponent + bit_length - 1;
    if top_exponent > 1023 {
        return None;
    }
    if top_exponent < -1100 {
        // Far below the smallest subnormal: underflows to zero, inexactly.
        return None;
    }
    let lsb_exponent = (top_exponent - 52).max(-1074);
    // How many low bits of the mantissa fall below the grid (negative: zeros to append).
    let shift = lsb_exponent - binary_exponent;
    let mut units: u64 = 0;
    let mut inexact = false;
    if shift <= 0 {
        for &bit in &bits {
            units = (units << 1) | u64::from(bit);
        }
        units <<= -shift;
    } else {
        let kept = (bit_length - shift).max(0) as usize;
        for &bit in &bits[..kept] {
            units = (units << 1) | u64::from(bit);
        }
        let dropped = &bits[kept..];
        // `dropped` is the remainder's top bits when `shift <= bit_length`; otherwise the
        // remainder sits below the half bit.
        let half_bit = shift <= bit_length && dropped[0];
        let below_half = if shift <= bit_length {
            dropped[1..].iter().any(|&bit| bit)
        } else {
            true
        };
        inexact = dropped.iter().any(|&bit| bit);
        if half_bit && (below_half || units & 1 == 1) {
            units += 1;
        }
    }

    let magnitude = scale_by_power_of_two(units, lsb_exponent);
    if magnitude.is_infinite() || (magnitude < DBL_MIN && inexact) {
        return None;
    }
    Some(if scanned.negative {
        -magnitude
    } else {
        magnitude
    })
}

/// `units * 2^exponent` for `units <= 2^53` and `-1074 <= exponent <= 971`, exactly (or
/// infinity when it overflows).
fn scale_by_power_of_two(units: u64, exponent: i64) -> f64 {
    if exponent == -1074 {
        // The subnormal grid: the bit pattern of `units` is `units * 2^-1074` (up to 2^53,
        // where it crosses into the normals with the same value).
        return f64::from_bits(units);
    }
    let power = if exponent >= -1022 {
        f64::from_bits(((exponent + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (exponent + 1074))
    };
    // Exact: both factors are exact and the product is a normal double (or overflows).
    units as f64 * power
}

/// The little exact unsigned integer arithmetic the subnormal test needs: 32-bit limbs,
/// least significant first, no leading zero limbs.
#[derive(PartialEq)]
struct BigUint {
    limbs: Vec<u32>,
}

impl BigUint {
    fn from_u64(value: u64) -> Self {
        let mut number = BigUint {
            limbs: vec![value as u32, (value >> 32) as u32],
        };
        number.trim();
        number
    }

    fn from_decimal(digits: &[u8]) -> Self {
        let mut number = BigUint { limbs: Vec::new() };
        for &digit in digits {
            number.mul_small(10);
            number.add_small(u32::from(digit - b'0'));
        }
        number
    }

    fn trim(&mut self) {
        while self.limbs.last() == Some(&0) {
            self.limbs.pop();
        }
    }

    fn mul_small(&mut self, factor: u32) {
        let mut carry = 0u64;
        for limb in &mut self.limbs {
            let product = u64::from(*limb) * u64::from(factor) + carry;
            *limb = product as u32;
            carry = product >> 32;
        }
        if carry != 0 {
            self.limbs.push(carry as u32);
        }
        self.trim();
    }

    fn add_small(&mut self, addend: u32) {
        let mut carry = u64::from(addend);
        for limb in &mut self.limbs {
            if carry == 0 {
                break;
            }
            let sum = u64::from(*limb) + carry;
            *limb = sum as u32;
            carry = sum >> 32;
        }
        if carry != 0 {
            self.limbs.push(carry as u32);
        }
    }

    fn shl(&mut self, bits: usize) {
        if self.limbs.is_empty() {
            return;
        }
        let (limb_shift, bit_shift) = (bits / 32, bits % 32);
        if bit_shift != 0 {
            let mut carry = 0u32;
            for limb in &mut self.limbs {
                let shifted = (u64::from(*limb) << bit_shift) | u64::from(carry);
                *limb = shifted as u32;
                carry = (shifted >> 32) as u32;
            }
            if carry != 0 {
                self.limbs.push(carry);
            }
        }
        self.limbs.splice(0..0, std::iter::repeat_n(0, limb_shift));
    }
}
