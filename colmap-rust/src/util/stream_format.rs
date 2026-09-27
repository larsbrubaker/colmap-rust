//! How a C++ `std::ostream` writes a double in its default float field (neither
//! `std::fixed` nor `std::scientific`), which is printf's `%.<precision>g`.
//!
//! COLMAP's check messages (`THROW_CHECK_GT(x, 0)` prints `(x vs. 0)` through an ostream at
//! precision 6), its `operator<<` for `Rigid3d`/`Sim3d` and its text reconstruction writers
//! (precision 17) all go through it, and their output is part of COLMAP's tested contract.
//! Port of colmap-sharp's `ColmapSharp/Util/CppStreamFormat.cs`, which was written from the
//! C standard's definition of `%g` (C11 7.21.6.1), not from any library.
//!
//! Tier A (exact): the digits are rounded from the double's exact decimal expansion,
//! round-half-even on exact ties, which is what glibc's and Apple's printf do in the default
//! rounding mode. Rust's `{:.N e}` formatting is exact for every precision (Grisu with a
//! Dragon4 fallback) and breaks exact ties to even, so only the `%g` layout is done here.
//! Tests: `tests/util/stream_format.rs`.

/// `std::ostream`'s default precision.
pub const DEFAULT_PRECISION: usize = 6;

/// `stream << value` with `stream.precision(precision)` and the default float field, i.e.
/// printf's `%.{precision}g`: at most `precision` significant digits, trailing zeros removed,
/// scientific notation when the decimal exponent is below -4 or at least the precision.
pub fn format_double(value: f64, precision: usize) -> String {
    if value.is_nan() {
        // libc++ on macOS (the pycolmap oracle's) prints "nan" for every NaN, whatever its
        // sign bit.
        return "nan".to_string();
    }
    if value.is_infinite() {
        return if value < 0.0 { "-inf" } else { "inf" }.to_string();
    }
    let negative = value.is_sign_negative();
    if value == 0.0 {
        return if negative { "-0" } else { "0" }.to_string();
    }

    // %g treats a precision of 0 as 1. Beyond 767 significant digits a double's exact
    // expansion has only zeros left, which %g strips anyway.
    let significant = precision.clamp(1, 800);
    let scientific = format!("{:.*e}", significant - 1, value.abs());
    let (mantissa, exponent) = scientific
        .split_once('e')
        .expect("Rust's {:e} output always contains 'e'");
    let exponent: i32 = exponent
        .parse()
        .expect("Rust's {:e} exponent is a plain integer");
    let mut digits: Vec<u8> = mantissa.bytes().filter(|b| *b != b'.').collect();
    while digits.len() > 1 && digits[digits.len() - 1] == b'0' {
        digits.pop();
    }
    layout(negative, &digits, exponent, significant)
}

/// `%g`'s layout of the rounded `digits` (trailing zeros already removed) whose first digit
/// has decimal exponent `exponent`.
fn layout(negative: bool, digits: &[u8], exponent: i32, significant: usize) -> String {
    let mut out = String::with_capacity(digits.len() + 16);
    if negative {
        out.push('-');
    }
    let push_digits = |out: &mut String, d: &[u8]| {
        out.extend(d.iter().map(|&b| b as char));
    };

    if exponent < -4 || exponent >= significant as i32 {
        push_digits(&mut out, &digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            push_digits(&mut out, &digits[1..]);
        }
        out.push('e');
        out.push(if exponent < 0 { '-' } else { '+' });
        // At least two exponent digits, as printf writes them.
        out.push_str(&format!("{:02}", exponent.unsigned_abs()));
    } else if exponent < 0 {
        out.push_str("0.");
        for _ in 0..(-exponent - 1) {
            out.push('0');
        }
        push_digits(&mut out, digits);
    } else {
        // The integer part keeps its zeros; only fraction zeros were stripped.
        let int_len = exponent as usize + 1;
        for i in 0..int_len {
            out.push(digits.get(i).map_or('0', |&b| b as char));
        }
        if digits.len() > int_len {
            out.push('.');
            push_digits(&mut out, &digits[int_len..]);
        }
    }
    out
}
