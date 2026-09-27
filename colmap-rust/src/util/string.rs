//! Port of COLMAP's `src/colmap/util/string.h/.cc`: the string helpers later code uses
//! (reconstruction text I/O, option parsing, CSV lists in `misc`).
//! Tests: `tests/util/string.rs` (`string_test.cc`).
//!
//! Replaced by Rust, not ported:
//! - `StringPrintf` (printf-style varargs) is `format!`: `%d`/`%s` are `{}`, `%.3f` is
//!   `{:.3}` (Rust's fixed formatting rounds from the exact binary value like printf), and
//!   `%g` is [`crate::util::stream_format::format_double`].
//! - `PlatformToUTF8` / `UTF8ToPlatform` are the identity on POSIX and code-page
//!   conversions on Windows; Rust strings are always UTF-8 and Rust's file APIs take them
//!   on every platform, so there is nothing to convert.
//!
//! Tier A (exact) for everything here. Whitespace for the trims is exactly COLMAP's
//! `IsNotWhiteSpace` set (' ', '\n', '\r', '\t'); case mapping is ASCII only, as `::tolower`
//! in the "C" locale.

use super::check::Result;

/// The characters C's `isspace` accepts in the "C" locale (what `operator>>` and
/// `std::stoi` skip).
pub(crate) const C_ISSPACE: [char; 6] = [' ', '\t', '\n', '\x0B', '\x0C', '\r'];

/// COLMAP's `IsNotWhiteSpace`, negated.
fn is_white_space(c: char) -> bool {
    matches!(c, ' ' | '\n' | '\r' | '\t')
}

/// Port of `StringReplace`: replaces every non-overlapping occurrence of `old_str`, left to
/// right. An empty `old_str` leaves the string unchanged (Rust's `str::replace` would insert
/// `new_str` between every character instead).
pub fn string_replace(s: &str, old_str: &str, new_str: &str) -> String {
    if old_str.is_empty() {
        return s.to_string();
    }
    // Same as COLMAP's find/replace loop, which resumes the search after the inserted text.
    s.replace(old_str, new_str)
}

/// Port of `StringGetAfter`: the part after the *last* occurrence of `key`; all of `s` for
/// an empty key, "" when `key` does not occur.
pub fn string_get_after(s: &str, key: &str) -> String {
    if key.is_empty() {
        return s.to_string();
    }
    match s.rfind(key) {
        Some(found) => s[found + key.len()..].to_string(),
        None => String::new(),
    }
}

/// Port of `StringSplit`: `boost::split(elems, str, boost::is_any_of(delim),
/// boost::token_compress_on)`. Splits at every character of `delim`; a run of adjacent
/// delimiters counts as one. A leading or trailing delimiter yields an empty first or last
/// element, an empty `delim` yields `[s]`, and an empty `s` yields `[""]`.
///
/// `boost::is_any_of` matches bytes; this matches `char`s, which is the same for the ASCII
/// delimiters COLMAP passes and keeps every piece valid UTF-8.
pub fn string_split(s: &str, delim: &str) -> Vec<String> {
    let mut elems = Vec::new();
    let mut current = String::new();
    let mut in_delim_run = false;
    for c in s.chars() {
        if delim.contains(c) {
            if !in_delim_run {
                elems.push(std::mem::take(&mut current));
                in_delim_run = true;
            }
        } else {
            in_delim_run = false;
            current.push(c);
        }
    }
    elems.push(current);
    elems
}

/// Port of `StringStartsWith`. An empty prefix never matches, as in COLMAP.
pub fn string_starts_with(s: &str, prefix: &str) -> bool {
    !prefix.is_empty() && s.starts_with(prefix)
}

/// Port of `StringLeftTrim`: removes leading ' ', '\n', '\r', '\t'.
pub fn string_left_trim(s: &mut String) {
    let start = s.len() - s.trim_start_matches(is_white_space).len();
    s.drain(..start);
}

/// Port of `StringRightTrim`: removes trailing ' ', '\n', '\r', '\t'.
pub fn string_right_trim(s: &mut String) {
    let end = s.trim_end_matches(is_white_space).len();
    s.truncate(end);
}

/// Port of `StringTrim`: [`string_left_trim`] then [`string_right_trim`].
pub fn string_trim(s: &mut String) {
    string_left_trim(s);
    string_right_trim(s);
}

/// Port of `StringToLower` (ASCII, as `::tolower` in the "C" locale).
pub fn string_to_lower(s: &mut str) {
    s.make_ascii_lowercase();
}

/// Port of `StringToUpper` (ASCII, as `::toupper` in the "C" locale).
pub fn string_to_upper(s: &mut str) {
    s.make_ascii_uppercase();
}

/// Port of `StringContains`. Every string contains "".
pub fn string_contains(s: &str, sub_str: &str) -> bool {
    s.contains(sub_str)
}

/// Port of `StringToDouble`: parses a floating-point value with the classic ("C") locale,
/// allowing surrounding white space but nothing else, and errors with COLMAP's
/// `THROW_CHECK` message ("Failed to parse floating-point value: <str>") otherwise.
///
/// COLMAP reads with `std::istringstream >> double`; this uses Rust's correctly rounded
/// parser on the token, which agrees on decimal and exponent notation (the only forms
/// COLMAP's writers produce). Spellings the two could treat differently are rejected here:
/// `inf`/`nan` and hexadecimal floats, and values that overflow to infinity (libc++ sets
/// `failbit` on `ERANGE`). `docs/CPP_DIVERGENCES.md` entry 60.
pub fn string_to_double(s: &str) -> Result<f64> {
    let value = parse_double_token(s);
    crate::check!(
        value.is_some(),
        "Failed to parse floating-point value: {}",
        s
    );
    // `value` is `Some` past the check.
    Ok(value.unwrap_or(f64::NAN))
}

/// The accepting core of [`string_to_double`]; `None` where COLMAP's stream read fails or
/// leaves trailing characters.
pub(crate) fn parse_double_token(s: &str) -> Option<f64> {
    // `>>` skips leading white space (C `isspace`), and the trailing-token check accepts
    // trailing white space.
    let token = s.trim_matches(C_ISSPACE);
    if token.is_empty()
        || !token
            .bytes()
            .all(|b| b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-'))
    {
        return None;
    }
    let value: f64 = token.parse().ok()?;
    if value.is_infinite() {
        return None;
    }
    Some(value)
}
