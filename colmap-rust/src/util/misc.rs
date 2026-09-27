//! Port of COLMAP's `src/colmap/util/misc.h/.cc`: vector membership helpers, the CSV list
//! conversions (`CSVToVector`, `VectorToCSV`, used for camera parameters and option lists)
//! and `RemoveCommandLineArgument`. Builds on [`super::string`].
//! Tests: `tests/util/misc.rs` (`misc_test.cc`).
//!
//! Not ported: `STRINGIFY` (Rust has `stringify!`) and `LOG_HEADING1/2` (log formatting
//! only; the core crate has no logger, and the app shows progress through its own UI).
//!
//! Tier A (exact). `VectorToCSV` writes each value as a C++ `std::ostream` would, through
//! [`CheckOpValue`] (doubles and floats at the default precision 6, `%g`).

use super::check::{CheckOpValue, ColmapError, ErrorKind, Result};
use super::string::{parse_double_token, string_split, string_trim, C_ISSPACE};

/// Port of `VectorContainsValue`.
pub fn vector_contains_value<T: PartialEq>(vector: &[T], value: &T) -> bool {
    vector.iter().any(|element| element == value)
}

/// Port of `VectorContainsDuplicateValues`. Like COLMAP (`std::unique` on an unsorted
/// copy), this finds only *adjacent* duplicates: `[1, 2, 1]` has none.
pub fn vector_contains_duplicate_values<T: PartialEq>(vector: &[T]) -> bool {
    vector.windows(2).any(|pair| pair[0] == pair[1])
}

/// The element types `CSVToVector<T>` supports (COLMAP's `static_assert`: `std::string`,
/// `int`, `float`, `double`).
pub trait CsvElement: Sized {
    /// Converts one trimmed, non-empty element. `Ok(None)` is COLMAP's caught
    /// `std::invalid_argument` (the whole list becomes empty); `Err` is an exception
    /// COLMAP does not catch (`std::out_of_range` from `std::stoi`).
    fn from_csv_element(elem: String) -> Result<Option<Self>>;
}

impl CsvElement for String {
    fn from_csv_element(elem: String) -> Result<Option<Self>> {
        Ok(Some(elem))
    }
}

impl CsvElement for i32 {
    fn from_csv_element(elem: String) -> Result<Option<Self>> {
        stoi(&elem)
    }
}

impl CsvElement for f32 {
    fn from_csv_element(elem: String) -> Result<Option<Self>> {
        // `static_cast<float>(StringToDouble(elem))`: round the double to nearest float.
        Ok(parse_double_token(&elem).map(|value| value as f32))
    }
}

impl CsvElement for f64 {
    fn from_csv_element(elem: String) -> Result<Option<Self>> {
        Ok(parse_double_token(&elem))
    }
}

/// `std::stoi(elem)` (base 10): skips leading white space, reads an optional sign and the
/// longest run of digits, and ignores what follows (`"3.5"` is 3). No digits is
/// `std::invalid_argument` (`Ok(None)`); a value outside `int` is `std::out_of_range`.
fn stoi(elem: &str) -> Result<Option<i32>> {
    let s = elem.trim_start_matches(C_ISSPACE);
    let bytes = s.as_bytes();
    let mut i = 0;
    let negative = match bytes.first() {
        Some(b'-') => {
            i = 1;
            true
        }
        Some(b'+') => {
            i = 1;
            false
        }
        _ => false,
    };
    let digits_start = i;
    let mut magnitude: i64 = 0;
    let mut overflow = false;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        magnitude = magnitude * 10 + i64::from(bytes[i] - b'0');
        // Beyond i32 either way; stop growing so the i64 cannot overflow.
        if magnitude > i64::from(i32::MAX) + 1 {
            overflow = true;
            magnitude = i64::from(i32::MAX) + 1;
        }
        i += 1;
    }
    if i == digits_start {
        return Ok(None);
    }
    let value = if negative { -magnitude } else { magnitude };
    if overflow || value > i64::from(i32::MAX) || value < i64::from(i32::MIN) {
        return Err(ColmapError::new(
            ErrorKind::OutOfRange,
            "stoi: out of range",
        ));
    }
    Ok(Some(value as i32))
}

/// Port of `CSVToVector<T>`: splits on ',' and ';' (adjacent delimiters merged), trims each
/// element, skips empty ones and converts the rest. If any element fails to convert the
/// result is empty, as in COLMAP (which also logs "Failed to convert CSV element"; the core
/// crate has no logger). `Err` only for an `int` element outside `i32`, which COLMAP does
/// not catch either.
pub fn csv_to_vector<T: CsvElement>(csv: &str) -> Result<Vec<T>> {
    let elems = string_split(csv, ",;");
    let mut values = Vec::with_capacity(elems.len());
    for mut elem in elems {
        string_trim(&mut elem);
        if elem.is_empty() {
            continue;
        }
        match T::from_csv_element(elem)? {
            Some(value) => values.push(value),
            None => return Ok(Vec::new()),
        }
    }
    Ok(values)
}

/// Port of `VectorToCSV<T>`: the values joined with ", ", each written as a classic-locale
/// `std::ostream` writes it; "" for no values.
pub fn vector_to_csv<T: CheckOpValue>(values: &[T]) -> String {
    values
        .iter()
        .map(CheckOpValue::check_op_string)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Port of `RemoveCommandLineArgument`: removes the first argument equal to `arg`, if any.
/// COLMAP's loop (`for j in i+1..argc: argv[i] = argv[j]`, then `argc -= 1`) writes every
/// later argument into the *same* slot `i`, so the last argument ends up in slot `i` and
/// the others stay where they were: that is `Vec::swap_remove`, not an order-preserving
/// shift. `["a", "b", "c", "d"]` minus "b" is `["a", "d", "c"]`.
pub fn remove_command_line_argument(arg: &str, args: &mut Vec<String>) {
    if let Some(index) = args.iter().position(|a| a == arg) {
        args.swap_remove(index);
    }
}
