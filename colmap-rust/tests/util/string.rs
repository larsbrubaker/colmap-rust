// Port of COLMAP's src/colmap/util/string_test.cc, 1:1 (Suite_Name -> suite_name, with
// COLMAP's own suite spellings kept: StringStrim, StringLeftString, StringStrimRight),
// testing colmap_rust::util::string. Tier A (exact). EXPECT_DOUBLE_EQ (4 ulp) is asserted
// as exact equality, which is stricter.
//
// Not ported:
// - StringPrintf.Nominal: StringPrintf is not ported; Rust's `format!` replaces printf-style
//   varargs (src/util/string.rs header).
// - ConversionBetweenPlatformAndUTF8.NonASCIIStringRoundtrip: PlatformToUTF8 /
//   UTF8ToPlatform are not ported; Rust strings are always UTF-8 on every platform.
// StringToDouble.LocaleIndependence keeps its StringToDouble assertions; the part that
// installs a comma-decimal global C++ locale and reads "1,23" through a stringstream tests
// libc++ and has no Rust counterpart (Rust parsing has no locale).

use colmap_rust::util::string::*;

fn inplace(f: impl Fn(&mut String), s: &str) -> String {
    let mut s = s.to_string();
    f(&mut s);
    s
}

#[test]
fn string_replace_nominal() {
    assert_eq!(string_replace("test", "-", ""), "test");
    assert_eq!(string_replace("test", "t", "a"), "aesa");
    assert_eq!(string_replace("test", "t", "---"), "---es---");
    assert_eq!(string_replace("test", "", "a"), "test");
    assert_eq!(string_replace("test", "", ""), "test");
    assert_eq!(string_replace("ttt", "ttt", "+++"), "+++");
}

#[test]
fn string_get_after_nominal() {
    assert_eq!(string_get_after("test", ""), "test");
    assert_eq!(string_get_after("test", "notinit"), "");
    assert_eq!(string_get_after("test", "e"), "st");
    assert_eq!(string_get_after("test, multiple tests", "test"), "s");
    assert_eq!(string_get_after("", ""), "");
    assert_eq!(
        string_get_after("path/to/dataset/sub1/image.png", "sub1/"),
        "image.png"
    );
}

#[test]
fn string_split_nominal() {
    assert_eq!(
        string_split("1,2,3,4,5 , 6", ","),
        ["1", "2", "3", "4", "5 ", " 6"]
    );
    assert_eq!(string_split("1,2,3,4,5 , 6", ""), ["1,2,3,4,5 , 6"]);
    assert_eq!(
        string_split("1,,2,,3,4,5 , 6", ","),
        ["1", "2", "3", "4", "5 ", " 6"]
    );
    assert_eq!(
        string_split("1,,2,,3,4,5 , 6", ",,"),
        ["1", "2", "3", "4", "5 ", " 6"]
    );
    assert_eq!(
        string_split("1,,2,,3,4,5 , 6", ", "),
        ["1", "2", "3", "4", "5", "6"]
    );
    assert_eq!(
        string_split(",1,,2,,3,4,5 , 6 ", ", "),
        ["", "1", "2", "3", "4", "5", "6", ""]
    );
}

#[test]
fn string_starts_with_nominal() {
    assert!(!string_starts_with("", ""));
    assert!(!string_starts_with("a", ""));
    assert!(!string_starts_with("", "a"));
    assert!(string_starts_with("a", "a"));
    assert!(string_starts_with("aa", "a"));
    assert!(string_starts_with("aa", "aa"));
    assert!(string_starts_with("aaaaaaaaa", "aa"));
}

#[test]
fn string_strim_nominal() {
    let cases = [
        ("", ""),
        (" ", ""),
        ("a", "a"),
        (" a", "a"),
        ("a ", "a"),
        (" a ", "a"),
        ("  a  ", "a"),
        ("aa  ", "aa"),
        ("a  a  ", "a  a"),
        ("a  a  a ", "a  a  a"),
        ("\n\r\t", ""),
    ];
    for (input, expected) in cases {
        assert_eq!(inplace(string_trim, input), expected, "{input:?}");
    }
}

#[test]
fn string_left_string_nominal() {
    let cases = [
        ("", ""),
        (" ", ""),
        ("a", "a"),
        (" a", "a"),
        ("a ", "a "),
        (" a ", "a "),
        ("  a  ", "a  "),
        ("aa  ", "aa  "),
        ("a  a  ", "a  a  "),
        ("  a  a", "a  a"),
        ("a  a  a ", "a  a  a "),
    ];
    for (input, expected) in cases {
        assert_eq!(inplace(string_left_trim, input), expected, "{input:?}");
    }
    assert_eq!(inplace(string_trim, "\n\r\ta"), "a");
}

#[test]
fn string_strim_right_nominal() {
    let cases = [
        ("", ""),
        (" ", ""),
        ("a", "a"),
        (" a", " a"),
        ("a ", "a"),
        (" a ", " a"),
        ("  a  ", "  a"),
        ("aa  ", "aa"),
        ("a  a  ", "a  a"),
        ("a  a  a ", "a  a  a"),
    ];
    for (input, expected) in cases {
        assert_eq!(inplace(string_right_trim, input), expected, "{input:?}");
    }
    assert_eq!(inplace(string_trim, "a\n\r\t"), "a");
}

#[test]
fn string_to_lower_nominal() {
    let cases = [
        ("", ""),
        (" ", " "),
        ("a", "a"),
        (" a", " a"),
        ("a ", "a "),
        (" a ", " a "),
        ("aa  ", "aa  "),
        ("A", "a"),
        (" A", " a"),
        ("A ", "a "),
        (" A ", " a "),
        ("AA  ", "aa  "),
        ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"),
        (
            "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            "0123456789 abcdefghijklmnopqrstuvwxyz",
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(inplace(|s| string_to_lower(s), input), expected);
    }
}

#[test]
fn string_to_upper_nominal() {
    let cases = [
        ("", ""),
        (" ", " "),
        ("A", "A"),
        (" A", " A"),
        ("A ", "A "),
        (" A ", " A "),
        ("AA  ", "AA  "),
        ("a", "A"),
        (" a", " A"),
        ("a ", "A "),
        (" a ", " A "),
        ("aa  ", "AA  "),
        ("abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        (
            "0123456789 abcdefghijklmnopqrstuvwxyz",
            "0123456789 ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        ),
    ];
    for (input, expected) in cases {
        assert_eq!(inplace(|s| string_to_upper(s), input), expected);
    }
}

#[test]
fn string_contains_nominal() {
    assert!(string_contains("", ""));
    assert!(string_contains("a", ""));
    assert!(string_contains("a", "a"));
    assert!(string_contains("ab", "a"));
    assert!(string_contains("ab", "ab"));
    assert!(!string_contains("", "a"));
    assert!(!string_contains("ab", "c"));
}

#[test]
fn string_to_double_nominal() {
    assert_eq!(string_to_double("0").unwrap(), 0.0);
    assert_eq!(string_to_double("0.5").unwrap(), 0.5);
    assert_eq!(string_to_double("-1.23").unwrap(), -1.23);
    assert_eq!(string_to_double("1.234e10").unwrap(), 1.234e10);
    assert_eq!(string_to_double("-5.67e-3").unwrap(), -5.67e-3);
    assert_eq!(string_to_double("100").unwrap(), 100.0);
    #[allow(clippy::excessive_precision)]
    let expected = 0.12345678901234567;
    assert_eq!(string_to_double("0.12345678901234567").unwrap(), expected);
}

#[test]
fn string_to_double_locale_independence() {
    assert_eq!(string_to_double("0.5").unwrap(), 0.5);
    assert_eq!(string_to_double("0.1").unwrap(), 0.1);
    assert_eq!(string_to_double("0.2").unwrap(), 0.2);
    assert_eq!(string_to_double("0.3").unwrap(), 0.3);
    assert_eq!(string_to_double("-1.23e5").unwrap(), -1.23e5);
}

// Rust-only: StringToDouble's failure cases (THROW_CHECK) and accepted white space.
#[test]
fn rust_only_string_to_double_rejects() {
    assert_eq!(string_to_double(" 2.5 \n").unwrap(), 2.5);
    for bad in [
        "", " ", "abc", "1.5x", "1.5 2", "1,5", "inf", "nan", "0x1p3", "1e400",
    ] {
        let err = string_to_double(bad).unwrap_err();
        assert_eq!(
            err.kind(),
            colmap_rust::ErrorKind::InvalidArgument,
            "{bad:?}"
        );
        assert!(
            err.message()
                .ends_with(&format!("Failed to parse floating-point value: {bad}")),
            "{}",
            err.message()
        );
    }
}
