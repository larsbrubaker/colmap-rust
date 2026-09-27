// Rust-only tests (COLMAP has no test of the C++ standard library) of
// colmap_rust::util::stream_format against printf's %g and %.17g. Port of the cases in
// colmap-sharp's ColmapSharp.Tests/Util/CppStreamFormatTests.cs, whose expected strings came
// from Python's '%g' % v and '%.17g' % v (C printf semantics: correctly rounded from the
// exact binary value, half-even on exact ties). Tier A.

use colmap_rust::util::stream_format::{format_double, DEFAULT_PRECISION};

#[test]
fn rust_only_format_double_matches_printf() {
    let cases: [(f64, &str, &str); 17] = [
        (0.0, "0", "0"),
        (-0.0, "-0", "-0"),
        (1.0, "1", "1"),
        (0.5, "0.5", "0.5"),
        (1234565.0, "1.23456e+06", "1234565"), // exact tie, rounds to even
        (1234575.0, "1.23458e+06", "1234575"), // exact tie, rounds to even
        (123456.0, "123456", "123456"),
        (1234567.0, "1.23457e+06", "1234567"),
        (0.0001, "0.0001", "0.0001"),
        (0.00001, "1e-05", "1.0000000000000001e-05"),
        (9.9999996, "10", "9.9999996000000007"), // carry into a new leading digit
        (99999.95, "99999.9", "99999.949999999997"), // below the tie in binary
        (1e-20, "1e-20", "9.9999999999999995e-21"),
        (1.0 / 3.0, "0.333333", "0.33333333333333331"),
        (-2.5e300, "-2.5e+300", "-2.5000000000000001e+300"),
        (5e-324, "4.94066e-324", "4.9406564584124654e-324"),
        (0.1, "0.1", "0.10000000000000001"),
    ];
    for (value, g6, g17) in cases {
        assert_eq!(
            format_double(value, DEFAULT_PRECISION),
            g6,
            "%g of {value:e}"
        );
        assert_eq!(format_double(value, 17), g17, "%.17g of {value:e}");
    }
}

#[test]
fn rust_only_format_double_non_finite() {
    assert_eq!(format_double(f64::INFINITY, 6), "inf");
    assert_eq!(format_double(f64::NEG_INFINITY, 6), "-inf");
    // macOS libc++ prints "nan" whatever the sign bit.
    assert_eq!(format_double(f64::NAN, 6), "nan");
    assert_eq!(format_double(-f64::NAN, 6), "nan");
    assert_eq!(format_double(f64::NAN, 17), "nan");
}

#[test]
fn rust_only_format_double_large_exponent_fixed_layout() {
    assert_eq!(format_double(-1e16, 17), "-10000000000000000");
    assert_eq!(format_double(1e20, 21), "100000000000000000000");
    assert_eq!(format_double(1e22, 30), "10000000000000000000000");
}

#[test]
fn rust_only_format_double_large_precision() {
    assert_eq!(format_double(123.0, 800), "123");
    assert_eq!(
        format_double(0.1, 60),
        "0.1000000000000000055511151231257827021181583404541015625"
    );
    // %g treats precision 0 as 1.
    assert_eq!(format_double(2.5, 0), "2");
    assert_eq!(format_double(3.5, 0), "4");
}
