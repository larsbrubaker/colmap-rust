// Port of COLMAP's src/colmap/util/misc_test.cc, 1:1 (Suite_Name -> suite_name), testing
// colmap_rust::util::misc. Tier A (exact). EXPECT_FLOAT_EQ / EXPECT_DOUBLE_EQ (4 ulp) are
// asserted as exact equality, which is stricter. RemoveCommandLineArgument takes a
// `Vec<String>` in place of (argc, argv).

// COLMAP's literal 3.14 / 3.14159 / 6.28318 are data, not approximations of pi.
#![allow(clippy::approx_constant)]

use colmap_rust::util::misc::*;

fn ints(csv: &str) -> Vec<i32> {
    csv_to_vector::<i32>(csv).unwrap()
}

fn floats(csv: &str) -> Vec<f32> {
    csv_to_vector::<f32>(csv).unwrap()
}

fn doubles(csv: &str) -> Vec<f64> {
    csv_to_vector::<f64>(csv).unwrap()
}

fn strings(csv: &str) -> Vec<String> {
    csv_to_vector::<String>(csv).unwrap()
}

#[test]
fn vector_contains_value_nominal() {
    assert!(vector_contains_value(&[1, 2, 3], &1));
    assert!(!vector_contains_value(&[2, 3], &1));
}

#[test]
fn vector_contains_duplicate_values_nominal() {
    assert!(!vector_contains_duplicate_values::<i32>(&[]));
    assert!(!vector_contains_duplicate_values(&[1]));
    assert!(!vector_contains_duplicate_values(&[1, 2]));
    assert!(!vector_contains_duplicate_values(&[1, 2, 3]));
    assert!(vector_contains_duplicate_values(&[1, 1, 2, 3]));
    assert!(vector_contains_duplicate_values(&[1, 1, 2, 2, 3]));
    assert!(vector_contains_duplicate_values(&[1, 2, 3, 3]));
    assert!(!vector_contains_duplicate_values(&["a"]));
    assert!(!vector_contains_duplicate_values(&["a", "b"]));
    assert!(vector_contains_duplicate_values(&["a", "a"]));
}

#[test]
fn csv_to_vector_int() {
    assert_eq!(ints("1, 2, 3 , 4,5,6 "), [1, 2, 3, 4, 5, 6]);
    assert_eq!(ints("1; 2; 3 ; 4;5;6 "), [1, 2, 3, 4, 5, 6]);
    assert_eq!(ints("1;, 2;; 3 ; 4;5;6 "), [1, 2, 3, 4, 5, 6]);
}

#[test]
fn csv_to_vector_int_empty() {
    assert!(ints("").is_empty());
    assert!(ints(" ").is_empty());
}

#[test]
fn csv_to_vector_int_invalid() {
    assert_eq!(ints("1, 2, invalid, 3").len(), 0);
}

#[test]
fn csv_to_vector_float() {
    assert_eq!(
        floats("1.5, 2.7, 3.14 , 4.0,5.9,6.2 "),
        [1.5f32, 2.7, 3.14, 4.0, 5.9, 6.2]
    );
    assert_eq!(floats("1.5; 2.7; 3.14"), [1.5f32, 2.7, 3.14]);
}

#[test]
fn csv_to_vector_float_empty() {
    assert!(floats("").is_empty());
    assert!(floats(" ").is_empty());
}

#[test]
fn csv_to_vector_float_invalid() {
    assert_eq!(floats("1.5, 2.7, invalid, 3.14").len(), 0);
}

#[test]
fn csv_to_vector_double() {
    assert_eq!(
        doubles("1.5, 2.7, 3.14159 , 4.0,5.9,6.28318 "),
        [1.5, 2.7, 3.14159, 4.0, 5.9, 6.28318]
    );
    assert_eq!(doubles("1.5; 2.7; 3.14159"), [1.5, 2.7, 3.14159]);
}

#[test]
fn csv_to_vector_double_empty() {
    assert!(doubles("").is_empty());
    assert!(doubles(" ").is_empty());
}

#[test]
fn csv_to_vector_double_invalid() {
    assert_eq!(doubles("1.5, 2.7, invalid, 3.14").len(), 0);
}

#[test]
fn csv_to_vector_string() {
    assert_eq!(
        strings("foo, bar, baz , qux,hello,world "),
        ["foo", "bar", "baz", "qux", "hello", "world"]
    );
    assert_eq!(strings("foo; bar; baz"), ["foo", "bar", "baz"]);
    assert_eq!(strings("foo;, bar;; baz"), ["foo", "bar", "baz"]);
}

#[test]
fn csv_to_vector_string_empty() {
    assert!(strings("").is_empty());
    assert!(strings(" ").is_empty());
}

#[test]
fn csv_to_vector_string_with_spaces() {
    assert_eq!(
        strings(" hello , world , test "),
        ["hello", "world", "test"]
    );
}

#[test]
fn vector_to_csv_nominal() {
    assert_eq!(vector_to_csv::<i32>(&[]), "");
    assert_eq!(vector_to_csv(&[1]), "1");
    assert_eq!(vector_to_csv(&[1, 2, 3]), "1, 2, 3");
}

#[test]
fn remove_command_line_argument_nominal() {
    let mut argv: Vec<String> = ["abc", "def", "ghi"].map(String::from).to_vec();

    remove_command_line_argument("abcd", &mut argv);
    assert_eq!(argv, ["abc", "def", "ghi"]);

    remove_command_line_argument("def", &mut argv);
    assert_eq!(argv, ["abc", "ghi"]);

    remove_command_line_argument("ghi", &mut argv);
    assert_eq!(argv, ["abc"]);

    remove_command_line_argument("abc", &mut argv);
    assert!(argv.is_empty());

    remove_command_line_argument("abc", &mut argv);
    assert!(argv.is_empty());
}

// Rust-only: std::stoi semantics inside CSVToVector<int> (trailing characters ignored,
// out-of-range values are std::out_of_range, which COLMAP does not catch), VectorToCSV of
// doubles at the ostream default precision, and VectorContainsDuplicateValues finding only
// adjacent duplicates (std::unique on an unsorted copy).
#[test]
fn rust_only_csv_edge_cases() {
    assert_eq!(ints("3.5, -7, +8"), [3, -7, 8]);
    assert_eq!(ints("2147483647, -2147483648"), [i32::MAX, i32::MIN]);
    let err = csv_to_vector::<i32>("1, 2147483648").unwrap_err();
    assert_eq!(err.kind(), colmap_rust::ErrorKind::OutOfRange);
    assert_eq!(
        vector_to_csv(&[1.0, 0.5, 1.0 / 3.0, 1234567.0]),
        "1, 0.5, 0.333333, 1.23457e+06"
    );
    assert!(!vector_contains_duplicate_values(&[1, 2, 1]));
}
