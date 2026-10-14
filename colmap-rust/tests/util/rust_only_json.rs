// Rust-only tests of colmap_rust::util::json, the core's JSON reader (written here in place of
// boost::property_tree's read_json; COLMAP has no test of its own for it): values, escapes,
// nesting, the property-tree view (first duplicate wins, children, stored text), malformed
// input and the nesting limit (docs/CPP_DIVERGENCES.md entry 275).

use colmap_rust::util::json::{parse_json, JsonValue, MAX_DEPTH};

fn error(text: &str) -> String {
    parse_json(text).unwrap_err().message().to_string()
}

#[test]
fn rust_only_json_scalars_and_containers() {
    assert_eq!(parse_json("null").unwrap(), JsonValue::Null);
    assert_eq!(parse_json(" true ").unwrap(), JsonValue::Bool(true));
    assert_eq!(parse_json("\nfalse\r\n").unwrap(), JsonValue::Bool(false));
    assert_eq!(
        parse_json("-1.5e3").unwrap(),
        JsonValue::Number("-1.5e3".to_string())
    );
    assert_eq!(
        parse_json(r#""text""#).unwrap(),
        JsonValue::String("text".to_string())
    );
    assert_eq!(parse_json("[]").unwrap(), JsonValue::Array(Vec::new()));
    assert_eq!(parse_json("{ }").unwrap(), JsonValue::Object(Vec::new()));
    assert_eq!(
        parse_json(r#"{"a": [1, {"b": null}], "c": "d"}"#).unwrap(),
        JsonValue::Object(vec![
            (
                "a".to_string(),
                JsonValue::Array(vec![
                    JsonValue::Number("1".to_string()),
                    JsonValue::Object(vec![("b".to_string(), JsonValue::Null)]),
                ])
            ),
            ("c".to_string(), JsonValue::String("d".to_string())),
        ])
    );
}

#[test]
fn rust_only_json_string_escapes() {
    let text = concat!(
        r#""q\" b\\ s\/ \b\f\n\r\t "#,
        r#"\u0041\u00e9\u20ac \ud83d\ude00 ü""#
    );
    let value = parse_json(text).unwrap();
    assert_eq!(
        value,
        JsonValue::String("q\" b\\ s/ \u{8}\u{c}\n\r\t Aé€ \u{1F600} ü".to_string())
    );
}

#[test]
fn rust_only_json_numbers_parse_like_str_parse() {
    for text in [
        "0",
        "-0",
        "1",
        "0.1",
        "1e5",
        "1E+5",
        "2.5e-3",
        "-123456789.123456789",
        "1.7976931348623157e308",
        "4.9406564584124654e-324",
        "1e400",
        "1e-400",
        "0.30000000000000004",
    ] {
        let value = parse_json(text).unwrap();
        assert_eq!(value, JsonValue::Number(text.to_string()));
        assert_eq!(
            value.number_value().unwrap().to_bits(),
            text.parse::<f64>().unwrap().to_bits(),
            "{text}"
        );
    }
    assert_eq!(JsonValue::String("1".to_string()).number_value(), None);
}

#[test]
fn rust_only_json_property_tree_view() {
    let value = parse_json(r#"{"k": 1, "k": 2, "o": {"x": true}, "a": [null, "s"]}"#).unwrap();
    // get_child finds the first member with the key.
    assert_eq!(value.get("k").unwrap().ptree_data(), "1");
    assert!(value.get("missing").is_none());
    // Array elements have no keys.
    assert!(value.get("a").unwrap().get("0").is_none());
    // Children: an object's member values (duplicates included), an array's elements.
    let children: Vec<&str> = value.children().map(JsonValue::ptree_data).collect();
    assert_eq!(children, ["1", "2", "", ""]);
    let elements: Vec<&str> = value
        .get("a")
        .unwrap()
        .children()
        .map(JsonValue::ptree_data)
        .collect();
    assert_eq!(elements, ["null", "s"]);
    assert_eq!(
        value.get("o").unwrap().get("x").unwrap().ptree_data(),
        "true"
    );
    assert_eq!(JsonValue::Bool(false).ptree_data(), "false");
    assert_eq!(JsonValue::Null.children().count(), 0);
}

#[test]
fn rust_only_json_malformed_input_is_an_error() {
    assert_eq!(
        error(""),
        "JSON parse error at line 1, column 1: expected value"
    );
    assert_eq!(
        error("[{"),
        "JSON parse error at line 1, column 3: expected key string"
    );
    assert_eq!(
        error("[1,\n 2,]"),
        "JSON parse error at line 2, column 4: expected value"
    );
    for (text, what) in [
        (r#"{"a": 1,}"#, "expected key string"),
        (r#"{"a" 1}"#, "expected ':'"),
        ("[1 2]", "expected ']' or ','"),
        (r#"{"a": 1 "b": 2}"#, "expected '}' or ','"),
        ("[1] x", "garbage after data"),
        ("tru", "expected 'true'"),
        ("nul", "expected 'null'"),
        ("01", "garbage after data"),
        ("-", "expected digits after '-'"),
        ("1.", "need at least one digit after '.'"),
        ("1e", "need at least one digit in exponent"),
        ("+1", "expected value"),
        (".5", "expected value"),
        ("'a'", "expected value"),
        ("// c\n[]", "expected value"),
        (r#""abc"#, "unterminated string"),
        ("\"a\nb\"", "control character in string"),
        (r#""\x""#, "invalid escape sequence"),
        (r#""\u12g4""#, "expected 4 hex digits"),
        (r#""\udc00""#, "stray low surrogate"),
        (
            r#""\ud83d""#,
            "expected codepoint reference after high surrogate",
        ),
        (
            "\"\\ud83d\\u0041\"",
            "expected low surrogate after high surrogate",
        ),
        (
            "\"\\ud83dA\"",
            "expected codepoint reference after high surrogate",
        ),
        ("NaN", "expected value"),
    ] {
        let message = error(text);
        assert!(message.contains(what), "{text:?}: {message}");
    }
}

#[test]
fn rust_only_json_nesting_limit() {
    let nested = |depth: usize| format!("{}{}", "[".repeat(depth), "]".repeat(depth));
    assert!(parse_json(&nested(MAX_DEPTH)).is_ok());
    assert!(error(&nested(MAX_DEPTH + 1)).contains("nesting deeper than"));
    // Far deeper input fails at the limit instead of overflowing the stack.
    assert!(parse_json(&"[".repeat(1_000_000)).is_err());
    let objects = format!(
        "{}1{}",
        r#"{"a":"#.repeat(MAX_DEPTH + 1),
        "}".repeat(MAX_DEPTH + 1)
    );
    assert!(error(&objects).contains("nesting deeper than"));
}
