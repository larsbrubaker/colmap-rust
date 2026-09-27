// Minimal JSON reader for the oracle fixtures in `tests/data/oracle/` (test support, not a
// test). The fixtures are written by Python's `json` module from `oracle/*.py`; this reads
// objects, arrays, numbers, strings (with the standard escapes), booleans and null, which is
// all they contain. Written here so the core crate's tests need no JSON dependency.
//
// Numbers parse with Rust's `str::parse::<f64>`, which is correctly rounded, and Python writes
// every double with `repr`, which round-trips, so fixture values arrive bit for bit.
//
// Include it from a test binary with `#[path = "support/oracle_json.rs"] mod oracle_json;`.

#![allow(dead_code)]

use std::collections::BTreeMap;

/// A parsed JSON value. Object keys keep a sorted map (the fixtures never depend on order).
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(BTreeMap<String, Json>),
}

impl Json {
    /// Parses a whole document; panics with the byte offset on malformed input (a broken
    /// fixture is a test failure).
    pub fn parse(text: &str) -> Json {
        let mut parser = Parser {
            bytes: text.as_bytes(),
            pos: 0,
        };
        let value = parser.value();
        parser.skip_whitespace();
        assert_eq!(
            parser.pos,
            parser.bytes.len(),
            "trailing data after JSON value"
        );
        value
    }

    /// The member `key` of an object; panics if absent.
    pub fn get(&self, key: &str) -> &Json {
        match self {
            Json::Object(map) => map.get(key).unwrap_or_else(|| panic!("missing key {key}")),
            _ => panic!("not an object (looking up {key})"),
        }
    }

    /// The elements of an array.
    pub fn as_array(&self) -> &[Json] {
        match self {
            Json::Array(items) => items,
            _ => panic!("not an array: {self:?}"),
        }
    }

    /// A number.
    pub fn as_f64(&self) -> f64 {
        match self {
            Json::Number(v) => *v,
            _ => panic!("not a number: {self:?}"),
        }
    }

    /// A string.
    pub fn as_str(&self) -> &str {
        match self {
            Json::String(s) => s,
            _ => panic!("not a string: {self:?}"),
        }
    }

    /// A number or an array of numbers, as a vector of doubles.
    pub fn as_f64s(&self) -> Vec<f64> {
        match self {
            Json::Number(v) => vec![*v],
            Json::Array(items) => items.iter().map(Json::as_f64).collect(),
            _ => panic!("not a number or number array: {self:?}"),
        }
    }
}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&mut self) -> u8 {
        self.skip_whitespace();
        *self
            .bytes
            .get(self.pos)
            .unwrap_or_else(|| panic!("unexpected end of JSON"))
    }

    fn expect(&mut self, byte: u8) {
        assert_eq!(
            self.peek(),
            byte,
            "expected '{}' at byte {}",
            byte as char,
            self.pos
        );
        self.pos += 1;
    }

    fn literal(&mut self, word: &str, value: Json) -> Json {
        assert!(
            self.bytes[self.pos..].starts_with(word.as_bytes()),
            "bad literal at byte {}",
            self.pos
        );
        self.pos += word.len();
        value
    }

    fn value(&mut self) -> Json {
        match self.peek() {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => Json::String(self.string()),
            b't' => self.literal("true", Json::Bool(true)),
            b'f' => self.literal("false", Json::Bool(false)),
            b'n' => self.literal("null", Json::Null),
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Json {
        self.expect(b'{');
        let mut map = BTreeMap::new();
        if self.peek() == b'}' {
            self.pos += 1;
            return Json::Object(map);
        }
        loop {
            let key = self.string();
            self.expect(b':');
            map.insert(key, self.value());
            match self.peek() {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    return Json::Object(map);
                }
                other => panic!("unexpected '{}' at byte {}", other as char, self.pos),
            }
        }
    }

    fn array(&mut self) -> Json {
        self.expect(b'[');
        let mut items = Vec::new();
        if self.peek() == b']' {
            self.pos += 1;
            return Json::Array(items);
        }
        loop {
            items.push(self.value());
            match self.peek() {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    return Json::Array(items);
                }
                other => panic!("unexpected '{}' at byte {}", other as char, self.pos),
            }
        }
    }

    fn string(&mut self) -> String {
        self.expect(b'"');
        let mut out = String::new();
        loop {
            let start = self.pos;
            while self.bytes[self.pos] != b'"' && self.bytes[self.pos] != b'\\' {
                self.pos += 1;
            }
            out.push_str(std::str::from_utf8(&self.bytes[start..self.pos]).expect("UTF-8"));
            let byte = self.bytes[self.pos];
            self.pos += 1;
            if byte == b'"' {
                return out;
            }
            let escape = self.bytes[self.pos];
            self.pos += 1;
            match escape {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'/' => out.push('/'),
                b'b' => out.push('\u{8}'),
                b'f' => out.push('\u{c}'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'u' => {
                    let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4]).unwrap();
                    let code = u32::from_str_radix(hex, 16).expect("\\u escape");
                    self.pos += 4;
                    // The fixtures are ASCII; surrogate pairs are not needed.
                    out.push(char::from_u32(code).expect("non-surrogate \\u escape"));
                }
                other => panic!("bad escape '\\{}'", other as char),
            }
        }
    }

    fn number(&mut self) -> Json {
        let start = self.pos;
        while self.pos < self.bytes.len()
            && matches!(
                self.bytes[self.pos],
                b'0'..=b'9' | b'-' | b'+' | b'.' | b'e' | b'E'
            )
        {
            self.pos += 1;
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos]).unwrap();
        // Python writes non-finite doubles as NaN / Infinity / -Infinity.
        let text = match text {
            "" if self.bytes[self.pos..].starts_with(b"NaN") => {
                self.pos += 3;
                "NaN"
            }
            "" if self.bytes[self.pos..].starts_with(b"Infinity") => {
                self.pos += 8;
                "inf"
            }
            "-" if self.bytes[self.pos..].starts_with(b"Infinity") => {
                self.pos += 8;
                "-inf"
            }
            t => t,
        };
        Json::Number(
            text.parse()
                .unwrap_or_else(|_| panic!("bad number {text:?} at {start}")),
        )
    }
}
