//! A small JSON reader for the core, written here (not ported): it replaces what COLMAP gets
//! from `boost::property_tree::read_json` for its rig configs (`scene/rig.cc`), so the core
//! stays std-only (PORTING_PLAN.md, Decisions). Its one user is
//! [`crate::scene::rig::read_rig_config`]; the ptree value translators it reads numbers and
//! booleans through are [`crate::util::ptree_translate`]. Tests:
//! `tests/util/rust_only_json.rs`.
//!
//! It accepts strict JSON (RFC 8259), which is what Boost's parser accepts: objects, arrays,
//! strings with every escape (`\uXXXX` with surrogate pairs), numbers, `true`, `false`,
//! `null`, any of them at the top level, and nothing but white space after it. Trailing
//! commas, comments, single quotes, raw control characters in strings and lone surrogates are
//! errors, as in Boost. Nesting deeper than [`MAX_DEPTH`] is an error (Boost recurses without
//! a limit; the divergence below).
//!
//! The tree keeps what `property_tree` keeps: a number keeps its source text (Boost stores
//! every value as text and converts on access), an object keeps every member in source order,
//! duplicates included, and [`JsonValue::get`] finds the first member with a key, as ptree's
//! `get_child` does. [`JsonValue::ptree_data`] and [`JsonValue::children`] are ptree's view
//! of a node.
//!
//! Deliberate divergence, the nesting limit: `boost::property_tree::read_json` recurses once per
//! nested array or object without a limit, so a deeply nested file overflows the stack.
//! [`parse_json`] returns an error ("nesting deeper than 512 levels") past [`MAX_DEPTH`] (512)
//! nested arrays and objects. Every document within the limit parses as Boost's parser does. A
//! stack overflow is a crash, natively and in the browser (whose wasm stack is small), so an
//! error is the only safe answer; real rig configs nest three levels.
//! `rust_only_json_nesting_limit` (`tests/util/rust_only_json.rs`) parses 512 levels, fails 513,
//! and fails a million without a crash.

use crate::util::check::{ColmapError, ErrorKind, Result};

/// The deepest nesting of arrays and objects the reader accepts. Real rig configs nest three
/// levels; the limit only stops hostile input from overflowing the stack (the divergence in
/// the file header).
pub const MAX_DEPTH: usize = 512;

/// A parsed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    /// `null`.
    Null,
    /// `true` or `false`.
    Bool(bool),
    /// A number, kept as its source text (`-1.5e3`); see [`JsonValue::number_value`].
    Number(String),
    /// A string, escapes resolved.
    String(String),
    /// An array's elements in order.
    Array(Vec<JsonValue>),
    /// An object's members in source order; a key may occur more than once.
    Object(Vec<(String, JsonValue)>),
}

impl JsonValue {
    /// `ptree::get_child_optional(key)`: the value of the first member with this key, `None`
    /// for a missing key or a node that is not an object (array elements have no keys).
    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        match self {
            JsonValue::Object(members) => members
                .iter()
                .find(|(member_key, _)| member_key == key)
                .map(|(_, value)| value),
            _ => None,
        }
    }

    /// ptree's children of the node: an array's elements or an object's member values, in
    /// order; a scalar has none.
    pub fn children(&self) -> Box<dyn Iterator<Item = &JsonValue> + '_> {
        match self {
            JsonValue::Array(elements) => Box::new(elements.iter()),
            JsonValue::Object(members) => Box::new(members.iter().map(|(_, value)| value)),
            _ => Box::new(std::iter::empty()),
        }
    }

    /// The text ptree stores for the node (`ptree::data()`): a string's value, a number's
    /// source text, `"true"`, `"false"`, `"null"`, and `""` for an array or object.
    pub fn ptree_data(&self) -> &str {
        match self {
            JsonValue::Null => "null",
            JsonValue::Bool(true) => "true",
            JsonValue::Bool(false) => "false",
            JsonValue::Number(text) | JsonValue::String(text) => text,
            JsonValue::Array(_) | JsonValue::Object(_) => "",
        }
    }

    /// A number's value, its source text through `str::parse::<f64>` (correctly rounded;
    /// out-of-range magnitudes become infinity or zero). `None` for anything but a number.
    pub fn number_value(&self) -> Option<f64> {
        match self {
            JsonValue::Number(text) => text.parse::<f64>().ok(),
            _ => None,
        }
    }
}

/// Parses JSON text. A malformed document is an `Err` naming the line and column (1-based,
/// columns in characters) and what was expected, e.g. `"JSON parse error at line 1, column
/// 3: expected value"`.
pub fn parse_json(text: &str) -> Result<JsonValue> {
    let mut parser = Parser {
        chars: text.char_indices().peekable(),
        text,
        line: 1,
        column: 1,
    };
    parser.skip_ws();
    let value = parser.parse_value(0)?;
    parser.skip_ws();
    if parser.peek().is_some() {
        return Err(parser.error("garbage after data"));
    }
    Ok(value)
}

struct Parser<'a> {
    chars: std::iter::Peekable<std::str::CharIndices<'a>>,
    text: &'a str,
    line: usize,
    column: usize,
}

impl Parser<'_> {
    fn peek(&mut self) -> Option<char> {
        self.chars.peek().map(|&(_, c)| c)
    }

    fn next(&mut self) -> Option<char> {
        let (_, c) = self.chars.next()?;
        if c == '\n' {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        Some(c)
    }

    fn offset(&mut self) -> usize {
        match self.chars.peek() {
            Some(&(offset, _)) => offset,
            None => self.text.len(),
        }
    }

    fn error(&self, what: &str) -> ColmapError {
        ColmapError::new(
            ErrorKind::RuntimeError,
            format!(
                "JSON parse error at line {}, column {}: {what}",
                self.line, self.column
            ),
        )
    }

    fn have(&mut self, c: char) -> bool {
        if self.peek() == Some(c) {
            self.next();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, c: char, what: &str) -> Result<()> {
        if self.have(c) {
            Ok(())
        } else {
            Err(self.error(what))
        }
    }

    fn skip_ws(&mut self) {
        while matches!(self.peek(), Some(' ' | '\t' | '\n' | '\r')) {
            self.next();
        }
    }

    fn parse_value(&mut self, depth: usize) -> Result<JsonValue> {
        match self.peek() {
            Some('{') => self.parse_object(depth + 1),
            Some('[') => self.parse_array(depth + 1),
            Some('"') => Ok(JsonValue::String(self.parse_string()?)),
            Some('t') => self.parse_keyword("true", JsonValue::Bool(true)),
            Some('f') => self.parse_keyword("false", JsonValue::Bool(false)),
            Some('n') => self.parse_keyword("null", JsonValue::Null),
            Some('-' | '0'..='9') => self.parse_number(),
            _ => Err(self.error("expected value")),
        }
    }

    fn check_depth(&self, depth: usize) -> Result<()> {
        if depth > MAX_DEPTH {
            return Err(self.error(&format!("nesting deeper than {MAX_DEPTH} levels")));
        }
        Ok(())
    }

    fn parse_object(&mut self, depth: usize) -> Result<JsonValue> {
        self.check_depth(depth)?;
        self.next(); // '{'
        self.skip_ws();
        let mut members = Vec::new();
        if self.have('}') {
            return Ok(JsonValue::Object(members));
        }
        loop {
            if self.peek() != Some('"') {
                return Err(self.error("expected key string"));
            }
            let key = self.parse_string()?;
            self.skip_ws();
            self.expect(':', "expected ':'")?;
            self.skip_ws();
            let value = self.parse_value(depth)?;
            members.push((key, value));
            self.skip_ws();
            if self.have(',') {
                self.skip_ws();
                continue;
            }
            self.expect('}', "expected '}' or ','")?;
            return Ok(JsonValue::Object(members));
        }
    }

    fn parse_array(&mut self, depth: usize) -> Result<JsonValue> {
        self.check_depth(depth)?;
        self.next(); // '['
        self.skip_ws();
        let mut elements = Vec::new();
        if self.have(']') {
            return Ok(JsonValue::Array(elements));
        }
        loop {
            elements.push(self.parse_value(depth)?);
            self.skip_ws();
            if self.have(',') {
                self.skip_ws();
                continue;
            }
            self.expect(']', "expected ']' or ','")?;
            return Ok(JsonValue::Array(elements));
        }
    }

    fn parse_keyword(&mut self, word: &str, value: JsonValue) -> Result<JsonValue> {
        for expected in word.chars() {
            if self.peek() != Some(expected) {
                return Err(self.error(&format!("expected '{word}'")));
            }
            self.next();
        }
        Ok(value)
    }

    /// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`, kept as text.
    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.offset();
        self.have('-');
        match self.peek() {
            Some('0') => {
                self.next();
            }
            Some('1'..='9') => self.skip_digits(),
            _ => return Err(self.error("expected digits after '-'")),
        }
        if self.have('.') {
            if !matches!(self.peek(), Some('0'..='9')) {
                return Err(self.error("need at least one digit after '.'"));
            }
            self.skip_digits();
        }
        if matches!(self.peek(), Some('e' | 'E')) {
            self.next();
            if !self.have('+') {
                self.have('-');
            }
            if !matches!(self.peek(), Some('0'..='9')) {
                return Err(self.error("need at least one digit in exponent"));
            }
            self.skip_digits();
        }
        let end = self.offset();
        Ok(JsonValue::Number(self.text[start..end].to_string()))
    }

    fn skip_digits(&mut self) {
        while matches!(self.peek(), Some('0'..='9')) {
            self.next();
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        self.next(); // '"'
        let mut value = String::new();
        loop {
            match self.next() {
                None => return Err(self.error("unterminated string")),
                Some('"') => return Ok(value),
                Some('\\') => self.parse_escape(&mut value)?,
                Some(c) if (c as u32) < 0x20 => {
                    return Err(self.error("invalid code sequence (control character in string)"))
                }
                Some(c) => value.push(c),
            }
        }
    }

    fn parse_escape(&mut self, value: &mut String) -> Result<()> {
        let c = match self.next() {
            Some('"') => '"',
            Some('\\') => '\\',
            Some('/') => '/',
            Some('b') => '\u{8}',
            Some('f') => '\u{c}',
            Some('n') => '\n',
            Some('r') => '\r',
            Some('t') => '\t',
            Some('u') => self.parse_codepoint()?,
            _ => return Err(self.error("invalid escape sequence")),
        };
        value.push(c);
        Ok(())
    }

    /// The code point of `XXXX` after `\u`, joining a surrogate pair `\uD8xx\uDCxx`.
    fn parse_codepoint(&mut self) -> Result<char> {
        let high = self.parse_hex4()?;
        let code = if (0xDC00..0xE000).contains(&high) {
            return Err(self.error("invalid codepoint, stray low surrogate"));
        } else if (0xD800..0xDC00).contains(&high) {
            if !(self.have('\\') && self.have('u')) {
                return Err(self.error("expected codepoint reference after high surrogate"));
            }
            let low = self.parse_hex4()?;
            if !(0xDC00..0xE000).contains(&low) {
                return Err(self.error("expected low surrogate after high surrogate"));
            }
            0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)
        } else {
            high
        };
        char::from_u32(code).ok_or_else(|| self.error("invalid codepoint"))
    }

    fn parse_hex4(&mut self) -> Result<u32> {
        let mut code = 0;
        for _ in 0..4 {
            let digit = self
                .peek()
                .and_then(|c| c.to_digit(16))
                .ok_or_else(|| self.error("invalid escape sequence (expected 4 hex digits)"))?;
            self.next();
            code = code * 16 + digit;
        }
        Ok(code)
    }
}
