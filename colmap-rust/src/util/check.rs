//! The `THROW_CHECK` family of COLMAP's `src/colmap/util/logging.h` (`THROW_CHECK`,
//! `THROW_CHECK_EQ/NE/LT/LE/GT/GE`, `THROW_CHECK_NOTNULL`, `LOG(FATAL_THROW)` and
//! `LOG_FATAL_THROW(exception)`), mapped to Rust's error returns.
//!
//! COLMAP throws; library Rust returns. Each macro evaluates its operands exactly once and,
//! when the check fails, does `return Err(From::from(ColmapError))` from the enclosing
//! function, so it works in any function whose error type implements `From<ColmapError>`.
//! The glog `CHECK`/`LOG(FATAL)` abort family is not ported here: CLAUDE.md maps each such
//! site to an error (or a panic where COLMAP's contract is an internal invariant).
//!
//! Messages follow COLMAP's `LogMessageFatalThrow` exactly in shape (same as colmap-sharp's
//! `ColmapSharp/Util/Check.cs`):
//!
//! ```text
//! "[<file>:<line>] Check failed: <expr> <message>"                        check!
//! "[<file>:<line>] Check failed: <a> <op> <b> (<va> vs. <vb>) <message>"  check_eq! ...
//! "[<file>:<line>] '<expr>' Must be non NULL"                              check_notnull!
//! "[<file>:<line>] <message>"                                              fatal_throw!
//! ```
//!
//! Two things differ by construction: `<file>:<line>` is the Rust caller's (base name only,
//! as COLMAP's `__GetConstFileBaseName`), and `<expr>` is Rust's `stringify!` of the
//! argument, not the C++ source text. Operand values print as an `std::ostream` would
//! ([`CheckOpValue`]): floating point at the default precision of 6 (`%g`), `bool` as 1/0.
//!
//! COLMAP throws `std::invalid_argument` for all of these (`LogMessageFatalThrowDefault`),
//! which is [`ErrorKind::InvalidArgument`]; `LOG_FATAL_THROW(std::logic_error)` is
//! [`ErrorKind::LogicError`]. Ported test: `tests/util/logging.rs` (`logging_test.cc`).

use std::fmt;

use super::stream_format::{format_double, DEFAULT_PRECISION};

/// Which C++ exception type COLMAP would have thrown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorKind {
    /// `std::invalid_argument`: every `THROW_CHECK*` and `LOG(FATAL_THROW)`.
    InvalidArgument,
    /// `std::logic_error`, from `LOG_FATAL_THROW(std::logic_error)`.
    LogicError,
}

/// An error raised where COLMAP throws: a failed `THROW_CHECK*` or a `LOG(FATAL_THROW)`.
/// `Display` prints COLMAP's `what()` string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColmapError {
    kind: ErrorKind,
    message: String,
}

/// `Result` with [`ColmapError`] as the default error type.
pub type Result<T, E = ColmapError> = std::result::Result<T, E>;

impl ColmapError {
    /// An error with a caller-built message (no `[file:line]` prefix).
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }

    /// Port of `LogMessageFatalThrow`'s `what()`: `"[<base name>:<line>] " + text`, as built
    /// by COLMAP's `__MakeExceptionPrefix`. Used by the check macros.
    pub fn at(kind: ErrorKind, file: &str, line: u32, text: &str) -> Self {
        Self::new(
            kind,
            format!("[{}:{}] {}", file_base_name(file), line, text),
        )
    }

    /// The C++ exception type COLMAP would have thrown.
    pub fn kind(&self) -> ErrorKind {
        self.kind
    }

    /// COLMAP's `what()` string.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for ColmapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for ColmapError {}

/// Port of `__GetConstFileBaseName`: the part after the last '/', or failing that after the
/// last '\\', or the whole path.
pub fn file_base_name(file: &str) -> &str {
    match file.rfind('/').or_else(|| file.rfind('\\')) {
        Some(i) => &file[i + 1..],
        None => file,
    }
}

/// How a check operand prints in `(<va> vs. <vb>)`: what `std::ostream << value` writes
/// (glog's `MakeCheckOpValueString`). Implement it for any type used with `check_eq!` etc.
pub trait CheckOpValue {
    /// The operand as an `std::ostream` in its default state would print it.
    fn check_op_string(&self) -> String;
}

impl<T: CheckOpValue + ?Sized> CheckOpValue for &T {
    fn check_op_string(&self) -> String {
        (**self).check_op_string()
    }
}

macro_rules! impl_check_op_value_display {
    ($($t:ty),*) => {$(
        impl CheckOpValue for $t {
            fn check_op_string(&self) -> String {
                self.to_string()
            }
        }
    )*};
}

impl_check_op_value_display!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize, str, String
);

impl CheckOpValue for f64 {
    fn check_op_string(&self) -> String {
        format_double(*self, DEFAULT_PRECISION)
    }
}

impl CheckOpValue for f32 {
    fn check_op_string(&self) -> String {
        // An ostream widens a float to double for %g; the widening is exact.
        format_double(f64::from(*self), DEFAULT_PRECISION)
    }
}

impl CheckOpValue for bool {
    fn check_op_string(&self) -> String {
        // No std::boolalpha: an ostream prints bool as an integer.
        if *self { "1" } else { "0" }.to_string()
    }
}

impl CheckOpValue for char {
    fn check_op_string(&self) -> String {
        // glog quotes a printable char and spells out anything else.
        if (' '..='~').contains(self) {
            format!("'{self}'")
        } else {
            format!("char value {}", u32::from(*self))
        }
    }
}

/// Port of `THROW_CHECK(condition) << message`: returns `Err` with
/// `"[file:line] Check failed: <condition> <message>"` when `condition` is false. The
/// message (`format!` arguments) is only formatted on failure.
#[macro_export]
macro_rules! check {
    ($cond:expr $(,)?) => {
        $crate::check!($cond, "")
    };
    ($cond:expr, $($arg:tt)+) => {
        if !($cond) {
            let text = ::std::format!(
                "Check failed: {} {}",
                ::core::stringify!($cond),
                ::std::format!($($arg)+)
            );
            return ::core::result::Result::Err(::core::convert::From::from(
                $crate::util::check::ColmapError::at(
                    $crate::util::check::ErrorKind::InvalidArgument,
                    ::core::file!(),
                    ::core::line!(),
                    &text,
                ),
            ));
        }
    };
}

/// Shared body of `check_eq!` .. `check_ge!` (glog's `CHECK_OP_LOG`). Not for direct use.
#[doc(hidden)]
#[macro_export]
macro_rules! __check_op {
    ($op:tt, $a:expr, $b:expr $(,)?) => {
        $crate::__check_op!($op, $a, $b, "")
    };
    ($op:tt, $a:expr, $b:expr, $($arg:tt)+) => {
        match (&$a, &$b) {
            (va, vb) => {
                #[allow(clippy::float_cmp)]
                let ok = *va $op *vb;
                if !ok {
                    let text = ::std::format!(
                        "Check failed: {} {} {} ({} vs. {}) {}",
                        ::core::stringify!($a),
                        ::core::stringify!($op),
                        ::core::stringify!($b),
                        $crate::util::check::CheckOpValue::check_op_string(va),
                        $crate::util::check::CheckOpValue::check_op_string(vb),
                        ::std::format!($($arg)+)
                    );
                    return ::core::result::Result::Err(::core::convert::From::from(
                        $crate::util::check::ColmapError::at(
                            $crate::util::check::ErrorKind::InvalidArgument,
                            ::core::file!(),
                            ::core::line!(),
                            &text,
                        ),
                    ));
                }
            }
        }
    };
}

/// Port of `THROW_CHECK_EQ(a, b) << message`.
#[macro_export]
macro_rules! check_eq {
    ($($t:tt)+) => { $crate::__check_op!(==, $($t)+) };
}

/// Port of `THROW_CHECK_NE(a, b) << message`.
#[macro_export]
macro_rules! check_ne {
    ($($t:tt)+) => { $crate::__check_op!(!=, $($t)+) };
}

/// Port of `THROW_CHECK_LT(a, b) << message`. NaN fails it, as the C++ `<` does.
#[macro_export]
macro_rules! check_lt {
    ($($t:tt)+) => { $crate::__check_op!(<, $($t)+) };
}

/// Port of `THROW_CHECK_LE(a, b) << message`. NaN fails it, as the C++ `<=` does.
#[macro_export]
macro_rules! check_le {
    ($($t:tt)+) => { $crate::__check_op!(<=, $($t)+) };
}

/// Port of `THROW_CHECK_GT(a, b) << message`. NaN fails it, as the C++ `>` does.
#[macro_export]
macro_rules! check_gt {
    ($($t:tt)+) => { $crate::__check_op!(>, $($t)+) };
}

/// Port of `THROW_CHECK_GE(a, b) << message`. NaN fails it, as the C++ `>=` does.
#[macro_export]
macro_rules! check_ge {
    ($($t:tt)+) => { $crate::__check_op!(>=, $($t)+) };
}

/// Port of `THROW_CHECK_NOTNULL(val)` for `Option`: evaluates to the inner value, or returns
/// `Err` with `"[file:line] '<val>' Must be non NULL"` on `None`.
#[macro_export]
macro_rules! check_notnull {
    ($val:expr $(,)?) => {
        match $val {
            ::core::option::Option::Some(v) => v,
            ::core::option::Option::None => {
                return ::core::result::Result::Err(::core::convert::From::from(
                    $crate::util::check::ColmapError::at(
                        $crate::util::check::ErrorKind::InvalidArgument,
                        ::core::file!(),
                        ::core::line!(),
                        ::core::concat!("'", ::core::stringify!($val), "' Must be non NULL"),
                    ),
                ));
            }
        }
    };
}

/// Port of `LOG(FATAL_THROW) << message` (`std::invalid_argument`) and, with
/// `kind = <ErrorKind variant>`, of `LOG_FATAL_THROW(exception) << message`: returns `Err`
/// with `"[file:line] <message>"`.
#[macro_export]
macro_rules! fatal_throw {
    (kind = $kind:ident, $($arg:tt)+) => {
        return ::core::result::Result::Err(::core::convert::From::from(
            $crate::util::check::ColmapError::at(
                $crate::util::check::ErrorKind::$kind,
                ::core::file!(),
                ::core::line!(),
                &::std::format!($($arg)+),
            ),
        ))
    };
    ($($arg:tt)+) => {
        $crate::fatal_throw!(kind = InvalidArgument, $($arg)+)
    };
}
