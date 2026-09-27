// Port of COLMAP's src/colmap/util/logging_test.cc (all three cases), plus Rust-only tests
// pinning the message shapes of colmap_rust::util::check against COLMAP's
// LogMessageFatalThrow (the same shapes colmap-sharp's CheckTests.cs pins).
//
// COLMAP's EXPECT_THROW(..., std::invalid_argument) becomes "returns Err with
// ErrorKind::InvalidArgument", since the check macros return errors instead of throwing.

use colmap_rust::{
    check, check_eq, check_ge, check_gt, check_le, check_lt, check_ne, check_notnull, fatal_throw,
    ColmapError, ErrorKind, Result,
};

fn printing_fn(message: &str) -> Result<String> {
    if message.is_empty() {
        fatal_throw!("Error in PrintingFn");
    }
    Ok(message.to_string())
}

fn throw_check(cond: bool) -> Result<()> {
    check!(cond, "Error!");
    Ok(())
}

fn throw_check_equal(val: i32) -> Result<()> {
    check_eq!(val, 1, "Error!");
    Ok(())
}

fn throw_check_notnull(val: Option<&i32>) -> Result<i32> {
    Ok(*check_notnull!(val))
}

fn log_fatal_throw() -> Result<()> {
    fatal_throw!("Error!");
}

fn log_fatal_throw_logic_error() -> Result<()> {
    fatal_throw!(kind = LogicError, "Error!");
}

fn kind_of<T: std::fmt::Debug>(result: Result<T>) -> ErrorKind {
    result.expect_err("expected an error").kind()
}

fn message_of<T: std::fmt::Debug>(result: Result<T>) -> String {
    result.expect_err("expected an error").message().to_string()
}

#[test]
fn exception_logging_nominal() {
    assert!(throw_check(true).is_ok());
    assert_eq!(kind_of(throw_check(false)), ErrorKind::InvalidArgument);
    assert!(throw_check_equal(1).is_ok());
    assert_eq!(kind_of(throw_check_equal(0)), ErrorKind::InvalidArgument);
    assert_eq!(
        kind_of(throw_check_notnull(None)),
        ErrorKind::InvalidArgument
    );
    assert_eq!(kind_of(log_fatal_throw()), ErrorKind::InvalidArgument);
    assert_eq!(
        kind_of(log_fatal_throw_logic_error()),
        ErrorKind::LogicError
    );
}

// Ensure that the condition is evaluated exactly once
// for both the positive and negative cases.
#[test]
fn exception_logging_num_condition_evals() {
    let mut num_calls = 0;
    let mut func = || {
        num_calls += 1;
        true
    };
    let mut run = |negate: bool| -> Result<()> {
        if negate {
            check!(!func());
        } else {
            check!(func());
        }
        Ok(())
    };
    assert!(run(false).is_ok());
    assert!(run(true).is_err());
    assert_eq!(num_calls, 2);
}

#[test]
fn exception_logging_nested() {
    assert!(printing_fn("message").is_ok());
    assert_eq!(kind_of(printing_fn("")), ErrorKind::InvalidArgument);
    // LOG(FATAL_THROW) << "Error: " << PrintingFn(...): the inner error propagates when the
    // inner call throws, and the outer one fires otherwise.
    let outer = |inner: &str| -> Result<()> {
        let text = printing_fn(inner)?;
        fatal_throw!("Error: {text}");
    };
    assert_eq!(kind_of(outer("message")), ErrorKind::InvalidArgument);
    assert!(message_of(outer("message")).ends_with("] Error: message"));
    assert_eq!(kind_of(outer("")), ErrorKind::InvalidArgument);
    assert!(message_of(outer("")).ends_with("] Error in PrintingFn"));
}

#[test]
fn rust_only_check_failure_message_matches_colmap() {
    let elems: Vec<i32> = Vec::new();
    let f = || -> Result<()> {
        check!(!elems.is_empty(), "extra {}", 7);
        Ok(())
    };
    let line = line!() - 3;
    assert_eq!(
        message_of(f()),
        format!("[logging.rs:{line}] Check failed: !elems.is_empty() extra 7")
    );
    let bare = |x: i32| -> Result<()> {
        check!(x < 0);
        Ok(())
    };
    let line = line!() - 3;
    assert_eq!(
        message_of(bare(1)),
        format!("[logging.rs:{line}] Check failed: x < 0 ")
    );
}

#[test]
fn rust_only_check_message_is_formatted_only_on_failure() {
    let mut formatted = 0;
    let mut counted = || {
        formatted += 1;
        "formatted"
    };
    let mut f = |cond: bool| -> Result<()> {
        check!(cond, "{}", counted());
        Ok(())
    };
    assert!(f(true).is_ok());
    assert!(f(false).is_err());
    assert_eq!(formatted, 1);
}

#[test]
fn rust_only_check_ge_failure_message_matches_colmap() {
    let p = -1.5;
    let f = || -> Result<()> {
        check_ge!(p, 0.0);
        Ok(())
    };
    let line = line!() - 3;
    assert_eq!(
        message_of(f()),
        format!("[logging.rs:{line}] Check failed: p >= 0.0 (-1.5 vs. 0) ")
    );
}

#[test]
fn rust_only_check_lt_formats_doubles_with_six_significant_digits() {
    // An ostream's default precision is 6 significant digits, so glog prints 1.23457.
    let x = 1.23456789;
    let f = || -> Result<()> {
        check_lt!(x, 1.0, "at {}", "x");
        Ok(())
    };
    let line = line!() - 3;
    assert_eq!(
        message_of(f()),
        format!("[logging.rs:{line}] Check failed: x < 1.0 (1.23457 vs. 1) at x")
    );
}

#[test]
fn rust_only_check_ops_pass_and_fail_like_their_operators() {
    fn run(f: impl Fn() -> Result<()>) -> bool {
        f().is_ok()
    }
    assert!(run(|| {
        check_eq!(1, 1);
        check_ne!(1, 2);
        check_lt!(1, 2);
        check_le!(2, 2);
        check_gt!(2, 1);
        check_ge!(2, 2);
        Ok(())
    }));
    assert!(!run(|| {
        check_eq!(1, 2);
        Ok(())
    }));
    assert!(!run(|| {
        check_ne!(1, 1);
        Ok(())
    }));
    assert!(!run(|| {
        check_lt!(2, 2);
        Ok(())
    }));
    assert!(!run(|| {
        check_le!(3, 2);
        Ok(())
    }));
    assert!(!run(|| {
        check_gt!(2, 2);
        Ok(())
    }));
    assert!(!run(|| {
        check_ge!(1, 2);
        Ok(())
    }));
    // NaN fails every ordered comparison, as the C++ operators do.
    assert!(!run(|| {
        check_ge!(f64::NAN, 0.0);
        Ok(())
    }));
}

#[test]
fn rust_only_check_op_values_print_like_an_ostream() {
    let f = || -> Result<()> {
        check_eq!(true, false);
        Ok(())
    };
    assert!(message_of(f()).ends_with("Check failed: true == false (1 vs. 0) "));
    let g = || -> Result<()> {
        check_eq!('a', 'b');
        Ok(())
    };
    assert!(message_of(g()).ends_with("Check failed: 'a' == 'b' ('a' vs. 'b') "));
    let h = || -> Result<()> {
        check_eq!(2.5f32, 1e-7f32);
        Ok(())
    };
    assert!(message_of(h()).ends_with("(2.5 vs. 1e-07) "));
}

#[test]
fn rust_only_check_notnull_returns_value_or_errs() {
    let present = Some(3);
    assert_eq!(throw_check_notnull(present.as_ref()).ok(), Some(3));
    let missing: Option<&i32> = None;
    let f = || -> Result<i32> { Ok(*check_notnull!(missing)) };
    let line = line!() - 1;
    assert_eq!(
        message_of(f()),
        format!("[logging.rs:{line}] 'missing' Must be non NULL")
    );
}

#[test]
fn rust_only_check_converts_into_callers_error_type() {
    #[derive(Debug)]
    struct AppError(ColmapError);
    impl From<ColmapError> for AppError {
        fn from(e: ColmapError) -> Self {
            AppError(e)
        }
    }
    let f = || -> std::result::Result<(), AppError> {
        check_gt!(0, 1);
        Ok(())
    };
    let err = f().expect_err("check must fail");
    assert!(err.0.to_string().contains("Check failed: 0 > 1 (0 vs. 1) "));
}

#[test]
fn rust_only_file_base_name_matches_colmap() {
    use colmap_rust::util::check::file_base_name;
    assert_eq!(file_base_name("a/b/c.rs"), "c.rs");
    assert_eq!(file_base_name("a\\b\\c.rs"), "c.rs");
    assert_eq!(file_base_name("c.rs"), "c.rs");
    // COLMAP looks for '/' first and only falls back to '\\' when there is none.
    assert_eq!(file_base_name("a\\b/c\\d.rs"), "c\\d.rs");
}
