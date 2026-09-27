// Rust-only probe (no COLMAP counterpart): the cross-platform guarantee for
// `colmap_rust::math::fns`. It pins the exact bits of every f64 and f32 function over a
// fixed, broad input set against checked-in tables (`tests/data/fns_probe_expected.txt` and
// `fns_probe_expected_f32.txt`), so any target whose results differ fails here with a
// per-function count and examples. CI runs it natively on Linux, macOS and Windows and as a
// `wasm32-wasip1` build under wasmtime, which enforces native == wasm on every push. The
// tables are embedded with `include_str!` so the wasm run needs no filesystem access.
//
// Why it exists: std's transcendentals call the platform C libm natively (Apple libm, glibc,
// UCRT) and compiler-builtins' musl port on wasm32, and differed (490 of 7154 f64 probes,
// 1-2 ulp, wasm vs macOS). `math::fns` therefore uses the pure-Rust `libm` crate, which is
// bit-identical native and wasm (docs/CPP_DIVERGENCES.md, entry 1).
//
// The inputs are built from bits and basic arithmetic only, never from the functions under
// test. Regenerate after changing the inputs or the backend:
//   COLMAP_REGENERATE_FNS_PROBE=1 cargo test -p colmap-rust --test math -- --test-threads=1

use colmap_rust::math::fns;
use std::fmt::Write as _;
use std::path::PathBuf;

/// Every probed function, by the name used in the table.
const FUNCTIONS: [&str; 15] = [
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "exp", "ln", "log2", "log10", "pow",
    "sqrt", "cbrt", "hypot",
];

fn eval(name: &str, x: f64, y: f64) -> f64 {
    match name {
        "sin" => fns::sin(x),
        "cos" => fns::cos(x),
        "tan" => fns::tan(x),
        "asin" => fns::asin(x),
        "acos" => fns::acos(x),
        "atan" => fns::atan(x),
        "atan2" => fns::atan2(x, y),
        "exp" => fns::exp(x),
        "ln" => fns::ln(x),
        "log2" => fns::log2(x),
        "log10" => fns::log10(x),
        "pow" => fns::pow(x, y),
        "sqrt" => fns::sqrt(x),
        "cbrt" => fns::cbrt(x),
        "hypot" => fns::hypot(x, y),
        _ => panic!("unknown probe function {name}"),
    }
}

/// Every probed f32 function, by the name used in the f32 table.
const FUNCTIONS_F32: [&str; 15] = [
    "sinf", "cosf", "tanf", "asinf", "acosf", "atanf", "atan2f", "expf", "logf", "log2f", "log10f",
    "powf", "sqrtf", "cbrtf", "hypotf",
];

fn eval_f32(name: &str, x: f32, y: f32) -> f32 {
    match name {
        "sinf" => fns::sinf(x),
        "cosf" => fns::cosf(x),
        "tanf" => fns::tanf(x),
        "asinf" => fns::asinf(x),
        "acosf" => fns::acosf(x),
        "atanf" => fns::atanf(x),
        "atan2f" => fns::atan2f(x, y),
        "expf" => fns::expf(x),
        "logf" => fns::logf(x),
        "log2f" => fns::log2f(x),
        "log10f" => fns::log10f(x),
        "powf" => fns::powf(x, y),
        "sqrtf" => fns::sqrtf(x),
        "cbrtf" => fns::cbrtf(x),
        "hypotf" => fns::hypotf(x, y),
        _ => panic!("unknown f32 probe function {name}"),
    }
}

/// splitmix64: a tiny deterministic generator so the inputs are reproducible anywhere.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [lo, hi) from the top 53 bits.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        lo + (hi - lo) * unit
    }

    /// A positive double with a uniformly drawn binary exponent in [min_exp, max_exp] and a
    /// random mantissa: covers many binades evenly. Built from bits, not `ln`/`exp`, so the
    /// inputs themselves never depend on the libm under test.
    fn random_binade(&mut self, min_exp: i64, max_exp: i64) -> f64 {
        let span = (max_exp - min_exp + 1) as u64;
        let exponent = min_exp + (self.next_u64() % span) as i64;
        let mantissa = self.next_u64() >> 12;
        f64::from_bits((((exponent + 1023) as u64) << 52) | mantissa)
    }
}

/// Hand-picked special and typical values, with their negations.
fn special_values() -> Vec<f64> {
    use std::f64::consts::{E, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, PI, TAU};
    let positive = [
        0.0,
        5e-324,
        2.2250738585072014e-308,
        1e-300,
        1e-10,
        1e-5,
        0.1,
        0.25,
        0.5,
        0.7,
        0.9999999999999999,
        1.0,
        1.0000000000000002,
        1.5,
        2.0,
        E,
        3.0,
        FRAC_PI_6,
        FRAC_PI_4,
        FRAC_PI_3,
        FRAC_PI_2,
        PI,
        TAU,
        10.0,
        100.0,
        709.78,
        1e3,
        1e5,
        1e10,
        1e22,
        1e300,
        f64::MAX,
        f64::INFINITY,
    ];
    let mut values: Vec<f64> = positive.to_vec();
    values.extend(positive.iter().map(|v| -v));
    values.push(f64::NAN);
    values
}

/// Inputs (x, y) for one function: the special values plus 200 random ones drawn from the
/// range where COLMAP actually calls it. `y` is 0 for unary functions.
fn inputs(name: &str) -> Vec<(f64, f64)> {
    let seed = 0x00C0_1A4F_0000_0000
        ^ name
            .bytes()
            .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
    let mut rng = SplitMix64(seed);
    let special = special_values();
    let mut out = Vec::new();
    let binary = matches!(name, "atan2" | "pow" | "hypot");
    if binary {
        for &x in &special {
            for &y in &[
                0.0,
                -0.0,
                0.5,
                1.0,
                -1.0,
                2.0,
                3.0,
                0.1,
                -2.5,
                f64::INFINITY,
            ] {
                out.push((x, y));
                if name != "pow" {
                    out.push((y, x));
                }
            }
        }
    } else {
        out.extend(special.iter().map(|&x| (x, 0.0)));
    }
    for i in 0..200 {
        let pair = match name {
            "sin" | "cos" | "tan" if i % 4 == 3 => (rng.uniform(-1e6, 1e6), 0.0),
            "sin" | "cos" | "tan" | "atan" => (rng.uniform(-10.0, 10.0), 0.0),
            "asin" | "acos" => (rng.uniform(-1.0, 1.0), 0.0),
            "exp" if i % 2 == 1 => (rng.uniform(-745.0, 709.0), 0.0),
            "exp" => (rng.uniform(-20.0, 20.0), 0.0),
            "ln" | "log2" | "log10" | "sqrt" if i % 4 == 3 => (rng.uniform(0.5, 2.0), 0.0),
            "ln" | "log2" | "log10" | "sqrt" => (rng.random_binade(-1000, 1000), 0.0),
            "cbrt" => (rng.uniform(-1e6, 1e6), 0.0),
            "pow" => (rng.uniform(0.0, 100.0), rng.uniform(-10.0, 10.0)),
            "atan2" | "hypot" => (rng.uniform(-100.0, 100.0), rng.uniform(-100.0, 100.0)),
            _ => panic!("no input range for {name}"),
        };
        out.push(pair);
    }
    out
}

/// Inputs for one f32 function: its f64 counterpart's inputs rounded to f32 (`as` rounds to
/// nearest, exactly, everywhere), plus 100 values from f32-specific ranges where the f64 ones
/// would mostly overflow or underflow.
fn inputs_f32(name: &str) -> Vec<(f32, f32)> {
    let base = match name {
        "logf" => "ln",
        _ => name.strip_suffix('f').expect("f32 names end in 'f'"),
    };
    let mut out: Vec<(f32, f32)> = inputs(base)
        .into_iter()
        .map(|(x, y)| (x as f32, y as f32))
        .collect();
    let seed = 0x00F3_2F32_0000_0000
        ^ name
            .bytes()
            .fold(0u64, |h, b| h.wrapping_mul(31).wrapping_add(u64::from(b)));
    let mut rng = SplitMix64(seed);
    for _ in 0..100 {
        let pair = match base {
            "exp" => (rng.uniform(-104.0, 89.0), 0.0),
            "ln" | "log2" | "log10" | "sqrt" => (rng.random_binade(-126, 127), 0.0),
            "pow" => (rng.uniform(0.0, 10.0), rng.uniform(-30.0, 30.0)),
            "asin" | "acos" => (rng.uniform(-1.0, 1.0), 0.0),
            "atan2" | "hypot" => (rng.uniform(-1e3, 1e3), rng.uniform(-1e3, 1e3)),
            _ => (rng.uniform(-100.0, 100.0), 0.0),
        };
        out.push((pair.0 as f32, pair.1 as f32));
    }
    out
}

/// One probe: function name, input bits and result bits (f64 or f32 bits widened to u64).
struct Probe {
    name: &'static str,
    x: u64,
    y: u64,
    result: u64,
}

/// A probe table: its file, its embedded contents, bit width, and how to evaluate one row.
struct Table {
    file: &'static str,
    /// The checked-in table, embedded at compile time so the check also runs on wasm32-wasip1
    /// without a preopened directory.
    text: &'static str,
    hex_width: usize,
    probes: fn() -> Vec<Probe>,
    is_nan: fn(u64) -> bool,
}

fn probes_f64() -> Vec<Probe> {
    FUNCTIONS
        .iter()
        .flat_map(|&name| {
            inputs(name).into_iter().map(move |(x, y)| Probe {
                name,
                x: x.to_bits(),
                y: y.to_bits(),
                result: eval(name, x, y).to_bits(),
            })
        })
        .collect()
}

fn probes_f32() -> Vec<Probe> {
    FUNCTIONS_F32
        .iter()
        .flat_map(|&name| {
            inputs_f32(name).into_iter().map(move |(x, y)| Probe {
                name,
                x: u64::from(x.to_bits()),
                y: u64::from(y.to_bits()),
                result: u64::from(eval_f32(name, x, y).to_bits()),
            })
        })
        .collect()
}

const F64_TABLE: Table = Table {
    file: "fns_probe_expected.txt",
    text: include_str!("../data/fns_probe_expected.txt"),
    hex_width: 16,
    probes: probes_f64,
    is_nan: |b| f64::from_bits(b).is_nan(),
};

const F32_TABLE: Table = Table {
    file: "fns_probe_expected_f32.txt",
    text: include_str!("../data/fns_probe_expected_f32.txt"),
    hex_width: 8,
    probes: probes_f32,
    is_nan: |b| f32::from_bits(b as u32).is_nan(),
};

/// Where regeneration writes the table (native only; the check itself reads `Table::text`).
fn table_path(table: &Table) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/data")
        .join(table.file)
}

fn render(table: &Table, probes: &[Probe]) -> String {
    let w = table.hex_width;
    let mut text = format!(
        "# math::fns probe ({} bits): <fn> <x bits> <y bits> <result bits> in hex.\n\
         # Backend: the libm crate. Generated by tests/math/fns_probe.rs.\n",
        w * 4
    );
    for p in probes {
        writeln!(
            text,
            "{} {:0w$x} {:0w$x} {:0w$x}",
            p.name, p.x, p.y, p.result
        )
        .expect("writing to a String cannot fail");
    }
    text
}

/// Rows of a table file, comments skipped.
fn read_rows(table: &Table) -> Vec<Vec<String>> {
    table
        .text
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.split(' ').map(str::to_string).collect())
        .collect()
}

fn check_table(table: &Table) {
    let current = (table.probes)();
    if std::env::var_os("COLMAP_REGENERATE_FNS_PROBE").is_some() {
        std::fs::write(table_path(table), render(table, &current)).expect("write table");
        return;
    }
    let rows = read_rows(table);
    assert!(rows.len() > 2000, "{}: too few probes", table.file);
    assert_eq!(
        rows.len(),
        current.len(),
        "{}: the input set changed; regenerate the table",
        table.file
    );
    let hex = |s: &str| u64::from_str_radix(s, 16).expect("hex bits");
    let mut mismatches = Vec::new();
    let mut per_function = std::collections::BTreeMap::<&str, usize>::new();
    for (row, probe) in rows.iter().zip(&current) {
        assert_eq!(row.len(), 4, "{}: malformed row {row:?}", table.file);
        assert!(
            row[0] == probe.name && hex(&row[1]) == probe.x && hex(&row[2]) == probe.y,
            "{}: the input set changed at {row:?}; regenerate the table",
            table.file
        );
        let want = hex(&row[3]);
        // NaN payloads and signs are not part of any contract we port; any NaN matches.
        let same = want == probe.result || ((table.is_nan)(want) && (table.is_nan)(probe.result));
        if !same {
            *per_function.entry(probe.name).or_default() += 1;
            mismatches.push(format!(
                "  {}(x={:x}, y={:x}): got {:x}, table {want:x} ({} ulp)",
                probe.name,
                probe.x,
                probe.y,
                probe.result,
                probe.result.abs_diff(want)
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "math::fns differs from {} on {} of {} probes on {}-{}.\n\
         Per function: {per_function:?}\nFirst mismatches:\n{}",
        table.file,
        mismatches.len(),
        rows.len(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        mismatches[..mismatches.len().min(40)].join("\n")
    );
}

#[test]
fn rust_only_fns_probe_f64_matches_table() {
    check_table(&F64_TABLE);
}

#[test]
fn rust_only_fns_probe_f32_matches_table() {
    check_table(&F32_TABLE);
}
