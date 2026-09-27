// Rust-only probe (no COLMAP counterpart): pins the exact bits of every function in
// `colmap_rust::math::fns` over a fixed, broad input set, against a table generated on
// macOS aarch64 (`tests/data/fns_probe_expected.txt`, Apple libm behind std).
//
// Purpose: decide the `math::fns` backend (PORTING_PLAN.md Phase 0). std calls the
// platform C libm natively (Apple libm, glibc, UCRT) and compiler-builtins' musl port on
// wasm32, so the same Rust code can round differently per target. CI runs this on Linux,
// macOS and Windows; any disagreement fails here with a per-function count and examples.
// The same table was checked against a `wasm32-unknown-unknown` build of these functions
// run under node: 490 of 7154 probes differ by 1-2 ulp (details in the Phase 0a commit
// message).
//
// Regenerate (on macOS only, since COLMAP/pycolmap oracles are macOS builds):
//   COLMAP_REGENERATE_FNS_PROBE=1 cargo test -p colmap-rust --test math rust_only_fns_probe

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

fn table_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/fns_probe_expected.txt")
}

/// The table: `<fn> <x bits> <y bits> <result bits>` in hex, one probe per line.
fn generate_table() -> String {
    let mut table = String::from(
        "# math::fns probe: <fn> <x bits> <y bits> <result bits>, f64 bit patterns in hex.\n\
         # Generated on macOS aarch64 (Apple libm behind std) by tests/math/fns_probe.rs.\n",
    );
    for name in FUNCTIONS {
        for (x, y) in inputs(name) {
            let r = eval(name, x, y);
            writeln!(
                table,
                "{name} {:016x} {:016x} {:016x}",
                x.to_bits(),
                y.to_bits(),
                r.to_bits()
            )
            .expect("writing to a String cannot fail");
        }
    }
    table
}

/// Distance in units in the last place between two finite doubles of the same sign.
fn ulps(a: f64, b: f64) -> u64 {
    let key = |v: f64| {
        let bits = v.to_bits() as i64;
        if bits < 0 {
            i64::MIN - bits
        } else {
            bits
        }
    };
    key(a).abs_diff(key(b))
}

#[test]
fn rust_only_fns_probe_matches_macos_table() {
    if std::env::var_os("COLMAP_REGENERATE_FNS_PROBE").is_some() {
        std::fs::write(table_path(), generate_table()).expect("write probe table");
        return;
    }
    let expected = std::fs::read_to_string(table_path()).expect("read probe table");
    let mut total = 0usize;
    let mut mismatches: Vec<String> = Vec::new();
    let mut per_function = std::collections::BTreeMap::<&str, usize>::new();
    for line in expected
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
    {
        let fields: Vec<&str> = line.split(' ').collect();
        assert_eq!(fields.len(), 4, "malformed probe line: {line}");
        let bits = |s: &str| u64::from_str_radix(s, 16).expect("hex bits");
        let name = FUNCTIONS
            .iter()
            .copied()
            .find(|f| *f == fields[0])
            .expect("known function");
        let (x, y) = (
            f64::from_bits(bits(fields[1])),
            f64::from_bits(bits(fields[2])),
        );
        let want = f64::from_bits(bits(fields[3]));
        let got = eval(name, x, y);
        total += 1;
        // NaN payloads and signs are not part of any contract we port; any NaN matches.
        let same = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
        if !same {
            *per_function.entry(name).or_default() += 1;
            mismatches.push(format!(
                "  {name}({x:e}, {y:e}): got {got:e} ({:016x}), macOS {want:e} ({:016x}), {} ulp",
                got.to_bits(),
                want.to_bits(),
                ulps(got, want)
            ));
        }
    }
    assert!(total > 3000, "probe table too small: {total} lines");
    assert!(
        mismatches.is_empty(),
        "math::fns differs from the macOS table on {} of {total} probes on {}-{}.\n\
         Per function: {per_function:?}\nFirst mismatches:\n{}",
        mismatches.len(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        mismatches[..mismatches.len().min(40)].join("\n")
    );
}

#[test]
fn rust_only_fns_probe_table_covers_current_inputs() {
    // The table must be regenerated when the input set changes; this catches forgetting to.
    let expected = std::fs::read_to_string(table_path()).expect("read probe table");
    let inputs_in_table: Vec<String> = expected
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| l.rsplit_once(' ').expect("4 fields").0.to_string())
        .collect();
    let current: Vec<String> = FUNCTIONS
        .iter()
        .flat_map(|&name| {
            inputs(name)
                .into_iter()
                .map(move |(x, y)| format!("{name} {:016x} {:016x}", x.to_bits(), y.to_bits()))
        })
        .collect();
    assert_eq!(inputs_in_table, current);
}
