// Rust-only (not a COLMAP test): `math::random`'s Mt19937, libc++ distributions and COLMAP
// API against tests/data/oracle/random.json, which oracle/fixture_random.py records from
// libc++'s <random> on macOS (the standard library the pycolmap wheel links). Tier A: every
// draw must be bit-identical. random_test.cc (ported 1:1 in random.rs) only checks ranges
// and moments; this is what pins the actual numbers. Port of colmap-sharp's
// RandomOracleTests.cs.
//
// The fixture's "cases" come from a -ffp-contract=off build, the arithmetic colmap-rust
// does. Its "contracted_cases" (Apple clang's default fused multiply-adds) are not asserted:
// docs/CPP_DIVERGENCES.md, entry 40. The fixture is embedded with include_str! so the
// wasm32-wasip1 run needs no filesystem access.

use colmap_rust::math::random::libcxx::{self, NormalDistribution};
use colmap_rust::math::random::{
    random_gaussian, random_uniform_integer, random_uniform_real, set_prng_seed, shuffle,
    with_prng, Mt19937,
};

const FIXTURE: &str = include_str!("../data/oracle/random.json");
const SEEDS: [u32; 4] = [0, 1, 42, 4294967295];

// The f64 Gaussian cases call `log`, which goes through `math::fns` (the `libm` crate) where
// the harness calls Apple libm: docs/CPP_DIVERGENCES.md, entry 1. A last-ulp difference in
// log(s) moves the drawn value by a few ulp (the accept/reject sequence does not depend on
// log, so the draws stay aligned). These cases are compared within this absolute tolerance;
// every other case, including the f32 Gaussians (`logf` agrees on these inputs), is
// bit-exact. With std's `f64::ln` (Apple libm natively) all 80 cases match bit for bit;
// with `math::fns` the largest difference is 8.9e-16 (4 ulp at magnitude 1).
const F64_LOG_CASES: [&str; 2] = ["gaussian_double_1_1", "normal_reused_0.5_3"];
const F64_LOG_TOLERANCE: f64 = 2e-15;

// One oracle value, compared exactly: integers by value, reals by bit pattern (so -0.0 vs
// 0.0 and last-ulp differences show).
#[derive(Debug, PartialEq)]
enum Value {
    Unsigned(u64),
    Signed(i64),
    Real(u64),
}

#[derive(Clone, Copy)]
enum Kind {
    Unsigned,
    Signed,
    Real,
}

fn real(v: f64) -> Value {
    Value::Real(v.to_bits())
}

// Minimal reader for the fixture's `"cases": { "name": [numbers], ... }` object. The file is
// written by json.dumps, so tokens are plain: quoted names, and numbers that Rust's
// correctly rounded parser reads back exactly (Python writes floats with repr).
fn fixture_cases() -> Vec<(String, Vec<String>)> {
    let start = FIXTURE.find("\"cases\"").expect("cases key") + "\"cases\"".len();
    let body = &FIXTURE[start..];
    let body = &body[body.find('{').expect("cases object") + 1..];
    let mut cases = Vec::new();
    let mut rest = body;
    loop {
        rest = rest.trim_start_matches(|c: char| c.is_whitespace() || c == ',');
        if rest.starts_with('}') {
            break;
        }
        assert!(rest.starts_with('"'), "unexpected fixture text");
        let name_end = rest[1..].find('"').expect("name end") + 1;
        let name = rest[1..name_end].to_string();
        let open = rest.find('[').expect("array start");
        let close = rest.find(']').expect("array end");
        let values = rest[open + 1..close]
            .split(',')
            .map(|t| t.trim().to_string())
            .filter(|t| !t.is_empty())
            .collect();
        cases.push((name, values));
        rest = &rest[close + 1..];
    }
    cases
}

fn parse(token: &str, kind: Kind) -> Value {
    match kind {
        Kind::Unsigned => Value::Unsigned(token.parse().expect("u64")),
        Kind::Signed => Value::Signed(token.parse().expect("i64")),
        Kind::Real => real(token.parse().expect("f64")),
    }
}

fn draws(seed: u32, count: usize, mut draw: impl FnMut() -> Value) -> Vec<Value> {
    set_prng_seed(seed);
    (0..count).map(|_| draw()).collect()
}

fn colmap_shuffle(seed: u32, num_to_shuffle: u32) -> Vec<Value> {
    set_prng_seed(seed);
    let mut values: Vec<i64> = (0..20).collect();
    shuffle(num_to_shuffle, &mut values).unwrap();
    values.into_iter().map(Value::Signed).collect()
}

// Harness case name (without "/seed<n>") -> value kind and the Rust draws for one seed.
// Each producer mirrors the matching block of oracle/random_harness.cc.
fn produce(name: &str, seed: u32) -> (Kind, Vec<Value>) {
    use Kind::*;
    match name {
        "mt19937" => (
            Unsigned,
            draws(seed, 32, || {
                Value::Unsigned(u64::from(with_prng(|g| g.next_u32())))
            }),
        ),
        "int_0_10000" => (
            Signed,
            draws(seed, 64, || {
                Value::Signed(random_uniform_integer::<i32>(0, 10000).into())
            }),
        ),
        "int_m100_100" => (
            Signed,
            draws(seed, 64, || {
                Value::Signed(random_uniform_integer::<i32>(-100, 100).into())
            }),
        ),
        "int_full" => (
            Signed,
            draws(seed, 16, || {
                Value::Signed(random_uniform_integer(i32::MIN, i32::MAX).into())
            }),
        ),
        "short_m7_300" => (
            Signed,
            draws(seed, 32, || {
                Value::Signed(random_uniform_integer::<i16>(-7, 300).into())
            }),
        ),
        "uint32_0_999" => (
            Unsigned,
            draws(seed, 64, || {
                Value::Unsigned(random_uniform_integer::<u32>(0, 999).into())
            }),
        ),
        "int64_pm1e12" => (
            Signed,
            draws(seed, 32, || {
                Value::Signed(random_uniform_integer::<i64>(
                    -1_000_000_000_000,
                    1_000_000_000_000,
                ))
            }),
        ),
        "uint64_0_2pow40p3" => (
            Unsigned,
            draws(seed, 32, || {
                Value::Unsigned(random_uniform_integer::<u64>(0, (1u64 << 40) + 3))
            }),
        ),
        "uint64_full" => (
            Unsigned,
            draws(seed, 16, || {
                Value::Unsigned(random_uniform_integer::<u64>(0, u64::MAX))
            }),
        ),
        // size_t is 64-bit in the harness; usize follows the target (32-bit on wasm32), and
        // [0, 5] draws the same through either width's path.
        "size_t_0_5" => (
            Unsigned,
            draws(seed, 32, || {
                Value::Unsigned(random_uniform_integer::<usize>(0, 5) as u64)
            }),
        ),
        "double_m100_100" => (
            Real,
            draws(seed, 32, || real(random_uniform_real(-100.0f64, 100.0))),
        ),
        "double_0_1" => (
            Real,
            draws(seed, 32, || real(random_uniform_real(0.0f64, 1.0))),
        ),
        // A float widens to double exactly, as the harness's printf("%a") does.
        "float_m1_1" => (
            Real,
            draws(seed, 32, || real(random_uniform_real(-1.0f32, 1.0).into())),
        ),
        "float_0_1000" => (
            Real,
            draws(seed, 32, || {
                real(random_uniform_real(0.0f32, 1000.0).into())
            }),
        ),
        "gaussian_double_1_1" => (Real, draws(seed, 32, || real(random_gaussian(1.0f64, 1.0)))),
        "gaussian_float_0_2" => (
            Real,
            draws(seed, 32, || real(random_gaussian(0.0f32, 2.0).into())),
        ),
        "normal_reused_0.5_3" => {
            // One distribution reused: exercises libc++'s cached second polar value.
            let mut distribution = NormalDistribution::new(0.5f64, 3.0);
            (
                Real,
                draws(seed, 32, || real(with_prng(|g| distribution.sample(g)))),
            )
        }
        "colmap_shuffle_5_of_20" => (Signed, colmap_shuffle(seed, 5)),
        "colmap_shuffle_20_of_20" => (Signed, colmap_shuffle(seed, 20)),
        "std_shuffle_30" => {
            set_prng_seed(seed);
            let mut values: Vec<i64> = (0..30).collect();
            with_prng(|g| libcxx::shuffle(&mut values, g));
            (Signed, values.into_iter().map(Value::Signed).collect())
        }
        other => panic!("fixture case {other} has no Rust producer"),
    }
}

#[test]
fn rust_only_mt19937_default_seed_10000th_draw() {
    let mut engine = Mt19937::default();
    for _ in 1..10000 {
        engine.next_u32();
    }
    let draw = engine.next_u32();
    // The C++ standard pins this value ([rand.predef]); the fixture confirms libc++ agrees.
    assert_eq!(draw, 4123659995);
    let cases = fixture_cases();
    let (_, expected) = cases
        .iter()
        .find(|(n, _)| n == "mt19937_default_10000th")
        .expect("case");
    assert_eq!(expected, &vec![draw.to_string()]);
}

#[test]
fn rust_only_all_cases_match_libcxx_bit_for_bit() {
    let mut mismatches = Vec::new();
    let mut compared = 0;
    let mut max_log_diff = 0.0f64;
    let cases = fixture_cases();
    for (key, tokens) in &cases {
        if key == "mt19937_default_10000th" {
            continue;
        }
        let (name, seed) = key.rsplit_once("/seed").expect("name/seed<n>");
        let seed: u32 = seed.parse().expect("seed");
        assert!(SEEDS.contains(&seed));
        let (kind, actual) = produce(name, seed);
        let expected: Vec<Value> = tokens.iter().map(|t| parse(t, kind)).collect();
        compared += 1;
        let tolerant = F64_LOG_CASES.contains(&name);
        let mut matches = |e: &Value, a: &Value| match (e, a) {
            (Value::Real(e), Value::Real(a)) if tolerant => {
                let (e, a) = (f64::from_bits(*e), f64::from_bits(*a));
                let diff = (e - a).abs();
                if diff > max_log_diff {
                    max_log_diff = diff;
                }
                diff <= F64_LOG_TOLERANCE
            }
            _ => e == a,
        };
        if expected.len() != actual.len() {
            mismatches.push(format!(
                "{key}: {} values vs {}",
                expected.len(),
                actual.len()
            ));
        } else if let Some(i) = (0..expected.len()).find(|&i| !matches(&expected[i], &actual[i])) {
            mismatches.push(format!(
                "{key}: first difference at [{i}]: expected {:?}, got {:?}",
                expected[i], actual[i]
            ));
        }
    }
    eprintln!("largest |difference| in the f64 log cases: {max_log_diff:e}");
    assert!(mismatches.is_empty(), "{}", mismatches.join("\n"));
    // 20 cases x 4 seeds: every harness case is compared.
    assert_eq!(compared, 80);
}
