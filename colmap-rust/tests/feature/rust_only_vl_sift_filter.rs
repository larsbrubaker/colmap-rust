// Rust-only tests for colmap_rust::feature::vlfeat (VLFeat's SIFT filter) against VLFeat
// itself: `sift_vlfeat.json`, written by oracle/fixture_sift.py from oracle/sift_harness.c,
// which compiles cpp-reference's VLFeat with -ffp-contract=off and drives it the way COLMAP's
// SiftCPUFeatureExtractor does. Tier A: every keypoint field, orientation and raw float
// descriptor value must be bit-identical, over the upsample (o_min -1, -2), copy (0) and
// downsample (1) scale-space paths and upright mode. VLFeat has no test suite of its own in
// COLMAP's tree, so this fixture is the specification for the port. Port of colmap-sharp's
// VlSiftFilterTests.CSharpOnly_MatchesUnfusedVLFeatExactly, one test per fixture case.
// The images come from `sift.json` (base64 grey pixels, row-major).

use colmap_rust::feature::vlfeat::mathop::{fast_expn, EXPN_MAX, EXPN_SZ, EXPN_TAB};
use colmap_rust::feature::vlfeat::sift_descriptor::VL_SIFT_DESCRIPTOR_LENGTH;
use colmap_rust::feature::vlfeat::sift_filter::VlSiftFilter;
use colmap_rust::math::fns;
use colmap_rust::util::cancellation::CancelToken;

use crate::oracle_json::Json;

const SIFT_FIXTURE: &str = include_str!("../data/oracle/sift.json");
const VLFEAT_FIXTURE: &str = include_str!("../data/oracle/sift_vlfeat.json");

/// Standard base64 (with `=` padding), as Python's `base64.b64encode` writes it.
fn decode_base64(text: &str) -> Vec<u8> {
    fn value(c: u8) -> u32 {
        match c {
            b'A'..=b'Z' => u32::from(c - b'A'),
            b'a'..=b'z' => u32::from(c - b'a') + 26,
            b'0'..=b'9' => u32::from(c - b'0') + 52,
            b'+' => 62,
            b'/' => 63,
            _ => panic!("not base64: {c}"),
        }
    }
    let bytes: Vec<u8> = text.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let mut acc = 0u32;
        for &c in chunk {
            acc = (acc << 6) | value(c);
        }
        let bits = chunk.len() * 6;
        acc <<= 24 - bits;
        for i in 0..bits / 8 {
            out.push((acc >> (16 - 8 * i)) as u8);
        }
    }
    out
}

/// Runs the filter over one fixture case and returns the first mismatch (empty when none)
/// with the number of keypoints matched.
fn run_case(name: &str) -> (String, usize, usize) {
    let sift_root = Json::parse(SIFT_FIXTURE);
    let vlfeat_root = Json::parse(VLFEAT_FIXTURE);
    let case = vlfeat_root
        .get("cases")
        .as_array()
        .iter()
        .find(|c| c.get("name").as_str() == name)
        .unwrap_or_else(|| panic!("no case {name}"));
    let image = sift_root.get("images").get(case.get("image").as_str());
    let width = image.get("width").as_f64() as i32;
    let height = image.get("height").as_f64() as i32;
    let pixels = decode_base64(image.get("pixels").as_str());
    assert_eq!(pixels.len(), (width * height) as usize);
    let upright = matches!(case.get("upright"), Json::Bool(true));

    let mut sift = VlSiftFilter::new(
        width,
        height,
        case.get("noctaves").as_f64() as i32,
        3,
        case.get("o_min").as_f64() as i32,
    );
    sift.peak_thresh = 0.02 / 3.0;
    sift.edge_thresh = 10.0;

    let image01: Vec<f32> = pixels.iter().map(|&v| f32::from(v) / 255.0f32).collect();
    let expected = case.get("keypoints").as_array();
    let cancel = CancelToken::new();

    let mut next = 0usize;
    let mut angles = [0.0f64; 4];
    let mut desc = [0.0f32; VL_SIFT_DESCRIPTOR_LENGTH];
    let mut first = true;
    loop {
        let more = if first {
            first = false;
            sift.process_first_octave(&image01, &cancel).unwrap()
        } else {
            sift.process_next_octave(&cancel).unwrap()
        };
        if !more {
            break;
        }

        sift.detect(&cancel).unwrap();
        for k in sift.keypoints().to_vec() {
            if next >= expected.len() {
                return (format!("extra keypoint {next}"), next, expected.len());
            }
            let ek = expected[next].get("k").as_f64s();
            let ints = [k.o, k.ix, k.iy, k.is];
            for (j, &v) in ints.iter().enumerate() {
                if f64::from(v) != ek[j] {
                    return (
                        format!("keypoint {next} field {j}: got {v}, expected {}", ek[j]),
                        next,
                        expected.len(),
                    );
                }
            }
            let floats = [k.x, k.y, k.s, k.sigma];
            for (j, &v) in floats.iter().enumerate() {
                if v != ek[4 + j] as f32 {
                    let e = ek[4 + j];
                    return (
                        format!("keypoint {next} field {}: got {v:?}, expected {e:?}", 4 + j),
                        next,
                        expected.len(),
                    );
                }
            }

            let num_orientations = if upright {
                angles[0] = 0.0;
                1
            } else {
                sift.calc_keypoint_orientations(&mut angles, &k)
            };

            let ea = expected[next].get("a").as_array();
            if ea.len() != num_orientations {
                let msg = format!(
                    "keypoint {next}: {num_orientations} orientations, expected {}",
                    ea.len()
                );
                return (msg, next, expected.len());
            }

            for (o, expected_angle) in ea.iter().enumerate() {
                sift.calc_keypoint_descriptor(&mut desc, &k, angles[o]);
                let want_angle = expected_angle.get("angle").as_f64();
                if angles[o] != want_angle {
                    let msg = format!(
                        "keypoint {next} angle {o}: got {:?}, expected {want_angle:?}",
                        angles[o]
                    );
                    return (msg, next, expected.len());
                }
                let bytes = decode_base64(expected_angle.get("descriptor").as_str());
                for (d, got) in desc.iter().enumerate() {
                    let b = &bytes[4 * d..4 * d + 4];
                    let want = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                    // Bit comparison, so a NaN or signed-zero difference would show too.
                    if got.to_bits() != want.to_bits() {
                        let msg = format!(
                            "keypoint {next} angle {o} bin {d}: got {got:?}, expected {want:?}"
                        );
                        return (msg, next, expected.len());
                    }
                }
            }
            next += 1;
        }
    }
    (String::new(), next, expected.len())
}

fn check_case(name: &str) {
    let (mismatch, matched, expected) = run_case(name);
    assert_eq!(mismatch, "", "{name}");
    assert_eq!(matched, expected, "{name}");
}

#[test]
fn rust_only_vl_sift_filter_matches_unfused_vlfeat_square256() {
    check_case("Square256");
}

#[test]
fn rust_only_vl_sift_filter_matches_unfused_vlfeat_texture() {
    check_case("Texture");
}

#[test]
fn rust_only_vl_sift_filter_matches_unfused_vlfeat_texture_upright() {
    check_case("TextureUpright");
}

#[test]
fn rust_only_vl_sift_filter_matches_unfused_vlfeat_texture_first_octave0() {
    check_case("TextureFirstOctave0");
}

#[test]
fn rust_only_vl_sift_filter_matches_unfused_vlfeat_texture_first_octave1() {
    check_case("TextureFirstOctave1");
}

#[test]
fn rust_only_vl_sift_filter_matches_unfused_vlfeat_texture_first_octave_minus2() {
    check_case("TextureFirstOctaveMinus2");
}

#[test]
fn rust_only_vl_sift_filter_reused_filter_matches_fresh_one() {
    // Entry 201: a filter reused for a second image gives what a fresh filter gives. With a
    // single octave, the first image's last octave with keypoints is the second image's first
    // octave, where VLFeat would keep the first image's gradient.
    let root = Json::parse(SIFT_FIXTURE);
    let image = root.get("images").get("texture");
    let width = image.get("width").as_f64() as usize;
    let height = image.get("height").as_f64() as usize;
    let pixels = decode_base64(image.get("pixels").as_str());
    let image_a: Vec<f32> = pixels.iter().map(|&v| f32::from(v) / 255.0f32).collect();
    // The same texture mirrored left to right.
    let image_b: Vec<f32> = (0..width * height)
        .map(|i| image_a[(i / width) * width + (width - 1 - i % width)])
        .collect();

    let cancel = CancelToken::new();
    let run = |sift: &mut VlSiftFilter, image: &[f32]| -> Vec<f32> {
        let mut out = Vec::new();
        let mut angles = [0.0f64; 4];
        let mut desc = [0.0f32; VL_SIFT_DESCRIPTOR_LENGTH];
        let mut more = sift.process_first_octave(image, &cancel).unwrap();
        while more {
            sift.detect(&cancel).unwrap();
            for k in sift.keypoints().to_vec() {
                let n = sift.calc_keypoint_orientations(&mut angles, &k);
                for &angle in &angles[..n] {
                    sift.calc_keypoint_descriptor(&mut desc, &k, angle);
                    out.push(angle as f32);
                    out.extend_from_slice(&desc);
                }
            }
            more = sift.process_next_octave(&cancel).unwrap();
        }
        out
    };
    let new_filter = || {
        let mut sift = VlSiftFilter::new(width as i32, height as i32, 1, 3, -1);
        sift.peak_thresh = 0.02 / 3.0;
        sift
    };
    let mut reused = new_filter();
    let first = run(&mut reused, &image_a);
    assert!(!first.is_empty());
    let second = run(&mut reused, &image_b);
    let fresh = run(&mut new_filter(), &image_b);
    assert!(!fresh.is_empty());
    assert_eq!(second.len(), fresh.len());
    assert!(second
        .iter()
        .zip(&fresh)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn rust_only_vl_sift_expn_table_is_exp() {
    // Entry 202: the constant table is exp(-k * 25 / 256), here within 1 ulp of the libm
    // crate's exp (the table holds the correctly rounded values).
    assert_eq!(EXPN_TAB.len(), EXPN_SZ + 1);
    assert_eq!(EXPN_TAB[0], 1.0);
    for (k, &t) in EXPN_TAB.iter().enumerate() {
        let e = fns::exp(-(k as f64) * (EXPN_MAX / EXPN_SZ as f64));
        let ulps = (t.to_bits() as i64 - e.to_bits() as i64).abs();
        assert!(ulps <= 1, "k {k}: table {t:e}, exp {e:e}");
    }
    assert_eq!(fast_expn(0.0), 1.0);
    assert_eq!(fast_expn(25.5), 0.0);
}

#[test]
fn rust_only_vl_sift_filter_cancelled_token_stops() {
    let cancel = CancelToken::new();
    cancel.cancel();
    let mut sift = VlSiftFilter::new(16, 16, 1, 3, 0);
    assert!(sift.process_first_octave(&[0.0; 256], &cancel).is_err());
}
