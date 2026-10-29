// Rust-only tests for colmap_rust::feature::vlfeat::covdet (VLFeat's covariant detector) against
// VLFeat itself: `covariant_sift_vlfeat.json`, written by oracle/fixture_covariant_sift.py from
// oracle/sift_harness.c's covdet mode (cpp-reference's VLFeat compiled with -ffp-contract=off,
// driven the way COLMAP's CovariantSiftCPUFeatureExtractor does with the default
// SiftExtractionOptions). Tier A: every feature (order, octave, level, frame, peak / edge /
// orientation scores) and every raw float descriptor of every domain-size-pooling scale must be
// bit-identical, over plain DoG, affine adaptation, upright, domain-size pooling and a first
// octave of 0, except that a frame matrix entry may be off by one float ulp of the frame's
// scale (libm, see `math::fns` and run_case). VLFeat has no test suite in COLMAP's tree, so this fixture is the specification.
// Port of colmap-sharp's VlCovDetTests.CSharpOnly_MatchesUnfusedVLFeatExactly, one test per
// fixture case. The per-scale descriptor loop is the harness's (COLMAP's sift.cc loop minus the
// pooling and normalization) over the production patch, gradient and raw-descriptor functions;
// the extractor's own loop is not ported yet. The images come from `sift.json`.

use colmap_rust::feature::vlfeat::covdet::{VlCovDet, VlCovDetFeature, VlFrameOrientedEllipse};
use colmap_rust::feature::vlfeat::imopv_smooth::{
    vl_imgradient_polar_f, GradientLayout, ImageRegion,
};
use colmap_rust::feature::vlfeat::sift_descriptor::VL_SIFT_DESCRIPTOR_LENGTH;
use colmap_rust::feature::vlfeat::sift_filter::VlSiftFilter;
use colmap_rust::feature::vlfeat::sift_raw_descriptor::RawDescriptorFrame;
use colmap_rust::util::cancellation::CancelToken;
use colmap_rust::util::check::ErrorKind;

use crate::oracle_json::Json;
use crate::rust_only_vl_sift_filter::decode_base64;

const SIFT_FIXTURE: &str = include_str!("../data/oracle/sift.json");
const COVDET_FIXTURE: &str = include_str!("../data/oracle/covariant_sift_vlfeat.json");

// sift.cc's covariant descriptor patch.
const PATCH_RESOLUTION: usize = 15;
const PATCH_SIDE: usize = 2 * PATCH_RESOLUTION + 1;
const PATCH_RELATIVE_EXTENT: f64 = 7.5;
const PATCH_RELATIVE_SMOOTHING: f64 = 1.0;
const PATCH_STEP: f64 = PATCH_RELATIVE_EXTENT / PATCH_RESOLUTION as f64;
const SIGMA: f64 = PATCH_RELATIVE_EXTENT / (3.0 * (4 + 1) as f64 / 2.0) / PATCH_STEP;

/// The grey image `name` of `sift.json` as VLFeat's [0, 1] floats, with its size.
fn load_image(name: &str) -> (Vec<f32>, usize, usize) {
    let root = Json::parse(SIFT_FIXTURE);
    let image = root.get("images").get(name);
    let width = image.get("width").as_f64() as usize;
    let height = image.get("height").as_f64() as usize;
    let pixels = decode_base64(image.get("pixels").as_str());
    assert_eq!(pixels.len(), width * height);
    let image01 = pixels.iter().map(|&v| f32::from(v) / 255.0f32).collect();
    (image01, width, height)
}

/// COLMAP's covdet setup with the default SiftExtractionOptions.
fn detector(first_octave: i32) -> VlCovDet {
    let mut covdet = VlCovDet::new();
    covdet.first_octave = first_octave;
    covdet.octave_resolution = 3;
    covdet.peak_threshold = 0.02 / 3.0;
    covdet.edge_threshold = 10.0;
    covdet
}

/// The harness's per-scale loop: the raw descriptor of each DSP scale of `frame`.
fn raw_descriptors(
    covdet: &mut VlCovDet,
    sift: &VlSiftFilter,
    frame: &VlFrameOrientedEllipse,
    dsp_num_scales: usize,
    dsp: bool,
) -> Vec<[f32; VL_SIFT_DESCRIPTOR_LENGTH]> {
    // COLMAP keeps these as float; `min + s * step` is a float sum stored in a double.
    let dsp_min_scale = if dsp { (1.0 / 6.0) as f32 } else { 1.0 };
    let dsp_scale_step = if dsp {
        ((3.0 - 1.0 / 6.0) / 10.0) as f32
    } else {
        0.0
    };
    let region = ImageRegion {
        offset: 0,
        width: PATCH_SIDE,
        height: PATCH_SIDE,
        stride: PATCH_SIDE,
    };
    let mut patch = vec![0.0f32; PATCH_SIDE * PATCH_SIDE];
    let mut modulus = vec![0.0f32; PATCH_SIDE * PATCH_SIDE];
    let mut angle = vec![0.0f32; PATCH_SIDE * PATCH_SIDE];
    (0..dsp_num_scales)
        .map(|s| {
            let dsp_scale = f64::from(dsp_min_scale + s as f32 * dsp_scale_step);
            let mut scaled = *frame;
            scaled.a11 = (f64::from(scaled.a11) * dsp_scale) as f32;
            scaled.a12 = (f64::from(scaled.a12) * dsp_scale) as f32;
            scaled.a21 = (f64::from(scaled.a21) * dsp_scale) as f32;
            scaled.a22 = (f64::from(scaled.a22) * dsp_scale) as f32;
            covdet
                .extract_patch_for_frame(
                    &mut patch,
                    PATCH_RESOLUTION,
                    PATCH_RELATIVE_EXTENT,
                    PATCH_RELATIVE_SMOOTHING,
                    &scaled,
                )
                .unwrap();
            // VLFeat writes modulus and angle interleaved into one buffer; each pixel's pair
            // depends only on the patch, so two planes interleaved afterwards are the same.
            vl_imgradient_polar_f(
                &mut modulus,
                &mut angle,
                GradientLayout {
                    first_offset: 0,
                    second_offset: 0,
                    width_stride: 1,
                    height_stride: PATCH_SIDE,
                },
                &patch,
                region,
            );
            let grad: Vec<f32> = modulus
                .iter()
                .zip(&angle)
                .flat_map(|(&m, &a)| [m, a])
                .collect();
            let mut desc = [0.0f32; VL_SIFT_DESCRIPTOR_LENGTH];
            sift.calc_raw_descriptor(
                &grad,
                &mut desc,
                PATCH_SIDE as i32,
                PATCH_SIDE as i32,
                RawDescriptorFrame {
                    x: PATCH_RESOLUTION as f64,
                    y: PATCH_RESOLUTION as f64,
                    sigma: SIGMA,
                    angle0: 0.0,
                },
            );
            desc
        })
        .collect()
}

/// Runs one fixture case; returns the first mismatch (empty when none).
fn run_case(name: &str) -> String {
    let root = Json::parse(COVDET_FIXTURE);
    let case = root
        .get("cases")
        .as_array()
        .iter()
        .find(|c| c.get("name").as_str() == name)
        .unwrap_or_else(|| panic!("no case {name}"));
    let flag = |key: &str| matches!(case.get(key), Json::Bool(true));
    let (image, width, height) = load_image(case.get("image").as_str());
    let dsp = flag("dsp");
    let dsp_num_scales = if dsp { 10 } else { 1 };

    let cancel = CancelToken::new();
    let mut covdet = detector(case.get("first_octave").as_f64() as i32);
    covdet.put_image(&image, width, height, &cancel).unwrap();
    covdet.detect(8192, &cancel).unwrap();
    if flag("affine") {
        covdet.extract_affine_shape(&cancel).unwrap();
    }
    if !flag("upright") {
        covdet.extract_orientations(&cancel).unwrap();
    }

    let expected = case.get("features").as_array();
    if covdet.num_features() != expected.len() {
        return format!(
            "{} features, expected {}",
            covdet.num_features(),
            expected.len()
        );
    }

    let mut sift = VlSiftFilter::new(16, 16, 1, 3, 0);
    sift.magnif = 3.0;
    let features: Vec<VlCovDetFeature> = covdet.features().to_vec();
    for (i, (f, e)) in features.iter().zip(expected).enumerate() {
        let os = e.get("os").as_f64s();
        if f64::from(f.o) != os[0] || f64::from(f.s) != os[1] {
            return format!(
                "feature {i}: (o, s) = ({}, {}), expected ({}, {})",
                f.o, f.s, os[0], os[1]
            );
        }
        let values = e.get("values").as_f64s();
        let got = [
            f.frame.x,
            f.frame.y,
            f.frame.a11,
            f.frame.a12,
            f.frame.a21,
            f.frame.a22,
            f.peak_score,
            f.edge_score,
            f.orientation_score,
        ];
        // The frame matrix may differ from VLFeat's by up to one float ulp of the frame's
        // scale (see the `math::fns` header): affine adaptation's make-upright rotation
        // takes atan2 / cos / sin, where the libm crate and Apple libm can differ by an ulp,
        // and a matrix entry that cancels to ~0 (Square256AffineUpright feature 8: a12 is 0.0
        // here, -2^-50 in VLFeat) shows that ulp. Everything else, including every descriptor
        // bin of the patch warped by that frame, is compared bit for bit.
        let scale = got[2..6].iter().fold(0.0f32, |m, a| m.max(a.abs()));
        for (k, (&g, &want)) in got.iter().zip(&values).enumerate() {
            let want = want as f32;
            let close = (2..6).contains(&k) && (g - want).abs() <= f32::EPSILON * scale;
            // Bit comparison, so a NaN or signed-zero difference would show too.
            if g.to_bits() != want.to_bits() && !close {
                return format!("feature {i} value {k}: {g:?}, expected {want:?}");
            }
        }

        let descriptors = raw_descriptors(&mut covdet, &sift, &f.frame, dsp_num_scales, dsp);
        let wanted = e.get("d").as_array();
        assert_eq!(wanted.len(), dsp_num_scales);
        for (s, (desc, want)) in descriptors.iter().zip(wanted).enumerate() {
            let bytes = decode_base64(want.as_str());
            for (d, got) in desc.iter().enumerate() {
                let b = &bytes[4 * d..4 * d + 4];
                let want = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                if got.to_bits() != want.to_bits() {
                    return format!("feature {i} scale {s} bin {d}: {got:?}, expected {want:?}");
                }
            }
        }
    }
    String::new()
}

#[test]
fn rust_only_vl_covdet_matches_unfused_vlfeat_square256() {
    assert_eq!(run_case("Square256"), "");
}

#[test]
fn rust_only_vl_covdet_matches_unfused_vlfeat_square256_affine() {
    assert_eq!(run_case("Square256Affine"), "");
}

#[test]
fn rust_only_vl_covdet_matches_unfused_vlfeat_square256_affine_upright() {
    assert_eq!(run_case("Square256AffineUpright"), "");
}

#[test]
fn rust_only_vl_covdet_matches_unfused_vlfeat_square256_dsp() {
    assert_eq!(run_case("Square256Dsp"), "");
}

#[test]
fn rust_only_vl_covdet_matches_unfused_vlfeat_texture() {
    assert_eq!(run_case("Texture"), "");
}

#[test]
fn rust_only_vl_covdet_matches_unfused_vlfeat_texture_affine() {
    assert_eq!(run_case("TextureAffine"), "");
}

#[test]
fn rust_only_vl_covdet_matches_unfused_vlfeat_texture_upright() {
    assert_eq!(run_case("TextureUpright"), "");
}

#[test]
fn rust_only_vl_covdet_matches_unfused_vlfeat_texture_first_octave0() {
    assert_eq!(run_case("TextureFirstOctave0"), "");
}

#[test]
fn rust_only_vl_covdet_rejects_bad_input() {
    let cancel = CancelToken::new();
    let mut covdet = VlCovDet::new();
    // VLFeat asserts these; here they are errors.
    assert!(covdet.detect(10, &cancel).is_err());
    assert!(covdet.put_image(&[], 0, 4, &cancel).is_err());
    assert!(covdet.put_image(&[0.0; 15], 4, 4, &cancel).is_err());
    // A 1-pixel side leaves no octave at all.
    assert!(covdet.put_image(&[0.0; 32], 1, 32, &cancel).is_err());
    // Below a 16-pixel side octave 0 does not fit, and VLFeat would upsample octave -1 from
    // it anyway (COLMAP's extractor never asks); from 16 on it works.
    assert!(covdet.put_image(&[0.0; 150], 10, 15, &cancel).is_err());
    covdet.first_octave = 0;
    assert!(covdet.put_image(&[0.0; 150], 10, 15, &cancel).is_err());
    covdet.first_octave = -1;
    covdet.put_image(&[0.0; 256], 16, 16, &cancel).unwrap();
    covdet.detect(10, &cancel).unwrap();
    assert_eq!(covdet.num_features(), 0);
}

#[test]
fn rust_only_vl_covdet_cancelled_token_stops() {
    let (image, width, height) = load_image("texture");
    let cancel = CancelToken::new();
    let mut covdet = detector(-1);
    covdet.put_image(&image, width, height, &cancel).unwrap();
    cancel.cancel();
    let err = covdet.detect(8192, &cancel).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Cancelled);
    assert!(covdet.put_image(&image, width, height, &cancel).is_err());
}

#[test]
fn rust_only_vl_covdet_non_extrema_suppression_drops_weaker_neighbours() {
    // With suppression off every refined extremum is kept; with VLFeat's default 0.5 the
    // weaker of two features at about the same place and scale goes, and the count says so.
    let (image, width, height) = load_image("texture");
    let cancel = CancelToken::new();
    let mut all = detector(-1);
    all.non_extrema_suppression = 0.0;
    all.put_image(&image, width, height, &cancel).unwrap();
    all.detect(8192, &cancel).unwrap();
    let mut kept = detector(-1);
    kept.put_image(&image, width, height, &cancel).unwrap();
    kept.detect(8192, &cancel).unwrap();
    assert_eq!(all.num_non_extrema_suppressed(), 0);
    assert!(kept.num_non_extrema_suppressed() > 0);
    assert_eq!(
        kept.num_features() + kept.num_non_extrema_suppressed(),
        all.num_features()
    );
    // The survivors keep their detection order.
    let mut j = 0;
    for f in kept.features() {
        while all.features()[j] != *f {
            j += 1;
        }
    }
}
