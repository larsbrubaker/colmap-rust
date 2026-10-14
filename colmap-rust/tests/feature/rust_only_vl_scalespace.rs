// Rust-only tests of colmap_rust::feature::vlfeat::imopv_smooth (VLFeat's vl_imsmooth_f and
// image gradients) and colmap_rust::feature::vlfeat::scalespace (VlScaleSpace), which have no
// COLMAP test of their own: the covariant detector's oracle fixture pins them end to end. These
// pin VLFeat's behaviour: the Gaussian filter's shape, the smoothing's impulse response
// (bit-exact: one nonzero tap per pass), padding by continuity, the gradients' one-sided and
// central differences, and the scale space's default geometry, level sizes, sigmas and
// cancellation.

use colmap_rust::feature::vlfeat::imopv_smooth::{
    vl_imgradient_f, vl_imgradient_polar_f, vl_imsmooth_f, vl_imsmooth_f_within,
    vl_new_gaussian_filter_f, GradientLayout, ImageRegion,
};
use colmap_rust::feature::vlfeat::scalespace::{VlScaleSpace, VlScaleSpaceGeometry};
use colmap_rust::math::fns;
use colmap_rust::util::cancellation::CancelToken;
use colmap_rust::ErrorKind;

fn region(width: usize, height: usize) -> ImageRegion {
    ImageRegion {
        offset: 0,
        width,
        height,
        stride: width,
    }
}

#[test]
fn rust_only_gaussian_filter_shape() {
    let filter = vl_new_gaussian_filter_f(1.0);
    // Half-width ceil(3 * sigma).
    assert_eq!(filter.len(), 7);
    for i in 0..3 {
        assert_eq!(filter[i], filter[6 - i]);
        assert!(filter[i] < filter[i + 1]);
    }
    let sum: f32 = filter.iter().sum();
    assert!((sum - 1.0).abs() < 1e-6);
    // A sigma just above an integer multiple of 1/3 rounds the support up.
    assert_eq!(vl_new_gaussian_filter_f(1.01).len(), 2 * 4 + 1);
}

#[test]
fn rust_only_imsmooth_impulse_response_is_the_filter_product() {
    // An impulse far from the border: each pass adds exactly one nonzero product, so the
    // result is fy[dy] * fx[dx] bit for bit, with the x filter along rows.
    let (w, h) = (21, 17);
    let (cx, cy) = (10, 8);
    let mut image = vec![0.0f32; w * h];
    image[cy * w + cx] = 1.0;
    let (sigmax, sigmay) = (1.5, 0.8);
    let mut smoothed = vec![-1.0f32; w * h];
    vl_imsmooth_f(&mut smoothed, 0, w, &image, region(w, h), sigmax, sigmay);

    let fx = vl_new_gaussian_filter_f(sigmax);
    let fy = vl_new_gaussian_filter_f(sigmay);
    let (hx, hy) = (fx.len() / 2, fy.len() / 2);
    for y in 0..h {
        for x in 0..w {
            let dx = x as isize - cx as isize + hx as isize;
            let dy = y as isize - cy as isize + hy as isize;
            let expected =
                if (0..fx.len() as isize).contains(&dx) && (0..fy.len() as isize).contains(&dy) {
                    fy[dy as usize] * fx[dx as usize]
                } else {
                    0.0
                };
            assert_eq!(
                smoothed[y * w + x].to_bits(),
                expected.to_bits(),
                "pixel ({x}, {y})"
            );
        }
    }
}

#[test]
fn rust_only_imsmooth_within_matches_separate_buffers() {
    let (w, h) = (9, 7);
    let image: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 11) as f32 / 10.0).collect();
    let mut expected = vec![0.0f32; w * h];
    vl_imsmooth_f(&mut expected, 0, w, &image, region(w, h), 1.2, 1.2);

    // In place.
    let mut data = image.clone();
    vl_imsmooth_f_within(&mut data, 0, w, region(w, h), 1.2, 1.2);
    assert_eq!(data, expected);

    // From one level of a buffer into the next.
    let mut levels = image.clone();
    levels.extend(vec![0.0f32; w * h]);
    vl_imsmooth_f_within(&mut levels, w * h, w, region(w, h), 1.2, 1.2);
    assert_eq!(&levels[w * h..], &expected[..]);
    assert_eq!(&levels[..w * h], &image[..]);
}

#[test]
fn rust_only_imsmooth_pads_by_continuity() {
    // A constant image stays (nearly) constant up to the border: padding repeats the edge.
    let (w, h) = (6, 5);
    let image = vec![0.75f32; w * h];
    let mut smoothed = vec![0.0f32; w * h];
    vl_imsmooth_f(&mut smoothed, 0, w, &image, region(w, h), 2.0, 2.0);
    for v in smoothed {
        assert!((v - 0.75).abs() < 1e-6, "{v}");
    }
}

#[test]
fn rust_only_imgradient_differences() {
    // f(x, y) = 2x + 3y + x^2: the central difference of x^2 is exactly 2x inside, the
    // one-sided ones are 2x + 1 (left) and 2x - 1 (right).
    let (w, h) = (5, 4);
    let image: Vec<f32> = (0..h)
        .flat_map(|y| (0..w).map(move |x| (2 * x + 3 * y + x * x) as f32))
        .collect();
    let mut gx = vec![0.0f32; w * h];
    let mut gy = vec![0.0f32; w * h];
    let layout = GradientLayout {
        first_offset: 0,
        second_offset: 0,
        width_stride: 1,
        height_stride: w,
    };
    vl_imgradient_f(&mut gx, &mut gy, layout, &image, region(w, h));
    for y in 0..h {
        for x in 0..w {
            let expected_gx = match x {
                0 => 3.0,
                4 => 9.0,
                _ => (2 + 2 * x) as f32,
            };
            assert_eq!(gx[y * w + x], expected_gx, "gx ({x}, {y})");
            assert_eq!(gy[y * w + x], 3.0, "gy ({x}, {y})");
        }
    }

    // The polar form: |(gx, gy)| and atan2 within the fast approximations' error (VLFeat
    // documents 0.0061 rad for vl_fast_atan2_f).
    let mut modulus = vec![0.0f32; w * h];
    let mut angle = vec![0.0f32; w * h];
    vl_imgradient_polar_f(&mut modulus, &mut angle, layout, &image, region(w, h));
    for i in 0..w * h {
        let expected_modulus = fns::sqrtf(gx[i] * gx[i] + gy[i] * gy[i]);
        assert!((modulus[i] - expected_modulus).abs() < 1e-2 * expected_modulus);
        let expected_angle = fns::atan2f(gy[i], gx[i]);
        assert!(
            (angle[i] - expected_angle).abs() <= 0.0061,
            "{} vs {}",
            angle[i],
            expected_angle
        );
    }
}

#[test]
fn rust_only_scalespace_default_geometry() {
    let geom = VlScaleSpaceGeometry::default_geometry(640, 480).unwrap();
    // floor(log2(480)) - 3 = 5.
    assert_eq!(geom.first_octave, 0);
    assert_eq!(geom.last_octave, 5);
    assert_eq!(geom.octave_resolution, 3);
    assert_eq!(geom.octave_first_subdivision, 0);
    assert_eq!(geom.octave_last_subdivision, 2);
    assert!((geom.base_scale - 1.6 * fns::pow(2.0, 1.0 / 3.0)).abs() < 1e-15);
    assert_eq!(geom.nominal_scale, 0.5);

    // Small images still get one octave.
    assert_eq!(
        VlScaleSpaceGeometry::default_geometry(7, 100)
            .unwrap()
            .last_octave,
        0
    );
    assert!(VlScaleSpaceGeometry::default_geometry(0, 10).is_err());
}

#[test]
fn rust_only_scalespace_octave_geometry_and_sigmas() {
    let mut geom = VlScaleSpaceGeometry::default_geometry(33, 20).unwrap();
    geom.first_octave = -1;
    geom.last_octave = 1;
    let space = VlScaleSpace::new(geom).unwrap();
    let o = space.octave_geometry(-1);
    assert_eq!((o.width, o.height, o.step), (66, 40, 0.5));
    let o = space.octave_geometry(1);
    assert_eq!((o.width, o.height, o.step), (16, 10, 2.0));
    assert_eq!(space.level(-1, 2).len(), 66 * 40);
    assert_eq!(space.level(1, 0).len(), 16 * 10);
    // sigma(o, s) = base * 2^(o + s / resolution).
    assert_eq!(space.level_sigma(0, 0), geom.base_scale);
    assert!((space.level_sigma(1, 3) - 4.0 * geom.base_scale).abs() < 1e-14);

    // Invalid geometries are rejected.
    let mut bad = geom;
    bad.first_octave = 2;
    assert!(VlScaleSpace::new(bad).is_err());
    let mut bad = geom;
    bad.octave_resolution = 0;
    assert!(VlScaleSpace::new(bad).is_err());
}

#[test]
fn rust_only_scalespace_put_image_keeps_a_constant_image() {
    for first_octave in [-1, 0] {
        let (w, h) = (40, 32);
        let mut geom = VlScaleSpaceGeometry::default_geometry(w, h).unwrap();
        geom.first_octave = first_octave;
        geom.octave_first_subdivision = -1;
        geom.octave_last_subdivision = 3;
        let mut space = VlScaleSpace::new(geom).unwrap();
        space
            .put_image(&vec![0.5f32; w * h], &CancelToken::new())
            .unwrap();
        for o in geom.first_octave..=geom.last_octave {
            for s in geom.octave_first_subdivision..=geom.octave_last_subdivision {
                for &v in space.level(o, s) {
                    assert!((v - 0.5).abs() < 1e-5, "level ({o}, {s}): {v}");
                }
            }
        }
    }
}

#[test]
fn rust_only_scalespace_levels_smooth_progressively() {
    // An impulse spreads as the level sigma grows: the peak drops level by level.
    let (w, h) = (64, 64);
    let mut image = vec![0.0f32; w * h];
    image[32 * w + 32] = 1.0;
    let mut space = VlScaleSpace::with_default_geometry(w, h).unwrap();
    space.put_image(&image, &CancelToken::new()).unwrap();
    let peak =
        |s: &VlScaleSpace, o: i32, l: i32| s.level(o, l).iter().cloned().fold(0.0f32, f32::max);
    assert!(peak(&space, 0, 0) > peak(&space, 0, 1));
    assert!(peak(&space, 0, 1) > peak(&space, 0, 2));
}

#[test]
fn rust_only_scalespace_put_image_cancelled() {
    let mut space = VlScaleSpace::with_default_geometry(16, 16).unwrap();
    let cancel = CancelToken::new();
    cancel.cancel();
    let error = space.put_image(&[0.0f32; 256], &cancel).unwrap_err();
    assert_eq!(error.kind(), ErrorKind::Cancelled);
}
