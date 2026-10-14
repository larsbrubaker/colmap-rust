// Rust-only tests of VlSiftFilter::calc_raw_descriptor (VLFeat's vl_sift_calc_raw_descriptor,
// the covariant extractor's descriptor), which has no COLMAP test of its own: the covariant
// oracle fixture pins it end to end once the detector lands. These pin VLFeat's behaviour: a
// frame outside the gradient image leaves the descriptor untouched, a gradient field of one
// orientation fills only that orientation's bins, the result is L2-normalized and truncated
// at 0.2, and the norm threshold scales with the number of samples.

use colmap_rust::feature::vlfeat::sift_descriptor::VL_SIFT_DESCRIPTOR_LENGTH;
use colmap_rust::feature::vlfeat::sift_filter::VlSiftFilter;
use colmap_rust::feature::vlfeat::sift_raw_descriptor::RawDescriptorFrame;

/// A `w` x `h` interleaved (modulus, angle) gradient image of one value.
fn uniform_gradient(w: usize, h: usize, modulus: f32, angle: f32) -> Vec<f32> {
    (0..w * h).flat_map(|_| [modulus, angle]).collect()
}

fn frame(x: f64, y: f64) -> RawDescriptorFrame {
    RawDescriptorFrame {
        x,
        y,
        sigma: 2.0,
        angle0: 0.0,
    }
}

#[test]
fn rust_only_raw_descriptor_outside_the_image_is_untouched() {
    let filter = VlSiftFilter::new(16, 16, 1, 3, 0);
    let grad = uniform_gradient(16, 16, 1.0, 0.0);
    let mut descr = [7.0f32; VL_SIFT_DESCRIPTOR_LENGTH];
    for (x, y) in [(-1.6, 5.0), (15.6, 5.0), (5.0, -1.6), (5.0, 14.6)] {
        filter.calc_raw_descriptor(&grad, &mut descr, 16, 16, frame(x, y));
        assert!(descr.iter().all(|&d| d == 7.0), "frame ({x}, {y})");
    }
    // `(int)(x + 0.5)` truncates toward zero, so x = -0.6 is pixel 0, inside. The last row is
    // excluded (yi >= h - 1), the last column is not.
    filter.calc_raw_descriptor(&grad, &mut descr, 16, 16, frame(-0.6, 5.0));
    assert!(descr.iter().any(|&d| d != 7.0));
    descr = [7.0f32; VL_SIFT_DESCRIPTOR_LENGTH];
    filter.calc_raw_descriptor(&grad, &mut descr, 16, 16, frame(15.0, 5.0));
    assert!(descr.iter().any(|&d| d != 7.0));
}

#[test]
fn rust_only_raw_descriptor_of_one_orientation() {
    let (w, h) = (48, 48);
    let filter = VlSiftFilter::new(w as i32, h as i32, 1, 3, 0);
    let grad = uniform_gradient(w, h, 1.0, 0.0);
    let mut descr = [0.0f32; VL_SIFT_DESCRIPTOR_LENGTH];
    filter.calc_raw_descriptor(&grad, &mut descr, w as i32, h as i32, frame(24.0, 24.0));

    // Bin t of spatial cell (bx, by) is at t + 8 * bx + 32 * by: angle 0 lands in t = 0 only.
    for (i, &d) in descr.iter().enumerate() {
        if i % 8 == 0 {
            assert!(d > 0.0, "bin {i}");
        } else {
            assert_eq!(d, 0.0, "bin {i}");
        }
    }
    // The Gaussian window weights the central cells above the corner cells.
    let cell = |bx: usize, by: usize| descr[8 * bx + 32 * by];
    assert!(cell(1, 1) > cell(0, 0));
    assert!(cell(2, 2) > cell(3, 3));
    let norm: f32 = descr.iter().map(|d| d * d).sum();
    assert!((norm - 1.0).abs() < 1e-3, "{norm}");
}

#[test]
fn rust_only_raw_descriptor_norm_threshold_scales_with_samples() {
    let (w, h) = (48, 48);
    let grad = uniform_gradient(w, h, 1e-3, 1.0);
    let mut filter = VlSiftFilter::new(w as i32, h as i32, 1, 3, 0);
    let mut descr = [0.0f32; VL_SIFT_DESCRIPTOR_LENGTH];

    // The norm before normalization is far above the threshold itself, but below it times
    // the number of samples in the window.
    filter.norm_thresh = 1e-3;
    filter.calc_raw_descriptor(&grad, &mut descr, w as i32, h as i32, frame(24.0, 24.0));
    assert!(descr.iter().all(|&d| d == 0.0));

    filter.norm_thresh = 0.0;
    filter.calc_raw_descriptor(&grad, &mut descr, w as i32, h as i32, frame(24.0, 24.0));
    assert!(descr.iter().any(|&d| d > 0.0));
}
