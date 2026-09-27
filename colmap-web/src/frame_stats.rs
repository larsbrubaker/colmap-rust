//! Pixel statistics of a rendered frame, for the browser smoke test (`web/tests/smoke.spec.ts`).
//!
//! Platform-neutral (compiles and is unit tested natively). The wasm side of `lib.rs` reads the
//! last rendered frame back from the GPU (`?test=1` only, see there) and hands the RGBA8 pixels to
//! [`frame_stats`]; the smoke test asserts on the result. Checking what the GPU rendered, rather
//! than a compositor screenshot, keeps the test independent of how headless Chrome composites the
//! page (on a GPU-less Linux runner its screenshots can come back blank while WebGPU renders fine).

/// Channel margin for a pixel to count as red/green/blue: that channel beats both others by more
/// than this, so the dark theme's greys, the text and anti-aliased grey edges never qualify.
pub const AXIS_MARGIN: i32 = 60;

/// Pixel classes counted in a frame.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameStats {
    pub width: u32,
    pub height: u32,
    /// Distinct colours after quantising each channel to 5 bits.
    pub distinct: u32,
    /// Share of pixels in the most common quantised colour (0 for an empty frame).
    pub dominant_fraction: f64,
    /// Pixels whose red channel beats green and blue by more than [`AXIS_MARGIN`].
    pub red: u32,
    pub green: u32,
    pub blue: u32,
}

/// Classify a tightly packed, top-down RGBA8 frame (`width * height * 4` bytes; alpha ignored).
/// Returns `None` when the buffer size doesn't match the dimensions.
pub fn frame_stats(rgba: &[u8], width: u32, height: u32) -> Option<FrameStats> {
    let pixels = (width as usize).checked_mul(height as usize)?;
    if rgba.len() != pixels.checked_mul(4)? {
        return None;
    }
    // 15-bit quantised colour key -> count; a flat array beats a hash map for 32768 keys.
    let mut buckets = vec![0u32; 1 << 15];
    let (mut red, mut green, mut blue) = (0u32, 0u32, 0u32);
    for px in rgba.chunks_exact(4) {
        let (r, g, b) = (px[0] as i32, px[1] as i32, px[2] as i32);
        let key = ((r >> 3) << 10) | ((g >> 3) << 5) | (b >> 3);
        buckets[key as usize] += 1;
        if r > g + AXIS_MARGIN && r > b + AXIS_MARGIN {
            red += 1;
        } else if g > r + AXIS_MARGIN && g > b + AXIS_MARGIN {
            green += 1;
        } else if b > r + AXIS_MARGIN && b > g + AXIS_MARGIN {
            blue += 1;
        }
    }
    let distinct = buckets.iter().filter(|&&c| c > 0).count() as u32;
    let dominant = buckets.iter().copied().max().unwrap_or(0);
    let dominant_fraction = if pixels == 0 {
        0.0
    } else {
        dominant as f64 / pixels as f64
    };
    Some(FrameStats {
        width,
        height,
        distinct,
        dominant_fraction,
        red,
        green,
        blue,
    })
}

/// Whether a page URL query string (`location.search`, with or without the leading `?`) asks for
/// the test hooks: a `test=1` parameter.
pub fn test_hooks_requested(search: &str) -> bool {
    search
        .trim_start_matches('?')
        .split('&')
        .any(|pair| pair == "test=1")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: u32, h: u32, fill: [u8; 4]) -> Vec<u8> {
        fill.iter()
            .copied()
            .cycle()
            .take((w * h * 4) as usize)
            .collect()
    }

    fn set(buf: &mut [u8], w: u32, x: u32, y: u32, c: [u8; 4]) {
        let i = ((y * w + x) * 4) as usize;
        buf[i..i + 4].copy_from_slice(&c);
    }

    #[test]
    fn flat_frame_is_one_dominant_colour_with_no_axis_pixels() {
        let s = frame_stats(&frame(4, 3, [30, 30, 34, 255]), 4, 3).unwrap();
        assert_eq!((s.width, s.height), (4, 3));
        assert_eq!(s.distinct, 1);
        assert_eq!(s.dominant_fraction, 1.0);
        assert_eq!((s.red, s.green, s.blue), (0, 0, 0));
    }

    #[test]
    fn counts_axis_colours_and_distinct_buckets() {
        let (w, h) = (10, 10);
        let mut buf = frame(w, h, [30, 30, 34, 255]);
        set(&mut buf, w, 0, 0, [230, 60, 60, 255]); // red
        set(&mut buf, w, 1, 0, [231, 61, 61, 255]); // red, same 5-bit bucket
        set(&mut buf, w, 2, 0, [60, 200, 60, 255]); // green
        set(&mut buf, w, 3, 0, [60, 90, 230, 255]); // blue
        set(&mut buf, w, 4, 0, [200, 200, 200, 255]); // light grey: no axis
        set(&mut buf, w, 5, 0, [150, 91, 91, 255]); // red by only 59: no axis
        let s = frame_stats(&buf, w, h).unwrap();
        assert_eq!((s.red, s.green, s.blue), (2, 1, 1));
        assert_eq!(s.distinct, 6);
        assert!((s.dominant_fraction - 0.94).abs() < 1e-12);
    }

    #[test]
    fn alpha_is_ignored() {
        let a = frame_stats(&frame(2, 2, [1, 2, 3, 0]), 2, 2).unwrap();
        let b = frame_stats(&frame(2, 2, [1, 2, 3, 255]), 2, 2).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn size_mismatch_is_rejected() {
        assert_eq!(frame_stats(&[0; 15], 2, 2), None);
        assert_eq!(frame_stats(&[0; 20], 2, 2), None);
    }

    #[test]
    fn test_flag_parsing() {
        assert!(test_hooks_requested("?test=1"));
        assert!(test_hooks_requested("?a=b&test=1"));
        assert!(test_hooks_requested("test=1"));
        assert!(!test_hooks_requested(""));
        assert!(!test_hooks_requested("?test=0"));
        assert!(!test_hooks_requested("?test=10"));
        assert!(!test_hooks_requested("?mytest=1"));
    }

    #[test]
    fn empty_frame_has_no_dominant_share() {
        let s = frame_stats(&[], 0, 0).unwrap();
        assert_eq!((s.distinct, s.dominant_fraction), (0, 0.0));
    }
}
