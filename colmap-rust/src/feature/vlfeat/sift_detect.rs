//! Port of `vl_sift_detect` from VLFeat's `sift.c`: the difference of Gaussians, the 3x3x3
//! extremum search and the quadratic refinement with the peak and edge tests. Part of
//! [`VlSiftFilter`] (scale space in [`super::sift_filter`], descriptors in
//! [`super::sift_descriptor`]). Port of colmap-sharp's `Feature/VLFeat/VlSiftFilter.Detect.cs`.
//!
//! Tier A (exact). The DoG is float; the refinement is double, with the float samples promoted
//! where the C promotes them (float sums and differences first, then the double scaling), and
//! the 3x3 solve is VLFeat's own Gauss elimination with partial pivoting.

use super::sift_filter::{VlSiftFilter, VlSiftKeypoint};
use crate::math::fns;
use crate::util::cancellation::CancelToken;
use crate::util::check::Result;

impl VlSiftFilter {
    /// Port of `vl_sift_detect`: finds the DoG extrema of the current octave and refines
    /// them. Results are in [`VlSiftFilter::keypoints`].
    pub fn detect(&mut self, cancel: &CancelToken) -> Result<()> {
        let w = self.octave_width as isize;
        let h = self.octave_height as isize;
        let te = self.edge_thresh;
        let tp = self.peak_thresh;

        let xo: isize = 1;
        let yo: isize = w;
        let so: isize = w * h;

        let xper = fns::pow(2.0, f64::from(self.o_cur));

        self.keys.clear();

        // Compute difference of Gaussians (DoG).
        let level = (w * h) as usize;
        let mut p = 0usize;
        for s in self.s_min..=self.s_max - 1 {
            let src_a = self.octave_offset(s);
            let src_b = self.octave_offset(s + 1);
            for i in 0..level {
                self.dog[p] = self.octave[src_b + i] - self.octave[src_a + i];
                p += 1;
            }
        }

        cancel.check_cancelled()?;

        // Find local maxima of DoG, starting from dog[1, 1, s_min + 1].
        let dg = &self.dog;
        let mut found: Vec<(i32, i32, i32)> = Vec::new();
        let mut p = xo + yo + so;
        let peak_floor = 0.8 * tp;
        for s in self.s_min + 1..=self.s_max - 2 {
            cancel.check_cancelled()?;
            for y in 1..h - 1 {
                for x in 1..w - 1 {
                    let v = dg[p as usize];
                    if is_extremum(dg, p, v, peak_floor, yo, so) {
                        found.push((x as i32, y as i32, s));
                    }
                    p += 1;
                }
                p += 2;
            }
            p += 2 * yo;
        }

        // Refine local maxima.
        let at = |pt: isize, ddx: isize, ddy: isize, dds: isize| -> f32 {
            dg[(pt + ddx * xo + ddy * yo + dds * so) as usize]
        };
        let mut keys = Vec::with_capacity(found.len());
        for &(x0, y0, s) in &found {
            let mut x = x0 as isize;
            let mut y = y0 as isize;
            let (mut dxv, mut dyv, mut dsv) = (0.0f64, 0.0f64, 0.0f64);
            let (mut dxx, mut dyy, mut dxy) = (0.0f64, 0.0f64, 0.0f64);
            let mut b = [0.0f64; 3];
            let (mut dx, mut dy) = (0isize, 0isize);
            let mut pt = 0isize;

            for _ in 0..5 {
                x += dx;
                y += dy;

                pt = xo * x + yo * y + so * (s - self.s_min) as isize;

                // Compute the gradient. The float differences are formed in float and then
                // promoted, like the C.
                dxv = 0.5 * f64::from(at(pt, 1, 0, 0) - at(pt, -1, 0, 0));
                dyv = 0.5 * f64::from(at(pt, 0, 1, 0) - at(pt, 0, -1, 0));
                dsv = 0.5 * f64::from(at(pt, 0, 0, 1) - at(pt, 0, 0, -1));

                // Compute the Hessian.
                let c = f64::from(at(pt, 0, 0, 0));
                dxx = f64::from(at(pt, 1, 0, 0) + at(pt, -1, 0, 0)) - 2.0 * c;
                dyy = f64::from(at(pt, 0, 1, 0) + at(pt, 0, -1, 0)) - 2.0 * c;
                let dss = f64::from(at(pt, 0, 0, 1) + at(pt, 0, 0, -1)) - 2.0 * c;

                dxy = 0.25
                    * f64::from(
                        at(pt, 1, 1, 0) + at(pt, -1, -1, 0) - at(pt, -1, 1, 0) - at(pt, 1, -1, 0),
                    );
                let dxs = 0.25
                    * f64::from(
                        at(pt, 1, 0, 1) + at(pt, -1, 0, -1) - at(pt, -1, 0, 1) - at(pt, 1, 0, -1),
                    );
                let dys = 0.25
                    * f64::from(
                        at(pt, 0, 1, 1) + at(pt, 0, -1, -1) - at(pt, 0, -1, 1) - at(pt, 0, 1, -1),
                    );

                // Solve the linear system; A is column-major like VLFeat's Aat(i, j).
                let mut a = [0.0f64; 9];
                a[0] = dxx;
                a[4] = dyy;
                a[8] = dss;
                a[1] = dxy;
                a[3] = dxy;
                a[2] = dxs;
                a[6] = dxs;
                a[5] = dys;
                a[7] = dys;

                b = [-dxv, -dyv, -dsv];
                solve_gauss_elimination(&mut a, &mut b);

                // If the translation of the keypoint is big, move the keypoint and re-iterate
                // the computation. Otherwise we are all set.
                dx = isize::from(b[0] > 0.6 && x < w - 2) - isize::from(b[0] < -0.6 && x > 1);
                dy = isize::from(b[1] > 0.6 && y < h - 2) - isize::from(b[1] < -0.6 && y > 1);

                if dx == 0 && dy == 0 {
                    break;
                }
            }

            // Check threshold and other conditions.
            let val = f64::from(at(pt, 0, 0, 0)) + 0.5 * (dxv * b[0] + dyv * b[1] + dsv * b[2]);
            let score = (dxx + dyy) * (dxx + dyy) / (dxx * dyy - dxy * dxy);
            let xn = x as f64 + b[0];
            let yn = y as f64 + b[1];
            let sn = f64::from(s) + b[2];

            let good = val.abs() > tp
                && score < (te + 1.0) * (te + 1.0) / te
                && score >= 0.0
                && b[0].abs() < 1.5
                && b[1].abs() < 1.5
                && b[2].abs() < 1.5
                && xn >= 0.0
                && xn <= (w - 1) as f64
                && yn >= 0.0
                && yn <= (h - 1) as f64
                && sn >= f64::from(self.s_min)
                && sn <= f64::from(self.s_max);

            if good {
                keys.push(VlSiftKeypoint {
                    o: self.o_cur,
                    ix: x as i32,
                    iy: y as i32,
                    is: s,
                    s: sn as f32,
                    x: (xn * xper) as f32,
                    y: (yn * xper) as f32,
                    sigma: (self.sigma0 * fns::pow(2.0, sn / f64::from(self.s_num)) * xper) as f32,
                });
            }
        }

        self.keys = keys;
        Ok(())
    }
}

/// `CHECK_NEIGHBORS(>, +) || CHECK_NEIGHBORS(<, -)`: `v` beats the peak floor and all 26
/// neighbors in space and scale. The float sample is promoted for the threshold test only.
fn is_extremum(d: &[f32], p: isize, v: f32, peak_floor: f64, yo: isize, so: isize) -> bool {
    const XO: isize = 1;
    // The 26 neighbor offsets in VLFeat's order (the comparison short-circuits, but every
    // neighbor is in bounds, so the order cannot change the result).
    let offsets = [
        XO,
        -XO,
        so,
        -so,
        yo,
        -yo,
        yo + XO,
        yo - XO,
        -yo + XO,
        -yo - XO,
        XO + so,
        -XO + so,
        yo + so,
        -yo + so,
        yo + XO + so,
        yo - XO + so,
        -yo + XO + so,
        -yo - XO + so,
        XO - so,
        -XO - so,
        yo - so,
        -yo - so,
        yo + XO - so,
        yo - XO - so,
        -yo + XO - so,
        -yo - XO - so,
    ];
    let value = f64::from(v);
    (value >= peak_floor && offsets.iter().all(|&o| v > d[(p + o) as usize]))
        || (value <= -peak_floor && offsets.iter().all(|&o| v < d[(p + o) as usize]))
}

/// VLFeat's 3x3 Gauss elimination with partial pivoting and back substitution, in place. `a`
/// is column-major (`Aat(i, j) = a[i + 3 * j]`). A singular system gives `b = 0`.
fn solve_gauss_elimination(a: &mut [f64; 9], b: &mut [f64; 3]) {
    for j in 0..3 {
        let mut maxa = 0.0f64;
        let mut maxabsa = 0.0f64;
        let mut maxi = 0usize;

        // Look for the maximally stable pivot.
        for i in j..3 {
            let v = a[i + 3 * j];
            let absa = v.abs();
            if absa > maxabsa {
                maxa = v;
                maxabsa = absa;
                maxi = i;
            }
        }

        // If singular give up (the threshold is the float literal 1e-10f).
        if maxabsa < f64::from(1e-10f32) {
            *b = [0.0; 3];
            break;
        }

        let r = maxi;

        // Swap j-th row with r-th row and normalize j-th row.
        for jj in j..3 {
            a.swap(r + 3 * jj, j + 3 * jj);
            a[j + 3 * jj] /= maxa;
        }
        b.swap(j, r);
        b[j] /= maxa;

        // Elimination.
        for ii in j + 1..3 {
            let x = a[ii + 3 * j];
            for jj in j..3 {
                a[ii + 3 * jj] -= x * a[j + 3 * jj];
            }
            b[ii] -= x * b[j];
        }
    }

    // Backward substitution.
    for i in (1..3).rev() {
        let x = b[i];
        for ii in (0..i).rev() {
            b[ii] -= x * a[ii + 3 * i];
        }
    }
}
