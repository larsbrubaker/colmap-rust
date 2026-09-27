//! Port of COLMAP's `colmap/geometry/bbox.h` and `bbox.cc`: splitting an axis-aligned box
//! into equal parts. Port of colmap-sharp's `Geometry/Bbox.cs`. Tests:
//! `tests/geometry/bbox.rs` (bbox_test.cc). Tier A: the same scalar arithmetic as COLMAP.

use crate::linalg::{AlignedBox3d, Vector3d};
use crate::{check_gt, Result};

/// Port of `colmap::ComputeEqualPartsBboxes`: divides `bbox` into `split[0] * split[1] *
/// split[2]` equal sub-boxes, x fastest, then y, then z. Each split count must be positive.
pub fn compute_equal_parts_bboxes(
    bbox: &AlignedBox3d,
    split: [i32; 3],
) -> Result<Vec<AlignedBox3d>> {
    check_gt!(split[0], 0);
    check_gt!(split[1], 0);
    check_gt!(split[2], 0);

    let extent = bbox.diagonal();
    let size = Vector3d::new(
        extent.x / f64::from(split[0]),
        extent.y / f64::from(split[1]),
        extent.z / f64::from(split[2]),
    );

    let mut bboxes = Vec::with_capacity((split[0] * split[1] * split[2]) as usize);
    for k in 0..split[2] {
        for j in 0..split[1] {
            for i in 0..split[0] {
                let min = Vector3d::new(
                    bbox.min.x + f64::from(i) * size.x,
                    bbox.min.y + f64::from(j) * size.y,
                    bbox.min.z + f64::from(k) * size.z,
                );
                bboxes.push(AlignedBox3d::new(min, min + size));
            }
        }
    }
    Ok(bboxes)
}
