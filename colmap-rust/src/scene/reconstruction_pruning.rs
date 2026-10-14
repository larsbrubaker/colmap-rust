//! Port of `colmap/scene/reconstruction_pruning.h` / `.cc` (and colmap-sharp's
//! `Scene/ReconstructionPruning.cs`): `FindRedundantPoints3D`, which greedily selects the 3D
//! points of a [`Reconstruction`] that add image coverage (an 8x8 tile grid per image) and
//! reports the rest as redundant. Tests: `tests/scene/reconstruction_pruning.rs`
//! (`reconstruction_pruning_test.cc` 1:1) and `tests/scene/rust_only_reconstruction_pruning.rs`.
//!
//! Tier A (exact). The priority queue orders by `(gain, point3D_id)`, a total order over the
//! queued entries (a point is re-queued only after it was popped), so the selection does not
//! depend on the heap implementation or on the order points are pushed. The result lists the
//! redundant ids in ascending id order (the reconstruction's `BTreeMap`, entry 213), where
//! COLMAP's is its hash map's; its tests compare order-insensitively. The per-image maps are
//! looked up only, never iterated.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

use crate::math::fns;
use crate::scene::point3d::Point3D;
use crate::scene::reconstruction::Reconstruction;
use crate::util::types::{ImageId, Point3DId};
use crate::Result;

const NUM_IMAGE_TILES_PER_DIM: i32 = 8;
const NUM_IMAGE_TILES: usize = (NUM_IMAGE_TILES_PER_DIM * NUM_IMAGE_TILES_PER_DIM) as usize;

/// A queued point: COLMAP's `Point3DInfo`, ordered by `std::tie(gain, point3D_id)`.
struct Point3DInfo {
    point3d_id: Point3DId,
    gain: f64,
}

impl PartialEq for Point3DInfo {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Point3DInfo {}

impl PartialOrd for Point3DInfo {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Point3DInfo {
    fn cmp(&self, other: &Self) -> Ordering {
        // Gains are finite sums of 1/sqrt(n) - 1/sqrt(n + 1) for n >= 1, never NaN.
        self.gain
            .total_cmp(&other.gain)
            .then(self.point3d_id.cmp(&other.point3d_id))
    }
}

// ComputeImageTileIdxs: the tile of every 2D point of every image.
fn compute_image_tile_idxs(
    num_tiles_per_dim: i32,
    reconstruction: &Reconstruction,
) -> Result<HashMap<ImageId, Vec<usize>>> {
    let mut image_tile_idxs = HashMap::with_capacity(reconstruction.num_images());
    for (&image_id, image) in reconstruction.images() {
        let camera = reconstruction.camera(image.camera_id())?;
        let tile_idxs = image
            .points2d()
            .iter()
            .map(|point2d| {
                // Clamp<int> converts its double argument to int first: truncation toward
                // zero. `as` saturates where C++'s conversion of an out-of-range value is UB.
                let tile_idx_x = ((f64::from(num_tiles_per_dim) * point2d.xy.x
                    / camera.width as f64) as i32)
                    .clamp(0, num_tiles_per_dim - 1);
                let tile_idx_y = ((f64::from(num_tiles_per_dim) * point2d.xy.y
                    / camera.height as f64) as i32)
                    .clamp(0, num_tiles_per_dim - 1);
                (tile_idx_x * num_tiles_per_dim + tile_idx_y) as usize
            })
            .collect();
        image_tile_idxs.insert(image_id, tile_idxs);
    }
    Ok(image_tile_idxs)
}

// ComputeCoverageGain: how much selecting the point would add to the coverage of the tiles
// its observations fall into.
fn compute_coverage_gain(
    point3d: &Point3D,
    num_selected_points3d_per_image_tile: &HashMap<ImageId, [i32; NUM_IMAGE_TILES]>,
    image_tile_idxs: &HashMap<ImageId, Vec<usize>>,
) -> Result<f64> {
    let mut gain = 0.0;
    for track_el in point3d.track.elements() {
        let tile_idx = tile_of(image_tile_idxs, track_el.image_id, track_el.point2d_idx)?;
        let selected = num_selected_points3d_per_image_tile.get(&track_el.image_id);
        crate::check!(selected.is_some());
        let n = 1 + selected.map_or(0, |tiles| tiles[tile_idx]);
        gain += 1. / fns::sqrt(f64::from(n)) - 1. / fns::sqrt(f64::from(1 + n));
    }
    Ok(gain)
}

// `image_tile_idxs.at(image_id).at(point2D_idx)`.
fn tile_of(
    image_tile_idxs: &HashMap<ImageId, Vec<usize>>,
    image_id: ImageId,
    point2d_idx: u32,
) -> Result<usize> {
    let tile = image_tile_idxs
        .get(&image_id)
        .and_then(|tiles| tiles.get(point2d_idx as usize));
    crate::check!(tile.is_some());
    Ok(tile.copied().unwrap_or_default())
}

/// Port of `colmap::FindRedundantPoints3D`: the ids of the 3D points whose coverage gain,
/// when greedily selecting points by largest gain, is at most `min_coverage_gain`, in
/// ascending id order.
pub fn find_redundant_points3d(
    min_coverage_gain: f64,
    reconstruction: &Reconstruction,
) -> Result<Vec<Point3DId>> {
    let image_tile_idxs = compute_image_tile_idxs(NUM_IMAGE_TILES_PER_DIM, reconstruction)?;
    let mut num_selected_points3d_per_image_tile: HashMap<ImageId, [i32; NUM_IMAGE_TILES]> =
        reconstruction
            .images()
            .keys()
            .map(|&image_id| (image_id, [0; NUM_IMAGE_TILES]))
            .collect();

    // std::priority_queue with has_left_smaller_gain is a max-heap, as BinaryHeap is.
    let mut priority_queue = BinaryHeap::with_capacity(reconstruction.num_points3d());
    for (&point3d_id, point3d) in reconstruction.points3d() {
        priority_queue.push(Point3DInfo {
            point3d_id,
            gain: compute_coverage_gain(
                point3d,
                &num_selected_points3d_per_image_tile,
                &image_tile_idxs,
            )?,
        });
    }

    // Lookup only, never iterated.
    let mut selected_point3d_ids = HashSet::with_capacity(reconstruction.num_points3d());
    while let Some(mut point3d_info) = priority_queue.pop() {
        if point3d_info.gain <= min_coverage_gain {
            break;
        }

        // If another point has been selected that shares an image with the current point,
        // then the gain of the current point might have changed.
        let point3d = reconstruction.point3d(point3d_info.point3d_id)?;
        let updated_gain = compute_coverage_gain(
            point3d,
            &num_selected_points3d_per_image_tile,
            &image_tile_idxs,
        )?;
        if updated_gain < point3d_info.gain {
            point3d_info.gain = updated_gain;
            priority_queue.push(point3d_info);
            continue;
        }

        for track_el in point3d.track.elements() {
            let tile_idx = tile_of(&image_tile_idxs, track_el.image_id, track_el.point2d_idx)?;
            let tiles = num_selected_points3d_per_image_tile.get_mut(&track_el.image_id);
            crate::check!(tiles.is_some());
            if let Some(tiles) = tiles {
                tiles[tile_idx] += 1;
            }
        }

        selected_point3d_ids.insert(point3d_info.point3d_id);
    }

    Ok(reconstruction
        .points3d()
        .keys()
        .copied()
        .filter(|point3d_id| !selected_point3d_ids.contains(point3d_id))
        .collect())
}
