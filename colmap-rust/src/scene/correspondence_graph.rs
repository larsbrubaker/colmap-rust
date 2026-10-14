//! Port of `colmap/scene/correspondence_graph.h` / `.cc`: [`CorrespondenceGraph`], the
//! graph of image-to-image and feature-to-feature correspondences of a dataset. For every
//! image point it stores the points in other images it was matched to (the verified inlier
//! matches of a [`TwoViewGeometry`]), plus one two-view geometry per image pair. The
//! database cache builds it and the incremental mapper triangulates and merges tracks by
//! walking it. Port of colmap-sharp's `Scene/CorrespondenceGraph.cs`.
//! Tests: `tests/scene/correspondence_graph.rs` (`correspondence_graph_test.cc` 1:1) and
//! `tests/scene/rust_only_correspondence_graph.rs`.
//!
//! Tier A (exact): pure bookkeeping, and every per-point correspondence list is in the same
//! order as COLMAP's (the order in which matches were added).
//!
//! Translation notes:
//! - COLMAP keeps images and image pairs in `NodeHashMap` / `FlatHashMap`, whose iteration
//!   order depends on the hash backend. The outputs that expose it, [`image_pairs`] and
//!   [`num_matches_between_all_images`], come out here in insertion order (the order
//!   `add_two_view_geometry` added the pairs), as in colmap-sharp: the pairs live in a `Vec`
//!   that is never removed from, with a `HashMap` from pair id to position for lookups;
//!   docs/CPP_DIVERGENCES.md entry 210. Images are in a `BTreeMap`; their order is never
//!   observable (each image is flattened independently).
//! - `CorrespondenceRange` (a `[beg, end)` pointer pair) is a slice borrowed from the graph.
//! - Self-matches and invalid and duplicate matches are dropped and counted as in COLMAP,
//!   without COLMAP's `LOG(WARNING)` lines (the core has no logger; entry 211).
//! - A missing image or point index is an [`ErrorKind::OutOfRange`] error, where COLMAP's
//!   `std::unordered_map::at` / `std::vector::at` throw `std::out_of_range`.
//! - A rejected `add_two_view_geometry` (missing image, pair already added) leaves the graph
//!   unchanged, where COLMAP has already added the match count to both images' totals
//!   (entry 212).
//!
//! [`image_pairs`]: CorrespondenceGraph::image_pairs
//! [`num_matches_between_all_images`]: CorrespondenceGraph::num_matches_between_all_images

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt;

use crate::feature::types::{FeatureMatch, FeatureMatches};
use crate::scene::two_view_geometry::TwoViewGeometry;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{
    image_pair_to_pair_id, should_swap_image_pair, ImageId, ImagePairId, Point2DIdx,
    INVALID_IMAGE_ID, INVALID_POINT2D_IDX,
};
use crate::Result;

/// Port of `CorrespondenceGraph::Correspondence`: one image point that corresponds to the
/// queried one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Correspondence {
    /// The identifier of the corresponding image.
    pub image_id: ImageId,
    /// The index of the corresponding point in the corresponding image.
    pub point2d_idx: Point2DIdx,
}

impl Correspondence {
    /// `Correspondence(image_id, point2D_idx)`.
    pub const fn new(image_id: ImageId, point2d_idx: Point2DIdx) -> Self {
        Self {
            image_id,
            point2d_idx,
        }
    }
}

impl Default for Correspondence {
    /// `Correspondence()`: both ids invalid.
    fn default() -> Self {
        Self::new(INVALID_IMAGE_ID, INVALID_POINT2D_IDX)
    }
}

impl fmt::Display for Correspondence {
    /// `operator<<`: `Correspondence(image_id=1, point2D_idx=2)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Correspondence(image_id={}, point2D_idx={})",
            self.image_id, self.point2d_idx
        )
    }
}

/// `CorrespondenceGraph::Image`: the correspondences of one image.
#[derive(Debug, Clone, Default)]
struct ImageEntry {
    /// Number of 2D points with at least one correspondence to another image.
    num_observations: Point2DIdx,
    /// Total number of correspondences to other images. This measure is useful to find a
    /// good initial pair, that is connected to many images.
    num_correspondences: Point2DIdx,
    /// Correspondences to other images per image point. Added correspondences before
    /// `finalize`, empty afterwards.
    corrs: Vec<Vec<Correspondence>>,
    /// Flattened correspondences after `finalize`, empty before.
    flat_corrs: Vec<Correspondence>,
    /// For each point, the beginning of its correspondences in `flat_corrs`; the end of
    /// point i is the beginning of point i + 1. Its length is num_points2D + 1, the last
    /// element being `flat_corrs.len()`.
    flat_corr_begs: Vec<usize>,
}

/// `CorrespondenceGraph::ImagePair`.
#[derive(Debug, Clone, Default)]
struct ImagePair {
    /// The number of inlier matches between pairs of images.
    num_matches: Point2DIdx,
    /// The two-view geometry of the image pair without matches.
    two_view_geometry: TwoViewGeometry,
}

/// Port of `colmap::CorrespondenceGraph`. It should be accessed from the database cache.
#[derive(Debug, Clone, Default)]
pub struct CorrespondenceGraph {
    finalized: bool,
    images: BTreeMap<ImageId, ImageEntry>,
    /// The image pairs in insertion order (entry 210); never removed from.
    image_pairs: Vec<(ImagePairId, ImagePair)>,
    /// Position of each pair id in `image_pairs`. Only looked up, never iterated.
    image_pair_index: HashMap<ImagePairId, usize>,
}

fn image_does_not_exist(image_id: ImageId) -> ColmapError {
    ColmapError::new(
        ErrorKind::OutOfRange,
        format!("Image with ID {image_id} does not exist"),
    )
}

impl CorrespondenceGraph {
    /// `CorrespondenceGraph()`: an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    fn image_pair(&self, pair_id: ImagePairId) -> Option<&ImagePair> {
        self.image_pair_index
            .get(&pair_id)
            .map(|&index| &self.image_pairs[index].1)
    }

    fn image(&self, image_id: ImageId) -> Result<&ImageEntry> {
        self.images
            .get(&image_id)
            .ok_or_else(|| image_does_not_exist(image_id))
    }

    /// `Finalize`: flattens every image's correspondences into one compact vector (after
    /// checking the incrementally tracked observation count), so lookups are a slice of it.
    /// Fails if the graph is already finalized.
    pub fn finalize(&mut self) -> Result<()> {
        crate::check!(!self.finalized);
        self.finalized = true;

        // Flatten all correspondences into a compact representation.
        for image in self.images.values_mut() {
            // Verify incremental num_observations tracking is consistent.
            let mut num_total_corrs = 0usize;
            let mut expected_num_observations: Point2DIdx = 0;
            for corr in &image.corrs {
                num_total_corrs += corr.len();
                if !corr.is_empty() {
                    expected_num_observations += 1;
                }
            }
            crate::check_eq!(image.num_observations, expected_num_observations);

            // Reshuffle correspondences into flattened vector.
            let num_points2d = image.corrs.len();
            image.flat_corrs.reserve(num_total_corrs);
            image.flat_corr_begs.resize(num_points2d + 1, 0);
            for point2d_idx in 0..num_points2d {
                image.flat_corr_begs[point2d_idx] = image.flat_corrs.len();
                image
                    .flat_corrs
                    .extend_from_slice(&image.corrs[point2d_idx]);
            }
            image.flat_corr_begs[num_points2d] = image.flat_corrs.len();

            // Ensure we reserved enough space before insertion.
            crate::check_eq!(image.flat_corrs.len(), num_total_corrs);

            // Deallocate original data.
            image.corrs = Vec::new();
        }
        Ok(())
    }

    /// `NumImages`: the number of added images.
    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    /// `NumImagePairs`: the number of added image pairs.
    pub fn num_image_pairs(&self) -> usize {
        self.image_pairs.len()
    }

    /// `ExistsImage`.
    pub fn exists_image(&self, image_id: ImageId) -> bool {
        self.images.contains_key(&image_id)
    }

    /// `NumObservationsForImage`: the number of image points with at least one
    /// correspondence.
    pub fn num_observations_for_image(&self, image_id: ImageId) -> Result<Point2DIdx> {
        Ok(self.image(image_id)?.num_observations)
    }

    /// `NumCorrespondencesForImage`: the number of correspondences of the image.
    pub fn num_correspondences_for_image(&self, image_id: ImageId) -> Result<Point2DIdx> {
        Ok(self.image(image_id)?.num_correspondences)
    }

    /// `NumMatchesBetweenImages`: the number of matches between a pair of images, 0 for a
    /// pair that was never added. Fails only for an image id beyond `kMaxNumImages`.
    pub fn num_matches_between_images(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
    ) -> Result<Point2DIdx> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        Ok(self.image_pair(pair_id).map_or(0, |pair| pair.num_matches))
    }

    /// `NumMatchesBetweenAllImages`: the number of matches of every pair as `(pair id,
    /// number of matches)`, in insertion order (entry 210).
    pub fn num_matches_between_all_images(&self) -> Vec<(ImagePairId, Point2DIdx)> {
        self.image_pairs
            .iter()
            .map(|(pair_id, pair)| (*pair_id, pair.num_matches))
            .collect()
    }

    /// `ImagePairs`: every image pair id in the graph, in insertion order (entry 210).
    pub fn image_pairs(&self) -> Vec<ImagePairId> {
        self.image_pairs
            .iter()
            .map(|(pair_id, _)| *pair_id)
            .collect()
    }

    /// `AddImage`: adds an image with `num_points` points. Fails if it already exists.
    pub fn add_image(&mut self, image_id: ImageId, num_points: usize) -> Result<()> {
        crate::check!(!self.exists_image(image_id));
        let image = self.images.entry(image_id).or_default();
        image.corrs.resize(num_points, Vec::new());
        Ok(())
    }

    /// `AddTwoViewGeometry`: adds the two-view geometry and inlier matches between two
    /// images. Ignores a self-match, matches whose point indices are out of bounds, and
    /// duplicate matches between the same image points (COLMAP warns about each; entry
    /// 211). Fails if either image is missing or the pair was already added.
    pub fn add_two_view_geometry(
        &mut self,
        image_id1: ImageId,
        image_id2: ImageId,
        mut two_view_geometry: TwoViewGeometry,
    ) -> Result<()> {
        // Avoid self-matches - should only happen, if user provides custom matches.
        if image_id1 == image_id2 {
            return Ok(());
        }

        // Corresponding images.
        let mut image1 = self
            .images
            .remove(&image_id1)
            .ok_or_else(|| image_does_not_exist(image_id1))?;
        let Some(mut image2) = self.images.remove(&image_id2) else {
            self.images.insert(image_id1, image1);
            return Err(image_does_not_exist(image_id2));
        };
        let result = self.add_two_view_geometry_to(
            image_id1,
            image_id2,
            &mut image1,
            &mut image2,
            &mut two_view_geometry,
        );
        self.images.insert(image_id1, image1);
        self.images.insert(image_id2, image2);
        result
    }

    /// The body of [`Self::add_two_view_geometry`] with both images taken out of the map,
    /// so both can be mutated at once.
    fn add_two_view_geometry_to(
        &mut self,
        image_id1: ImageId,
        image_id2: ImageId,
        image1: &mut ImageEntry,
        image2: &mut ImageEntry,
        two_view_geometry: &mut TwoViewGeometry,
    ) -> Result<()> {
        // Set the number of all correspondences for this image pair. Further below, we will
        // make sure that only unique correspondences are counted.
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        crate::check!(
            !self.image_pair_index.contains_key(&pair_id),
            "Two view geometry for image pair was already added: image_id1={image_id1}, \
             image_id2={image_id2}"
        );

        // `static_cast<point2D_t>` of the match count: a pair never has 2^32 matches.
        let num_inlier_matches = two_view_geometry.inlier_matches.len() as Point2DIdx;
        // Store number of correspondences for each image to find good initial pair.
        image1.num_correspondences = image1.num_correspondences.wrapping_add(num_inlier_matches);
        image2.num_correspondences = image2.num_correspondences.wrapping_add(num_inlier_matches);
        let mut num_matches = num_inlier_matches;

        // Store all matches in correspondence graph data structure. This data-structure uses
        // more memory than storing the raw match matrices, but is significantly more
        // efficient when updating the correspondences in case an observation is
        // triangulated.
        for feature_match in &two_view_geometry.inlier_matches {
            let idx1 = feature_match.point2d_idx1 as usize;
            let idx2 = feature_match.point2d_idx2 as usize;
            let valid_idx1 = idx1 < image1.corrs.len();
            let valid_idx2 = idx2 < image2.corrs.len();

            if valid_idx1 && valid_idx2 {
                let corrs1 = &mut image1.corrs[idx1];
                // We add valid correspondences bidirectionally, so checking from only one
                // side is sufficient to detect duplicated matches.
                let duplicate = corrs1.iter().any(|corr| {
                    corr.image_id == image_id2 && corr.point2d_idx == feature_match.point2d_idx2
                });
                if duplicate {
                    image1.num_correspondences = image1.num_correspondences.wrapping_sub(1);
                    image2.num_correspondences = image2.num_correspondences.wrapping_sub(1);
                    num_matches = num_matches.wrapping_sub(1);
                } else {
                    corrs1.push(Correspondence::new(image_id2, feature_match.point2d_idx2));
                    // First correspondence makes this point an observation.
                    if corrs1.len() == 1 {
                        image1.num_observations += 1;
                    }
                    let corrs2 = &mut image2.corrs[idx2];
                    corrs2.push(Correspondence::new(image_id1, feature_match.point2d_idx1));
                    if corrs2.len() == 1 {
                        image2.num_observations += 1;
                    }
                }
            } else {
                image1.num_correspondences = image1.num_correspondences.wrapping_sub(1);
                image2.num_correspondences = image2.num_correspondences.wrapping_sub(1);
                num_matches = num_matches.wrapping_sub(1);
            }
        }

        // Clear and deallocate matches.
        let mut stored = std::mem::take(two_view_geometry);
        stored.inlier_matches = FeatureMatches::new();
        if should_swap_image_pair(image_id1, image_id2) {
            stored.invert();
        }
        self.image_pair_index
            .insert(pair_id, self.image_pairs.len());
        self.image_pairs.push((
            pair_id,
            ImagePair {
                num_matches,
                two_view_geometry: stored,
            },
        ));
        Ok(())
    }

    /// `FindCorrespondences`: the correspondences of an image point to all other images.
    /// Fails for a missing image or an out-of-range point index.
    pub fn find_correspondences(
        &self,
        image_id: ImageId,
        point2d_idx: Point2DIdx,
    ) -> Result<&[Correspondence]> {
        let image = self.image(image_id)?;
        let idx = point2d_idx as usize;
        let out_of_range = || {
            ColmapError::new(
                ErrorKind::OutOfRange,
                format!("point2D_idx={point2d_idx} in image_id={image_id} does not exist"),
            )
        };
        if !self.finalized {
            return image
                .corrs
                .get(idx)
                .map(Vec::as_slice)
                .ok_or_else(out_of_range);
        }
        match (
            image.flat_corr_begs.get(idx),
            image.flat_corr_begs.get(idx + 1),
        ) {
            (Some(&beg), Some(&end)) => Ok(&image.flat_corrs[beg..end]),
            _ => Err(out_of_range()),
        }
    }

    /// `ExtractCorrespondences`: copies the correspondences of an image point into `corrs`
    /// (cleared first; the vector is reused so per-point loops do not allocate).
    pub fn extract_correspondences(
        &self,
        image_id: ImageId,
        point2d_idx: Point2DIdx,
        corrs: &mut Vec<Correspondence>,
    ) -> Result<()> {
        let range = self.find_correspondences(image_id, point2d_idx)?;
        corrs.clear();
        corrs.extend_from_slice(range);
        Ok(())
    }

    /// `ExtractTransitiveCorrespondences`: transitively collects the correspondences of an
    /// image point, first its own, then those of the collected ones, and so forth until
    /// `transitivity` levels are exhausted or no more are found. The result has no
    /// duplicates and does not contain the given observation.
    pub fn extract_transitive_correspondences(
        &self,
        image_id: ImageId,
        point2d_idx: Point2DIdx,
        transitivity: usize,
        corrs: &mut Vec<Correspondence>,
    ) -> Result<()> {
        if transitivity == 1 {
            return self.extract_correspondences(image_id, point2d_idx, corrs);
        }

        corrs.clear();
        if !self.has_correspondences(image_id, point2d_idx)? {
            return Ok(());
        }

        // Push requested image point on queue to visit. Will be removed later.
        corrs.push(Correspondence::new(image_id, point2d_idx));

        // Visited (image, point2D) observations, for deduplication only; the iteration order
        // is never observed (the output is the discovery-ordered `corrs`), so a std hash set
        // is safe here.
        let mut image_corrs: HashSet<(ImageId, Point2DIdx)> = HashSet::new();
        image_corrs.insert((image_id, point2d_idx));

        let mut corr_queue_beg = 0usize;
        let mut corr_queue_end = 1usize;

        for _ in 0..transitivity {
            // Collect correspondences at transitive level t to all correspondences that were
            // collected at transitive level t - 1.
            for i in corr_queue_beg..corr_queue_end {
                let ref_corr = corrs[i];
                let ref_corr_range =
                    self.find_correspondences(ref_corr.image_id, ref_corr.point2d_idx)?;
                for corr in ref_corr_range {
                    // Check if correspondence already collected, otherwise collect.
                    if image_corrs.insert((corr.image_id, corr.point2d_idx)) {
                        corrs.push(*corr);
                    }
                }
            }

            // Move on to the next block of correspondences at next transitive level.
            corr_queue_beg = corr_queue_end;
            corr_queue_end = corrs.len();

            // No new correspondences collected in last transitivity level.
            if corr_queue_beg == corr_queue_end {
                break;
            }
        }

        // Remove first element, which is the given observation by swapping it with the last
        // collected correspondence.
        if corrs.len() > 1 {
            let last = corrs[corrs.len() - 1];
            corrs[0] = last;
        }
        corrs.pop();
        Ok(())
    }

    /// `ExtractMatchesBetweenImages`: every match between two images, in image 1's point
    /// order, written into `matches` (cleared first).
    pub fn extract_matches_between_images(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        matches: &mut FeatureMatches,
    ) -> Result<()> {
        matches.clear();

        let num_correspondences = self.num_matches_between_images(image_id1, image_id2)?;
        if num_correspondences == 0 {
            return Ok(());
        }

        matches.reserve(num_correspondences as usize);

        let image1 = self.image(image_id1)?;
        let num_points2d1 = if self.finalized {
            image1.flat_corr_begs.len() - 1
        } else {
            image1.corrs.len()
        };
        // Point indices of an image come from its `num_points` (a `point2D_t` in COLMAP).
        for point2d_idx1 in 0..num_points2d1 as Point2DIdx {
            for corr in self.find_correspondences(image_id1, point2d_idx1)? {
                if corr.image_id == image_id2 {
                    matches.push(FeatureMatch::new(point2d_idx1, corr.point2d_idx));
                }
            }
        }
        Ok(())
    }

    /// `ExtractTwoViewGeometry`: the two-view geometry of the pair, inverted if the ids are
    /// given in swapped order, with the inlier matches if `extract_inlier_matches`. Fails if
    /// the pair was never added.
    pub fn extract_two_view_geometry(
        &self,
        image_id1: ImageId,
        image_id2: ImageId,
        extract_inlier_matches: bool,
    ) -> Result<TwoViewGeometry> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        let image_pair = self.image_pair(pair_id);
        crate::check!(image_pair.is_some());
        let mut two_view_geometry = image_pair
            .map(|pair| pair.two_view_geometry.clone())
            .unwrap_or_default();
        if should_swap_image_pair(image_id1, image_id2) {
            two_view_geometry.invert();
        }
        // Extract after inversion, as they are extracted in the correct order.
        if extract_inlier_matches {
            self.extract_matches_between_images(
                image_id1,
                image_id2,
                &mut two_view_geometry.inlier_matches,
            )?;
        }
        Ok(two_view_geometry)
    }

    /// `UpdateTwoViewGeometry`: replaces the two-view geometry of an existing pair. The
    /// inlier matches of the given geometry are ignored (matches are stored separately).
    /// Fails if the pair was never added.
    pub fn update_two_view_geometry(
        &mut self,
        image_id1: ImageId,
        image_id2: ImageId,
        mut two_view_geometry: TwoViewGeometry,
    ) -> Result<()> {
        let pair_id = image_pair_to_pair_id(image_id1, image_id2)?;
        let index = self.image_pair_index.get(&pair_id).copied();
        crate::check!(index.is_some());
        two_view_geometry.inlier_matches = FeatureMatches::new();
        if should_swap_image_pair(image_id1, image_id2) {
            two_view_geometry.invert();
        }
        if let Some(index) = index {
            self.image_pairs[index].1.two_view_geometry = two_view_geometry;
        }
        Ok(())
    }

    /// `HasCorrespondences`: whether the image point has any correspondence.
    pub fn has_correspondences(&self, image_id: ImageId, point2d_idx: Point2DIdx) -> Result<bool> {
        Ok(!self.find_correspondences(image_id, point2d_idx)?.is_empty())
    }

    /// `IsTwoViewObservation`: whether the observation is part of a two-view track, i.e. it
    /// has exactly one correspondence, whose only correspondence is the observation.
    pub fn is_two_view_observation(
        &self,
        image_id: ImageId,
        point2d_idx: Point2DIdx,
    ) -> Result<bool> {
        let range = self.find_correspondences(image_id, point2d_idx)?;
        if range.len() != 1 {
            return Ok(false);
        }
        let other_range = self.find_correspondences(range[0].image_id, range[0].point2d_idx)?;
        Ok(other_range.len() == 1)
    }
}

impl fmt::Display for CorrespondenceGraph {
    /// `operator<<`: `CorrespondenceGraph(num_images=2, num_image_pairs=1)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "CorrespondenceGraph(num_images={}, num_image_pairs={})",
            self.num_images(),
            self.num_image_pairs()
        )
    }
}
