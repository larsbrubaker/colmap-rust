//! Port of `colmap/scene/track.h` / `track.cc`: [`TrackElement`] (one observation of a 3D
//! point: an image and a 2D point index in it) and [`Track`], all observations of a 3D
//! point. Used by [`super::point3d::Point3D`]. Tests: `tests/scene/track.rs`
//! (`track_test.cc` 1:1).
//!
//! Tier A (plain data). Translation notes:
//! - `Element(idx)` / `SetElement(idx, ..)` use `std::vector::at`, which throws
//!   `std::out_of_range`; here they index the `Vec` and panic out of range. They are internal
//!   invariants (callers iterate `0..length()`), and a `Result` on every element access of
//!   every track would burden the hot loops of triangulation and bundle adjustment.
//! - `DeleteElement(idx)` is a `THROW_CHECK_LT` and returns `Err` like the other checks.

use std::fmt;

use crate::check_lt;
use crate::util::check::Result;
use crate::util::types::{ImageId, Point2DIdx, INVALID_IMAGE_ID, INVALID_POINT2D_IDX};

/// Port of `colmap::TrackElement`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackElement {
    /// The image in which the track element is observed.
    pub image_id: ImageId,
    /// The point in the image that the track element is observed.
    pub point2d_idx: Point2DIdx,
}

impl Default for TrackElement {
    /// Both ids invalid, as COLMAP's default constructor sets them.
    fn default() -> Self {
        Self {
            image_id: INVALID_IMAGE_ID,
            point2d_idx: INVALID_POINT2D_IDX,
        }
    }
}

impl TrackElement {
    /// `TrackElement(image_id, point2D_idx)`.
    pub const fn new(image_id: ImageId, point2d_idx: Point2DIdx) -> Self {
        Self {
            image_id,
            point2d_idx,
        }
    }
}

/// Port of `colmap::Track`: stores all observations of a 3D point.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Track {
    elements: Vec<TrackElement>,
}

impl Track {
    /// An empty track.
    pub fn new() -> Self {
        Self::default()
    }

    /// `Length`: the number of track elements.
    pub fn length(&self) -> usize {
        self.elements.len()
    }

    /// `Elements() const`.
    pub fn elements(&self) -> &Vec<TrackElement> {
        &self.elements
    }

    /// `Elements()`, the mutable overload.
    pub fn elements_mut(&mut self) -> &mut Vec<TrackElement> {
        &mut self.elements
    }

    /// `SetElements`.
    pub fn set_elements(&mut self, elements: Vec<TrackElement>) {
        self.elements = elements;
    }

    /// `Element(idx) const`. Panics when `idx` is out of range (see the module header).
    pub fn element(&self, idx: usize) -> &TrackElement {
        &self.elements[idx]
    }

    /// `Element(idx)`, the mutable overload. Panics when `idx` is out of range.
    pub fn element_mut(&mut self, idx: usize) -> &mut TrackElement {
        &mut self.elements[idx]
    }

    /// `SetElement`. Panics when `idx` is out of range.
    pub fn set_element(&mut self, idx: usize, element: TrackElement) {
        self.elements[idx] = element;
    }

    /// `AddElement(const TrackElement&)`.
    pub fn add_element(&mut self, element: TrackElement) {
        self.elements.push(element);
    }

    /// `AddElement(image_id, point2D_idx)`.
    pub fn add_element_ids(&mut self, image_id: ImageId, point2d_idx: Point2DIdx) {
        self.elements.push(TrackElement::new(image_id, point2d_idx));
    }

    /// `AddElements`: appends `elements` in order.
    pub fn add_elements(&mut self, elements: &[TrackElement]) {
        self.elements.extend_from_slice(elements);
    }

    /// `DeleteElement(idx)`: removes the element at `idx`, keeping the order of the rest.
    pub fn delete_element(&mut self, idx: usize) -> Result<()> {
        check_lt!(idx, self.elements.len());
        self.elements.remove(idx);
        Ok(())
    }

    /// `DeleteElement(image_id, point2D_idx)`: removes every element equal to the pair
    /// (`std::remove_if` + `erase`, order kept).
    pub fn delete_element_ids(&mut self, image_id: ImageId, point2d_idx: Point2DIdx) {
        self.elements
            .retain(|e| !(e.image_id == image_id && e.point2d_idx == point2d_idx));
    }

    /// `Reserve`: capacity for at least `num_elements` elements in total. Like
    /// `std::vector::reserve`, the request is a total, not an addition (Rust's
    /// `Vec::reserve_exact` takes the additional count, hence the subtraction).
    pub fn reserve(&mut self, num_elements: usize) {
        let additional = num_elements.saturating_sub(self.elements.len());
        self.elements.reserve_exact(additional);
    }

    /// `Compress`: shrinks the capacity to the length to save memory.
    pub fn compress(&mut self) {
        self.elements.shrink_to_fit();
    }
}

impl fmt::Display for TrackElement {
    /// `TrackElement(image_id=<id>, point2D_idx=<idx>)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TrackElement(image_id={}, point2D_idx={})",
            self.image_id, self.point2d_idx
        )
    }
}

impl fmt::Display for Track {
    /// `Track(elements=[TrackElement(..), ...])`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Track(elements=[")?;
        for (i, element) in self.elements.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{element}")?;
        }
        f.write_str("])")
    }
}
