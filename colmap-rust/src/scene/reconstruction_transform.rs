//! Port of `colmap/scene/reconstruction.cc`, part 3: the [`Reconstruction`] methods that
//! measure or move the whole model in space — `Normalize`, `ComputeCentroid`,
//! `ComputeBoundingBox` (both through `ComputeBBBoxAndCentroid`), `Transform` and `Crop`.
//! The type is in [`crate::scene::reconstruction`], its queries in
//! [`crate::scene::reconstruction_queries`]. Port of colmap-sharp's
//! `Scene/Reconstruction.Queries.cs` (the same methods).
//! Tests: `tests/scene/reconstruction_transform.rs` (`reconstruction_test.cc` 1:1).
//!
//! Tier A: plain IEEE in COLMAP's order, except the centroid's summation order (the sort
//! that replaces `std::nth_element`, `geometry/normalization.rs` header) and `Transform`'s
//! rotation of vectors (`q * v`: `Quaterniond`'s `Mul<Vector3d>` in `linalg/quaternion.rs`,
//! and the FMA divergence in the `geometry/rigid3.rs` header).
//!
//! Translation notes: COLMAP's default arguments are the `DEFAULT_*` constants below; Rust
//! has none, so every caller passes them. Points are visited in ascending id order (the
//! `scene/reconstruction.rs` header), which only changes the centroid's summation order.

use std::collections::BTreeSet;

use crate::geometry::normalization::compute_bounding_box_and_centroid;
use crate::geometry::pose::transform_camera_world;
use crate::geometry::sim3::Sim3d;
use crate::linalg::{AlignedBox3d, Quaterniond, Vector3d};
use crate::scene::reconstruction::{image_id_of, Reconstruction};
use crate::util::types::FrameId;
use crate::Result;

/// `Normalize`'s default `extent`.
pub const DEFAULT_NORMALIZE_EXTENT: f64 = 10.0;
/// `Normalize`'s and `ComputeCentroid`'s default `min_percentile`.
pub const DEFAULT_NORMALIZE_MIN_PERCENTILE: f64 = 0.1;
/// `Normalize`'s and `ComputeCentroid`'s default `max_percentile`.
pub const DEFAULT_NORMALIZE_MAX_PERCENTILE: f64 = 0.9;

impl Reconstruction {
    /// `Normalize(fixed_scale, extent, min_percentile, max_percentile, use_images)`:
    /// translates the scene so the mean of the camera centers (`use_images`) or point
    /// locations within the percentiles is the origin and, unless `fixed_scale`, scales it so
    /// the percentile box's diagonal is `extent`. Returns the applied transform; with fewer
    /// than two registered frames (or points) nothing changes and it is the identity.
    /// COLMAP's defaults: `false`, [`DEFAULT_NORMALIZE_EXTENT`],
    /// [`DEFAULT_NORMALIZE_MIN_PERCENTILE`], [`DEFAULT_NORMALIZE_MAX_PERCENTILE`], `true`.
    pub fn normalize(
        &mut self,
        fixed_scale: bool,
        extent: f64,
        min_percentile: f64,
        max_percentile: f64,
        use_images: bool,
    ) -> Result<Sim3d> {
        crate::check_gt!(extent, 0.0);

        if (use_images && self.num_reg_frames() < 2) || (!use_images && self.points3d.len() < 2) {
            return Ok(Sim3d::identity());
        }

        let (bbox, centroid) =
            self.compute_bbox_and_centroid(min_percentile, max_percentile, use_images)?;

        // Calculate scale and translation, such that
        // translation is applied before scaling.
        let mut scale = 1.0;
        if !fixed_scale {
            let old_extent = bbox.diagonal().norm();
            if old_extent >= f64::EPSILON {
                scale = extent / old_extent;
            }
        }

        let tform = Sim3d::new(scale, Quaterniond::identity(), centroid * -scale);
        self.transform(&tform);

        Ok(tform)
    }

    /// `ComputeCentroid(min_percentile, max_percentile, use_images)`: the mean of the camera
    /// centers or point locations within the percentiles. COLMAP's defaults: 0.1, 0.9,
    /// `false`.
    pub fn compute_centroid(
        &self,
        min_percentile: f64,
        max_percentile: f64,
        use_images: bool,
    ) -> Result<Vector3d> {
        Ok(self
            .compute_bbox_and_centroid(min_percentile, max_percentile, use_images)?
            .1)
    }

    /// `ComputeBoundingBox(min_percentile, max_percentile, use_images)`: the box of the
    /// camera centers or point locations within the percentiles. COLMAP's defaults: 0.0,
    /// 1.0, `false`.
    pub fn compute_bounding_box(
        &self,
        min_percentile: f64,
        max_percentile: f64,
        use_images: bool,
    ) -> Result<AlignedBox3d> {
        Ok(self
            .compute_bbox_and_centroid(min_percentile, max_percentile, use_images)?
            .0)
    }

    /// `ComputeBBBoxAndCentroid`: both of the above; an empty model gives a zero box and
    /// centroid.
    fn compute_bbox_and_centroid(
        &self,
        min_percentile: f64,
        max_percentile: f64,
        use_images: bool,
    ) -> Result<(AlignedBox3d, Vector3d)> {
        let num_elements = if use_images {
            self.num_reg_frames()
        } else {
            self.points3d.len()
        };
        let zero = Vector3d::new(0.0, 0.0, 0.0);
        if num_elements == 0 {
            return Ok((AlignedBox3d::new(zero, zero), zero));
        }

        // Coordinates of image centers or point locations.
        let mut coords_x = Vec::with_capacity(num_elements);
        let mut coords_y = Vec::with_capacity(num_elements);
        let mut coords_z = Vec::with_capacity(num_elements);
        if use_images {
            for &frame_id in &self.reg_frame_ids {
                let frame = self.frame(frame_id)?;
                let rig = self.rig(frame.rig_id())?;
                for data_id in frame.image_ids() {
                    let proj_center = self
                        .image(image_id_of(data_id.id))?
                        .projection_center(frame, rig)?;
                    coords_x.push(proj_center.x);
                    coords_y.push(proj_center.y);
                    coords_z.push(proj_center.z);
                }
            }
        } else {
            for point3d in self.points3d.values() {
                coords_x.push(point3d.xyz.x);
                coords_y.push(point3d.xyz.y);
                coords_z.push(point3d.xyz.z);
            }
        }

        compute_bounding_box_and_centroid(
            min_percentile,
            max_percentile,
            coords_x,
            coords_y,
            coords_z,
        )
    }

    /// `Transform(new_from_old_world)`: applies the similarity to every rig's sensor
    /// offsets (scale only), posed frame and 3D point.
    pub fn transform(&mut self, new_from_old_world: &Sim3d) {
        for rig in self.rigs.values_mut() {
            for sensor_from_rig in rig.non_ref_sensors_mut().values_mut().flatten() {
                sensor_from_rig.translation *= new_from_old_world.scale;
            }
        }
        for frame in self.frames.values_mut() {
            if let Some(rig_from_world) = frame.maybe_rig_from_world() {
                frame.set_rig_from_world(transform_camera_world(
                    new_from_old_world,
                    &rig_from_world,
                ));
            }
        }
        for point3d in self.points3d.values_mut() {
            point3d.xyz = *new_from_old_world * point3d.xyz;
        }
    }

    /// `Crop(bbox)`: a copy with the same rigs, cameras, frames and images (image points no
    /// longer observing anything), the 3D points inside `bbox` (under new ids, ascending in
    /// the old ids' order) and only the frames that observe one of them registered.
    pub fn crop(&self, bbox: &AlignedBox3d) -> Result<Reconstruction> {
        let mut cropped_reconstruction = Reconstruction::new();
        for camera in self.cameras.values() {
            cropped_reconstruction.add_camera(camera.clone())?;
        }
        for rig in self.rigs.values() {
            cropped_reconstruction.add_rig(rig.clone())?;
        }
        for frame in self.frames.values() {
            cropped_reconstruction.add_frame(frame.clone())?;
        }
        for image in self.images.values() {
            let mut image = image.clone();
            for point2d_idx in 0..image.num_points2d() {
                image.reset_point3d_for_point2d(point2d_idx);
            }
            cropped_reconstruction.add_image(image)?;
        }
        let mut cropped_frame_ids: BTreeSet<FrameId> = BTreeSet::new();
        for point3d in self.points3d.values() {
            if bbox.contains(point3d.xyz) {
                for track_el in point3d.track.elements() {
                    cropped_frame_ids.insert(self.image(track_el.image_id)?.frame_id());
                }
                cropped_reconstruction.add_point3d(
                    point3d.xyz,
                    point3d.track.clone(),
                    point3d.color,
                )?;
            }
        }
        let frame_ids: Vec<FrameId> = cropped_reconstruction.frames.keys().copied().collect();
        for frame_id in frame_ids {
            if !cropped_frame_ids.contains(&frame_id) {
                cropped_reconstruction.deregister_frame(frame_id)?;
            }
        }
        Ok(cropped_reconstruction)
    }
}
