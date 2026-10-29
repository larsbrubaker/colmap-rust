//! Port of `colmap/scene/reconstruction.cc`, part 2: the [`Reconstruction`] methods that
//! read or summarize the whole model — `IsValid`, `TearDown`, `FindImageWithName`,
//! `FindCommonRegImageIds`, the `Compute*` statistics, `UpdatePoint3DErrors` and
//! `operator<<`. The type and its add/delete/register operations are in
//! [`crate::scene::reconstruction`]. Port of colmap-sharp's `Scene/Reconstruction.Queries.cs`.
//! Tests: `tests/scene/reconstruction_queries.rs` and `tests/scene/rust_only_reconstruction.rs`.
//!
//! Translation notes:
//! - `IsValid`'s pointer checks (`HasRigPtr`, `RigPtr() == &rigs_.at(..)`, and the camera
//!   and frame ones) become the id checks they guarded: the referenced object exists and
//!   holds this one (the ownership divergence in the [`crate::scene`] module header).
//!   COLMAP's `LOG(WARNING)` naming the first broken association is not emitted (the core
//!   has no logger, and the line is diagnostic only).
//! - Iteration is in ascending id order (the `scene/reconstruction.rs` header):
//!   `find_image_with_name` returns the
//!   smallest-id image among several with the same name.

use std::collections::BTreeSet;
use std::fmt;

use crate::math::fns;
use crate::scene::image::Image;
use crate::scene::projection::calculate_squared_reprojection_error;
use crate::scene::reconstruction::{image_id_of, Reconstruction};
use crate::util::types::{ImageId, RigId, SensorType};
use crate::Result;

impl Reconstruction {
    /// `IsValid`: whether every association is consistent — rigs' cameras exist, frames'
    /// rigs exist and hold their sensors, frames' images exist, images' cameras and frames
    /// exist and the frame holds the image, the 2D-3D associations agree in both
    /// directions, and every registered frame exists and has a pose.
    pub fn is_valid(&self) -> bool {
        // Check object associations: rig-frame-image-camera references.
        // Check rigs.
        for rig in self.rigs.values() {
            for sensor_id in rig.sensor_ids() {
                match sensor_id.sensor_type {
                    SensorType::Camera => {
                        if !self.exists_camera(sensor_id.id) {
                            return false;
                        }
                    }
                    // Only camera sensors are currently supported.
                    SensorType::Imu | SensorType::Invalid => {}
                }
            }
        }
        // Check frames.
        for frame in self.frames.values() {
            if !frame.has_rig_id() {
                return false;
            }
            let Some(rig) = self.rigs.get(&frame.rig_id()) else {
                return false;
            };
            for data_id in frame.data_ids() {
                if !rig.has_sensor(data_id.sensor_id) {
                    return false;
                }
                match data_id.sensor_id.sensor_type {
                    SensorType::Camera => {
                        if !self.exists_image(image_id_of(data_id.id)) {
                            return false;
                        }
                    }
                    // Only camera data is currently supported.
                    SensorType::Imu | SensorType::Invalid => {}
                }
            }
        }
        // Check images.
        for image in self.images.values() {
            if !image.has_camera_id() || !self.exists_camera(image.camera_id()) {
                return false;
            }
            if !image.has_frame_id() {
                return false;
            }
            let Some(frame) = self.frames.get(&image.frame_id()) else {
                return false;
            };
            if !frame.has_data_id(image.data_id()) {
                return false;
            }
            // Check 2D-3D associations: point2D -> point3D direction.
            let mut actual_num_points3d = 0;
            for point2d in image.points2d() {
                if point2d.has_point3d() {
                    actual_num_points3d += 1;
                    if !self.exists_point3d(point2d.point3d_id) {
                        return false;
                    }
                }
            }
            if image.num_points3d() != actual_num_points3d {
                return false;
            }
        }
        // Check 2D-3D associations: point3D -> point2D direction.
        for (&point3d_id, point3d) in &self.points3d {
            for track_el in point3d.track.elements() {
                let Some(image) = self.images.get(&track_el.image_id) else {
                    return false;
                };
                if track_el.point2d_idx >= image.num_points2d() {
                    return false;
                }
                let point2d = image.point2d(track_el.point2d_idx);
                if !point2d.has_point3d() || point2d.point3d_id != point3d_id {
                    return false;
                }
            }
        }
        // Check registered frames exist and have poses.
        for frame_id in &self.reg_frame_ids {
            match self.frames.get(frame_id) {
                Some(frame) if frame.has_pose() => {}
                _ => return false,
            }
        }
        true
    }

    /// `TearDown`: removes the frames without a pose with their images, then the rigs no
    /// remaining frame uses with their cameras, and compresses every track.
    pub fn tear_down(&mut self) {
        // Remove all non-registered frames/images.
        let mut keep_rig_ids: BTreeSet<RigId> = BTreeSet::new();
        let images = &mut self.images;
        self.frames.retain(|_, frame| {
            if frame.has_pose() {
                keep_rig_ids.insert(frame.rig_id());
                true
            } else {
                for data_id in frame.image_ids() {
                    images.remove(&image_id_of(data_id.id));
                }
                false
            }
        });

        // Remove all unused rigs and corresponding sensors.
        let cameras = &mut self.cameras;
        self.rigs.retain(|rig_id, rig| {
            if keep_rig_ids.contains(rig_id) {
                return true;
            }
            for sensor_id in rig.sensor_ids() {
                match sensor_id.sensor_type {
                    SensorType::Camera => {
                        cameras.remove(&sensor_id.id);
                    }
                    SensorType::Imu | SensorType::Invalid => {}
                }
            }
            false
        });

        // Compress tracks.
        for point3d in self.points3d.values_mut() {
            point3d.track.compress();
        }
    }

    /// `FindImageWithName`: the image with the given name, the smallest id among several
    /// (the `scene/reconstruction.rs` header); `None` if there is none.
    pub fn find_image_with_name(&self, name: &str) -> Option<&Image> {
        self.images.values().find(|image| image.name() == name)
    }

    /// `FindCommonRegImageIds`: `(this image id, other image id)` for every registered image
    /// of this reconstruction whose name matches an image with a pose in `other`, in this
    /// reconstruction's registration order.
    pub fn find_common_reg_image_ids(
        &self,
        other: &Reconstruction,
    ) -> Result<Vec<(ImageId, ImageId)>> {
        let mut common_reg_image_ids = Vec::new();
        for &frame_id in &self.reg_frame_ids {
            let frame = self.frame(frame_id)?;
            for data_id in frame.image_ids() {
                let image = self.image(image_id_of(data_id.id))?;
                if let Some(other_image) = other.find_image_with_name(image.name()) {
                    if other.frame(other_image.frame_id())?.has_pose() {
                        common_reg_image_ids.push((image.image_id(), other_image.image_id()));
                    }
                }
            }
        }
        Ok(common_reg_image_ids)
    }

    /// `ComputeNumObservations`: the number of triangulated points over the registered
    /// images.
    pub fn compute_num_observations(&self) -> Result<usize> {
        let mut num_obs = 0;
        for image_id in self.reg_image_ids()? {
            num_obs += self.image(image_id)?.num_points3d() as usize;
        }
        Ok(num_obs)
    }

    /// `ComputeMeanTrackLength`: observations per 3D point, 0 without points.
    pub fn compute_mean_track_length(&self) -> Result<f64> {
        if self.points3d.is_empty() {
            Ok(0.0)
        } else {
            Ok(self.compute_num_observations()? as f64 / self.points3d.len() as f64)
        }
    }

    /// `ComputeMeanObservationsPerRegImage`: observations per registered image, 0 without
    /// registered images.
    pub fn compute_mean_observations_per_reg_image(&self) -> Result<f64> {
        if self.num_reg_images() == 0 {
            Ok(0.0)
        } else {
            Ok(self.compute_num_observations()? as f64 / self.num_reg_images() as f64)
        }
    }

    /// `ComputeMeanReprojectionError`: the mean error of the points that have one, 0 if
    /// none has. Sums in ascending point id order (the
    /// `scene/reconstruction.rs` header).
    pub fn compute_mean_reprojection_error(&self) -> f64 {
        let mut error_sum = 0.0;
        let mut num_valid_errors = 0usize;
        for point3d in self.points3d.values() {
            if point3d.has_error() {
                error_sum += point3d.error;
                num_valid_errors += 1;
            }
        }

        if num_valid_errors == 0 {
            0.0
        } else {
            error_sum / num_valid_errors as f64
        }
    }

    /// `UpdatePoint3DErrors`: sets each 3D point's error to the mean reprojection error of
    /// its observations (0 for an empty track). Fails if an observing image's frame has no
    /// pose or an object it refers to is missing.
    pub fn update_point3d_errors(&mut self) -> Result<()> {
        // Errors are computed against the immutable model, then written.
        let mut errors = Vec::with_capacity(self.points3d.len());
        for (&point3d_id, point3d) in &self.points3d {
            if point3d.track.length() == 0 {
                errors.push((point3d_id, 0.0));
                continue;
            }
            let mut error = 0.0;
            for track_el in point3d.track.elements() {
                let image = self.image(track_el.image_id)?;
                let point2d = image.point2d(track_el.point2d_idx);
                let camera = self.camera(image.camera_id())?;
                let frame = self.frame(image.frame_id())?;
                let rig = self.rig(frame.rig_id())?;
                let cam_from_world = image.cam_from_world(frame, rig)?;
                error += fns::sqrt(calculate_squared_reprojection_error(
                    point2d.xy,
                    point3d.xyz,
                    &cam_from_world,
                    camera,
                ));
            }
            error /= point3d.track.length() as f64;
            errors.push((point3d_id, error));
        }
        for (point3d_id, error) in errors {
            self.point3d_mut(point3d_id)?.error = error;
        }
        Ok(())
    }
}

impl fmt::Display for Reconstruction {
    /// `operator<<`: `Reconstruction(num_rigs=.., num_cameras=.., num_frames=..,
    /// num_reg_frames=.., num_images=.., num_points3D=..)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Reconstruction(num_rigs={}, num_cameras={}, num_frames={}, num_reg_frames={}, \
             num_images={}, num_points3D={})",
            self.num_rigs(),
            self.num_cameras(),
            self.num_frames(),
            self.num_reg_frames(),
            self.num_images(),
            self.num_points3d()
        )
    }
}
