//! Port of `colmap/scene/image.h` / `image.cc`: [`Image`], one camera exposure with its 2D
//! features ([`super::point2d::Point2D`]) and the ids of its camera
//! ([`super::camera::Camera`]) and frame ([`super::frame::Frame`]), which holds the pose.
//! Tests: `tests/scene/image.rs` (`image_test.cc` 1:1).
//!
//! Tier A (plain data and pose composition in COLMAP's order).
//!
//! **Ids into owning maps** (the ownership divergence in the [`super`] module header). COLMAP's
//! `Image` holds raw `Camera* camera_ptr_` and `Frame* frame_ptr_` that the reconstruction sets
//! and clears. Here an image stores only the ids; whoever owns the cameras, frames and rigs (the
//! reconstruction) passes them to the methods that need them, and each method checks it was handed
//! the image's own camera / frame and the frame's rig:
//! - `SetCameraPtr` / `SetFramePtr` are [`Image::verify_camera`] / [`Image::verify_frame`]:
//!   the same checks, with nothing stored;
//! - `CamFromWorld`, `ProjectionCenter`, `ViewingDirection`, `IsRefInFrame` take
//!   `(&Frame, &Rig)`; `ProjectPoint` also takes `&Camera`; `HasPose` takes the image's frame
//!   if it has one (`frames.get(&image.frame_id())`);
//! - `operator==` without the frames compares the image's own data
//!   ([`PartialEq`]); [`Image::eq_with_frames`] is COLMAP's full comparison, including the
//!   pose; `operator<<` is [`Image::display`], which takes the frame for `has_pose`;
//! - `CameraPtr`, `FramePtr`, `Reset*Ptr` and `Has*Ptr` have no counterpart, and
//!   `SetCameraId` / `SetFrameId` no longer refuse to change the id of a bound image.
//!
//! Point access by index panics out of range where COLMAP throws `std::out_of_range`
//! (`std::vector::at`), like `Track` (the `scene/track.rs` header). That is a deliberate
//! divergence: COLMAP's `Image::Point2D(idx)`, `SetPoint3DForPoint2D` and
//! `ResetPoint3DForPoint2D` go through `std::vector::at`; [`Image::point2d`],
//! [`Image::point2d_mut`], [`Image::set_point3d_for_point2d`] and
//! [`Image::reset_point3d_for_point2d`] index the `Vec` and panic.
//! `SetPoint3DForPoint2D`'s `THROW_CHECK_NE(point3D_id, kInvalidPoint3DId)` stays an `Err`. No
//! COLMAP code catches the exception: callers index with feature indices taken from the image
//! itself or from its correspondences, so an out-of-range index is a caller bug, an internal
//! invariant where CLAUDE.md's error rule allows a panic, and a `Result` on every point access
//! would burden triangulation and bundle adjustment. `tests/scene/rust_only_frame_image.rs`
//! pins it in `rust_only_image_point2d_out_of_range_panics` and
//! `rust_only_image_set_point3d_out_of_range_panics`.

use std::fmt;

use super::camera::Camera;
use super::frame::Frame;
use super::point2d::Point2D;
use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{Vector2d, Vector3d};
use crate::sensor::rig::Rig;
use crate::util::check::Result;
use crate::util::types::{
    CameraId, DataId, FrameId, ImageId, Point2DIdx, Point3DId, SensorId, SensorType,
    INVALID_CAMERA_ID, INVALID_FRAME_ID, INVALID_IMAGE_ID, INVALID_POINT3D_ID,
};
use crate::{check, check_eq, check_ne};

/// Port of `colmap::Image`. An image is the product of one camera exposure at a certain
/// location (parameterized as the pose of its frame). An image may share a camera with other
/// images if its intrinsics are the same, and a frame with the other sensors of its rig.
#[derive(Debug, Clone)]
pub struct Image {
    /// The name of the image, i.e. the relative path.
    name: String,
    /// All image points, including points that are not part of a 3D point track.
    points2d: Vec<Point2D>,
    /// Identifier of the image, [`INVALID_IMAGE_ID`] if not specified.
    image_id: ImageId,
    /// Identifier of the associated camera, [`INVALID_CAMERA_ID`] if not specified.
    camera_id: CameraId,
    /// Identifier of the frame holding the pose, [`INVALID_FRAME_ID`] if not specified.
    frame_id: FrameId,
    /// The number of 3D points the image observes: the `points2d` with a 3D point.
    num_points3d: Point2DIdx,
}

impl Default for Image {
    fn default() -> Self {
        Image {
            name: String::new(),
            points2d: Vec::new(),
            image_id: INVALID_IMAGE_ID,
            camera_id: INVALID_CAMERA_ID,
            frame_id: INVALID_FRAME_ID,
            num_points3d: 0,
        }
    }
}

impl PartialEq for Image {
    /// COLMAP's `operator==` without the pose, which lives in the frame: ids, name, points and
    /// the triangulated count. [`Image::eq_with_frames`] adds the pose.
    fn eq(&self, other: &Self) -> bool {
        self.image_id == other.image_id
            && self.camera_id == other.camera_id
            && self.frame_id == other.frame_id
            && self.name == other.name
            && self.num_points3d == other.num_points3d
            && self.points2d == other.points2d
    }
}

impl Image {
    /// An image with invalid ids, an empty name and no points.
    pub fn new() -> Self {
        Self::default()
    }

    /// `ImageId`: unique identifier of the image.
    pub fn image_id(&self) -> ImageId {
        self.image_id
    }

    /// `SetImageId`.
    pub fn set_image_id(&mut self, image_id: ImageId) {
        self.image_id = image_id;
    }

    /// `Name`: the image's name, i.e. its relative path.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// `Name()` (non-const), for in-place edits.
    pub fn name_mut(&mut self) -> &mut String {
        &mut self.name
    }

    /// `SetName`.
    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    /// `CameraId`. Multiple images may share a camera.
    pub fn camera_id(&self) -> CameraId {
        self.camera_id
    }

    /// `SetCameraId`. The id must be valid.
    pub fn set_camera_id(&mut self, camera_id: CameraId) -> Result<()> {
        check_ne!(camera_id, INVALID_CAMERA_ID);
        self.camera_id = camera_id;
        Ok(())
    }

    /// `HasCameraId`.
    pub fn has_camera_id(&self) -> bool {
        self.camera_id != INVALID_CAMERA_ID
    }

    /// `DataId`: the image as a measurement of its camera, `((CAMERA, camera_id), image_id)`.
    pub fn data_id(&self) -> DataId {
        DataId::new(self.sensor_id(), self.image_id)
    }

    /// Port of `SetCameraPtr`'s checks: `camera` has a valid id equal to the image's camera
    /// id. To bind the image to another camera, set its camera id first.
    pub fn verify_camera(&self, camera: &Camera) -> Result<()> {
        check_ne!(camera.camera_id, INVALID_CAMERA_ID);
        check_eq!(camera.camera_id, self.camera_id);
        Ok(())
    }

    /// `NumPoints2D`.
    pub fn num_points2d(&self) -> Point2DIdx {
        // `static_cast<point2D_t>(size())`: an image never has 2^32 features.
        self.points2d.len() as Point2DIdx
    }

    /// `NumPoints3D`: the number of points that are part of a 3D point track.
    pub fn num_points3d(&self) -> Point2DIdx {
        self.num_points3d
    }

    /// `FrameId`. Multiple images may share a frame.
    pub fn frame_id(&self) -> FrameId {
        self.frame_id
    }

    /// `SetFrameId`.
    pub fn set_frame_id(&mut self, frame_id: FrameId) {
        self.frame_id = frame_id;
    }

    /// `HasFrameId`.
    pub fn has_frame_id(&self) -> bool {
        self.frame_id != INVALID_FRAME_ID
    }

    /// Port of `SetFramePtr`'s checks: `frame` has a valid id equal to the image's frame id
    /// and contains the image's [`Image::data_id`]. To bind the image to another frame, set
    /// its frame id first.
    pub fn verify_frame(&self, frame: &Frame) -> Result<()> {
        check_ne!(frame.frame_id(), INVALID_FRAME_ID);
        check!(
            frame.has_data_id(self.data_id()),
            "Image {} does not exist in frame {}",
            self.image_id,
            frame.frame_id()
        );
        check_eq!(frame.frame_id(), self.frame_id);
        Ok(())
    }

    /// `IsRefInFrame`: whether the image's camera is the reference sensor of the frame's rig.
    pub fn is_ref_in_frame(&self, frame: &Frame, rig: &Rig) -> Result<bool> {
        self.check_is_image_frame(frame)?;
        check_eq!(rig.rig_id(), frame.rig_id());
        Ok(rig.is_ref_sensor(self.sensor_id()))
    }

    /// `CamFromWorld`: the frame's `sensor_from_rig * rig_from_world` for the image's camera
    /// (just `rig_from_world` for the rig's reference camera). Fails when the frame has no
    /// pose or `frame` / `rig` are not the image's.
    pub fn cam_from_world(&self, frame: &Frame, rig: &Rig) -> Result<Rigid3d> {
        self.check_is_image_frame(frame)?;
        frame.sensor_from_world(rig, self.sensor_id())
    }

    /// `HasPose`: whether the image's frame is posed. `frame` is the image's frame if it has
    /// one (an image without a frame has no pose); another frame has no pose for this image.
    pub fn has_pose(&self, frame: Option<&Frame>) -> bool {
        frame.is_some_and(|frame| frame.frame_id() == self.frame_id && frame.has_pose())
    }

    /// `Point2D(idx)`. Panics out of range (the file header).
    pub fn point2d(&self, point2d_idx: Point2DIdx) -> &Point2D {
        &self.points2d[point2d_idx as usize]
    }

    /// `Point2D(idx)` (non-const). Panics out of range (the file header).
    pub fn point2d_mut(&mut self, point2d_idx: Point2DIdx) -> &mut Point2D {
        &mut self.points2d[point2d_idx as usize]
    }

    /// `Points2D`.
    pub fn points2d(&self) -> &[Point2D] {
        &self.points2d
    }

    /// `Points2D()` (non-const). Changing which 3D points they observe through this bypasses
    /// [`Image::num_points3d`]'s bookkeeping, as in COLMAP.
    pub fn points2d_mut(&mut self) -> &mut Vec<Point2D> {
        &mut self.points2d
    }

    /// `SetPoints2D(const std::vector<Eigen::Vector2d>&)`: resize the points to `points` and
    /// set their coordinates. Like COLMAP, kept points keep their 3D point ids and the
    /// triangulated count is not recomputed.
    pub fn set_points2d_from_xy(&mut self, points: &[Vector2d]) {
        self.points2d.resize(points.len(), Point2D::default());
        for (point2d, xy) in self.points2d.iter_mut().zip(points) {
            point2d.xy = *xy;
        }
    }

    /// `SetPoints2D(std::vector<Point2D>)`: only on an image without points; counts the ones
    /// with a 3D point.
    pub fn set_points2d(&mut self, points: Vec<Point2D>) -> Result<()> {
        check!(self.points2d.is_empty());
        self.points2d = points;
        self.num_points3d = 0;
        for point2d in &self.points2d {
            if point2d.has_point3d() {
                self.num_points3d += 1;
            }
        }
        Ok(())
    }

    /// `SetPoint3DForPoint2D`: mark the point as triangulated, i.e. part of the track of
    /// `point3d_id`. Panics out of range (the file header).
    pub fn set_point3d_for_point2d(
        &mut self,
        point2d_idx: Point2DIdx,
        point3d_id: Point3DId,
    ) -> Result<()> {
        check_ne!(point3d_id, INVALID_POINT3D_ID);
        let point2d = &mut self.points2d[point2d_idx as usize];
        if !point2d.has_point3d() {
            self.num_points3d += 1;
        }
        point2d.point3d_id = point3d_id;
        Ok(())
    }

    /// `ResetPoint3DForPoint2D`: mark the point as not triangulated. Panics out of range
    /// (the file header).
    pub fn reset_point3d_for_point2d(&mut self, point2d_idx: Point2DIdx) {
        let point2d = &mut self.points2d[point2d_idx as usize];
        if point2d.has_point3d() {
            point2d.point3d_id = INVALID_POINT3D_ID;
            self.num_points3d -= 1;
        }
    }

    /// `HasPoint3D`: whether one of the image points observes `point3d_id`.
    pub fn has_point3d(&self, point3d_id: Point3DId) -> bool {
        self.points2d
            .iter()
            .any(|point2d| point2d.point3d_id == point3d_id)
    }

    /// `ProjectionCenter`: the camera center in world space.
    pub fn projection_center(&self, frame: &Frame, rig: &Rig) -> Result<Vector3d> {
        Ok(self.cam_from_world(frame, rig)?.tgt_origin_in_src())
    }

    /// `ViewingDirection`: the third row of the cam_from_world rotation.
    pub fn viewing_direction(&self, frame: &Frame, rig: &Rig) -> Result<Vector3d> {
        Ok(self
            .cam_from_world(frame, rig)?
            .rotation
            .to_rotation_matrix()
            .row(2))
    }

    /// `ProjectPoint`: reproject a world point into the image in pixels. `None` when the
    /// point is behind the camera; fails when `camera`, `frame` or `rig` are not the image's
    /// or the frame has no pose.
    pub fn project_point(
        &self,
        frame: &Frame,
        rig: &Rig,
        camera: &Camera,
        point3d: Vector3d,
    ) -> Result<Option<Vector2d>> {
        // COLMAP's THROW_CHECK(HasCameraPtr()).
        self.verify_camera(camera)?;
        let point3d_in_cam = self.cam_from_world(frame, rig)? * point3d;
        Ok(camera.img_from_cam(point3d_in_cam, true))
    }

    /// COLMAP's full `operator==`: [`PartialEq`] plus equal [`Image::has_pose`] and, when
    /// posed, equal `rig_from_world`. `frame` / `other_frame` are the images' frames, if any.
    pub fn eq_with_frames(
        &self,
        frame: Option<&Frame>,
        other: &Image,
        other_frame: Option<&Frame>,
    ) -> bool {
        let has_pose = self.has_pose(frame);
        let result = self == other && has_pose == other.has_pose(other_frame);
        match (frame, other_frame) {
            (Some(frame), Some(other_frame)) if has_pose => {
                result && frame.maybe_rig_from_world() == other_frame.maybe_rig_from_world()
            }
            _ => result,
        }
    }

    /// Port of `operator<<(std::ostream&, const Image&)`, which needs the image's frame (if
    /// any) for `has_pose`:
    /// `Image(image_id=1, camera_id=2, frame_id=3, name="test", has_pose=0, triangulated=0/0)`.
    pub fn display<'a>(&'a self, frame: Option<&'a Frame>) -> impl fmt::Display + 'a {
        ImageDisplay { image: self, frame }
    }

    /// The image's camera as a sensor, `(CAMERA, camera_id)`.
    fn sensor_id(&self) -> SensorId {
        SensorId::new(SensorType::Camera, self.camera_id)
    }

    /// The id-based form of COLMAP's `THROW_CHECK_NOTNULL(frame_ptr_)`: the frame passed in
    /// must be the image's.
    fn check_is_image_frame(&self, frame: &Frame) -> Result<()> {
        check_ne!(self.frame_id, INVALID_FRAME_ID);
        check_eq!(frame.frame_id(), self.frame_id);
        Ok(())
    }
}

/// [`Image::display`]'s formatter.
struct ImageDisplay<'a> {
    image: &'a Image,
    frame: Option<&'a Frame>,
}

impl fmt::Display for ImageDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let image = self.image;
        let id_or_invalid = |valid: bool, id: u32| {
            if valid {
                id.to_string()
            } else {
                "Invalid".to_string()
            }
        };
        write!(
            f,
            "Image(image_id={}, camera_id={}, frame_id={}, name=\"{}\", has_pose={}, \
             triangulated={}/{})",
            id_or_invalid(image.image_id != INVALID_IMAGE_ID, image.image_id),
            id_or_invalid(image.has_camera_id(), image.camera_id),
            id_or_invalid(image.has_frame_id(), image.frame_id),
            image.name,
            if image.has_pose(self.frame) { 1 } else { 0 },
            image.num_points3d,
            image.num_points2d()
        )
    }
}
