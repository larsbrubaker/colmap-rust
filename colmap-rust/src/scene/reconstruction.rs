//! Port of `colmap/scene/reconstruction.h` / `.cc`, part 1: [`Reconstruction`], the owner of
//! a reconstruction's rigs, cameras, frames, images and 3D points, with its accessors and the
//! operations that add, delete, register and de-register them. Its queries, statistics,
//! validity check and tear-down are in [`crate::scene::reconstruction_queries`]. Port of
//! colmap-sharp's `Scene/Reconstruction.cs`.
//! Tests: `tests/scene/reconstruction.rs` and `tests/scene/reconstruction_queries.rs`
//! (`reconstruction_test.cc` 1:1 for the ported methods) and
//! `tests/scene/rust_only_reconstruction.rs`.
//!
//! Tier A: pure bookkeeping, except `merge_points3d`'s weighted mean (plain IEEE in COLMAP's
//! order).
//!
//! Translation notes:
//! - Ownership (docs/CPP_DIVERGENCES.md entry 170): frames and images store ids, not
//!   pointers. Where COLMAP's `AddFrame` / `AddImage` set or compare a `Set*Ptr`, this checks
//!   the same conditions with `Frame::verify_rig`, `Image::verify_camera` and
//!   `Image::verify_frame` against the objects in its maps. A copy is the derived `Clone`;
//!   COLMAP's copy constructor re-links pointers, which ids do not need.
//! - The maps are `BTreeMap`s, so every iteration (and `find_image_with_name` among
//!   duplicate names) is in ascending id order, where COLMAP iterates `NodeHashMap`s
//!   (entry 213). `reg_frame_ids` is COLMAP's vector, in registration order.
//! - A missing object is an [`ErrorKind::OutOfRange`] error with COLMAP's message
//!   ("Rig with ID 1 does not exist"), where COLMAP throws `std::out_of_range`.
//! - The mutable accessors hand out `&mut` objects as COLMAP's non-const accessors do.
//!   Changing an attached image's camera or frame id, or a frame's rig id, through them is
//!   not rejected at the setter (there is no pointer to guard it, entry 170); `is_valid`
//!   reports it, and every owner-taking method of `Frame` / `Image` rejects the mismatched
//!   objects (entry 214).
//! - Not yet ported (later Phase 4 slices): `Load` (needs `DatabaseCache`), `Normalize`,
//!   `ComputeCentroid`, `ComputeBoundingBox`, `Transform`, `Crop` (need `Sim3d`),
//!   `TranscribeImageIdsToDatabase` (needs `Database`), `Read*` / `Write*`
//!   (`reconstruction_io`), the PLY import/export, `ExtractColors*` and `CreateImageDirs`.

use std::collections::{BTreeMap, HashMap};

use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{Vector3d, Vector3ub};
use crate::scene::camera::Camera;
use crate::scene::frame::Frame;
use crate::scene::image::Image;
use crate::scene::point3d::Point3D;
use crate::scene::track::{Track, TrackElement};
use crate::sensor::rig::Rig;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::types::{
    CameraId, FrameId, ImageId, Point2DIdx, Point3DId, RigId, SensorId, SensorType,
    INVALID_FRAME_ID,
};
use crate::Result;

/// Port of `colmap::Reconstruction`: the cameras, rigs, frames, images and 3D points of a
/// sparse model, and which frames are registered.
#[derive(Debug, Clone, Default)]
pub struct Reconstruction {
    pub(super) rigs: BTreeMap<RigId, Rig>,
    pub(super) cameras: BTreeMap<CameraId, Camera>,
    pub(super) frames: BTreeMap<FrameId, Frame>,
    pub(super) images: BTreeMap<ImageId, Image>,
    pub(super) points3d: BTreeMap<Point3DId, Point3D>,
    /// `{frame_id, ...}` of the registered frames, in registration order.
    pub(super) reg_frame_ids: Vec<FrameId>,
    /// The number of images in the registered frames.
    pub(super) num_reg_images: usize,
    /// The largest 3D point id used so far; new points get the next one.
    pub(super) max_point3d_id: Point3DId,
}

fn does_not_exist(kind: &str, id: impl std::fmt::Display) -> ColmapError {
    ColmapError::new(
        ErrorKind::OutOfRange,
        format!("{kind} with ID {id} does not exist"),
    )
}

/// The image id of a frame's camera measurement: data ids of images hold an `image_t`
/// (`static_cast<image_t>` of the 64-bit `data_t::id` in COLMAP).
pub(super) fn image_id_of(data_id_id: u64) -> ImageId {
    data_id_id as ImageId
}

impl Reconstruction {
    /// `Reconstruction()`: empty.
    pub fn new() -> Self {
        Self::default()
    }

    /// `NumRigs`.
    pub fn num_rigs(&self) -> usize {
        self.rigs.len()
    }

    /// `NumCameras`.
    pub fn num_cameras(&self) -> usize {
        self.cameras.len()
    }

    /// `NumFrames`.
    pub fn num_frames(&self) -> usize {
        self.frames.len()
    }

    /// `NumRegFrames`.
    pub fn num_reg_frames(&self) -> usize {
        self.reg_frame_ids.len()
    }

    /// `NumRegImages`: the number of images in the registered frames.
    pub fn num_reg_images(&self) -> usize {
        self.num_reg_images
    }

    /// `NumImages`.
    pub fn num_images(&self) -> usize {
        self.images.len()
    }

    /// `NumPoints3D`.
    pub fn num_points3d(&self) -> usize {
        self.points3d.len()
    }

    /// `Rig(rig_id)`.
    pub fn rig(&self, rig_id: RigId) -> Result<&Rig> {
        self.rigs
            .get(&rig_id)
            .ok_or_else(|| does_not_exist("Rig", rig_id))
    }

    /// `Rig(rig_id)`, mutable.
    pub fn rig_mut(&mut self, rig_id: RigId) -> Result<&mut Rig> {
        self.rigs
            .get_mut(&rig_id)
            .ok_or_else(|| does_not_exist("Rig", rig_id))
    }

    /// `Camera(camera_id)`.
    pub fn camera(&self, camera_id: CameraId) -> Result<&Camera> {
        self.cameras
            .get(&camera_id)
            .ok_or_else(|| does_not_exist("Camera", camera_id))
    }

    /// `Camera(camera_id)`, mutable.
    pub fn camera_mut(&mut self, camera_id: CameraId) -> Result<&mut Camera> {
        self.cameras
            .get_mut(&camera_id)
            .ok_or_else(|| does_not_exist("Camera", camera_id))
    }

    /// `Frame(frame_id)`.
    pub fn frame(&self, frame_id: FrameId) -> Result<&Frame> {
        self.frames
            .get(&frame_id)
            .ok_or_else(|| does_not_exist("Frame", frame_id))
    }

    /// `Frame(frame_id)`, mutable.
    pub fn frame_mut(&mut self, frame_id: FrameId) -> Result<&mut Frame> {
        self.frames
            .get_mut(&frame_id)
            .ok_or_else(|| does_not_exist("Frame", frame_id))
    }

    /// `Image(image_id)`.
    pub fn image(&self, image_id: ImageId) -> Result<&Image> {
        self.images
            .get(&image_id)
            .ok_or_else(|| does_not_exist("Image", image_id))
    }

    /// `Image(image_id)`, mutable.
    pub fn image_mut(&mut self, image_id: ImageId) -> Result<&mut Image> {
        self.images
            .get_mut(&image_id)
            .ok_or_else(|| does_not_exist("Image", image_id))
    }

    /// `Point3D(point3D_id)`.
    pub fn point3d(&self, point3d_id: Point3DId) -> Result<&Point3D> {
        self.points3d
            .get(&point3d_id)
            .ok_or_else(|| does_not_exist("Point3D", point3d_id))
    }

    /// `Point3D(point3D_id)`, mutable.
    pub fn point3d_mut(&mut self, point3d_id: Point3DId) -> Result<&mut Point3D> {
        self.points3d
            .get_mut(&point3d_id)
            .ok_or_else(|| does_not_exist("Point3D", point3d_id))
    }

    /// `Rigs`, in ascending id order (entry 213).
    pub fn rigs(&self) -> &BTreeMap<RigId, Rig> {
        &self.rigs
    }

    /// `Cameras`, in ascending id order.
    pub fn cameras(&self) -> &BTreeMap<CameraId, Camera> {
        &self.cameras
    }

    /// `Frames`, in ascending id order.
    pub fn frames(&self) -> &BTreeMap<FrameId, Frame> {
        &self.frames
    }

    /// `RegFrameIds`, in registration order.
    pub fn reg_frame_ids(&self) -> &[FrameId] {
        &self.reg_frame_ids
    }

    /// `Images`, in ascending id order.
    pub fn images(&self) -> &BTreeMap<ImageId, Image> {
        &self.images
    }

    /// `Points3D`, in ascending id order.
    pub fn points3d(&self) -> &BTreeMap<Point3DId, Point3D> {
        &self.points3d
    }

    /// `ExistsRig`.
    pub fn exists_rig(&self, rig_id: RigId) -> bool {
        self.rigs.contains_key(&rig_id)
    }

    /// `ExistsCamera`.
    pub fn exists_camera(&self, camera_id: CameraId) -> bool {
        self.cameras.contains_key(&camera_id)
    }

    /// `ExistsFrame`.
    pub fn exists_frame(&self, frame_id: FrameId) -> bool {
        self.frames.contains_key(&frame_id)
    }

    /// `ExistsImage`.
    pub fn exists_image(&self, image_id: ImageId) -> bool {
        self.images.contains_key(&image_id)
    }

    /// `ExistsPoint3D`.
    pub fn exists_point3d(&self, point3d_id: Point3DId) -> bool {
        self.points3d.contains_key(&point3d_id)
    }

    /// `RegImageIds`: the images of the registered frames, frame by frame in registration
    /// order. Fails if a registered frame refers to an image that was never added.
    pub fn reg_image_ids(&self) -> Result<Vec<ImageId>> {
        let mut reg_image_ids = Vec::new();
        for &frame_id in &self.reg_frame_ids {
            let frame = self.frame(frame_id)?;
            for data_id in frame.image_ids() {
                let image_id = image_id_of(data_id.id);
                crate::check!(
                    self.exists_image(image_id),
                    "The reconstruction object is broken as image {} in frame {} does not \
                     exist in the reconstruction. The most likely cause is missing \
                     AddImage(*) calls after adding frames.",
                    image_id,
                    frame.frame_id()
                );
                reg_image_ids.push(image_id);
            }
        }
        Ok(reg_image_ids)
    }

    /// `Point3DIds`, in ascending order.
    pub fn point3d_ids(&self) -> Vec<Point3DId> {
        self.points3d.keys().copied().collect()
    }

    /// `AddRig`: every camera sensor of the rig must already be in the reconstruction
    /// (add cameras before their rig), and the rig id must be new.
    pub fn add_rig(&mut self, rig: Rig) -> Result<()> {
        let check_exists_sensor = |sensor_id: SensorId| -> Result<()> {
            match sensor_id.sensor_type {
                SensorType::Camera => crate::check!(
                    self.exists_camera(sensor_id.id),
                    "Camera {} from rig {} not found in the reconstruction. Note that \
                     AddCamera should be called before AddRig.",
                    sensor_id.id,
                    rig.rig_id()
                ),
                SensorType::Imu | SensorType::Invalid => {}
            }
            Ok(())
        };
        check_exists_sensor(rig.ref_sensor_id())?;
        for &sensor_id in rig.non_ref_sensors().keys() {
            check_exists_sensor(sensor_id)?;
        }
        crate::check!(!self.rigs.contains_key(&rig.rig_id()));
        self.rigs.insert(rig.rig_id(), rig);
        Ok(())
    }

    /// `AddCamera`: the parameters must be valid for the model and the camera id new.
    pub fn add_camera(&mut self, camera: Camera) -> Result<()> {
        crate::check!(camera.verify_params()?);
        crate::check!(!self.cameras.contains_key(&camera.camera_id));
        self.cameras.insert(camera.camera_id, camera);
        Ok(())
    }

    /// `AddCameraWithTrivialRig`: adds the camera and a rig of the same id with the camera
    /// as its only (reference) sensor.
    pub fn add_camera_with_trivial_rig(&mut self, camera: Camera) -> Result<()> {
        crate::check!(
            !self.exists_rig(camera.camera_id),
            "AddCameraWithTrivialRig tried to add a rig with the same id as the camera, but \
             failed because Rig {}already exists in the reconstruction. ",
            camera.camera_id
        );
        let mut rig = Rig::new();
        rig.set_rig_id(camera.camera_id);
        rig.add_ref_sensor(camera.sensor_id())?;
        self.add_camera(camera)?;
        self.add_rig(rig)
    }

    /// `AddFrame`: the frame's rig must exist and have every camera the frame has data of.
    /// A frame with a pose is registered.
    pub fn add_frame(&mut self, mut frame: Frame) -> Result<()> {
        crate::check!(frame.has_rig_id());
        let rig = self.rig(frame.rig_id())?;
        for data_id in frame.data_ids() {
            match data_id.sensor_id.sensor_type {
                SensorType::Camera => crate::check!(rig.has_sensor(data_id.sensor_id)),
                // Note that we do not (yet) support IMU measurement data.
                SensorType::Imu => {}
                SensorType::Invalid => {
                    crate::fatal_throw!("Invalid sensor type: {}", data_id.sensor_id.sensor_type)
                }
            }
        }
        // COLMAP's SetRigPtr (or its pointer comparison); entry 170.
        frame.verify_rig(rig)?;
        let is_registered = frame.has_pose();
        let frame_id = frame.frame_id();
        crate::check!(!self.frames.contains_key(&frame_id));
        // We finalize the data ids, otherwise some internal bookkeeping (e.g., counting
        // reg_image_ids_) will be incorrect.
        frame.finalize_data_ids();
        self.frames.insert(frame_id, frame);
        if is_registered {
            crate::check_ne!(frame_id, INVALID_FRAME_ID);
            self.register_frame(frame_id)?;
        }
        Ok(())
    }

    /// `AddImage`: the image's camera and frame must exist and the frame must contain the
    /// image's data id; the image id must be new.
    pub fn add_image(&mut self, image: Image) -> Result<()> {
        crate::check!(image.has_camera_id());
        image.verify_camera(self.camera(image.camera_id())?)?;
        crate::check!(image.has_frame_id());
        let frame = self.frame(image.frame_id())?;
        crate::check!(frame.has_data_id(image.data_id()));
        image.verify_frame(frame)?;
        crate::check!(!self.images.contains_key(&image.image_id()));
        self.images.insert(image.image_id(), image);
        Ok(())
    }

    /// `AddImageWithTrivialFrame(image)`: adds the image with a frame of the same id that
    /// holds only it, in the trivial rig of its camera (`add_camera_with_trivial_rig`).
    pub fn add_image_with_trivial_frame(&mut self, mut image: Image) -> Result<()> {
        crate::check!(
            !self.exists_frame(image.image_id()),
            "AddImageWithTrivialFrame tried to add a frame with the same id as the image, \
             but failed because Frame {}already exists in the reconstruction.",
            image.image_id()
        );
        crate::check!(
            self.exists_rig(image.camera_id()),
            "Rig {} that contains Camera {} does not exist in the reconstruction.",
            image.camera_id(),
            image.camera_id()
        );
        let rig = self.rig(image.camera_id())?;
        crate::check_eq!(
            rig.num_sensors(),
            1,
            "AddImageWithTrivialFrame requires that the camera is from a rig that contains \
             exactly one sensor (the camera itself)."
        );
        crate::check!(rig.is_ref_sensor(self.camera(image.camera_id())?.sensor_id()));
        let mut frame = Frame::new();
        frame.set_frame_id(image.image_id());
        frame.set_rig_id(image.camera_id())?;
        frame.add_data_id(image.data_id())?;
        if image.has_frame_id() {
            crate::check_eq!(image.frame_id(), frame.frame_id());
        } else {
            image.set_frame_id(frame.frame_id());
        }
        self.add_frame(frame)?;
        self.add_image(image)
    }

    /// `AddImageWithTrivialFrame(image, cam_from_world)`: as above, then poses and
    /// registers the frame.
    pub fn add_image_with_trivial_frame_posed(
        &mut self,
        image: Image,
        cam_from_world: Rigid3d,
    ) -> Result<()> {
        let frame_id = image.image_id();
        self.add_image_with_trivial_frame(image)?;
        self.frame_mut(frame_id)?.set_rig_from_world(cam_from_world);
        self.register_frame(frame_id)
    }

    /// `AddPoint3D(point3D_id, point3D)`: adds the point under the given id and marks every
    /// track element's image point as observing it. Fails, leaving the reconstruction
    /// unchanged (entry 216), if the id is taken, a track image is missing or a track image
    /// point already observes another 3D point.
    pub fn add_point3d_with_id(&mut self, point3d_id: Point3DId, point3d: Point3D) -> Result<()> {
        crate::check!(!self.points3d.contains_key(&point3d_id));
        for track_el in point3d.track.elements() {
            let point2d = self.image(track_el.image_id)?.point2d(track_el.point2d_idx);
            if point2d.has_point3d() {
                crate::check_eq!(point2d.point3d_id, point3d_id);
            }
        }

        self.max_point3d_id = self.max_point3d_id.max(point3d_id);
        for track_el in point3d.track.elements() {
            let image = self.image_mut(track_el.image_id)?;
            let point2d = image.point2d(track_el.point2d_idx);
            if point2d.has_point3d() {
                crate::check_eq!(point2d.point3d_id, point3d_id);
            } else {
                image.set_point3d_for_point2d(track_el.point2d_idx, point3d_id)?;
            }
            crate::check_le!(image.num_points3d(), image.num_points2d());
        }
        self.points3d.insert(point3d_id, point3d);
        Ok(())
    }

    /// `AddPoint3D(xyz, track, color)`: adds a point under the next free id and returns it.
    pub fn add_point3d(
        &mut self,
        xyz: Vector3d,
        track: Track,
        color: Vector3ub,
    ) -> Result<Point3DId> {
        self.max_point3d_id += 1;
        let point3d_id = self.max_point3d_id;
        let point3d = Point3D {
            xyz,
            track,
            color,
            ..Point3D::default()
        };
        self.add_point3d_with_id(point3d_id, point3d)?;
        Ok(point3d_id)
    }

    /// `AddObservation`: adds an observation of an untriangulated image point to a 3D point.
    pub fn add_observation(&mut self, point3d_id: Point3DId, track_el: TrackElement) -> Result<()> {
        // Look the point up first so a missing point leaves the image unchanged (entry 216).
        if !self.exists_point3d(point3d_id) {
            return Err(does_not_exist("Point3D", point3d_id));
        }
        let image = self.image_mut(track_el.image_id)?;
        crate::check!(!image.point2d(track_el.point2d_idx).has_point3d());

        image.set_point3d_for_point2d(track_el.point2d_idx, point3d_id)?;
        crate::check_le!(image.num_points3d(), image.num_points2d());

        self.point3d_mut(point3d_id)?.track.add_element(track_el);
        Ok(())
    }

    /// `MergePoints3D`: replaces two 3D points by one at their track-length-weighted mean
    /// position and color, observed by both tracks; returns its new id.
    pub fn merge_points3d(
        &mut self,
        point3d_id1: Point3DId,
        point3d_id2: Point3DId,
    ) -> Result<Point3DId> {
        let point3d1 = self.point3d(point3d_id1)?;
        let point3d2 = self.point3d(point3d_id2)?;

        let length1 = point3d1.track.length() as f64;
        let length2 = point3d2.track.length() as f64;
        let sum = length1 + length2;
        let merged_xyz = Vector3d::new(
            (length1 * point3d1.xyz.x + length2 * point3d2.xyz.x) / sum,
            (length1 * point3d1.xyz.y + length2 * point3d2.xyz.y) / sum,
            (length1 * point3d1.xyz.z + length2 * point3d2.xyz.z) / sum,
        );
        let merged_channel = |c1: u8, c2: u8| -> u8 {
            // `cast<uint8_t>()` truncates; the weighted mean of two u8 stays in 0..=255.
            ((length1 * f64::from(c1) + length2 * f64::from(c2)) / sum) as u8
        };
        let merged_rgb = Vector3ub::new(
            merged_channel(point3d1.color.x, point3d2.color.x),
            merged_channel(point3d1.color.y, point3d2.color.y),
            merged_channel(point3d1.color.z, point3d2.color.z),
        );

        let mut merged_track = Track::new();
        merged_track.reserve(point3d1.track.length() + point3d2.track.length());
        merged_track.add_elements(point3d1.track.elements());
        merged_track.add_elements(point3d2.track.elements());

        self.delete_point3d(point3d_id1)?;
        self.delete_point3d(point3d_id2)?;

        self.add_point3d(merged_xyz, merged_track, merged_rgb)
    }

    /// `DeletePoint3D`: removes the point and resets every image point that observed it.
    pub fn delete_point3d(&mut self, point3d_id: Point3DId) -> Result<()> {
        // Note: Do not change order of these instructions.
        // Every track image is looked up before anything changes (entry 216).
        for track_el in self.point3d(point3d_id)?.track.elements() {
            self.image(track_el.image_id)?;
        }
        let point3d = self
            .points3d
            .remove(&point3d_id)
            .ok_or_else(|| does_not_exist("Point3D", point3d_id))?;
        for track_el in point3d.track.elements() {
            self.image_mut(track_el.image_id)?
                .reset_point3d_for_point2d(track_el.point2d_idx);
        }
        Ok(())
    }

    /// `DeleteObservation`: removes an image point's observation of its 3D point; a point
    /// left with fewer than two observations is deleted.
    pub fn delete_observation(&mut self, image_id: ImageId, point2d_idx: Point2DIdx) -> Result<()> {
        // Note: Do not change order of these instructions.
        let point3d_id = self.image(image_id)?.point2d(point2d_idx).point3d_id;
        let point3d = self.point3d_mut(point3d_id)?;

        if point3d.track.length() <= 2 {
            return self.delete_point3d(point3d_id);
        }

        point3d.track.delete_element_ids(image_id, point2d_idx);

        self.image_mut(image_id)?
            .reset_point3d_for_point2d(point2d_idx);
        Ok(())
    }

    /// `DeleteAllPoints2DAndPoints3D`: removes every 3D point and every image's points,
    /// leaving each image's triangulated count at 0 (COLMAP leaves it stale; entry 215).
    pub fn delete_all_points2d_and_points3d(&mut self) -> Result<()> {
        self.points3d.clear();
        for image in self.images.values_mut() {
            image.points2d_mut().clear();
            // COLMAP's `SetPoints2D(std::vector<Eigen::Vector2d>(0))` keeps `num_points3D_`;
            // `set_points2d` of no points recomputes it (to 0).
            image.set_points2d(Vec::new())?;
        }
        Ok(())
    }

    /// `SetRigsAndFrames`: replaces all rigs and frames (registering the posed frames) and
    /// moves every image to the frame that now holds it. Every image must be in exactly
    /// one frame.
    pub fn set_rigs_and_frames(&mut self, rigs: Vec<Rig>, frames: Vec<Frame>) -> Result<()> {
        self.rigs.clear();
        for rig in rigs {
            self.add_rig(rig)?;
        }

        self.frames.clear();
        self.reg_frame_ids.clear();
        self.num_reg_images = 0;
        // Lookups only, never iterated.
        let mut image_to_frame_ids: HashMap<ImageId, FrameId> = HashMap::new();
        for frame in frames {
            for data_id in frame.image_ids() {
                crate::check!(image_to_frame_ids
                    .insert(image_id_of(data_id.id), frame.frame_id())
                    .is_none());
            }
            self.add_frame(frame)?;
        }

        for (&image_id, image) in &mut self.images {
            let frame_id = *image_to_frame_ids
                .get(&image_id)
                .ok_or_else(|| does_not_exist("Frame for image", image_id))?;
            image.set_frame_id(frame_id);
            let frame = self
                .frames
                .get(&frame_id)
                .ok_or_else(|| does_not_exist("Frame", frame_id))?;
            image.verify_frame(frame)?;
        }
        Ok(())
    }

    /// `RegisterFrame`: marks a posed frame as registered (a no-op if it already is).
    pub fn register_frame(&mut self, frame_id: FrameId) -> Result<()> {
        let frame = self.frame(frame_id)?;
        crate::check!(frame.has_pose());
        if !self.reg_frame_ids.contains(&frame_id) {
            self.num_reg_images += frame.image_ids().count();
            self.reg_frame_ids.push(frame_id);
        }
        Ok(())
    }

    /// `DeRegisterFrame`: deletes the observations of the frame's images, resets its pose and
    /// unregisters it. A frame that is not registered is ignored (COLMAP warns; entry 211's
    /// reasoning, the core has no logger).
    pub fn deregister_frame(&mut self, frame_id: FrameId) -> Result<()> {
        if !self.reg_frame_ids.contains(&frame_id) {
            return Ok(());
        }

        let image_ids: Vec<ImageId> = self
            .frame(frame_id)?
            .image_ids()
            .map(|data_id| image_id_of(data_id.id))
            .collect();
        for image_id in image_ids {
            let num_points2d = self.image(image_id)?.num_points2d();
            for point2d_idx in 0..num_points2d {
                if self.image(image_id)?.point2d(point2d_idx).has_point3d() {
                    self.delete_observation(image_id, point2d_idx)?;
                }
            }
            self.num_reg_images -= 1;
        }

        self.frame_mut(frame_id)?.reset_pose();
        self.reg_frame_ids.retain(|&id| id != frame_id);
        Ok(())
    }
}
