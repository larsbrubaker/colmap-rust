//! Port of `colmap/scene/rig.h` and the `ReadRigConfig` half of `rig.cc`: [`RigConfig`], a
//! rig configuration that groups images into multi-camera rigs by image name prefix, and
//! [`read_rig_config`]. The `ApplyRigConfig` half, which writes the configured rigs and frames
//! into a [`crate::scene::database::Database`] (and optionally a reconstruction), is
//! `scene/rig_apply.rs`; the rig itself is [`crate::sensor::rig::Rig`]. Port of colmap-sharp's
//! `ColmapSharp/Scene/RigConfig.cs`. Tests: `tests/scene/rig.rs` (`rig_test.cc` 1:1) and
//! `tests/scene/rust_only_rig.rs`.
//!
//! COLMAP reads the file with `boost::property_tree::read_json`, which keeps every value as
//! text and converts on access. Where COLMAP takes a file path, this takes the file's text
//! (the host reads files, natively and in the browser). The JSON goes through the core's own
//! reader ([`crate::util::json`]), which mirrors the property tree:
//! - a node's children are an array's elements or an object's values, and a scalar has none;
//! - a lookup by key finds the first member with that key (ptree `get_child`);
//! - a scalar reads as its text (a number's source text, "true"/"false", a string's value), a
//!   container as "" (ptree's empty data), and numbers and booleans parse from that text as
//!   the ptree stream translator does ([`crate::util::ptree_translate`]), so "1.5" and 1.5
//!   are both a double.
//!
//! Strict JSON (no trailing commas, no comments) matches Boost's parser, which rejects both.
//! The rotation and translation arrays must have exactly 4 and 3 entries (the divergence in
//! `read_doubles`' doc).

use crate::geometry::rigid3::Rigid3d;
use crate::linalg::{Quaterniond, Vector3d};
use crate::scene::camera::Camera;
use crate::sensor::models::camera_model_name_to_id;
use crate::util::check::{ColmapError, ErrorKind};
use crate::util::json::{parse_json, JsonValue};
use crate::util::ptree_translate::{ptree_get_bool, ptree_get_double};
use crate::{check, Result};

/// Port of `colmap::RigConfig`: one rig of a rig configuration. For each rig, the
/// configuration specifies a list of cameras with exactly one camera specified as the
/// reference sensor as well as all cameras with a specified image prefix. All images with the
/// given name prefix will be associated with the camera. In addition, each camera may specify
/// an optional known pose in the rig - except for the reference camera whose pose is defined
/// as identity. The rotation is expected in the order [w, x, y, z]. Furthermore, each camera
/// may specify a custom camera model and parameters.
///
/// Example for eth3d/delivery_area:
/// ```json
/// [
///   {
///     "cameras": [
///       { "image_prefix": "images_rig_cam4_undistorted/", "ref_sensor": true },
///       { "image_prefix": "images_rig_cam5_undistorted/" },
///       { "image_prefix": "images_rig_cam6_undistorted/" },
///       { "image_prefix": "images_rig_cam7_undistorted/" }
///     ]
///   }
/// ]
/// ```
///
/// Example for GoPro cubemaps:
/// ```json
/// [
///   {
///     "cameras": [
///       { "image_prefix": "0/", "ref_sensor": true },
///       {
///         "image_prefix": "1/",
///         "cam_from_rig_rotation": [0.7071067811865475, 0.0, 0.7071067811865476, 0.0],
///         "cam_from_rig_translation": [0, 0, 0]
///       }
///     ]
///   }
/// ]
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RigConfig {
    /// The cameras of the rig.
    pub cameras: Vec<RigCamera>,
}

/// Port of `colmap::RigConfig::RigCamera`: one camera of a rig configuration.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RigCamera {
    /// Whether this camera is the rig's reference sensor.
    pub ref_sensor: bool,
    /// All images whose name starts with this prefix belong to this camera.
    pub image_prefix: String,
    /// The known `cam_from_rig` pose, or `None` to derive it (or leave it unknown).
    pub cam_from_rig: Option<Rigid3d>,
    /// Custom intrinsics (model, parameters, prior focal length flag) to apply to the camera,
    /// or `None` to keep the database's.
    pub camera: Option<Camera>,
}

/// Port of `colmap::ReadRigConfig`, on the text of the `.json` file (see [`RigConfig`] for the
/// format). Errors on invalid JSON, a rig without "cameras", a camera without
/// "image_prefix", a value that does not convert, or a rig without exactly one reference
/// sensor.
pub fn read_rig_config(json: &str) -> Result<Vec<RigConfig>> {
    let pt = parse_json(json)?;

    let mut configs = Vec::new();
    for rig_node in pt.children() {
        let mut config = RigConfig::default();
        let mut has_ref_sensor = false;
        for camera in get_child(rig_node, "cameras")?.children() {
            let mut config_camera = RigCamera {
                image_prefix: get_child(camera, "image_prefix")?.ptree_data().to_string(),
                ..RigCamera::default()
            };

            let cam_from_rig_rotation_node = camera.get("cam_from_rig_rotation");
            let cam_from_rig_translation_node = camera.get("cam_from_rig_translation");
            if let (Some(rotation_node), Some(translation_node)) =
                (cam_from_rig_rotation_node, cam_from_rig_translation_node)
            {
                let wxyz = read_doubles(rotation_node, 4, "cam_from_rig_rotation")?;
                let translation = read_doubles(translation_node, 3, "cam_from_rig_translation")?;
                config_camera.cam_from_rig = Some(Rigid3d::new(
                    Quaterniond::new(wxyz[0], wxyz[1], wxyz[2], wxyz[3]),
                    Vector3d::new(translation[0], translation[1], translation[2]),
                ));
            }

            if let Some(ref_sensor_node) = camera.get("ref_sensor") {
                if ptree_get_bool(ref_sensor_node.ptree_data())? {
                    check!(
                        cam_from_rig_rotation_node.is_none()
                            && cam_from_rig_translation_node.is_none(),
                        "Reference sensor must not have cam_from_rig"
                    );
                    check!(!has_ref_sensor, "Rig must only have one reference sensor");
                    config_camera.ref_sensor = true;
                    has_ref_sensor = true;
                }
            }

            let camera_model_name_node = camera.get("camera_model_name");
            let camera_params_node = camera.get("camera_params");
            if let (Some(model_name_node), Some(params_node)) =
                (camera_model_name_node, camera_params_node)
            {
                let mut params = Vec::new();
                for node in params_node.children() {
                    params.push(ptree_get_double(node.ptree_data())?);
                }
                config_camera.camera = Some(Camera {
                    model_id: camera_model_name_to_id(model_name_node.ptree_data()),
                    has_prior_focal_length: true,
                    params,
                    ..Camera::default()
                });
            }

            config.cameras.push(config_camera);
        }

        check!(has_ref_sensor, "Rig must have one reference sensor");
        configs.push(config);
    }

    Ok(configs)
}

/// `ptree::get_child(key)`: the first member with the key, or Boost's `ptree_bad_path`.
fn get_child<'a>(node: &'a JsonValue, key: &str) -> Result<&'a JsonValue> {
    node.get(key)
        .ok_or_else(|| ColmapError::new(ErrorKind::RuntimeError, format!("No such node ({key})")))
}

/// The node's children as doubles. COLMAP writes them into a fixed-size Eigen vector by
/// running index without a size check (undefined behavior for any other count); here the
/// count must be exactly `count`.
///
/// That is a deliberate divergence. COLMAP's `ReadRigConfig` writes the entries of
/// `cam_from_rig_rotation` and `cam_from_rig_translation` into an `Eigen::Vector4d` and
/// `Rigid3d::translation()` by running index; [`read_rig_config`] returns a "Check failed"
/// error ("... must have 4 entries" / "... must have 3 entries") unless there are exactly 4 and
/// 3 entries. Fewer entries leave COLMAP with uninitialized coefficients and more entries write
/// out of bounds; both are undefined behavior with no result to match. Every well-formed config
/// reads the same: `tests/scene/rig.rs` (`rig_test.cc` 1:1) passes, including
/// `read_rig_config_nominal`, which reads a 4-entry rotation and a 3-entry translation, and
/// `rust_only_read_rig_config_pose_needs_four_and_three_entries`
/// (`tests/scene/rust_only_rig.rs`) pins the check. Same as colmap-sharp entry 80.
fn read_doubles(node: &JsonValue, count: usize, key: &str) -> Result<Vec<f64>> {
    let mut values = Vec::new();
    for child in node.children() {
        values.push(ptree_get_double(child.ptree_data())?);
    }
    check!(values.len() == count, "{key} must have {count} entries");
    Ok(values)
}
