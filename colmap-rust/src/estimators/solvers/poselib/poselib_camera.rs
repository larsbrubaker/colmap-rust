//! Port of the camera description of PoseLib's `PoseLib/misc/camera_models.h` and `.cc`
//! (`struct Camera`: model id, size, parameters; `id_from_string`, `name_from_id`,
//! `init_params`, `focal`, and each model's name, id, `num_params`, `focal_idx` and
//! `principal_point_idx`; BSD-3-Clause, Copyright (c) 2020 Viktor Larsson; see
//! THIRD_PARTY_NOTICES.md; commit fa7280fee27f97aff31ae7f98bab7f583fac7d08), through
//! colmap-sharp's `Estimators/Solvers/PoseLib/PoseLibCamera.cs`. COLMAP's
//! `estimators/solvers/poselib_utils` converts its `Camera` to and from it.
//!
//! PoseLib's projection and distortion functions (project/unproject and their Jacobians) are
//! not ported: COLMAP only ever converts cameras to PoseLib and back (`poselib_utils.cc`), and
//! every solver COLMAP calls works on bearings or normalized points. Port them with the
//! PoseLib code that first needs them. Tests: `tests/estimators/poselib_utils.rs`.

/// One entry of PoseLib's `SWITCH_CAMERA_MODELS`.
struct ModelInfo {
    name: &'static str,
    id: i32,
    num_params: usize,
    focal_idx: &'static [usize],
    principal_point_idx: &'static [usize],
}

const fn model(
    name: &'static str,
    id: i32,
    num_params: usize,
    focal_idx: &'static [usize],
    principal_point_idx: &'static [usize],
) -> ModelInfo {
    ModelInfo {
        name,
        id,
        num_params,
        focal_idx,
        principal_point_idx,
    }
}

/// `SWITCH_CAMERA_MODELS`, in PoseLib's order (the name lookup returns the first match).
const MODELS: [ModelInfo; 17] = [
    model("NULL", -1, 0, &[], &[]),
    model("SIMPLE_PINHOLE", 0, 3, &[0], &[1, 2]),
    model("PINHOLE", 1, 4, &[0, 1], &[2, 3]),
    model("SIMPLE_RADIAL", 2, 4, &[0], &[1, 2]),
    model("RADIAL", 3, 5, &[0], &[1, 2]),
    model("OPENCV", 4, 8, &[0, 1], &[2, 3]),
    model("FULL_OPENCV", 6, 12, &[0, 1], &[2, 3]),
    model("OPENCV_FISHEYE", 5, 8, &[0, 1], &[2, 3]),
    model("FOV", 7, 5, &[0, 1], &[2, 3]),
    model("SIMPLE_RADIAL_FISHEYE", 8, 4, &[0], &[1, 2]),
    model("RADIAL_FISHEYE", 9, 5, &[0], &[1, 2]),
    model("THIN_PRISM_FISHEYE", 10, 12, &[0, 1], &[2, 3]),
    model("RAD_TAN_THIN_PRISM_FISHEYE", 11, 16, &[0, 1], &[2, 3]),
    model("1D_RADIAL", 99, 2, &[], &[0, 1]),
    model("SPHERICAL", 100, 2, &[], &[]),
    model("DIVISION", 101, 5, &[0, 1], &[2, 3]),
    model("SIMPLE_DIVISION", 102, 4, &[0], &[1, 2]),
];

/// PoseLib's `CameraModelId::INVALID`.
pub const INVALID_MODEL_ID: i32 = -1;

/// A PoseLib camera: model id, image size and model parameters. Port of the data members,
/// constructors and name lookups of `poselib::Camera`. The size is `int` in PoseLib, kept as
/// `i32`.
#[derive(Clone, Debug, PartialEq)]
pub struct PoseLibCamera {
    /// PoseLib's model id (`poselib::CameraModelId`; -1 is INVALID).
    pub model_id: i32,
    /// Image width in pixels.
    pub width: i32,
    /// Image height in pixels.
    pub height: i32,
    /// The model parameters.
    pub params: Vec<f64>,
}

impl PoseLibCamera {
    /// `Camera(model_name, params, width, height)`: looks the model up by name (INVALID = -1
    /// when unknown) and runs `init_params`.
    pub fn from_name(model_name: &str, params: &[f64], width: i32, height: i32) -> Self {
        Self::new(id_from_string(model_name), params, width, height)
    }

    /// `Camera(model_id, params, width, height)`, followed by `init_params`.
    pub fn new(model_id: i32, params: &[f64], width: i32, height: i32) -> Self {
        let mut camera = Self {
            model_id,
            width,
            height,
            params: params.to_vec(),
        };
        camera.init_params();
        camera
    }

    /// The model name, or "INVALID_MODEL" for an unknown id (`model_name()`).
    pub fn model_name(&self) -> &'static str {
        name_from_id(self.model_id)
    }

    /// The mean of the model's focal parameters: 1 for an empty (identity) camera, and also 1
    /// for a model without focal parameters or an unknown id. Port of `Camera::focal`.
    pub fn focal(&self) -> f64 {
        if self.params.is_empty() {
            return 1.0; // empty camera assumed to be identity
        }

        let mut focal = 1.0;
        if let Some(model) = find(self.model_id).filter(|m| !m.focal_idx.is_empty()) {
            focal = 0.0;
            for &idx in model.focal_idx {
                focal += self.params[idx] / model.focal_idx.len() as f64;
            }
        }

        focal
    }

    /// Port of `Camera::init_params`: resizes the parameters to the model's count (padding
    /// with zeros) and, only when none were given, sets the focal length to
    /// `1.2 * max(width, height)` and the principal point to the image center (or 1 and 0
    /// without a size).
    fn init_params(&mut self) {
        let init_params = self.params.is_empty();
        let Some(model) = find(self.model_id) else {
            return;
        };

        self.params.resize(model.num_params, 0.0);

        if !init_params {
            return;
        }

        let (mut focal, mut cx, mut cy) = (1.0, 0.0, 0.0);
        if self.width > 0 && self.height > 0 {
            focal = f64::from(self.width.max(self.height)) * 1.2;
            cx = f64::from(self.width) / 2.0;
            cy = f64::from(self.height) / 2.0;
        }

        for &k in model.focal_idx {
            self.params[k] = focal;
        }

        if let [px, py] = model.principal_point_idx {
            self.params[*px] = cx;
            self.params[*py] = cy;
        }
    }
}

/// Port of `Camera::id_from_string`.
pub fn id_from_string(model_name: &str) -> i32 {
    MODELS
        .iter()
        .find(|m| m.name == model_name)
        .map_or(INVALID_MODEL_ID, |m| m.id)
}

/// Port of `Camera::name_from_id`.
pub fn name_from_id(model_id: i32) -> &'static str {
    find(model_id).map_or("INVALID_MODEL", |m| m.name)
}

fn find(model_id: i32) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.id == model_id)
}
