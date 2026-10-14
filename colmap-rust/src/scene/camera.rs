//! Port of `colmap/scene/camera.h` / `camera.cc`: [`Camera`], a camera model id with its
//! parameters and image size, and the per-camera wrappers around the model dispatch in
//! [`crate::sensor::models`]. Tests: `tests/scene/camera.rs` (`camera_test.cc` 1:1).
//!
//! Tier A: every number comes from the camera models (Tier A themselves) or plain
//! arithmetic in COLMAP's order.
//!
//! Errors follow docs/CPP_DIVERGENCES.md entry 101. The methods that go through the model
//! metadata (parameter groups, `ParamsInfo`, `VerifyParams`, `HasBogusParams`, the focal
//! length and principal point accessors, `CalibrationMatrix`, `MeanFocalLength`,
//! `IsUndistorted`, `Rescale(scale)`) return `Err(DomainError, "Camera model does not
//! exist")` for an unknown model, and their `THROW_CHECK`s return `Err` too. The per-point
//! methods (`CamFromImg`, `CamRayFromImg`, `CamFromImgThreshold`, `ImgFromCam`,
//! `ImgFromCamWithJac`, `CamRayFromImgWithJac`, the `Is*` model-kind queries and
//! `Rescale(width, height)`) panic with that message instead: a camera must be checked with
//! [`Camera::verify_params`] (or built through [`Camera::create_from_model_id`]) before it is
//! used per point. A `params` vector shorter than the model's is out-of-bounds indexing in
//! COLMAP (undefined behavior); here it panics.

use std::fmt;

use crate::geometry::pose::CamRayWithJac;
use crate::linalg::{Matrix2x3d, Matrix3d, Vector2d, Vector3d};
use crate::sensor::models::{
    cam_ray_from_img_jacobian, camera_model_cam_from_img, camera_model_cam_from_img_threshold,
    camera_model_cam_ray_from_img, camera_model_extra_params_idxs, camera_model_focal_length_idxs,
    camera_model_has_bogus_params, camera_model_id_to_name, camera_model_img_from_cam,
    camera_model_img_from_cam_with_jac, camera_model_initialize_params,
    camera_model_is_perspective, camera_model_is_perspective_fisheye,
    camera_model_is_perspective_pinhole, camera_model_is_spherical,
    camera_model_meta_data_params_idxs, camera_model_name_to_id, camera_model_params_info,
    camera_model_principal_point_idxs, camera_model_rescale, camera_model_verify_params,
    exists_camera_model_with_id, CameraModelId,
};
use crate::util::check::Result;
use crate::util::misc::{csv_to_vector, vector_to_csv};
use crate::util::types::{CameraId, SensorId, SensorType, INVALID_CAMERA_ID};
use crate::{check, check_eq, check_gt};

/// Port of `colmap::Camera`: camera class that holds the intrinsic parameters. Cameras may
/// be shared between multiple images, e.g., if the same "physical" camera took multiple
/// pictures with the exact same lens and intrinsics (focal length, etc.). This class has a
/// specific distortion model defined by a camera model class.
#[derive(Debug, Clone, PartialEq)]
pub struct Camera {
    /// The unique identifier of the camera.
    pub camera_id: CameraId,
    /// The identifier of the camera model.
    pub model_id: CameraModelId,
    /// The image width in pixels.
    pub width: usize,
    /// The image height in pixels.
    pub height: usize,
    /// The focal length, principal point, and extra parameters. If the camera model is not
    /// specified, this vector is empty.
    pub params: Vec<f64>,
    /// Whether there is a safe prior for the focal length, e.g. manually provided or
    /// extracted from EXIF.
    pub has_prior_focal_length: bool,
}

impl Default for Camera {
    fn default() -> Self {
        Self {
            camera_id: INVALID_CAMERA_ID,
            model_id: CameraModelId::Invalid,
            width: 0,
            height: 0,
            params: Vec::new(),
            has_prior_focal_length: false,
        }
    }
}

impl Camera {
    /// `CreateFromModelId`: a camera with the model's initial parameters for this focal
    /// length and image size. An unknown model is an error.
    pub fn create_from_model_id(
        camera_id: CameraId,
        model_id: CameraModelId,
        focal_length: f64,
        width: usize,
        height: usize,
    ) -> Result<Camera> {
        check!(exists_camera_model_with_id(model_id));
        Ok(Camera {
            camera_id,
            model_id,
            width,
            height,
            params: camera_model_initialize_params(model_id, focal_length, width, height)?,
            has_prior_focal_length: false,
        })
    }

    /// `CreateFromModelName`.
    pub fn create_from_model_name(
        camera_id: CameraId,
        model_name: &str,
        focal_length: f64,
        width: usize,
        height: usize,
    ) -> Result<Camera> {
        Self::create_from_model_id(
            camera_id,
            camera_model_name_to_id(model_name),
            focal_length,
            width,
            height,
        )
    }

    /// `ModelName`: the model's name, "" for an unknown model.
    pub fn model_name(&self) -> &'static str {
        camera_model_id_to_name(self.model_id)
    }

    /// `SensorId`: `(CAMERA, camera_id)`.
    pub fn sensor_id(&self) -> SensorId {
        SensorId::new(SensorType::Camera, self.camera_id)
    }

    /// `MeanFocalLength`: the mean of the focal length parameters; 0 for the spherical
    /// EQUIRECTANGULAR model, which has no focal length.
    pub fn mean_focal_length(&self) -> Result<f64> {
        let focal_length_idxs = self.focal_length_idxs()?;
        if self.is_spherical() {
            return Ok(0.0);
        }
        let mut focal_length = 0.0;
        for &idx in focal_length_idxs {
            focal_length += self.params[idx];
        }
        Ok(focal_length / focal_length_idxs.len() as f64)
    }

    /// `FocalLength`: the single focal length; an error for models with separate fx, fy.
    pub fn focal_length(&self) -> Result<f64> {
        let idxs = self.focal_length_idxs()?;
        check_eq!(idxs.len(), 1);
        Ok(self.params[idxs[0]])
    }

    /// `FocalLengthX`.
    pub fn focal_length_x(&self) -> Result<f64> {
        let idxs = self.focal_length_idxs()?;
        Ok(self.params[idxs[0]])
    }

    /// `FocalLengthY`: fy, or the single focal length.
    pub fn focal_length_y(&self) -> Result<f64> {
        let idxs = self.focal_length_idxs()?;
        Ok(self.params[idxs[if idxs.len() == 1 { 0 } else { 1 }]])
    }

    /// `SetFocalLength`: sets every focal length parameter to `f`.
    pub fn set_focal_length(&mut self, f: f64) -> Result<()> {
        for &idx in self.focal_length_idxs()? {
            self.params[idx] = f;
        }
        Ok(())
    }

    /// `SetFocalLengthX`: only for models with separate fx, fy.
    pub fn set_focal_length_x(&mut self, fx: f64) -> Result<()> {
        let idxs = self.focal_length_idxs()?;
        check_eq!(idxs.len(), 2);
        self.params[idxs[0]] = fx;
        Ok(())
    }

    /// `SetFocalLengthY`: only for models with separate fx, fy.
    pub fn set_focal_length_y(&mut self, fy: f64) -> Result<()> {
        let idxs = self.focal_length_idxs()?;
        check_eq!(idxs.len(), 2);
        self.params[idxs[1]] = fy;
        Ok(())
    }

    /// `PrincipalPointX`.
    pub fn principal_point_x(&self) -> Result<f64> {
        let idxs = self.principal_point_idxs()?;
        check_eq!(idxs.len(), 2);
        Ok(self.params[idxs[0]])
    }

    /// `PrincipalPointY`.
    pub fn principal_point_y(&self) -> Result<f64> {
        let idxs = self.principal_point_idxs()?;
        check_eq!(idxs.len(), 2);
        Ok(self.params[idxs[1]])
    }

    /// `PrincipalPoint`: `(cx, cy)`.
    pub fn principal_point(&self) -> Result<Vector2d> {
        Ok(Vector2d::new(
            self.principal_point_x()?,
            self.principal_point_y()?,
        ))
    }

    /// `SetPrincipalPointX`.
    pub fn set_principal_point_x(&mut self, cx: f64) -> Result<()> {
        let idxs = self.principal_point_idxs()?;
        check_eq!(idxs.len(), 2);
        self.params[idxs[0]] = cx;
        Ok(())
    }

    /// `SetPrincipalPointY`.
    pub fn set_principal_point_y(&mut self, cy: f64) -> Result<()> {
        let idxs = self.principal_point_idxs()?;
        check_eq!(idxs.len(), 2);
        self.params[idxs[1]] = cy;
        Ok(())
    }

    /// `FocalLengthIdxs`: indices of the focal length parameters in `params`.
    pub fn focal_length_idxs(&self) -> Result<&'static [usize]> {
        camera_model_focal_length_idxs(self.model_id)
    }

    /// `PrincipalPointIdxs`.
    pub fn principal_point_idxs(&self) -> Result<&'static [usize]> {
        camera_model_principal_point_idxs(self.model_id)
    }

    /// `ExtraParamsIdxs`: the distortion parameters.
    pub fn extra_params_idxs(&self) -> Result<&'static [usize]> {
        camera_model_extra_params_idxs(self.model_id)
    }

    /// `MetaDataParamsIdxs`: parameters that are neither intrinsics nor distortion (the
    /// EQUIRECTANGULAR model's image size).
    pub fn meta_data_params_idxs(&self) -> Result<&'static [usize]> {
        camera_model_meta_data_params_idxs(self.model_id)
    }

    /// `CalibrationMatrix`: the 3x3 calibration matrix K. Only defined for perspective
    /// cameras.
    pub fn calibration_matrix(&self) -> Result<Matrix3d> {
        // The existence check comes first so an unknown model is the DomainError of
        // entry 101, as COLMAP's IsPerspective() throws it, rather than a panic.
        self.focal_length_idxs()?;
        check!(
            self.is_perspective(),
            "CalibrationMatrix() only defined for perspective cameras."
        );
        let mut k = Matrix3d::identity();
        k[(0, 0)] = self.focal_length_x()?;
        k[(1, 1)] = self.focal_length_y()?;
        k[(0, 2)] = self.principal_point_x()?;
        k[(1, 2)] = self.principal_point_y()?;
        Ok(k)
    }

    /// `ParamsInfo`: the model's human-readable parameter list, e.g. "f, cx, cy".
    pub fn params_info(&self) -> Result<&'static str> {
        camera_model_params_info(self.model_id)
    }

    /// `ParamsToString`: the parameters as comma-separated values.
    pub fn params_to_string(&self) -> String {
        vector_to_csv(&self.params)
    }

    /// `SetParamsFromString`: parses comma-separated values and keeps them only if they fit
    /// the model; `Ok(false)` (parameters unchanged) otherwise.
    pub fn set_params_from_string(&mut self, string: &str) -> Result<bool> {
        let new_camera_params: Vec<f64> = csv_to_vector(string)?;
        if !camera_model_verify_params(self.model_id, &new_camera_params)? {
            return Ok(false);
        }
        self.params = new_camera_params;
        Ok(true)
    }

    /// `VerifyParams`: whether the number of parameters fits the model. An unknown model is
    /// `Err(DomainError)` (docs/CPP_DIVERGENCES.md entry 101).
    pub fn verify_params(&self) -> Result<bool> {
        camera_model_verify_params(self.model_id, &self.params)
    }

    /// `IsUndistorted`: whether every extra (distortion) parameter is within 1e-8 of zero.
    /// Non-perspective cameras (e.g. EQUIRECTANGULAR) have no pinhole image plane to
    /// undistort to; they count as already undistorted so undistortion is a no-op.
    pub fn is_undistorted(&self) -> Result<bool> {
        let extra_params_idxs = self.extra_params_idxs()?;
        if self.is_spherical() {
            return Ok(true);
        }
        for &idx in extra_params_idxs {
            if self.params[idx].abs() > 1e-8 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// `IsPerspective`. Panics for an unknown model.
    pub fn is_perspective(&self) -> bool {
        camera_model_is_perspective(self.model_id)
    }

    /// `IsSpherical`. Panics for an unknown model.
    pub fn is_spherical(&self) -> bool {
        camera_model_is_spherical(self.model_id)
    }

    /// `IsPerspectiveFisheye` (false for an unknown model, as in COLMAP).
    pub fn is_perspective_fisheye(&self) -> bool {
        camera_model_is_perspective_fisheye(self.model_id)
    }

    /// `IsPerspectivePinhole`. Panics for an unknown model.
    pub fn is_perspective_pinhole(&self) -> bool {
        camera_model_is_perspective_pinhole(self.model_id)
    }

    /// `HasBogusParams`: whether the focal length ratio or an extra parameter is out of the
    /// given bounds.
    pub fn has_bogus_params(
        &self,
        min_focal_length_ratio: f64,
        max_focal_length_ratio: f64,
        max_extra_param: f64,
    ) -> Result<bool> {
        camera_model_has_bogus_params(
            self.model_id,
            &self.params,
            self.width,
            self.height,
            min_focal_length_ratio,
            max_focal_length_ratio,
            max_extra_param,
        )
    }

    /// `CamFromImg`: project a point from image to camera space (normalized image plane).
    pub fn cam_from_img(&self, image_point: Vector2d) -> Option<Vector2d> {
        camera_model_cam_from_img(self.model_id, &self.params, image_point)
    }

    /// `CamRayFromImg`: the unit bearing of an image point; also defined for spherical
    /// models, where [`Camera::cam_from_img`] fails on the back hemisphere.
    pub fn cam_ray_from_img(&self, image_point: Vector2d) -> Option<Vector3d> {
        camera_model_cam_ray_from_img(self.model_id, &self.params, image_point)
    }

    /// `CamFromImgThreshold`: convert a pixel threshold to the normalized image plane.
    pub fn cam_from_img_threshold(&self, threshold: f64) -> f64 {
        camera_model_cam_from_img_threshold(self.model_id, &self.params, threshold)
    }

    /// `ImgFromCam`: project a camera-space point to the image. With `check_cheirality`
    /// (COLMAP's default `true`) points behind the camera have no projection.
    pub fn img_from_cam(&self, cam_point: Vector3d, check_cheirality: bool) -> Option<Vector2d> {
        camera_model_img_from_cam(self.model_id, &self.params, cam_point, check_cheirality)
    }

    /// `ImgFromCamWithJac`: as [`Camera::img_from_cam`], also writing the 2x3 Jacobian of
    /// the projection with respect to `cam_point` into `j_uvw` when given.
    pub fn img_from_cam_with_jac(
        &self,
        cam_point: Vector3d,
        j_uvw: Option<&mut Matrix2x3d>,
        check_cheirality: bool,
    ) -> Option<Vector2d> {
        camera_model_img_from_cam_with_jac(
            self.model_id,
            &self.params,
            cam_point,
            j_uvw,
            check_cheirality,
        )
    }

    /// `CamRayFromImgWithJac`: the unit bearing of an image point and its 3x2 Jacobian with
    /// respect to the pixel; `None` when the pixel cannot be unprojected or the projection
    /// Jacobian is degenerate there.
    pub fn cam_ray_from_img_with_jac(&self, image_point: Vector2d) -> Option<CamRayWithJac> {
        let cam_ray = self.cam_ray_from_img(image_point)?;
        let mut j_uvw = Matrix2x3d::zeros();
        self.img_from_cam_with_jac(cam_ray, Some(&mut j_uvw), true)?;
        let j_ray = cam_ray_from_img_jacobian(cam_ray, j_uvw)?;
        Some(CamRayWithJac {
            ray: cam_ray,
            jacobian: j_ray,
        })
    }

    /// `Rescale(scale)`: rescale the image size and the camera parameters by `scale`. The
    /// image size is rounded (half away from zero, `std::round`) and the parameters are
    /// scaled by the ratio actually achieved per axis.
    pub fn rescale(&mut self, scale: f64) -> Result<()> {
        // Validated before the check so an unknown model is entry 101's DomainError.
        self.focal_length_idxs()?;
        check_gt!(scale, 0.0);
        let scale_x = (scale * self.width as f64).round() / self.width as f64;
        let scale_y = (scale * self.height as f64).round() / self.height as f64;
        // `static_cast<size_t>` of a non-negative rounded double; `as` saturates where C++
        // would be undefined (beyond usize).
        self.width = (scale * self.width as f64).round() as usize;
        self.height = (scale * self.height as f64).round() as usize;
        camera_model_rescale(self.model_id, scale_x, scale_y, &mut self.params);
        Ok(())
    }

    /// `Rescale(new_width, new_height)`: rescale to an explicit image size. Panics for an
    /// unknown model.
    pub fn rescale_to(&mut self, new_width: usize, new_height: usize) {
        let scale_x = new_width as f64 / self.width as f64;
        let scale_y = new_height as f64 / self.height as f64;
        self.width = new_width;
        self.height = new_height;
        camera_model_rescale(self.model_id, scale_x, scale_y, &mut self.params);
    }
}

impl fmt::Display for Camera {
    /// Port of `operator<<(std::ostream&, const Camera&)`: `Camera(camera_id=<id|Invalid>,
    /// model=<name|Invalid>, width=w, height=h, params=[..] (<info|?>))`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let valid_model = exists_camera_model_with_id(self.model_id);
        let camera_id = if self.camera_id != INVALID_CAMERA_ID {
            self.camera_id.to_string()
        } else {
            "Invalid".to_string()
        };
        let params_info = self.params_info().unwrap_or("?");
        let model_name = if valid_model {
            self.model_name()
        } else {
            "Invalid"
        };
        write!(
            f,
            "Camera(camera_id={camera_id}, model={model_name}, width={}, height={}, \
             params=[{}] ({params_info}))",
            self.width,
            self.height,
            self.params_to_string()
        )
    }
}
