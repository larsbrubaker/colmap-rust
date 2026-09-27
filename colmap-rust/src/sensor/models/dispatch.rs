//! The runtime-dispatched free functions of `colmap/sensor/models.h` and `models.cc`
//! (`CameraModelImgFromCam`, `CameraModelNameToId`, ...). COLMAP writes each as a `switch`
//! over `CAMERA_MODEL_CASES`; here the `dispatch!` macro expands the same `match`, binding
//! the model type, so a dispatched call is the same computation as calling the model
//! directly (COLMAP's tests check `EXPECT_EQ` between the two).
//!
//! An unknown model id is COLMAP's `std::domain_error("Camera model does not exist")`. The only
//! unknown id a Rust [`CameraModelId`] can hold is `Invalid`, and COLMAP does pass it in at its
//! input boundaries (a text reader looks a model name up with `CameraModelNameToId` and asks
//! `CameraModelNumParams` / `CameraModelVerifyParams` without checking it exists). So the
//! metadata and validation functions those boundaries call (initialize, params info, the
//! index groups, num params, verify, has-bogus) return `Err` with
//! [`ErrorKind::DomainError`] and COLMAP's message. The per-point functions (projection,
//! unprojection, threshold, rescale, the kind predicates) stay infallible and panic with the
//! same message on `Invalid`: callers must hold a verified model id
//! (docs/CPP_DIVERGENCES.md, entry 101). The functions COLMAP does not throw from
//! (`ExistsCameraModelWithId`, `CameraModelIdToName`, `CameraModelNameToId`,
//! `CameraModelIsPerspectiveFisheye`) answer for `Invalid` instead.
//!
//! [`camera_model_img_from_cam_with_jac`] dispatches to the analytic per-model kernels of
//! `models_jacobian.h` (`jacobian*.rs`); binding `M` to a model without
//! [`CameraModelWithJac`] fails to compile, as COLMAP's `static_assert` does.

use crate::linalg::{Matrix2x3d, Matrix3x2d, Vector2d, Vector3d};
use crate::util::check::{ColmapError, ErrorKind, Result};

use super::{
    CameraModel, CameraModelId, CameraModelWithJac, CameraModelKind, DivisionCameraModel, EUCMCameraModel,
    EquirectangularCameraModel, FOVCameraModel, FisheyeCameraModel, FullOpenCVCameraModel,
    OpenCVCameraModel, OpenCVFisheyeCameraModel, PinholeCameraModel, RadTanThinPrismFisheyeModel,
    RadialCameraModel, RadialFisheyeCameraModel, SimpleDivisionCameraModel,
    SimpleFisheyeCameraModel, SimplePinholeCameraModel, SimpleRadialCameraModel,
    SimpleRadialFisheyeCameraModel, ThinPrismFisheyeCameraModel,
};

/// `CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION`.
#[cold]
fn camera_model_does_not_exist() -> ! {
    panic!("Camera model does not exist")
}

/// `CAMERA_MODEL_DOES_NOT_EXIST_EXCEPTION` as an error, for the fallible functions.
fn check_camera_model_exists(model_id: CameraModelId) -> Result<()> {
    if model_id == CameraModelId::Invalid {
        return Err(ColmapError::new(
            ErrorKind::DomainError,
            "Camera model does not exist",
        ));
    }
    Ok(())
}

/// `switch (model_id) { CAMERA_MODEL_SWITCH_CASES }`: evaluates `$body` with `$M` bound to
/// the model type; `Invalid` panics like COLMAP's default case throws.
macro_rules! dispatch {
    ($id:expr, $M:ident => $body:expr) => {
        match $id {
            CameraModelId::SimplePinhole => {
                type $M = SimplePinholeCameraModel;
                $body
            }
            CameraModelId::Pinhole => {
                type $M = PinholeCameraModel;
                $body
            }
            CameraModelId::SimpleRadial => {
                type $M = SimpleRadialCameraModel;
                $body
            }
            CameraModelId::Radial => {
                type $M = RadialCameraModel;
                $body
            }
            CameraModelId::OpenCV => {
                type $M = OpenCVCameraModel;
                $body
            }
            CameraModelId::OpenCVFisheye => {
                type $M = OpenCVFisheyeCameraModel;
                $body
            }
            CameraModelId::FullOpenCV => {
                type $M = FullOpenCVCameraModel;
                $body
            }
            CameraModelId::FOV => {
                type $M = FOVCameraModel;
                $body
            }
            CameraModelId::SimpleRadialFisheye => {
                type $M = SimpleRadialFisheyeCameraModel;
                $body
            }
            CameraModelId::RadialFisheye => {
                type $M = RadialFisheyeCameraModel;
                $body
            }
            CameraModelId::ThinPrismFisheye => {
                type $M = ThinPrismFisheyeCameraModel;
                $body
            }
            CameraModelId::RadTanThinPrismFisheye => {
                type $M = RadTanThinPrismFisheyeModel;
                $body
            }
            CameraModelId::SimpleDivision => {
                type $M = SimpleDivisionCameraModel;
                $body
            }
            CameraModelId::Division => {
                type $M = DivisionCameraModel;
                $body
            }
            CameraModelId::SimpleFisheye => {
                type $M = SimpleFisheyeCameraModel;
                $body
            }
            CameraModelId::Fisheye => {
                type $M = FisheyeCameraModel;
                $body
            }
            CameraModelId::EUCM => {
                type $M = EUCMCameraModel;
                $body
            }
            CameraModelId::Equirectangular => {
                type $M = EquirectangularCameraModel;
                $body
            }
            CameraModelId::Invalid => camera_model_does_not_exist(),
        }
    };
}

/// `ExistsCameraModelWithName`.
pub fn exists_camera_model_with_name(model_name: &str) -> bool {
    camera_model_name_to_id(model_name) != CameraModelId::Invalid
}

/// `ExistsCameraModelWithId`: false for `Invalid`.
pub fn exists_camera_model_with_id(model_id: CameraModelId) -> bool {
    model_id != CameraModelId::Invalid
}

/// `CameraModelNameToId`: the id of the model named `model_name`, or `Invalid`.
pub fn camera_model_name_to_id(model_name: &str) -> CameraModelId {
    CameraModelId::ALL
        .iter()
        .copied()
        .filter(|&id| id != CameraModelId::Invalid)
        .find(|&id| camera_model_id_to_name(id) == model_name)
        .unwrap_or(CameraModelId::Invalid)
}

/// `CameraModelIdToName`: the model's name, or "" for `Invalid`.
pub fn camera_model_id_to_name(model_id: CameraModelId) -> &'static str {
    if model_id == CameraModelId::Invalid {
        return "";
    }
    dispatch!(model_id, M => M::MODEL_NAME)
}

/// `CameraModelInitializeParams`: all focal lengths `focal_length`, the principal point at
/// the image center. Assumes image measurements within `[0, dim]`, i.e. the upper left
/// corner is the `(0, 0)` coordinate (rather than the center of the upper left pixel).
///
/// # Errors
///
/// `DomainError` "Camera model does not exist" for `Invalid`.
pub fn camera_model_initialize_params(
    model_id: CameraModelId,
    focal_length: f64,
    width: usize,
    height: usize,
) -> Result<Vec<f64>> {
    check_camera_model_exists(model_id)?;
    Ok(dispatch!(model_id, M => M::initialize_params(focal_length, width, height)))
}

/// `CameraModelParamsInfo`: the human-readable parameter order.
///
/// # Errors
///
/// `DomainError` "Camera model does not exist" for `Invalid`.
pub fn camera_model_params_info(model_id: CameraModelId) -> Result<&'static str> {
    check_camera_model_exists(model_id)?;
    Ok(dispatch!(model_id, M => M::PARAMS_INFO))
}

/// `CameraModelFocalLengthIdxs` (empty for spherical models).
///
/// # Errors
///
/// `DomainError` "Camera model does not exist" for `Invalid`.
pub fn camera_model_focal_length_idxs(model_id: CameraModelId) -> Result<&'static [usize]> {
    check_camera_model_exists(model_id)?;
    Ok(dispatch!(model_id, M => M::FOCAL_LENGTH_IDXS))
}

/// `CameraModelPrincipalPointIdxs` (empty for spherical models).
///
/// # Errors
///
/// `DomainError` "Camera model does not exist" for `Invalid`.
pub fn camera_model_principal_point_idxs(model_id: CameraModelId) -> Result<&'static [usize]> {
    check_camera_model_exists(model_id)?;
    Ok(dispatch!(model_id, M => M::PRINCIPAL_POINT_IDXS))
}

/// `CameraModelExtraParamsIdxs` (empty for spherical models).
///
/// # Errors
///
/// `DomainError` "Camera model does not exist" for `Invalid`.
pub fn camera_model_extra_params_idxs(model_id: CameraModelId) -> Result<&'static [usize]> {
    check_camera_model_exists(model_id)?;
    Ok(dispatch!(model_id, M => M::EXTRA_PARAMS_IDXS))
}

/// `CameraModelMetaDataParamsIdxs` (empty for perspective models).
///
/// # Errors
///
/// `DomainError` "Camera model does not exist" for `Invalid`.
pub fn camera_model_meta_data_params_idxs(model_id: CameraModelId) -> Result<&'static [usize]> {
    check_camera_model_exists(model_id)?;
    Ok(dispatch!(model_id, M => M::METADATA_IDXS))
}

/// `CameraModelNumParams`.
///
/// # Errors
///
/// `DomainError` "Camera model does not exist" for `Invalid`.
pub fn camera_model_num_params(model_id: CameraModelId) -> Result<usize> {
    check_camera_model_exists(model_id)?;
    Ok(dispatch!(model_id, M => M::NUM_PARAMS))
}

/// `CameraModelRescale`: rescales the parameters in place for a new image resolution, given
/// the per-axis scale factors `new_dim / old_dim`.
///
/// # Panics
///
/// On `CameraModelId::Invalid` ("Camera model does not exist"): callers must hold a
/// verified model id (docs/CPP_DIVERGENCES.md, entry 101).
pub fn camera_model_rescale(
    model_id: CameraModelId,
    scale_x: f64,
    scale_y: f64,
    params: &mut [f64],
) {
    dispatch!(model_id, M => M::rescale(scale_x, scale_y, params))
}

/// `CameraModelVerifyParams`: whether the parameter count matches the model.
///
/// # Errors
///
/// `DomainError` "Camera model does not exist" for `Invalid`.
pub fn camera_model_verify_params(model_id: CameraModelId, params: &[f64]) -> Result<bool> {
    check_camera_model_exists(model_id)?;
    Ok(dispatch!(model_id, M => params.len() == M::NUM_PARAMS))
}

/// `CameraModelHasBogusParams`: principal point outside the image, a focal length over the
/// larger image side outside `[min, max]`, or an extra parameter's magnitude above
/// `max_extra_param` (plus the model's own checks, e.g. EUCM's).
///
/// # Errors
///
/// `DomainError` "Camera model does not exist" for `Invalid`.
pub fn camera_model_has_bogus_params(
    model_id: CameraModelId,
    params: &[f64],
    width: usize,
    height: usize,
    min_focal_length_ratio: f64,
    max_focal_length_ratio: f64,
    max_extra_param: f64,
) -> Result<bool> {
    check_camera_model_exists(model_id)?;
    Ok(dispatch!(model_id, M => M::has_bogus_params(
        params,
        width,
        height,
        min_focal_length_ratio,
        max_focal_length_ratio,
        max_extra_param,
    )))
}

/// `CameraModelImgFromCam`: camera coordinates `(u, v, w)` to pixels, or `None` if the
/// projection fails. The inverse of [`camera_model_cam_from_img`]. COLMAP's default for
/// `check_cheirality` is true.
///
/// # Panics
///
/// On `CameraModelId::Invalid` ("Camera model does not exist"): callers must hold a
/// verified model id (docs/CPP_DIVERGENCES.md, entry 101).
pub fn camera_model_img_from_cam(
    model_id: CameraModelId,
    params: &[f64],
    uvw: Vector3d,
    check_cheirality: bool,
) -> Option<Vector2d> {
    let mut x = 0.0;
    let mut y = 0.0;
    let ok = dispatch!(model_id, M => M::img_from_cam(
        params,
        uvw.x,
        uvw.y,
        uvw.z,
        &mut x,
        &mut y,
        check_cheirality,
    ));
    ok.then(|| Vector2d::new(x, y))
}

/// `CameraModelImgFromCamWithJac`: [`camera_model_img_from_cam`] through the model's
/// analytic `ImgFromCamWithJac`, also writing the projection Jacobian
/// `d(x, y) / d(u, v, w)` into `j_uvw` when it is given (`None` skips it, COLMAP's
/// `nullptr`). `None` on failure, with `j_uvw` untouched. COLMAP's default for
/// `check_cheirality` is true.
pub fn camera_model_img_from_cam_with_jac(
    model_id: CameraModelId,
    params: &[f64],
    uvw: Vector3d,
    j_uvw: Option<&mut Matrix2x3d>,
    check_cheirality: bool,
) -> Option<Vector2d> {
    let mut x = 0.0;
    let mut y = 0.0;
    // 2x3 row-major Jacobian, zero-initialized like COLMAP's, which copies it out whole.
    let mut j_uvw_data = [0.0; 6];
    let with_jac = j_uvw.is_some();
    let ok = dispatch!(model_id, M => M::img_from_cam_with_jac(
        params,
        uvw.x,
        uvw.y,
        uvw.z,
        &mut x,
        &mut y,
        None,
        with_jac.then_some(&mut j_uvw_data),
        check_cheirality,
    ));
    if !ok {
        return None;
    }
    if let Some(j_uvw) = j_uvw {
        *j_uvw = Matrix2x3d::from_row_major(j_uvw_data);
    }
    Some(Vector2d::new(x, y))
}

/// `CameraModelCamFromImg`: pixels to normalized camera coordinates `(u, v)`, or `None` if
/// lifting fails. Limited to the forward hemisphere; see [`camera_model_cam_ray_from_img`].
///
/// # Panics
///
/// On `CameraModelId::Invalid` ("Camera model does not exist"): callers must hold a
/// verified model id (docs/CPP_DIVERGENCES.md, entry 101).
pub fn camera_model_cam_from_img(
    model_id: CameraModelId,
    params: &[f64],
    xy: Vector2d,
) -> Option<Vector2d> {
    let mut u = 0.0;
    let mut v = 0.0;
    let ok = dispatch!(model_id, M => M::cam_from_img(params, xy.x, xy.y, &mut u, &mut v));
    ok.then(|| Vector2d::new(u, v))
}

/// `CameraModelCamRayFromImg`: a pixel to a unit bearing vector in the camera frame, for
/// any pixel the model can unproject, including back-facing rays of omnidirectional
/// cameras. Prefer this to lifting, homogenizing and normalizing when a 3D ray is needed.
///
/// # Panics
///
/// On `CameraModelId::Invalid` ("Camera model does not exist"): callers must hold a
/// verified model id (docs/CPP_DIVERGENCES.md, entry 101).
pub fn camera_model_cam_ray_from_img(
    model_id: CameraModelId,
    params: &[f64],
    xy: Vector2d,
) -> Option<Vector3d> {
    let mut rx = 0.0;
    let mut ry = 0.0;
    let mut rz = 0.0;
    let ok = dispatch!(model_id, M => M::cam_ray_from_img(
        params,
        xy.x,
        xy.y,
        &mut rx,
        &mut ry,
        &mut rz,
    ));
    ok.then(|| Vector3d::new(rx, ry, rz))
}

/// `CameraModelCamFromImgThreshold`: a pixel threshold in normalized camera units (divided
/// by the mean focal length for perspective models).
///
/// # Panics
///
/// On `CameraModelId::Invalid` ("Camera model does not exist"): callers must hold a
/// verified model id (docs/CPP_DIVERGENCES.md, entry 101).
pub fn camera_model_cam_from_img_threshold(
    model_id: CameraModelId,
    params: &[f64],
    threshold: f64,
) -> f64 {
    dispatch!(model_id, M => M::cam_from_img_threshold(params, threshold))
}

/// `CameraModelIsPerspectiveFisheye`: false for `Invalid`, as in COLMAP.
pub fn camera_model_is_perspective_fisheye(model_id: CameraModelId) -> bool {
    if model_id == CameraModelId::Invalid {
        return false;
    }
    dispatch!(model_id, M => M::KIND == CameraModelKind::PerspectiveFisheye)
}

/// `CameraModelIsPerspective`: has a focal length and a finite image plane.
///
/// # Panics
///
/// On `CameraModelId::Invalid` ("Camera model does not exist"): callers must hold a
/// verified model id (docs/CPP_DIVERGENCES.md, entry 101).
pub fn camera_model_is_perspective(model_id: CameraModelId) -> bool {
    dispatch!(model_id, M => M::KIND != CameraModelKind::Spherical)
}

/// `CameraModelIsPerspectivePinhole`: projects as `X / Z`, then deforms the plane, so a
/// calibration matrix K is meaningful.
///
/// # Panics
///
/// On `CameraModelId::Invalid` ("Camera model does not exist"): callers must hold a
/// verified model id (docs/CPP_DIVERGENCES.md, entry 101).
pub fn camera_model_is_perspective_pinhole(model_id: CameraModelId) -> bool {
    dispatch!(model_id, M => M::KIND == CameraModelKind::PerspectivePinhole)
}

/// `CameraModelIsSpherical`.
///
/// # Panics
///
/// On `CameraModelId::Invalid` ("Camera model does not exist"): callers must hold a
/// verified model id (docs/CPP_DIVERGENCES.md, entry 101).
pub fn camera_model_is_spherical(model_id: CameraModelId) -> bool {
    dispatch!(model_id, M => M::KIND == CameraModelKind::Spherical)
}

/// `CamRayFromImgJacobian`: the Jacobian of [`camera_model_cam_ray_from_img`],
/// `d(u, v, w) / d(x, y)`, from the projection Jacobian `j_uvw = d(x, y) / d(u, v, w)` at the
/// unit bearing `cam_ray`. `None` if `j_uvw` is rank deficient.
///
/// Central projection depends only on the direction of the ray, so the ray lies in the null
/// space of `j_uvw`, which has rank 2. For a *unit* ray its Moore-Penrose pseudo-inverse is
/// exactly the Jacobian of the normalized unprojection, and its range is the tangent plane
/// of the unit sphere at the ray, so no explicit tangent basis is required. Uses the closed
/// form of Terekhov and Larsson, "Tangent Sampson Error", ICCV 2023, Lemma 1:
/// `J_uvw^+ = 1 / (d . (g_x x g_y)) * [ (g_y x d), (d x g_x) ]`, where `g_x` and `g_y` are
/// the rows of `J_uvw`. This is cheaper than forming `J^T (J J^T)^-1` and exposes the rank
/// condition directly as the scalar triple product in the denominator.
pub fn cam_ray_from_img_jacobian(cam_ray: Vector3d, j_uvw: Matrix2x3d) -> Option<Matrix3x2d> {
    let g_x = j_uvw.row(0);
    let g_y = j_uvw.row(1);
    let alpha = cam_ray.dot(g_x.cross(g_y));
    // Since the projection is degree-zero homogeneous, g_x x g_y is parallel to the ray, so
    // for a unit ray |alpha| == ||g_x x g_y|| and alpha^2 is exactly det(J J^T), the product
    // of the squared singular values. Requiring
    // |alpha| > kMinRelAlpha * (||g_x||^2 + ||g_y||^2) therefore rejects singular value
    // ratios below kMinRelAlpha, i.e. condition numbers worse than ~1e6. Relative, so the
    // test is invariant to focal length.
    const MIN_REL_ALPHA: f64 = 1e-6;
    // Negated on purpose: a NaN alpha is rejected, as in COLMAP.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    let rank_deficient = !(alpha.abs() > MIN_REL_ALPHA * (g_x.squared_norm() + g_y.squared_norm()));
    if rank_deficient {
        return None;
    }
    Some(Matrix3x2d::from_columns(g_y.cross(cam_ray), cam_ray.cross(g_x)) / alpha)
}
