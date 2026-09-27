//! COLMAP's camera models: port of `colmap/sensor/models.h` and `models.cc` (4.2.0).
//!
//! A camera model maps camera-frame points `(u, v, w)` to pixels (`img_from_cam`) and pixels
//! back to normalized coordinates `(u, v, 1)` (`cam_from_img`). The pixel convention: the
//! upper left image corner is `(0, 0)`, the lower right `(width, height)`, so the upper left
//! pixel center is `(0.5, 0.5)`.
//!
//! Layout:
//! - this file: [`CameraModelId`], the [`CameraModel`] trait (COLMAP's CRTP base
//!   `BaseCameraModel` / `BasePerspectiveCameraModel` / `BaseSphericalCameraModel`, with their
//!   shared behavior as default methods), [`DistortedCameraModel`] and
//!   [`PerspectiveFisheyeModel`], and [`has_projectable_depth`];
//! - `pinhole.rs`: SIMPLE_PINHOLE, PINHOLE, SIMPLE_RADIAL, RADIAL, OPENCV, FULL_OPENCV;
//! - `fisheye.rs`: the perspective fisheye models;
//! - `other.rs`: FOV, SIMPLE_DIVISION, DIVISION, EUCM, EQUIRECTANGULAR;
//! - `dispatch.rs`: the runtime-dispatched free functions (`CameraModelImgFromCam`, ...);
//! - `undistortion.rs`: `IterativeUndistortion`, evaluated on `jet.rs`'s Jet;
//! - `scalar.rs`: the [`Scalar`] trait the templated model code is generic over.
//!
//! Design (an API later phases build on):
//! - COLMAP's CRTP (`struct M : BasePerspectivePinholeCameraModel<M>`) becomes a trait with
//!   associated constants; every model is a unit struct. Generic code takes `M: CameraModel`
//!   and calls `M::img_from_cam(...)`, the Rust spelling of `CameraModel::ImgFromCam`, so
//!   COLMAP's templated tests and, later, its templated cost functions port one to one.
//! - The functions COLMAP templates over `T` (`ImgFromCam`, `Distortion`, the fisheye
//!   helpers) are generic over [`Scalar`]: `f64` now, a Ceres-style Jet for the iterative
//!   undistortion now and for bundle adjustment later. Functions COLMAP only ever
//!   instantiates for `double` (`CamFromImg`, `CamRayFromImg`, the bogus-parameter checks,
//!   `CamFromImgThreshold`) take `f64`.
//! - The per-model functions keep COLMAP's out-parameter shape (`&mut` outputs, `bool`
//!   result): `CamFromImg` writes its outputs even when it fails (the Newton iteration's last
//!   iterate), and COLMAP's own tests read them. The dispatch functions return `Option`.
//! - `const T* params` is a slice; `&params[4]` is `&params[4..]`.
//!
//! Tier A: every operation keeps COLMAP's evaluation order. Differences from the pycolmap
//! wheel come only from its FMA contraction and its libm (docs/CPP_DIVERGENCES.md, entries 1
//! and 100); `tests/sensor/rust_only_camera_model_oracle.rs` pins what is bit-identical.
//! Tests: `colmap-rust/tests/sensor.rs`.

mod dispatch;
mod fisheye;
mod jet;
mod other;
mod pinhole;
mod scalar;
mod undistortion;

use std::fmt;

pub use dispatch::*;
pub use fisheye::{
    FisheyeCameraModel, OpenCVFisheyeCameraModel, RadTanThinPrismFisheyeModel,
    RadialFisheyeCameraModel, SimpleFisheyeCameraModel, SimpleRadialFisheyeCameraModel,
    ThinPrismFisheyeCameraModel,
};
pub use other::{
    DivisionCameraModel, EUCMCameraModel, EquirectangularCameraModel, FOVCameraModel,
    SimpleDivisionCameraModel,
};
pub use pinhole::{
    FullOpenCVCameraModel, OpenCVCameraModel, PinholeCameraModel, RadialCameraModel,
    SimplePinholeCameraModel, SimpleRadialCameraModel,
};
pub use scalar::Scalar;
pub use undistortion::iterative_undistortion;

use crate::math::fns;

/// Port of `colmap::CameraModelId`. The numeric values are part of COLMAP's database and
/// reconstruction file formats and never change.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i32)]
pub enum CameraModelId {
    /// `kInvalid` (-1).
    Invalid = -1,
    /// `kSimplePinhole` (0): SIMPLE_PINHOLE.
    SimplePinhole = 0,
    /// `kPinhole` (1): PINHOLE.
    Pinhole = 1,
    /// `kSimpleRadial` (2): SIMPLE_RADIAL.
    SimpleRadial = 2,
    /// `kRadial` (3): RADIAL.
    Radial = 3,
    /// `kOpenCV` (4): OPENCV.
    OpenCV = 4,
    /// `kOpenCVFisheye` (5): OPENCV_FISHEYE.
    OpenCVFisheye = 5,
    /// `kFullOpenCV` (6): FULL_OPENCV.
    FullOpenCV = 6,
    /// `kFOV` (7): FOV.
    FOV = 7,
    /// `kSimpleRadialFisheye` (8): SIMPLE_RADIAL_FISHEYE.
    SimpleRadialFisheye = 8,
    /// `kRadialFisheye` (9): RADIAL_FISHEYE.
    RadialFisheye = 9,
    /// `kThinPrismFisheye` (10): THIN_PRISM_FISHEYE.
    ThinPrismFisheye = 10,
    /// `kRadTanThinPrismFisheye` (11): RAD_TAN_THIN_PRISM_FISHEYE.
    RadTanThinPrismFisheye = 11,
    /// `kSimpleDivision` (12): SIMPLE_DIVISION.
    SimpleDivision = 12,
    /// `kDivision` (13): DIVISION.
    Division = 13,
    /// `kSimpleFisheye` (14): SIMPLE_FISHEYE.
    SimpleFisheye = 14,
    /// `kFisheye` (15): FISHEYE.
    Fisheye = 15,
    /// `kEUCM` (16): EUCM.
    EUCM = 16,
    /// `kEquirectangular` (17): EQUIRECTANGULAR.
    Equirectangular = 17,
}

impl CameraModelId {
    /// Every id in numeric order, `Invalid` first.
    pub const ALL: [CameraModelId; 19] = [
        CameraModelId::Invalid,
        CameraModelId::SimplePinhole,
        CameraModelId::Pinhole,
        CameraModelId::SimpleRadial,
        CameraModelId::Radial,
        CameraModelId::OpenCV,
        CameraModelId::OpenCVFisheye,
        CameraModelId::FullOpenCV,
        CameraModelId::FOV,
        CameraModelId::SimpleRadialFisheye,
        CameraModelId::RadialFisheye,
        CameraModelId::ThinPrismFisheye,
        CameraModelId::RadTanThinPrismFisheye,
        CameraModelId::SimpleDivision,
        CameraModelId::Division,
        CameraModelId::SimpleFisheye,
        CameraModelId::Fisheye,
        CameraModelId::EUCM,
        CameraModelId::Equirectangular,
    ];

    /// `static_cast<CameraModelId>(value)` for a value that names an enumerator; `None`
    /// otherwise (a Rust enum cannot hold an unnamed value, unlike a C++ enum class).
    pub fn from_i32(value: i32) -> Option<CameraModelId> {
        Self::ALL.iter().copied().find(|id| *id as i32 == value)
    }

    /// `CameraModelIdToString`: the enumerator's name, e.g. "kSimplePinhole" (what
    /// `MAKE_ENUM_CLASS_OVERLOAD_STREAM`'s `operator<<` prints). Not the model name; see
    /// [`camera_model_id_to_name`].
    pub fn as_str(self) -> &'static str {
        match self {
            CameraModelId::Invalid => "kInvalid",
            CameraModelId::SimplePinhole => "kSimplePinhole",
            CameraModelId::Pinhole => "kPinhole",
            CameraModelId::SimpleRadial => "kSimpleRadial",
            CameraModelId::Radial => "kRadial",
            CameraModelId::OpenCV => "kOpenCV",
            CameraModelId::OpenCVFisheye => "kOpenCVFisheye",
            CameraModelId::FullOpenCV => "kFullOpenCV",
            CameraModelId::FOV => "kFOV",
            CameraModelId::SimpleRadialFisheye => "kSimpleRadialFisheye",
            CameraModelId::RadialFisheye => "kRadialFisheye",
            CameraModelId::ThinPrismFisheye => "kThinPrismFisheye",
            CameraModelId::RadTanThinPrismFisheye => "kRadTanThinPrismFisheye",
            CameraModelId::SimpleDivision => "kSimpleDivision",
            CameraModelId::Division => "kDivision",
            CameraModelId::SimpleFisheye => "kSimpleFisheye",
            CameraModelId::Fisheye => "kFisheye",
            CameraModelId::EUCM => "kEUCM",
            CameraModelId::Equirectangular => "kEquirectangular",
        }
    }
}

impl fmt::Display for CameraModelId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where a model sits in COLMAP's CRTP hierarchy, which is how COLMAP classifies it
/// (`CameraModelIsPerspective`, `...IsPerspectivePinhole`, `...IsPerspectiveFisheye`,
/// `...IsSpherical` test `std::is_base_of_v`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraModelKind {
    /// Derives from `BasePerspectivePinholeCameraModel`.
    PerspectivePinhole,
    /// Derives from `BasePerspectiveFisheyeCameraModel`.
    PerspectiveFisheye,
    /// Derives from `BaseSphericalCameraModel`.
    Spherical,
}

/// Port of `HasProjectableDepth`: rejects points at or behind the camera plane if
/// `check_cheirality`, otherwise only points on the plane, where the projection diverges.
#[inline]
pub fn has_projectable_depth<T: Scalar>(w: T, check_cheirality: bool) -> bool {
    let epsilon = T::from_f64(f64::EPSILON);
    if check_cheirality {
        w >= epsilon
    } else {
        w.abs() >= epsilon
    }
}

/// A camera model: port of the members COLMAP's `CAMERA_MODEL_*_DEFINITIONS` macros declare
/// on every model, with `BasePerspectiveCameraModel`'s and `BaseSphericalCameraModel`'s
/// shared behavior as default methods (the spherical model overrides them).
pub trait CameraModel {
    /// `model_id`.
    const MODEL_ID: CameraModelId;
    /// `model_name`, e.g. "SIMPLE_RADIAL".
    const MODEL_NAME: &'static str;
    /// `params_info`: the parameter order, e.g. "f, cx, cy, k".
    const PARAMS_INFO: &'static str;
    /// `num_params`.
    const NUM_PARAMS: usize;
    /// The model's base class.
    const KIND: CameraModelKind;
    /// `focal_length_idxs` (empty for spherical models).
    const FOCAL_LENGTH_IDXS: &'static [usize];
    /// `principal_point_idxs` (empty for spherical models).
    const PRINCIPAL_POINT_IDXS: &'static [usize];
    /// `extra_params_idxs` (empty for spherical models).
    const EXTRA_PARAMS_IDXS: &'static [usize];
    /// `metadata_idxs` (empty for perspective models).
    const METADATA_IDXS: &'static [usize];

    /// `InitializeParams`: all focal lengths `focal_length`, the principal point at the
    /// image center, distortion at its neutral value.
    fn initialize_params(focal_length: f64, width: usize, height: usize) -> Vec<f64>;

    /// `ImgFromCam`: camera coordinates `(u, v, w)` to pixels `(x, y)`. Returns false if
    /// the projection failed; `check_cheirality` selects whether points behind the camera
    /// are rejected ([`has_projectable_depth`]).
    fn img_from_cam<T: Scalar>(
        params: &[T],
        u: T,
        v: T,
        w: T,
        x: &mut T,
        y: &mut T,
        check_cheirality: bool,
    ) -> bool;

    /// `CamFromImg`: pixels to normalized camera coordinates `(u, v)` (the ray `(u, v, 1)`).
    /// Returns false if lifting failed; the outputs may still have been written.
    fn cam_from_img(params: &[f64], x: f64, y: f64, u: &mut f64, v: &mut f64) -> bool;

    /// `HasBogusParams`: principal point outside the image, a focal length ratio outside
    /// `[min, max]`, or an extra parameter out of range.
    fn has_bogus_params(
        params: &[f64],
        width: usize,
        height: usize,
        min_focal_length_ratio: f64,
        max_focal_length_ratio: f64,
        max_extra_param: f64,
    ) -> bool {
        Self::has_bogus_principal_point(params, width, height)
            || Self::has_bogus_focal_length(
                params,
                width,
                height,
                min_focal_length_ratio,
                max_focal_length_ratio,
            )
            || Self::has_bogus_extra_params(params, max_extra_param)
    }

    /// `HasBogusFocalLength`: some `f / max(width, height)` outside `[min, max]`.
    fn has_bogus_focal_length(
        params: &[f64],
        width: usize,
        height: usize,
        min_focal_length_ratio: f64,
        max_focal_length_ratio: f64,
    ) -> bool {
        let inv_max_size = 1.0 / width.max(height) as f64;
        Self::FOCAL_LENGTH_IDXS.iter().any(|&idx| {
            let focal_length_ratio = params[idx] * inv_max_size;
            focal_length_ratio < min_focal_length_ratio
                || focal_length_ratio > max_focal_length_ratio
        })
    }

    /// `HasBogusPrincipalPoint`: the principal point outside `[0, width] x [0, height]`.
    fn has_bogus_principal_point(params: &[f64], width: usize, height: usize) -> bool {
        let cx = params[Self::PRINCIPAL_POINT_IDXS[0]];
        let cy = params[Self::PRINCIPAL_POINT_IDXS[1]];
        cx < 0.0 || cx > width as f64 || cy < 0.0 || cy > height as f64
    }

    /// `HasBogusExtraParams`: some `|extra| > max_extra_param`.
    fn has_bogus_extra_params(params: &[f64], max_extra_param: f64) -> bool {
        Self::EXTRA_PARAMS_IDXS
            .iter()
            .any(|&idx| params[idx].abs() > max_extra_param)
    }

    /// `CamFromImgThreshold`: a pixel threshold divided by the mean focal length.
    fn cam_from_img_threshold(params: &[f64], threshold: f64) -> f64 {
        let mut mean_focal_length = 0.0;
        for &idx in Self::FOCAL_LENGTH_IDXS {
            mean_focal_length += params[idx];
        }
        mean_focal_length /= Self::FOCAL_LENGTH_IDXS.len() as f64;
        threshold / mean_focal_length
    }

    /// `CamRayFromImg`: a pixel to a unit bearing vector. The perspective default lifts with
    /// `cam_from_img` and normalizes `(u, v, 1)`, so the ray always has `rz > 0`.
    fn cam_ray_from_img(
        params: &[f64],
        x: f64,
        y: f64,
        rx: &mut f64,
        ry: &mut f64,
        rz: &mut f64,
    ) -> bool {
        let mut u = 0.0;
        let mut v = 0.0;
        if !Self::cam_from_img(params, x, y, &mut u, &mut v) {
            return false;
        }
        let norm = fns::sqrt(u * u + v * v + 1.0);
        *rx = u / norm;
        *ry = v / norm;
        *rz = 1.0 / norm;
        true
    }

    /// `Rescale`: scales the parameters in place for a new resolution. Perspective: one
    /// shared focal length scales by the mean factor, `fx`/`fy` independently, and the
    /// principal point follows the image; distortion is untouched. Spherical: the `(w, h)`
    /// metadata scale.
    fn rescale(scale_x: f64, scale_y: f64, params: &mut [f64]) {
        match Self::KIND {
            CameraModelKind::Spherical => {
                params[Self::METADATA_IDXS[0]] *= scale_x;
                params[Self::METADATA_IDXS[1]] *= scale_y;
            }
            CameraModelKind::PerspectivePinhole | CameraModelKind::PerspectiveFisheye => {
                if Self::FOCAL_LENGTH_IDXS.len() == 1 {
                    params[Self::FOCAL_LENGTH_IDXS[0]] *= 0.5 * (scale_x + scale_y);
                } else {
                    params[Self::FOCAL_LENGTH_IDXS[0]] *= scale_x;
                    params[Self::FOCAL_LENGTH_IDXS[1]] *= scale_y;
                }
                params[Self::PRINCIPAL_POINT_IDXS[0]] *= scale_x;
                params[Self::PRINCIPAL_POINT_IDXS[1]] *= scale_y;
            }
        }
    }
}

/// A perspective model with an additive distortion `x + d(x)` of the normalized (or fisheye)
/// plane, the `Distortion` member COLMAP declares on every perspective model. The iterative
/// undistortion inverts it. A separate trait because COLMAP only defines `Distortion` for the
/// models that have one, and "no stubs" rules out empty bodies.
pub trait DistortedCameraModel: CameraModel {
    /// `Distortion`: `(du, dv)` at `(u, v)` given the extra parameters only.
    fn distortion<T: Scalar>(extra_params: &[T], u: T, v: T, du: &mut T, dv: &mut T);
}

/// Port of `BasePerspectiveFisheyeCameraModel` plus the `ImgFromFisheye` / `FisheyeFromImg`
/// members of `PERSPECTIVE_FISHEYE_CAMERA_MODEL_DEFINITIONS`.
pub trait PerspectiveFisheyeModel: CameraModel {
    /// `ImgFromFisheye`: fisheye-plane coordinates to pixels.
    fn img_from_fisheye<T: Scalar>(params: &[T], uu: T, vv: T, x: &mut T, y: &mut T);

    /// `FisheyeFromImg`: pixels to fisheye-plane coordinates.
    fn fisheye_from_img<T: Scalar>(params: &[T], x: T, y: T, uu: &mut T, vv: &mut T);

    /// `FisheyeFromNormal`: the equidistant mapping `theta = atan(r)` of the normalized plane.
    fn fisheye_from_normal<T: Scalar>(u: T, v: T, uu: &mut T, vv: &mut T) {
        *uu = u;
        *vv = v;
        let r = (u * u + v * v).sqrt();
        if r > T::from_f64(f64::EPSILON) {
            let theta = r.atan();
            *uu = *uu * (theta / r);
            *vv = *vv * (theta / r);
        }
    }

    /// `NormalFromFisheye`: the inverse, `r = tan(theta)`, written as `sin / cos`.
    fn normal_from_fisheye<T: Scalar>(uu: T, vv: T, u: &mut T, v: &mut T) {
        *u = uu;
        *v = vv;
        let theta = (uu * uu + vv * vv).sqrt();
        let theta_cos_theta = theta * theta.cos();
        if theta_cos_theta > T::from_f64(f64::EPSILON) {
            let scale = theta.sin() / theta_cos_theta;
            *u = *u * scale;
            *v = *v * scale;
        }
    }
}
